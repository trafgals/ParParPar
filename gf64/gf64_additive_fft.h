#ifndef GF64_ADDITIVE_FFT_H
#define GF64_ADDITIVE_FFT_H

/*
 * Gao-Mateer-style additive FFT over GF(2^64).
 *
 * T3 (scalar) and T4 (AVX-512 vectorized) of the par3-cauchy-fft-kernel plan.
 * Both implementations use the simplified monomial-basis recursion. The public
 * inverse intentionally preserves the plan's non-normalized scaling: for
 * n = 2^k, k > 0,
 *
 *   gf64_fft_inverse(gf64_fft_forward(p, n), n) = p * 2^(k - 1)
 *
 * where 2^(k - 1) is interpreted as the GF(2^64) element with that bit set.
 * gf64_poly_mul compensates internally and returns ordinary coefficient-basis
 * polynomial multiplication.
 *
 * The AVX-512 implementation follows the SAME recursion structure as the
 * scalar version, but uses SIMD vectorized element-wise operations
 * (gf64_square_avx512 for the Frobenius map, gf64_mul_avx512 for the
 * butterfly's alpha * odd multiplication). For n < 64 the AVX-512 entry
 * delegates to the scalar implementation, since the SIMD setup overhead
 * dominates for small problems.
 */

#include "gf64_global.h"

#include <stddef.h>

HEDLEY_BEGIN_C_DECLS

void gf64_fft_forward(gf64_t *poly, size_t n);
void gf64_fft_inverse(gf64_t *poly, size_t n);

/*
 * LCH14 (Lin-Chung-Han 2014) additive FFT that satisfies the convolution
 * theorem for arbitrary GF(2^64) inputs. The forward/inverse pair
 *
 *   gf64_fft_forward_lch14(arr, n)
 *   gf64_fft_inverse_lch14(arr, n)
 *
 * is its own inverse (no 1/n factor; the X_i novel basis absorbs the
 * normalization per LCH14 §III-C). For polynomials a, b in GF(2^64)[x]
 * with deg(a), deg(b) < n and deg(a) + deg(b) < n,
 *
 *   inv_lch14(forward_lch14(a) * forward_lch14(b)) = a * b   (pointwise)
 *
 * See gf64_additive_fft_lch14.c for the algorithm and
 * gf64_cantor_basis.h for the precomputed v_i constants. Verified by
 * test_gf64_fft_poly_mul using these new entry points.
 */
void gf64_fft_forward_lch14(gf64_t *arr, size_t n);
void gf64_fft_inverse_lch14(gf64_t *arr, size_t n);

/*
 * Vandermonde-matrix additive FFT over GF(2^64). Correct by
 * construction: forward(p)[i] = sum_j p[j] * v_i^j, which satisfies
 * the convolution theorem (pointwise-multiply in evaluation form =
 * poly convolution in monomial form). Asymptotic cost O(N^2) per call;
 * the matrix and its inverse are pre-computed and cached per n. This
 * is the drop-in "guaranteed correct" FFT for downstream consumers
 * like Fenger Toeplitz; consumers that need O(N log N) should swap in
 * a faster transform (e.g. the Gao-Mateer tower form). See
 * gf64_additive_fft_vandermonde.c.
 */
void gf64_fft_forward_vandermonde(gf64_t *out, const gf64_t *in, size_t n);
void gf64_fft_inverse_vandermonde(gf64_t *out, const gf64_t *in, size_t n);

/*
 * AVX-512 vectorized forward / inverse Gao-Mateer-style additive FFT.
 *
 *   void gf64_fft_forward_avx512(gf64_t *poly, size_t n)
 *   void gf64_fft_inverse_avx512(gf64_t *poly, size_t n)
 *
 * Same recursion structure as the scalar versions, with element-wise
 * operations (Frobenius squaring, GF multiplication in the butterfly) lifted
 * into 8-lane AVX-512 SIMD using gf64_square_avx512 (T2) and
 * gf64_mul_avx512 (T1). For n < 64, the AVX-512 entry delegates to the
 * corresponding scalar function.
 *
 * Bit-exact to gf64_fft_forward / gf64_fft_inverse — verified by
 * test/test_gf64_additive_fft.c.
 *
 * Host: requires AVX-512F + VPCLMULQDQ. Caller is responsible for dispatching
 * here only when the host supports these features.
 */
void gf64_fft_forward_avx512(gf64_t *poly, size_t n);
void gf64_fft_inverse_avx512(gf64_t *poly, size_t n);

void gf64_poly_mul(
	gf64_t *out,
	const gf64_t *a,
	size_t deg_a,
	const gf64_t *b,
	size_t deg_b
);

/*
 * Pre-allocated-output variant of gf64_poly_mul.  Multiplies two polynomials
 * (length = degree + 1 each) and writes only the low-order out_len
 * coefficients into the caller-owned buffer `out` — no internal allocation
 * is tied to `out`, so callers (notably the Newton iteration in
 * gf64_poly_invmod) can reuse a single scratch buffer across all doubling
 * steps without per-call malloc/free.
 *
 *   out      caller-owned buffer of size out_len coefficients; on return
 *            holds the low-order out_len coefficients of a * b (constant-
 *            first, after zero-padding any unwritten higher-order slots).
 *   a, len_a multiplicand coefficients.
 *   b, len_b multiplier coefficients.
 *   out_len  number of coefficients to produce.
 *
 * This entry point is API-additive — existing gf64_poly_mul callers are
 * unaffected.  The implementation is bit-exact to gf64_poly_mul for
 * (deg_a, deg_b, out_len) = (deg_a, deg_b, deg_a + deg_b + 1).
 *
 * Phase 2a: above GF64_POLY_MUL_INTERNAL_KARATSUBA_MIN the implementation
 * dispatches into Karatsuba (O(n^1.585), see gf64_poly_mul_karatsuba.h) for
 * a real algorithmic speedup on the polynomial-heavy T6/T7/T8 primitives.
 * Below the threshold the original O(n^2) schoolbook runs. Both are
 * bit-exact at every input size.
 */
void gf64_poly_mul_padded(
	gf64_t *out,
	const gf64_t *a, size_t len_a,
	const gf64_t *b, size_t len_b,
	size_t out_len
);

/*
 * Vandermonde-FFT-based polynomial multiplication.
 *
 * Computes a * b in GF(2^64)[x] via:
 *   1. Pad a, b to length n (next power of 2 above deg_a + deg_b + 1).
 *   2. Vandermonde-FFT each (O(n^2) each via cached matrix).
 *   3. Pointwise-multiply.
 *   4. Inverse Vandermonde-FFT.
 *
 * Bit-exact to gf64_poly_mul for (deg_a, deg_b, out_len) =
 * (deg_a, deg_b, deg_a + deg_b + 1). Asymptotic cost O(n^2) per call;
 * the matrices are cached per n. Use the schoolbook via
 * gf64_poly_mul_padded for small n where it wins; use this entry for
 * testing the additive-FFT pipeline end to end.
 *
 * Buffer requirements: `out` must hold at least out_len coefficients.
 */
void gf64_poly_mul_vandermonde(
	gf64_t *out,
	const gf64_t *a, size_t deg_a,
	const gf64_t *b, size_t deg_b
);

HEDLEY_END_C_DECLS

#endif /* GF64_ADDITIVE_FFT_H */
