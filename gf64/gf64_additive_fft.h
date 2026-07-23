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
 * Writes exactly out_len low-order coefficients of a * b. The output is
 * zero-padded when out_len exceeds the full product and truncated otherwise.
 * The output range must not overlap either input range; violations abort with
 * GF64_ALIASING_ERROR.
 */
void gf64_poly_mul_padded(
	gf64_t *out,
	const gf64_t *a, size_t len_a,
	const gf64_t *b, size_t len_b,
	size_t out_len
);

HEDLEY_END_C_DECLS

#endif /* GF64_ADDITIVE_FFT_H */
