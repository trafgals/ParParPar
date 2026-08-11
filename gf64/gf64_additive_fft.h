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

/*
 * HQC 2026 TCHES §2.3 Algorithm 2 (LCH14) additive FFT — Cantor-basis
 * variant implemented in gf64_additive_fft_hqc2026.c.
 *
 * Same O(N log N) complexity as the Gao-Mateer variant above, but uses the
 * nested vanishing-subspace decomposition V_{i-1} ⊂ V_i with basis vectors
 * {v_0, ..., v_{LM-1}} (see gf64_cantor_basis.h). The forward transform
 * evaluates f at the affine line a + V_{logn}; the inverse evaluates the
 * novel polynomial representation.
 *
 * PUBLIC API — scratch-required, two-tier length caps.
 *
 *   Every entry point takes a caller-allocated scratch buffer and a
 *   scratch_words size. Use the companion *_scratch_words(n) function to
 *   size the buffer. Mirrors the convention established by
 *   gf64_poly_divmod_scratch (gf64_mpe.h:115).
 *
 * Length caps (two tiers; sizes in (131072, 2^20] require General
 * Algorithm 1, filed as a separate work item):
 *
 *   GF64_HQC_MAX_MATRIXFORM_N = 16384   — matrix-form path (cache-backed)
 *                                          _fwd_scratch, _inv_scratch,
 *                                          _poly_mul_scratch
 *   GF64_HQC_MAX_LM_N = 2^20 = 1048576  — matrix-free recursive path
 *                                          _fwd_recursive_scratch,
 *                                          _inv_recursive_scratch,
 *                                          _poly_mul_recursive_scratch
 *
 *   gf64_addfft64_poly_mul_recursive_scratch
 *       has the largest scratch demand (4n) — it is the recommended entry
 *       for production dispatch once the caller knows n <= 2^20.
 *
 * Bit-exact for n <= 4096 (see test/test_gf64_additive_fft_hqc2026.c);
 * recursive path additionally verified at all powers of 2 up to 2^20
 * (the LM_N cap), including non-simple-2-term sizes (65536, 262144,
 * etc.) that Chen 2018 Algorithm 1 (the basisCvt_recursive_v2 polyeval
 * port) handles directly via 2-term XOR-shift division at every level.
 */

/* ----- Length caps (exposed for dispatch) ----- */
#define GF64_HQC_MAX_MATRIXFORM_N ((size_t)16384)
/* Matrix-free recursive path: Chen 2018 Algorithm 1 general case. At
 * power-of-2 n ≤ 2^20 (= 1 M), the polyeval-form `hqc_cvt` /
 * `basisCvt_recursive_v2` does 2-term XOR-shift divisions at multiple
 * scales within the poly block, yielding the O(n log n) recursion
 * described in PR #49 /tmp/general_algorithm_1_research.md. Pure XOR
 * + a single gf64_mul_reference in the butterfly → trivially host-
 * portable (no AVX-512 gating needed, since gf64_mul_reference is
 * ISA-agnostic). */
#define GF64_HQC_MAX_LM_N         ((size_t)(1 << 20))

/* ----- Scratch size queries ----- */
size_t gf64_addfft64_fwd_scratch_words(size_t n);              /* 4n */
size_t gf64_addfft64_inv_scratch_words(size_t n);              /* 4n */
size_t gf64_addfft64_poly_mul_scratch_words(size_t n);         /* 4n */
size_t gf64_addfft64_fwd_recursive_scratch_words(size_t n);    /* 2n */
size_t gf64_addfft64_inv_recursive_scratch_words(size_t n);    /* 2n */
size_t gf64_addfft64_poly_mul_recursive_scratch_words(size_t n); /* 4n */

/* ----- Matrix-form path (cap: GF64_HQC_MAX_MATRIXFORM_N = 16384) ----- */
void gf64_addfft64_fwd_scratch(gf64_t *arr, size_t n,
                               gf64_t *scratch, size_t scratch_words);
void gf64_addfft64_inv_scratch(gf64_t *arr, size_t n,
                               gf64_t *scratch, size_t scratch_words);
void gf64_addfft64_poly_mul_scratch(
	gf64_t *out,
	const gf64_t *a, size_t len_a,
	const gf64_t *b, size_t len_b,
	size_t out_len,
	gf64_t *scratch, size_t scratch_words
);

/* ----- Matrix-free recursive path (cap: GF64_HQC_MAX_LM_N = 131072) ----- */
void gf64_addfft64_fwd_recursive_scratch(gf64_t *arr, size_t n,
                                         gf64_t *scratch, size_t scratch_words);
void gf64_addfft64_inv_recursive_scratch(gf64_t *arr, size_t n,
                                         gf64_t *scratch, size_t scratch_words);

/*
 * gf64_addfft64_poly_mul_recursive_scratch — out[0..out_len) = a * b
 * (full product, truncated). Pads inputs with zeros up to
 * next_pow2(len_a + len_b - 1), runs fwd/inv internally. Convolution
 * theorem holds within the _recursive family only.
 */
void gf64_addfft64_poly_mul_recursive_scratch(
	gf64_t *out,
	const gf64_t *a, size_t len_a,
	const gf64_t *b, size_t len_b,
	size_t out_len,
	gf64_t *scratch, size_t scratch_words
);

/* ----- AVX-512 (PCLMULQDQ) accelerated recursive path -----
 *
 * Same scratch layout and cap as the scalar _recursive_scratch entries
 * above. Use these on AVX-512F + VPCLMULQDQ hosts for 4-6× speedup at
 * n = 4096 over the scalar butterfly (file header benchmark).
 *
 * Bit-exact to the scalar _recursive_scratch family — verified by
 * test_gf64_additive_fft_hqc2026 at small n (32, 512).
 *
 * Host: requires AVX-512F + VPCLMULQDQ. Caller is responsible for
 * dispatching here only when the host supports these features.
 */
void gf64_addfft64_fwd_recursive_scratch_avx512(gf64_t *arr, size_t n,
                                                gf64_t *scratch, size_t scratch_words);
void gf64_addfft64_inv_recursive_scratch_avx512(gf64_t *arr, size_t n,
                                                gf64_t *scratch, size_t scratch_words);
void gf64_addfft64_poly_mul_recursive_scratch_avx512(
	gf64_t *out,
	const gf64_t *a, size_t len_a,
	const gf64_t *b, size_t len_b,
	size_t out_len,
	gf64_t *scratch, size_t scratch_words
);

HEDLEY_END_C_DECLS

#endif /* GF64_ADDITIVE_FFT_H */
