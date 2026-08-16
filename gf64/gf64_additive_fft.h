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
/* Minimum operand/out length for the HQC FFT tier (same value as the
 * private GF64_HQC_FFT_MIN in gf64_additive_fft.c — the dispatch gate
 * in gf64_additive_fft.c requires all three of len_a/len_b/out_len
 * >= this). */
#define GF64_HQC_FFT_MIN          ((size_t)96)
#define GF64_HQC_MAX_MATRIXFORM_N ((size_t)16384)
/* Matrix-free recursive path: Chen 2018 Algorithm 1 general case. At
 * power-of-2 n ≤ 2^20 (= 1 M), the polyeval-form `hqc_cvt` /
 * `basisCvt_recursive_v2` does 2-term XOR-shift divisions at multiple
 * scales within the poly block, yielding the O(n log n) recursion
 * described in PR #49 /tmp/general_algorithm_1_research.md. Pure XOR
 * + a single gf64_mul_reference in the butterfly → trivially host-
 * portable (no AVX-512 gating needed, since gf64_mul_reference is
 * ISA-agnostic).
 *
 * IMPORTANT (cubic review 4910960162 P1): any future dispatcher that
 * routes to the recursive path MUST check the padded transform size
 * `n_pad = next_pow2(la + lb - 1)` against this cap BEFORE calling
 * the library. E.g. inputs (600000, 600000) pad to n_pad = 2^21 > cap
 * — the library's public entries assert this and abort in debug;
 * release builds have it as a documented undefined-behaviour input.
 * Use the query `gf64_hqc_supports_size(n)` (declared below) for the
 * cap-aware gate. */
#define GF64_HQC_MAX_LM_N         ((size_t)(1 << 20))

/* Returns 1 if every *_recursive_scratch public entry accepts n (i.e.
 * n <= GF64_HQC_MAX_LM_N AND n is a power of 2), 0 otherwise. Use this
 * in dispatcher code to decide whether HQC FFT is a valid path before
 * passing the padded size to the library. */
int gf64_hqc_supports_size(size_t n);

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

/*
 * gf64_addfft64_poly_mul_batch_shared — K products with ONE shared operand
 * (issue #59 §4 A3 word-batching primitive).
 *
 *   outs[k][0..out_len) = shared[0..len_shared) * f_k[0..len_f)
 *   for k in [0, K), where f_k = f + k*len_f (K contiguous slabs).
 *
 * The shared operand's forward transform is computed ONCE per call;
 * each f_k is forward-transformed, pointwise-multiplied by the shared
 * transform, and inverse-transformed. Bit-exact to K sequential
 * gf64_addfft64_poly_mul_recursive_scratch calls — identical pipeline
 * and per-word operation order; the shared transform is a pure hoist
 * (pinned by test_gf64_mul_batch_shared).
 *
 * Constraints (the caller's responsibility, mirroring the single-mul
 * dispatch): all K products must map onto the SAME padded size n —
 * uniform len_f and out_len; variable-length operands must be
 * zero-extended by the caller. Sizes outside the HQC window
 * (n < GF64_HQC_FFT_MIN or n > GF64_HQC_MAX_LM_N) are not handled
 * here; the caller falls back to per-word muls.
 *
 * Scratch: [pt: n | pf: n | inner: 2n] = 4n total (same budget as the
 * single-mul entry — pt holds the shared transform across the K words,
 * pf is reused per word). n = next_pow2(max(len_shared + len_f - 1,
 * out_len)); assert n <= GF64_HQC_MAX_LM_N.
 */
void gf64_addfft64_poly_mul_batch_shared(
	gf64_t *const *outs, size_t K,
	const gf64_t *shared, size_t len_shared,
	const gf64_t *f, size_t len_f,
	size_t out_len,
	gf64_t *scratch, size_t scratch_words
);
void gf64_addfft64_poly_mul_batch_shared_avx512(
	gf64_t *const *outs, size_t K,
	const gf64_t *shared, size_t len_shared,
	const gf64_t *f, size_t len_f,
	size_t out_len,
	gf64_t *scratch, size_t scratch_words
);
/* Interleaved batch FFT: coefficient-major layout arr[i*8 + k] = word k's
 * i-th coefficient; all K <= 8 words are transformed together (one zmm per
 * coefficient index). Bit-exact per lane to the scalar FFT. Scratch for
 * fwd/inv: 2*8*n words. Scratch for the batch-shared mul: 4*8*n words
 * (pt: 8n | pf: 8n | inner: 16n). Requires AVX-512F + VPCLMULQDQ. */
void gf64_addfft64_fwd_batch_avx512(gf64_t *arr, size_t n,
	gf64_t *scratch, size_t scratch_words);
void gf64_addfft64_inv_batch_avx512(gf64_t *arr, size_t n,
	gf64_t *scratch, size_t scratch_words);
void gf64_addfft64_poly_mul_batch_shared_interleaved_avx512(
	gf64_t *const *outs, size_t K,
	const gf64_t *shared, size_t len_shared,
	const gf64_t *f, size_t len_f,
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

/*
 * Dispatch counters (exposed for boundary regression tests).
 * Incremented once per gf64_poly_mul_internal invocation per code path
 * taken. Fields are plain uint64_t (NOT stdatomic) so this header stays
 * C89/C99-clean; all increments go through GF64_DISPATCH_COUNTER_INC,
 * which is a lock-free atomic add on both toolchains (MSVC
 * _InterlockedIncrement64 / GCC __atomic_fetch_add). The kernel can be
 * entered from concurrent workers (e.g. parallel creates on separate
 * worker_threads), so a plain ++ would lose increments; the atomic add
 * keeps the counters exact even then. Reads are relaxed — exact totals
 * are only meaningful in the single-threaded test harness. See
 * gf64/test/test_gf64_poly_mul_internal_dispatch.c for boundary uses.
 */
typedef struct gf64_dispatch_counts {
	uint64_t schoolbook;
	uint64_t karatsuba;
	uint64_t toom3;
	uint64_t fft;
	uint64_t hqc_fft;  /* Phase 2 — HQC 2026 TCHES §2.3 additive FFT */
} gf64_dispatch_counts_t;

extern gf64_dispatch_counts_t gf64_dispatch_counts;
void gf64_dispatch_counts_reset(void);

#if defined(_MSC_VER)
#include <intrin.h>
#define GF64_DISPATCH_COUNTER_INC(p) \
	((void)_InterlockedIncrement64((volatile long long *)(p)))
#else
#define GF64_DISPATCH_COUNTER_INC(p) \
	((void)__atomic_fetch_add((p), 1, __ATOMIC_RELAXED))
#endif

/*
 * Test-only HQC cap override. 0 (default) = use GF64_HQC_MAX_LM_N.
 * Lets boundary tests exercise the >cap fallthrough tier (Karatsuba)
 * without a 2^20-length operand — the real cap forces a
 * next_pow2(2^21) balanced Karatsuba recursion (~3^15 leaves, minutes
 * of CI wall-time). Single-threaded test harness only; see
 * gf64/test/test_gf64_poly_mul_internal_dispatch.c case [7].
 */
extern size_t gf64_hqc_max_lm_n_override;

HEDLEY_END_C_DECLS

#endif /* GF64_ADDITIVE_FFT_H */
