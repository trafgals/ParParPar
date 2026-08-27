/*
 * ============================================================================
 * gf64/gf64_fenger.c — Fenger Toeplitz pipeline (issue #28).
 *
 * See gf64_fenger.h for the algorithm and asymptotics.
 *
 * The implementation is split into prepare/execute/release so the B-axis
 * can be sharded across multiple threads without re-building the trees
 * per slice. The convenience wrapper gf64_fenger_matvec handles the
 * single-threaded case.
 *
 * Asymptotic correctness: the per-word interpolation step is Bostan-Schost
 * top-down via the weights-aware gf64_multi_point_interp_weights (T8b) —
 * derivative-based Lagrange form since issue #59 A1, needing no cached
 * tree inverses; the prepare phase's barycentric weights (ctx->V_prime_inv)
 * skip the per-word P'-MPE + N inversions. (The gated public wrapper
 * gf64_multi_point_interp is NOT the Fenger entry point; cubic review
 * 50f46d24 P3.) The per-word
 * evaluation step is Bostan-Schost top-down via gf64_multi_point_eval
 * (T8a). The product gives p(y_r) in O((N + R) log² (N + R)) per word.
 *
 * Field operations are delegated to gf64_mul_reference (scalar SSE2) and
 * gf64_invert_ita_one (T5). Stages 4a and 4d (the per-word weighted-V'
 * and divide-by-V(y_r) element-wise multiplies) use gf64_elem_mul, the
 * ISA-dispatched wrapper from gf64_elem_mul.c — AVX-512 hosts run the
 * clmul-Barrett lane primitive (8 elements per VPCLMULQDQ pair); AVX-2 /
 * SSSE3 / scalar hosts fall back to gf64_mul_reference (issue #59 T3).
 * ============================================================================
 */

#include "gf64_fenger.h"
#include "gf64_additive_fft.h"
#include "gf64_barycentric.h"
#include "gf64_elem_mul.h"
#include "gf64_invert_ita.h"
#include "gf64_mpe.h"
#include "gf64_mul.h"
#include "gf64_subproduct.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Single source of truth for the batched-execute K cap: used by the
 * PAR3_FENGER_BATCH_WORDS clamp (fenger_batch_words) and by every
 * fixed-size batch array below (ptrs, left_degs/right_degs, degs,
 * child_degs = 2 * K_MAX). (cubic review 91c44c50 P3) */
#define FENGER_BATCH_K_MAX 32

/* Forward declarations — the batched execute (§2b below) is defined
 * after the single-word gf64_fenger_execute but dispatched from it. */
static size_t fenger_batch_words(void);
static void gf64_fenger_execute_batched(
	const gf64_fenger_ctx *ctx,
	const gf64_t *in,  size_t B,
	      gf64_t *out,
	size_t w_start, size_t w_end,
	size_t K
);

/* Scalar SSE2 reference mul, declared in gf64_single.c. */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* ----------------------------------------------------------------------------
 * Opaque context: holds everything shared across per-word execute calls.
 * --------------------------------------------------------------------------*/
struct gf64_fenger_ctx {
	SubproductTree tree_x; /* input-point tree       */
	SubproductTree tree_y; /* recovery-point tree    */
	gf64_t *V_prime;       /* V'(x_c) for each input */
	gf64_t *V_prime_inv;   /* 1/V'(x_c) for each input (barycentric weights;
	                        * reused per word by the weights-aware interp) */
	gf64_t *V_at_y_inv;    /* 1/V(y_r) for each r    */
	/* T5 (issue #59): precomputed per-node reciprocals for the eval walk
	 * over tree_y. Eliminates the per-batch gf64_poly_invmod calls inside
	 * fenger_eval_recurse_batch (the reciprocals depend only on the tree,
	 * which is fixed at prepare). Storage = (num_levels-1)*R/2 words ≈
	 * R/2 * log2(R); at R=32k that's ~250 KiB (trivial vs the tree's
	 * own ~20 MiB). NULL if num_levels <= 1 (R == 1, no internal nodes). */
	gf64_t *recip_pool;
	size_t  recip_pool_words;
	size_t  N;             /* padded input count (power of 2)  */
	size_t  R;             /* padded recovery count (power of 2) */
	size_t  num_inputs_real;   /* real input blocks (< N when padded) */
	size_t  num_recovery_real; /* real recovery rows (< R when padded) */
};

/* ----------------------------------------------------------------------------
 * 1. gf64_fenger_prepare: build trees, weights, V(y_r) inverse.
 *
 * Core builder shared by gf64_fenger_prepare (power-of-2, exact counts)
 * and gf64_fenger_prepare_padded (synthetic power-of-2 padding). Builds
 * T_X over the N_pad input points, T_Y over the R_pad recovery points,
 * the V'(x_c) weights, and the V(y_r) inverses.
 *
 * Point placement:
 *   - input points:   x_c = firstInput + c          for c in [0, N_pad)
 *   - recovery points: y_r = firstRecovery + r       for r in [0, R_pad)
 *
 * For the padded variant the caller passes numInputsPadded/numRecoveryPadded
 * (>= real counts, both 0/1/power-of-2) and numInputsReal/numRecoveryReal;
 * the engine must ensure the padded point ranges do not collide with each
 * other (see ComputeRecoveryBlocksFenger). The Fenger identity
 *   sum_c in[c]/(y - x_c) = p(y)/V(y)
 * holds for ANY point set, and synthetic zero-data inputs contribute 0, so
 * the real recovery rows are bit-identical to the unpadded computation.
 * --------------------------------------------------------------------------*/
static gf64_fenger_ctx *gf64_fenger_prepare_core(
	uint64_t firstInput,
	uint64_t firstRecovery,
	size_t numInputsPadded,
	size_t numRecoveryPadded,
	size_t numInputsReal,
	size_t numRecoveryReal,
	uint64_t syntheticInputBase
) {
	gf64_fenger_ctx *ctx = (gf64_fenger_ctx *)malloc(sizeof(gf64_fenger_ctx));
	if (ctx == NULL) abort();
	ctx->N = numInputsPadded;
	ctx->R = numRecoveryPadded;
	ctx->num_inputs_real   = numInputsReal;
	ctx->num_recovery_real = numRecoveryReal;
	ctx->V_prime    = NULL;
	ctx->V_prime_inv = NULL;
	ctx->V_at_y_inv = NULL;

	/* ---- Build T_X over input points ----
	 * Real inputs occupy [firstInput, firstInput + numInputsReal).
	 * When padding is active (numInputsPadded > numInputsReal), the
	 * synthetic zero-data inputs sit at
	 *   [syntheticInputBase, syntheticInputBase + (numInputsPadded - numInputsReal))
	 * which the caller must place disjoint from the recovery range
	 * [firstRecovery, firstRecovery + numRecoveryPadded) so V(y_r) != 0.
	 * The Fenger identity holds for ANY point set, so the padding is
	 * bit-exact (synthetic in[c] = 0 contributes 0 to the sum). */
	gf64_t *x_points = (gf64_t *)malloc(numInputsPadded * sizeof(gf64_t));
	if (x_points == NULL) abort();
	for (size_t c = 0; c < numInputsPadded; c++) {
		if (c < numInputsReal) {
			x_points[c] = (gf64_t)(firstInput + (uint64_t)c);
		} else {
			x_points[c] = (gf64_t)(syntheticInputBase + (uint64_t)(c - numInputsReal));
		}
	}
	gf64_subproduct_tree_build(x_points, numInputsPadded, &ctx->tree_x);
	free(x_points);

	/* ---- Build T_Y over recovery points ---- */
	gf64_t *y_points = (gf64_t *)malloc(numRecoveryPadded * sizeof(gf64_t));
	if (y_points == NULL) abort();
	for (size_t r = 0; r < numRecoveryPadded; r++) {
		y_points[r] = (gf64_t)(firstRecovery + (uint64_t)r);
	}
	gf64_subproduct_tree_build(y_points, numRecoveryPadded, &ctx->tree_y);
	free(y_points);

	/* ---- Barycentric weights (1/V'(x_c)) and their inverses ----
	 * The weights are kept in ctx->V_prime_inv: the per-word
	 * interpolation (gf64_multi_point_interp_weights) consumes exactly
	 * 1/P'(x_c) = 1/V'(x_c), so keeping them avoids re-running the
	 * P'-MPE + N inversions once per word (issue #59 A1). */
	gf64_t *bary = (gf64_t *)malloc(numInputsPadded * sizeof(gf64_t));
	if (bary == NULL) abort();
	gf64_barycentric_weights(&ctx->tree_x, bary);

	ctx->V_prime_inv = bary;
	ctx->V_prime = (gf64_t *)malloc(numInputsPadded * sizeof(gf64_t));
	if (ctx->V_prime == NULL) abort();
	/*
	 * Issue #59 T2 (P2.2): invert V'(x_c) for each input via the AVX-512
	 * 8-lane ZMM batch primitive gf64_invert_ita_batch when the host has a
	 * fully functional VPCLMULQDQ+ZMM pipeline; otherwise fall through to
	 * the scalar gf64_invert_ita_one loop. The batch primitive itself
	 * dispatches the count % 8 tail to the scalar fallback inside
	 * gf64_invert_ita_batch (gf64_invert_ita_avx512.c:121-124), so we
	 * never round-down the input count.
	 *
	 * Mirrors the dispatch pattern in gf64_barycentric.c:208-212 (issue
	 * #62/cubic-review 4914681432 P1): gate on gf64_has_vpclmulqdq (the
	 * CPUID+XCR0+ZMM-probe-ANDed flag), NOT on gf64_current_method (the
	 * PD2 workload-chosen method), so PD2 downclock downgrades do not
	 * silently disable the batch path on hosts that genuinely support
	 * the ISA.
	 *
	 * ISA fallback: on hosts without working AVX-512F+VPCLMULQDQ the
	 * loop falls through to the scalar gf64_invert_ita_one path — the
	 * bit-exact output is identical (verified by test_gf64_fenger_pipeline
	 * and the new test_gf64_fenger_prepare_batch).
	 */
	if (gf64_has_vpclmulqdq) {
		gf64_invert_ita_batch(ctx->V_prime, bary, numInputsPadded);
	} else {
		for (size_t c = 0; c < numInputsPadded; c++) {
			ctx->V_prime[c] = gf64_invert_ita_one(bary[c]);
		}
	}

	/* ---- V(y_r) for each recovery point (MPE on tree_y) ---- */
	gf64_t *V_at_y = (gf64_t *)malloc(numRecoveryPadded * sizeof(gf64_t));
	if (V_at_y == NULL) abort();
	gf64_multi_point_eval(
		ctx->tree_x.level_data[0],
		ctx->tree_x.num_points,
		&ctx->tree_y,
		V_at_y
	);

	ctx->V_at_y_inv = (gf64_t *)malloc(numRecoveryPadded * sizeof(gf64_t));
	if (ctx->V_at_y_inv == NULL) abort();
	/* Same dispatch as the V_prime loop above — see the comment there.
	 * At N=262144/R=32768 (1G/262144/4KiB bench target shape) this loop
	 * alone runs R=32768 scalar inversions of 63 squarings + 62 muls
	 * each; the batch path drops the wall-time by the ZMM lane count. */
	if (gf64_has_vpclmulqdq) {
		gf64_invert_ita_batch(ctx->V_at_y_inv, V_at_y, numRecoveryPadded);
	} else {
		for (size_t r = 0; r < numRecoveryPadded; r++) {
			ctx->V_at_y_inv[r] = gf64_invert_ita_one(V_at_y[r]);
		}
	}
	free(V_at_y);

	/* T5 (issue #59): precompute reciprocals of every internal child
	 * polynomial in tree_y. The per-batch gf64_poly_invmod calls inside
	 * the execute walk collapse to pointer dereferences. The execute
	 * walk's per-node call to fenger_eval_recurse_batch computes the
	 * reciprocals of BOTH children (P_L and P_R) at every internal
	 * level, so we must store 2(R-1) reciprocals = (L-1)*R words
	 * (~8 MiB at R=65536 - trivial vs the tree's ~20 MiB). */
	{
		const size_t num_levels = ctx->tree_y.num_levels;
		if (num_levels > 1) {
			const size_t total_words = (num_levels - 1) *
			                            numRecoveryPadded;
			ctx->recip_pool = (gf64_t *)malloc(total_words *
			                                    sizeof(gf64_t));
			if (ctx->recip_pool == NULL) abort();
			ctx->recip_pool_words = total_words;
			/* BFS over CHILD levels. At child_lev the tree has
			 * 2^child_lev child polys, each of degree R/2^child_lev
			 * (so stride child_deg+1 in level_data[child_lev]). */
			gf64_t *cursor = ctx->recip_pool;
			/* cubic P3 (task 17): exclude the leaf level (child_deg==1) —
			 * leaf reciprocals are never consumed (HQC eligibility
			 * requires child_deg >= 96). Loop bound stops at
			 * num_levels-2 inclusive (i.e. skips child_lev == num_levels-1,
			 * which is the leaf where child_deg == 1). */
			for (size_t child_lev = 1; child_lev + 1 < num_levels; child_lev++) {
				const size_t child_deg = numRecoveryPadded >>
				                          child_lev;
				const size_t child_stride = child_deg + 1;
				const gf64_t *level_data = ctx->tree_y.level_data[child_lev];
				const size_t num_children = (size_t)1 << child_lev;
				/* cubic P2 (task 18): allocate rev once per level, reuse across
				 * num_children iterations. Saves num_children-1 mallocs per
				 * level — at R=32768 the deepest levels have 16384 children. */
				gf64_t *rev = (gf64_t *)malloc(child_stride * sizeof(gf64_t));
				if (rev == NULL) abort();
				for (size_t i = 0; i < num_children; i++) {
					const gf64_t *p = level_data + i * child_stride;
					for (size_t k = 0; k <= child_deg; k++) {
						rev[k] = p[child_deg - k];
					}
					gf64_poly_invmod(rev, child_deg, child_deg,
					                  cursor);
					cursor += child_deg;
				}
				free(rev);
			}
		} else {
			ctx->recip_pool = NULL;
			ctx->recip_pool_words = 0;
		}
	}

	return ctx;
}

gf64_fenger_ctx *gf64_fenger_prepare(
	uint64_t firstInput,
	uint64_t firstRecovery,
	size_t N,
	size_t R
) {
	if (N == 0 || R == 0) {
		return NULL;
	}
	return gf64_fenger_prepare_core(
		firstInput, firstRecovery, N, R, N, R, 0
	);
}

/*
 * Padded variant: numInputsPadded / numRecoveryPadded must be 0/1/
 * power-of-2 and >= the real counts. syntheticInputBase places the
 * padded synthetic input points; the caller must ensure
 *   [syntheticInputBase, syntheticInputBase + numInputsPadded - numInputsReal)
 * is disjoint from both the real input range and the recovery range
 * [firstRecovery, firstRecovery + numRecoveryPadded) — the engine uses
 * firstRecovery + numRecoveryPadded for this (see
 * GF64Controller::ComputeRecoveryBlocksFenger). Returns NULL when either
 * padded count is 0.
 */
gf64_fenger_ctx *gf64_fenger_prepare_padded(
	uint64_t firstInput,
	uint64_t firstRecovery,
	size_t numInputs,
	size_t numRecovery,
	size_t numInputsPadded,
	size_t numRecoveryPadded,
	uint64_t syntheticInputBase
) {
	if (numInputs == 0 || numRecovery == 0 ||
	    numInputsPadded == 0 || numRecoveryPadded == 0) {
		return NULL;
	}
	return gf64_fenger_prepare_core(
		firstInput, firstRecovery,
		numInputsPadded, numRecoveryPadded,
		numInputs, numRecovery,
		syntheticInputBase
	);
}

/* ----------------------------------------------------------------------------
 * 2. gf64_fenger_execute: per-word pipeline over [w_start, w_end).
 *
 * Each invocation allocates its own N+R scratch buffers for the
 * interpolation and evaluation steps, so concurrent invocations on the
 * same ctx are safe.
 * --------------------------------------------------------------------------*/
void gf64_fenger_execute(
	const gf64_fenger_ctx *ctx,
	const gf64_t *in,  size_t B,
	      gf64_t *out,
	size_t w_start, size_t w_end
) {
	if (ctx == NULL || B == 0 || w_start >= w_end) {
		return;
	}
	/* A3: word-batched path (issue #59 §4). K >= 2 batches the tree
	 * passes; K == 1 is the per-word reference loop below (bit-exact to
	 * the batched path — pinned by test_gf64_fenger_batch). */
	const size_t K = fenger_batch_words();
	if (K >= 2) {
		gf64_fenger_execute_batched(ctx, in, B, out, w_start, w_end, K);
		return;
	}
	const size_t N = ctx->N;
	const size_t R = ctx->R;
	const SubproductTree *tree_x = &ctx->tree_x;
	const SubproductTree *tree_y = &ctx->tree_y;

	/* Per-thread scratch. Alloc'd once per call; could be reused across
	 * w-iterations but the inner loops allocate nothing. */
	gf64_t *weighted = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *poly_p   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *p_at_y   = (gf64_t *)malloc(R * sizeof(gf64_t));
	if (weighted == NULL || poly_p == NULL || p_at_y == NULL) {
		abort();
	}

	const size_t deg_p = (tree_x->num_points > 1)
	                       ? (tree_x->num_points - 1) : 0;

	for (size_t w = w_start; w < w_end; w++) {
		/* 4a: weighted values. Synthetic padded inputs (c >=
		 * num_inputs_real) have zero DATA (the caller's buffer only
		 * holds real blocks), so weight them 0 without reading `in`.
		 *
		 * Issue #59 T3: strided gather into a contiguous slab, then a
		 * contiguous element-wise mul. The gather is the same memory
		 * traffic as the original per-element scalar mul load (one
		 * word per c), but the mul is now ISA-dispatched: AVX-512
		 * uses the clmul-Barrett lane primitive (8 elements per
		 * VPCLMULQDQ pair), AVX-2 / SSSE3 / scalar fall back to
		 * gf64_mul_reference. Bit-exact because the gather reads the
		 * same value the scalar mul would have read, and the
		 * contiguous mul is bit-exact to the scalar reference
		 * (test_gf64_elem_mul.c).
		 *
		 * Padded region (c >= num_inputs_real) gets weighted[c] = 0
		 * via memset; the subsequent mul reads V_prime[c] for all c
		 * (a valid array index), but 0 * anything = 0, so the result
		 * matches the original scalar code exactly. */
		for (size_t c = 0; c < ctx->num_inputs_real; c++) {
			weighted[c] = in[c * B + w];
		}
		if (ctx->num_inputs_real < N) {
			memset(weighted + ctx->num_inputs_real, 0,
			       (N - ctx->num_inputs_real) * sizeof(gf64_t));
		}
		gf64_elem_mul(weighted, weighted, ctx->V_prime, N);
		/* 4b: interpolate to polynomial p_w of degree < N. Fenger is the
		 * sanctioned consumer of the Bostan-Schost body, so it calls the
		 * ungated weights-aware variant directly — the
		 * PAR3_GF64_USE_INTERP opt-in gate is for production code that
		 * does not require interpolation. The prepare phase precomputed
		 * ctx->V_prime_inv = 1/V'(x_c) (barycentric weights), so this
		 * call skips the per-word P'-MPE + N inversions (issue #59 A1). */
		gf64_multi_point_interp_weights(tree_x, weighted,
		                                ctx->V_prime_inv, poly_p);
		/* 4c: evaluate p_w at each recovery point. */
		gf64_multi_point_eval(poly_p, deg_p, tree_y, p_at_y);
		/* 4d: divide by V(y_r) — only real recovery rows are written
		 * (synthetic padded rows would overflow the caller's buffer).
		 *
		 * Issue #59 T3: contiguous vectorized mul into a local slab,
		 * then strided scatter to the caller's column-major buffer.
		 * `p_at_y` is a private scratch slab (allocated above) that
		 * is discarded at the end of this call, so aliasing it with
		 * itself (out == a in gf64_elem_mul) is safe. */
		if (ctx->num_recovery_real > 0) {
			gf64_elem_mul(p_at_y, p_at_y, ctx->V_at_y_inv,
			              ctx->num_recovery_real);
			for (size_t r = 0; r < ctx->num_recovery_real; r++) {
				out[r * B + w] = p_at_y[r];
			}
		}
	}

	free(weighted);
	free(poly_p);
	free(p_at_y);
}

/* ----------------------------------------------------------------------------
 * 2b. Word-batched execute (issue #59 §4 A3).
 *
 * The per-word loop above re-runs the SAME tree walks for every word:
 * the interp combine multiplies every word's fL/fR by the SAME sibling
 * polynomials (P_R/P_L), and the MPE divmods re-compute the SAME
 * Newton reciprocals per word. Batching K words per pass hoists the
 * shared work:
 *
 *   - interp combine: the sibling's forward transform is computed once
 *     per node per batch (gf64_addfft64_poly_mul_batch_shared);
 *   - MPE divmod: the reciprocal rev(P)^{-1} mod x^{m_max} is computed
 *     once per node per batch (m_max = child_deg, level-uniform; the
 *     per-word quotient uses the true m_true = deg_f - deg_g + 1 by
 *     truncating the full-precision product, which is exact since the
 *     low coefficients of rev(f)·inv mod x^{m_max} equal
 *     rev(f)·inv mod x^{m_true}).
 *
 * Per-word operation order is UNCHANGED (same transforms, same
 * pointwise, same combine), so the batched result is bit-exact to the
 * per-word loop — pinned by test_gf64_fenger_batch.
 *
 * The env var PAR3_FENGER_BATCH_WORDS (default 4, clamp [1,
 * FENGER_BATCH_K_MAX]) selects K; 1 selects the per-word loop above.
 * Read per call (no cache) so tests can flip it between runs.
 * --------------------------------------------------------------------------*/

static size_t fenger_batch_words(void) {
	const char *e = getenv("PAR3_FENGER_BATCH_WORDS");
	long v = (e != NULL) ? strtol(e, NULL, 10) : 4;
	if (v < 1) v = 1;
	if (v > FENGER_BATCH_K_MAX) v = FENGER_BATCH_K_MAX;
	return (size_t)v;
}

/* Mirror gf64_poly_mul_internal's HQC gate (gf64_additive_fft.c). */
static int fenger_hqc_eligible(size_t len_a, size_t len_b, size_t out_len) {
	size_t cap = gf64_hqc_max_lm_n_override
		? gf64_hqc_max_lm_n_override : GF64_HQC_MAX_LM_N;
	size_t max_len = (len_a > len_b) ? len_a : len_b;
	size_t full_len = 2 * max_len - 1;
	if (out_len > full_len) full_len = out_len;
	return (len_a >= GF64_HQC_FFT_MIN &&
	        len_b >= GF64_HQC_FFT_MIN &&
	        out_len >= GF64_HQC_FFT_MIN &&
	        len_a <= cap && len_b <= cap && full_len <= cap);
}

/* Char-2 Horner (mirrors horner_eval_poly in gf64_mpe.c). */
static gf64_t fenger_horner(const gf64_t *f, size_t deg, gf64_t x) {
	gf64_t acc = f[deg];
	for (size_t i = deg; i-- > 0; ) {
		acc = gf64_mul_reference(acc, x) ^ f[i];
	}
	return acc;
}

static void fenger_batch_shared_mul(
	gf64_t *const *outs, size_t K,
	const gf64_t *shared, size_t len_shared,
	const gf64_t *f, size_t len_f,
	size_t out_len,
	gf64_t *scratch, size_t scratch_words)
{
	if (gf64_current_method == GF64_AVX512) {
		/* Interleaved path: K <= 8 words transformed together in the
		 * coefficient-major layout. Needs 32n scratch (4 * 8 * n);
		 * falls back to the within-word SIMD path otherwise. */
		size_t full_len = len_shared + len_f - 1;
		if (full_len < out_len) full_len = out_len;
		size_t n = 1;
		while (n < full_len) n <<= 1;
		if (K <= 8 && scratch_words >= 32 * n) {
			gf64_addfft64_poly_mul_batch_shared_interleaved_avx512(
				outs, K, shared, len_shared, f, len_f, out_len,
				scratch, scratch_words);
			return;
		}
		gf64_addfft64_poly_mul_batch_shared_avx512(
			outs, K, shared, len_shared, f, len_f, out_len,
			scratch, scratch_words);
	} else {
		gf64_addfft64_poly_mul_batch_shared(
			outs, K, shared, len_shared, f, len_f, out_len,
			scratch, scratch_words);
	}
}

/* ---- Batched interpolation combine (top-down derivative form) ----
 *
 * z: K slabs of tree->num_points (already scaled by the barycentric
 * weights); out: K slabs of the same size (the interpolants).
 * Frame layout per node (size 3K·N_at_lev):
 *   [fL: K·f_size | fR: K·f_size | t1: K·N_at_lev | t2: K·N_at_lev]
 * mul_scratch: one region of 4·N words reused at every depth (the max
 * batch mul n is 2·f_size = N at the root).
 */
static void fenger_interp_recurse_batch(
	const SubproductTree *tree,
	size_t lev, size_t node_idx, size_t out_offset,
	const gf64_t *z,
	gf64_t *out,
	gf64_t *scratch,
	gf64_t *mul_scratch, size_t mul_scratch_words,
	size_t K)
{
	const size_t num_levels = tree->num_levels;
	const size_t N_at_lev = tree->num_points >> lev;
	const size_t f_size = N_at_lev >> 1;

	if (lev + 1 == num_levels) {
		for (size_t k = 0; k < K; k++) {
			out[k * N_at_lev] = z[k * tree->num_points + out_offset];
		}
		return;
	}

	const size_t child_stride = f_size + 1;
	const gf64_t *children = tree->level_data[lev + 1];
	const gf64_t *P_L = children + (2 * node_idx)     * child_stride;
	const gf64_t *P_R = children + (2 * node_idx + 1) * child_stride;

	gf64_t *slot0 = scratch;
	gf64_t *slot1 = scratch + K * f_size;
	gf64_t *slot2 = scratch + 2 * K * f_size;
	gf64_t *slot3 = scratch + 2 * K * f_size + K * N_at_lev;
	gf64_t *child_scratch = scratch + 3 * K * N_at_lev;

	fenger_interp_recurse_batch(tree, lev + 1, 2 * node_idx,
	                            out_offset, z, slot0, child_scratch,
	                            mul_scratch, mul_scratch_words, K);
	fenger_interp_recurse_batch(tree, lev + 1, 2 * node_idx + 1,
	                            out_offset + f_size, z, slot1, child_scratch,
	                            mul_scratch, mul_scratch_words, K);

	if (fenger_hqc_eligible(f_size + 1, f_size, N_at_lev)) {
		const size_t n = 2 * f_size; /* next_pow2(max(2·f_size, N_at_lev)) */
		/* 32n (not 4n) so the interleaved batch FFT engages (K <= 8). */
		const size_t sw = 32 * n;
		gf64_t *ptrs[FENGER_BATCH_K_MAX];
		for (size_t k = 0; k < K; k++) ptrs[k] = slot2 + k * N_at_lev;
		fenger_batch_shared_mul(ptrs, K, P_R, f_size + 1, slot0, f_size,
		                        N_at_lev, mul_scratch, sw);
		for (size_t k = 0; k < K; k++) ptrs[k] = slot3 + k * N_at_lev;
		fenger_batch_shared_mul(ptrs, K, P_L, f_size + 1, slot1, f_size,
		                        N_at_lev, mul_scratch, sw);
	} else {
		/* Per-word fallback (identical to the single-word combine). */
		for (size_t k = 0; k < K; k++) {
			gf64_poly_mul_padded(slot2 + k * N_at_lev, P_R, f_size + 1,
			                     slot0 + k * f_size, f_size, N_at_lev);
			gf64_poly_mul_padded(slot3 + k * N_at_lev, P_L, f_size + 1,
			                     slot1 + k * f_size, f_size, N_at_lev);
		}
	}
	for (size_t k = 0; k < K; k++) {
		gf64_t *o = out + k * N_at_lev;
		const gf64_t *t1 = slot2 + k * N_at_lev;
		const gf64_t *t2 = slot3 + k * N_at_lev;
		for (size_t i = 0; i < N_at_lev; i++) o[i] = t1[i] ^ t2[i];
	}
}

static void fenger_interp_batch(
	const SubproductTree *tree,
	const gf64_t *z, size_t K,
	gf64_t *out,
	gf64_t *scratch,
	gf64_t *mul_scratch, size_t mul_scratch_words)
{
	if (tree->num_points == 1) {
		for (size_t k = 0; k < K; k++) out[k] = z[k];
		return;
	}
	fenger_interp_recurse_batch(tree, 0, 0, 0, z, out,
	                            scratch, mul_scratch, mul_scratch_words, K);
}

/* ---- Batched multi-point evaluation (top-down divmod walk) ----
 *
 * f: K slabs of N_at_lev (each word's polynomial, degree deg_fs[k]); the
 *    per-word stride is f_stride (the ROOT caller's slab layout — e.g.
 *    the interp output is N-strided while N_at_lev at the root is R —
 *    deeper levels pass their r_L/r_R buffers at N_at_lev stride);
 * out: K slabs of tree->num_points (the leaf values).
 *
 * Frame layout per node (size 2K·N_at_lev + (2K+2)·child_deg):
 *   [r_L: K·N_at_lev | r_R: K·N_at_lev
 *    | revq: K·child_deg (rev_f input, then q_L in place)
 *    | q2: K·child_deg (q_R for the per-word fallback)
 *    | inv_L: child_deg | inv_R: child_deg]
 * r_L/r_R are N_at_lev wide because gf64_poly_divmod copies f into r
 * first (the contract requires r to hold deg_f + 1 <= N_at_lev words);
 * the remainder itself only occupies [0, child_deg).
 * mul_scratch: one region of 4·N words reused at every depth.
 *
 * m_max = child_deg (level-uniform upper bound on the quotient length);
 * words whose degree < child_deg need no division (r = f). The shared
 * reciprocals are computed once per node per batch via
 * gf64_poly_invmod; the per-word quotient/remainder apply the
 * precomputed reciprocal (exact by truncation, see the section comment).
 *
 * The walk is only valid for deg_f < N (the interpolation bound, as in
 * gf64_multi_point_eval); the caller (gf64_fenger_execute_batched)
 * routes deg_p >= R (and R == 1) through a per-word Horner fallback.
 */
/* T5: look up the precomputed reciprocal of an internal child node in
 * tree_y by (level, local index). The pool is laid out BFS over child
 * levels: child_lev=1 first (R/2 words), then child_lev=2 (R/4 per
 * child * 4 = R words), etc. (see gf64_fenger_prepare_core). child_lev
 * is the level of the CHILD whose reciprocal we want (>= 1, since
 * leaves have no reciprocal). child_node_idx is the local index within
 * that level. The reciprocal has child_deg = R/2^child_lev words (one
 * polynomial of degree < child_deg). */
static inline gf64_t *fenger_recip_at(
	gf64_fenger_ctx *ctx,
	size_t child_lev, size_t child_node_idx
) {
	const size_t R = ctx->R;
	return ctx->recip_pool + (child_lev - 1) * R +
	       child_node_idx * (R >> child_lev);
}

static void fenger_eval_recurse_batch(
	const gf64_fenger_ctx *ctx,
	const SubproductTree *tree,
	size_t lev, size_t node_idx,
	const gf64_t *f, size_t f_stride, const size_t *deg_fs,
	gf64_t *out,
	gf64_t *scratch,
	gf64_arena_t *arena,
	gf64_t *mul_scratch, size_t mul_scratch_words,
	size_t K)
{
	const size_t num_levels = tree->num_levels;

	if (lev + 1 == num_levels) {
		/* Leaf: f(x_j) = f[0] (degree < 1 after the divmod chain). */
		const gf64_t *leaves = tree->level_data[lev];
		const gf64_t xj = leaves[2 * node_idx];
		for (size_t k = 0; k < K; k++) {
			const gf64_t *fk = f + k * f_stride;
			out[k * tree->num_points + node_idx] =
				(deg_fs[k] == 0) ? fk[0] : fenger_horner(fk, deg_fs[k], xj);
		}
		return;
	}

	const size_t N_at_lev = tree->num_points >> lev;
	const size_t child_deg = tree->level_degs[lev + 1];
	const size_t child_stride = child_deg + 1;
	const gf64_t *children = tree->level_data[lev + 1];
	const gf64_t *P_L = children + (2 * node_idx)     * child_stride;
	const gf64_t *P_R = children + (2 * node_idx + 1) * child_stride;

	gf64_t *r_L  = scratch;
	gf64_t *r_R  = scratch + K * N_at_lev;
	gf64_t *revq = scratch + 2 * K * N_at_lev;
	gf64_t *q2   = revq + K * child_deg;
	gf64_t *inv_L = q2 + K * child_deg;
	gf64_t *inv_R = inv_L + child_deg;
	gf64_t *child_scratch = scratch + (2 * K * N_at_lev + (2 * K + 2) * child_deg);

	const size_t m_max = child_deg;
	int any_needs = 0;
	for (size_t k = 0; k < K; k++) {
		if (deg_fs[k] >= child_deg) { any_needs = 1; break; }
	}

	/* 2K entries: left children occupy [0, K), right children [K, 2K). */
	size_t child_degs[2 * FENGER_BATCH_K_MAX];
	if (any_needs && fenger_hqc_eligible(child_deg + 1, m_max, child_deg)) {
		/* T5: reciprocals were precomputed at prepare time. */
		inv_L = fenger_recip_at((gf64_fenger_ctx *)ctx,
		                       lev + 1, 2 * node_idx);
		inv_R = fenger_recip_at((gf64_fenger_ctx *)ctx,
		                       lev + 1, 2 * node_idx + 1);

		const size_t n = 2 * child_deg;
		const size_t sw = gf64_addfft64_poly_mul_recursive_scratch_words(n);
		gf64_t *ptrs[FENGER_BATCH_K_MAX];

		/* rev_f slabs (zero-padded to m_max; aliases the q output slab —
		 * safe: the batch mul reads f_k into its transform buffer before
		 * writing outs[k]). */
		for (size_t k = 0; k < K; k++) {
			const gf64_t *fk = f + k * f_stride;
			gf64_t *rk = revq + k * m_max;
			if (deg_fs[k] >= child_deg) {
				for (size_t i = 0; i < m_max; i++) {
					rk[i] = (i <= deg_fs[k]) ? fk[deg_fs[k] - i] : 0;
				}
			} else {
				memset(rk, 0, m_max * sizeof(gf64_t));
			}
		}
		/* q_k = rev_f_k · inv_L mod x^{m_true_k} (un-reverse in place).
		 * Words with deg_fs[k] < child_deg need no division: their
		 * rev_f was zeroed, so the product is exactly 0 — skip the
		 * un-reverse (m_true would underflow for them). */
		for (size_t k = 0; k < K; k++) ptrs[k] = revq + k * m_max;
		fenger_batch_shared_mul(ptrs, K, inv_L, m_max, revq, m_max,
		                        m_max, mul_scratch, sw);
		for (size_t k = 0; k < K; k++) {
			if (deg_fs[k] < child_deg) continue;
			const size_t m_true = deg_fs[k] - child_deg + 1;
			gf64_t *rk = revq + k * m_max;
			for (size_t i = 0; i < m_true / 2; i++) {
				gf64_t t = rk[i]; rk[i] = rk[m_true - 1 - i]; rk[m_true - 1 - i] = t;
			}
			if (m_true < m_max) memset(rk + m_true, 0, (m_max - m_true) * sizeof(gf64_t));
			}
			/* r_L = f ^ P_L·q (batch-shared; only the low child_deg coefs). */
			for (size_t k = 0; k < K; k++) ptrs[k] = r_L + k * N_at_lev;
			fenger_batch_shared_mul(ptrs, K, P_L, child_stride, revq, m_max,
			child_deg, mul_scratch, sw);
			/* Same for the right child. */
		for (size_t k = 0; k < K; k++) {
			const gf64_t *fk = f + k * f_stride;
			gf64_t *rk = revq + k * m_max;
			if (deg_fs[k] >= child_deg) {
				for (size_t i = 0; i < m_max; i++) {
					rk[i] = (i <= deg_fs[k]) ? fk[deg_fs[k] - i] : 0;
				}
			}
		}
		for (size_t k = 0; k < K; k++) ptrs[k] = revq + k * m_max;
		fenger_batch_shared_mul(ptrs, K, inv_R, m_max, revq, m_max,
		                        m_max, mul_scratch, sw);
		for (size_t k = 0; k < K; k++) {
			if (deg_fs[k] < child_deg) continue;
			const size_t m_true = deg_fs[k] - child_deg + 1;
			gf64_t *rk = revq + k * m_max;
			for (size_t i = 0; i < m_true / 2; i++) {
				gf64_t t = rk[i]; rk[i] = rk[m_true - 1 - i]; rk[m_true - 1 - i] = t;
			}
			if (m_true < m_max) memset(rk + m_true, 0, (m_max - m_true) * sizeof(gf64_t));
		}
		for (size_t k = 0; k < K; k++) ptrs[k] = r_R + k * N_at_lev;
		fenger_batch_shared_mul(ptrs, K, P_R, child_stride, revq, m_max,
		                        child_deg, mul_scratch, sw);

		/* Remainders + per-word degrees. */
		for (size_t k = 0; k < K; k++) {
			const gf64_t *fk = f + k * f_stride;
			gf64_t *rLk = r_L + k * N_at_lev;
			gf64_t *rRk = r_R + k * N_at_lev;
			if (deg_fs[k] >= child_deg) {
				for (size_t i = 0; i < child_deg; i++) {
					rLk[i] = fk[i] ^ rLk[i];
					rRk[i] = fk[i] ^ rRk[i];
				}
			} else {
				memcpy(rLk, fk, child_deg * sizeof(gf64_t));
				memcpy(rRk, fk, child_deg * sizeof(gf64_t));
			}
			size_t dL = 0, dR = 0;
			for (size_t i = child_deg; i-- > 0; ) {
				if (rLk[i] != 0) { dL = i; break; }
			}
			for (size_t i = child_deg; i-- > 0; ) {
				if (rRk[i] != 0) { dR = i; break; }
			}
			if (deg_fs[k] < child_deg) { dL = dR = deg_fs[k]; }
			child_degs[2 * k]     = dL;
			child_degs[2 * k + 1] = dR;
		}
	} else {
		/* Per-word fallback (identical to the single-word walk). q_L lives in
		 * revq, q_R in the separate q2 slab — the divmod contract forbids q
		 * aliasing r (it mutates r in place while writing q). */
		for (size_t k = 0; k < K; k++) {
			const gf64_t *fk = f + k * f_stride;
			gf64_t *rLk = r_L + k * N_at_lev;
			gf64_t *rRk = r_R + k * N_at_lev;
			gf64_t *q = revq + k * m_max;
			(void)q2;
			if (K == 1) {
				gf64_poly_divmod_scratch(fk, deg_fs[k], P_L, child_deg, q, rLk, arena);
				gf64_poly_divmod_scratch(fk, deg_fs[k], P_R, child_deg, revq, rRk, arena);
			} else {
				gf64_t *qR = q2 + k * m_max; /* separate from rRk (no aliasing) */
				gf64_poly_divmod_scratch(fk, deg_fs[k], P_L, child_deg, q, rLk, arena);
				gf64_poly_divmod_scratch(fk, deg_fs[k], P_R, child_deg, qR, rRk, arena);
			}
			size_t dL = 0, dR = 0;
			for (size_t i = child_deg; i-- > 0; ) {
				if (rLk[i] != 0) { dL = i; break; }
			}
			for (size_t i = child_deg; i-- > 0; ) {
				if (rRk[i] != 0) { dR = i; break; }
			}
			if (deg_fs[k] < child_deg) { dL = dR = deg_fs[k]; }
			child_degs[2 * k]     = dL;
			child_degs[2 * k + 1] = dR;
		}
		}

	/* Recurse into both children (K slabs each; the child's f degree
	 * array is child_degs[0..K) for the left and [K..2K) for the right). */
	{
		size_t left_degs[FENGER_BATCH_K_MAX], right_degs[FENGER_BATCH_K_MAX];
		for (size_t k = 0; k < K; k++) {
			left_degs[k]  = child_degs[2 * k];
			right_degs[k] = child_degs[2 * k + 1];
		}
		fenger_eval_recurse_batch(ctx, tree, lev + 1, 2 * node_idx,
		                          r_L, N_at_lev, left_degs, out, child_scratch,
		                          arena, mul_scratch, mul_scratch_words, K);
		fenger_eval_recurse_batch(ctx, tree, lev + 1, 2 * node_idx + 1,
		                          r_R, N_at_lev, right_degs, out, child_scratch,
		                          arena, mul_scratch, mul_scratch_words, K);
	}
}

/* ---- Batched execute entry (K words per pass) ---- */
static void gf64_fenger_execute_batched(
	const gf64_fenger_ctx *ctx,
	const gf64_t *in,  size_t B,
	      gf64_t *out,
	size_t w_start, size_t w_end,
	size_t K)
{
	const size_t N = ctx->N;
	const size_t R = ctx->R;
	const SubproductTree *tree_x = &ctx->tree_x;
	const SubproductTree *tree_y = &ctx->tree_y;
	const size_t deg_p = (tree_x->num_points > 1)
	                       ? (tree_x->num_points - 1) : 0;
	const size_t R_stride = tree_y->num_points;

	gf64_t *weighted = (gf64_t *)malloc(K * N * sizeof(gf64_t));
	gf64_t *poly_p   = (gf64_t *)malloc(K * N * sizeof(gf64_t));
	gf64_t *p_at_y   = (gf64_t *)malloc(K * R * sizeof(gf64_t));
	/* interp frames: 3K·N_at_lev per depth -> 6K·N total; slack 8K·N. */
	gf64_t *interp_scratch = (gf64_t *)malloc(8 * K * N * sizeof(gf64_t));
	/* eval frames: (3K+1)·N_at_lev per depth -> (6K+2)·R total — the
	 * walk is over tree_y, whose root size is R (NOT N; R > N is the
	 * walk's raison d'être). */
	gf64_t *eval_scratch = (gf64_t *)malloc((6 * K + 4) * R * sizeof(gf64_t));
	/* batch mul scratch: max eval n = 2·(R/2) = R at the root -> 4R words.
	 * The interp's scratch is 8× (32N) so the interleaved batch FFT
	 * (K <= 8) engages; the scalar/within-word paths use only the first 4N. */
	gf64_t *interp_mul = (gf64_t *)malloc(32 * N * sizeof(gf64_t));
	gf64_t *eval_mul   = (gf64_t *)malloc(4 * R * sizeof(gf64_t));
	/* T4 (issue #59): arena for the eval walk's per-node divmod/invmod
	 * scratch (Newton reciprocals + rev buffers). Degrees scale with the
	 * recovery tree (R); worst single-call demand is ~6·max(N,R) words,
	 * so 8× covers it with slack. Zero heap traffic inside the walk. */
	gf64_arena_t ev_arena;
	size_t arena_words = 8 * ((N > R) ? N : R);
	if (!weighted || !poly_p || !p_at_y ||
	    !interp_scratch || !eval_scratch || !interp_mul || !eval_mul ||
	    gf64_arena_init(&ev_arena, arena_words) != 0) {
		abort();
	}

	const size_t mul_sw = gf64_addfft64_poly_mul_recursive_scratch_words(N);

	for (size_t w0 = w_start; w0 < w_end; w0 += K) {
		const size_t K_eff = (w_end - w0 < K) ? (w_end - w0) : K;

		/* 4a: weighted slabs. Synthetic padded inputs (c >= real) have
		 * zero DATA (never read from `in`), so weight them 0.
		 *
		 * Issue #59 T3: per-word strided gather into the contiguous
		 * `weighted` slab, then a contiguous ISA-dispatched
		 * element-wise mul. The gather reads the same bytes the
		 * scalar mul would have read (one word per c per word k);
		 * the mul is bit-exact to gf64_mul_reference (see
		 * test_gf64_elem_mul.c). Padded region (c >= num_inputs_real)
		 * gets memset to 0; the subsequent mul reads V_prime[c]
		 * for those indices (a valid array index) but produces 0
		 * because the operand is 0. */
		for (size_t k = 0; k < K_eff; k++) {
			const size_t w = w0 + k;
			gf64_t *wk = weighted + k * N;
			for (size_t c = 0; c < ctx->num_inputs_real; c++) {
				wk[c] = in[c * B + w];
			}
			if (ctx->num_inputs_real < N) {
				memset(wk + ctx->num_inputs_real, 0,
				       (N - ctx->num_inputs_real) * sizeof(gf64_t));
			}
		}
		for (size_t k = 0; k < K_eff; k++) {
			gf64_t *wk = weighted + k * N;
			gf64_elem_mul(wk, wk, ctx->V_prime, N);
		}
		/* 4b: z = weighted · 1/V'(x_c), then the batched combine.
		 * `weighted` slabs are contiguous — direct ISA-dispatched
		 * element-wise mul, aliasing-safe (wk == a). */
		for (size_t k = 0; k < K_eff; k++) {
			gf64_t *zk = weighted + k * N;
			gf64_elem_mul(zk, zk, ctx->V_prime_inv, N);
		}
		fenger_interp_batch(tree_x, weighted, K_eff, poly_p,
		                    interp_scratch, interp_mul, 32 * N);

		/* 4c: batched MPE over the recovery tree. The tree walk is only
		 * valid for deg_p < R (the interpolation bound); for R == 1 or
		 * deg_p >= R, mirror gf64_multi_point_eval's Horner fallback.
		 * The interp output (poly_p) is N-strided per word — the eval
		 * root's f_stride. */
		{
			size_t degs[FENGER_BATCH_K_MAX];
			if (R == 1 || deg_p >= R) {
				const gf64_t *leaves =
					tree_y->level_data[tree_y->num_levels - 1];
				for (size_t k = 0; k < K_eff; k++) {
					const gf64_t *fk = poly_p + k * N;
					gf64_t *pk = p_at_y + k * R;
					for (size_t j = 0; j < R; j++) {
						pk[j] = fenger_horner(fk, deg_p, leaves[2 * j]);
					}
				}
			} else {
				for (size_t k = 0; k < K_eff; k++) degs[k] = deg_p;
				fenger_eval_recurse_batch(ctx, tree_y, 0, 0, poly_p, N, degs,
				                          p_at_y, eval_scratch, &ev_arena,
				                          eval_mul, mul_sw, K_eff);
			}
		}
		/* 4d: divide by V(y_r) — only real recovery rows are written.
		 *
		 * Issue #59 T3: contiguous vectorized mul into the per-word
		 * `p_at_y` slab, then strided scatter to the caller's
		 * column-major buffer. `p_at_y` is private scratch (allocated
		 * above, freed at the end of this call), so the in-place
		 * mul (out == a aliasing) is safe. */
		for (size_t k = 0; k < K_eff; k++) {
			const size_t w = w0 + k;
			gf64_t *pk = p_at_y + k * R_stride;
			if (ctx->num_recovery_real > 0) {
				gf64_elem_mul(pk, pk, ctx->V_at_y_inv,
				              ctx->num_recovery_real);
				for (size_t r = 0; r < ctx->num_recovery_real; r++) {
					out[r * B + w] = pk[r];
				}
			}
		}
	}

	free(weighted); free(poly_p); free(p_at_y);
	free(interp_scratch); free(eval_scratch);
	free(interp_mul); free(eval_mul);
	gf64_arena_free(&ev_arena);
}

/* ----------------------------------------------------------------------------
 * 3. gf64_fenger_release.
 * --------------------------------------------------------------------------*/
void gf64_fenger_release(gf64_fenger_ctx *ctx) {
	if (ctx == NULL) return;
	gf64_subproduct_tree_free(&ctx->tree_x);
	gf64_subproduct_tree_free(&ctx->tree_y);
	free(ctx->V_prime);
	free(ctx->V_prime_inv);
	free(ctx->V_at_y_inv);
	free(ctx->recip_pool);
	free(ctx);
}

/* ----------------------------------------------------------------------------
 * 4. Convenience wrapper: prepare + execute (single thread, full B) + release.
 * --------------------------------------------------------------------------*/
void gf64_fenger_matvec(
	const gf64_t *in,  size_t N, size_t B,
	      gf64_t *out, size_t R,
	uint64_t firstInput,
	uint64_t firstRecovery
) {
	if (N == 0 || R == 0 || B == 0) {
		return;
	}
	gf64_fenger_ctx *ctx = gf64_fenger_prepare(
		firstInput, firstRecovery, N, R
	);
	gf64_fenger_execute(ctx, in, B, out, 0, B);
	gf64_fenger_release(ctx);
}
