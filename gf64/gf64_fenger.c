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
 * top-down via gf64_multi_point_interp (T8b) which already uses the
 * cached inv_mod_data from the tree build. The per-word evaluation step
 * is Bostan-Schost top-down via gf64_multi_point_eval (T8a). The product
 * gives p(y_r) in O((N + R) log² (N + R)) per word.
 *
 * Field operations are delegated to gf64_mul_reference (scalar SSE2) and
 * gf64_invert_ita_one (T5). The AVX-512 swap-in is deferred; the
 * pipeline's call shape is independent of the field-multiplication
 * primitive.
 * ============================================================================
 */

#include "gf64_fenger.h"
#include "gf64_barycentric.h"
#include "gf64_invert_ita.h"
#include "gf64_mpe.h"
#include "gf64_mul.h"
#include "gf64_subproduct.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Scalar SSE2 reference mul, declared in gf64_single.c. */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* ----------------------------------------------------------------------------
 * Opaque context: holds everything shared across per-word execute calls.
 * --------------------------------------------------------------------------*/
struct gf64_fenger_ctx {
	SubproductTree tree_x; /* input-point tree       */
	SubproductTree tree_y; /* recovery-point tree    */
	gf64_t *V_prime;       /* V'(x_c) for each input */
	gf64_t *V_at_y_inv;    /* 1/V(y_r) for each r    */
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

	/* ---- Barycentric weights (1/V'(x_c)) then invert to V'(x_c) ---- */
	gf64_t *bary = (gf64_t *)malloc(numInputsPadded * sizeof(gf64_t));
	if (bary == NULL) abort();
	gf64_barycentric_weights(&ctx->tree_x, bary);

	ctx->V_prime = (gf64_t *)malloc(numInputsPadded * sizeof(gf64_t));
	if (ctx->V_prime == NULL) abort();
	for (size_t c = 0; c < numInputsPadded; c++) {
		ctx->V_prime[c] = gf64_invert_ita_one(bary[c]);
	}
	free(bary);

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
	for (size_t r = 0; r < numRecoveryPadded; r++) {
		ctx->V_at_y_inv[r] = gf64_invert_ita_one(V_at_y[r]);
	}
	free(V_at_y);

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
		 * holds real blocks), so weight them 0 without reading `in`. */
		for (size_t c = 0; c < N; c++) {
			if (c < ctx->num_inputs_real) {
				weighted[c] = gf64_mul_reference(in[c * B + w], ctx->V_prime[c]);
			} else {
				weighted[c] = 0;
			}
		}
		/* 4b: interpolate to polynomial p_w of degree < N. Fenger is the
		 * sanctioned consumer of the Bostan-Schost body, so it calls the
		 * ungated gf64_multi_point_interp_internal directly — the
		 * PAR3_GF64_USE_INTERP opt-in gate is for production code that
		 * does not require interpolation. */
		gf64_multi_point_interp_internal(tree_x, weighted, poly_p);
		/* 4c: evaluate p_w at each recovery point. */
		gf64_multi_point_eval(poly_p, deg_p, tree_y, p_at_y);
		/* 4d: divide by V(y_r) — only real recovery rows are written
		 * (synthetic padded rows would overflow the caller's buffer). */
		for (size_t r = 0; r < ctx->num_recovery_real; r++) {
			out[r * B + w] = gf64_mul_reference(p_at_y[r], ctx->V_at_y_inv[r]);
		}
	}

	free(weighted);
	free(poly_p);
	free(p_at_y);
}

/* ----------------------------------------------------------------------------
 * 3. gf64_fenger_release.
 * --------------------------------------------------------------------------*/
void gf64_fenger_release(gf64_fenger_ctx *ctx) {
	if (ctx == NULL) return;
	gf64_subproduct_tree_free(&ctx->tree_x);
	gf64_subproduct_tree_free(&ctx->tree_y);
	free(ctx->V_prime);
	free(ctx->V_at_y_inv);
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
