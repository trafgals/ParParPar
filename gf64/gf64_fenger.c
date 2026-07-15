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
	size_t  N;
	size_t  R;
};

/* ----------------------------------------------------------------------------
 * 1. gf64_fenger_prepare: build trees, weights, V(y_r) inverse.
 * --------------------------------------------------------------------------*/
gf64_fenger_ctx *gf64_fenger_prepare(
	uint64_t firstInput,
	uint64_t firstRecovery,
	size_t N,
	size_t R
) {
	if (N == 0 || R == 0) {
		return NULL;
	}

	gf64_fenger_ctx *ctx = (gf64_fenger_ctx *)malloc(sizeof(gf64_fenger_ctx));
	if (ctx == NULL) abort();
	ctx->N = N;
	ctx->R = R;
	ctx->V_prime    = NULL;
	ctx->V_at_y_inv = NULL;

	/* ---- Build T_X over input points ---- */
	gf64_t *x_points = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (x_points == NULL) abort();
	for (size_t c = 0; c < N; c++) {
		x_points[c] = (gf64_t)(firstInput + (uint64_t)c);
	}
	gf64_subproduct_tree_build(x_points, N, &ctx->tree_x);
	free(x_points);

	/* ---- Build T_Y over recovery points ---- */
	gf64_t *y_points = (gf64_t *)malloc(R * sizeof(gf64_t));
	if (y_points == NULL) abort();
	for (size_t r = 0; r < R; r++) {
		y_points[r] = (gf64_t)(firstRecovery + (uint64_t)r);
	}
	gf64_subproduct_tree_build(y_points, R, &ctx->tree_y);
	free(y_points);

	/* ---- Barycentric weights (1/V'(x_c)) then invert to V'(x_c) ---- */
	gf64_t *bary = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (bary == NULL) abort();
	gf64_barycentric_weights(&ctx->tree_x, bary);

	ctx->V_prime = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (ctx->V_prime == NULL) abort();
	for (size_t c = 0; c < N; c++) {
		ctx->V_prime[c] = gf64_invert_ita_one(bary[c]);
	}
	free(bary);

	/* ---- V(y_r) for each recovery point (MPE on tree_y) ---- */
	gf64_t *V_at_y = (gf64_t *)malloc(R * sizeof(gf64_t));
	if (V_at_y == NULL) abort();
	gf64_multi_point_eval(
		ctx->tree_x.level_data[0],
		ctx->tree_x.num_points,
		&ctx->tree_y,
		V_at_y
	);

	ctx->V_at_y_inv = (gf64_t *)malloc(R * sizeof(gf64_t));
	if (ctx->V_at_y_inv == NULL) abort();
	for (size_t r = 0; r < R; r++) {
		ctx->V_at_y_inv[r] = gf64_invert_ita_one(V_at_y[r]);
	}
	free(V_at_y);

	return ctx;
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
		/* 4a: weighted values. */
		for (size_t c = 0; c < N; c++) {
			weighted[c] = gf64_mul_reference(in[c * B + w], ctx->V_prime[c]);
		}
		/* 4b: interpolate to polynomial p_w of degree < N. */
		gf64_multi_point_interp(tree_x, weighted, poly_p);
		/* 4c: evaluate p_w at each recovery point. */
		gf64_multi_point_eval(poly_p, deg_p, tree_y, p_at_y);
		/* 4d: divide by V(y_r). */
		for (size_t r = 0; r < R; r++) {
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
