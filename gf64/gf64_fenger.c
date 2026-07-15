/*
 * ============================================================================
 * gf64/gf64_fenger.c — Fenger Toeplitz pipeline (issue #28).
 *
 * See gf64_fenger.h for the algorithm and asymptotics. This file is the
 * reference implementation: each call builds the trees, computes the
 * weights, and runs the per-word pipeline. Caller-side caching of T_X +
 * V'(x_c) across recovery runs is deferred to Phase 2 (when the NAPI
 * binding learns the {input-chunk-set} lifetime).
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

void gf64_fenger_matvec(
	const gf64_t *in,  size_t N, size_t B,
	      gf64_t *out, size_t R,
	uint64_t firstInput,
	uint64_t firstRecovery
) {
	/* Empty cases: no work, no allocation. */
	if (N == 0 || R == 0 || B == 0) {
		return;
	}

	/* ---------------------------------------------------------------------
	 * Step 1a: build subproduct tree T_X over the input points.
	 * ------------------------------------------------------------------- */
	gf64_t *x_points = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (x_points == NULL) abort();
	for (size_t c = 0; c < N; c++) {
		x_points[c] = (gf64_t)(firstInput + (uint64_t)c);
	}
	SubproductTree tree_x;
	gf64_subproduct_tree_build(x_points, N, &tree_x);
	free(x_points);

	/* ---------------------------------------------------------------------
	 * Step 1b: build subproduct tree T_Y over the recovery points.
	 * ------------------------------------------------------------------- */
	gf64_t *y_points = (gf64_t *)malloc(R * sizeof(gf64_t));
	if (y_points == NULL) abort();
	for (size_t r = 0; r < R; r++) {
		y_points[r] = (gf64_t)(firstRecovery + (uint64_t)r);
	}
	SubproductTree tree_y;
	gf64_subproduct_tree_build(y_points, R, &tree_y);
	free(y_points);

	/* ---------------------------------------------------------------------
	 * Step 2: barycentric weights (1 / V'(x_c)) at each input point.
	 *
	 * gf64_barycentric_weights currently uses an O(N²) Horner loop in
	 * step 2 (per the gf64_barycentric.h TODO). For Phase 1 we accept
	 * this — it's amortized across B words and across multiple recovery
	 * runs once Phase 2 caches tree_x + weights per input-chunk-set.
	 * ------------------------------------------------------------------- */
	gf64_t *bary = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (bary == NULL) abort();
	gf64_barycentric_weights(&tree_x, bary);

	/* V'(x_c) = 1 / bary[c] (since bary[c] = 1 / V'(x_c)). */
	gf64_t *V_prime = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (V_prime == NULL) abort();
	for (size_t c = 0; c < N; c++) {
		V_prime[c] = gf64_invert_ita_one(bary[c]);
	}

	/* ---------------------------------------------------------------------
	 * Step 3: V(y_r) for each recovery point. The root polynomial of T_X
	 * is V(x), with degree N and coefficients constant-first at
	 * tree_x.level_data[0]. Evaluating V at T_Y's leaves uses Bostan-
	 * Schost top-down MPE: O((N + R) log² (N + R)) total — same cost
	 * shape as the per-word eval, but a single call amortized across
	 * all B words.
	 * ------------------------------------------------------------------- */
	gf64_t *V_at_y = (gf64_t *)malloc(R * sizeof(gf64_t));
	if (V_at_y == NULL) abort();
	gf64_multi_point_eval(
		tree_x.level_data[0],
		(tree_x.num_points > 0 ? tree_x.num_points : 0),
		&tree_y,
		V_at_y
	);

	/* Pre-invert V(y_r). */
	gf64_t *V_at_y_inv = (gf64_t *)malloc(R * sizeof(gf64_t));
	if (V_at_y_inv == NULL) abort();
	for (size_t r = 0; r < R; r++) {
		V_at_y_inv[r] = gf64_invert_ita_one(V_at_y[r]);
	}

	/* ---------------------------------------------------------------------
	 * Per-word working buffers.
	 * ------------------------------------------------------------------- */
	gf64_t *weighted = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *poly_p   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *p_at_y   = (gf64_t *)malloc(R * sizeof(gf64_t));
	if (weighted == NULL || poly_p == NULL || p_at_y == NULL) {
		abort();
	}

	/* ---------------------------------------------------------------------
	 * Step 4: per-word pipeline.
	 *
	 *   weighted[c] = in[c][w] * V'(x_c)
	 *   p_w(x)     = interp(weighted at x_c)
	 *   p_at_y[r]  = p_w(y_r)
	 *   out[r][w]  = p_at_y[r] / V(y_r)
	 * ------------------------------------------------------------------- */
	for (size_t w = 0; w < B; w++) {
		/* 4a: weighted values. */
		for (size_t c = 0; c < N; c++) {
			weighted[c] = gf64_mul_reference(in[c * B + w], V_prime[c]);
		}
		/* 4b: interpolate to polynomial p_w of degree < N. */
		gf64_multi_point_interp(&tree_x, weighted, poly_p);
		/* 4c: evaluate p_w at each recovery point. The poly has at most
		 * N-1 degree (Bostan-Schost interp contract), but the MPE
		 * accepts "deg" as a hint. Pass N-1 to keep the scratch tight;
		 * the leading coefficient (poly_p[N-1]) is always zero. */
		gf64_multi_point_eval(
			poly_p,
			(tree_x.num_points > 1 ? tree_x.num_points - 1 : 0),
			&tree_y,
			p_at_y
		);
		/* 4d: divide by V(y_r). */
		for (size_t r = 0; r < R; r++) {
			out[r * B + w] = gf64_mul_reference(p_at_y[r], V_at_y_inv[r]);
		}
	}

	/* Cleanup. */
	free(weighted);
	free(poly_p);
	free(p_at_y);
	free(bary);
	free(V_prime);
	free(V_at_y);
	free(V_at_y_inv);
	gf64_subproduct_tree_free(&tree_x);
	gf64_subproduct_tree_free(&tree_y);
}
