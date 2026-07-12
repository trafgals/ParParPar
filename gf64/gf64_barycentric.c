/*
 * ============================================================================
 * gf64/gf64_barycentric.c — Barycentric weight extraction over GF(2^64).
 *
 * T7 of the par3-cauchy-fft-kernel plan.
 *
 * Given a subproduct tree built over input points {x_0, ..., x_{N-1}},
 * computes barycentric weights
 *
 *     W_j = 1 / P'(x_j)    for j = 0..N-1
 *
 * where P(x) = ∏_i (x + x_i) is the tree root and P'(x) is its formal
 * derivative in GF(2^64).
 *
 * The flow is:
 *   1. Extract P'(x) from the root polynomial — in char 2, the derivative's
 *      coefficient array of length N drops the constant and even-indexed
 *      coefficients to zero, and copies the odd-indexed coefficients of P
 *      one position to the left:
 *
 *         deriv[j] = root[j + 1]   if j is even
 *         deriv[j] = 0             if j is odd   (for j = 0..N-1)
 *
 *      P' has degree N - 1 (N odd) or N - 2 (N even), so passing degree
 *      N - 1 to the MPE is always safe — the leading zero just propagates.
 *
 *   2. Evaluate P'(x_j) at each input point via gf64_multi_point_eval
 *      (T8 — gf64_mpe.{c,h}). The subproduct tree is built over the
 *      input points {x_j}; MPE walks its leaves and writes P'(x_j)
 *      directly into deriv_at_points. No local Horner loop here.
 *
 *   3. Invert the N values in one shot via Itoh-Tsujii batched inversion
 *      (gf64_invert_ita_batch from T5).
 *
 * Allocation strategy: deriv (N coefficients) and deriv_at_points (N values)
 * are still temporary scratch — both heap-allocated and freed before return.
 * When PR-2 lands an FFT-based MPE the second allocation may be moveable
 * into a caller-provided buffer; until then the malloc churn stays.
 * ============================================================================
 */

#include "gf64_barycentric.h"
#include "gf64_global.h"
#include "gf64_invert_ita.h"
#include "gf64_mpe.h"
#include "gf64_subproduct.h"

#include <stdint.h>
#include <stdlib.h>

HEDLEY_BEGIN_C_DECLS

/* Public reference for field multiplication, declared in gf64_single.c.
 * The scalar CLMUL-based reference is appropriate here: T7 is a correctness
 * milestone (the O(N^2) Horner loop will be replaced by T8's Bostan-Schost
 * MPE); using the ISA-independent reference avoids a self-test ordering
 * problem with the build target's ISA gating. */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

void gf64_barycentric_weights(const SubproductTree *tree, gf64_t *weights_out) {
	if (tree == NULL || tree->num_points == 0) {
		return;
	}

	const size_t N = tree->num_points;

	/*
	 * Step 1: Formal derivative of the root polynomial.
	 *
	 * The root polynomial at level_data[0] has degree N, so it occupies
	 * N + 1 coefficients root[0..N] (constant-first). In characteristic 2,
	 *
	 *     P(x)  = sum_{k=0}^{N}  root[k] * x^k
	 *     P'(x) = sum_{k odd, k>=1} root[k] * x^{k-1}
	 *
	 * which puts a copy of root[k+1] at the even index k (and zero at the
	 * odd indices) in `deriv`:
	 *
	 *     deriv[j] = root[j + 1]   if j is even
	 *     deriv[j] = 0             if j is odd
	 *
	 * The allocated buffer has N slots. The actual degree of P' is
	 * N - 1 (N odd) or N - 2 (N even); using deg = N - 1 below is safe
	 * because the leading zero pads one extra iteration of Horner harmlessly.
	 *
	 * Edge cases:
	 *   - N = 1: root = [x_0, 1]; loop reads root[1] = 1 → deriv = [1].
	 *            P'(x) = 1 → W_0 = 1.
	 *   - N = 0: early-returned above (root has no coefficients anyway).
	 */
	gf64_t *deriv = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (deriv == NULL) {
		abort();
	}

	{
		const gf64_t *HEDLEY_RESTRICT root = tree->level_data[0];
		for (size_t j = 0; j < N; j++) {
			deriv[j] = ((j & 1U) == 0U) ? root[j + 1] : (gf64_t)0;
		}
	}

	/*
	 * Step 2: Evaluate P'(x_j) at each input point x_j.
	 *
	 * Now delegated to gf64_multi_point_eval (T8) — the subproduct tree
	 * was built over the input points {x_j}, so MPE walks its leaves and
	 * writes P'(x_j) directly into deriv_at_points.
	 *
	 * The current implementation in gf64_multi_point_eval is a naive
	 * Horner per leaf (O(N * deg_f)) — bit-equivalent to the T7 stop-gap.
	 * PR-2 will replace that body with a top-down Bostan-Schost walk once
	 * the FFT-multiply primitive is available. The public signature of
	 * gf64_multi_point_eval is stable, so this swap here is permanent.
	 *
	 * Input points are stored at the leaves (level_data[num_levels-1]):
	 * leaf j has coefficients [x_j, 1] (constant-first). MPE consumes
	 * this layout directly — no manual leaf walk needed at this call site.
	 */
	gf64_t *deriv_at_points = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (deriv_at_points == NULL) {
		free(deriv);
		abort();
	}

	/*
	 * deriv has degree at most N - 1 (formal derivative of a degree-N P).
	 * For N >= 1 this is well-defined and >= 0; N == 0 is short-circuited
	 * at function entry.
	 */
	const size_t deriv_degree = (N == 0) ? (size_t)(-(intptr_t)1) : (N - 1);
	gf64_multi_point_eval(deriv, deriv_degree, tree, deriv_at_points);

	/*
	 * Step 3: Batch invert via Itoh-Tsujii (T5).
	 *
	 * weights_out[j] = 1 / deriv_at_points[j].
	 * For inputs equal to 0 (which arise when some x_j is duplicated, so
	 * that point is a repeated root of P), the invert_ita_batch entry
	 * point returns 0 by convention. The barycentric weight is therefore
	 * 0 for that j — matching the "duplicate input -> zero weight"
	 * edge case in the plan.
	 */
	gf64_invert_ita_batch(weights_out, deriv_at_points, N);

	free(deriv_at_points);
	free(deriv);
}

HEDLEY_END_C_DECLS
