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
 *      N - 1 to Horner is always safe — the leading zero just propagates
 *      and one extra iteration is wasted.
 *
 *   2. Evaluate P'(x_j) at each input point. The naive O(N^2) Horner loop
 *      below is the T7 stop-gap. The plan describes a Bostan-Schost
 *      multi-point evaluation that drives the root's evaluation in
 *      O(N log^2 N) using the pre-built subproduct tree — that is T8
 *      (gf64_mpe.{c,h}) and not yet implemented.
 *
 *   3. Invert the N values in one shot via Itoh-Tsujii batched inversion
 *      (gf64_invert_ita_batch from T5).
 *
 * Allocation strategy: the derivative (N coefficients) and the
 * deriv-at-points buffer (N values) are temporary scratch — both are
 * heap-allocated and freed before return. (Future T8 will be free of the
 * second allocation as well.)
 * ============================================================================
 */

#include "gf64_barycentric.h"
#include "gf64_global.h"
#include "gf64_invert_ita.h"
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

/*
 * Horner evaluation of a constant-first polynomial c[0..deg] at x = r.
 * Returns 0 when the polynomial is empty (deg == (size_t)(-1) sentinel).
 *
 * For characteristic 2 the "addition" step is XOR. This is the same
 * evaluation loop that T8 (Bostan-Schost MPE) will need internally; the
 * helper is kept static here and T8 may either lift it to its own TU or
 * share it via the header.
 */
static gf64_t horner_eval_poly(const gf64_t *c, size_t deg, gf64_t r) {
	if (deg == (size_t)(-(intptr_t)1)) {
		/* Empty polynomial sentinel. Not expected in T7 because the root
		 * of any non-empty subproduct tree has degree N >= 1, so P' has
		 * degree N-1 >= 0. Guarded defensively. */
		return 0;
	}

	gf64_t acc = c[deg];
	for (size_t i = deg; i > 0; i--) {
		acc = gf64_mul_reference(acc, r) ^ c[i - 1];
	}
	return acc;
}

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
	 * TODO(gf64-mpe, T8): replace this naive O(N^2) Horner loop with the
	 * Bostan-Schost multi-point evaluation that walks the subproduct tree
	 * from leaves to root in O(N log^2 N). The Horner loop here is the T7
	 * stop-gap agreed in the plan; the existing subproduct-tree layout is
	 * exactly what T8 will consume.
	 *
	 * Input points are stored at the leaves (level_data[num_levels-1]):
	 * leaf j has coefficients [x_j, 1] (constant-first). We read only
	 * the constant term, leaf_j[0] = x_j.
	 */
	gf64_t *deriv_at_points = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (deriv_at_points == NULL) {
		free(deriv);
		abort();
	}

	{
		const gf64_t *HEDLEY_RESTRICT leaves = tree->level_data[tree->num_levels - 1];
		const size_t deriv_degree = (N == 0) ? (size_t)(-(intptr_t)1) : (N - 1);
		/*
		 * N == 0 is short-circuited above; for N >= 1 the derivative has
		 * degree N - 1, so deriv_degree is well-defined and >= 0.
		 */
		for (size_t j = 0; j < N; j++) {
			const gf64_t xj = leaves[2 * j];
			deriv_at_points[j] = horner_eval_poly(deriv, deriv_degree, xj);
		}
	}

	/*
	 * Step 3: Batch invert via Itoh-Tsujii (T5).
	 *
	 * weights_out[j] = 1 / deriv_at_points[j].
	 * For inputs equal to 0 (which arise when some x_j is duplicated, so
	 * that point is a repeated root of P), the invert_ita_batch entry
	 * point returns 0 by convention. The barycentric weight is therefore
	 * 0 for that j — matching the "duplicate input -> zero weight"
	 * edge case in the plan.
	 *
	 * gf64_invert_ita_batch is compiled with __attribute__((target
	 * ("avx512f,vpclmulqdq"))) and executes ZMM instructions
	 * UNCONDITIONALLY — calling it on an AVX2-only host SIGILLs. Gate it
	 * on the runtime dispatch (set by gf64_init_dispatch / gf64_apply_method)
	 * and fall back to the scalar one-at-a-time inverter otherwise.
	 */
	if (gf64_current_method == GF64_AVX512) {
		gf64_invert_ita_batch(weights_out, deriv_at_points, N);
	} else {
		for (size_t j = 0; j < N; j++) {
			weights_out[j] = gf64_invert_ita_one(deriv_at_points[j]);
		}
	}

	free(deriv_at_points);
	free(deriv);
}

HEDLEY_END_C_DECLS
