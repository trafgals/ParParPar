/*
 * ============================================================================
 * src/par3_engine_barycentric.cc — Barycentric Lagrange interpolation kernel
 *
 * T9 of the par3-cauchy-fft-kernel plan.
 *
 * Implements GF64Controller::ComputeRecoveryBlocksBarycentric. The kernel
 * computes, for each output block r and word position w:
 *
 *     out[r][w] = num[w](y_r) / den(y_r)
 *
 * where the input points x_c = firstInput + c, the recovery points
 * y_r = firstRecovery + r, the barycentric weights W_j = 1 / P'(x_j)
 * (computed via T7's gf64_barycentric_weights), and
 *
 *     num[w](y) = sum_j in[j][w] * W_j / (y + x_j)
 *     den(y)    = sum_j W_j / (y + x_j)
 *
 * Equivalently (via the partial-fraction identity sum_j W_j/(y+x_j) = 1/P(y)),
 * the output is the Lagrange interpolation of {in[j][w]} at the input
 * points evaluated at the recovery points — *not* the Cauchy matrix-vector
 * product (which is a different linear map; see par3_engine.h for the
 * kernel-vs-Cauchy note).
 *
 * PIPELINE
 * --------
 *   1. Build the input-point grid x_c = firstInput + c for c in [0, numInputs).
 *   2. Build a subproduct tree over those points (T6).
 *      - The tree requires numInputs to be 0, 1, or a power of 2. For non
 *        powers of two we round numInputs up to the next power of two and
 *        pad the leaves with arbitrary values; the corresponding weights
 *        produced by T7's gf64_barycentric_weights will be undefined for the
 *        padded slots, so we must NOT read them. The inner loop is bounded
 *        to j < numInputs, which discards the padded slots.
 *   3. Compute barycentric weights W_j for the tree's first numInputs leaves
 *      (T7's gf64_barycentric_weights). The remaining padded slots are
 *      ignored by the loop in step 5 below.
 *   4. Build the recovery-point grid y_r = firstRecovery + r. No tree
 *      required for recovery points — we evaluate directly using the
 *      per-point partial-fraction identity.
 *   5. For each r, compute den(y_r) once (loop body: 1 mul + 1 invert + 1
 *      XOR per input j).
 *   6. For each (r, w) triple, compute the numerator sum and the final
 *      division by den(y_r):
 *        - num_sum[r][w] = sum_j in[j][w] * W_j * inv(y_r + x_j)
 *        - out[r][w]     = num_sum[r][w] * inv(den(y_r))
 *
 * CURRENT VERSION
 * ---------------
 * T9 first version uses a naive O(N²) per-recovery inner loop. The accumulator
 * recomputes the full sum for every (r, w) pair. The TODO below marks the
 * Bostan-Schost fast-path landing site.
 *
 * TODO(bostan-schost-fast-path): replace the inner j-loop with a Bostan-Schost
 * multi-point evaluation. The numerator polynomial num[w](y) is a polynomial
 * of degree N-1 in y (after multiplying through by the input subproduct P(y)
 * the partial fractions collapse into a single polynomial). Evaluating it at
 * all numRecovery recovery points via the tree-driven multi-point evaluator
 * reduces the per-call cost from O(numRecovery * numInputs * blockSize64) to
 * O((D + R) log²(D + R)) where D = numInputs - 1 and R = numRecovery. The
 * shared denominator den(y_r) is evaluated once via the same MPE applied to
 * the constant-summed weight polynomial. T8's currently-Horner fallback
 * gf64_multi_point_eval will become the O(N log²N) Bostan-Schost evaluator
 * before this fast-path goes live.
 * ============================================================================
 */

#include "par3_engine.h"

extern "C" {
#include "../gf64/gf64_barycentric.h"
#include "../gf64/gf64_mpe.h"
#include "../gf64/gf64_subproduct.h"
#include "../gf64/gf64_mul.h"
#include "../gf64/gf64_invert_ita.h"
}

#include <cstring>
#include <vector>

/*
 * Field multiplication primitive. Public reference for GF(2^64) multiply,
 * declared in gf64_single.c (T1). Bit-exact to gf64_mul in gf64_solve.c.
 * Using the ISA-independent reference avoids build-target ISA gating here:
 * the barycentric kernel's correctness contract is "matches the JS reference
 * at the bit level", which is what gf64_mul_reference gives us.
 */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/*
 * GF(2^64) is the field of characteristic 2, so subtraction equals XOR
 * equals addition. Express that as a tiny inline helper so the loop bodies
 * read as math, not bit ops.
 */
static inline gf64_t gf_add(gf64_t a, gf64_t b) {
	return a ^ b;
}

void GF64Controller::ComputeRecoveryBlocksBarycentric(
	const gf64_t* inputs, size_t numInputs,
	gf64_t* recovery, size_t numRecovery,
	size_t blockSize64,
	uint64_t firstInput, uint64_t firstRecovery,
	int /*numThreads — unused in T9 first version, see TODO above*/
) {
	/*
	 * Trivial-input short-circuit. Matches the engine convention (see
	 * ComputeRecoveryBlocks) — empty problems return without writing the
	 * output and without doing any work.
	 */
	if (numInputs == 0 || numRecovery == 0 || blockSize64 == 0) {
		return;
	}

	/*
	 * Step 1: Build input points x_c = firstInput + c.
	 *
	 * In char-2 GF(2^64), "addition" of integers is XOR of the field
	 * elements. The values firstInput + c, etc. are interpreted as field
	 * elements (their bit patterns), not as integers in Z. So
	 *
	 *     x_c = (gf64_t)(firstInput + c)
	 *
	 * is the natural construction: re-interpret the integer x as a
	 * 64-bit field element. We use this same construction for the
	 * recovery points below.
	 */
	std::vector<gf64_t> input_points(numInputs);
	for (size_t c = 0; c < numInputs; c++) {
		input_points[c] = (gf64_t)(firstInput + c);
	}

	/*
	 * Step 2: Build the subproduct tree.
	 *
	 * gf64_subproduct_tree_build requires N to be 0, 1, or a power of 2
	 * (see gf64_subproduct.h:29). For non-power-of-two numInputs we pad
	 * the leaves up to the next power of two with arbitrary values; the
	 * corresponding barycentric weights will be ill-defined for those
	 * padded slots, but the loops below bound to j < numInputs so the
	 * padded weights are never read. The padding values themselves are
	 * unimportant.
	 */
	size_t N_pow2 = 1;
	while (N_pow2 < numInputs) N_pow2 *= 2;

	SubproductTree tree;
	if (N_pow2 == numInputs) {
		gf64_subproduct_tree_build(input_points.data(), numInputs, &tree);
	} else {
		/* Pad with placeholder leaves. */
		std::vector<gf64_t> padded(N_pow2);
		for (size_t i = 0; i < numInputs; i++) padded[i] = input_points[i];
		for (size_t i = numInputs; i < N_pow2; i++) padded[i] = (gf64_t)i;
		gf64_subproduct_tree_build(padded.data(), N_pow2, &tree);
	}

	/*
	 * Step 3: Compute barycentric weights W_j = 1 / P'(x_j) for the
	 * leaves we care about. The padded slots (j >= numInputs) get
	 * weights that are not meaningful; we do not read them.
	 *
	 * gf64_barycentric_weights requires a buffer of N_pow2 gf64_t slots.
	 */
	std::vector<gf64_t> weights(N_pow2);
	gf64_barycentric_weights(&tree, weights.data());

	/*
	 * Step 4: Build the recovery point grid y_r = firstRecovery + r. No
	 * tree is built here — the naive T9 kernel evaluates each y_r
	 * individually.
	 */
	std::vector<gf64_t> recovery_points(numRecovery);
	for (size_t r = 0; r < numRecovery; r++) {
		recovery_points[r] = (gf64_t)(firstRecovery + r);
	}

	/*
	 * Step 5: Precompute den(y_r) = sum_j W_j / (y_r + x_j) once per
	 * recovery point. The result is independent of w (it's the
	 * denominator of the barycentric formula), so we do this in a
	 * single r-only loop and cache the value for the (r, w) loop below.
	 *
	 * T9 inner loop (per r): numInputs GF(2^64) inversions + numInputs
	 * field multiplications + numInputs GF(2^64) XORs.
	 */
	std::vector<gf64_t> den_at_recovery(numRecovery);
	for (size_t r = 0; r < numRecovery; r++) {
		const gf64_t yr = recovery_points[r];
		gf64_t acc = 0;
		for (size_t j = 0; j < numInputs; j++) {
			const gf64_t xj = input_points[j];
			const gf64_t denom_term = gf_add(yr, xj);
			const gf64_t inv_denom = gf64_invert_ita_one(denom_term);
			acc ^= gf64_mul_reference(weights[j], inv_denom);
		}
		den_at_recovery[r] = acc;
	}

	/*
	 * Step 6: For each word position w and recovery point r, compute
	 *
	 *     num_sum[r][w] = sum_j in[j][w] * W_j * inv(y_r + x_j)
	 *     out[r][w]     = num_sum[r][w] * inv(den(y_r))
	 *
	 * The naive loop below is O(numRecovery * numInputs * blockSize64).
	 * The TODO at the top of this file marks the Bostan-Schost fast-path
	 * landing site.
	 */
	for (size_t w = 0; w < blockSize64; w++) {
		for (size_t r = 0; r < numRecovery; r++) {
			const gf64_t yr = recovery_points[r];
			gf64_t acc = 0;
			for (size_t j = 0; j < numInputs; j++) {
				const gf64_t xj = input_points[j];
				const gf64_t in_jw = inputs[j * blockSize64 + w];
				const gf64_t denom_term = gf_add(yr, xj);
				const gf64_t inv_denom = gf64_invert_ita_one(denom_term);
				const gf64_t num_term = gf64_mul_reference(in_jw, weights[j]);
				acc ^= gf64_mul_reference(num_term, inv_denom);
			}
			recovery[r * blockSize64 + w] = acc;
		}
	}

	/*
	 * Step 7: Divide through by den(y_r). Done as a separate pass so the
	 * num_sum scratch can reuse the recovery[] output buffer (one fewer
	 * numRecovery * blockSize64-word allocation).
	 *
	 * Cost: numRecovery GF(2^64) inversions + numRecovery * blockSize64
	 * field multiplications.
	 */
	for (size_t r = 0; r < numRecovery; r++) {
		const gf64_t inv_den = gf64_invert_ita_one(den_at_recovery[r]);
		if (inv_den == 0) {
			/* den was zero; convention keeps the existing values. */
			continue;
		}
		gf64_t* row = recovery + r * blockSize64;
		for (size_t w = 0; w < blockSize64; w++) {
			row[w] = gf64_mul_reference(row[w], inv_den);
		}
	}

	/*
	 * Release the subproduct tree's backing storage before returning.
	 * Free is a no-op on a zero-initialized SubproductTree struct and
	 * idempotent on a valid one — matches the contract in
	 * gf64_subproduct_tree_free.
	 */
	gf64_subproduct_tree_free(&tree);
}
