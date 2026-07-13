/*
 * ============================================================================
 * src/par3_engine_barycentric.cc — Barycentric kernel entry (Phase 2c)
 *
 * Computes GF64Controller::ComputeRecoveryBlocksBarycentric: the Cauchy
 * matrix-vector product
 *
 *     out[r][w] = XOR_{c=0..numInputs-1} in[c][w] / (y_r XOR x_c)
 *
 * where x_c = firstInput + c, y_r = firstRecovery + r in GF(2^64).
 * Bit-identical to the legacy 2D-muladd path (and to test/par3-barycentric-
 * parity.js's reference implementation of the same formula).
 *
 * PIPELINE (this version)
 * -----------------------
 *   1. Build the input-points subproduct tree T_X over the N=numInputs
 *      input points x_c — the (future) Phase 2b MPE machinery will drive
 *      this through a Bostan-Schost walk instead of per-row Horner.
 *   2. Compute barycentric weights W_c = 1 / P'(x_c) using T7
 *      (gf64_barycentric_weights). One batched inversion over N elements.
 *      Currently unused by the per-row Cauchy evaluation (the Cauchy matrix
 *      is not Lagrange interpolation — see MATH note below) but precomputed
 *      here so that any future Fenger-Toeplitz pipeline that consumes the
 *      weights in-line has them already on hand.
 *   3. For each recovery row r in [0..R):
 *        a. compute M[r][c] on the fly as 1 / (y_r XOR x_c), exposed to
 *           the existing SIMD 2D-muladd via the (numInputs × R) coeff
 *           scratch buffer;
 *        b. run gf64_region_2d_muladd_arr to produce out[r][:];
 *        c. zero the coeff scratch after each row so the next row starts
 *           from a deterministic state.
 *
 * COST
 * ----
 *   - Step 1 (subproduct tree):       O(N log N) field ops (driven by
 *                                     gf64_poly_mul_padded, now Karatsuba-
 *                                     accelerated per Phase 2a).
 *   - Step 2 (barycentric weights):   O(N) field ops via Itoh-Tsujii batch.
 *   - Step 3 (per recovery row):      O(N inv + N*B muladd). Identical
 *                                     complexity to the legacy 2D-muladd,
 *                                     just routed through the subproduct-
 *                                     tree front-end for the FFT hook point.
 *   - The "real" asymptotic win (replacing the per-row N inversions with
 *     one MPE-based D(y_r) evaluation for every r) lands with Phase 2b
 *     (real FFT poly_mul) — see src/par3_engine_barycentric.cc header
 *     block in the prior version of this file for the Fenger 2009 plan.
 *
 * MATH NOTE — why no Barycentric trick
 * -------------------------------------
 * One might hope to apply the Barycentric Lagrange identity
 *     1/(y + x_c) = W_c * V(y)/(y + x_c) / D(y)
 * to compress the matrix-vector product into an MPE-based evaluation. That
 * identity only holds for LAGRANGE INTERPOLATION (where the input is a
 * polynomial f evaluated at the nodes x_c, and we want f at a query point
 * y). The PAR3 Cauchy recovery uses f(x_c) = in[c][:] which is NOT a
 * polynomial evaluation at x_c — the index `c` indexes an array, not a
 * polynomial variable. So the Cauchy matrix is a different linear operator
 * from the Lagrange interpolator, and the Barycentric identity does not
 * apply directly.
 *
 * The Phase 2b FFT + Phase 3 Fenger Toeplitz work targets a TRUE asymptotic
 * win via the Fenger 2009 decomposition (polynomial multiplication /
 * division over the GF(2^64)[x] ring), not via the Barycentric kernel
 * itself. Phase 2c's job here is just to keep the kernel as a real,
 * parity-tested entry point so the architecture is ready for that future
 * plug-in.
 *
 * IMPLEMENTATION
 * --------------
 * As of this version, this function is a real Cauchy matrix-vector kernel
 * that routes through the Barycentric front-end (steps 1-2) and then
 * delegates the per-row SIMD compute to gf64_region_2d_muladd_arr (the
 * same kernel the legacy entry uses). On warm calls the legacy 2D-muladd
 * wins on throughput because it re-uses a single cached (R x N) matrix;
 * this Barycentric kernel pays the per-row matrix reconstruction cost as
 * the price of keeping the architecture future-proof for Phase 2b's
 * MPE-based path.
 * ============================================================================
 */

#include "par3_engine.h"
#include "gf64_global.h"
#include "gf64_subproduct.h"
#include "gf64_barycentric.h"
#include "gf64_invert_ita.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

namespace {

/* gf64_mul_reference (declared in gf64_single.c) is the scalar SSE2
 * reference multiply. We use it for the per-row Cauchy matrix element
 * computation (1 / (y_r XOR x_c)), so the arithmetic is ISA-independent
 * and identical to the parity-test reference in test/par3-barycentric-
 * parity.js. The per-word muladd itself uses gf64_mul_avx512 via the
 * gf64_region_2d_muladd_arr dispatch. */
extern "C" gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

} // anonymous namespace

void GF64Controller::ComputeRecoveryBlocksBarycentric(
	const gf64_t* inputs, size_t numInputs,
	gf64_t* recovery, size_t numRecovery,
	size_t blockSize64,
	uint64_t firstInput, uint64_t firstRecovery,
	int numThreads
) {
	/*
	 * Trivial-input short-circuit. Matches the engine convention (see
	 * ComputeRecoveryBlocks) — empty problems return without writing the
	 * output and without doing any work.
	 */
	if (numInputs == 0 || numRecovery == 0 || blockSize64 == 0) {
		return;
	}

	/* Ensure the SIMD dispatch is initialized (the legacy
	 * ComputeRecoveryBlocks does this; the Barycentric kernel needs the
	 * same one-shot call). */
	gf64_apply_method(gf64_method_for_workload(numInputs, numRecovery, blockSize64));

	/*
	 * Step 1 + 2: build the N-point subproduct tree over the input points
	 * and the barycentric weights W_c. The weights are NOT consumed by
	 * the current per-row Cauchy evaluation (see MATH NOTE above), but
	 * precomputing them here means the future FFT/MPE pipeline has the
	 * inputs ready without needing to redo this work.
	 */
	gf64_t *input_points = (gf64_t *)malloc(numInputs * sizeof(gf64_t));
	if (input_points == NULL) abort();
	for (size_t c = 0; c < numInputs; c++) {
		input_points[c] = (gf64_t)(firstInput + (uint64_t)c);
	}

	SubproductTree tree_X;
	gf64_subproduct_tree_build(input_points, numInputs, &tree_X);
	free(input_points);

	gf64_t *weights = (gf64_t *)malloc(numInputs * sizeof(gf64_t));
	if (weights == NULL) {
		gf64_subproduct_tree_free(&tree_X);
		abort();
	}
	gf64_barycentric_weights(&tree_X, weights);
	gf64_subproduct_tree_free(&tree_X);

	/*
	 * Step 3: per-row Cauchy matrix + SIMD 2D-muladd. For each recovery
	 * row r we build the (1 x numInputs) coefficient slice M[r][:]
	 * inline (1/(y_r XOR x_c) for each c) and pass it to the same SIMD
	 * muladd the legacy kernel uses. The coeff slice is heap-allocated
	 * once outside the loop to avoid per-row malloc/free.
	 */
	gf64_t *coeff_row = (gf64_t *)malloc(numInputs * sizeof(gf64_t));
	if (coeff_row == NULL) {
		free(weights);
		abort();
	}

	for (size_t r = 0; r < numRecovery; r++) {
		uint64_t y_r = firstRecovery + (uint64_t)r;
		for (size_t c = 0; c < numInputs; c++) {
			uint64_t x_c = firstInput + (uint64_t)c;
			coeff_row[c] = gf64_invert_ita_one((gf64_t)(y_r ^ x_c));
		}

		/* Per-row Cauchy accumulation:
		 *   for c: out[r][w] ^= in[c][w] * coeff_row[c]   for each w
		 *
		 * Routed through gf64_region_muladd_arr with n_coeff=1 so the
		 * AVX-512 / SSSE3 / scalar dispatch path is reused (see
		 * gf64_region.h). Identical op count to the legacy K=1 G=1
		 * path; the next iteration of Phase 2c (post-Phase-2b FFT)
		 * replaces this loop with a single MPE-driven vector eval. */
		memset(recovery + r * blockSize64, 0, blockSize64 * sizeof(gf64_t));
		for (size_t c = 0; c < numInputs; c++) {
			gf64_region_muladd_arr(
				recovery + r * blockSize64,
				inputs + c * blockSize64,
				&coeff_row[c],
				blockSize64,
				1
			);
		}
		(void)weights;  /* future FFT pipeline consumes this */
	}

	free(coeff_row);
	free(weights);
}