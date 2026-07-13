/*
 * ============================================================================
 * src/par3_engine_barycentric.cc — Barycentric kernel entry (Phase 2c, rev 2)
 *
 * Computes GF64Controller::ComputeRecoveryBlocksBarycentric: the Cauchy
 * matrix-vector product
 *
 *     out[r][w] = XOR_{c=0..numInputs-1} in[c][w] / (y_r XOR x_c)
 *
 * where x_c = firstInput + c, y_r = firstRecovery + r in GF(2^64).
 * Bit-identical to the legacy 2D-muladd path (verified by 56-case
 * parity test + 29-case dispatch test + 1-case threshold crossover).
 *
 * PIPELINE
 * --------
 *   1. Build the N-point subproduct tree T_X over {x_c}.
 *   2. Compute barycentric weights W_c = 1 / P'(x_c) via T7.
 *      Currently unused by the per-row SIMD compute (see MATH NOTE
 *      below), but precomputed so that any future Fenger-Toeplitz
 *      pipeline that consumes the weights in-line has them ready.
 *   3. Delegate the per-row Cauchy accumulation to the LEGACY
 *      ComputeRecoveryBlocks entry. This routes through the LRU
 *      CoeffCache so warm calls reuse the cached (R x N) matrix
 *      instead of recomputing 1/(y_r XOR x_c) per element — the
 *      pre-Phase-2c inline-reconstruction path was 24-138x slower
 *      than the LRU-cached version on the bench.
 *
 * MATH NOTE — this is NOT a Lagrange-interpolation kernel
 *   The Barycentric identity 1/(y+x_c) = W_c * V(y)/(y+x_c) / D(y)
 *   only holds for LAGRANGE INTERPOLATION — where f(x_c) is a
 *   polynomial evaluated at {x_c} and the query is f(y). PAR3's
 *   Cauchy recovery uses f(c) = in[c][:] indexed by array position
 *   c, NOT a polynomial evaluation. So the Cauchy matrix is a
 *   different linear operator from the Lagrange interpolator, and
 *   the Barycentric identity does not apply directly.
 *
 *   The Phase 2b FFT + Phase 3 Fenger Toeplitz work targets a TRUE
 *   asymptotic win via the Fenger 2009 decomposition (polynomial
 *   multiplication over GF(2^64)[x] reduced to O((N+R) log²(N+R))
 *   complexity). Until those land, the kernel stays equivalent to
 *   legacy 2D-muladd throughput-wise and exists primarily to keep
 *   the entry bit-exact to legacy so the parity suite stays green.
 *
 * PERF (WSL Ubuntu gcc 15.2.0, Zen4 7800X3D, forced Barycentric):
 *   Inherits legacy throughput via the LRU-cached matrix path —
 *   matches compute_recovery_full on warm calls.
 * ============================================================================
 */

#include "par3_engine.h"
#include "gf64_global.h"
#include "gf64_subproduct.h"
#include "gf64_barycentric.h"

#include <stdint.h>
#include <stdlib.h>

namespace {
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

	/*
	 * Steps 1 + 2: build the Barycentric front-end (subproduct tree +
	 * weights). Currently unused by the per-row SIMD compute — kept as
	 * the architectural hook point that Phase 2b's MPE-driven vector
	 * eval will plug into. The cost is O(N log N) for the tree (driven
	 * by gf64_poly_mul_padded, now Karatsuba-accelerated per Phase 2a)
	 * plus O(N) for the Itoh-Tsujii batched inversion.
	 *
	 * The subproduct tree (T6) requires N to be a power of 2 (or 0/1).
	 * The PAR3 create path allows any N; for non-power-of-2 N we skip the
	 * front-end and just delegate to legacy, paying only the dispatch
	 * overhead. With Phase 2b the front-end will instead pad N to the
	 * next power of 2 explicitly so the MPE machinery has the right
	 * tree shape — that swap localises to this block.
	 */
	int numInputs_pow2 = (numInputs >= 1) && ((numInputs & (numInputs - 1)) == 0);
	if (numInputs_pow2) {
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

		/* The weights are precomputed so that a future FFT/MPE-based
		 * drop-in does not have to redo the tree build. Currently the
		 * per-row Cauchy accumulation is delegated to the LRU-cached
		 * ComputeRecoveryBlocks path below; the Barycentric structure
		 * is not yet used to compress the work. */
		(void)weights;
		free(weights);
	}

	/*
	 * Step 3: delegate to the legacy Cauchy matrix-vector product. This
	 * routes through the LRU CoeffCache so the (R x N) matrix is built
	 * once and reused across calls — matching legacy 2D-muladd
	 * throughput. With Phase 2b (real FFT) plus Phase 3 (Fenger Toeplitz),
	 * the body of this delegate is replaced by a polynomial-multiply
	 * pipeline that achieves O((N+R) log²(N+R)) complexity, which is
	 * the asymptotic path to beating PAR2's 622 MB/s.
	 */
	ComputeRecoveryBlocks(
		inputs, numInputs,
		recovery, numRecovery,
		blockSize64,
		firstInput, firstRecovery,
		numThreads
	);
}