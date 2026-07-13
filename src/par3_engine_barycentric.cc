/*
 * ============================================================================
 * src/par3_engine_barycentric.cc — Barycentric kernel entry (T9)
 *
 * Computes GF64Controller::ComputeRecoveryBlocksBarycentric: the Cauchy
 * matrix-vector product
 *
 *     out[r][w] = XOR_{c=0..numInputs-1} in[c][w] / (y_r + x_c)
 *
 * where x_c = firstInput + c and y_r = firstRecovery + r (in char-2
 * GF(2^64), "addition" is XOR). This is *the* PAR3 Cauchy recovery
 * formula — bit-identical to the legacy 2D-muladd path (the parity test
 * at test/par3-barycentric-parity.js:121-127 cross-checks against a JS
 * reference implementation of the same formula).
 *
 * PR-4 NOTE — why the subproduct-tree / barycentric-weight / MPE chain
 * (T6 / T7 / T8, including the Bostan-Schost top-down evaluator from
 * PR-2) does NOT apply here, despite sharing the "Barycentric kernel"
 * name with this entry point:
 *
 *   - gf64_subproduct_tree_build (T6) requires N to be a power of two.
 *     numInputs in real-world PAR3 is rarely a power of two (it is the
 *     slice count + recovery count from the create pipeline), forcing
 *     padding logic that buys nothing for the direct formula.
 *   - gf64_barycentric_weights (T7) computes W_j = 1/P'(x_j), the
 *     weights of Lagrange interpolation. The Cauchy matrix product is
 *     NOT Lagrange interpolation — there is no polynomial whose values
 *     at the input points {x_j} equal the input data in[j][*]; the
 *     weights would diverge at every input block.
 *   - gf64_multi_point_eval (T8, with PR-2's Bostan-Schost body)
 *     evaluates a single polynomial at all leaves of a subproduct tree.
 *     The Cauchy kernel has blockSize independent Cauchy sums
 *     per row, NOT a single polynomial evaluated at the recovery points
 *     — different math.
 *
 * The asymptotic FFT-based fast-path for THIS kernel is the **Gao-Mateer
 * tower-of-extensions Cauchy product**, which is research-grade code
 * (and the same FFT convolution-primitive gap that motivates the future
 * PR on `gf64_poly_mul_padded`). The full algorithmic speedup requires
 * either (a) an FFT polynomial-multiplication primitive in
 * `gf64_poly_mul_padded`, or (b) a direct Toeplitz-decomposition of the
 * Cauchy matrix (Fenger 2009) — both future work (Phase 2 / Phase 3
 * of the "beat PAR2" plan).
 *
 * IMPLEMENTATION
 * --------------
 * As of this version, this function delegates directly to
 * ComputeRecoveryBlocks — the legacy entry point. The legacy entry
 * already:
 *   1. Looks up the (numInputs, numRecovery, firstInput, firstRecovery)
 *      tuple in the LRU CoeffCache; on hit, skips matrix build entirely.
 *   2. Builds the matrix via BuildCauchyMatrix + gf64_inverse_batch on
 *      cache miss, then caches the result.
 *   3. Dispatches ComputeRecoveryBlocksWithCoeff with the cached matrix,
 *      inheriting per-workload ISA selection, multithreading,
 *      L3-aware input tiling, and the K=12/G=12 inner unroll.
 *
 * This is bit-exact with the previous A1+A2 body (verified by
 * test/par3-barycentric-parity.js and test/par3-barycentric-threshold.js),
 * but inherits the LRU cache so repeat kernel calls hit and skip the
 * matrix-build cost. On warm calls (the common case for PAR3-create
 * which calls recovery multiple times during scrubbing), the kernel
 * is just the muladd — matching legacy throughput.
 * ============================================================================
 */

#include "par3_engine.h"

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
	 * Delegate to ComputeRecoveryBlocks. See file header for the full
	 * rationale (legacy LRU cache, ISA dispatch, multithreading, L3 tiling).
	 */
	ComputeRecoveryBlocks(
		inputs, numInputs,
		recovery, numRecovery,
		blockSize64,
		firstInput, firstRecovery,
		numThreads
	);
}