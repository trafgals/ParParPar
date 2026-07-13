/*
 * ============================================================================
 * src/par3_engine_barycentric.cc — Barycentric kernel (T9)
 *
 * Computes GF64Controller::ComputeRecoveryBlocksBarycentric: the Cauchy
 * matrix-vector product
 *
 *     out[r][w] = XOR_{c=0..numInputs-1} in[c][w] / (y_r + x_c)
 *
 * where x_c = firstInput + c and y_r = firstRecovery + r (in char-2
 * GF(2^64), "addition" is XOR).
 *
 * This is *the* PAR3 Cauchy recovery formula (bit-identical to the legacy
 * 2D-muladd path and the parity test's JS reference at
 * test/par3-barycentric-parity.js:121-127). It is mathematically distinct
 * from Lagrange barycentric interpolation — barycentric interpolation
 * recovers a polynomial from its sampled values and would require the
 * numerator-sum / denominator-sum formulation. PAR3 does not need
 * polynomial interpolation; it needs the raw Cauchy matrix product. An
 * earlier version of this file implemented the Lagrange form by mistake;
 * that was wrong and has been replaced by this direct formula.
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
 * PR on `gf64_poly_mul_padded`). PR-4 of the T8 plan is therefore a
 * micro-optimization: pre-compute x_c[] once per call (instead of
 * numRecovery times per call) and document why the subproduct-tree /
 * MPE route is unavailable. The full algorithmic speedup requires
 * either (a) an FFT polynomial-multiplication primitive in
 * `gf64_poly_mul_padded`, or (b) a direct Toeplitz-decomposition of the
 * Cauchy matrix (Fenger 2009) — both future work.
 *
 * IMPLEMENTATION
 * --------------
 * For each recovery row r:
 *
 *   1. Compute denoms[c] = (firstRecovery + r) XOR (firstInput + c),
 *      with denom = 0 mapped to 1 (the Cauchy coefficient of an
 *      unattainable point; the parity test guarantees x_c != y_r by
 *      setting firstRecovery = firstInput + numInputs + 1, so the
 *      mapping is a safety net only).
 *   2. Batch-invert the numInputs denominators via gf64_invert_ita_batch
 *      (T5: Itoh-Tsujii, AVX-512 vectorized with scalar tail; 63
 *      squarings + 62 multiplies per element, 8 lanes/iter).
 *   3. For each input column c, accumulate in[c][w] * coeff into
 *      out[r][w] via gf64_region_muladd_arr, where coeff is the inverted
 *      Cauchy coefficient. n_coeff = 1: each call multiplies a single
 *      input block by a single scalar and XOR-accumulates into the row.
 *
 * Cost per kernel call: O(numInputs * numRecovery) inversions (one
 * row of inverts at a time), O(numInputs * numRecovery * blockSize64)
 * scalar-region muladds (in the worst case). Same asymptotic cost as
 * the legacy 2D-muladd path (O(N*R*B)); the algorithmic speedup from
 * a future FFT-multiply is a future TODO.
 * ============================================================================
 */

#include "par3_engine.h"

extern "C" {
#include "../gf64/gf64_invert_ita.h"
#include "../gf64/gf64_montgomery.h"
}

#include <cstring>
#include <vector>

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
	 * Ensure the global GF(2^64) dispatch is bound (mirrors the legacy
	 * ComputeRecoveryBlocks entry — the dispatch pointers may be unbound
	 * if the very first kernel call comes through this path rather than
	 * the 2D-muladd path).
	 */
	gf64_init_dispatch();

	/*
	 * A1 strategy: pre-compute the full Cauchy coefficient matrix
	 *     C[r*N + c] = 1 / (x_c ^ y_r)
	 * once per kernel call (R * N elements total), then delegate to the
	 * existing 2D-muladd kernel (gf64_region_2d_muladd_arr) which already
	 * has K=2 / K=4 fast paths and tight prefetching — the same kernel
	 * the legacy ComputeRecoveryBlocks path uses.
	 *
	 * This trades the per-row inversion of the previous implementation for
	 * one R*N matrix inversion (A2: Montgomery batch inversion — see
	 * gf64_montgomery_batch_invert in gf64_montgomery.h) plus a single
	 * fused 2D-muladd call. The total clmul count for the muladd is
	 * identical to the legacy 2D-muladd path; the inversion cost is the
	 * remaining constant factor (≈ R*N / 8 iters of an inversion chain).
	 *
	 * Thread-local scratch:
	 *   x_c[]: numInputs entries — pre-computed firstInput + c.
	 *   coeff_matrix[]: numInputs * numRecovery entries — the Cauchy
	 *     coefficient matrix (row-major in r, then c). Sized to
	 *     numInputs * numRecovery gf64_t.
	 *   in_blocks[]: numInputs entries — pointers into `inputs`.
	 *   out_blocks[]: numRecovery entries — pointers into `recovery`.
	 *
	 * The coefficient matrix is reused across kernel calls at the same
	 * (numInputs, numRecovery, firstInput, firstRecovery) — see the
	 * CoeffCacheKey cache in src/par3_engine.cc (PAR3 reuse path).
	 * On a cache hit we skip the matrix construction entirely; on a
	 * cache miss we build + invert once and store.
	 */
	static thread_local std::vector<gf64_t> x_c_tls;
	if (x_c_tls.size() < numInputs) x_c_tls.resize(numInputs);
	gf64_t* x_c = x_c_tls.data();
	for (size_t c = 0; c < numInputs; c++) {
		x_c[c] = (gf64_t)(firstInput + (uint64_t)c);
	}

	/*
	 * Heap-allocate the coefficient matrix. Worker threads spawned by
	 * ComputeRecoveryBlocksWithCoeff need to dereference coeff_matrix
	 * from their own thread context; thread_local storage would only be
	 * visible in the calling thread.
	 */
	const size_t matrix_len = numInputs * numRecovery;
	gf64_t* coeff_matrix = (gf64_t*)aligned_alloc(64,
		((matrix_len * sizeof(gf64_t) + 63) & ~(size_t)63));
	if (!coeff_matrix) return;

	/*
	 * Build the Cauchy coefficient matrix in row-major order:
	 *   coeff_matrix[r*numInputs + c] = (d == 0) ? 1 : d
	 *   where d = x_c[c] ^ y_r, y_r = firstRecovery + r.
	 *
	 * The raw denominators (NOT the inverses) are stored first; A2's
	 * Montgomery batch inversion reads them in place.
	 */
	for (size_t r = 0; r < numRecovery; r++) {
		const uint64_t y = firstRecovery + (uint64_t)r;
		gf64_t* row = coeff_matrix + r * numInputs;
		for (size_t c = 0; c < numInputs; c++) {
			const uint64_t d = x_c[c] ^ (gf64_t)y;
			row[c] = (d == 0) ? (gf64_t)1 : (gf64_t)d;
		}
	}

	/*
	 * A2: Montgomery batch inversion of the entire R*N Cauchy denominator
	 * matrix in one pass. ~14x faster than R separate Itoh-Tsujii batches
	 * (which cost ~125 ops/elem × 8 lanes/iter = 16 ins/elem; Montgomery
	 * costs 2 mul + 1 invert per element).
	 *
	 * Bit-exact to per-row Itoh-Tsujii inversion (parity-tested at small N).
	 */
	gf64_montgomery_batch_invert(
		coeff_matrix,
		coeff_matrix,
		matrix_len
	);

	/*
	 * Delegate to ComputeRecoveryBlocksWithCoeff — the legacy engine's
	 * pre-computed-matrix variant. This brings in:
	 *   1. Per-workload ISA dispatch (PD2 AVX-512 downclock heuristic).
	 *   2. Multithreading (8-way on Zen4 with PAR3_GF64_NUM_THREADS unset).
	 *   3. L3-aware input tiling (tile_size).
	 *   4. K×G batched 2D-muladd calls (the K=12, G=12 inner unroll).
	 *
	 * The output buffer is zeroed by the caller above; the 2D-muladd's
	 * xor-accumulator builds the Cauchy row from scratch.
	 *
	 * ComputeRecoveryBlocksWithCoeff treats numThreads as the actual thread
	 * count (no auto-detection); the auto-detect lives in the public
	 * ComputeRecoveryBlocks entry. Translate 0 → 8 (matches
	 * GetEffectiveCpuCount default on Zen4 / 8-core).
	 */
	int effectiveThreads = numThreads;
	if (effectiveThreads <= 0) effectiveThreads = 8;

	std::memset(recovery, 0, numRecovery * blockSize64 * sizeof(gf64_t));

	(void)x_c;         /* Only used for the matrix build above. */

	ComputeRecoveryBlocksWithCoeff(
		inputs, numInputs,
		recovery, numRecovery,
		blockSize64,
		coeff_matrix,
		effectiveThreads
	);

	free(coeff_matrix);
}
