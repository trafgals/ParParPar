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
 * see the commit history (and test/par3-barycentric-parity.js) for the
 * correction.
 *
 * DESIGN NOTE — why the tree-based primitives are NOT used here (issue #46
 * root-cause context):
 *
 * The Cauchy matrix-vector product is a *linear operator* over the input
 * blocks. The subproduct-tree / MPE / Fenger primitives shipped in
 * gf64_subproduct.c / gf64_mpe.c / gf64_fenger.c all target *polynomial*
 * evaluation (Lagrange form, e.g. computing f_w(y_r) / V(y_r) for
 * Cauchy recovery). They are deliberately *not* invoked here because:
 *
 *   - gf64_subproduct_tree_build (T6) requires N to be a power of two,
 *     forcing padding logic that buys nothing for the direct formula.
 *   - gf64_barycentric_weights (T7) computes W_j = 1/P'(x_j), which is
 *     a Lagrange-interpolation primitive with no role in the Cauchy
 *     matrix-vector product.
 *   - gf64_multi_point_eval (T8) provides multi-point polynomial
 *     evaluation; it reduces the *polynomial* evaluation cost from
 *     O(N*R) to O((N+R) log²(N+R)), but only when the inputs are the
 *     polynomial coefficients — for the raw Cauchy matrix product the
 *     reduction is applied via batched Itoh-Tsujii inversion below.
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
 *      squarings + 62 multiplies per element, 8 lanes/iter) on hosts
 *      with VPCLMULQDQ; otherwise dispatch through gf64_inverse_batch
 *      (scalar/SSSE3/AVX-2 specialised batch inverters bound at
 *      gf64_init_dispatch time). Input and output buffers are kept
 *      distinct to satisfy HEDLEY_RESTRICT — see the coeffs/denoms
 *      scratch pair in processRow below.
 *   3. For each input column c, accumulate in[c][w] * coeff into
 *      out[r][w] via gf64_region_muladd_arr, where coeff is the inverted
 *      Cauchy coefficient. n_coeff = 1: each call multiplies a single
 *      input block by a single scalar and XOR-accumulates into the row.
 *
 * Cost per kernel call: O(numInputs) inversions per row,
 * O(numInputs * blockSize64) scalar-region muladds per row, numRecovery
 * rows. Same asymptotic cost as the legacy 2D-muladd path
 * (O(N*R*B)); the algorithmic speedup from subproduct-tree MPE is a
 * future TODO.
 *
 * PARALLELIZATION (issue #46 Phase A — 2026-08-13 native-Windows session)
 * ------------------------------------------------------------------------
 * The v1 implementation ran the row loop single-threaded regardless of
 * numThreads, which regressed create throughput ~6x vs the legacy
 * 8-thread L3-aware Cauchy path (1 GiB / 10K slices: 28.3 -> 4.78 MB/s
 * on native Windows). The row loop is embarrassingly parallel: each
 * row r writes a distinct out_row and reads only the shared inputs, and
 * gf64_invert_ita_batch / gf64_inverse_batch / gf64_region_muladd_arr
 * are pure (no shared mutable state — verified by inspection), so rows
 * can be distributed across workers with per-worker scratch. The
 * parallel path mirrors BuildCauchyMatrix's row-stealing std::async
 * pattern (proven load-balanced in this engine); the serial path keeps
 * the original thread_local scratch for the single-worker case.
 *
 * TODO(bostan-schost-fast-path): see the file header doc. The future
 * fast-path replaces step 3 above with a single polynomial evaluation
 * via gf64_multi_point_eval, driving the kernel from O(N*R*B) to
 * O((N+R)*B + (D+N)*log²(D+N)) where D = N.
 * ============================================================================
 */

#include "par3_engine.h"

extern "C" {
#include "../gf64/gf64_invert_ita.h"
}

#include <atomic>
#include <cstring>
#include <future>
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

	if (numRecovery <= 32) {
		/* Keep the small-R workload on the engine's 1D muladd shortcut.
		 * This entry can still be selected by the JS large-input gate, but
		 * the Barycentric algorithm itself is reserved for larger R. */
		ComputeRecoveryBlocks(inputs, numInputs, recovery, numRecovery,
		                     blockSize64, firstInput, firstRecovery, 0);
		return;
	}

	/*
	 * Ensure the global GF(2^64) dispatch is bound (mirrors the legacy
	 * ComputeRecoveryBlocks entry — the dispatch pointers may be unbound
	 * if the very first kernel call comes through this path rather than
	 * the 2D-muladd path). Must complete BEFORE the parallel workers
	 * spawn: the workers only read the bound function pointers
	 * (gf64_inverse_batch / gf64_region_muladd_arr), and the one-shot
	 * init guard is not thread-safe against concurrent first calls.
	 */
	gf64_init_dispatch();

	/*
	 * Worker count: honour the caller's numThreads (<= 0 = auto), fall
	 * back to the effective CPU count (affinity-aware), and cap at the
	 * row count — mirroring ComputeRepairBlocks / BuildCauchyMatrix.
	 */
	size_t n_workers = (numThreads <= 0)
		? GetEffectiveCpuCount()
		: (size_t)numThreads;
	if (n_workers == 0) n_workers = 1;
	if (n_workers > numRecovery) n_workers = numRecovery;

	/*
	 * Per-row kernel, factored so the serial and parallel paths share one
	 * implementation. `denoms` / `coeffs` are caller-owned scratch:
	 * thread_local vectors in the serial path (reused across rows, no
	 * reallocation), per-worker vectors in the parallel path.
	 *
	 * Per-row memory: numInputs * sizeof(gf64_t) for the raw denominators
	 * (`denoms`) PLUS numInputs * sizeof(gf64_t) for the inverted
	 * coefficients (`coeffs`). The two buffers are deliberately
	 * distinct: gf64_inverse_batch's non-aliasing HEDLEY_RESTRICT contract
	 * forbids passing the same array as both input and output. Cubic
	 * review 4891289950 (item 3) flagged the prior in-place scalar
	 * Itoh-Tsujii fallback for losing the SIMD batch path on AVX2/SSSE3
	 * hosts and on the PD2 downclock downgrade; the fix is to ship a
	 * separate output buffer and dispatch through gf64_inverse_batch in
	 * the fallback branch.
	 */
	auto processRow = [&](size_t r, gf64_t* denoms, gf64_t* coeffs) {
		const uint64_t y = firstRecovery + (uint64_t)r;

		/* Step 1: raw denominators. */
		for (size_t c = 0; c < numInputs; c++) {
			const uint64_t x = firstInput + (uint64_t)c;
			const uint64_t d = x ^ y;
			denoms[c] = (d == 0) ? (gf64_t)1 : (gf64_t)d;
		}

		/* Step 2: batched Itoh-Tsujii inversion (T5).
		 *
		 * The VPCLMULQDQ codepath uses gf64_invert_ita_batch, which
		 * __attribute__((target("avx512f,vpclmulqdq")))-compiles and
		 * would SIGILL on a host without VPCLMULQDQ. Gate on
		 * gf64_has_vpclmulqdq (a host-availability flag set once from
		 * CPUID), NOT on gf64_current_method — the latter is workload-
		 * chosen and the PD2 downclock heuristic may downgrade it to
		 * GF64_AVX2 even when the host supports the VPCLMULQDQ codepath.
		 *
		 * The non-VPCLMULQDQ fallback dispatches through the GF64Method
		 * table's gf64_inverse_batch (scalar / SSSE3 / AVX-2 / AVX-512
		 * inverted batch on the free side). This preserves the
		 * AVX2/SSSE3 SIMD batch path the cubic review (4891289950,
		 * item 3) flagged as lost by the prior scalar Itoh-Tsujii loop.
		 * The function pointer carries HEDLEY_RESTRICT on both
		 * arguments, so input (`denoms`) and output (`coeffs`) must be
		 * distinct arrays. */
		if (gf64_has_vpclmulqdq) {
			gf64_invert_ita_batch(coeffs, denoms, numInputs);
		} else {
			gf64_inverse_batch(coeffs, denoms, numInputs);
		}

		/* Step 3: out[r][w] = XOR_c in[c][w] * coeffs[c].
		 *
		 * gf64_region_muladd_arr performs out[w] ^= in[w] * coeff[w]
		 * for the n_coeff = 1 special case. The dispatch picks the
		 * ISA-appropriate path (scalar / SSSE3 / AVX-2 / AVX-512).
		 *
		 * `coeffs[c]` is the inverted Cauchy coefficient 1 / (x_c ^ y_r),
		 * written into the per-row output scratch by Step 2's batch
		 * inversion (gf64_invert_ita_batch or the dispatched
		 * gf64_inverse_batch fallback). The raw `denoms` buffer holds
		 * the pre-inversion inputs and is not re-read.
		 *
		 * Zero out_row first so the XOR-accumulator below builds the
		 * Cauchy row from scratch (no need to depend on the caller's
		 * prior contents).
		 */
		gf64_t* out_row = recovery + r * blockSize64;
		std::memset(out_row, 0, blockSize64 * sizeof(gf64_t));

		for (size_t c = 0; c < numInputs; c++) {
			const gf64_t coeff = coeffs[c];
			gf64_region_muladd_arr(
				out_row,
				inputs + c * blockSize64,
				&coeff,
				blockSize64,
				/*n_coeff=*/1
			);
		}
	};

	if (n_workers <= 1) {
		/* Serial path: thread_local scratch, reused across rows (no
		 * per-row heap traffic). */
		static thread_local std::vector<gf64_t> denoms_tls;
		if (denoms_tls.size() < numInputs) denoms_tls.resize(numInputs);
		static thread_local std::vector<gf64_t> coeffs_tls;
		if (coeffs_tls.size() < numInputs) coeffs_tls.resize(numInputs);

		for (size_t r = 0; r < numRecovery; r++) {
			processRow(r, denoms_tls.data(), coeffs_tls.data());
		}
		return;
	}

	/* Parallel path: row-stealing std::async workers (mirrors
	 * BuildCauchyMatrix). `nextRow` is destroyed only after all futures
	 * complete (we block below), so workers can safely reference it by
	 * reference. Each worker owns its denoms/coeffs scratch, reused per
	 * row. Rows are independent: distinct out_row, read-only inputs,
	 * pure kernels. */
	std::atomic<size_t> nextRow{0};
	std::vector<std::future<void>> futures;
	futures.reserve(n_workers);

	for (size_t w = 0; w < n_workers; w++) {
		futures.push_back(std::async(std::launch::async, [&, w]() {
			std::vector<gf64_t> denoms(numInputs);
			std::vector<gf64_t> coeffs(numInputs);
			for (size_t r = nextRow.fetch_add(1, std::memory_order_relaxed);
			     r < numRecovery;
			     r = nextRow.fetch_add(1, std::memory_order_relaxed)) {
				processRow(r, denoms.data(), coeffs.data());
			}
		}));
	}

	/* f.get() (not wait()) so a worker exception (e.g. std::bad_alloc
	 * from the per-worker scratch vectors) propagates to the caller
	 * instead of silently returning with `recovery` partially written
	 * (cubic review f96ee17b P2). */
	for (auto& f : futures) {
		f.get();
	}
}
