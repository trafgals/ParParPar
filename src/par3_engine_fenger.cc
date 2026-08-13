/*
 * ============================================================================
 * src/par3_engine_fenger.cc — Fenger Toeplitz engine entry (issue #28).
 *
 * Computes GF64Controller::ComputeRecoveryBlocksFenger: the Cauchy
 * matrix-vector product
 *
 *     out[r][w] = XOR_{c=0..numInputs-1} in[c][w] / (y_r XOR x_c)
 *
 * via the Bostan-Schost top-down Fenger pipeline (gf64_fenger_matvec /
 * gf64_fenger_{prepare,execute,release}). Bit-identical to the legacy
 * 2D-muladd path (verified by Phase 1's bit-exact parity test: 12/12
 * PASS across N∈{1,2,4,8,16,32}, R∈{1,2,4,8,16}, B∈{1,2,4,8}).
 *
 * ASYMPTOTICS
 * -----------
 *   Per call:
 *     Setup (one-time):    O(M(N) log N + M(R) log R + M(N) log^2 N
 *                          + M(R) log^2 R + N + R)
 *     Per word (B words):  O(M(N) log^2 N + R)
 *     Total:               O(B * M(N) log^2 N + setup)
 *
 *   The setup is amortized across the B words. Per-word cost is
 *   INDEPENDENT of R once R is large enough that the per-row division
 *   step (O(R)) is dominated by the per-word interp/eval cost.
 *
 *   For the canonical PAR3-create workload (N = 10K, R = 1K,
 *   B = ~12.5K gf64_t per input block), the legacy Cauchy path is
 *   O(B * N * R) ≈ 10^11 GF(2^64) ops; the Fenger pipeline drops this
 *   to O(B * M(N) log^2 N) ≈ 2 * 10^10 — about 5x faster on the
 *   arithmetic count alone, with the AVX-512 muladd constant factor
 *   favouring the Fenger pipeline further (it has fewer branches and
 *   no scatter-gather on the input chunks).
 *
 * CONSTRAINTS
 * -----------
 *   - Any numInputs / numRecovery (issue #59 A2 always-pad routing):
 *     non-power-of-2 counts are padded to the next power of 2 with
 *     synthetic zero-data inputs placed at a base above both the real
 *     input range and the (padded) recovery range. Synthetic data is
 *     never read, so the real recovery rows are bit-identical to the
 *     unpadded computation (test_gf64_fenger_padded + the engine
 *     routing contract).
 *   - The disjointness condition (no x_c coincides with y_r) is enforced
 *     by gf64_fenger_matvec itself via the V(y_r) inverse; the padded
 *     routing additionally falls back to the legacy path when the
 *     synthetic base would overflow uint64 or when a padded recovery
 *     range would reach into the real input range (non-canonical id
 *     ordering).
 *
 * THREADING
 * ---------
 *   Per-thread sharding across the B-axis: the prepare phase builds the
 *   trees and weights ONCE in the calling thread, then B is split into
 *   numThreads slices and each thread runs gf64_fenger_execute on its
 *   slice. The threads share read-only state (T_X, T_Y, V_prime,
 *   V_at_y_inv); the execute function allocates per-call scratch via
 *   gf64_multi_point_* and is therefore reentrant.
 *
 *   numThreads = 0 selects "auto": cap at the smaller of (blockSize64,
 *   GetEffectiveCpuCount()) to avoid spinning more threads than words.
 *
 * FALLBACK
 * --------
 *   Non-power-of-2 counts are padded (issue #59 A2 always-pad routing,
 *   see CONSTRAINTS). The only fallbacks to ComputeRecoveryBlocks are
 *   synthetic-base uint64 overflow and padded-recovery collision with
 *   the real input range (non-canonical id ordering). The legacy path
 *   is bit-exact and works for any shape, so the NAPI binding contract
 *   stays safe for all workloads.
 * ============================================================================
 */

#include "par3_engine.h"
#include "gf64_fenger.h"
#include "gf64_global.h"

#include <stdint.h>
#include <stdlib.h>
#include <thread>
#include <vector>

/* C-linkage declarations for the Fenger pipeline. */
extern "C" gf64_fenger_ctx *gf64_fenger_prepare(
	uint64_t firstInput,
	uint64_t firstRecovery,
	size_t N,
	size_t R
);
/* Padded route exists for bit-exact correctness on small N; not routed
 * from production dispatch (see FALLBACK above and header comment). */
extern "C" gf64_fenger_ctx *gf64_fenger_prepare_padded(
	uint64_t firstInput,
	uint64_t firstRecovery,
	size_t numInputs,
	size_t numRecovery,
	size_t numInputsPadded,
	size_t numRecoveryPadded,
	uint64_t syntheticInputBase
);
extern "C" void gf64_fenger_execute(
	const gf64_fenger_ctx *ctx,
	const gf64_t *in,  size_t B,
	      gf64_t *out,
	size_t w_start, size_t w_end
);
extern "C" void gf64_fenger_release(gf64_fenger_ctx *ctx);

void GF64Controller::ComputeRecoveryBlocksFenger(
	const gf64_t* inputs, size_t numInputs,
	gf64_t* recovery, size_t numRecovery,
	size_t blockSize64,
	uint64_t firstInput, uint64_t firstRecovery,
	int numThreads
) {
	/* Trivial-input short-circuit, matching the engine convention. */
	if (numInputs == 0 || numRecovery == 0 || blockSize64 == 0) {
		return;
	}

	/* Zero the output buffer. The Fenger matvec is overwrite-semantics;
	 * any output row that no input contributes to (impossible under the
	 * disjointness condition, but kept for parity with the legacy path)
	 * must be explicitly zeroed. */
	const size_t out_words = numRecovery * blockSize64;
	for (size_t i = 0; i < out_words; i++) {
		recovery[i] = 0;
	}

	/* Auto thread count: cap at the smaller of (blockSize64, cpu count).
	 * blockSize64 is the B-axis size; no point in spinning more threads
	 * than words when each thread takes a contiguous B-slice. */
	if (numThreads <= 0) {
		int cpu = GetEffectiveCpuCount();
		if (cpu <= 0) cpu = 1;
		size_t max_useful = (size_t)cpu;
		if (max_useful > blockSize64) max_useful = blockSize64;
		numThreads = (int)max_useful;
		if (numThreads < 1) numThreads = 1;
	}

	/* Issue #59 §3 / A2: always-pad routing. The subproduct tree needs
	 * power-of-2 point counts; non-power-of-2 workloads are padded with
	 * synthetic zero-data inputs at a base disjoint from both the real
	 * input range and the recovery range. The Fenger identity holds for
	 * any point set and synthetic inputs contribute 0 (their DATA is
	 * never read), so the real recovery rows are bit-identical to the
	 * unpadded computation (verified by test_gf64_fenger_padded and the
	 * engine's padded routing contract). */
	size_t numInputsPadded = 1;
	while (numInputsPadded < numInputs) {
		/* Overflow guard: shifting a count >= 2^63 wraps to 0 and would
		 * loop forever. Such counts cannot be padded — fall back to the
		 * legacy path (cubic review f70a81ef P2). */
		if (numInputsPadded > (SIZE_MAX >> 1)) {
			GF64Controller::ComputeRecoveryBlocks(
				inputs, numInputs,
				recovery, numRecovery,
				blockSize64,
				firstInput, firstRecovery,
				/* numThreads */ 0
			);
			return;
		}
		numInputsPadded <<= 1;
	}
	size_t numRecoveryPadded = 1;
	while (numRecoveryPadded < numRecovery) {
		if (numRecoveryPadded > (SIZE_MAX >> 1)) {
			GF64Controller::ComputeRecoveryBlocks(
				inputs, numInputs,
				recovery, numRecovery,
				blockSize64,
				firstInput, firstRecovery,
				/* numThreads */ 0
			);
			return;
		}
		numRecoveryPadded <<= 1;
	}

	const bool padInputs   = (numInputsPadded != numInputs);
	const bool padRecovery = (numRecoveryPadded != numRecovery);

	/* Any overlap of the REAL input range [firstInput, inEnd) with the
	 * (padded) recovery range [firstRecovery, recEnd) forces the legacy
	 * fallback: Fenger would compute V(y_r) == 0 on a colliding point
	 * and emit a zero recovery row instead of the legacy row. Checked
	 * for EVERY workload — including unpadded power-of-2 shapes, which
	 * never enter the synthetic-base block below (cubic review f70a81ef
	 * P2 / f44ead49 P1 / ce3679b0 P2). inEnd/recEnd saturate on uint64
	 * overflow so the comparison stays sound. */
	const uint64_t inEnd  = (firstInput > UINT64_MAX - (uint64_t)numInputs)
	                        ? UINT64_MAX
	                        : firstInput + (uint64_t)numInputs;
	const uint64_t recEnd = (firstRecovery > UINT64_MAX - (uint64_t)numRecoveryPadded)
	                        ? UINT64_MAX
	                        : firstRecovery + (uint64_t)numRecoveryPadded;
	if ((firstRecovery < inEnd) && (firstInput < recEnd)) {
		GF64Controller::ComputeRecoveryBlocks(
			inputs, numInputs,
			recovery, numRecovery,
			blockSize64,
			firstInput, firstRecovery,
			/* numThreads */ 0
		);
		return;
	}

	/* Synthetic input base: strictly above both the real input range
	 * [firstInput, inEnd) and the (padded) recovery range
	 * [firstRecovery, recEnd), so the synthetic points collide with
	 * nothing. On uint64 overflow (the synthetic base itself would
	 * wrap), fall back to the legacy path rather than risking a point
	 * collision (V(y_r) == 0 would abort). */
	uint64_t syntheticInputBase = 0;
	if (padInputs || padRecovery) {
		const uint64_t padCount = (uint64_t)(numInputsPadded - numInputs);
		const bool overflow =
			firstInput > UINT64_MAX - (uint64_t)numInputs ||
			firstRecovery > UINT64_MAX - (uint64_t)numRecoveryPadded ||
			(inEnd > recEnd ? inEnd : recEnd) > UINT64_MAX - padCount;
		if (overflow) {
			GF64Controller::ComputeRecoveryBlocks(
				inputs, numInputs,
				recovery, numRecovery,
				blockSize64,
				firstInput, firstRecovery,
				/* numThreads */ 0
			);
			return;
		}
		syntheticInputBase = (inEnd > recEnd) ? inEnd : recEnd;
	}

	gf64_fenger_ctx *ctx;
	if (padInputs || padRecovery) {
		ctx = gf64_fenger_prepare_padded(
			firstInput, firstRecovery,
			numInputs, numRecovery,
			numInputsPadded, numRecoveryPadded,
			syntheticInputBase
		);
	} else {
		ctx = gf64_fenger_prepare(
			firstInput, firstRecovery, numInputs, numRecovery
		);
	}

	/* Single-thread fast path: avoid the std::thread overhead. */
	if (numThreads == 1) {
		gf64_fenger_execute(ctx, inputs, blockSize64, recovery, 0, blockSize64);
		gf64_fenger_release(ctx);
		return;
	}

	/* Per-thread sharding: divide B across numThreads contiguous slices.
	 * Each slice is roughly ceil(B / numThreads); the last slice absorbs
	 * the remainder. */
	std::vector<std::thread> workers;
	workers.reserve((size_t)numThreads);
	const size_t chunk = (blockSize64 + (size_t)numThreads - 1)
	                     / (size_t)numThreads;
	size_t w_start = 0;
	for (int t = 0; t < numThreads && w_start < blockSize64; t++) {
		size_t w_end = w_start + chunk;
		if (w_end > blockSize64) w_end = blockSize64;
		workers.emplace_back(
			[ctx, inputs, blockSize64, recovery, w_start, w_end]() {
				gf64_fenger_execute(
					ctx, inputs, blockSize64, recovery, w_start, w_end
				);
			}
		);
		w_start = w_end;
	}
	for (auto& th : workers) th.join();

	gf64_fenger_release(ctx);
}
