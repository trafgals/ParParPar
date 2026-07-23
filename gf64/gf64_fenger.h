/*
 * ============================================================================
 * gf64/gf64_fenger.h — Fenger Toeplitz pipeline over GF(2^64)[x].
 *
 * Issue #28 — replaces the per-recovery polynomial interpolation currently
 * embedded in ComputeRecoveryBlocks with the asymptotic Bostan-Schost
 * multi-point interpolation + evaluation pipeline.
 *
 * The Cauchy matrix-vector product
 *
 *   out[r][w] = XOR_c in[c][w] / (y_r XOR x_c)
 *
 * with x_c = firstInput + c (input points) and y_r = firstRecovery + r
 * (recovery points) is decomposed via the Fenger 2009 identity
 *
 *   sum_c in[c] / (y - x_c)  =  p(y) / V(y)
 *
 * where V(x) = prod_c (x XOR x_c) is the subproduct-tree root polynomial,
 * p(x) is the unique polynomial of degree < N satisfying
 *
 *   p(x_c) = in[c] * V'(x_c)
 *
 * and V'(x_c) is the formal derivative of V at x_c.
 *
 * TWO PUBLIC APIs
 * ---------------
 *
 * 1. Convenience wrapper — rebuilds state per call, single-threaded:
 *
 *      void gf64_fenger_matvec(in, N, B, out, R, firstInput, firstRecovery);
 *
 *    Use this when the input/recovery sets change every call.
 *
 * 2. Prepare/execute/release — supports per-thread sharding over B words:
 *
 *      gf64_fenger_ctx *ctx = gf64_fenger_prepare(firstInput, firstRecovery, N, R);
 *      // Spawn threads; each runs gf64_fenger_execute on its own B-slice:
 *      gf64_fenger_execute(ctx, in, B, out, w_start, w_end);
 *      // ... wait for threads ...
 *      gf64_fenger_release(ctx);
 *
 *    The trees and weights are computed once in prepare() and shared
 *    read-only across all execute() calls (which only use stack-local
 *    scratch via gf64_multi_point_interp / gf64_multi_point_eval).
 *
 * Asymptotic cost (per call, all steps):
 *
 *   setup (one-time per call):
 *     - Build T_X:                O(M(N) log N)
 *     - Build T_Y:                O(M(R) log R)
 *     - Barycentric weights:      O(M(N) log² N + N)        [T7 — Bostan-Schost MPE]
 *     - V(y_r) for each r:        O(M(N) log² N + M(R) log² R)
 *
 *   per-word (B words):
 *     - Interp:  O(M(N) log² N)
 *     - Eval:    O(M(N) log² N)    (since deg p_w < N)
 *     - Divide:  O(R)
 *
 * Total: O(B · M(N) log² N + setup). The per-word cost is INDEPENDENT
 * of R once the setup amortizes across the B words — the structural
 * win over the current O(B · N · R) Cauchy path that gates issue #27.
 *
 * The V(y_r) MPE in step 4 currently evaluates V(x) (the root polynomial
 * of T_X) at the leaves of T_Y using gf64_multi_point_eval, which is
 * Bostan-Schost top-down. Per-recovery-point Horner is NOT used here —
 * the MPE form keeps step 4 at O((N + R) log² (N + R)).
 *
 * Memory:
 *   - Two subproduct trees (T_X, T_Y): O(N) and O(R) storage.
 *   - One V'(x_c) buffer of N elements.
 *   - One V(y_r)^(-1) buffer of R elements.
 *   - Per-execute scratch (allocated inside gf64_multi_point_*):
 *     8·N + 4·max(N+1, B) gf64_t.
 * ============================================================================
 */

#ifndef GF64_FENGER_H
#define GF64_FENGER_H

#include "gf64_global.h"
#include "gf64_subproduct.h"

#include <stddef.h>
#include <stdint.h>

HEDLEY_BEGIN_C_DECLS

/* Opaque prepare/execute context. Fields are private; use the
 * gf64_fenger_prepare / execute / release API below. */
typedef struct gf64_fenger_ctx gf64_fenger_ctx;

/*
 * Prepare the Fenger Toeplitz pipeline for a fixed (firstInput,
 * firstRecovery, N, R) configuration. Builds T_X, T_Y, the barycentric
 * weights (1/V'(x_c)), the V'(x_c) buffer, and the V(y_r) inverse
 * buffer. Returns a heap-allocated context to be passed to
 * gf64_fenger_execute and freed with gf64_fenger_release.
 *
 * On allocation failure: calls abort() (matches the abort-on-OOM
 * convention in gf64_alloc_halves and the Fenger pipeline's other
 * allocs).
 *
 * Returns NULL when N == 0 or R == 0 (no work to do).
 */
gf64_fenger_ctx *gf64_fenger_prepare(
	uint64_t firstInput,
	uint64_t firstRecovery,
	size_t N,
	size_t R
);

/*
 * Run the per-word Fenger pipeline on the input/output data over the
 * word range [w_start, w_end). Used to sharded the B-axis across
 * multiple threads — each thread owns a disjoint slice.
 *
 * Input layout: in[c * B + w] for c in [0, N), w in [0, B).
 * Output layout: out[r * B + w] for r in [0, R), w in [0, B).
 *
 * ctx must have been returned by gf64_fenger_prepare with matching
 * (N, R, firstInput, firstRecovery). The function does not modify ctx.
 *
 * Threads concurrently invoking gf64_fenger_execute on the SAME ctx
 * are safe as long as each thread's [w_start, w_end) is disjoint and
 * the underlying gf64_multi_point_* functions are reentrant (they
 * are — they malloc/free their own scratch).
 *
 * w_start must be <= w_end; w_end must be <= B.
 */
void gf64_fenger_execute(
	const gf64_fenger_ctx *ctx,
	const gf64_t *in,  size_t B,
	      gf64_t *out,
	size_t w_start, size_t w_end
);

/*
 * Free the resources held by a context returned by gf64_fenger_prepare.
 * Safe on a NULL pointer (no-op).
 */
void gf64_fenger_release(gf64_fenger_ctx *ctx);

/*
 * Convenience wrapper: prepare + execute + release on the full B range
 * in a single thread. Equivalent to:
 *
 *   ctx = gf64_fenger_prepare(firstInput, firstRecovery, N, R);
 *   gf64_fenger_execute(ctx, in, B, out, 0, B);
 *   gf64_fenger_release(ctx);
 *
 * N must be 0, 1, or a power of 2 (the subproduct tree builder's
 * constraint). R must similarly be 0, 1, or a power of 2.
 *
 * firstInput and firstRecovery must produce DISTINCT evaluation points:
 *   {firstInput + c} ∩ {firstRecovery + r} = ∅
 * (equivalently V(y_r) ≠ 0 for all r).
 */
void gf64_fenger_matvec(
	const gf64_t *in,  size_t N, size_t B,
	      gf64_t *out, size_t R,
	uint64_t firstInput,
	uint64_t firstRecovery
);

HEDLEY_END_C_DECLS

#endif /* GF64_FENGER_H */
