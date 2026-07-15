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
 * and V'(x_c) is the formal derivative of V at x_c. The pipeline:
 *
 *   1. Build subproduct tree T_X over the input points.
 *   2. Build subproduct tree T_Y over the recovery points.
 *   3. Compute barycentric weights (1 / V'(x_c)) at each x_c.
 *   4. Compute V(y_r) for each r (MPE of V(x) at T_Y's leaves).
 *   5. Per word w:
 *        a. weighted[c] = in[c][w] * V'(x_c)
 *        b. p_w(x) = gf64_multi_point_interp(T_X, weighted)
 *        c. p_at_y[r] = gf64_multi_point_eval(p_w, T_Y)
 *        d. out[r][w] = p_at_y[r] / V(y_r)
 *
 * Asymptotic cost (per call, all steps):
 *
 *   setup (one-time per call):
 *     - Build T_X:                O(M(N) log N)
 *     - Build T_Y:                O(M(R) log R)
 *     - Barycentric weights:      O(M(N) log² N + N)        [T7 — naive Horner
 *                                                                  for now]
 *     - V(y_r) for each r:        O(M(N) log² N + M(R) log² R)
 *
 *   per-word (B words):
 *     - Interp:  O(M(N) log² N)
 *     - Eval:    O(M(N) log² N)    (since deg p_w < N)
 *     - Divide:  O(R)
 *
 * Total: O(B · M(N) log² N + setup). This is INDEPENDENT of R once the
 * setup amortizes across the B words — the structural win over the
 * current O(B · N · R) Cauchy path that gates issue #27.
 *
 * The V(y_r) MPE in step 4 currently evaluates V(x) (the root polynomial
 * of T_X) at the leaves of T_Y using gf64_multi_point_eval, which is
 * already Bostan-Schost top-down. Per-recovery-point Horner is NOT used
 * here — the MPE form keeps step 4 at O((N + R) log² (N + R)).
 *
 * Memory:
 *   - Two subproduct trees (T_X, T_Y): O(N) and O(R) storage.
 *   - One V'(x_c) buffer of N elements.
 *   - Per-word working buffers of N + R elements (reused across B).
 * ============================================================================
 */

#ifndef GF64_FENGER_H
#define GF64_FENGER_H

#include "gf64_global.h"

#include <stddef.h>
#include <stdint.h>

HEDLEY_BEGIN_C_DECLS

/*
 * Cauchy matrix-vector product via the Fenger Toeplitz pipeline.
 *
 * Computes
 *
 *   for r in [0, R), w in [0, B):
 *     out[r * B + w] = XOR_{c = 0..N-1} in[c * B + w] / ((firstRecovery + r) XOR (firstInput + c))
 *
 * using the Bostan-Schost decomposition documented above. Both `in` and
 * `out` are caller-owned, with no aliasing between them. `in` and `out`
 * must not alias each other; `in` and `out` themselves must be sized to
 * at least N*B and R*B gf64_t values respectively.
 *
 * N must be 0, 1, or a power of 2 (the subproduct tree builder's
 * constraint). R must similarly be 0, 1, or a power of 2.
 *
 * firstInput and firstRecovery must produce DISTINCT evaluation points:
 *   {firstInput + c} ∩ {firstRecovery + r} = ∅
 * (equivalently V(y_r) ≠ 0 for all r). The function aborts on a violation
 * because the inverse V(y_r)⁻¹ is undefined at the degenerate points.
 *
 * Cost: see file header for asymptotics.
 */
void gf64_fenger_matvec(
	const gf64_t *in,  size_t N, size_t B,
	      gf64_t *out, size_t R,
	uint64_t firstInput,
	uint64_t firstRecovery
);

HEDLEY_END_C_DECLS

#endif /* GF64_FENGER_H */
