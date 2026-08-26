/*
 * ============================================================================
 * gf64/gf64_elem_mul.c — ISA-dispatched element-wise GF(2^64) multiply
 *
 * Issue #59 Wave 1 / T3. Replaces scalar `gf64_mul_reference` calls in
 * the Fenger Toeplitz pipeline's execute stages 4a/4d (gf64_fenger.c:269-296
 * and 706-803) with a single contiguous element-wise multiply, dispatched
 * via the standard GF64Method table (`gf64_current_method`).
 *
 * SCOPE
 *   This file is a NEW translation unit. It does NOT modify
 *   gf64/gf64_solve.c:7-37 (the existing scalar gf64_mul is preserved
 *   untouched for the Solve path). On AVX-512 hosts the existing
 *   clmul-Barrett lane primitive (gf64_mul_avx512.c) processes 8 GF(2^64)
 *   elements per VPCLMULQDQ pair; AVX-2 / SSSE3 / scalar hosts fall back
 *   to a scalar loop using the public SSE2 reference. The reduction
 *   arithmetic is bit-exact to the scalar reference, so the two paths are
 *   interchangeable.
 *
 * ISA / DISPATCH
 *   The dispatch gate uses TWO runtime flags, not just gf64_current_method:
 *     1. gf64_current_method == GF64_AVX512  — workload chose AVX-512
 *     2. gf64_has_vpclmulqdq                  — host actually has it
 *
 *   The second gate is required because gf64_current_method may be bound
 *   to GF64_AVX512 on hosts that lack VPCLMULQDQ (e.g. PAR3_GF64_USE_AVX512=1
 *   operator force-on). Without the SIGILL-tested ZMM probe gate, a kernel
 *   call would raise SIGILL on first ZMM instruction (cubic review
 *   4914681432 P1 — the same discipline already applied in
 *   gf64_barycentric.c:208 and gf64_invert_ita.c).
 *
 *   On non-x86_64 builds (no AVX-512 available), the file's only
 *   AVX-512 branch is compiled out, so dispatch always picks scalar.
 *
 * TAIL
 *   The scalar fallback handles N == 0 as a no-op (the loop body is
 *   trivially empty). The AVX-512 path delegates the tail to its own
 *   scalar epilog (gf64_mul_avx512.c:313-315).
 *
 * RE-ENTRANCY
 *   The function is re-entrant and stateless; safe for the Fenger
 *   pipeline's multi-threaded execute (each worker thread calls into
 *   its own copy via the caller's slab partitioning).
 * ============================================================================
 */

#include "gf64_elem_mul.h"

#if defined(__x86_64__) || defined(_M_X64)
#include "gf64_mul.h"  /* gf64_mul_avx512 */
#endif

#include <stddef.h>

HEDLEY_BEGIN_C_DECLS

/* Public scalar SSE2 reference (gf64/gf64_single.c). */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

void gf64_elem_mul(
	gf64_t *out,
	const gf64_t *a,
	const gf64_t *b,
	size_t N
) {
#if defined(__x86_64__) || defined(_M_X64)
	/* AVX-512 fast path: gated on BOTH the workload-chosen method AND
	 * the host-validated VPCLMULQDQ capability (the latter includes
	 * a SIGILL-tested ZMM probe — see gf64_dispatch.c:122). Without
	 * the second gate, a PAR3_GF64_USE_AVX512=1 force-on on a host
	 * lacking AVX-512 would SIGILL at first ZMM instruction.
	 *
	 * gf64_mul_avx512 is aliasing-safe (`out MAY alias a or b`,
	 * gf64_mul.h:30) and handles N == 0 as a no-op (its body only
	 * enters the per-iteration loop when blocks > 0; the tail loop
	 * is empty when N % 8 == 0). */
	if (gf64_current_method == GF64_AVX512 && gf64_has_vpclmulqdq) {
		gf64_mul_avx512(out, a, b, N);
		return;
	}
#endif
	/* Scalar fallback: GF64_AVX2 / GF64_SSSE3 / GF64_SCALAR, or
	 * AVX-512 host lacking the SIGILL-tested ZMM probe. The loop
	 * body is trivially empty when N == 0. */
	for (size_t i = 0; i < N; i++) {
		out[i] = gf64_mul_reference(a[i], b[i]);
	}
}

HEDLEY_END_C_DECLS