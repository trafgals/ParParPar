/*
 * ============================================================================
 * gf64/gf64_invert_ita.c — Itoh-Tsujii batched inversion in GF(2^64)
 *
 * T5 of the par3-cauchy-fft-kernel plan. Provides:
 *   gf64_t gf64_invert_ita_one (gf64_t a)
 *
 * The AVX-512 batched kernel (gf64_invert_ita_batch) was split into
 * gf64/gf64_invert_ita_avx512.c (issue #62): the MSVC /arch flag is
 * whole-TU, and compiling the scalar inversion with /arch:AVX512 let the
 * auto-vectorizer emit EVEX (ZMM) instructions into the scalar path,
 * which SIGILLs on hosts without AVX-512 even when the dispatcher
 * correctly selected the AVX-2 method.
 *
 * MATH
 *   In GF(2^k), a^(2^k - 1) = 1 for any non-zero a (Fermat's Little
 *   Theorem). Therefore a^(-1) = a^(2^k - 2). For k = 64 we compute
 *   a^(2^64 - 2) using the Itoh-Tsujii chain derived from the recurrence
 *
 *       t_1  = a
 *       t_k  = (t_{k-1})^2 * a       for k = 2, 3, ..., k_max
 *       a^(2^k - 1)
 *
 *   PROOF OF INVARIANT  (by induction on k)
 *     base:    t_1 = a = a^(2^1 - 1)
 *     step:    t_{k-1} = a^(2^(k-1) - 1)
 *              t_k = t_{k-1}^2 * a
 *                  = (a^(2^(k-1) - 1))^2 * a
 *                  = a^(2 * (2^(k-1) - 1)) * a
 *                  = a^(2^k - 2) * a
 *                  = a^(2^k - 1)               [since -2 + 1 = -1 in GF(2)]
 *
 *   Picking k_max = 63, we get t_63 = a^(2^63 - 1). One final squaring
 *   produces a^(2^64 - 2) = a^(-1):
 *
 *       (a^(2^63 - 1))^2 = a^(2 * (2^63 - 1)) = a^(2^64 - 2)
 *
 *   Total per-element cost: 62 iterations * (1 square + 1 mul) + 1 square
 *                         = 63 squarings + 62 multiplications.
 *
 * ALGORITHM SCOPE (vs the plan's spec)
 *   The plan's "Must have" header mentions an addition chain
 *   [1, 2, 3, 7, 15, 31, 63] producing a^(2^6 - 1) = a^63 (8 multiplications).
 *   That computes a^(2^6 - 1), NOT a^(2^64 - 2). The plan's task spec
 *   (this file) is the right reduction: we need a^(2^64 - 2), not a^63. The
 *   chain used here is the implicit binary one embedded in the iterative
 *   recurrence t_k = t_{k-1}^2 * a — 62 squarings + 62 multiplications + 1
 *   final squaring, exactly as the task header specifies. This is the
 *   textbook Itoh-Tsujii construction for GF(2^k) when k itself is a power
 *   of 2 (k = 64 = 2^6); the [1,2,3,7,15,31,63] chain is the special case
 *   that stops at k=6, intended for further optimization.
 *
 * ISA / DISPATCH
 *   gf64_invert_ita_one is portable C (single-element, scalar). The
 *   AVX-512 version of gf64_invert_ita_batch lives in
 *   gf64_invert_ita_avx512.c (compiled with /arch:AVX512 on MSVC,
 *   __attribute__((target("avx512f,vpclmulqdq"))) on GCC/Clang) — it calls
 *   gf64_mul_avx512 (which requires VPCLMULQDQ) and gf64_square_avx512
 *   (which requires only AVX-512F but is a separate TU compiled with that
 *   target). The N % 8 tail falls back to gf64_invert_ita_one (scalar).
 *   Caller (gf64_dispatch.c) is responsible for binding this only when
 *   the host supports AVX-512F + VPCLMULQDQ; on smaller ISAs the
 *   existing EEA-based gf64_inverse_batch_* path is used.
 *
 * CONVENTION
 *   gf64_invert_ita_one(0) returns 0 (the multiplicative inverse of 0
 *   does not exist; the convention used by gf64_inverse in
 *   gf64_single.c is to return 0 for zero input).
 * ============================================================================
 */

#include "gf64_invert_ita.h"
#include "gf64_square.h"
#include "gf64_mul.h"
#include "gf64_global.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include <stdint.h>
#include <stddef.h>

#ifndef __GNUC__
/* Stub out GCC __attribute__((target(...))) under MSVC.
 * Use variadic macro so the entire trailing ((...)) parens are eaten as
 * a single comma-separated argument list. */
#define __attribute__(...) /* __attribute__ not supported under MSVC */
#endif

HEDLEY_BEGIN_C_DECLS

#if defined(__x86_64__) || defined(_M_X64)
#define GF64_INVERT_ITA_TU_BODY 1
#else
#define GF64_INVERT_ITA_TU_BODY 0
#endif
#if GF64_INVERT_ITA_TU_BODY

/* Public scalar SSE2 multiplication, defined in gf64/gf64_single.c:6.
 * Bit-exact to gf64_mul in gf64_solve.c:7-37. */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* ---------------------------------------------------------------------------
 * Scalar Itoh-Tsujii inversion of a single GF(2^64) element.
 *
 *   result = a^(2^64 - 2) = a^(-1) (in GF(2^64), for a != 0)
 *
 * Uses the iterative chain t_k = t_{k-1}^2 * a for k = 2..63 (62 iterations,
 * 62 squarings + 62 multiplications), then squares once more (1 final
 * squaring) to get a^(2^64 - 2). Total per-element cost:
 *   63 squarings + 62 multiplications.
 *
 * Squarings go through gf64_square (the T2 Frobenius primitive — bit-exact
 * to gf64_mul_reference(a, a)). Multiplications by `a` go through
 * gf64_mul_reference (the scalar reference mul — bit-exact to gf64_mul in
 * gf64_solve.c:7-37). All operations are branchless in the hot path; the
 * `a == 0` short-circuit runs first.
 * --------------------------------------------------------------------------- */
gf64_t gf64_invert_ita_one(gf64_t a) {
	if (a == 0) return 0;  /* multiplicative inverse of 0 is undefined; 0 convention. */

	gf64_t t = a;
	gf64_t sq = 0;          /* temporary for gf64_square N=1 output */

	/* Itoh-Tsujii chain: t_k = t_{k-1}^2 * a for k = 2..63.
	 * Each iteration: 1 squaring (frobenius, bit-exact to mul(a,a)) then
	 * 1 multiplication by `a` (gf64_mul_reference). */
	for (int k = 2; k <= 63; k++) {
		/* t = t^2 in GF(2^64). */
		gf64_square(&sq, &t, 1);
		t = sq;
		/* t = t * a in GF(2^64). */
		t = gf64_mul_reference(t, a);
	}

	/* After loop: t = a^(2^63 - 1). One more squaring yields a^(2^64 - 2)
	 * = a^(-1) by Fermat's Little Theorem (a^(2^64 - 1) = 1 in GF(2^64)). */
	gf64_square(&sq, &t, 1);
	return sq;
}

#endif /* GF64_INVERT_ITA_TU_BODY */
HEDLEY_END_C_DECLS
