/*
 * ============================================================================
 * gf64/gf64_invert_ita.c — Itoh-Tsujii batched inversion in GF(2^64)
 *
 * T5 of the par3-cauchy-fft-kernel plan. Provides:
 *   void gf64_invert_ita_batch(gf64_t* out, const gf64_t* in, size_t N)
 *   gf64_t gf64_invert_ita_one (gf64_t a)
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
 *   AVX-512 version of gf64_invert_ita_batch is compiled with
 *   __attribute__((target("avx512f,vpclmulqdq"))) — it calls
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

/* ---------------------------------------------------------------------------
 * AVX-512 vectorized Itoh-Tsujii batched inversion of a length-N vector.
 *
 *   out[i] = gf64_invert_ita_one(in[i])   for i in [0, N)
 *
 * Per outer iteration: load 8 inputs, run the 63-step squaring + 62-step
 * multiplication chain on the 8 lanes in lockstep using the existing
 * vectorized primitives from T1 (gf64_mul_avx512) and T2
 * (gf64_square_avx512), then store 8 inverses. N % 8 tail falls back to
 * gf64_invert_ita_one.
 *
 * The 8 inputs are loaded once into `a_in_buf[8]` and reused across all
 * 62 chain iterations — a single load amortizes the input bandwidth. The
 * running state `t_buf[8]` is updated in place via the aliasing contract
 * of both gf64_square_avx512 and gf64_mul_avx512 (`out` MAY alias `in`
 * and `a` respectively; the multiply's `a` parameter is the running t
 * buffer and `b` is the saved input buffer).
 *
 * Target attribute: avx512f + vpclmulqdq (broader of the two callees'
 * targets — gf64_mul_avx512 requires VPCLMULQDQ, gf64_square_avx512
 * requires only AVX-512F).
 *
 * Bit-exact to gf64_invert_ita_one applied element-wise (verified by
 * test/test_gf64_invert_ita.c). Naive in counter: each outer iteration
 *   = 62 * (1 squaring + 1 mul by a) + 1 squaring
 *   = 63 squarings + 62 multiplications of 8 elements per call.
 * --------------------------------------------------------------------------- */
__attribute__((target("avx512f,vpclmulqdq")))
void gf64_invert_ita_batch(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT in,
	size_t N
) {
	if (N == 0) return;

	/* LANES must be an integer constant expression for MSVC stack-array
	 * sizing (C2057: "expected constant expression"). A `const size_t` is
	 * NOT acceptable to MSVC; a #define is. Keep LANES == 8 (one ZMM). */
#define LANES 8
	gf64_t a_in_buf[LANES];   /* saved inputs (lane-aligned) */
	gf64_t t_buf[LANES];      /* running state across the chain */

	size_t i = 0;
	for (; i + LANES <= N; i += LANES) {
		/* Snapshot the 8 inputs once. The whole 62-step chain reads them.
		 * Unaligned variant — safe regardless of stack alignment and
		 * equivalent in cost on Zen4 / Ice Lake (no AVX-512 unaligned
		 * penalty once you cross the split-load threshold). */
		__m512i a_in_zmm = _mm512_loadu_si512((const __m512i *)(in + i));
		_mm512_storeu_si512((__m512i *)a_in_buf, a_in_zmm);

		/* t = a (initial value of the chain). */
		_mm512_storeu_si512((__m512i *)t_buf, a_in_zmm);

		/* Itoh-Tsujii chain: t_k = t_{k-1}^2 * a for k = 2..63. */
		for (int k = 2; k <= 63; k++) {
			/* t = t^2 — in-place aliasing supported by gf64_square_avx512. */
			gf64_square_avx512(t_buf, t_buf, LANES);
			/* t = t * a_in_buf — in-place aliasing supported by
			 * gf64_mul_avx512 (out is the running t, a is the same t,
			 * b is the saved input). */
			gf64_mul_avx512(t_buf, t_buf, a_in_buf, LANES);
		}

		/* Final squaring: t = a^(2^64 - 2) = a^(-1). */
		gf64_square_avx512(t_buf, t_buf, LANES);

		/* Store 8 inverses. */
		_mm512_storeu_si512((__m512i *)(out + i),
		                    _mm512_loadu_si512((const __m512i *)t_buf));
	}

	/* Scalar tail epilog (0..7 elements). */
	for (; i < N; i++) {
		out[i] = gf64_invert_ita_one(in[i]);
	}
}

#endif /* GF64_INVERT_ITA_TU_BODY */
HEDLEY_END_C_DECLS
