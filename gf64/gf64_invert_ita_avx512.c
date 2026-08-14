/*
 * ============================================================================
 * gf64/gf64_invert_ita_avx512.c — Itoh-Tsujii batched inversion, AVX-512 kernel
 *
 * Split from gf64_invert_ita.c (issue #62): the AVX-512 batch lives in its
 * own translation unit so the scalar TU (gf64_invert_ita.c) can be compiled
 * WITHOUT /arch:AVX512. The MSVC /arch flag is whole-TU — compiling the
 * scalar inversion with /arch:AVX512 let the auto-vectorizer emit EVEX
 * (ZMM) instructions into the scalar path, which SIGILLs on hosts without
 * AVX-512 (windows-2025 fleet AMD Milan runners) even when the dispatcher
 * correctly selected the AVX-2 method.
 *
 * This TU is compiled with /arch:AVX512 (MSVC) or per-function
 * __attribute__((target("avx512f,vpclmulqdq"))) (GCC/Clang) and is the
 * ONLY home of the batched inversion's _mm512 intrinsics.
 *
 * Math: identical to gf64_invert_ita.c — bit-exact to
 * gf64_invert_ita_one applied element-wise.
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
#define GF64_INVERT_ITA_AVX512_TU_BODY 1
#else
#define GF64_INVERT_ITA_AVX512_TU_BODY 0
#endif
#if GF64_INVERT_ITA_AVX512_TU_BODY

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

#endif /* GF64_INVERT_ITA_AVX512_TU_BODY */
HEDLEY_END_C_DECLS
