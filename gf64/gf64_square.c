/*
 * ============================================================================
 * gf64/gf64_square.c — GF(2^64) Frobenius squaring (scalar)
 *
 * T2 of the par3-cauchy-fft-kernel plan. Provides:
 *   void gf64_square(gf64_t* out, const gf64_t* in, size_t N)
 *
 * The AVX-512 kernel (gf64_square_avx512) was split into
 * gf64/gf64_square_avx512.c (issue #62): the MSVC /arch flag is whole-TU,
 * and compiling the scalar squaring with /arch:AVX512 let the
 * auto-vectorizer emit EVEX (ZMM) instructions into the scalar path,
 * which SIGILLs on hosts without AVX-512 even when the dispatcher
 * correctly selected the AVX-2 method.
 *
 * MATH
 *   Squaring in GF(2^k) is the Frobenius endomorphism f(x) -> f(x)^2.
 *   For each input bit i, the output bit sits at position 2i. The full
 *   128-bit intermediate result (lo, hi) is then reduced modulo the
 *   irreducible polynomial
 *     x^64 + x^4 + x^3 + x + 1          (low 64 bits = 0x1B)
 *   using the SAME arithmetic as gf64/gf64_solve.c:14-36 (the existing
 *   scalar gf64_mul is preserved untouched per scope discipline).
 *
 * SCOPE
 *   This file is a NEW translation unit. It does NOT modify
 *   gf64/gf64_solve.c:7-37, gf64_mul_avx512.c, or any existing AVX-512
 *   kernel. The reduction arithmetic mirrors gf64_solve.c:14-36 and the
 *   vectorized reduction in gf64_mul_avx512.c:96-171, so the squaring is
 *   bit-exact to gf64_mul_reference(a, a).
 *
 * SCALAR ALGORITHM
 *   SWAR cascade (Hacker's Delight / Bit Twiddling Hacks) on each 32-bit
 *   half of the input — bit k -> bit 2k within a 64-bit value:
 *     x = (x | (x << 16)) & 0x0000FFFF0000FFFF
 *     x = (x | (x <<  8)) & 0x00FF00FF00FF00FF
 *     x = (x | (x <<  4)) & 0x0F0F0F0F0F0F0F0F
 *     x = (x | (x <<  2)) & 0x3333333333333333
 *     x = (x | (x <<  1)) & 0x5555555555555555
 *   Applied to a_lo = a & 0xFFFFFFFF and a_hi = a >> 32. After the cascade,
 *   bit k of a_lo sits at position 2k of `lo`, and bit k of a_hi sits at
 *   position 2k of `hi`. The full 128-bit squaring is (hi << 64) | lo.
 *   Then reduce lo by hi * 0x1B (with overflow R_hi captured in the low
 *   4 bits and re-reduced).
 *
 * TAIL
 *   The AVX-512 kernel falls back to this scalar function for the N % 8
 *   epilog (mirrors the tail strategy of every other SIMD kernel in this
 *   codebase: gf64_mul_avx512.c, gf64_region_*_arr.c, etc.).
 * ============================================================================
 */

#include "gf64_square.h"
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
#define GF64_SQUARE_TU_BODY 1
#else
#define GF64_SQUARE_TU_BODY 0
#endif
#if GF64_SQUARE_TU_BODY

/* ---------------------------------------------------------------------------
 * Scalar Frobenius squaring of a single GF(2^64) element.
 *
 * Splits the input into two 32-bit halves, runs the SWAR cascade on each
 * (bit k -> bit 2k within the 64-bit value), then reduces the resulting
 * (lo, hi) pair modulo the irreducible polynomial x^64 + x^4 + x^3 + x + 1.
 *
 * Bit-exact to gf64_mul_reference(a, a). The reduction arithmetic is
 * the SAME as gf64_solve.c:14-36 — only the "hi" here is the upper 64
 * bits of the squaring result, NOT the upper 64 bits of a carry-less
 * product. Otherwise the R_hi / t2 logic is identical.
 * --------------------------------------------------------------------------- */
static inline gf64_t gf64_square_one(gf64_t a) {
	/* SWAR cascade: spread bit k -> bit 2k within each 64-bit value. */
	gf64_t lo = a & 0xFFFFFFFFULL;
	gf64_t hi = a >> 32;

	lo = (lo | (lo << 16)) & (gf64_t)0x0000FFFF0000FFFFULL;
	lo = (lo | (lo <<  8)) & (gf64_t)0x00FF00FF00FF00FFULL;
	lo = (lo | (lo <<  4)) & (gf64_t)0x0F0F0F0F0F0F0F0FULL;
	lo = (lo | (lo <<  2)) & (gf64_t)0x3333333333333333ULL;
	lo = (lo | (lo <<  1)) & (gf64_t)0x5555555555555555ULL;

	hi = (hi | (hi << 16)) & (gf64_t)0x0000FFFF0000FFFFULL;
	hi = (hi | (hi <<  8)) & (gf64_t)0x00FF00FF00FF00FFULL;
	hi = (hi | (hi <<  4)) & (gf64_t)0x0F0F0F0F0F0F0F0FULL;
	hi = (hi | (hi <<  2)) & (gf64_t)0x3333333333333333ULL;
	hi = (hi | (hi <<  1)) & (gf64_t)0x5555555555555555ULL;

	/* Reduce: result = lo ^ (hi * 0x1B) (in GF(2)), with overflow R_hi.
	 * Mirrors gf64_solve.c:14-36 exactly. */
	gf64_t t_lo = (hi << 4) ^ (hi << 3) ^ (hi << 1) ^ hi;

	gf64_t R_hi =
		(((hi >> 60) ^ (hi >> 61) ^ (hi >> 63)) & 1) |
		((((hi >> 61) ^ (hi >> 62)) & 1) << 1) |
		((((hi >> 62) ^ (hi >> 63)) & 1) << 2) |
		(((hi >> 63) & 1) << 3);

	gf64_t t2 = (R_hi << 4) ^ (R_hi << 3) ^ (R_hi << 1) ^ R_hi;

	return lo ^ t_lo ^ t2;
}

/* ---------------------------------------------------------------------------
 * Scalar element-by-element Frobenius squaring of a length-N vector.
 * --------------------------------------------------------------------------- */
void gf64_square(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT in,
	size_t N
) {
	for (size_t i = 0; i < N; i++) {
		out[i] = gf64_square_one(in[i]);
	}
}

#endif /* GF64_SQUARE_TU_BODY */
HEDLEY_END_C_DECLS
