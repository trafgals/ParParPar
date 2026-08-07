/*
 * ============================================================================
 * gf64/gf64_square.c — GF(2^64) Frobenius squaring (scalar + AVX-512)
 *
 * T2 of the par3-cauchy-fft-kernel plan. Provides:
 *   void gf64_square       (gf64_t* out, const gf64_t* in, size_t N)
 *   void gf64_square_avx512(gf64_t* out, const gf64_t* in, size_t N)
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
 * AVX-512 ALGORITHM
 *   Same SWAR cascade lifted into ZMM registers via _mm512_slli_epi64
 *   (constant per-lane shift) + _mm512_or_si512 + _mm512_and_si512
 *   (masks broadcast via _mm512_set1_epi64). 8 lanes processed per
 *   iteration; reduction is a 1:1 vectorization of the scalar reduction.
 *   No VPCLMULQDQ needed — squaring is a pure shift/xor map, no
 *   carry-less multiplies.
 *
 * TAIL
 *   Both functions fall back to the scalar gf64_square_one for the N % 8
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

/* ---------------------------------------------------------------------------
 * AVX-512 vectorized SWAR cascade — 8 lanes in parallel.
 *
 * Each 64-bit lane independently undergoes the 5-step cascade that maps
 * bit k -> bit 2k. Used twice per outer iteration (once for the low 32
 * bits of each input element, once for the high 32 bits) to build the
 * (lo, hi) pair of the full 128-bit squaring.
 *
 * Pure AVX-512F — no VPCLMULQDQ / IFMA / BMI2 / VBMI dependency. The masks
 * are broadcast via _mm512_set1_epi64 (one 64-bit imm64 per mask); the
 * shifts are constant per-lane (1, 2, 4, 8, 16) via _mm512_slli_epi64.
 * --------------------------------------------------------------------------- */
__attribute__((target("avx512f")))
static inline __m512i gf64_swar_spread_512(__m512i x) {
	const __m512i m16 = _mm512_set1_epi64((int64_t)0x0000FFFF0000FFFFULL);
	const __m512i m8  = _mm512_set1_epi64((int64_t)0x00FF00FF00FF00FFULL);
	const __m512i m4  = _mm512_set1_epi64((int64_t)0x0F0F0F0F0F0F0F0FULL);
	const __m512i m2  = _mm512_set1_epi64((int64_t)0x3333333333333333ULL);
	const __m512i m1  = _mm512_set1_epi64((int64_t)0x5555555555555555ULL);

	x = _mm512_and_si512(_mm512_or_si512(x, _mm512_slli_epi64(x, 16)), m16);
	x = _mm512_and_si512(_mm512_or_si512(x, _mm512_slli_epi64(x,  8)), m8);
	x = _mm512_and_si512(_mm512_or_si512(x, _mm512_slli_epi64(x,  4)), m4);
	x = _mm512_and_si512(_mm512_or_si512(x, _mm512_slli_epi64(x,  2)), m2);
	x = _mm512_and_si512(_mm512_or_si512(x, _mm512_slli_epi64(x,  1)), m1);
	return x;
}

/* ---------------------------------------------------------------------------
 * Vectorized GF(2^64) reduction, operating on a ZMM of 8 (lo, hi) pairs.
 *
 * Mirrors the scalar reduction in gf64_solve.c:14-36 lifted into ZMM
 * registers (same logic as gf64_reduce_mul_512 in gf64_mul_avx512.c:96-171,
 * but applied here to the squaring's (lo, hi) pair rather than to a clmul
 * product).
 *
 * Bit-exact to the scalar reduction; verified by test/test_gf64_square.c
 * against gf64_mul_reference.
 * --------------------------------------------------------------------------- */
__attribute__((target("avx512f")))
static inline __m512i gf64_reduce_square_512(__m512i lo_v, __m512i hi_v) {
	const __m512i one = _mm512_set1_epi64(1);

	/* t_lo = (hi<<4) ^ (hi<<3) ^ (hi<<1) ^ hi  -- low 64 bits of hi*0x1B */
	__m512i t_lo = _mm512_xor_si512(
		_mm512_xor_si512(_mm512_slli_epi64(hi_v, 4), _mm512_slli_epi64(hi_v, 3)),
		_mm512_xor_si512(_mm512_slli_epi64(hi_v, 1), hi_v)
	);

	/* R_hi: bit-extracted overflow of hi * 0x1B (low 4 bits per lane). */
	__m512i bit0 = _mm512_and_si512(
		_mm512_xor_si512(
			_mm512_srli_epi64(hi_v, 60),
			_mm512_xor_si512(_mm512_srli_epi64(hi_v, 61), _mm512_srli_epi64(hi_v, 63))
		),
		one
	);
	__m512i bit1 = _mm512_slli_epi64(
		_mm512_and_si512(
			_mm512_xor_si512(_mm512_srli_epi64(hi_v, 61), _mm512_srli_epi64(hi_v, 62)),
			one
		),
		1
	);
	__m512i bit2 = _mm512_slli_epi64(
		_mm512_and_si512(
			_mm512_xor_si512(_mm512_srli_epi64(hi_v, 62), _mm512_srli_epi64(hi_v, 63)),
			one
		),
		2
	);
	__m512i bit3 = _mm512_slli_epi64(_mm512_and_si512(_mm512_srli_epi64(hi_v, 63), one), 3);
	__m512i R_hi = _mm512_or_si512(
		_mm512_or_si512(bit0, bit1),
		_mm512_or_si512(bit2, bit3)
	);

	/* t2 = (R_hi<<4) ^ (R_hi<<3) ^ (R_hi<<1) ^ R_hi  -- R_hi * 0x1B (no overflow) */
	__m512i t2 = _mm512_xor_si512(
		_mm512_xor_si512(_mm512_slli_epi64(R_hi, 4), _mm512_slli_epi64(R_hi, 3)),
		_mm512_xor_si512(_mm512_slli_epi64(R_hi, 1), R_hi)
	);

	/* result = lo ^ t_lo ^ t2 */
	return _mm512_xor_si512(_mm512_xor_si512(lo_v, t_lo), t2);
}

/* ---------------------------------------------------------------------------
 * AVX-512 element-by-element Frobenius squaring of a length-N vector.
 *
 * Per outer iteration: load 8 input elements, split each into a low /
 * high 32-bit half, SWAR-spread each half independently to produce the
 * (lo, hi) pair of the 128-bit squaring result, then reduce via
 * gf64_reduce_square_512. 8 elements per store.
 *
 * The N % 8 tail is handled by the scalar gf64_square_one (matches the
 * tail strategy of gf64_mul_avx512.c, gf64_invert_avx512.c, etc.).
 * --------------------------------------------------------------------------- */
__attribute__((target("avx512f")))
void gf64_square_avx512(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT in,
	size_t N
) {
	const size_t LANES = 8;
	const __m512i lo32_mask = _mm512_set1_epi64((int64_t)0xFFFFFFFFULL);
	const size_t blocks = N / LANES;
	size_t i = 0;

	for (size_t bl = 0; bl < blocks; bl++) {
		__m512i v    = _mm512_loadu_si512((const __m512i *)(in + i));
		__m512i a_lo = _mm512_and_si512(v, lo32_mask);
		__m512i a_hi = _mm512_srli_epi64(v, 32);

		__m512i lo_spread = gf64_swar_spread_512(a_lo);
		__m512i hi_spread = gf64_swar_spread_512(a_hi);

		__m512i result = gf64_reduce_square_512(lo_spread, hi_spread);
		_mm512_storeu_si512((__m512i *)(out + i), result);

		i += LANES;
	}

	/* Scalar tail epilog (0..7 elements). */
	for (; i < N; i++) {
		out[i] = gf64_square_one(in[i]);
	}
}

#endif /* GF64_SQUARE_TU_BODY */
HEDLEY_END_C_DECLS