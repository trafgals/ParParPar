/*
 * ============================================================================
 * gf64/gf64_square_avx512.c — GF(2^64) Frobenius squaring, AVX-512 kernel
 *
 * Split from gf64_square.c (issue #62): the AVX-512 functions live in their
 * own translation unit so the scalar TU (gf64_square.c) can be compiled
 * WITHOUT /arch:AVX512. The MSVC /arch flag is whole-TU — compiling the
 * scalar squaring with /arch:AVX512 let the auto-vectorizer emit EVEX
 * (ZMM) instructions into the scalar path, which SIGILLs on hosts without
 * AVX-512 (windows-2025 fleet AMD Milan runners) even when the dispatcher
 * correctly selected the AVX-2 method.
 *
 * This TU is compiled with /arch:AVX512 (MSVC) or per-function
 * __attribute__((target("avx512f"))) (GCC/Clang) and is the ONLY home of
 * the squaring kernel's _mm512 intrinsics.
 *
 * Math: identical to gf64_square.c — bit-exact to gf64_mul_reference(a, a).
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
#define GF64_SQUARE_AVX512_TU_BODY 1
#else
#define GF64_SQUARE_AVX512_TU_BODY 0
#endif
#if GF64_SQUARE_AVX512_TU_BODY

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
 * The N % 8 tail is handled by the scalar gf64_square (matches the
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

	/* Scalar tail epilog (0..7 elements) — bit-exact via the scalar TU.
	 * Guarded: with N == 0 both pointers may be NULL and out + 0 is
	 * undefined pointer arithmetic. */
	if (i < N) {
		gf64_square(out + i, in + i, N - i);
	}
}

#endif /* GF64_SQUARE_AVX512_TU_BODY */
HEDLEY_END_C_DECLS
