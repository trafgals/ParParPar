/*
 * ============================================================================
 * gf64/gf64_mul_avx512.c — AVX-512 vectorized element-by-element GF(2^64) mul
 *
 * T1 of the par3-cauchy-fft-kernel plan (Barycentric path). Provides
 *   void gf64_mul_avx512(gf64_t* out, const gf64_t* a, const gf64_t* b, size_t N)
 * that computes out[i] = gf64_mul(a[i], b[i]) for i in [0, N).
 *
 * SCOPE
 *   This file is a NEW translation unit. It does NOT modify
 *   gf64/gf64_solve.c:7-37 (the existing scalar gf64_mul is preserved
 *   untouched for the Solve path). The reduction arithmetic (x^64 + x^4 +
 *   x^3 + x + 1, low 64 bits = 0x1B) is mirrored bit-exact from the scalar
 *   reference, so the two paths are interchangeable.
 *
 * ISA / DISPATCH
 *   Compiled with __attribute__((target("avx512f,vpclmulqdq"))). Caller
 *   (GF64Method dispatch in gf64/gf64_dispatch.c) is responsible for ensuring
 *   the host supports these features before binding this function — see
 *   gf64/cpu_detect.c (CPUID + XCR0 + SIGILL probe, plus PAR3_GF64_USE_AVX512
 *   operator escape hatch). On non-AVX-512 hosts, the Barycentric path falls
 *   back to the scalar gf64_mul; this function is never reached there.
 *
 * SIMD SHAPE (per outer iteration)
 *   Two VPCLMULQDQ calls process 8 input pairs. Each call takes 4 element
 *   pairs (a[i+k], b[i+k]) for k=0..3 from independent 128-bit lanes of two
 *   ZMM registers, produces four 128-bit carry-less products, and we then
 *   reduce each product to a single 64-bit GF(2^64) element via
 *   gf64_reduce_512 — a vectorized rewrite of the scalar reduction in
 *   gf64_solve.c:14-36 (mirrors gf64_reduce_512 in gf64_region_avx512_arr.c
 *   but operates on a 4-lane half-pack rather than a 4-product ZMM).
 *
 *   Lane layout inside the VPCLMULQDQ result:
 *     128-bit lane k = [ hi_k | lo_k ] = 64x64 -> 128 clmul product
 *   After reduction each lane contributes one 64-bit GF element.
 *
 * REDUCTION (mirrors gf64_solve.c:14-36)
 *   For each lane: result = lo ^ (hi << 4) ^ (hi << 3) ^ (hi << 1) ^ hi
 *                          ^ (R_hi << 4) ^ (R_hi << 3) ^ (R_hi << 1) ^ R_hi
 *   where R_hi is the 4-bit overflow of (hi * 0x1B) into positions [64:67]:
 *     R_hi[0] = hi[60] ^ hi[61] ^ hi[63]
 *     R_hi[1] = hi[61] ^ hi[62]
 *     R_hi[2] = hi[62] ^ hi[63]
 *     R_hi[3] = hi[63]
 *   R_hi < 16, so R_hi * 0x1B fits safely in uint64_t (no further overflow).
 *
 * TAIL
 *   The body processes N/8 full blocks of 8. Any remainder (0..7 elements)
 *   is handled by a scalar epilog calling gf64_mul_reference (the public
 *   scalar SSE2 path defined in gf64/gf64_single.c) — matches the tail
 *   strategy of the existing gf64_region_*_arr.c kernels.
 * ============================================================================
 */

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

#define GF64_MUL_AVX512_TU_BODY 1
#if GF64_MUL_AVX512_TU_BODY

/* Public scalar SSE2 reduction path, used for the N % 8 tail.
 * Defined in gf64/gf64_single.c; bit-exact to gf64_solve.c:7-37. */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* ---------------------------------------------------------------------------
 * Vectorized GF(2^64) reduction, operating on a half-pack of 4 active lanes
 * (lanes 0..3 carry (lo_i, hi_i) for i=0..3; lanes 4..7 are ignored).
 *
 * Mirrors gf64_reduce_128 from gf64_solve.c:7-37, lifted into ZMM registers:
 *
 *   t_lo = (hi<<4) ^ (hi<<3) ^ (hi<<1) ^ hi      // lo(hi * 0x1B)
 *   R_hi = bit-extracted 4-bit overflow of hi*0x1B
 *   t2   = (R_hi<<4) ^ (R_hi<<3) ^ (R_hi<<1) ^ R_hi   // R_hi * 0x1B (no overflow)
 *   return lo ^ t_lo ^ t2
 *
 * The bit-extraction of R_hi uses individual bit masks + shifts rather than
 * a single 64-bit shift/mask. Reason: we want the per-lane result to live in
 * the LOW 4 bits of each 64-bit lane (so subsequent (R_hi<<k) shifts
 * position each bit to its final location), which keeps the reduction a
 * pure AVX-512F (no AVX-512DQ/VBMI dependency) — same approach as
 * gf64_reduce_512 in gf64_region_avx512_arr.c.
 *
 * Bit-exact to gf64_mul_reference, verified by test/test_gf64_mul_avx512_parity.c.
 * --------------------------------------------------------------------------- */
__attribute__((target("avx512f")))
static inline __m512i gf64_reduce_mul_512(__m512i lo_v, __m512i hi_v) {
	const __m512i one = _mm512_set1_epi64(1);

	/* t_lo = (hi<<4) ^ (hi<<3) ^ (hi<<1) ^ hi */
	__m512i t_lo = _mm512_xor_si512(
		_mm512_xor_si512(
			_mm512_slli_epi64(hi_v, 4),
			_mm512_slli_epi64(hi_v, 3)
		),
		_mm512_xor_si512(
			_mm512_slli_epi64(hi_v, 1),
			hi_v
		)
	);

	/* R_hi: 4-bit overflow of hi * 0x1B per lane.
	 *   bit0 = (hi >> 60) ^ (hi >> 61) ^ (hi >> 63)
	 *   bit1 = (hi >> 61) ^ (hi >> 62)
	 *   bit2 = (hi >> 62) ^ (hi >> 63)
	 *   bit3 = (hi >> 63)
	 * Each bit is computed masked to 1 bit, then shifted to its position
	 * in R_hi (low 4 bits of each 64-bit lane). */
	__m512i bit0 = _mm512_and_si512(
		_mm512_xor_si512(
			_mm512_srli_epi64(hi_v, 60),
			_mm512_xor_si512(
				_mm512_srli_epi64(hi_v, 61),
				_mm512_srli_epi64(hi_v, 63)
			)
		),
		one
	);
	__m512i bit1 = _mm512_slli_epi64(
		_mm512_and_si512(
			_mm512_xor_si512(
				_mm512_srli_epi64(hi_v, 61),
				_mm512_srli_epi64(hi_v, 62)
			),
			one
		),
		1
	);
	__m512i bit2 = _mm512_slli_epi64(
		_mm512_and_si512(
			_mm512_xor_si512(
				_mm512_srli_epi64(hi_v, 62),
				_mm512_srli_epi64(hi_v, 63)
			),
			one
		),
		2
	);
	__m512i bit3 = _mm512_slli_epi64(
		_mm512_and_si512(_mm512_srli_epi64(hi_v, 63), one),
		3
	);
	__m512i R_hi = _mm512_or_si512(
		_mm512_or_si512(bit0, bit1),
		_mm512_or_si512(bit2, bit3)
	);

	/* t2 = (R_hi<<4) ^ (R_hi<<3) ^ (R_hi<<1) ^ R_hi  (R_hi * 0x1B, fits in uint64) */
	__m512i t2 = _mm512_xor_si512(
		_mm512_xor_si512(
			_mm512_slli_epi64(R_hi, 4),
			_mm512_slli_epi64(R_hi, 3)
		),
		_mm512_xor_si512(
			_mm512_slli_epi64(R_hi, 1),
			R_hi
		)
	);

	/* result = lo ^ t_lo ^ t2 */
	return _mm512_xor_si512(_mm512_xor_si512(lo_v, t_lo), t2);
}

/* ---------------------------------------------------------------------------
 * VPCLMULQDQ lane-k helper. For k in {0..3}, each 128-bit lane of the two
 * ZMM inputs is laid out as [hi_qword | lo_qword] = [garbage | element].
 *
 * With imm8=0x00 (select low qword of each operand) and inputs constructed
 * via _mm512_set_epi64(0, ..., 0, element_k), each lane performs
 * clmul(element_k, element_k_of_b). We use imm8=0x11 (select high qword of
 * each operand) when constructing via the inverse layout.
 *
 * To keep the data-movement story simple and obvious (and to match the
 * construction style of gf64_region_avx512_arr.c), we pack 4 element pairs
 * per clmul call with the lower qword of each 128-bit lane carrying the
 * element. The two halves of an 8-element outer iteration are issued as
 * two separate clmul calls; the half-packs are then reduced in lockstep.
 * --------------------------------------------------------------------------- */
__attribute__((target("avx512f,vpclmulqdq")))
void gf64_mul_avx512(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT a,
	const gf64_t *HEDLEY_RESTRICT b,
	size_t N
) {
	size_t i = 0;
	const size_t LANES_PER_BLOCK = 8;
	const size_t blocks = N / LANES_PER_BLOCK;

	for (size_t bl = 0; bl < blocks; bl++) {
		/* Build two ZMMs, each holding 4 (zero, element_k) 128-bit lane pairs.
		 *
		 * _mm512_set_epi64 args map highest-lane-first: arg #0 → lane 7,
		 * arg #7 → lane 0. The 128-bit VPCLMULQDQ lane layout is therefore:
		 *   128-bit lane 0 = [ arg #1 | arg #0 ]   // [ high qword | low qword ]
		 *   128-bit lane 1 = [ arg #3 | arg #2 ]
		 *   128-bit lane 2 = [ arg #5 | arg #4 ]
		 *   128-bit lane 3 = [ arg #7 | arg #6 ]
		 *
		 * To place elements k=0..3 in the low qwords of lanes 0..3 (with the
		 * high qword zeroed — VPCLMULQDQ imm8=0x00 selects the low qword), we
		 * interleave (zero, element) pairs: arg #0 = element 0, arg #1 = 0,
		 * arg #2 = element 1, arg #3 = 0, etc. This mirrors the construction
		 * style used in gf64_region_avx512_arr.c (which broadcasts a single
		 * coefficient across all lanes for the mul_arr fast path).
		 */
		__m512i va_lo = _mm512_set_epi64(
			(int64_t)0,                    /* lane 7 (high qword of 128-bit lane 3) */
			(int64_t)a[i + 3],             /* lane 6 (low  qword of 128-bit lane 3) */
			(int64_t)0,                    /* lane 5 (high qword of 128-bit lane 2) */
			(int64_t)a[i + 2],             /* lane 4 (low  qword of 128-bit lane 2) */
			(int64_t)0,                    /* lane 3 (high qword of 128-bit lane 1) */
			(int64_t)a[i + 1],             /* lane 2 (low  qword of 128-bit lane 1) */
			(int64_t)0,                    /* lane 1 (high qword of 128-bit lane 0) */
			(int64_t)a[i + 0]              /* lane 0 (low  qword of 128-bit lane 0) */
		);
		__m512i vb_lo = _mm512_set_epi64(
			(int64_t)0,                    /* lane 7 */
			(int64_t)b[i + 3],             /* lane 6 */
			(int64_t)0,                    /* lane 5 */
			(int64_t)b[i + 2],             /* lane 4 */
			(int64_t)0,                    /* lane 3 */
			(int64_t)b[i + 1],             /* lane 2 */
			(int64_t)0,                    /* lane 1 */
			(int64_t)b[i + 0]              /* lane 0 */
		);
		__m512i va_hi = _mm512_set_epi64(
			(int64_t)0,
			(int64_t)a[i + 7],
			(int64_t)0,
			(int64_t)a[i + 6],
			(int64_t)0,
			(int64_t)a[i + 5],
			(int64_t)0,
			(int64_t)a[i + 4]
		);
		__m512i vb_hi = _mm512_set_epi64(
			(int64_t)0,
			(int64_t)b[i + 7],
			(int64_t)0,
			(int64_t)b[i + 6],
			(int64_t)0,
			(int64_t)b[i + 5],
			(int64_t)0,
			(int64_t)b[i + 4]
		);

		/* Two VPCLMULQDQ calls — each emits 4 128-bit products. */
		__m512i prod_lo = _mm512_clmulepi64_epi128(va_lo, vb_lo, 0x00);
		__m512i prod_hi = _mm512_clmulepi64_epi128(va_hi, vb_hi, 0x00);

		/* Each 128-bit product lane is [ hi_64 | lo_64 ]. Split into two
		 * ZMMs of 4 active lo's / 4 active hi's. */
		__m512i lo_v_lo = _mm512_permutex2var_epi64(
			prod_lo,
			_mm512_setr_epi64(0, 2, 4, 6, 0, 0, 0, 0),
			_mm512_setzero_si512()
		);
		__m512i hi_v_lo = _mm512_permutex2var_epi64(
			prod_lo,
			_mm512_setr_epi64(1, 3, 5, 7, 0, 0, 0, 0),
			_mm512_setzero_si512()
		);
		__m512i lo_v_hi = _mm512_permutex2var_epi64(
			prod_hi,
			_mm512_setr_epi64(0, 2, 4, 6, 0, 0, 0, 0),
			_mm512_setzero_si512()
		);
		__m512i hi_v_hi = _mm512_permutex2var_epi64(
			prod_hi,
			_mm512_setr_epi64(1, 3, 5, 7, 0, 0, 0, 0),
			_mm512_setzero_si512()
		);

		/* Reduce each half-pack independently. */
		__m512i red_lo = gf64_reduce_mul_512(lo_v_lo, hi_v_lo);
		__m512i red_hi = gf64_reduce_mul_512(lo_v_hi, hi_v_hi);

		/* Pack the two 4-element reductions back into a single 8-element
		 * ZMM for a single unaligned store. */
		__m512i packed = _mm512_permutex2var_epi64(
			red_lo,
			_mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11),
			red_hi
		);

		_mm512_storeu_si512((__m512i *)(out + i), packed);

		i += LANES_PER_BLOCK;
	}

	/* Scalar tail epilog (0..7 elements). Mirrors the tail strategy used
	 * by every other SIMD region kernel in this codebase. */
	for (; i < N; i++) {
		out[i] = gf64_mul_reference(a[i], b[i]);
	}
}

#endif /* GF64_MUL_AVX512_TU_BODY */
HEDLEY_END_C_DECLS