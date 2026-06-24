#include "gf64_global.h"
#include <immintrin.h>
#include <wmmintrin.h>
#include <stdint.h>
#include <stddef.h>

HEDLEY_BEGIN_C_DECLS

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* Reduce a 128-bit carry-less product (lo:hi) to a single 64-bit GF(2^64) element.
 * Mirrors the reducer used in the other region files; inlined to allow
 * the per-lane scalar reduction after each VPCLMULQDQ call.
 */
static inline uint64_t gf64_reduce_128(uint64_t lo, uint64_t hi) {
	/* Lower 64 bits of hi * 0x1B (truncated at 64 bits by uint64_t). */
	uint64_t t_lo = (hi << 4) ^ (hi << 3) ^ (hi << 1) ^ hi;

	/* Overflow bits (64-67) of hi * 0x1B:
	 * (hi<<4) overflow: hi[60:63] -> full_product[64:67]
	 * (hi<<3) overflow: hi[61:63] -> full_product[64:66]
	 * (hi<<1) overflow: hi[63]   -> full_product[64]
	 * R_hi[0] = full_product bit 64 = hi[60] ^ hi[61] ^ hi[63]
	 * R_hi[1] = full_product bit 65 = hi[61] ^ hi[62]
	 * R_hi[2] = full_product bit 66 = hi[62] ^ hi[63]
	 * R_hi[3] = full_product bit 67 = hi[63]
	 */
	uint64_t R_hi =
		(((hi >> 60) ^ (hi >> 61) ^ (hi >> 63)) & 1) |
		((((hi >> 61) ^ (hi >> 62)) & 1) << 1) |
		((((hi >> 62) ^ (hi >> 63)) & 1) << 2) |
		(((hi >> 63) & 1) << 3);

	/* Reduce R_hi: x^64 = 0x1B, so R_hi * x^64 = R_hi * 0x1B.
	 * R_hi < 16, so R_hi * 0x1B fits safely in uint64_t. */
	uint64_t t2 = (R_hi << 4) ^ (R_hi << 3) ^ (R_hi << 1) ^ R_hi;

	return lo ^ t_lo ^ t2;
}

/* VEX.256 VPCLMULQDQ notes (verified against Intel docs, felixcloutier.com/x86/pclmulqdq):
 *   - Takes 256-bit (YMM) operands; operates independently on each 128-bit lane.
 *   - Each lane performs one 64x64->128 carry-less multiply.
 *   - Output is a 256-bit YMM holding 2 x 128-bit products (NOT 4 x 128-bit;
 *     4 x 128-bit output requires VPCLMULQDQ AVX-512F with ZMM registers).
 *   - imm8 bits: [4] selects qword of src2, [0] selects qword of src1, per lane.
 *     imm8=0x00 -> low qword of each lane in both operands (what we want: GF
 *     elements live in the low qword of each lane, coefficients broadcast the same).
 *
 * Layout of a 256-bit vector `in_vec` constructed via
 *   _mm256_setr_epi64x(in[i+0], 0, in[i+1], 0):
 *   lane 0 (bits[127:0])   = [ in[i+0] | 0 ]
 *   lane 1 (bits[255:128]) = [ in[i+1] | 0 ]
 *   With coeff_broadcast = [c, c, c, c] and imm8=0x00:
 *     result lane 0 = clmul(in[i+0], c)  (128-bit)
 *     result lane 1 = clmul(in[i+1], c)  (128-bit)
 *   Two GF element products per VPCLMULQDQ call.
 *
 * Each 128-bit product's lo/hi halves are then reduced to 64 bits via the
 * vectorized gf64_reduce_ymm helper (see below), which keeps both halves in
 * YMM registers throughout and produces 2 reduced 64-bit values per call.
 */

/* Vectorized GF(2^64) reduction for a YMM holding 2 carryless products.
 *
 * Inputs:
 *   lo_vec YMM: lane 0 = lo0, lane 1 = lo1 (lanes 2-3 must be zero or
 *               carry the same values; they are not consumed).
 *   hi_vec YMM: lane 0 = hi0, lane 1 = hi1 (same lane convention).
 *
 * Output YMM: lane 0 = gf64_reduce_128(lo0, hi0),
 *             lane 1 = gf64_reduce_128(lo1, hi1).
 *             (Lanes 2-3 also hold valid reduced values, but callers
 *              consume lanes 0-1 via _mm256_castsi256_si128.)
 *
 * Math is bit-exact identical to the scalar gf64_reduce_128 above:
 *   t_lo = (hi<<4) ^ (hi<<3) ^ (hi<<1) ^ hi
 *   R_hi bits extracted from hi[60..63] (see scalar version for derivation)
 *   t2 = (R_hi<<4) ^ (R_hi<<3) ^ (R_hi<<1) ^ R_hi
 *   result = lo ^ t_lo ^ t2
 */
static inline __m256i gf64_reduce_ymm(__m256i lo_vec, __m256i hi_vec) {
	/* t_lo = (hi<<4) ^ (hi<<3) ^ (hi<<1) ^ hi */
	__m256i t_lo = _mm256_xor_si256(
		_mm256_xor_si256(
			_mm256_slli_epi64(hi_vec, 4),
			_mm256_slli_epi64(hi_vec, 3)
		),
		_mm256_xor_si256(
			_mm256_slli_epi64(hi_vec, 1),
			hi_vec
		)
	);

	/* R_hi bit 0 = hi[60] ^ hi[61] ^ hi[63] */
	__m256i bit0 = _mm256_and_si256(
		_mm256_xor_si256(
			_mm256_srli_epi64(hi_vec, 60),
			_mm256_xor_si256(
				_mm256_srli_epi64(hi_vec, 61),
				_mm256_srli_epi64(hi_vec, 63)
			)
		),
		_mm256_set1_epi64x(1)
	);
	/* R_hi bit 1 = hi[61] ^ hi[62] */
	__m256i bit1 = _mm256_slli_epi64(
		_mm256_and_si256(
			_mm256_xor_si256(
				_mm256_srli_epi64(hi_vec, 61),
				_mm256_srli_epi64(hi_vec, 62)
			),
			_mm256_set1_epi64x(1)
		),
		1
	);
	/* R_hi bit 2 = hi[62] ^ hi[63] */
	__m256i bit2 = _mm256_slli_epi64(
		_mm256_and_si256(
			_mm256_xor_si256(
				_mm256_srli_epi64(hi_vec, 62),
				_mm256_srli_epi64(hi_vec, 63)
			),
			_mm256_set1_epi64x(1)
		),
		2
	);
	/* R_hi bit 3 = hi[63] */
	__m256i bit3 = _mm256_slli_epi64(
		_mm256_and_si256(_mm256_srli_epi64(hi_vec, 63), _mm256_set1_epi64x(1)),
		3
	);
	/* R_hi = bit0 | bit1 | bit2 | bit3 (4-bit value, fits in lane low nibble) */
	__m256i R_hi = _mm256_or_si256(
		_mm256_or_si256(bit0, bit1),
		_mm256_or_si256(bit2, bit3)
	);

	/* t2 = (R_hi<<4) ^ (R_hi<<3) ^ (R_hi<<1) ^ R_hi — same shape as t_lo */
	__m256i t2 = _mm256_xor_si256(
		_mm256_xor_si256(
			_mm256_slli_epi64(R_hi, 4),
			_mm256_slli_epi64(R_hi, 3)
		),
		_mm256_xor_si256(
			_mm256_slli_epi64(R_hi, 1),
			R_hi
		)
	);

	/* result = lo ^ t_lo ^ t2 (XOR is associative). */
	return _mm256_xor_si256(_mm256_xor_si256(lo_vec, t_lo), t2);
}

/* Permute a YMM `prod` (carrying 2 carryless products in 4 64-bit lanes
 * [lo0, hi0, lo1, hi1]) into two YMMs grouping like halves:
 *   lo_vec = [lo0, lo1, 0, 0]
 *   hi_vec = [hi0, hi1, 0, 0]
 * Implemented via 128-bit extraction + 64-bit unpack + 256-bit cast.
 * (Note: _mm256_permute4x64_epi64 cannot select from source lane 3 — the
 * imm8==3 case returns 0 per the Intel SDM, so we cannot use it here to
 * gather hi1 = prod[255:192] into output lane 1.)
 */
static inline void gf64_split_prod_ymm(__m256i prod, __m256i *lo_vec, __m256i *hi_vec) {
	__m128i prod_lo128 = _mm256_castsi256_si128(prod);    /* [lo0, hi0] */
	__m128i prod_hi128 = _mm256_extracti128_si256(prod, 1); /* [lo1, hi1] */
	__m128i lo_pair = _mm_unpacklo_epi64(prod_lo128, prod_hi128); /* [lo0, lo1] */
	__m128i hi_pair = _mm_unpackhi_epi64(prod_lo128, prod_hi128); /* [hi0, hi1] */
	*lo_vec = _mm256_castsi128_si256(lo_pair); /* [lo0, lo1, 0, 0] */
	*hi_vec = _mm256_castsi128_si256(hi_pair); /* [hi0, hi1, 0, 0] */
}

__attribute__((target("avx2,vpclmulqdq")))
void gf64_region_mul_avx2_arr(gf64_t *HEDLEY_RESTRICT out, const gf64_t *HEDLEY_RESTRICT in, const gf64_t *HEDLEY_RESTRICT coeff, size_t len, size_t n_coeff) {
	size_t i = 0;

	if (n_coeff == 1) {
		/* Fast path: single coefficient broadcast across all lanes.
		 * Broadcast once outside the loop, then do 2 VPCLMULQDQ per outer
		 * iteration (4 GF elements processed, 2 elements per call).
		 * Each VPCLMULQDQ produces a YMM with 2 carryless products;
		 * gf64_reduce_ymm reduces both to 64 bits in one SIMD pass without
		 * the per-lane scalar extract round trip.
		 */
		uint64_t c0 = coeff[0];
		__m256i coeff_broadcast = _mm256_set1_epi64x((int64_t)c0);

		size_t blocks = len / 4;
		for (size_t b = 0; b < blocks; b++) {
			/* Pair 1: out[i+0] = clmul(in[i+0], c0), out[i+1] = clmul(in[i+1], c0) */
			__m256i in01 = _mm256_setr_epi64x((int64_t)in[i + 0], 0, (int64_t)in[i + 1], 0);
			__m256i prod01 = _mm256_clmulepi64_epi128(in01, coeff_broadcast, 0x00);
			__m256i lo_vec;
			__m256i hi_vec;
			gf64_split_prod_ymm(prod01, &lo_vec, &hi_vec);
			__m256i result01 = gf64_reduce_ymm(lo_vec, hi_vec);
			_mm_storeu_si128((__m128i *)(out + i + 0), _mm256_castsi256_si128(result01));

			/* Pair 2: out[i+2] = clmul(in[i+2], c0), out[i+3] = clmul(in[i+3], c0) */
			__m256i in23 = _mm256_setr_epi64x((int64_t)in[i + 2], 0, (int64_t)in[i + 3], 0);
			__m256i prod23 = _mm256_clmulepi64_epi128(in23, coeff_broadcast, 0x00);
			gf64_split_prod_ymm(prod23, &lo_vec, &hi_vec);
			__m256i result23 = gf64_reduce_ymm(lo_vec, hi_vec);
			_mm_storeu_si128((__m128i *)(out + i + 2), _mm256_castsi256_si128(result23));

			i += 4;
		}

		/* Tail (0..3 elements) — scalar epilog. */
		while (i < len) {
			out[i] = gf64_mul_reference(in[i], c0);
			i++;
		}
	} else {
		/* General case: each input element is multiplied by ALL coefficients
		 * and the products are XORed together (dot product per element).
		 * Process 2 elements per VPCLMULQDQ call (one per 128-bit lane).
		 * Accumulate lo and hi halves in YMM registers across the inner
		 * loop, then apply the vectorized reduction once at the end.
		 * Two VPCLMULQDQ calls per 4-element block.
		 */
		size_t blocks = len / 4;
		for (size_t b = 0; b < blocks; b++) {
			/* Pair 1: out[i+0], out[i+1] */
			__m256i in01 = _mm256_setr_epi64x((int64_t)in[i + 0], 0, (int64_t)in[i + 1], 0);
			__m256i acc_lo_01 = _mm256_setzero_si256();
			__m256i acc_hi_01 = _mm256_setzero_si256();

			for (size_t c = 0; c < n_coeff; c++) {
				__m256i coeff_bc = _mm256_set1_epi64x((int64_t)coeff[c]);
				__m256i prod = _mm256_clmulepi64_epi128(in01, coeff_bc, 0x00);
				__m256i lo_vec;
				__m256i hi_vec;
				gf64_split_prod_ymm(prod, &lo_vec, &hi_vec);
				acc_lo_01 = _mm256_xor_si256(acc_lo_01, lo_vec);
				acc_hi_01 = _mm256_xor_si256(acc_hi_01, hi_vec);
			}

			__m256i result01 = gf64_reduce_ymm(acc_lo_01, acc_hi_01);
			_mm_storeu_si128((__m128i *)(out + i + 0), _mm256_castsi256_si128(result01));

			/* Pair 2: out[i+2], out[i+3] */
			__m256i in23 = _mm256_setr_epi64x((int64_t)in[i + 2], 0, (int64_t)in[i + 3], 0);
			__m256i acc_lo_23 = _mm256_setzero_si256();
			__m256i acc_hi_23 = _mm256_setzero_si256();

			for (size_t c = 0; c < n_coeff; c++) {
				__m256i coeff_bc = _mm256_set1_epi64x((int64_t)coeff[c]);
				__m256i prod = _mm256_clmulepi64_epi128(in23, coeff_bc, 0x00);
				__m256i lo_vec;
				__m256i hi_vec;
				gf64_split_prod_ymm(prod, &lo_vec, &hi_vec);
				acc_lo_23 = _mm256_xor_si256(acc_lo_23, lo_vec);
				acc_hi_23 = _mm256_xor_si256(acc_hi_23, hi_vec);
			}

			__m256i result23 = gf64_reduce_ymm(acc_lo_23, acc_hi_23);
			_mm_storeu_si128((__m128i *)(out + i + 2), _mm256_castsi256_si128(result23));

			i += 4;
		}

		/* Tail (0..3 elements) — scalar epilog with SUM semantics. */
		while (i < len) {
			uint64_t sum = 0;
			for (size_t c = 0; c < n_coeff; c++) {
				sum ^= gf64_mul_reference(in[i], coeff[c]);
			}
			out[i] = sum;
			i++;
		}
	}
}

__attribute__((target("avx2,vpclmulqdq")))
void gf64_region_muladd_avx2_arr(gf64_t *HEDLEY_RESTRICT out, const gf64_t *HEDLEY_RESTRICT in, const gf64_t *HEDLEY_RESTRICT coeff, size_t len, size_t n_coeff) {
	size_t i = 0;

	if (n_coeff == 1) {
		/* Fast path: same vectorized reduction as gf64_region_mul_avx2_arr,
		 * but XOR the 2 reduced values into the existing out[i+0..i+1]
		 * (and out[i+2..i+3]) instead of overwriting them.
		 */
		uint64_t c0 = coeff[0];
		__m256i coeff_broadcast = _mm256_set1_epi64x((int64_t)c0);

		size_t blocks = len / 4;
		for (size_t b = 0; b < blocks; b++) {
			__m256i in01 = _mm256_setr_epi64x((int64_t)in[i + 0], 0, (int64_t)in[i + 1], 0);
			__m256i prod01 = _mm256_clmulepi64_epi128(in01, coeff_broadcast, 0x00);
			__m256i lo_vec;
			__m256i hi_vec;
			gf64_split_prod_ymm(prod01, &lo_vec, &hi_vec);
			__m256i result01 = gf64_reduce_ymm(lo_vec, hi_vec);
			__m128i prev01 = _mm_loadu_si128((const __m128i *)(out + i + 0));
			_mm_storeu_si128((__m128i *)(out + i + 0),
			                 _mm_xor_si128(prev01, _mm256_castsi256_si128(result01)));

			__m256i in23 = _mm256_setr_epi64x((int64_t)in[i + 2], 0, (int64_t)in[i + 3], 0);
			__m256i prod23 = _mm256_clmulepi64_epi128(in23, coeff_broadcast, 0x00);
			gf64_split_prod_ymm(prod23, &lo_vec, &hi_vec);
			__m256i result23 = gf64_reduce_ymm(lo_vec, hi_vec);
			__m128i prev23 = _mm_loadu_si128((const __m128i *)(out + i + 2));
			_mm_storeu_si128((__m128i *)(out + i + 2),
			                 _mm_xor_si128(prev23, _mm256_castsi256_si128(result23)));

			i += 4;
		}

		while (i < len) {
			out[i] ^= gf64_mul_reference(in[i], c0);
			i++;
		}
	} else {
		/* General muladd: accumulate lo/hi per element across all coefficients
		 * in YMM registers, vectorized-reduce once at end, then XOR into out.
		 */
		size_t blocks = len / 4;
		for (size_t b = 0; b < blocks; b++) {
			__m256i in01 = _mm256_setr_epi64x((int64_t)in[i + 0], 0, (int64_t)in[i + 1], 0);
			__m256i acc_lo_01 = _mm256_setzero_si256();
			__m256i acc_hi_01 = _mm256_setzero_si256();

			for (size_t c = 0; c < n_coeff; c++) {
				__m256i coeff_bc = _mm256_set1_epi64x((int64_t)coeff[c]);
				__m256i prod = _mm256_clmulepi64_epi128(in01, coeff_bc, 0x00);
				__m256i lo_vec;
				__m256i hi_vec;
				gf64_split_prod_ymm(prod, &lo_vec, &hi_vec);
				acc_lo_01 = _mm256_xor_si256(acc_lo_01, lo_vec);
				acc_hi_01 = _mm256_xor_si256(acc_hi_01, hi_vec);
			}

			__m256i result01 = gf64_reduce_ymm(acc_lo_01, acc_hi_01);
			__m128i prev01 = _mm_loadu_si128((const __m128i *)(out + i + 0));
			_mm_storeu_si128((__m128i *)(out + i + 0),
			                 _mm_xor_si128(prev01, _mm256_castsi256_si128(result01)));

			__m256i in23 = _mm256_setr_epi64x((int64_t)in[i + 2], 0, (int64_t)in[i + 3], 0);
			__m256i acc_lo_23 = _mm256_setzero_si256();
			__m256i acc_hi_23 = _mm256_setzero_si256();

			for (size_t c = 0; c < n_coeff; c++) {
				__m256i coeff_bc = _mm256_set1_epi64x((int64_t)coeff[c]);
				__m256i prod = _mm256_clmulepi64_epi128(in23, coeff_bc, 0x00);
				__m256i lo_vec;
				__m256i hi_vec;
				gf64_split_prod_ymm(prod, &lo_vec, &hi_vec);
				acc_lo_23 = _mm256_xor_si256(acc_lo_23, lo_vec);
				acc_hi_23 = _mm256_xor_si256(acc_hi_23, hi_vec);
			}

			__m256i result23 = gf64_reduce_ymm(acc_lo_23, acc_hi_23);
			__m128i prev23 = _mm_loadu_si128((const __m128i *)(out + i + 2));
			_mm_storeu_si128((__m128i *)(out + i + 2),
			                 _mm_xor_si128(prev23, _mm256_castsi256_si128(result23)));

			i += 4;
		}

		while (i < len) {
			uint64_t sum = 0;
			for (size_t c = 0; c < n_coeff; c++) {
				sum ^= gf64_mul_reference(in[i], coeff[c]);
			}
			out[i] ^= sum;
			i++;
		}
	}
}

HEDLEY_END_C_DECLS
