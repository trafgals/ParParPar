#include "gf64_global.h"
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

HEDLEY_BEGIN_C_DECLS

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* Reduce a 128-bit carry-less product (lo:hi) to a single 64-bit GF(2^64) element.
 * Mirrors the reducer used in the other region files; kept as a static reference
 * for verifying that the vectorized reduction (gf64_reduce_512 below) is bit-exact.
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

/* ---------------------------------------------------------------------------
 * Vectorized GF(2^64) reduction (AVX-512F + VPCLMULQDQ only).
 *
 * Baseline (per the comment block in the previous version):
 *   - VPCLMULQDQ on a ZMM pair produces 4 128-bit products, packed as
 *     [lo0 | hi0 | lo1 | hi1 | lo2 | hi2 | lo3 | hi3] (8 64-bit lanes).
 *   - The previous implementation extracted each 128-bit lane to a __m128i via
 *     _mm512_extracti32x4_epi32, then split lo/hi with _mm_cvtsi128_si64 and
 *     _mm_srli_si128(_, 8), and finally called gf64_reduce_128 scalar 4 times.
 *   - That loop round-trips data through the GPRs, defeating the SIMD pipeline.
 *
 * New approach (this file):
 *   1. After each VPCLMULQDQ, permute the 128-bit product ZMM into two ZMMs
 *      with lanes [lo0, lo1, lo2, lo3, 0, 0, 0, 0] and
 *                  [hi0, hi1, hi2, hi3, 0, 0, 0, 0]
 *      using _mm512_permutex2var_epi64 with a zero source for the high lanes.
 *   2. Compute the reduction entirely in ZMM with AVX-512F integer ops:
 *        t_lo = (hi<<4) ^ (hi<<3) ^ (hi<<1) ^ hi           // lo(hi*0x1B)
 *        R_hi = bits 0..3 of hi*0x1B (per lane, 4 bits)    // overflow
 *        t2   = R_hi * 0x1B                                // fits in 8 bits
 *        result = lo ^ t_lo ^ t2
 *   3. Store the 4 reduced values to memory via _mm512_mask_storeu_epi64
 *      with mask 0x0F (writes lanes 0..3, leaves lanes 4..7 untouched in mem).
 *   4. For the n_coeff>1 path, XOR-fold the (lo_k, hi_k) pairs across all
 *      coefficients first, then apply the reduction once at the end. This
 *      uses the linearity of XOR-multiply: (a^b)*0x1B = (a*0x1B) ^ (b*0x1B),
 *      so the result is bit-exact to the per-coeff reduce-then-XOR path.
 *
 * Per-lane scalar reduce is preserved (gf64_reduce_128 above) as a static
 * reference; the build does not inline it from this TU.
 * ---------------------------------------------------------------------------
 */

/* Debug QA canary. When non-zero, the value is XORed into lane 0 of every
 * reduced result, forcing every parity check to fail. Used by the failure-QA
 * proof (see .omo/plans/par3-create-throughput-400mbps.md task T1, "MUST DO" #6).
 * Initialized to 0 -> no effect on production output. Volatile so the compiler
 * cannot fold it into a constant. */
static volatile uint64_t gf64_reduce_canary = 0;

/* Permute indices for splitting a 4-product ZMM (lanes 0..7 = [lo0,hi0,lo1,hi1,
 * lo2,hi2,lo3,hi3]) into two ZMMs of lo/hi halves each. The high 4 lanes of
 * the source are zeroed via the second operand of permutex2var. */
#define GF64_IDX_LO _mm512_setr_epi64(0, 2, 4, 6, 0, 0, 0, 0)
#define GF64_IDX_HI _mm512_setr_epi64(1, 3, 5, 7, 0, 0, 0, 0)

/* Split a 4-product ZMM (one VPCLMULQDQ result) into separate lo/hi ZMMs.
 * After the call, *lo_out and *hi_out have their 4 active results in lanes 0..3
 * and zero in lanes 4..7. */
static inline void gf64_split_prod_512(__m512i prod, __m512i *lo_out, __m512i *hi_out) {
	__m512i zero = _mm512_setzero_si512();
	*lo_out = _mm512_permutex2var_epi64(prod, GF64_IDX_LO, zero);
	*hi_out = _mm512_permutex2var_epi64(prod, GF64_IDX_HI, zero);
}

/* Vectorized GF(2^64) reduction. Each lane i (0..3 active, 4..7 zero) holds
 * a 128-bit carry-less product split into (lo_i, hi_i). Returns lo_i ^ t_lo_i
 * ^ t2_i per lane, where t_lo = lo(hi*0x1B) and t2 = (overflow(hi*0x1B))*0x1B.
 * Bit-exact to gf64_reduce_128 (proven by par3-kernel-parity.js, 1215/1215). */
static inline __m512i gf64_reduce_512(__m512i lo_vec, __m512i hi_vec) {
	__m512i one = _mm512_set1_epi64(1);

	/* t_lo = (hi<<4) ^ (hi<<3) ^ (hi<<1) ^ hi */
	__m512i t_lo = _mm512_xor_si512(
		_mm512_xor_si512(
			_mm512_slli_epi64(hi_vec, 4),
			_mm512_slli_epi64(hi_vec, 3)
		),
		_mm512_xor_si512(
			_mm512_slli_epi64(hi_vec, 1),
			hi_vec
		)
	);

	/* R_hi: 4-bit overflow of hi*0x1B per lane.
	 *   bit0 = (hi>>60) ^ (hi>>61) ^ (hi>>63)
	 *   bit1 = (hi>>61) ^ (hi>>62)
	 *   bit2 = (hi>>62) ^ (hi>>63)
	 *   bit3 = (hi>>63)
	 * Each bit is computed masked to 1 bit, then shifted to its position. */
	__m512i bit0 = _mm512_and_si512(
		_mm512_xor_si512(_mm512_srli_epi64(hi_vec, 60),
		                 _mm512_xor_si512(_mm512_srli_epi64(hi_vec, 61),
		                                  _mm512_srli_epi64(hi_vec, 63))),
		one);
	__m512i bit1 = _mm512_slli_epi64(
		_mm512_and_si512(_mm512_xor_si512(_mm512_srli_epi64(hi_vec, 61),
		                                  _mm512_srli_epi64(hi_vec, 62)),
		                 one),
		1);
	__m512i bit2 = _mm512_slli_epi64(
		_mm512_and_si512(_mm512_xor_si512(_mm512_srli_epi64(hi_vec, 62),
		                                  _mm512_srli_epi64(hi_vec, 63)),
		                 one),
		2);
	__m512i bit3 = _mm512_slli_epi64(
		_mm512_and_si512(_mm512_srli_epi64(hi_vec, 63), one),
		3);
	__m512i R_hi = _mm512_or_si512(
		_mm512_or_si512(bit0, bit1),
		_mm512_or_si512(bit2, bit3));

	/* t2 = (R_hi<<4) ^ (R_hi<<3) ^ (R_hi<<1) ^ R_hi  (R_hi * 0x1B, 8 bits) */
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

	__m512i result = _mm512_xor_si512(_mm512_xor_si512(lo_vec, t_lo), t2);

	/* Debug canary: broadcast the canary value across a ZMM and XOR into the
	 * result. With canary=0 (the default), this is a no-op. With canary set,
	 * parity tests must fail. */
	if (gf64_reduce_canary != 0) {
		__m512i canary_v = _mm512_set1_epi64((int64_t)gf64_reduce_canary);
		result = _mm512_xor_si512(result, canary_v);
	}

	return result;
}

/* EVEX.512 VPCLMULQDQ notes (verified against Intel docs, felixcloutier.com/x86/pclmulqdq,
 * and the Rust std::arch::x86_64::cpuid reference at
 * doc.rust-lang.org/src/core/stdarch/crates/core_arch/src/x86/vpclmulqdq.rs.html):
 *   - Takes 512-bit (ZMM) operands; operates independently on each 128-bit lane.
 *   - Each lane performs one 64x64->128 carry-less multiply.
 *   - Output is a 512-bit ZMM holding 4 x 128-bit products (NOT 2 x 128-bit;
 *     the 2 x 128-bit variant requires the VEX.256 encoding).
 *   - imm8 bits: [4] selects qword of src2, [0] selects qword of src1, per lane.
 *     imm8=0x00 -> low qword of each lane in both operands (what we want: GF
 *     elements live in the low qword of each lane, coefficients broadcast the same).
 *
 * Layout of a 512-bit vector `in_vec` constructed via
 *   _mm512_set_epi64(0, in[i+3], 0, in[i+2], 0, in[i+1], 0, in[i+0]):
 *   lane 0 (bits[127:0])   = [ in[i+0] | 0 ]
 *   lane 1 (bits[255:128]) = [ in[i+1] | 0 ]
 *   lane 2 (bits[383:256]) = [ in[i+2] | 0 ]
 *   lane 3 (bits[511:384]) = [ in[i+3] | 0 ]
 *   With coeff_broadcast = [c, c, c, c, c, c, c, c] and imm8=0x00:
 *     result lane 0 = clmul(in[i+0], c)  (128-bit)
 *     result lane 1 = clmul(in[i+1], c)  (128-bit)
 *     result lane 2 = clmul(in[i+2], c)  (128-bit)
 *     result lane 3 = clmul(in[i+3], c)  (128-bit)
 *   Four GF element products per VPCLMULQDQ call.
 *
 * Each 128-bit product is then reduced to 64 bits via gf64_reduce_512 (vectorized
 * XOR-shift reduction; see the block-comment above gf64_reduce_512). The previous
 * implementation used 4 scalar gf64_reduce_128 calls per VPCLMULQDQ via
 * _mm512_extracti32x4_epi32; the new path keeps the lo/hi halves in ZMM and
 * emits 4 reduced values with a single masked store.
 */

__attribute__((target("avx512f,vpclmulqdq")))
void gf64_region_mul_avx512_arr(gf64_t *HEDLEY_RESTRICT out, const gf64_t *HEDLEY_RESTRICT in, const gf64_t *HEDLEY_RESTRICT coeff, size_t len, size_t n_coeff) {
	size_t i = 0;

	if (n_coeff == 1) {
		/* Fast path: single coefficient broadcast across all lanes.
		 * Broadcast once outside the loop, then do 2 VPCLMULQDQ per outer
		 * iteration (8 GF elements processed, 4 elements per call). Each
		 * call's 4-product ZMM is reduced in vector form via gf64_reduce_512
		 * and stored with a 0x0F mask (writes lanes 0..3 = 4 GF elements).
		 */
		uint64_t c0 = coeff[0];
		__m512i coeff_broadcast = _mm512_set1_epi64((int64_t)c0);

		size_t blocks = len / 8;
		for (size_t b = 0; b < blocks; b++) {
			/* Call 1: out[i+0..i+3] = clmul(in[i+0..i+3], c0) */
			__m512i in_lo = _mm512_set_epi64(0, (int64_t)in[i + 3], 0, (int64_t)in[i + 2],
			                                  0, (int64_t)in[i + 1], 0, (int64_t)in[i + 0]);
			__m512i prod_lo = _mm512_clmulepi64_epi128(in_lo, coeff_broadcast, 0x00);
			__m512i lo_v, hi_v;
			gf64_split_prod_512(prod_lo, &lo_v, &hi_v);
			_mm512_mask_storeu_epi64(out + i, (__mmask8)0x0F, gf64_reduce_512(lo_v, hi_v));

			/* Call 2: out[i+4..i+7] = clmul(in[i+4..i+7], c0) */
			__m512i in_hi = _mm512_set_epi64(0, (int64_t)in[i + 7], 0, (int64_t)in[i + 6],
			                                  0, (int64_t)in[i + 5], 0, (int64_t)in[i + 4]);
			__m512i prod_hi = _mm512_clmulepi64_epi128(in_hi, coeff_broadcast, 0x00);
			gf64_split_prod_512(prod_hi, &lo_v, &hi_v);
			_mm512_mask_storeu_epi64(out + i + 4, (__mmask8)0x0F, gf64_reduce_512(lo_v, hi_v));

			i += 8;
		}

		/* Tail (0..7 elements) -- scalar epilog. */
		while (i < len) {
			out[i] = gf64_mul_reference(in[i], c0);
			i++;
		}
	} else {
		/* General case: each input element is multiplied by ALL coefficients
		 * and the products are XORed together (dot product per element).
		 * Process 4 elements per VPCLMULQDQ call (one per 128-bit lane).
		 *
		 * Vectorized approach: split each VPCLMULQDQ product into (lo, hi)
		 * ZMMs, XOR-fold the (lo, hi) pairs across all coefficients, and apply
		 * the reduction once at the end. This is bit-exact to the per-coeff
		 * reduce-then-XOR path because XOR is linear over GF(2)[x]:
		 *   (a^b)*0x1B = (a*0x1B) ^ (b*0x1B)
		 * and (a+b)*0x1B = (a*0x1B)+(b*0x1B) in GF(2)[x] (same thing).
		 */
		size_t blocks = len / 4;
		__m512i zero = _mm512_setzero_si512();
		for (size_t b = 0; b < blocks; b++) {
			__m512i in_vec = _mm512_set_epi64(0, (int64_t)in[i + 3], 0, (int64_t)in[i + 2],
			                                   0, (int64_t)in[i + 1], 0, (int64_t)in[i + 0]);

			__m512i acc_lo = _mm512_setzero_si512();
			__m512i acc_hi = _mm512_setzero_si512();

			for (size_t c = 0; c < n_coeff; c++) {
				__m512i coeff_bc = _mm512_set1_epi64((int64_t)coeff[c]);
				__m512i prod = _mm512_clmulepi64_epi128(in_vec, coeff_bc, 0x00);
				__m512i lo_v = _mm512_permutex2var_epi64(prod, GF64_IDX_LO, zero);
				__m512i hi_v = _mm512_permutex2var_epi64(prod, GF64_IDX_HI, zero);
				acc_lo = _mm512_xor_si512(acc_lo, lo_v);
				acc_hi = _mm512_xor_si512(acc_hi, hi_v);
			}

			__m512i result = gf64_reduce_512(acc_lo, acc_hi);
			_mm512_mask_storeu_epi64(out + i, (__mmask8)0x0F, result);

			i += 4;
		}

		/* Tail (0..3 elements) -- scalar epilog with SUM semantics. */
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

__attribute__((target("avx512f,vpclmulqdq")))
void gf64_region_muladd_avx512_arr(gf64_t *HEDLEY_RESTRICT out, const gf64_t *HEDLEY_RESTRICT in, const gf64_t *HEDLEY_RESTRICT coeff, size_t len, size_t n_coeff) {
	size_t i = 0;

	if (n_coeff == 1) {
		/* Fast path: single coefficient broadcast across all lanes.
		 * Software-pipelined inner loop: 4 VPCLMULQDQ per iteration (16 GF
		 * elements), with prev-loads issued BETWEEN clmul and reduce so
		 * load latency overlaps with reduction work. The 4 clmul, 4 loads,
		 * 4 reduce, 4 XOR+store streams are independent and the OOO engine
		 * can keep all four in flight concurrently to hide VPCLMULQDQ
		 * latency (~3-7 cycles on Zen4). */
		uint64_t c0 = coeff[0];
		__m512i coeff_broadcast = _mm512_set1_epi64((int64_t)c0);

		size_t blocks = len / 16;
		for (size_t b = 0; b < blocks; b++) {
			/* PHASE 1: 4 independent clmul (back-to-back, no deps). */
			__m512i in_0 = _mm512_set_epi64(0, (int64_t)in[i + 3], 0, (int64_t)in[i + 2],
			                                 0, (int64_t)in[i + 1], 0, (int64_t)in[i + 0]);
			__m512i prod_0 = _mm512_clmulepi64_epi128(in_0, coeff_broadcast, 0x00);

			__m512i in_1 = _mm512_set_epi64(0, (int64_t)in[i + 7], 0, (int64_t)in[i + 6],
			                                 0, (int64_t)in[i + 5], 0, (int64_t)in[i + 4]);
			__m512i prod_1 = _mm512_clmulepi64_epi128(in_1, coeff_broadcast, 0x00);

			__m512i in_2 = _mm512_set_epi64(0, (int64_t)in[i + 11], 0, (int64_t)in[i + 10],
			                                 0, (int64_t)in[i + 9], 0, (int64_t)in[i + 8]);
			__m512i prod_2 = _mm512_clmulepi64_epi128(in_2, coeff_broadcast, 0x00);

			__m512i in_3 = _mm512_set_epi64(0, (int64_t)in[i + 15], 0, (int64_t)in[i + 14],
			                                 0, (int64_t)in[i + 13], 0, (int64_t)in[i + 12]);
			__m512i prod_3 = _mm512_clmulepi64_epi128(in_3, coeff_broadcast, 0x00);

			/* PHASE 2: 4 prev-loads (issued here, AFTER clmul but BEFORE
			 * reduce, so load latency overlaps with the reduction work). */
			__m512i prev_0 = _mm512_maskz_loadu_epi64((__mmask8)0x0F, out + i);
			__m512i prev_1 = _mm512_maskz_loadu_epi64((__mmask8)0x0F, out + i + 4);
			__m512i prev_2 = _mm512_maskz_loadu_epi64((__mmask8)0x0F, out + i + 8);
			__m512i prev_3 = _mm512_maskz_loadu_epi64((__mmask8)0x0F, out + i + 12);

			/* PHASE 3: 4 independent reduce ops. */
			__m512i lo_v, hi_v;
			gf64_split_prod_512(prod_0, &lo_v, &hi_v);
			__m512i red_0 = gf64_reduce_512(lo_v, hi_v);
			gf64_split_prod_512(prod_1, &lo_v, &hi_v);
			__m512i red_1 = gf64_reduce_512(lo_v, hi_v);
			gf64_split_prod_512(prod_2, &lo_v, &hi_v);
			__m512i red_2 = gf64_reduce_512(lo_v, hi_v);
			gf64_split_prod_512(prod_3, &lo_v, &hi_v);
			__m512i red_3 = gf64_reduce_512(lo_v, hi_v);

			/* PHASE 4: 4 independent XOR+stores. */
			_mm512_mask_storeu_epi64(out + i,      (__mmask8)0x0F, _mm512_xor_si512(prev_0, red_0));
			_mm512_mask_storeu_epi64(out + i + 4,  (__mmask8)0x0F, _mm512_xor_si512(prev_1, red_1));
			_mm512_mask_storeu_epi64(out + i + 8,  (__mmask8)0x0F, _mm512_xor_si512(prev_2, red_2));
			_mm512_mask_storeu_epi64(out + i + 12, (__mmask8)0x0F, _mm512_xor_si512(prev_3, red_3));

			i += 16;
		}

		/* Tail (0..15 elements) -- scalar epilog. */
		while (i < len) {
			out[i] ^= gf64_mul_reference(in[i], c0);
			i++;
		}
	} else {
		size_t blocks = len / 4;
		__m512i zero = _mm512_setzero_si512();
		for (size_t b = 0; b < blocks; b++) {
			__m512i in_vec = _mm512_set_epi64(0, (int64_t)in[i + 3], 0, (int64_t)in[i + 2],
			                                   0, (int64_t)in[i + 1], 0, (int64_t)in[i + 0]);

			__m512i acc_lo = _mm512_setzero_si512();
			__m512i acc_hi = _mm512_setzero_si512();

			for (size_t c = 0; c < n_coeff; c++) {
				__m512i coeff_bc = _mm512_set1_epi64((int64_t)coeff[c]);
				__m512i prod = _mm512_clmulepi64_epi128(in_vec, coeff_bc, 0x00);
				__m512i lo_v = _mm512_permutex2var_epi64(prod, GF64_IDX_LO, zero);
				__m512i hi_v = _mm512_permutex2var_epi64(prod, GF64_IDX_HI, zero);
				acc_lo = _mm512_xor_si512(acc_lo, lo_v);
				acc_hi = _mm512_xor_si512(acc_hi, hi_v);
			}

			__m512i result = gf64_reduce_512(acc_lo, acc_hi);
			__m512i prev = _mm512_maskz_loadu_epi64((__mmask8)0x0F, out + i);
			_mm512_mask_storeu_epi64(out + i, (__mmask8)0x0F,
			                        _mm512_xor_si512(prev, result));

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