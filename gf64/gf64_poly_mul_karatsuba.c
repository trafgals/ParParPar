/*
 * ============================================================================
 * gf64/gf64_poly_mul_karatsuba.c — Karatsuba over GF(2^64)[x]
 *
 * Recurrence: A*B = A_lo*B_lo + Z1*x^k + A_hi*B_hi*x^(2k), where
 *   A = A_lo + A_hi * x^k,  B = B_lo + B_hi * x^k,  k = n/2
 *   Z1 = (A_lo+A_hi)*(B_lo+B_hi) + A_lo*B_lo + A_hi*B_hi
 * The subtraction collapses to XOR in characteristic 2.
 *
 * Cost: 3 half-size multiplications instead of 4. Asymptotically
 *   T(n) = 3 T(n/2) + O(n) ⇒ O(n^log2 3) ≈ O(n^1.585).
 *
 * Implementation choices
 * ----------------------
 * 1. Power-of-2 padding. Inputs are zero-extended to the next power of two
 *    of max(len_a, len_b) after capping both at `out_len`. This makes the
 *    recursive split always at half the size with both halves non-empty,
 *    simplifying the base case.
 *
 * 2. Full-result compute + truncate. We compute the full 2n-1 coefficients
 *    of the padded result and copy the low `out_len` to the caller. For the
 *    Newton-iteration use pattern in gf64_poly_invmod, both operands are
 *    truncated to `out_len` at the entry, so the padding waste is bounded
 *    by one extra power-of-2 doubling at most.
 *
 * 3. Per-call malloc for scratch. The recursion needs 6n words of scratch
 *    (3 sub-products + 2 sum buffers + 4h bytes for the next-level
 *    recursion). Per-call allocation is the simplest correct approach for
 *    this iteration; a thread_local scratch pool is a future optimization
 *    (and trivial to add behind this same entry point).
 *
 * 4. Schoolbook base case at GF64_KARATSUBA_THRESHOLD (64). The constant
 *    overhead of recursion (3 small mallocs, copying, combine loop) only
 *    pays off once n exceeds the threshold by a comfortable margin.
 * ============================================================================
 */

#include "gf64_poly_mul_karatsuba.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

HEDLEY_BEGIN_C_DECLS

/*
 * Scalar GF(2^64) reference, declared in gf64_single.c. Using the reference
 * (rather than the AVX-512 vectorized gf64_mul) keeps this TU portable and
 * makes the Karatsuba recurrence directly comparable to the existing
 * gf64_poly_mul_internal in gf64_additive_fft.c — both are scalar.
 */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

#define GF64_KARATSUBA_THRESHOLD ((size_t)64)

/* ----------------------------------------------------------------------------
 * next_pow2_size(n): smallest power of two >= n. Returns 1 for n=0.
 * Used to pad the operands before the recursive Karatsuba split.
 * ---------------------------------------------------------------------------- */
static size_t next_pow2_size(size_t n) {
	size_t p = 1;
	while (p < n) {
		p <<= 1;
	}
	return p;
}

/* ----------------------------------------------------------------------------
 * kschoolbook: the Karatsuba leaf. Identical in body to
 * gf64_poly_mul_internal in gf64_additive_fft.c (truncation via
 * a_cap/b_cap, zero-skip on a). Inlined into the recursion's base case so
 * the threshold check fires once per level.
 * ---------------------------------------------------------------------------- */
static void kschoolbook(
	gf64_t *out,
	const gf64_t *a, size_t len_a,
	const gf64_t *b, size_t len_b,
	size_t out_len
) {
	assert(out != NULL);
	assert(a != NULL);
	assert(b != NULL);

	memset(out, 0, out_len * sizeof(gf64_t));

	size_t a_cap = (len_a < out_len) ? len_a : out_len;
	size_t b_cap = (len_b < out_len) ? len_b : out_len;

	for (size_t i = 0; i < a_cap; i++) {
		gf64_t ai = a[i];
		if (ai == 0) {
			continue;
		}
		size_t j_max = (b_cap < out_len - i) ? b_cap : (out_len - i);
		for (size_t j = 0; j < j_max; j++) {
			out[i + j] ^= gf64_mul_reference(ai, b[j]);
		}
	}
}

/* ----------------------------------------------------------------------------
 * kbal_full: balanced Karatsuba.
 *
 * Both a and b have length n (a power of two; the caller has padded to
 * next_pow2_size(max(len_a, len_b)) at the entry). Writes the full 2n-1
 * convolution into `out` (size 2n-1). `scratch` of size `scratch_size`
 * is a caller-owned buffer reused across the recursion; sub-calls receive
 * the EXACT number of slots they have left (pass-through sizing) so the
 * allocator only needs to satisfy the top-level 8n ceiling.
 *
 * Layout in `scratch` (offsets relative to scratch base):
 *   [z0: 2h-1 | z2: 2h-1 | a_sum: h | b_sum: h | z1: 2h-1 | recurse: rest]
 *
 * The header (z0..z1) is exactly 7h-3 slots. Whatever remains is the
 * recurse region; we hand that to the three sub-calls. Each sub-call
 * does the same accounting, so the recursion bottoms out cleanly with
 * the schoolbook base case at GF64_KARATSUBA_THRESHOLD.
 *
 * Top-level scratch size needed = 8n (the closed-form sum
 *   Σ_{k=0..K-1} (2*(n/2^k) - 3)  where K = log2(n/THRESHOLD)
 *   ≈ 4n - 3*log2(n/THRESHOLD)
 * bounds the per-level recurse requirement; 8n is a comfortable over-approx
 * that absorbs the small constants).
 * ---------------------------------------------------------------------------- */
static void kbal_full(
	gf64_t *HEDLEY_RESTRICT out,           /* 2n-1 coeffs */
	const gf64_t *HEDLEY_RESTRICT a,
	const gf64_t *HEDLEY_RESTRICT b,
	size_t n,
	gf64_t *HEDLEY_RESTRICT scratch,       /* scratch_size coeffs */
	size_t scratch_size
) {
	if (n <= GF64_KARATSUBA_THRESHOLD) {
		kschoolbook(out, a, n, b, n, (2 * n) - 1);
		return;
	}

	size_t half = n / 2;

	size_t len_z    = (2 * half) - 1;
	size_t len_sum  = half;
	size_t header   = 3 * len_z + 2 * len_sum;  /* 7h - 3 */

	assert(scratch_size > header);
	size_t recurse_size = scratch_size - header;

	gf64_t *z0     = scratch;
	gf64_t *z2     = z0 + len_z;
	gf64_t *a_sum  = z2 + len_z;
	gf64_t *b_sum  = a_sum + len_sum;
	gf64_t *z1     = b_sum + len_sum;
	gf64_t *recurse = z1 + len_z;

	/* z0 = A_lo * B_lo */
	kbal_full(z0, a, b, half, recurse, recurse_size);

	/* z2 = A_hi * B_hi */
	kbal_full(z2, a + half, b + half, half, recurse, recurse_size);

	/* a_sum = A_lo + A_hi,  b_sum = B_lo + B_hi */
	for (size_t i = 0; i < half; i++) {
		a_sum[i] = a[i] ^ a[i + half];
		b_sum[i] = b[i] ^ b[i + half];
	}

	/* z1 = a_sum * b_sum */
	kbal_full(z1, a_sum, b_sum, half, recurse, recurse_size);

	/* z1 = z1 + z0 + z2 (char-2 subtraction) */
	for (size_t i = 0; i < len_z; i++) {
		z1[i] ^= z0[i] ^ z2[i];
	}

	/* Combine: out = z0 + (z1 << half) + (z2 << n) */
	memset(out, 0, ((2 * n) - 1) * sizeof(gf64_t));
	for (size_t i = 0; i < len_z; i++) {
		out[i]        ^= z0[i];
		out[i + half] ^= z1[i];
		out[i + n]    ^= z2[i];
	}
}

/* ----------------------------------------------------------------------------
 * Public entry: Karatsuba with truncation.
 *
 * Cap both inputs at `out_len` (consistent with gf64_poly_mul_internal's
 * a_cap/b_cap discipline — coefficients beyond out_len can't land in the
 * output anyway), then dispatch:
 *   - max(len_a, len_b) <= THRESHOLD  → schoolbook
 *   - else                            → balanced Karatsuba (full result + truncate)
 * ---------------------------------------------------------------------------- */
void gf64_poly_mul_karatsuba(
	gf64_t *out,
	const gf64_t *a, size_t len_a,
	const gf64_t *b, size_t len_b,
	size_t out_len
) {
	if (out_len == 0) {
		return;
	}

	/* Truncate inputs at out_len. Coefficients beyond out_len can't land
	 * in the output regardless of the algorithm. */
	if (len_a > out_len) len_a = out_len;
	if (len_b > out_len) len_b = out_len;

	if (len_a == 0 || len_b == 0) {
		memset(out, 0, out_len * sizeof(gf64_t));
		return;
	}

	size_t n_raw = (len_a > len_b) ? len_a : len_b;

	/* Below threshold: defer to the existing schoolbook logic. This is
	 * the same code path used by the pre-Karatsuba
	 * gf64_poly_mul_internal — bit-exact by construction. */
	if (n_raw <= GF64_KARATSUBA_THRESHOLD) {
		kschoolbook(out, a, len_a, b, len_b, out_len);
		return;
	}

	size_t n = next_pow2_size(n_raw);

	/* Power-of-2-padded operands. */
	gf64_t *a_pad = (gf64_t *)calloc(n, sizeof(gf64_t));
	gf64_t *b_pad = (gf64_t *)calloc(n, sizeof(gf64_t));
	if (a_pad == NULL || b_pad == NULL) {
		free(a_pad);
		free(b_pad);
		abort();
	}
	memcpy(a_pad, a, len_a * sizeof(gf64_t));
	memcpy(b_pad, b, len_b * sizeof(gf64_t));

	/* Full result + scratch. Top-level scratch size = 8n is a safe
	 * over-approximation of the closed-form sum Σ (2*(n/2^k) - 3) over
	 * the K recursion levels; see kbal_full's header comment.
	 *
	 * full_result is sized to max(2n-1, out_len). The Karatsuba
	 * recursion writes up to index 2n-2; if the caller asks for more
	 * coefficients than the polynomial product has (e.g. out_len == 256
	 * for two 128-coefficient operands whose product has only 255
	 * coefficients), the surplus slots must come back as zero. calloc
	 * gives us that for free. */
	size_t full_len = (2 * n) - 1;
	if (out_len > full_len) {
		full_len = out_len;
	}
	gf64_t *full_result = (gf64_t *)calloc(full_len, sizeof(gf64_t));
	gf64_t *scratch     = (gf64_t *)malloc(8 * n * sizeof(gf64_t));
	if (full_result == NULL || scratch == NULL) {
		free(a_pad);
		free(b_pad);
		free(full_result);
		free(scratch);
		abort();
	}

	kbal_full(full_result, a_pad, b_pad, n, scratch, 8 * n);

	/* Copy the requested range. Surplus slots are guaranteed zero from
	 * the calloc above. */
	memcpy(out, full_result, out_len * sizeof(gf64_t));

	free(a_pad);
	free(b_pad);
	free(full_result);
	free(scratch);
}

HEDLEY_END_C_DECLS