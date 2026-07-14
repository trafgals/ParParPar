/*
 * ============================================================================
 * gf64/gf64_additive_fft.c — Gao-Mateer-style additive FFT over GF(2^64)
 *
 * T3 (scalar) and T4 (AVX-512 vectorized) of the par3-cauchy-fft-kernel plan.
 *
 * SCALAR FORWARD (T3)
 *   The forward transform follows the simplified monomial-basis recursion:
 *     1. split coefficients into even and odd halves;
 *     2. apply the Frobenius map (square) to each half;
 *     3. recursively transform each half;
 *     4. combine with the scalar additive butterfly
 *          y[i]     = even[i] + i * odd[i]
 *          y[i+h]   = y[i] + odd[i]
 *        over GF(2^64), where h = n / 2.
 *
 * SCALAR INVERSE (T3)
 *   The exact inverse must undo the Frobenius map with 63 repeated squarings;
 *   squaring itself is not self-inverse in GF(2^64). The public inverse then
 *   applies the non-normalized T3 round-trip scale 2^(log2(n)-1).
 *
 * AVX-512 FORWARD / INVERSE (T4)
 *   The AVX-512 entry points mirror the scalar recursion exactly, but lift
 *   the element-wise operations into 8-lane SIMD:
 *     - Frobenius squaring uses gf64_square_avx512 (T2)
 *     - The butterfly's alpha * odd multiplications use gf64_mul_avx512 (T1)
 *   For n < 64 the AVX-512 entries delegate to the scalar functions, since
 *   the SIMD setup overhead dominates for small problems. Bit-exactness is
 *   guaranteed by the bit-exactness of gf64_square_avx512 vs.
 *   gf64_mul_reference(x, x) and gf64_mul_avx512 vs. gf64_mul_reference —
 *   verified by the existing T1 / T2 parity tests and the new AVX-512
 *   parity tests in test_gf64_additive_fft.c.
 *
 * SCOPE
 *   This file consolidates the T3 (scalar) and T4 (AVX-512) implementations
 *   into a single translation unit. The AVX-512 code does NOT modify any
 *   scalar primitive (gf64/gf64_solve.c, gf64/gf64_square.c,
 *   gf64/gf64_mul_avx512.c are all read-only dependencies).
 * ============================================================================
 */

#include "gf64_additive_fft.h"
#include "gf64_mul.h"
#include "gf64_square.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

HEDLEY_BEGIN_C_DECLS

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

static int gf64_is_power_of_two(size_t n) {
	return n != 0 && (n & (n - 1)) == 0;
}

static size_t gf64_log2_size(size_t n) {
	size_t log_n = 0;
	while (n > 1) {
		n >>= 1;
		log_n++;
	}
	return log_n;
}

static gf64_t gf64_fft_scale(size_t n) {
	size_t log_n = gf64_log2_size(n);
	if (log_n == 0) {
		return 1;
	}
	return (gf64_t)1ULL << (log_n - 1);
}

static gf64_t gf64_pow_u64(gf64_t base, uint64_t exponent) {
	gf64_t result = 1;
	while (exponent != 0) {
		if ((exponent & 1U) != 0) {
			result = gf64_mul_reference(result, base);
		}
		exponent >>= 1;
		if (exponent != 0) {
			base = gf64_mul_reference(base, base);
		}
	}
	return result;
}

static gf64_t gf64_inverse_nonzero(gf64_t a) {
	assert(a != 0);
	return gf64_pow_u64(a, UINT64_MAX - 1ULL);
}

static void gf64_alloc_halves(size_t half, gf64_t **even_half, gf64_t **odd_half) {
	*even_half = (gf64_t *)malloc(half * sizeof(gf64_t));
	*odd_half = (gf64_t *)malloc(half * sizeof(gf64_t));
	if (*even_half == NULL || *odd_half == NULL) {
		free(*even_half);
		free(*odd_half);
		abort();
	}
}

static void gf64_square_vector_in_place(gf64_t *values, size_t n) {
	for (size_t i = 0; i < n; i++) {
		gf64_t in = values[i];
		gf64_t out;
		gf64_square(&out, &in, 1);
		values[i] = out;
	}
}

static void gf64_apply_inverse_frobenius(gf64_t *values, size_t n) {
	for (size_t round = 0; round < 63; round++) {
		gf64_square_vector_in_place(values, n);
	}
}

static void gf64_scale_vector(gf64_t *values, size_t n, gf64_t scale) {
	if (scale == 1) {
		return;
	}
	for (size_t i = 0; i < n; i++) {
		values[i] = gf64_mul_reference(values[i], scale);
	}
}

static void gf64_fft_inverse_unscaled(gf64_t *poly, size_t n) {
	if (n <= 1) {
		return;
	}

	size_t half = n / 2;
	gf64_t *even_half;
	gf64_t *odd_half;
	gf64_alloc_halves(half, &even_half, &odd_half);

	for (size_t i = 0; i < half; i++) {
		gf64_t alpha = (gf64_t)i;
		gf64_t odd = poly[i] ^ poly[i + half];
		odd_half[i] = odd;
		even_half[i] = poly[i] ^ gf64_mul_reference(alpha, odd);
	}

	gf64_fft_inverse_unscaled(even_half, half);
	gf64_fft_inverse_unscaled(odd_half, half);
	gf64_apply_inverse_frobenius(even_half, half);
	gf64_apply_inverse_frobenius(odd_half, half);

	for (size_t i = 0; i < half; i++) {
		poly[2 * i] = even_half[i];
		poly[2 * i + 1] = odd_half[i];
	}

	free(even_half);
	free(odd_half);
}

void gf64_fft_forward(gf64_t *poly, size_t n) {
	if (n <= 1) {
		return;
	}
	assert(poly != NULL);
	assert(gf64_is_power_of_two(n));

	size_t half = n / 2;
	gf64_t *even_half;
	gf64_t *odd_half;
	gf64_alloc_halves(half, &even_half, &odd_half);

	for (size_t i = 0; i < half; i++) {
		even_half[i] = poly[2 * i];
		odd_half[i] = poly[2 * i + 1];
	}

	gf64_square_vector_in_place(even_half, half);
	gf64_square_vector_in_place(odd_half, half);

	gf64_fft_forward(even_half, half);
	gf64_fft_forward(odd_half, half);

	for (size_t i = 0; i < half; i++) {
		gf64_t alpha = (gf64_t)i;
		gf64_t beta = even_half[i];
		gf64_t gamma = gf64_mul_reference(alpha, odd_half[i]);
		poly[i] = beta ^ gamma;
		poly[i + half] = poly[i] ^ odd_half[i];
	}

	free(even_half);
	free(odd_half);
}

void gf64_fft_inverse(gf64_t *poly, size_t n) {
	if (n <= 1) {
		return;
	}
	assert(poly != NULL);
	assert(gf64_is_power_of_two(n));

	gf64_fft_inverse_unscaled(poly, n);
	gf64_scale_vector(poly, n, gf64_fft_scale(n));
}

/*
 * Forward declaration for the Karatsuba drop-in (see
 * gf64_poly_mul_karatsuba.c). Defined in its own TU to keep this file's
 * focus on the additive FFT primitive. Phase 2a of the "beat PAR2" plan.
 */
extern void gf64_poly_mul_karatsuba(
	gf64_t *out,
	const gf64_t *a, size_t len_a,
	const gf64_t *b, size_t len_b,
	size_t out_len
);
extern void gf64_poly_mul_toom3(
	gf64_t *out, const gf64_t *a, size_t len_a,
	const gf64_t *b, size_t len_b, size_t out_len
);

/* Karatsuba crossover. Below this size, schoolbook wins on constant overhead.
 * Above it (and the operands being at-or-above this size after truncation),
 * Karatsuba wins asymptotically at O(n^1.585) vs O(n^2). The threshold is
 * also the schoolbook base case inside gf64_poly_mul_karatsuba's recursion,
 * so the two stay in sync. */
#define GF64_POLY_MUL_INTERNAL_KARATSUBA_MIN ((size_t)128)

/* Toom-Cook 3 (2.3): O(n^1.465), beats Karatsuba's O(n^1.585) above
 * a measured crossover. DISABLED by default — the existing
 * gf64_poly_mul_toom3 implementation does per-recursion mallocs for
 * 6 limb buffers × 5 recursive sub-products, which costs more than
 * Karatsuba at all sizes measured (n=64..4096; bench log in commit).
 * Set to a non-SIZE_MAX value to enable once a thread_local scratch
 * pool is added (or once the malloc overhead is otherwise amortized). */
#define GF64_POLY_MUL_INTERNAL_TOOM3_MIN ((size_t)SIZE_MAX)

/*
 * Shared convolution kernel for gf64_poly_mul and gf64_poly_mul_padded.
 *
 * Computes out[0 .. out_len) as the low-order coefficients of the convolution
 * a * b in GF(2^64)[x].  `a` has len_a coefficients and `b` has len_b
 * coefficients (constant-first, length = degree + 1).  Coefficients of
 * index >= out_len are discarded; trailing slots [0, out_len) are zeroed
 * before writing.
 *
 * Implementation: dispatches to either the schoolbook triple loop (below
 * GF64_POLY_MUL_INTERNAL_KARATSUBA_MIN) or to gf64_poly_mul_karatsuba
 * (above). The Gao-Mateer additive FFT in this TU does NOT implement the
 * convolution theorem for arbitrary GF(2^64) inputs — the recursive
 * structure is a transform of the monomial basis into a "twiddle"-laden
 * evaluation-like representation where pointwise multiplication is not
 * equivalent to polynomial convolution.  A correct FFT-based multiplication
 * would need either:
 *
 *   (a) the full Gao-Mateer "tower of extensions" pipeline (works in
 *       characteristic 2 but needs to use the *evaluation* basis at the
 *       top level, not just the monomial-basis transform), or
 *   (b) an NTT over a prime subfield of GF(2^64) with roots of unity.
 *
 * Both are research-level primitives. Karatsuba (Phase 2a) is the
 * pragmatic, immediately useful intermediate that gets the
 * polynomial-heavy T6/T7/T8 primitives to O(n^1.585) without the
 * research-grade FFT work.
 */
static void gf64_poly_mul_internal(
	gf64_t *out,
	const gf64_t *a, size_t len_a,
	const gf64_t *b, size_t len_b,
	size_t out_len
) {
	assert(out != NULL);
	assert(a != NULL);
	assert(b != NULL);

	if (out_len == 0) {
		return;
	}

	/* Karatsuba above the threshold. We require ALL of (len_a, len_b,
	 * out_len) to be at-or-above the crossover so we don't pay Karatsuba's
	 * malloc/scratch overhead when the operands are small (the schoolbook
	 * base case would fire immediately anyway, but with extra setup cost). */
	if (len_a >= GF64_POLY_MUL_INTERNAL_TOOM3_MIN &&
	    len_b >= GF64_POLY_MUL_INTERNAL_TOOM3_MIN &&
	    out_len >= GF64_POLY_MUL_INTERNAL_TOOM3_MIN) {
		gf64_poly_mul_toom3(out, a, len_a, b, len_b, out_len);
		return;
	}
	if (len_a >= GF64_POLY_MUL_INTERNAL_KARATSUBA_MIN &&
	    len_b >= GF64_POLY_MUL_INTERNAL_KARATSUBA_MIN &&
	    out_len >= GF64_POLY_MUL_INTERNAL_KARATSUBA_MIN) {
		gf64_poly_mul_karatsuba(out, a, len_a, b, len_b, out_len);
		return;
	}

	memset(out, 0, out_len * sizeof(gf64_t));

	/* Cap reads at the truncation point: any coefficient of index >= out_len
	 * in either input contributes only to output slots [out_len, ...), which
	 * are out of contract.  Skipping the long-tail inner products gives the
	 * Newton-iteration truncation idiom for free without needing a separate
	 * "low-order only" loop. */
	size_t a_cap = (len_a < out_len) ? len_a : out_len;
	size_t b_cap = (len_b < out_len) ? len_b : out_len;

	for (size_t i = 0; i < a_cap; i++) {
		gf64_t ai = a[i];
		if (ai == 0) continue;
		/* Output slot (i + j) must be < out_len for this product to land
		 * inside the truncation; clamp j at out_len - i - 1. */
		size_t j_max = (b_cap < out_len - i) ? b_cap : (out_len - i);
		for (size_t j = 0; j < j_max; j++) {
			out[i + j] ^= gf64_mul_reference(ai, b[j]);
		}
	}
}

/*
 * Public polynomial multiplication: out = a * b (constant-first, length =
 * deg_a + deg_b + 1 coefficients).  Schoolbook O((deg_a+1)(deg_b+1)) —
 * see gf64_poly_mul_internal for why an FFT path is not used here yet.
 */
void gf64_poly_mul(
	gf64_t *out,
	const gf64_t *a,
	size_t deg_a,
	const gf64_t *b,
	size_t deg_b
) {
	assert(out != NULL);
	assert(a != NULL);
	assert(b != NULL);

	gf64_poly_mul_internal(
		out,
		a, (size_t)deg_a + 1U,
		b, (size_t)deg_b + 1U,
		(size_t)deg_a + (size_t)deg_b + 1U
	);
}

/*
 * Pre-allocated-output variant used by Newton's iteration in
 * gf64_poly_invmod.  Multiplies two polynomials (of length len_a and len_b)
 * and writes only the low-order out_len coefficients into caller-owned
 * `out`.  Cost is O(len_a * len_b) schoolbook; future FFT-multiply work
 * (Gao-Mateer tower-of-extensions or subfield NTT) can drop in here.
 *
 * `out` must point to a buffer of size out_len coefficients.
 */
void gf64_poly_mul_padded(
	gf64_t *out,
	const gf64_t *a, size_t len_a,
	const gf64_t *b, size_t len_b,
	size_t out_len
) {
	assert(out != NULL);
	assert(a != NULL);
	assert(b != NULL);
	assert(out_len > 0);

	gf64_poly_mul_internal(out, a, len_a, b, len_b, out_len);
}

/* ===========================================================================
 * T4: AVX-512 vectorized Gao-Mateer-style additive FFT.
 *
 * The recursion structure mirrors the scalar implementation exactly. The
 * element-wise operations are lifted into 8-lane SIMD:
 *   - Frobenius squaring uses gf64_square_avx512 (T2)
 *   - The butterfly's alpha * odd multiplications use gf64_mul_avx512 (T1)
 *
 * Both T1 and T2 are bit-exact to the scalar GF(2^64) reference, so the
 * recursion is bit-exact to gf64_fft_forward / gf64_fft_inverse. The N % 8
 * tail and the small-N (n < 64) inner recursion delegate to the scalar
 * implementation, since SIMD setup overhead dominates for small problems.
 * =========================================================================== */

#include <immintrin.h>

#ifndef __GNUC__
/* Stub out GCC __attribute__((target(...))) under MSVC. */
#define __attribute__(...)
#endif

#define GF64_FFT_AVX512_MIN  ((size_t)64)
#define GF64_FFT_AVX512_LANES ((size_t)8)

/* Apply Frobenius 63 times (the inverse of x -> x^2 in GF(2^64)).
 * SIMD via 63 calls to gf64_square_avx512 — matches the scalar
 * gf64_apply_inverse_frobenius in lockstep, since both delegate to the
 * same bit-exact squaring primitive. The in-place aliasing is safe:
 * gf64_square_avx512 does load -> compute -> store per block. */
static void gf64_apply_inverse_frobenius_avx512(gf64_t *values, size_t n) {
	for (size_t round = 0; round < 63; round++) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
		gf64_square_avx512(values, values, n);
#pragma GCC diagnostic pop
	}
}

/* AVX-512 vectorized in-place Gao-Mateer forward FFT.
 *
 * For n >= GF64_FFT_AVX512_MIN, mirrors the scalar gf64_fft_forward
 * recursion, lifting the element-wise Frobenius squaring into 8-lane
 * SIMD and the alpha * odd butterfly multiplications into 8-lane SIMD.
 * For n < GF64_FFT_AVX512_MIN, delegates to the scalar implementation. */
__attribute__((target("avx512f,vpclmulqdq")))
void gf64_fft_forward_avx512(gf64_t *poly, size_t n) {
	if (n <= 1) {
		return;
	}

	if (n < GF64_FFT_AVX512_MIN) {
		gf64_fft_forward(poly, n);
		return;
	}

	assert(gf64_is_power_of_two(n));

	size_t half = n / 2;
	gf64_t *even_half = (gf64_t *)malloc(half * sizeof(gf64_t));
	gf64_t *odd_half = (gf64_t *)malloc(half * sizeof(gf64_t));
	if (even_half == NULL || odd_half == NULL) {
		free(even_half);
		free(odd_half);
		abort();
	}

	/* Split poly into even/odd halves. The scalar deinterleave is bandwidth-
	 * bound; the SIMD ops (square, mul) below carry the actual compute. */
	for (size_t i = 0; i < half; i++) {
		even_half[i] = poly[2 * i];
		odd_half[i] = poly[2 * i + 1];
	}

	/* Frobenius (square) each half — 8 lanes per iteration via T2.
	 * In-place aliasing is safe: gf64_square_avx512 does load -> compute
	 * -> store per block. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
	gf64_square_avx512(even_half, even_half, half);
	gf64_square_avx512(odd_half, odd_half, half);
#pragma GCC diagnostic pop

	/* Recursive transform on each half — recurses down to n=64 in SIMD,
	 * then drops into scalar for the bottom levels. */
	gf64_fft_forward_avx512(even_half, half);
	gf64_fft_forward_avx512(odd_half, half);

	/* Butterfly combine. For each batch of 8 positions i, alpha is the
	 * 8-element vector {i, i+1, ..., i+7}; one SIMD multiplication per
	 * batch covers all 8 alpha * odd products. */
	gf64_t alpha_buf[GF64_FFT_AVX512_LANES];
	gf64_t prod_buf[GF64_FFT_AVX512_LANES];

	for (size_t i = 0; i < half; i += GF64_FFT_AVX512_LANES) {
		for (size_t k = 0; k < GF64_FFT_AVX512_LANES; k++) {
			alpha_buf[k] = (gf64_t)(i + k);
		}
		gf64_mul_avx512(prod_buf, alpha_buf, odd_half + i, GF64_FFT_AVX512_LANES);

		for (size_t k = 0; k < GF64_FFT_AVX512_LANES; k++) {
			gf64_t beta = even_half[i + k];
			gf64_t gamma = prod_buf[k];
			poly[i + k] = beta ^ gamma;
			poly[i + k + half] = poly[i + k] ^ odd_half[i + k];
		}
	}

	free(even_half);
	free(odd_half);
}

/* AVX-512 vectorized unscaled inverse — counterpart to the static
 * gf64_fft_inverse_unscaled above. File-local (same TU as the scalar
 * version), so it can call gf64_fft_inverse_unscaled directly for the
 * small-N scalar fallback. */
__attribute__((target("avx512f,vpclmulqdq")))
static void gf64_fft_inverse_unscaled_avx512(gf64_t *poly, size_t n) {
	if (n <= 1) {
		return;
	}

	if (n < GF64_FFT_AVX512_MIN) {
		gf64_fft_inverse_unscaled(poly, n);
		return;
	}

	size_t half = n / 2;
	gf64_t *even_half = (gf64_t *)malloc(half * sizeof(gf64_t));
	gf64_t *odd_half = (gf64_t *)malloc(half * sizeof(gf64_t));
	if (even_half == NULL || odd_half == NULL) {
		free(even_half);
		free(odd_half);
		abort();
	}

	/* Step 1: inverse butterfly — split poly into even/odd halves.
	 *
	 * Math: given y_lo = poly[i], y_hi = poly[i + half], the inverse of the
	 * forward butterfly (y_lo = e + i*o, y_hi = y_lo + o) gives
	 *   o = y_lo + y_hi
	 *   e = y_lo + i*o
	 * The product is alpha * o (the SUM, not alpha * y_lo), so we must first
	 * compute o = lo ^ hi, then multiply alpha by o, then xor with lo.
	 * Mirrors the scalar gf64_fft_inverse_unscaled exactly. */
	gf64_t alpha_buf[GF64_FFT_AVX512_LANES];
	gf64_t odd_buf[GF64_FFT_AVX512_LANES];
	gf64_t prod_buf[GF64_FFT_AVX512_LANES];

	for (size_t i = 0; i < half; i += GF64_FFT_AVX512_LANES) {
		for (size_t k = 0; k < GF64_FFT_AVX512_LANES; k++) {
			gf64_t lo = poly[i + k];
			gf64_t odd = lo ^ poly[i + k + half];
			odd_buf[k] = odd;
			odd_half[i + k] = odd;
			alpha_buf[k] = (gf64_t)(i + k);
		}
		gf64_mul_avx512(prod_buf, alpha_buf, odd_buf, GF64_FFT_AVX512_LANES);

		for (size_t k = 0; k < GF64_FFT_AVX512_LANES; k++) {
			gf64_t lo = poly[i + k];
			even_half[i + k] = lo ^ prod_buf[k];
		}
	}

	/* Step 2: recurse on each half. */
	gf64_fft_inverse_unscaled_avx512(even_half, half);
	gf64_fft_inverse_unscaled_avx512(odd_half, half);

	/* Step 3: apply inverse Frobenius (63 squarings) to each half. */
	gf64_apply_inverse_frobenius_avx512(even_half, half);
	gf64_apply_inverse_frobenius_avx512(odd_half, half);

	/* Step 4: interleave. Scalar epilog; bandwidth-bound, the next pass
	 * would SIMDify the deinterleave via vinserti/vpermi but T4 keeps it
	 * simple. */
	for (size_t i = 0; i < half; i++) {
		poly[2 * i] = even_half[i];
		poly[2 * i + 1] = odd_half[i];
	}

	free(even_half);
	free(odd_half);
}

/* Public AVX-512 inverse: unscaled recursion + T3 round-trip scale. */
__attribute__((target("avx512f,vpclmulqdq")))
void gf64_fft_inverse_avx512(gf64_t *poly, size_t n) {
	if (n <= 1) {
		return;
	}
	gf64_fft_inverse_unscaled_avx512(poly, n);
	gf64_scale_vector(poly, n, gf64_fft_scale(n));
}

HEDLEY_END_C_DECLS
