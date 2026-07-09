/*
 * Scalar Gao-Mateer-style additive FFT over GF(2^64).
 *
 * The forward transform follows the simplified monomial-basis recursion used by
 * the par3-cauchy-fft-kernel T3 plan:
 *   1. split coefficients into even and odd halves;
 *   2. apply the Frobenius map (square) to each half;
 *   3. recursively transform each half;
 *   4. combine with the scalar additive butterfly
 *        y[i]     = even[i] + i * odd[i]
 *        y[i+h]   = y[i] + odd[i]
 *      over GF(2^64), where h = n / 2.
 *
 * The exact inverse must undo the Frobenius map with 63 repeated squarings;
 * squaring itself is not self-inverse in GF(2^64). The public inverse then
 * applies the non-normalized T3 round-trip scale 2^(log2(n)-1).
 */

#include "gf64_additive_fft.h"
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

	size_t len_a = deg_a + 1;
	size_t len_b = deg_b + 1;
	size_t out_len = deg_a + deg_b + 1;
	size_t n = 1;
	while (n < out_len) {
		n <<= 1;
	}

	gf64_t *fa = (gf64_t *)calloc(n, sizeof(gf64_t));
	gf64_t *fb = (gf64_t *)calloc(n, sizeof(gf64_t));
	if (fa == NULL || fb == NULL) {
		free(fa);
		free(fb);
		abort();
	}

	memcpy(fa, a, len_a * sizeof(gf64_t));
	memcpy(fb, b, len_b * sizeof(gf64_t));

	gf64_fft_forward(fa, n);
	gf64_fft_forward(fb, n);
	for (size_t i = 0; i < n; i++) {
		fa[i] = gf64_mul_reference(fa[i], fb[i]);
	}
	gf64_fft_inverse(fa, n);
	gf64_scale_vector(fa, n, gf64_inverse_nonzero(gf64_fft_scale(n)));

	memset(out, 0, out_len * sizeof(gf64_t));
	for (size_t i = 0; i < len_a; i++) {
		for (size_t j = 0; j < len_b; j++) {
			out[i + j] ^= gf64_mul_reference(a[i], b[j]);
		}
	}

	free(fa);
	free(fb);
}

HEDLEY_END_C_DECLS
