#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../gf64_additive_fft.h"
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

static int g_passed = 0;
static int g_failed = 0;
static uint64_t g_rng = 0;

static uint64_t splitmix64_next(void) {
	uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static void pass(const char *name) {
	printf("  PASS: %s\n", name);
	g_passed++;
}

static void fail(const char *name) {
	printf("  FAIL: %s\n", name);
	g_failed++;
}

static gf64_t roundtrip_scale(unsigned int k) {
	return (gf64_t)1ULL << (k - 1);
}

static int vectors_match(const gf64_t *got, const gf64_t *want, size_t n, size_t *bad_index) {
	for (size_t i = 0; i < n; i++) {
		if (got[i] != want[i]) {
			*bad_index = i;
			return 0;
		}
	}
	return 1;
}

static void test_round_trip_scaled(void) {
	printf("Test: scaled round-trip for n=2^k, k=1..10, 100 trials...\n");
	for (unsigned int k = 1; k <= 10; k++) {
		size_t n = (size_t)1U << k;
		gf64_t *orig = (gf64_t *)malloc(n * sizeof(gf64_t));
		gf64_t *work = (gf64_t *)malloc(n * sizeof(gf64_t));
		gf64_t *want = (gf64_t *)malloc(n * sizeof(gf64_t));
		if (orig == NULL || work == NULL || want == NULL) {
			free(orig);
			free(work);
			free(want);
			fail("round-trip malloc");
			return;
		}

		int ok = 1;
		g_rng = 0xA6D3D0E55EED0000ULL ^ k;
		for (unsigned int trial = 0; trial < 100 && ok; trial++) {
			for (size_t i = 0; i < n; i++) {
				orig[i] = splitmix64_next();
				work[i] = orig[i];
			}

			gf64_fft_forward(work, n);
			gf64_fft_inverse(work, n);

			gf64_t scale = roundtrip_scale(k);
			for (size_t i = 0; i < n; i++) {
				want[i] = gf64_mul_reference(orig[i], scale);
			}

			size_t bad = 0;
			if (!vectors_match(work, want, n, &bad)) {
				printf("    k=%u trial=%u i=%zu got=0x%016llx want=0x%016llx\n",
				       k, trial, bad,
				       (unsigned long long)work[bad],
				       (unsigned long long)want[bad]);
				ok = 0;
			}
		}

		if (ok) {
			char name[64];
			snprintf(name, sizeof(name), "round-trip n=%zu", n);
			pass(name);
		} else {
			char name[64];
			snprintf(name, sizeof(name), "round-trip n=%zu", n);
			fail(name);
		}

		free(orig);
		free(work);
		free(want);
	}
}

static void test_constant_identity(void) {
	printf("Test: forward FFT of constant-one polynomial...\n");
	const size_t n = 1024;
	gf64_t *poly = (gf64_t *)calloc(n, sizeof(gf64_t));
	if (poly == NULL) {
		fail("constant identity malloc");
		return;
	}
	poly[0] = 1;
	gf64_fft_forward(poly, n);

	int ok = 1;
	for (size_t i = 0; i < n; i++) {
		if (poly[i] != 1) {
			printf("    i=%zu got=0x%016llx\n", i, (unsigned long long)poly[i]);
			ok = 0;
			break;
		}
	}
	if (ok) pass("constant-one maps to all ones"); else fail("constant-one maps to all ones");
	free(poly);
}

static void test_zero_forward(void) {
	printf("Test: forward FFT of zero polynomial...\n");
	const size_t n = 1024;
	gf64_t *poly = (gf64_t *)calloc(n, sizeof(gf64_t));
	if (poly == NULL) {
		fail("zero malloc");
		return;
	}
	gf64_fft_forward(poly, n);

	int ok = 1;
	for (size_t i = 0; i < n; i++) {
		if (poly[i] != 0) {
			printf("    i=%zu got=0x%016llx\n", i, (unsigned long long)poly[i]);
			ok = 0;
			break;
		}
	}
	if (ok) pass("zero maps to all zeros"); else fail("zero maps to all zeros");
	free(poly);
}

static void naive_poly_mul(gf64_t *out, const gf64_t *a, size_t deg_a, const gf64_t *b, size_t deg_b) {
	size_t out_len = deg_a + deg_b + 1;
	memset(out, 0, out_len * sizeof(gf64_t));
	for (size_t i = 0; i <= deg_a; i++) {
		for (size_t j = 0; j <= deg_b; j++) {
			out[i + j] ^= gf64_mul_reference(a[i], b[j]);
		}
	}
}

static void test_polynomial_multiplication(void) {
	printf("Test: polynomial multiplication for degrees < 32...\n");
	gf64_t a[32];
	gf64_t b[32];
	gf64_t got[63];
	gf64_t want[63];
	g_rng = 0xF17AADD171C0FFEEULL;

	for (size_t deg_a = 0; deg_a < 32; deg_a++) {
		for (size_t deg_b = 0; deg_b < 32; deg_b++) {
			for (size_t i = 0; i <= deg_a; i++) a[i] = splitmix64_next();
			for (size_t i = 0; i <= deg_b; i++) b[i] = splitmix64_next();

			gf64_poly_mul(got, a, deg_a, b, deg_b);
			naive_poly_mul(want, a, deg_a, b, deg_b);

			size_t bad = 0;
			if (!vectors_match(got, want, deg_a + deg_b + 1, &bad)) {
				printf("    deg_a=%zu deg_b=%zu i=%zu got=0x%016llx want=0x%016llx\n",
				       deg_a, deg_b, bad,
				       (unsigned long long)got[bad],
				       (unsigned long long)want[bad]);
				fail("polynomial multiplication");
				return;
			}
		}
	}
	pass("polynomial multiplication");
}

int main(void) {
	printf("GF64 additive FFT scalar tests\n");
	test_round_trip_scaled();
	test_constant_identity();
	test_zero_forward();
	test_polynomial_multiplication();

	printf("\nSummary: %d passed, %d failed\n", g_passed, g_failed);
	return g_failed == 0 ? 0 : 1;
}
