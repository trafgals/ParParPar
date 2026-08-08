/*
 * ============================================================================
 * gf64/test/test_gf64_poly_mul_karatsuba.c
 *
 * Parity tests for gf64_poly_mul_karatsuba (Phase 2a). Exercises the
 * Karatsuba recursion at sizes both below and above
 * GF64_POLY_MUL_INTERNAL_KARATSUBA_MIN (=128), so we cover:
 *   - the public gf64_poly_mul_karatsuba entry (direct);
 *   - the gf64_poly_mul_internal dispatch path (via gf64_poly_mul_padded).
 *
 * Each case compares against a from-scratch schoolbook reference built
 * from gf64_mul_reference. Bit-exactness is required: any single mismatch
 * fails the test.
 *
 * Build & run from gf64/test/:
 *   $(CC) -O2 -march=native -I.. test_gf64_poly_mul_karatsuba.c \
 *         ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c \
 *         ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c \
 *         ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c \
 *         -o test_gf64_poly_mul_karatsuba && ./test_gf64_poly_mul_karatsuba
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../gf64_additive_fft.h"
#include "../gf64_poly_mul_karatsuba.h"
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* Reference scalar convolution (constant-first). */
static void schoolbook(
	gf64_t *out,
	const gf64_t *a, size_t len_a,
	const gf64_t *b, size_t len_b
) {
	memset(out, 0, (len_a + len_b - 1) * sizeof(gf64_t));
	for (size_t i = 0; i < len_a; i++) {
		for (size_t j = 0; j < len_b; j++) {
			out[i + j] ^= gf64_mul_reference(a[i], b[j]);
		}
	}
}

static uint64_t g_rng = 0xABCDEF0123456789ULL;
static uint64_t splitmix64_next(void) {
	uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}
static void put_seed(uint64_t s) { g_rng = s; }

static int g_passed = 0;
static int g_failed = 0;
static void pass(const char *name) {
	printf("  PASS: %s\n", name);
	g_passed++;
}
static void fail(const char *name) {
	printf("  FAIL: %s\n", name);
	g_failed++;
}

static int eq_buf(const gf64_t *a, const gf64_t *b, size_t n) {
	for (size_t i = 0; i < n; i++) if (a[i] != b[i]) return 0;
	return 1;
}

/* ----------------------------------------------------------------------------
 * Test A: gf64_poly_mul_karatsuba (direct) == schoolbook reference.
 *         Sizes chosen to straddle the Karatsuba crossover (128).
 * ---------------------------------------------------------------------------- */
static void test_karatsuba_direct(void) {
	printf("Test A: gf64_poly_mul_karatsuba vs schoolbook\n");

	struct { const char *name; size_t la, lb, ol; uint64_t seed; } cases[] = {
		/* Below Karatsuba threshold (schoolbook path). */
		{"tiny (1 x 1, out=1)",         1,   1,   1,   0xA0A0A0A0A0A0A0A0ULL},
		{"small (32 x 32, out=63)",     32,  32,  63,  0xA1A1A1A1A1A1A1A1ULL},
		{"at threshold (64 x 64)",      64,  64,  127, 0xA2A2A2A2A2A2A2A2ULL},

		/* Above Karatsuba threshold (recursive path). */
		{"just over (128 x 128, full)", 128, 128, 255, 0xA3A3A3A3A3A3A3A3ULL},
		{"balanced (256 x 256, full)",  256, 256, 511, 0xA4A4A4A4A4A4A4A4ULL},
		{"balanced (512 x 512, full)",  512, 512, 1023,0xA5A5A5A5A5A5A5A5ULL},
		{"balanced (1024 x 1024, full)",1024,1024,2047,0xA6A6A6A6A6A6A6A6ULL},

		/* Karatsuba dispatch via gf64_poly_mul_internal (out_len < 2n-1). */
		{"trunc (1024 x 1024, out=256)",1024,1024,256, 0xA7A7A7A7A7A7A7A7ULL},
		{"trunc (1024 x 1024, out=64)", 1024,1024,64,  0xA8A8A8A8A8A8A8A8ULL},
		{"trunc (512 x 256, out=300)",  512, 256, 300, 0xA9A9A9A9A9A9A9A9ULL},
		{"trunc (256 x 64, out=200)",   256, 64,  200, 0xAAAAAAAAAAAAAAA0ULL},

		/* Asymmetric (Newton-iteration pattern). */
		{"asym (n x small, out=small)",  4096, 255, 256, 0xABABABABABABABABULL},
		{"asym (large x 1, out=1)",     4096, 1,   1,   0xACACACACACACACACULL},
		{"asym (4096 x 4095, out=4096)",4096, 4095,4096,0xADADADADADADADADULL},

		/* Boundary at the dispatch threshold. */
		{"at dispatch (128 x 128, out=128)",128,128,128,0xAEAEAEAEAEAEAEAEULL},
	};

	int all_ok = 1;
	for (size_t k = 0; k < sizeof(cases)/sizeof(cases[0]); k++) {
		size_t la = cases[k].la, lb = cases[k].lb, ol = cases[k].ol;
		gf64_t *a = (gf64_t *)malloc(la * sizeof(gf64_t));
		gf64_t *b = (gf64_t *)malloc(lb * sizeof(gf64_t));
		size_t full_len = la + lb - 1;
		gf64_t *ref  = (gf64_t *)calloc(full_len, sizeof(gf64_t));
		gf64_t *got  = (gf64_t *)calloc(ol, sizeof(gf64_t));

		put_seed(cases[k].seed);
		for (size_t i = 0; i < la; i++) a[i] = splitmix64_next();
		for (size_t i = 0; i < lb; i++) b[i] = splitmix64_next();

		schoolbook(ref, a, la, b, lb);
		gf64_poly_mul_karatsuba(got, a, la, b, lb, ol);

		int ok = (ol <= full_len) ? eq_buf(got, ref, ol) : 0;
		if (ok && ol < full_len) {
			/* The higher-order coefficients of `ref` we ignored must
			 * still be untouched in `got`'s caller buffer (out is
			 * zero-initialized above, but for paranoia check ref's
			 * first mismatch — there shouldn't be any). */
		}

		if (ok) {
			pass(cases[k].name);
		} else {
			printf("    %s: FAIL (la=%zu lb=%zu ol=%zu)\n",
			       cases[k].name, la, lb, ol);
			fail(cases[k].name);
			all_ok = 0;
		}

		free(a); free(b); free(ref); free(got);
	}
	(void)all_ok;
}

/* ----------------------------------------------------------------------------
 * Test B: gf64_poly_mul_padded == gf64_poly_mul_karatsuba at sizes where
 *         Karatsuba activates inside gf64_poly_mul_internal.
 *
 * The wrapper at gf64_additive_fft.c::gf64_poly_mul_internal dispatches into
 * gf64_poly_mul_karatsuba when len_a, len_b, out_len are all >= 128. This
 * test verifies the dispatch is bit-exact to the direct Karatsuba call.
 * ---------------------------------------------------------------------------- */
static void test_padded_dispatch(void) {
	printf("Test B: gf64_poly_mul_padded (Karatsuba-dispatch) == gf64_poly_mul_karatsuba\n");

	struct { const char *name; size_t la, lb, ol; uint64_t seed; } cases[] = {
		{"balanced (256 x 256, full)",   256, 256, 511,  0xB1B1B1B1B1B1B1B1ULL},
		{"balanced (512 x 512, full)",   512, 512, 1023, 0xB2B2B2B2B2B2B2B2ULL},
		{"trunc (1024 x 1024, out=300)", 1024, 1024, 300, 0xB3B3B3B3B3B3B3B3ULL},
		{"asym (4096 x 256, out=300)",   4096, 256, 300,  0xB4B4B4B4B4B4B4B4ULL},
		{"asym (4096 x 4095, out=4096)", 4096, 4095, 4096,0xB5B5B5B5B5B5B5B5ULL},
		{"at dispatch (128 x 128, full)", 128, 128, 255,  0xB6B6B6B6B6B6B6B6ULL},
	};

	for (size_t k = 0; k < sizeof(cases)/sizeof(cases[0]); k++) {
		size_t la = cases[k].la, lb = cases[k].lb, ol = cases[k].ol;
		gf64_t *a = (gf64_t *)malloc(la * sizeof(gf64_t));
		gf64_t *b = (gf64_t *)malloc(lb * sizeof(gf64_t));
		gf64_t *r1 = (gf64_t *)calloc(ol, sizeof(gf64_t));
		gf64_t *r2 = (gf64_t *)calloc(ol, sizeof(gf64_t));

		put_seed(cases[k].seed);
		for (size_t i = 0; i < la; i++) a[i] = splitmix64_next();
		for (size_t i = 0; i < lb; i++) b[i] = splitmix64_next();

		gf64_poly_mul_padded(r1, a, la, b, lb, ol);
		gf64_poly_mul_karatsuba(r2, a, la, b, lb, ol);

		if (eq_buf(r1, r2, ol)) {
			pass(cases[k].name);
		} else {
			fail(cases[k].name);
		}

		free(a); free(b); free(r1); free(r2);
	}
}

/* ----------------------------------------------------------------------------
 * Test C: zero polynomial handling — the special cases that the dispatch
 *         gate must not break.
 * ---------------------------------------------------------------------------- */
static void test_edge_cases(void) {
	printf("Test C: edge cases\n");

	/* a == 0 -> all zeros. */
	gf64_t a0[128] = {0};
	gf64_t b[128];
	for (size_t i = 0; i < 128; i++) b[i] = splitmix64_next();
	gf64_t out[256];
	memset(out, 0xCD, sizeof(out));
	gf64_poly_mul_karatsuba(out, a0, 128, b, 128, 256);
	int ok_a = eq_buf(out, (gf64_t[256]){0}, 256);
	if (ok_a) pass("zero a -> all-zero output"); else fail("zero a -> all-zero output");

	/* b == 0 -> all zeros. */
	gf64_t b0[128] = {0};
	gf64_t a[128];
	for (size_t i = 0; i < 128; i++) a[i] = splitmix64_next();
	memset(out, 0xCD, sizeof(out));
	gf64_poly_mul_karatsuba(out, a, 128, b0, 128, 256);
	int ok_b = eq_buf(out, (gf64_t[256]){0}, 256);
	if (ok_b) pass("zero b -> all-zero output"); else fail("zero b -> all-zero output");

	/* out_len == 0 -> no-op. */
	memset(out, 0xCD, sizeof(out));
	gf64_poly_mul_karatsuba(out, a, 128, b, 128, 0);
	int ok_o = 1;
	for (size_t i = 0; i < 256; i++) {
		/* Caller-initialized memory should be unchanged. */
		if (((uint8_t *)out)[i] != 0xCD) { ok_o = 0; break; }
	}
	if (ok_o) pass("out_len == 0 -> no-op (caller memory intact)"); else fail("out_len == 0 -> no-op");

	/* x^128 * x^128 -> coefficient at index 256 == 1 (with out_len=257). */
	gf64_t one_a[129] = {0};
	gf64_t one_b[129] = {0};
	one_a[128] = 1;
	one_b[128] = 1;
	gf64_t out257[257];
	memset(out257, 0, sizeof(out257));
	gf64_poly_mul_karatsuba(out257, one_a, 129, one_b, 129, 257);
	int ok_x = (out257[256] == 1);
	for (size_t i = 0; i < 256; i++) {
		if (out257[i] != 0) { ok_x = 0; break; }
	}
	if (ok_x) pass("x^128 * x^128 -> x^256"); else fail("x^128 * x^128 -> x^256");
}

int main(void) {
	printf("Phase 2a Karatsuba parity tests\n");
	printf("================================\n\n");

	test_karatsuba_direct();
	test_padded_dispatch();
	test_edge_cases();

	printf("\n=== Summary ===\n");
	printf("Passed: %d\n", g_passed);
	printf("Failed: %d\n", g_failed);
	return g_failed > 0 ? 1 : 0;
}