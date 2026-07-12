/*
 * ============================================================================
 * gf64/test/test_gf64_poly_invmod_fft.c
 *
 * PR-1 of the par3-cauchy-fft-kernel T8 follow-up:
 *
 *   - gf64_poly_mul       : cleaned up to share a private core with the
 *                           padded-output variant (bit-exact with the
 *                           pre-PR-1 schoolbook fallback — see the doc on
 *                           gf64_poly_mul_internal in gf64_additive_fft.c
 *                           for why the additive FFT does not implement
 *                           the convolution theorem directly in GF(2^64)).
 *   - gf64_poly_mul_padded: new pre-allocated-output entry point used by
 *                           the Newton iteration in gf64_poly_invmod.
 *   - gf64_poly_invmod    : Newton iteration now delegates both the squaring
 *                           and the multiplication to gf64_poly_mul_padded,
 *                           giving the routine a stable signature for the
 *                           future FFT-multiply drop-in.
 *
 * This test exercises the cleaned-up contract end-to-end:
 *   Test B: gf64_poly_mul_padded matches gf64_poly_mul for the case where
 *           the padded output is the full convolution (out_len ==
 *           deg_a + deg_b + 1).
 *   Test C: gf64_poly_mul_padded produces the correct low-order truncation
 *           when out_len < deg_a + deg_b + 1 (the Newton iteration's typical
 *           use pattern).
 *   Test D: gf64_poly_invmod invariant g * inv ≡ 1 (mod x^n) holds across
 *           a wider n grid than the legacy test (1 .. 256, stepping by
 *           powers of 2 and odd sizes), using g * inv reconstructed with
 *           gf64_poly_mul.
 *   Test E: g * inv ≡ 1 (mod x^n) when reconstructed with gf64_poly_mul_padded
 *           (the path PR-1 promotes into the Newton iteration itself).
 *
 * Build & run from gf64/test/:
 *   $(CC) -O2 -march=native -I.. test_gf64_poly_invmod_fft.c \
 *         ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c \
 *         ../gf64_additive_fft.c ../gf64_mul_avx512.c \
 *         ../gf64_square.c ../gf64_single.c \
 *         -o test_gf64_poly_invmod_fft && ./test_gf64_poly_invmod_fft
 *
 * Exits 0 on full pass; non-zero on any failure.
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../gf64_mpe.h"
#include "../gf64_invert_ita.h"
#include "../gf64_additive_fft.h"
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

static uint64_t g_rng = 0xC0DEFACE7E51C0DEULL;

static uint64_t splitmix64_next(void) {
	uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static void put_seed(uint64_t s) {
	g_rng = s;
}

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

/* Reference scalar convolution. Returns deg_a + deg_b + 1 coefficients. */
static void schoolbook_poly_mul(
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

/* ----------------------------------------------------------------------------
 * Test A: gf64_poly_mul == schoolbook on random coefficients at three
 *         degree-ratio regimes:
 *         - tiny_a_tiny_b: deg_a=2, deg_b=3
 *         - balanced:      deg_a=15, deg_b=15
 *         - imbalanced:    deg_a=2, deg_b=63 (the next pow-2 padded length is
 *                            larger than deg_a + deg_b + 1)
 *         - imbalanced2:   deg_a=63, deg_b=63 (next pow-2 = 128, force
 *                            FFT round-up)
 *         - larger:        deg_a=255, deg_b=127
 * ---------------------------------------------------------------------------- */
static void test_poly_mul_vs_schoolbook(void) {
	printf("Test A: gf64_poly_mul vs schoolbook at varied (len_a, len_b)\n");

	struct {
		const char *name;
		size_t len_a, len_b;
		uint64_t seed;
	} cases[] = {
		{"tiny (2 x 3)",       3,  4,  0xA1A2A3A4A5A6A7A8ULL},
		{"balanced (16 x 16)", 16, 16, 0xB1B2B3B4B5B6B7B8ULL},
		{"imbalanced (3 x 64)",3,  64, 0xC1C2C3C4C5C6C7C8ULL},
		{"pow2 round (64 x 64)",64, 64, 0xD1D2D3D4D5D6D7D8ULL},
		{"larger (256 x 128)", 256, 128, 0xE1E2E3E4E5E6E7E8ULL},
	};

	int all_ok = 1;
	for (size_t k = 0; k < sizeof(cases)/sizeof(cases[0]); k++) {
		const size_t len_a = cases[k].len_a;
		const size_t len_b = cases[k].len_b;
		const size_t out_len = len_a + len_b - 1;

		gf64_t *a = (gf64_t *)malloc(len_a * sizeof(gf64_t));
		gf64_t *b = (gf64_t *)malloc(len_b * sizeof(gf64_t));
		gf64_t *out_fft = (gf64_t *)calloc(out_len, sizeof(gf64_t));
		gf64_t *out_ref = (gf64_t *)calloc(out_len, sizeof(gf64_t));

		put_seed(cases[k].seed);
		for (size_t i = 0; i < len_a; i++) a[i] = splitmix64_next();
		for (size_t j = 0; j < len_b; j++) b[j] = splitmix64_next();

		gf64_poly_mul(out_fft, a, len_a - 1, b, len_b - 1);
		schoolbook_poly_mul(out_ref, a, len_a, b, len_b);

		int ok = 1;
		int first = -1;
		for (size_t i = 0; i < out_len; i++) {
			if (out_fft[i] != out_ref[i]) {
				if (first < 0) first = (int)i;
				ok = 0;
			}
		}
		if (ok) {
			pass(cases[k].name);
		} else {
			printf("    %s: first mismatch i=%d fft=0x%016llx ref=0x%016llx\n",
			       cases[k].name, first,
			       (unsigned long long)out_fft[first],
			       (unsigned long long)out_ref[first]);
			fail(cases[k].name);
			all_ok = 0;
		}

		free(a); free(b); free(out_fft); free(out_ref);
	}
	(void)all_ok;
}

/* ----------------------------------------------------------------------------
 * Test B: gf64_poly_mul_padded(out_len = full conv) == gf64_poly_mul.
 * ---------------------------------------------------------------------------- */
static void test_poly_mul_padded_matches_poly_mul(void) {
	printf("Test B: gf64_poly_mul_padded (full conv) == gf64_poly_mul\n");

	struct { size_t la, lb; uint64_t seed; } cases[] = {
		{1, 1, 0x1010101010101010ULL},
		{3, 5, 0x2020202020202020ULL},
		{16, 17, 0x3030303030303030ULL},
		{64, 64, 0x4040404040404040ULL},
		{256, 64, 0x5050505050505050ULL},
	};

	for (size_t k = 0; k < sizeof(cases)/sizeof(cases[0]); k++) {
		const size_t la = cases[k].la;
		const size_t lb = cases[k].lb;
		const size_t out_len = la + lb - 1;
		gf64_t *a = (gf64_t *)malloc(la * sizeof(gf64_t));
		gf64_t *b = (gf64_t *)malloc(lb * sizeof(gf64_t));
		gf64_t *r1 = (gf64_t *)calloc(out_len, sizeof(gf64_t));
		gf64_t *r2 = (gf64_t *)calloc(out_len, sizeof(gf64_t));

		put_seed(cases[k].seed);
		for (size_t i = 0; i < la; i++) a[i] = splitmix64_next();
		for (size_t j = 0; j < lb; j++) b[j] = splitmix64_next();

		gf64_poly_mul(r1, a, la - 1, b, lb - 1);
		gf64_poly_mul_padded(r2, a, la, b, lb, out_len);

		int ok = 1;
		for (size_t i = 0; i < out_len; i++) {
			if (r1[i] != r2[i]) { ok = 0; break; }
		}
		if (ok) {
			char name[64];
			snprintf(name, sizeof(name), "(la=%zu, lb=%zu)", la, lb);
			pass(name);
		} else {
			fail("padded vs unpadded");
		}

		free(a); free(b); free(r1); free(r2);
	}
}

/* ----------------------------------------------------------------------------
 * Test C: gf64_poly_mul_padded(out_len < full) returns the correct low-order
 *         coefficients (the Newton iteration's typical usage).
 * ---------------------------------------------------------------------------- */
static void test_poly_mul_padded_truncation(void) {
	printf("Test C: gf64_poly_mul_padded truncated output\n");

	const size_t la = 9, lb = 9;  /* full conv = 17 coefficients */
	const size_t full = la + lb - 1;
	gf64_t *a = (gf64_t *)malloc(la * sizeof(gf64_t));
	gf64_t *b = (gf64_t *)malloc(lb * sizeof(gf64_t));
	gf64_t *full_conv = (gf64_t *)calloc(full, sizeof(gf64_t));
	gf64_t *trunc = (gf64_t *)calloc(full, sizeof(gf64_t));

	put_seed(0x7172737475767778ULL);
	for (size_t i = 0; i < la; i++) a[i] = splitmix64_next();
	for (size_t j = 0; j < lb; j++) b[j] = splitmix64_next();

	schoolbook_poly_mul(full_conv, a, la, b, lb);

	int all_ok = 1;
	size_t cuts[] = { 1, 2, 4, 7, 16, 17 };
	for (size_t c = 0; c < sizeof(cuts)/sizeof(cuts[0]); c++) {
		size_t out_len = cuts[c];
		memset(trunc, 0, full * sizeof(gf64_t));
		gf64_poly_mul_padded(trunc, a, la, b, lb, out_len);

		int ok = 1;
		for (size_t i = 0; i < out_len; i++) {
			if (trunc[i] != full_conv[i]) { ok = 0; break; }
		}
		for (size_t i = out_len; i < full; i++) {
			if (trunc[i] != 0) { ok = 0; break; } /* deterministic zero */
		}
		if (!ok) {
			printf("    out_len=%zu mismatch\n", out_len);
			all_ok = 0;
			fail("truncation correctness");
		}
	}
	if (all_ok) pass("truncated low-order coefficients match schoolbook");

	free(a); free(b); free(full_conv); free(trunc);
}

/* ----------------------------------------------------------------------------
 * Test D: gf64_poly_invmod invariant g * inv_g ≡ 1 (mod x^n) at a wider
 *         grid than test_gf64_mpe.c, using gf64_poly_mul to build the
 *         product.
 * ---------------------------------------------------------------------------- */
static void test_invmod_wide_grid_via_poly_mul(void) {
	printf("Test D: gf64_poly_invmod invariant via gf64_poly_mul (wider grid)\n");

	const size_t n_cases[] = {
		1, 2, 3, 7, 8, 16, 32, 64, 100, 128, 200, 256
	};
	const size_t num_n = sizeof(n_cases) / sizeof(n_cases[0]);

	for (size_t ci = 0; ci < num_n; ci++) {
		const size_t n = n_cases[ci];
		const size_t deg_g = (n < 32) ? 7 : 31;

		gf64_t *g     = (gf64_t *)malloc((deg_g + 1) * sizeof(gf64_t));
		gf64_t *inv   = (gf64_t *)calloc(n,                 sizeof(gf64_t));
		/* The product has up to (deg_g + n) coefficients. */
		gf64_t *prod  = (gf64_t *)calloc(deg_g + n + 1,     sizeof(gf64_t));

		if (!g || !inv || !prod) {
			free(g); free(inv); free(prod);
			fail("alloc");
			continue;
		}

		put_seed(0xD0D0D0D0D0D0D0D0ULL ^ (uint64_t)ci);
		for (size_t i = 0; i <= deg_g; i++) g[i] = splitmix64_next();
		if (g[0] == 0) g[0] = 1ULL;

		gf64_poly_invmod(g, deg_g, n, inv);
		gf64_poly_mul(prod, g, deg_g, inv, n - 1);

		int ok = 1;
		if (prod[0] != 1ULL) ok = 0;
		for (size_t k = 1; k < n; k++) {
			if (prod[k] != 0) { ok = 0; break; }
		}
		char name[96];
		if (ok) {
			snprintf(name, sizeof(name),
			         "invmod (n=%zu, deg_g=%zu) via gf64_poly_mul", n, deg_g);
			pass(name);
		} else {
			snprintf(name, sizeof(name),
			         "invmod (n=%zu, deg_g=%zu) via gf64_poly_mul", n, deg_g);
			fail(name);
		}
		free(g); free(inv); free(prod);
	}
}

/* ----------------------------------------------------------------------------
 * Test E: same invariant but reconstructed via gf64_poly_mul_padded (the
 *         path the Newton iteration itself takes internally).
 * ---------------------------------------------------------------------------- */
static void test_invmod_via_poly_mul_padded(void) {
	printf("Test E: gf64_poly_invmod invariant via gf64_poly_mul_padded\n");

	const size_t n_cases[] = { 1, 2, 4, 8, 16, 32, 64, 128 };
	const size_t num_n = sizeof(n_cases) / sizeof(n_cases[0]);

	for (size_t ci = 0; ci < num_n; ci++) {
		const size_t n = n_cases[ci];
		const size_t deg_g = (n < 8) ? 3 : 7;

		gf64_t *g     = (gf64_t *)malloc((deg_g + 1) * sizeof(gf64_t));
		gf64_t *inv   = (gf64_t *)calloc(n,                 sizeof(gf64_t));
		gf64_t *prod  = (gf64_t *)calloc(deg_g + n + 1,     sizeof(gf64_t));

		put_seed(0xE0E0E0E0E0E0E0E0ULL ^ (uint64_t)ci);
		for (size_t i = 0; i <= deg_g; i++) g[i] = splitmix64_next();
		if (g[0] == 0) g[0] = 1ULL;

		gf64_poly_invmod(g, deg_g, n, inv);
		gf64_poly_mul_padded(prod, g, deg_g + 1, inv, n, n);

		int ok = 1;
		if (prod[0] != 1ULL) ok = 0;
		for (size_t k = 1; k < n; k++) {
			if (prod[k] != 0) { ok = 0; break; }
		}
		char name[96];
		if (ok) {
			snprintf(name, sizeof(name),
			         "invmod (n=%zu, deg_g=%zu) via padded", n, deg_g);
			pass(name);
		} else {
			snprintf(name, sizeof(name),
			         "invmod (n=%zu, deg_g=%zu) via padded", n, deg_g);
			fail(name);
		}
		free(g); free(inv); free(prod);
	}
}

int main(void) {
	printf("PR-1 parity tests: gf64_poly_mul(FFT), gf64_poly_mul_padded, gf64_poly_invmod\n");
	printf("================================================================================\n\n");

	test_poly_mul_vs_schoolbook();
	test_poly_mul_padded_matches_poly_mul();
	test_poly_mul_padded_truncation();
	test_invmod_wide_grid_via_poly_mul();
	test_invmod_via_poly_mul_padded();

	printf("\n=== Summary ===\n");
	printf("Passed: %d\n", g_passed);
	printf("Failed: %d\n", g_failed);
	return g_failed > 0 ? 1 : 0;
}
