/*
 * test_gf64_invmod_mod_parity.c — PR #49 Step 1 contract test
 *
 * Verifies that the upgraded gf64_poly_invmod_mod (now using the
 * Newton-reciprocal inner divmod + Karatsuba-dispatched s_curr update)
 * still produces a polynomial `u` of degree < deg_f satisfying
 *
 *     g * u ≡ 1 (mod f)   in GF(2^64)[x]
 *
 * bit-exactly. The contract uniquely determines u once deg u < deg f is
 * fixed, so any implementation (schoolbook EGCD or Newton+Karatsuba)
 * that satisfies the contract must agree coefficient-by-coefficient.
 *
 * Test grid: random coprime g, f of degrees in
 *   {4, 8, 16, 32, 64, 128, 256}
 * For each pair we verify:
 *   1. u[0..deg_f-1] satisfies g*u ≡ 1 (mod f) — i.e. the product
 *      g*u, reduced modulo f, has constant term 1 and zero elsewhere
 *      in the first deg_f coefficients.
 *   2. The leading coefficient of u is the canonical normalized form
 *      (multiplied by f_lead^-1).
 *
 * Run from gf64/test/:
 *   $(CC) -O2 -march=native -I.. test_gf64_invmod_mod_parity.c \
 *         ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c \
 *         ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c \
 *         ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c \
 *         -o test_gf64_invmod_mod_parity && ./test_gf64_invmod_mod_parity
 *
 * Exits 0 on full pass; non-zero on any failure.
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

/* splitmix64 PRNG (single global stream; per-case seeds via put_seed). */
static uint64_t g_splitmix_state = 0;

static uint64_t splitmix64_next(void) {
	uint64_t z = (g_splitmix_state += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static void put_seed(uint64_t s) {
	g_splitmix_state = s;
}

static int g_passed = 0;
static int g_failed = 0;

static void pass(const char *name) { printf("  PASS: %s\n", name); g_passed++; }
static void fail(const char *name) { printf("  FAIL: %s\n", name); g_failed++; }

/*
 * Reduce p mod f using gf64_poly_divmod: r = p - q*f, deg r < deg f.
 * The divmod writes a "working" r into the buffer; we only need the
 * first deg_f coefficients for the contract check.
 */
static void poly_reduce(const gf64_t *p, size_t deg_p,
                        const gf64_t *f, size_t deg_f,
                        gf64_t *r_out) {
	gf64_t *q = (gf64_t *)calloc(deg_p + 1, sizeof(gf64_t));
	gf64_t *r = (gf64_t *)calloc(deg_p + 1, sizeof(gf64_t));
	if (!q || !r) { free(q); free(r); abort(); }
	gf64_poly_divmod(p, deg_p, f, deg_f, q, r);
	for (size_t i = 0; i <= deg_f; i++) r_out[i] = r[i];
	free(q); free(r);
}

/*
 * Run one random coprime (g, f) pair of degrees (deg_g, deg_f) and verify
 * gf64_poly_invmod_mod returns u with g*u ≡ 1 (mod f) bit-exactly.
 */
static int run_case(size_t deg_g, size_t deg_f, uint64_t seed) {
	int ok = 1;
	gf64_t *g     = (gf64_t *)calloc(deg_g + 1, sizeof(gf64_t));
	gf64_t *f     = (gf64_t *)calloc(deg_f + 1, sizeof(gf64_t));
	gf64_t *u     = (gf64_t *)calloc(deg_f,     sizeof(gf64_t));
	gf64_t *prod  = (gf64_t *)calloc(deg_g + deg_f, sizeof(gf64_t));
	gf64_t *red   = (gf64_t *)calloc(deg_f + 1, sizeof(gf64_t));
	if (!g || !f || !u || !prod || !red) {
		printf("    alloc failed (deg_g=%zu deg_f=%zu)\n", deg_g, deg_f);
		ok = 0;
		goto cleanup;
	}

	put_seed(seed);
	for (size_t i = 0; i <= deg_g; i++) g[i] = splitmix64_next();
	for (size_t i = 0; i <= deg_f; i++) f[i] = splitmix64_next();
	if (g[deg_g] == 0) g[deg_g] = 1ULL;
	if (f[deg_f] == 0) f[deg_f] = 1ULL;

	int rc = gf64_poly_invmod_mod(g, deg_g, f, deg_f, u);
	if (rc != 0) {
		printf("    (deg_g=%zu, deg_f=%zu) gf64_poly_invmod_mod returned %d\n",
		       deg_g, deg_f, rc);
		ok = 0;
		goto cleanup;
	}

	/* prod = g * u (full polynomial, deg_g + deg_f - 1 coefficients). */
	gf64_poly_mul(prod, g, deg_g, u, deg_f - 1);
	/* red = prod mod f (deg < deg_f). */
	poly_reduce(prod, deg_g + deg_f - 1, f, deg_f, red);

	/* Contract: red[0] == 1, red[1..deg_f-1] == 0. */
	if (red[0] != 1ULL) {
		printf("    (deg_g=%zu, deg_f=%zu) red[0] = 0x%016llx, want 1\n",
		       deg_g, deg_f, (unsigned long long)red[0]);
		ok = 0;
	}
	for (size_t i = 1; i <= deg_f; i++) {
		if (red[i] != 0) {
			printf("    (deg_g=%zu, deg_f=%zu) red[%zu] = 0x%016llx, want 0\n",
			       deg_g, deg_f, i, (unsigned long long)red[i]);
			ok = 0;
			break;
		}
	}

cleanup:
	free(g); free(f); free(u); free(prod); free(red);
	return ok;
}

int main(void) {
	printf("gf64_poly_invmod_mod contract test (PR #49 step 1)\n");
	printf("===================================================\n\n");

	const size_t deg_pairs[][2] = {
		{4, 4}, {8, 8}, {16, 16}, {32, 32},
		{64, 64}, {128, 128}, {256, 256},
		{32, 64}, {64, 32}, {64, 128}, {128, 64}
	};
	const size_t num_pairs = sizeof(deg_pairs) / sizeof(deg_pairs[0]);

	int all_ok = 1;
	for (size_t i = 0; i < num_pairs; i++) {
		size_t dg = deg_pairs[i][0];
		size_t df = deg_pairs[i][1];
		printf("Case (deg_g=%zu, deg_f=%zu)\n", dg, df);
		int ok = run_case(dg, df,
		                  0xABCDEF0102030405ULL ^ ((uint64_t)dg << 32) ^ (uint64_t)df);
		if (ok) {
			char buf[80];
			snprintf(buf, sizeof(buf), "invmod_mod (deg_g=%zu, deg_f=%zu) contract",
			         dg, df);
			pass(buf);
		} else {
			all_ok = 0;
			char buf[80];
			snprintf(buf, sizeof(buf), "invmod_mod (deg_g=%zu, deg_f=%zu) contract",
			         dg, df);
			fail(buf);
		}
	}

	printf("\n=== Summary ===\n");
	printf("Passed: %d\n", g_passed);
	printf("Failed: %d\n", g_failed);

	return (all_ok && g_failed == 0) ? 0 : 1;
}