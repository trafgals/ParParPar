/*
 * ============================================================================
 * gf64/test/test_gf64_mul_avx512.c — Parity test for gf64_mul_avx512
 *
 * T1 verification harness (par3-cauchy-fft-kernel plan). Confirms the
 * AVX-512 vectorized gf64_mul_avx512 produces bit-exact results matching
 * the scalar gf64_mul_reference (gf64/gf64_single.c, which itself mirrors
 * gf64_solve.c:7-37) across:
 *
 *   1. N=1024 randomized inputs seeded with mulberry32(0xDEADBEEF).
 *   2. Tail-boundary lengths (0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 24,
 *      1023, 1024, 1025) to exercise every off-by-one in the N % 8 epilog.
 *   3. Identity (a * 1 = a) and zero (a * 0 = 0) edge cases.
 *   4. Aliasing: out == a and out == b (in-place semantics).
 *
 * Build & run from gf64/test/:
 *   $(CC) -O2 -march=native -mavx512f -mvpclmulqdq -I.. \
 *         test_gf64_mul_avx512.c ../gf64_mul_avx512.c ../gf64_single.c \
 *         -o test_gf64_mul_avx512 && ./test_gf64_mul_avx512
 *
 * Exit 0 on full pass; non-zero on any mismatch. Output is plain-text so it
 * can be piped to .omo/evidence/task-1-par3-cauchy-fft-kernel.log.
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../gf64_mul.h"
#include "../gf64_global.h"

/* Public scalar SSE2 reduction (bit-exact to gf64_solve.c:7-37). */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* ----------------------------------------------------------------------------
 * mulberry32 PRNG — direct C port of the JS implementation in
 * test/par3-gf64-mularr-parity.js (so test inputs match the parity test that
 * ships in the project). Seed = 0xDEADBEEF per T1 spec.
 *
 * The JS version uses Math.imul and `| 0` for 32-bit signed semantics; we
 * match with explicit uint32_t casts to keep the output stream identical.
 * ---------------------------------------------------------------------------- */
static uint32_t g_seed32 = 0;

static uint32_t mulberry32_step(void) {
	g_seed32 = (uint32_t)(g_seed32 + 0x6D2B79F5u);
	uint32_t t = g_seed32;
	t = (uint32_t)((t ^ (t >> 15)) * (1u | t));
	t = (uint32_t)(t + (uint32_t)((t ^ (t >> 7)) * (61u | t)) ^ t);
	return (uint32_t)((t ^ (t >> 14)) & 0xFFFFFFFFu);
}

static double mulberry32_double(void) {
	return (double)mulberry32_step() / 4294967296.0;
}

static uint64_t mulberry32_u64(void) {
	uint64_t hi = (uint64_t)(mulberry32_double() * 4294967296.0);
	uint64_t lo = (uint64_t)(mulberry32_double() * 4294967296.0);
	return (hi << 32) | lo;
}

/* ----------------------------------------------------------------------------
 * Test framework
 * ---------------------------------------------------------------------------- */
static int g_passed = 0;
static int g_failed = 0;

#define EXPECT_EQ(a, b, msg) do { \
	if ((a) == (b)) { \
		g_passed++; \
	} else { \
		g_failed++; \
		printf("FAIL: %s (got 0x%016llx, expected 0x%016llx)\n", \
		       msg, (unsigned long long)(a), (unsigned long long)(b)); \
	} \
} while(0)

/* ----------------------------------------------------------------------------
 * Test 1: N=1024 randomized parity. The headline check from the T1 spec.
 * ---------------------------------------------------------------------------- */
static void test_basic_parity_n1024(void) {
	printf("Test 1: N=1024 randomized parity (seed=0xDEADBEEF)...\n");
	const size_t N = 1024;
	gf64_t *a     = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *b     = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *ref   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *got   = (gf64_t *)malloc(N * sizeof(gf64_t));

	if (!a || !b || !ref || !got) {
		printf("  SKIP: malloc failed\n");
		g_failed++;
		goto cleanup;
	}

	g_seed32 = 0xDEADBEEFu;
	for (size_t i = 0; i < N; i++) {
		a[i] = mulberry32_u64();
		b[i] = mulberry32_u64();
	}

	for (size_t i = 0; i < N; i++) {
		ref[i] = gf64_mul_reference(a[i], b[i]);
	}

	gf64_mul_avx512(got, a, b, N);

	int mismatches = 0;
	for (size_t i = 0; i < N; i++) {
		if (got[i] != ref[i]) {
			mismatches++;
			if (mismatches <= 5) {
				printf("  Mismatch at i=%zu: a=0x%016llx b=0x%016llx got=0x%016llx ref=0x%016llx\n",
				       i, (unsigned long long)a[i], (unsigned long long)b[i],
				       (unsigned long long)got[i], (unsigned long long)ref[i]);
			}
		}
	}
	if (mismatches == 0) {
		printf("  PASS: %zu/%zu elements match\n", N, N);
		g_passed++;
	} else {
		printf("  FAIL: %d/%zu mismatches\n", mismatches, N);
		g_failed++;
	}

cleanup:
	free(a); free(b); free(ref); free(got);
}

/* ----------------------------------------------------------------------------
 * Test 2: Tail-boundary lengths. Every off-by-one in the N % 8 epilog should
 * trigger here.
 * ---------------------------------------------------------------------------- */
static void test_tail_lengths(void) {
	printf("Test 2: tail-boundary lengths (0..24, 1023..1025)...\n");
	const size_t test_lens[] = {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 24, 1023, 1024, 1025};
	const size_t num_lens = sizeof(test_lens) / sizeof(test_lens[0]);
	int all_ok = 1;

	for (size_t t = 0; t < num_lens; t++) {
		size_t N = test_lens[t];
		if (N == 0) {
			/* N=0 must be a no-op. Call with NULLs; the body only iterates
			 * the scalar tail (which is `for (; i < N; i++)`), so this is safe. */
			gf64_mul_avx512(NULL, NULL, NULL, 0);
			g_passed++;
			printf("  PASS: N=0 (no-op)\n");
			continue;
		}

		gf64_t *a   = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *b   = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *ref = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *got = (gf64_t *)malloc(N * sizeof(gf64_t));
		if (!a || !b || !ref || !got) {
			printf("  SKIP N=%zu: malloc failed\n", N);
			g_failed++;
			all_ok = 0;
			free(a); free(b); free(ref); free(got);
			continue;
		}

		g_seed32 = 0xDEADBEEFu ^ (uint32_t)t;
		for (size_t i = 0; i < N; i++) {
			a[i] = mulberry32_u64();
			b[i] = mulberry32_u64();
		}
		for (size_t i = 0; i < N; i++) {
			ref[i] = gf64_mul_reference(a[i], b[i]);
		}

		gf64_mul_avx512(got, a, b, N);

		int mismatches = 0;
		for (size_t i = 0; i < N; i++) {
			if (got[i] != ref[i]) mismatches++;
		}
		if (mismatches == 0) {
			g_passed++;
			printf("  PASS: N=%zu\n", N);
		} else {
			g_failed++;
			all_ok = 0;
			printf("  FAIL: N=%zu had %d mismatches\n", N, mismatches);
		}
		free(a); free(b); free(ref); free(got);
	}
	(void)all_ok;
}

/* ----------------------------------------------------------------------------
 * Test 3: Identity (a * 1 = a) and zero (a * 0 = 0).
 * ---------------------------------------------------------------------------- */
static void test_identity_and_zero(void) {
	printf("Test 3: identity (a*1==a) and zero (a*0==0)...\n");
	const size_t N = 64;
	gf64_t *a   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *one = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *z   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *got = (gf64_t *)malloc(N * sizeof(gf64_t));

	g_seed32 = 0xCAFEBABEu;
	for (size_t i = 0; i < N; i++) a[i] = mulberry32_u64();
	for (size_t i = 0; i < N; i++) one[i] = 1;
	for (size_t i = 0; i < N; i++) z[i] = 0;

	/* a * 1 */
	gf64_mul_avx512(got, a, one, N);
	int ok = 1;
	for (size_t i = 0; i < N; i++) {
		if (got[i] != a[i]) { ok = 0; break; }
	}
	if (ok) { g_passed++; printf("  PASS: a*1 == a (all %zu)\n", N); }
	else    { g_failed++; printf("  FAIL: a*1 != a\n"); }

	/* a * 0 */
	gf64_mul_avx512(got, a, z, N);
	ok = 1;
	for (size_t i = 0; i < N; i++) {
		if (got[i] != 0) { ok = 0; break; }
	}
	if (ok) { g_passed++; printf("  PASS: a*0 == 0 (all %zu)\n", N); }
	else    { g_failed++; printf("  FAIL: a*0 != 0\n"); }

	free(a); free(one); free(z); free(got);
}

/* ----------------------------------------------------------------------------
 * Test 4: Aliasing — out == a and out == b. The T1 spec mandates the function
 * allow out to alias either input.
 * ---------------------------------------------------------------------------- */
static void test_aliasing(void) {
	printf("Test 4: aliasing (out == a, out == b)...\n");
	const size_t N = 1024;
	gf64_t *a   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *b   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *ref = (gf64_t *)malloc(N * sizeof(gf64_t));

	g_seed32 = 0x12345678u;
	for (size_t i = 0; i < N; i++) {
		a[i] = mulberry32_u64();
		b[i] = mulberry32_u64();
	}
	for (size_t i = 0; i < N; i++) {
		ref[i] = gf64_mul_reference(a[i], b[i]);
	}

	/* Snapshot a and b before any in-place call */
	gf64_t *a0 = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *b0 = (gf64_t *)malloc(N * sizeof(gf64_t));
	memcpy(a0, a, N * sizeof(gf64_t));
	memcpy(b0, b, N * sizeof(gf64_t));

	/* out == a */
	gf64_mul_avx512(a, a, b, N);
	int mismatches_ab = 0;
	for (size_t i = 0; i < N; i++) {
		if (a[i] != ref[i]) mismatches_ab++;
	}
	if (mismatches_ab == 0) {
		g_passed++; printf("  PASS: out==a aliasing\n");
	} else {
		g_failed++;
		printf("  FAIL: out==a had %d mismatches\n", mismatches_ab);
	}

	/* Restore a for the next aliasing test, then run with out == b */
	memcpy(a, a0, N * sizeof(gf64_t));
	gf64_mul_avx512(b, a, b, N);
	int mismatches_bb = 0;
	for (size_t i = 0; i < N; i++) {
		if (b[i] != ref[i]) mismatches_bb++;
	}
	if (mismatches_bb == 0) {
		g_passed++; printf("  PASS: out==b aliasing\n");
	} else {
		g_failed++;
		printf("  FAIL: out==b had %d mismatches\n", mismatches_bb);
	}

	free(a); free(b); free(ref); free(a0); free(b0);
}

/* ----------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------------- */
int main(void) {
	printf("GF64 mul AVX-512 Parity Test (par3-cauchy-fft-kernel T1)\n");
	printf("========================================================\n\n");

	test_basic_parity_n1024();
	test_tail_lengths();
	test_identity_and_zero();
	test_aliasing();

	printf("\n=== Summary ===\n");
	printf("Passed: %d\n", g_passed);
	printf("Failed: %d\n", g_failed);

	return g_failed > 0 ? 1 : 0;
}