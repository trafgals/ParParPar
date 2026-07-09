/*
 * ============================================================================
 * gf64/test/test_gf64_square.c — Parity test for gf64_square / gf64_square_avx512
 *
 * T2 verification harness (par3-cauchy-fft-kernel plan). Confirms both the
 * scalar gf64_square and the AVX-512 vectorized gf64_square_avx512 produce
 * bit-exact results matching the scalar gf64_mul_reference (gf64/gf64_single.c,
 * which itself mirrors gf64_solve.c:7-37) across:
 *
 *   1. 1000 randomized inputs seeded with mulberry32(0xCAFEBABE).
 *   2. Tail-boundary lengths (0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 24,
 *      1023, 1024, 1025) to exercise every off-by-one in the N % 8 epilog.
 *   3. Edge cases: zero input -> zero output; all-ones input (0xFF...FF)
 *      squared matches mul_reference; single-bit inputs across all 64 bit
 *      positions exercise the cross-byte / cross-halfword / cross-dword
 *      boundary in the SWAR cascade.
 *   4. Aliasing: out == in (in-place semantics).
 *
 * Build & run from gf64/test/:
 *   $(CC) -O2 -march=native -mavx512f -I.. \
 *         test_gf64_square.c ../gf64_square.c ../gf64_single.c \
 *         -o test_gf64_square && ./test_gf64_square
 *
 * Exit 0 on full pass; non-zero on any mismatch. Output is plain-text so it
 * can be piped to .omo/evidence/task-2-par3-cauchy-fft-kernel.log.
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../gf64_square.h"
#include "../gf64_global.h"

/* Public scalar SSE2 reference (bit-exact to gf64_solve.c:7-37). */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* ----------------------------------------------------------------------------
 * mulberry32 PRNG — same construction as test_gf64_mul_avx512.c.
 * Seed = 0xCAFEBABE for T2.
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

/* ----------------------------------------------------------------------------
 * Test 1: 1000 randomized parity (scalar + AVX-512 vs gf64_mul_reference).
 * This is the headline check from the T2 spec.
 * ---------------------------------------------------------------------------- */
static void test_basic_parity_n1000(void) {
	printf("Test 1: N=1000 randomized parity (seed=0xCAFEBABE)...\n");
	const size_t N = 1000;
	gf64_t *in   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *ref  = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *sc   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *avx  = (gf64_t *)malloc(N * sizeof(gf64_t));

	if (!in || !ref || !sc || !avx) {
		printf("  SKIP: malloc failed\n");
		g_failed++;
		goto cleanup;
	}

	g_seed32 = 0xCAFEBABEu;
	for (size_t i = 0; i < N; i++) {
		in[i] = mulberry32_u64();
	}

	for (size_t i = 0; i < N; i++) {
		ref[i] = gf64_mul_reference(in[i], in[i]);
	}

	gf64_square(sc, in, N);
	gf64_square_avx512(avx, in, N);

	int sc_mismatches  = 0;
	int avx_mismatches = 0;
	for (size_t i = 0; i < N; i++) {
		if (sc[i]  != ref[i]) sc_mismatches++;
		if (avx[i] != ref[i]) avx_mismatches++;
	}

	if (sc_mismatches == 0) {
		printf("  PASS: scalar matches gf64_mul_reference for all %zu elements\n", N);
		g_passed++;
	} else {
		printf("  FAIL: scalar had %d/%zu mismatches\n", sc_mismatches, N);
		g_failed++;
		int printed = 0;
		for (size_t i = 0; i < N && printed < 5; i++) {
			if (sc[i] != ref[i]) {
				printf("    i=%zu in=0x%016llx got=0x%016llx ref=0x%016llx\n",
				       i, (unsigned long long)in[i],
				       (unsigned long long)sc[i], (unsigned long long)ref[i]);
				printed++;
			}
		}
	}

	if (avx_mismatches == 0) {
		printf("  PASS: AVX-512 matches gf64_mul_reference for all %zu elements\n", N);
		g_passed++;
	} else {
		printf("  FAIL: AVX-512 had %d/%zu mismatches\n", avx_mismatches, N);
		g_failed++;
		int printed = 0;
		for (size_t i = 0; i < N && printed < 5; i++) {
			if (avx[i] != ref[i]) {
				printf("    i=%zu in=0x%016llx got=0x%016llx ref=0x%016llx\n",
				       i, (unsigned long long)in[i],
				       (unsigned long long)avx[i], (unsigned long long)ref[i]);
				printed++;
			}
		}
	}

	/* Also cross-check: scalar vs AVX-512 must agree element-by-element. */
	int sc_vs_avx = 0;
	for (size_t i = 0; i < N; i++) {
		if (sc[i] != avx[i]) sc_vs_avx++;
	}
	if (sc_vs_avx == 0) {
		printf("  PASS: scalar and AVX-512 agree on all %zu elements\n", N);
		g_passed++;
	} else {
		printf("  FAIL: scalar vs AVX-512 had %d/%zu mismatches\n", sc_vs_avx, N);
		g_failed++;
	}

cleanup:
	free(in); free(ref); free(sc); free(avx);
}

/* ----------------------------------------------------------------------------
 * Test 2: Tail-boundary lengths. Every off-by-one in the N % 8 epilog should
 * trigger here.
 * ---------------------------------------------------------------------------- */
static void test_tail_lengths(void) {
	printf("Test 2: tail-boundary lengths (0..24, 1023..1025)...\n");
	const size_t test_lens[] = {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 24, 1023, 1024, 1025};
	const size_t num_lens = sizeof(test_lens) / sizeof(test_lens[0]);

	for (size_t t = 0; t < num_lens; t++) {
		size_t N = test_lens[t];
		if (N == 0) {
			/* N=0 must be a no-op. Safe to call with NULLs. */
			gf64_square(NULL, NULL, 0);
			gf64_square_avx512(NULL, NULL, 0);
			g_passed += 2;
			printf("  PASS: N=0 (no-op, scalar + AVX-512)\n");
			continue;
		}

		gf64_t *in  = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *ref = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *sc  = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *avx = (gf64_t *)malloc(N * sizeof(gf64_t));
		if (!in || !ref || !sc || !avx) {
			printf("  SKIP N=%zu: malloc failed\n", N);
			g_failed += 2;
			free(in); free(ref); free(sc); free(avx);
			continue;
		}

		g_seed32 = 0xCAFEBABEu ^ (uint32_t)t;
		for (size_t i = 0; i < N; i++) {
			in[i] = mulberry32_u64();
		}
		for (size_t i = 0; i < N; i++) {
			ref[i] = gf64_mul_reference(in[i], in[i]);
		}

		gf64_square(sc, in, N);
		gf64_square_avx512(avx, in, N);

		int sc_m = 0, avx_m = 0;
		for (size_t i = 0; i < N; i++) {
			if (sc[i]  != ref[i]) sc_m++;
			if (avx[i] != ref[i]) avx_m++;
		}
		if (sc_m == 0 && avx_m == 0) {
			g_passed += 2;
			printf("  PASS: N=%zu (scalar + AVX-512)\n", N);
		} else {
			g_failed += 2;
			printf("  FAIL: N=%zu scalar=%d avx=%d mismatches\n", N, sc_m, avx_m);
		}
		free(in); free(ref); free(sc); free(avx);
	}
}

/* ----------------------------------------------------------------------------
 * Test 3: Edge cases.
 *   - zero input -> zero output (0^2 = 0).
 *   - all-ones input -> squared output matches mul_reference (all-ones^2 in
 *     GF(2^64) is NOT all-ones; reduction maps the 128-bit squaring back
 *     to a non-trivial 64-bit element).
 *   - single-bit inputs across all 64 positions, exercising cross-byte,
 *     cross-halfword, and cross-dword boundaries in the SWAR cascade.
 * ---------------------------------------------------------------------------- */
static void test_edge_cases(void) {
	printf("Test 3: edge cases (zero, all-ones, single-bit sweep)...\n");
	const size_t N = 64;
	gf64_t *in   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *ref  = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *sc   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *avx  = (gf64_t *)malloc(N * sizeof(gf64_t));

	/* Zero input */
	for (size_t i = 0; i < N; i++) in[i] = 0;
	for (size_t i = 0; i < N; i++) ref[i] = 0;
	gf64_square(sc, in, N);
	gf64_square_avx512(avx, in, N);
	int sc_m = 0, avx_m = 0;
	for (size_t i = 0; i < N; i++) {
		if (sc[i]  != 0) sc_m++;
		if (avx[i] != 0) avx_m++;
	}
	if (sc_m == 0 && avx_m == 0) {
		g_passed += 2;
		printf("  PASS: 0^2 == 0 (scalar + AVX-512)\n");
	} else {
		g_failed += 2;
		printf("  FAIL: zero input: scalar=%d avx=%d non-zero outputs\n", sc_m, avx_m);
	}

	/* All-ones input */
	for (size_t i = 0; i < N; i++) in[i] = ~(gf64_t)0;
	for (size_t i = 0; i < N; i++) ref[i] = gf64_mul_reference(in[i], in[i]);
	gf64_square(sc, in, N);
	gf64_square_avx512(avx, in, N);
	sc_m = 0; avx_m = 0;
	for (size_t i = 0; i < N; i++) {
		if (sc[i]  != ref[i]) sc_m++;
		if (avx[i] != ref[i]) avx_m++;
	}
	if (sc_m == 0 && avx_m == 0) {
		g_passed += 2;
		printf("  PASS: all-ones squared matches mul_reference (scalar + AVX-512), result=0x%016llx\n",
		       (unsigned long long)ref[0]);
	} else {
		g_failed += 2;
		printf("  FAIL: all-ones squared: scalar=%d avx=%d mismatches (ref result=0x%016llx)\n",
		       sc_m, avx_m, (unsigned long long)ref[0]);
	}

	/* Single-bit inputs across the bit range (one element at a time). */
	int single_bit_ok = 1;
	for (int bit = 0; bit < 64; bit++) {
		gf64_t a = (gf64_t)1ULL << bit;
		gf64_t r = gf64_mul_reference(a, a);
		gf64_t s_scalar = 0;
		gf64_t s_avx    = 0;
		gf64_square(&s_scalar, &a, 1);
		gf64_square_avx512(&s_avx, &a, 1);
		if (s_scalar != r) {
			printf("  FAIL: single-bit pos=%d scalar: got 0x%016llx, expected 0x%016llx\n",
			       bit, (unsigned long long)s_scalar, (unsigned long long)r);
			single_bit_ok = 0;
		}
		if (s_avx != r) {
			printf("  FAIL: single-bit pos=%d avx: got 0x%016llx, expected 0x%016llx\n",
			       bit, (unsigned long long)s_avx, (unsigned long long)r);
			single_bit_ok = 0;
		}
	}
	if (single_bit_ok) {
		g_passed++;
		printf("  PASS: all 64 single-bit positions square correctly (scalar + AVX-512)\n");
	} else {
		g_failed++;
	}

	free(in); free(ref); free(sc); free(avx);
}

/* ----------------------------------------------------------------------------
 * Test 4: Aliasing — out == in (in-place semantics).
 * Both functions must support this.
 * ---------------------------------------------------------------------------- */
static void test_aliasing(void) {
	printf("Test 4: aliasing (out == in)...\n");
	const size_t N = 1024;
	gf64_t *buf  = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *ref  = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *snap = (gf64_t *)malloc(N * sizeof(gf64_t));

	g_seed32 = 0x12345678u;
	for (size_t i = 0; i < N; i++) {
		buf[i] = mulberry32_u64();
	}
	memcpy(snap, buf, N * sizeof(gf64_t));
	for (size_t i = 0; i < N; i++) {
		ref[i] = gf64_mul_reference(snap[i], snap[i]);
	}

	/* Scalar in-place */
	gf64_square(buf, buf, N);
	int m = 0;
	for (size_t i = 0; i < N; i++) {
		if (buf[i] != ref[i]) m++;
	}
	if (m == 0) {
		g_passed++;
		printf("  PASS: scalar in-place (out == in)\n");
	} else {
		g_failed++;
		printf("  FAIL: scalar in-place had %d mismatches\n", m);
	}

	/* Restore and run AVX-512 in-place */
	memcpy(buf, snap, N * sizeof(gf64_t));
	gf64_square_avx512(buf, buf, N);
	m = 0;
	for (size_t i = 0; i < N; i++) {
		if (buf[i] != ref[i]) m++;
	}
	if (m == 0) {
		g_passed++;
		printf("  PASS: AVX-512 in-place (out == in)\n");
	} else {
		g_failed++;
		printf("  FAIL: AVX-512 in-place had %d mismatches\n", m);
	}

	free(buf); free(ref); free(snap);
}

/* ----------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------------- */
int main(void) {
	printf("GF64 Frobenius Squaring Parity Test (par3-cauchy-fft-kernel T2)\n");
	printf("==============================================================\n\n");

	test_basic_parity_n1000();
	test_tail_lengths();
	test_edge_cases();
	test_aliasing();

	printf("\n=== Summary ===\n");
	printf("Passed: %d\n", g_passed);
	printf("Failed: %d\n", g_failed);

	return g_failed > 0 ? 1 : 0;
}