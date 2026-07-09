/*
 * ============================================================================
 * gf64/test/test_gf64_invert_ita.c — Parity test for gf64_invert_ita_batch
 *
 * T5 verification harness (par3-cauchy-fft-kernel plan). Confirms both the
 * scalar gf64_invert_ita_one and the AVX-512 vectorized gf64_invert_ita_batch
 * produce bit-exact multiplicative inverses in GF(2^64):
 *
 *   1. 1000 randomized non-zero inputs seeded with mulberry32(0xDEADBEEF);
 *      for each input `a`, the output `inv` must satisfy
 *        gf64_mul_reference(a, inv) == 1   (== the multiplicative identity).
 *   2. Zero-input edge case: zero in -> zero out (convention).
 *   3. Identity case: input == 1 -> output == 1.
 *   4. Tail-boundary lengths (0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 24,
 *      1023, 1024, 1025) to exercise every off-by-one in the N % 8 epilog
 *      of the AVX-512 batched path.
 *   5. Aliasing: out == in (in-place semantics).
 *
 * Build & run from gf64/test/:
 *   $(CC) -O2 -march=native -mavx512f -mvpclmulqdq -I.. \
 *         test_gf64_invert_ita.c ../gf64_invert_ita.c \
 *         ../gf64_square.c ../gf64_mul_avx512.c ../gf64_single.c \
 *         -o test_gf64_invert_ita && ./test_gf64_invert_ita
 *
 * Exit 0 on full pass; non-zero on any mismatch. Output is plain-text so
 * it can be piped to .omo/evidence/task-5-par3-cauchy-fft-kernel.log.
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../gf64_invert_ita.h"
#include "../gf64_global.h"

/* Public scalar SSE2 multiplication, bit-exact to gf64_solve.c:7-37.
 * Used here as the reference for verifying a * a^(-1) == 1. */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* ----------------------------------------------------------------------------
 * mulberry32 PRNG — direct C port of the JS implementation in
 * test/par3-gf64-mularr-parity.js (so test inputs match the project parity
 * tests). Seed = 0xDEADBEEF per T5 spec.
 *
 * Same uint32_t-signed-semantics matching as test_gf64_mul_avx512.c.
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

/* Multiplicative identity in GF(2^64) is the constant 1 (gf64_t is uint64_t,
 * and the field's identity element is the byte 0x01). The T5 spec verifies
 * "mul(a, out[i]) == 1" for every input, which is the multiplicative identity
 * (NOT the byte 1 — same value in this case; explicit constant for clarity). */
#define GF64_ONE ((gf64_t)1ULL)

/* Helper: count how many outputs satisfy a * inv == 1. */
static int count_inverse_ok(
	const gf64_t *in,
	const gf64_t *inv,
	size_t N,
	int print_first_few_mismatches
) {
	int mismatches = 0;
	for (size_t i = 0; i < N; i++) {
		if (gf64_mul_reference(in[i], inv[i]) != GF64_ONE) {
			mismatches++;
			if (print_first_few_mismatches && mismatches <= 5) {
				printf("    i=%zu in=0x%016llx inv=0x%016llx a*inv=0x%016llx\n",
				       i, (unsigned long long)in[i],
				       (unsigned long long)inv[i],
				       (unsigned long long)gf64_mul_reference(in[i], inv[i]));
			}
		}
	}
	return mismatches;
}

/* ----------------------------------------------------------------------------
 * Test 1: 1000 randomized non-zero parity.
 * The headline check from the T5 spec — for every non-zero a, a * a^(-1) = 1
 * in GF(2^64).
 * ---------------------------------------------------------------------------- */
static void test_basic_parity_n1000(void) {
	printf("Test 1: N=1000 randomized non-zero parity (seed=0xDEADBEEF)...\n");
	const size_t N = 1000;
	gf64_t *in  = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *sc  = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *avx = (gf64_t *)malloc(N * sizeof(gf64_t));

	if (!in || !sc || !avx) {
		printf("  SKIP: malloc failed\n");
		g_failed++;
		goto cleanup;
	}

	g_seed32 = 0xDEADBEEFu;
	for (size_t i = 0; i < N; i++) {
		gf64_t v;
		do {
			v = mulberry32_u64();
		} while (v == 0);  /* skip any zero draw — every input must be non-zero */
		in[i] = v;
	}

	gf64_invert_ita_one(in[0]);  /* touch the symbol so it isn't dead-stripped in some linkers */
	for (size_t i = 0; i < N; i++) {
		sc[i] = gf64_invert_ita_one(in[i]);
	}
	gf64_invert_ita_batch(avx, in, N);

	int sc_m  = count_inverse_ok(in, sc,  N, 1);
	int avx_m = count_inverse_ok(in, avx, N, 1);

	if (sc_m == 0) {
		printf("  PASS: scalar gf64_invert_ita_one — all %zu elements satisfy a*a^(-1)=1\n", N);
		g_passed++;
	} else {
		printf("  FAIL: scalar gf64_invert_ita_one had %d/%zu mismatches\n", sc_m, N);
		g_failed++;
	}

	if (avx_m == 0) {
		printf("  PASS: AVX-512 gf64_invert_ita_batch — all %zu elements satisfy a*a^(-1)=1\n", N);
		g_passed++;
	} else {
		printf("  FAIL: AVX-512 gf64_invert_ita_batch had %d/%zu mismatches\n", avx_m, N);
		g_failed++;
	}

	/* Cross-check: scalar and AVX-512 must agree element-by-element. */
	int cross_m = 0;
	for (size_t i = 0; i < N; i++) {
		if (sc[i] != avx[i]) cross_m++;
	}
	if (cross_m == 0) {
		printf("  PASS: scalar and AVX-512 agree on all %zu outputs\n", N);
		g_passed++;
	} else {
		printf("  FAIL: scalar vs AVX-512 had %d/%zu mismatches\n", cross_m, N);
		g_failed++;
		int printed = 0;
		for (size_t i = 0; i < N && printed < 5; i++) {
			if (sc[i] != avx[i]) {
				printf("    i=%zu in=0x%016llx sc=0x%016llx avx=0x%016llx\n",
				       i, (unsigned long long)in[i],
				       (unsigned long long)sc[i], (unsigned long long)avx[i]);
				printed++;
			}
		}
	}

cleanup:
	free(in); free(sc); free(avx);
}

/* ----------------------------------------------------------------------------
 * Test 2: Zero-input edge case.
 * Convention: gf64_invert_ita_one(0) == 0. The batched path produces 0
 * for every zero in the input array (each zero passes through the chain
 * t = 0; t^2 = 0; 0 * a = 0; etc.). The T5 spec demands this case be
 * verified explicitly.
 * ---------------------------------------------------------------------------- */
static void test_zero_input(void) {
	printf("Test 2: zero input -> zero output (convention)...\n");
	const size_t N = 1024;
	gf64_t *in  = (gf64_t *)calloc(N, sizeof(gf64_t));
	gf64_t *out = (gf64_t *)malloc(N * sizeof(gf64_t));

	if (!in || !out) {
		printf("  SKIP: alloc failed\n");
		g_failed++;
		goto cleanup;
	}

	/* Scalar path */
	int sc_m = 0;
	for (size_t i = 0; i < N; i++) {
		gf64_t r = gf64_invert_ita_one(in[i]);
		if (r != 0) sc_m++;
	}
	if (sc_m == 0) {
		printf("  PASS: scalar — all %zu zero inputs produce zero outputs\n", N);
		g_passed++;
	} else {
		printf("  FAIL: scalar — %d/%zu zero inputs did NOT return zero\n", sc_m, N);
		g_failed++;
	}

	/* Batched path */
	gf64_invert_ita_batch(out, in, N);
	int avx_m = 0;
	for (size_t i = 0; i < N; i++) {
		if (out[i] != 0) avx_m++;
	}
	if (avx_m == 0) {
		printf("  PASS: AVX-512 — all %zu zero inputs produce zero outputs\n", N);
		g_passed++;
	} else {
		printf("  FAIL: AVX-512 — %d/%zu zero inputs did NOT return zero\n", avx_m, N);
		g_failed++;
	}

cleanup:
	free(in); free(out);
}

/* ----------------------------------------------------------------------------
 * Test 3: Identity case.
 * The inverse of 1 is 1 (since 1 * 1 = 1 in any field). Verify both
 * paths honor this on the first, last, and middle positions.
 * ---------------------------------------------------------------------------- */
static void test_identity(void) {
	printf("Test 3: identity — input 1 -> output 1...\n");
	const size_t N = 64;
	gf64_t *in  = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *out = (gf64_t *)malloc(N * sizeof(gf64_t));

	if (!in || !out) {
		printf("  SKIP: alloc failed\n");
		g_failed += 2;
		goto cleanup;
	}

	for (size_t i = 0; i < N; i++) in[i] = GF64_ONE;

	/* Scalar spot-checks */
	int sc_ok = 1;
	gf64_t r_first = gf64_invert_ita_one(in[0]);
	gf64_t r_mid   = gf64_invert_ita_one(in[N / 2]);
	gf64_t r_last  = gf64_invert_ita_one(in[N - 1]);
	if (r_first != GF64_ONE || r_mid != GF64_ONE || r_last != GF64_ONE) {
		printf("  FAIL: scalar — first=0x%016llx mid=0x%016llx last=0x%016llx\n",
		       (unsigned long long)r_first, (unsigned long long)r_mid,
		       (unsigned long long)r_last);
		sc_ok = 0;
	}
	if (sc_ok) {
		printf("  PASS: scalar — 1^(-1) == 1 at first/middle/last\n");
		g_passed++;
	} else {
		g_failed++;
	}

	/* Batched full sweep */
	gf64_invert_ita_batch(out, in, N);
	int avx_m = 0;
	for (size_t i = 0; i < N; i++) {
		if (out[i] != GF64_ONE) avx_m++;
	}
	if (avx_m == 0) {
		printf("  PASS: AVX-512 — all %zu identity inputs return 1\n", N);
		g_passed++;
	} else {
		printf("  FAIL: AVX-512 — %d/%zu identity inputs did NOT return 1\n", avx_m, N);
		g_failed++;
	}

cleanup:
	free(in); free(out);
}

/* ----------------------------------------------------------------------------
 * Test 4: Tail-boundary lengths.
 * Every off-by-one in the AVX-512 batched N % 8 epilog should hit here.
 * ---------------------------------------------------------------------------- */
static void test_tail_lengths(void) {
	printf("Test 4: tail-boundary lengths (0..24, 1023..1025)...\n");
	const size_t test_lens[] = {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 24, 1023, 1024, 1025};
	const size_t num_lens = sizeof(test_lens) / sizeof(test_lens[0]);

	for (size_t t = 0; t < num_lens; t++) {
		size_t N = test_lens[t];
		if (N == 0) {
			/* N=0 must be a no-op. Safe to call with NULLs. */
			gf64_invert_ita_batch(NULL, NULL, 0);
			g_passed++;
			printf("  PASS: N=0 (no-op)\n");
			continue;
		}

		gf64_t *in  = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *out = (gf64_t *)malloc(N * sizeof(gf64_t));
		if (!in || !out) {
			printf("  SKIP N=%zu: malloc failed\n", N);
			g_failed++;
			free(in); free(out);
			continue;
		}

		g_seed32 = 0xDEADBEEFu ^ (uint32_t)t;
		for (size_t i = 0; i < N; i++) {
			gf64_t v;
			do {
				v = mulberry32_u64();
			} while (v == 0);
			in[i] = v;
		}

		gf64_invert_ita_batch(out, in, N);

		int m = count_inverse_ok(in, out, N, 0);
		if (m == 0) {
			g_passed++;
			printf("  PASS: N=%zu (all elements satisfy a*a^(-1)=1)\n", N);
		} else {
			g_failed++;
			printf("  FAIL: N=%zu had %d/%zu mismatches\n", N, m, N);
		}
		free(in); free(out);
	}
}

/* ----------------------------------------------------------------------------
 * Test 5: Mixed inputs — include some zeros in the middle of an array.
 * Verifies the chain preserves zero through the multiplication path
 * (a zero input stays zero no matter what a_it is on each iteration since
 * the chain starts with t = a and operates only on non-zero a_inv for the
 * multiplications; for zero a, t starts as zero and stays zero).
 * ---------------------------------------------------------------------------- */
static void test_mixed_with_zeros(void) {
	printf("Test 5: mixed inputs (non-zero + zeros at positions 0, 7, 8, 999)...\n");
	const size_t N = 1000;
	gf64_t *in  = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *out = (gf64_t *)malloc(N * sizeof(gf64_t));

	if (!in || !out) {
		printf("  SKIP: alloc failed\n");
		g_failed++;
		goto cleanup;
	}

	g_seed32 = 0xDEADBEEFu;
	for (size_t i = 0; i < N; i++) {
		gf64_t v;
		do {
			v = mulberry32_u64();
		} while (v == 0);
		in[i] = v;
	}
	in[0]   = 0;   /* tail (N=1000 % 8 == 0 — slot 0 still in the AVX-512 lane) */
	in[7]   = 0;   /* last lane of the first AVX-512 group */
	in[8]   = 0;   /* first lane of the second AVX-512 group */
	in[999] = 0;   /* last element — falls in scalar tail */

	gf64_invert_ita_batch(out, in, N);

	int zero_m = 0;
	int nonzero_m = 0;
	for (size_t i = 0; i < N; i++) {
		gf64_t prod = gf64_mul_reference(in[i], out[i]);
		if (in[i] == 0) {
			if (out[i] != 0) zero_m++;
		} else {
			if (prod != GF64_ONE) nonzero_m++;
		}
	}
	if (zero_m == 0 && nonzero_m == 0) {
		g_passed++;
		printf("  PASS: zeros produce zeros, non-zeros satisfy a*a^(-1)=1 (total N=%zu)\n", N);
	} else {
		g_failed++;
		printf("  FAIL: zero_m=%d nonzero_m=%d on N=%zu\n", zero_m, nonzero_m, N);
	}

cleanup:
	free(in); free(out);
}

/* ----------------------------------------------------------------------------
 * Test 6: Aliasing — out == in (in-place semantics).
 * The chain loads `a_in` once at the top of each 8-lane iteration and reuses
 * it across all 62 squaring+multiplication steps, so in-place semantics are
 * trivially safe. Verify both paths.
 * ---------------------------------------------------------------------------- */
static void test_aliasing(void) {
	printf("Test 6: aliasing (out == in)...\n");
	const size_t N = 1024;
	gf64_t *buf = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *ref = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *snap = (gf64_t *)malloc(N * sizeof(gf64_t));

	if (!buf || !ref || !snap) {
		printf("  SKIP: alloc failed\n");
		g_failed += 2;
		goto cleanup;
	}

	g_seed32 = 0x12345678u;
	for (size_t i = 0; i < N; i++) {
		gf64_t v;
		do {
			v = mulberry32_u64();
		} while (v == 0);
		buf[i] = v;
	}
	memcpy(snap, buf, N * sizeof(gf64_t));

	/* Compute reference into a separate buffer */
	for (size_t i = 0; i < N; i++) {
		ref[i] = gf64_invert_ita_one(snap[i]);
	}

	/* In-place: gf64_invert_ita_batch(buf, buf, N) */
	gf64_invert_ita_batch(buf, buf, N);
	int alias_m = count_inverse_ok(snap, buf, N, 0);
	if (alias_m == 0) {
		g_passed++;
		printf("  PASS: in-place (out == in) — all %zu elements satisfy a*a^(-1)=1\n", N);
	} else {
		g_failed++;
		printf("  FAIL: in-place — %d/%zu elements failed\n", alias_m, N);
	}

	/* Also check exact byte-equality of inverses */
	int eq_m = 0;
	for (size_t i = 0; i < N; i++) {
		if (buf[i] != ref[i]) eq_m++;
	}
	if (eq_m == 0) {
		g_passed++;
		printf("  PASS: in-place output equals scalar reference on all %zu elements\n", N);
	} else {
		g_failed++;
		printf("  FAIL: in-place output differs from scalar reference on %d/%zu elements\n", eq_m, N);
	}

cleanup:
	free(buf); free(ref); free(snap);
}

/* ----------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------------- */
int main(void) {
	printf("GF64 Itoh-Tsujii Inversion Parity Test (par3-cauchy-fft-kernel T5)\n");
	printf("==================================================================\n\n");

	test_basic_parity_n1000();
	test_zero_input();
	test_identity();
	test_tail_lengths();
	test_mixed_with_zeros();
	test_aliasing();

	printf("\n=== Summary ===\n");
	printf("Passed: %d\n", g_passed);
	printf("Failed: %d\n", g_failed);

	return g_failed > 0 ? 1 : 0;
}
