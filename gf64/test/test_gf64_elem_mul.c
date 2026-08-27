/*
 * ============================================================================
 * gf64/test/test_gf64_elem_mul.c — Parity test for gf64_elem_mul
 *
 * Issue #59 Wave 1 / T3. Verifies that the ISA-dispatched
 * `gf64_elem_mul(out, a, b, N)` wrapper produces bit-exact results
 * matching the scalar `gf64_mul_reference` across:
 *
 *   1. Randomized bulk inputs at N ∈ {1, 8, 1024, 4096, 4099, 65536}.
 *      Each iteration exercises one full inner loop (and any
 *      tail-epilog path) at a different cache footprint.
 *   2. Tail-boundary lengths (0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 24,
 *      1023, 1024, 1025, 4095, 4096, 4097) to exercise every off-by-one
 *      in the AVX-512 lane loop's N % 8 epilog.
 *   3. Identity (a * 1 = a) and zero (a * 0 = 0) edge cases.
 *   4. Aliasing: out == a and out == b (in-place semantics).
 *
 * ISA gating
 *   The AVX-512 path is exercised only when `gf64_current_method` resolves
 *   to GF64_AVX512 AND `gf64_has_vpclmulqdq` is set (the SIGILL-tested
 *   ZMM probe — see gf64_dispatch.c:122). On other hosts the test runs
 *   only the scalar path; the dispatcher in gf64_elem_mul.c returns the
 *   scalar loop and bit-exactness is trivially satisfied (scalar vs
 *   scalar). The test exits 0 on either branch; per-ISA verdict is
 *   logged so the receiving CI lane can see which path was actually
 *   verified.
 *
 * Build & run from gf64/test/:
 *   make test_gf64_elem_mul && ./test_gf64_elem_mul
 *
 * Exit 0 on full pass; non-zero on any mismatch.
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../gf64_elem_mul.h"
#include "../gf64_global.h"

/* Public scalar SSE2 reduction (bit-exact to gf64_solve.c:7-37). */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* ----------------------------------------------------------------------------
 * mulberry32 PRNG — same scheme as test_gf64_mul_avx512.c so the seed
 * produces a deterministic stream across both test binaries. Seed = 0xDEADBEEF
 * for the bulk tests, seed xor'd with iteration index for the tail sweep.
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

static int g_avx512_path_ran = 0;
static int g_scalar_path_ran = 0;

/* ----------------------------------------------------------------------------
 * Test 1: Bulk randomized parity across cache footprints.
 * Covers N ∈ {1, 8, 1024, 4096, 4099, 65536}. Each N exercises one full
 * inner-loop iteration and (for N % 8 != 0) the tail epilog.
 * ---------------------------------------------------------------------------- */
static void test_bulk_parity(void) {
	printf("Test 1: bulk randomized parity (seed=0xDEADBEEF)...\n");
	const size_t test_ns[] = {1, 8, 1024, 4096, 4099, 65536};
	const size_t num_ns = sizeof(test_ns) / sizeof(test_ns[0]);

	for (size_t t = 0; t < num_ns; t++) {
		size_t N = test_ns[t];
		gf64_t *a   = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *b   = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *ref = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *got = (gf64_t *)malloc(N * sizeof(gf64_t));
		if (!a || !b || !ref || !got) {
			printf("  SKIP N=%zu: malloc failed\n", N);
			g_failed++;
			free(a); free(b); free(ref); free(got);
			continue;
		}

		g_seed32 = (uint32_t)(0xDEADBEEFu ^ (uint32_t)t);
		for (size_t i = 0; i < N; i++) {
			a[i] = mulberry32_u64();
			b[i] = mulberry32_u64();
		}
		for (size_t i = 0; i < N; i++) {
			ref[i] = gf64_mul_reference(a[i], b[i]);
		}

		gf64_elem_mul(got, a, b, N);

		int mismatches = 0;
		for (size_t i = 0; i < N; i++) {
			if (got[i] != ref[i]) mismatches++;
		}
		if (mismatches == 0) {
			printf("  PASS: N=%zu\n", N);
			g_passed++;
		} else {
			printf("  FAIL: N=%zu had %d mismatches\n", N, mismatches);
			g_failed++;
		}
		free(a); free(b); free(ref); free(got);
	}

	/* Mark which path actually executed, for the per-ISA verdict. */
	if (gf64_current_method == GF64_AVX512 && gf64_has_vpclmulqdq) {
		g_avx512_path_ran = 1;
	} else {
		g_scalar_path_ran = 1;
	}
}

/* ----------------------------------------------------------------------------
 * Test 2: Tail-boundary lengths. Every off-by-one in the AVX-512 lane
 * loop's N % 8 epilog should trigger here. The scalar fallback is
 * exercised on every length; the AVX-512 path is only triggered on
 * lengths >= 8 (smaller N hits the scalar tail epilog).
 * ---------------------------------------------------------------------------- */
static void test_tail_lengths(void) {
	printf("Test 2: tail-boundary lengths (0..24, 1023..1025, 4095..4097)...\n");
	const size_t test_lens[] = {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 24,
	                            1023, 1024, 1025, 4095, 4096, 4097};
	const size_t num_lens = sizeof(test_lens) / sizeof(test_lens[0]);

	for (size_t t = 0; t < num_lens; t++) {
		size_t N = test_lens[t];
		if (N == 0) {
			/* N=0 must be a no-op. The dispatcher's scalar loop and
			 * gf64_mul_avx512 (when bound) both handle N=0 trivially. */
			gf64_elem_mul(NULL, NULL, NULL, 0);
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
			free(a); free(b); free(ref); free(got);
			continue;
		}

		g_seed32 = (uint32_t)(0xDEADBEEFu ^ (uint32_t)t);
		for (size_t i = 0; i < N; i++) {
			a[i] = mulberry32_u64();
			b[i] = mulberry32_u64();
		}
		for (size_t i = 0; i < N; i++) {
			ref[i] = gf64_mul_reference(a[i], b[i]);
		}

		gf64_elem_mul(got, a, b, N);

		int mismatches = 0;
		for (size_t i = 0; i < N; i++) {
			if (got[i] != ref[i]) mismatches++;
		}
		if (mismatches == 0) {
			g_passed++;
			printf("  PASS: N=%zu\n", N);
		} else {
			g_failed++;
			printf("  FAIL: N=%zu had %d mismatches\n", N, mismatches);
		}
		free(a); free(b); free(ref); free(got);
	}
}

/* ----------------------------------------------------------------------------
 * Test 3: Identity (a * 1 = a) and zero (a * 0 = 0). Exercises the
 * element-wise mul with extreme operand values.
 * ---------------------------------------------------------------------------- */
static void test_identity_and_zero(void) {
	printf("Test 3: identity (a*1==a) and zero (a*0==0)...\n");
	const size_t N = 64;
	gf64_t *a   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *one = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *z   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *got = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (!a || !one || !z || !got) {
		printf("  SKIP: malloc failed\n");
		g_failed++;
		free(a); free(one); free(z); free(got);
		return;
	}

	g_seed32 = 0xCAFEBABEu;
	for (size_t i = 0; i < N; i++) a[i] = mulberry32_u64();
	for (size_t i = 0; i < N; i++) one[i] = 1;
	for (size_t i = 0; i < N; i++) z[i] = 0;

	gf64_elem_mul(got, a, one, N);
	int ok = 1;
	for (size_t i = 0; i < N; i++) {
		if (got[i] != a[i]) { ok = 0; break; }
	}
	if (ok) { g_passed++; printf("  PASS: a*1 == a (all %zu)\n", N); }
	else    { g_failed++; printf("  FAIL: a*1 != a\n"); }

	gf64_elem_mul(got, a, z, N);
	ok = 1;
	for (size_t i = 0; i < N; i++) {
		if (got[i] != 0) { ok = 0; break; }
	}
	if (ok) { g_passed++; printf("  PASS: a*0 == 0 (all %zu)\n", N); }
	else    { g_failed++; printf("  FAIL: a*0 != 0\n"); }

	free(a); free(one); free(z); free(got);
}

/* ----------------------------------------------------------------------------
 * Test 4: Aliasing — out == a and out == b. The dispatcher's contract
 * (gf64_elem_mul.h:68) requires aliasing support; the AVX-512 path
 * already supports it (gf64_mul.h:30), and the scalar fallback
 * trivially does.
 * ---------------------------------------------------------------------------- */
static void test_aliasing(void) {
	printf("Test 4: aliasing (out == a, out == b)...\n");
	const size_t N = 1024;
	gf64_t *a   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *b   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *ref = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (!a || !b || !ref) {
		printf("  SKIP: malloc failed\n");
		g_failed++;
		free(a); free(b); free(ref);
		return;
	}

	g_seed32 = 0x12345678u;
	for (size_t i = 0; i < N; i++) {
		a[i] = mulberry32_u64();
		b[i] = mulberry32_u64();
	}
	for (size_t i = 0; i < N; i++) {
		ref[i] = gf64_mul_reference(a[i], b[i]);
	}

	gf64_t *a0 = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *b0 = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (!a0 || !b0) {
		printf("  SKIP: snapshot malloc failed\n");
		g_failed++;
		free(a); free(b); free(ref); free(a0); free(b0);
		return;
	}
	memcpy(a0, a, N * sizeof(gf64_t));
	memcpy(b0, b, N * sizeof(gf64_t));

	/* out == a */
	gf64_elem_mul(a, a, b, N);
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

	memcpy(a, a0, N * sizeof(gf64_t));
	gf64_elem_mul(b, a, b, N);
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
	printf("GF64 element-wise mul parity test (issue #59 T3)\n");
	printf("==================================================\n\n");

	/* Initialise dispatch so the per-ISA verdict is meaningful. */
	int rc = gf64_init_dispatch();
	if (rc != 0) {
		printf("FATAL: gf64_init_dispatch returned %d\n", rc);
		return 2;
	}

	test_bulk_parity();
	test_tail_lengths();
	test_identity_and_zero();
	test_aliasing();

	printf("\n=== Per-ISA verdict ===\n");
	if (g_avx512_path_ran) {
		printf("AVX-512 path exercised: gf64_mul_avx512 (VPCLMULQDQ)\n");
	} else if (g_scalar_path_ran) {
		const char *m =
			(gf64_current_method == GF64_AVX2)  ? "AVX-2"  :
			(gf64_current_method == GF64_SSSE3) ? "SSSE3"  :
			(gf64_current_method == GF64_SCALAR) ? "SCALAR" :
			"unknown";
		printf("Scalar fallback exercised (gf64_current_method = %s)\n", m);
		printf("AVX-512 path NOT exercised on this host — pass marks\n");
		printf("the scalar contract only. To exercise AVX-512, run on a\n");
		printf("host with VPCLMULQDQ+ZMM (e.g. PAR3_GF64_USE_AVX512=1).\n");
	} else {
		printf("(no path marked — bulk test did not run)\n");
	}

	printf("\n=== Summary ===\n");
	printf("Passed: %d\n", g_passed);
	printf("Failed: %d\n", g_failed);

	return g_failed > 0 ? 1 : 0;
}