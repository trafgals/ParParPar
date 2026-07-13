/*
 * ============================================================================
 * gf64/test/test_gf64_fft_poly_mul.c
 *
 * Phase 2b FFT convolution-theorem probe.
 *
 * The Gao-Mateer additive FFT in gf64_additive_fft.c implements a recursive
 * transform that maps monomial-basis polynomials to (what the existing code
 * calls) an "evaluation-like" representation. If the recursion's character
 * basis satisfies the convolution theorem, then pointwise multiply +
 * inverse-transform computes polynomial convolution in O(N log N).
 *
 * This test PROBES that hypothesis directly:
 *   1. Pick random a and b of length n (n a power of 2);
 *   2. compute schoolbook a*b;
 *   3. copy a and b into scratch buffers of length n, zero-padded;
 *   4. forward FFT both;
 *   5. pointwise multiply in the field;
 *   6. inverse FFT the result;
 *   7. compare to schoolbook (with the round-trip scale absorbed).
 *
 * We compare both the bit-exact coefficient match AND the integer
 * element count of any mismatch. The test is expected to FAIL on the
 * first run (the existing Gao-Mateer implementation is the
 * monomial-basis transform, NOT the evaluation-basis transform — see
 * the doc on gf64_poly_mul_internal in gf64_additive_fft.c) and the
 * failure mode here documents the precise algebraic gap.
 *
 * Build & run from gf64/test/:
 *   $(CC) -O2 -march=native -I.. test_gf64_fft_poly_mul.c \
 *         ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c \
 *         ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c \
 *         ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c \
 *         -o test_gf64_fft_poly_mul && ./test_gf64_fft_poly_mul
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../gf64_additive_fft.h"
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

static uint64_t g_rng = 0xC0DEBA5EFACE0001ULL;
static uint64_t splitmix64_next(void) {
	uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static int g_passed = 0;
static int g_failed = 0;
static void pass(const char *name) { printf("  PASS: %s\n", name); g_passed++; }
static void fail(const char *name) { printf("  FAIL: %s\n", name); g_failed++; }

/* Schoolbook reference. */
static void schoolbook(
	gf64_t *out, const gf64_t *a, size_t la, const gf64_t *b, size_t lb
) {
	memset(out, 0, (la + lb - 1) * sizeof(gf64_t));
	for (size_t i = 0; i < la; i++) {
		for (size_t j = 0; j < lb; j++) {
			out[i + j] ^= gf64_mul_reference(a[i], b[j]);
		}
	}
}

static int test_one(const char *name, size_t n, uint64_t seed) {
	g_rng = seed;
	gf64_t *a = (gf64_t *)malloc(n * sizeof(gf64_t));
	gf64_t *b = (gf64_t *)malloc(n * sizeof(gf64_t));
	for (size_t i = 0; i < n; i++) a[i] = splitmix64_next();
	for (size_t i = 0; i < n; i++) b[i] = splitmix64_next();

	gf64_t *ab_ref = (gf64_t *)calloc(2*n - 1, sizeof(gf64_t));
	schoolbook(ab_ref, a, n, b, n);

	/* Build zero-padded length-n buffers, forward-FFT each, pointwise multiply, inverse-FFT. */
	gf64_t *A = (gf64_t *)calloc(n, sizeof(gf64_t));
	gf64_t *B = (gf64_t *)calloc(n, sizeof(gf64_t));
	memcpy(A, a, n * sizeof(gf64_t));
	memcpy(B, b, n * sizeof(gf64_t));

	gf64_fft_forward(A, n);
	gf64_fft_forward(B, n);
	for (size_t i = 0; i < n; i++) {
		A[i] = gf64_mul_reference(A[i], B[i]);
	}
	gf64_fft_inverse(A, n);

	int mismatches = 0;
	for (size_t i = 0; i < 2*n - 1; i++) {
		if (A[i] != ab_ref[i]) {
			if (mismatches < 3) {
				printf("    %s @ n=%zu: a*b[%zu] fft=0x%016llx ref=0x%016llx\n",
				       name, n, i,
				       (unsigned long long)A[i], (unsigned long long)ab_ref[i]);
			}
			mismatches++;
		}
	}

	free(a); free(b); free(ab_ref); free(A); free(B);

	if (mismatches == 0) {
		pass(name);
		return 0;
	} else {
		printf("    %s @ n=%zu: %d mismatches\n", name, n, mismatches);
		fail(name);
		return 1;
	}
}

int main(void) {
	printf("Phase 2b FFT convolution-theorem probe\n");
	printf("=======================================\n\n");
	printf("Hypothesis: gf64_fft_forward + gf64_fft_inverse + pointwise multiply ==\n");
	printf("            polynomial convolution in GF(2^64)[x].\n\n");
	printf("If the additive FFT's character basis satisfies the convolution\n");
	printf("theorem, this test passes. Otherwise it fails with coefficient-\n");
	printf("by-coefficient mismatches, documenting the algebraic gap that\n");
	printf("Phase 2b needs to close (full Gao-Mateer evaluation-basis or a\n");
	printf("subfield NTT).\n\n");

	test_one("n=2",   2,   0xA1A1A1A1A1A1A1A1ULL);
	test_one("n=4",   4,   0xA2A2A2A2A2A2A2A2ULL);
	test_one("n=8",   8,   0xA3A3A3A3A3A3A3A3ULL);
	test_one("n=16",  16,  0xA4A4A4A4A4A4A4A4ULL);
	test_one("n=32",  32,  0xA5A5A5A5A5A5A5A5ULL);
	test_one("n=64",  64,  0xA6A6A6A6A6A6A6A6ULL);
	test_one("n=128", 128, 0xA7A7A7A7A7A7A7A7ULL);
	test_one("n=256", 256, 0xA8A8A8A8A8A8A8A8ULL);

	printf("\n=== Summary ===\n");
	printf("Passed: %d\n", g_passed);
	printf("Failed: %d\n", g_failed);
	printf("\nInterpretation:\n");
	printf("  - All PASS: the existing additive FFT does implement the convolution\n");
	printf("    theorem (FFT poly_mul is a drop-in), giving Phase 2b O(n log n)\n");
	printf("    for free. Proceed to Fenger Toeplitz (Phase 3).\n");
	printf("  - Some FAIL: the existing additive FFT is monomial-basis, not\n");
	printf("    evaluation-basis. Closing this gap requires an evaluation-basis\n");
	printf("    transform — a research-grade multi-week implementation.\n");
	return g_failed > 0 ? 1 : 0;
}