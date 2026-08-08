/*
 * ============================================================================
 * test_gf64_fenger_pipeline.c — Phase 1 bit-exact parity check for the
 *   Fenger Toeplitz pipeline (issue #28).
 *
 * Compares:
 *
 *   cauchy_reference: out[r][w] = XOR_c in[c][w] / (y_r XOR x_c)
 *                     (explicit O(N) per output, in-place Horner)
 *
 *   gf64_fenger_matvec: the Bostan-Schost top-down pipeline from
 *                     gf64_fenger.c (issue #28 Phase 1).
 *
 * Configurations cover:
 *   - N in {1, 2, 4, 8, 16, 32}
 *   - R in {1, 2, 4, 8, 16}
 *   - B in {1, 2, 4, 8}
 *   - disjoint firstInput / firstRecovery offsets (V(y_r) nonzero)
 *
 * Phase 0's naive Fenger reference (Lagrange interp + V'(x_c) weighting)
 * verified the algebra; Phase 1 verifies the Bostan-Schost tree-walking
 * implementation in gf64_fenger.c agrees with the same algebra.
 * ============================================================================
 */

#include "../gf64_fenger.h"
#include "../gf64_invert.h"
#include "../gf64_mul.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);
extern gf64_t gf64_inverse(gf64_t a);

/* ----- Reference: explicit Cauchy matrix-vector product. */
static void cauchy_reference(
	const gf64_t *in, size_t N, size_t B,
	gf64_t *out, size_t R,
	uint64_t firstInput, uint64_t firstRecovery)
{
	for (size_t r = 0; r < R; r++) {
		gf64_t yr = (gf64_t)(firstRecovery + (uint64_t)r);
		gf64_t *out_row = out + r * B;
		for (size_t w = 0; w < B; w++) out_row[w] = 0;
		for (size_t c = 0; c < N; c++) {
			gf64_t xc = (gf64_t)(firstInput + (uint64_t)c);
			gf64_t denom = gf64_inverse(yr ^ xc);
			const gf64_t *in_row = in + c * B;
			for (size_t w = 0; w < B; w++) {
				out_row[w] ^= gf64_mul_reference(in_row[w], denom);
			}
		}
	}
}

/* ----- xorshift64 PRNG. */
static uint64_t xs_state = 0xDEADBEEFCAFEBABEULL;
static uint64_t xorshift64(void) {
	uint64_t x = xs_state;
	x ^= x << 13; x ^= x >> 7; x ^= x << 17;
	xs_state = x;
	return x;
}

static int test_config(
	size_t N, size_t R, size_t B,
	uint64_t firstInput, uint64_t firstRecovery)
{
	gf64_t *in        = (gf64_t *)malloc(N * B * sizeof(gf64_t));
	gf64_t *cauchy_out = (gf64_t *)calloc(R * B, sizeof(gf64_t));
	gf64_t *fenger_out = (gf64_t *)calloc(R * B, sizeof(gf64_t));
	if (!in || !cauchy_out || !fenger_out) {
		fprintf(stderr, "  alloc fail N=%zu R=%zu B=%zu\n", N, R, B);
		return 1;
	}
	for (size_t i = 0; i < N * B; i++) in[i] = xorshift64();

	cauchy_reference(in, N, B, cauchy_out, R, firstInput, firstRecovery);
	gf64_fenger_matvec (in, N, B, fenger_out, R, firstInput, firstRecovery);

	int ok = 1;
	for (size_t i = 0; i < R * B; i++) {
		if (cauchy_out[i] != fenger_out[i]) {
			fprintf(stderr,
				"  MISMATCH N=%zu R=%zu B=%zu idx=%zu: "
				"cauchy=0x%016llx fenger=0x%016llx\n",
				N, R, B, i,
				(unsigned long long)cauchy_out[i],
				(unsigned long long)fenger_out[i]);
			ok = 0;
			break;
		}
	}
	if (ok) {
		printf("  PASS: N=%zu R=%zu B=%zu firstInput=0x%llx firstRecovery=0x%llx\n",
		       N, R, B,
		       (unsigned long long)firstInput,
		       (unsigned long long)firstRecovery);
	}
	free(in); free(cauchy_out); free(fenger_out);
	return ok ? 0 : 1;
}

int main(void)
{
	printf("Fenger Toeplitz pipeline — Phase 1 bit-exact parity (issue #28)\n");
	printf("Verifying gf64_fenger_matvec against explicit Cauchy matvec\n");
	printf("over GF(2^64)[x] at small (N, R, B).\n\n");

	int failures = 0;
	/* Powers of 2 from the subproduct-tree contract. */
	failures += test_config(1,  1,  1,  0,   100);
	failures += test_config(2,  2,  1,  0,   100);
	failures += test_config(2,  2,  4,  0,   100);
	failures += test_config(4,  2,  2,  5,   200);
	failures += test_config(4,  4,  4,  10,  300);
	failures += test_config(8,  4,  2,  100, 1000);
	failures += test_config(8,  8,  4,  1,   20);
	failures += test_config(16, 4,  2,  0x100, 0x200);
	failures += test_config(16, 8,  4,  0,   500);
	failures += test_config(16, 16, 8, 0,   1000);
	failures += test_config(32, 8,  4,  0,   500);
	failures += test_config(32, 16, 2, 0,   1000);

	printf("\n%s: %d test(s) failed\n", failures == 0 ? "OK" : "FAIL", failures);
	return failures;
}
