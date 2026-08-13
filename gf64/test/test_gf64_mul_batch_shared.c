/*
 * ============================================================================
 * gf64/test/test_gf64_mul_batch_shared.c — batch-shared mul parity tests
 *
 * Issue #59 §4 A3 (word-batched tree passes): K products with one shared
 * operand must be BIT-EXACT to K sequential
 * gf64_addfft64_poly_mul_recursive_scratch calls — the shared forward
 * transform is a pure hoist and the per-word pipeline (fwd, pointwise,
 * inv) is unchanged, so the batch must agree coefficient-for-coefficient.
 *
 * Coverage:
 *   1. HQC-window sizes (n_pad 128..65536), K = 4, uniform lens.
 *   2. Asymmetric shared/f lens (len_shared != len_f).
 *   3. K = 1 (degenerate batch == single mul) and K = 8.
 *   4. AVX-512 batch vs AVX-512 sequential (host-gated).
 *   5. Zero-length operands (all outputs zero).
 *
 * Build & run from gf64/test/:
 *   make test_gf64_mul_batch_shared && ./test_gf64_mul_batch_shared
 *
 * Exits 0 on full pass; non-zero on any failure.
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../gf64_additive_fft.h"
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* ----------------------------------------------------------------------------
 * splitmix64 PRNG.
 * ---------------------------------------------------------------------------- */
static uint64_t g_rng = 0;

static uint64_t splitmix64_next(void) {
	uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static void put_seed(uint64_t s) {
	g_rng = s;
}

/* ----------------------------------------------------------------------------
 * Test framework
 * ---------------------------------------------------------------------------- */
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

/* ----------------------------------------------------------------------------
 * One parity case: K sequential single muls vs one batch call.
 * ---------------------------------------------------------------------------- */
static void run_case(size_t len_shared, size_t len_f, size_t out_len,
                     size_t K, uint64_t seed, int use_avx512) {
	char label[160];
	snprintf(label, sizeof(label),
	         "batch == %zu sequential muls (len_shared=%zu, len_f=%zu, out_len=%zu, %s)",
	         K, len_shared, len_f, out_len, use_avx512 ? "avx512" : "scalar");

	gf64_t *shared = (gf64_t *)malloc(len_shared * sizeof(gf64_t));
	gf64_t *f      = (gf64_t *)malloc(K * len_f * sizeof(gf64_t));
	gf64_t *refs   = (gf64_t *)malloc(K * out_len * sizeof(gf64_t));
	gf64_t *outs   = (gf64_t *)malloc(K * out_len * sizeof(gf64_t));
	if (!shared || !f || !refs || !outs) {
		fail("alloc");
		free(shared); free(f); free(refs); free(outs);
		return;
	}

	put_seed(seed);
	for (size_t i = 0; i < len_shared; i++) shared[i] = splitmix64_next();
	for (size_t i = 0; i < K * len_f; i++) f[i] = splitmix64_next();

	/* Reference: K sequential single muls (same scratch budget). */
	size_t full_len = (len_shared + len_f - 1 > out_len)
	                  ? (len_shared + len_f - 1) : out_len;
	size_t n = 1;
	while (n < full_len) n <<= 1;
	size_t sw = gf64_addfft64_poly_mul_recursive_scratch_words(n);
	gf64_t *scratch = (gf64_t *)malloc(sw * sizeof(gf64_t));
	if (scratch == NULL) abort();

	for (size_t k = 0; k < K; k++) {
		gf64_t *out_k = refs + k * out_len;
		const gf64_t *f_k = f + k * len_f;
		if (use_avx512) {
			gf64_addfft64_poly_mul_recursive_scratch_avx512(
				out_k, shared, len_shared, f_k, len_f, out_len, scratch, sw);
		} else {
			gf64_addfft64_poly_mul_recursive_scratch(
				out_k, shared, len_shared, f_k, len_f, out_len, scratch, sw);
		}
	}

	/* Batch call. */
	gf64_t *out_ptrs[16];
	if (K > 16) abort();
	for (size_t k = 0; k < K; k++) out_ptrs[k] = outs + k * out_len;
	if (use_avx512) {
		gf64_addfft64_poly_mul_batch_shared_avx512(
			out_ptrs, K, shared, len_shared, f, len_f, out_len, scratch, sw);
	} else {
		gf64_addfft64_poly_mul_batch_shared(
			out_ptrs, K, shared, len_shared, f, len_f, out_len, scratch, sw);
	}

	int ok = 1;
	for (size_t k = 0; k < K && ok; k++) {
		for (size_t i = 0; i < out_len; i++) {
			if (outs[k * out_len + i] != refs[k * out_len + i]) {
				printf("    k=%zu coeff[%zu]: batch=0x%016llx ref=0x%016llx\n",
				       k, i,
				       (unsigned long long)outs[k * out_len + i],
				       (unsigned long long)refs[k * out_len + i]);
				ok = 0;
				break;
			}
		}
	}

	if (ok) {
		pass(label);
	} else {
		fail(label);
	}

	free(shared); free(f); free(refs); free(outs); free(scratch);
}

int main(void) {
	printf("GF64 batch-shared mul parity tests (issue #59 A3)\n");
	printf("=================================================\n\n");

	/* HQC-window sizes. len_shared/len_f/out_len chosen so the padded
	 * n lands on the listed value: n = next_pow2(len_shared+len_f-1). */
	run_case(64, 65, 128, 4, 0x1111111111111111ULL, 0); /* n=128   */
	run_case(256, 257, 512, 4, 0x2222222222222222ULL, 0); /* n=512   */
	run_case(2048, 2049, 4096, 4, 0x3333333333333333ULL, 0); /* n=4096 */
	run_case(32768, 32769, 65536, 4, 0x4444444444444444ULL, 0); /* n=65536 */

	/* Asymmetric operands (shared much longer than f). */
	run_case(600, 100, 700, 4, 0x5555555555555555ULL, 0);

	/* Degenerate K=1 and large K=8. */
	run_case(128, 129, 256, 1, 0x6666666666666666ULL, 0);
	run_case(128, 129, 256, 8, 0x7777777777777777ULL, 0);

	/* Truncated output (out_len < full product). */
	run_case(256, 257, 300, 4, 0x8888888888888888ULL, 0);

	/* AVX-512 variants (host-gated). */
	run_case(64, 65, 128, 4, 0x9999999999999999ULL, 1);
	run_case(2048, 2049, 4096, 8, 0xAAAAAAAAAAAAAAAAULL, 1);

	/* Zero-length operands: all outputs must be zero. */
	{
		gf64_t shared[8] = {0};
		gf64_t f[8] = {0};
		gf64_t outs[4][8];
		memset(outs, 0xAA, sizeof(outs));
		gf64_t *ptrs[4] = { outs[0], outs[1], outs[2], outs[3] };
		size_t sw = gf64_addfft64_poly_mul_recursive_scratch_words(8);
		gf64_t *scratch = (gf64_t *)malloc(sw * sizeof(gf64_t));
		if (scratch == NULL) abort();
		gf64_addfft64_poly_mul_batch_shared(ptrs, 4, shared, 0, f, 8, 8, scratch, sw);
		int ok = 1;
		for (size_t k = 0; k < 4 && ok; k++) {
			for (size_t i = 0; i < 8; i++) {
				if (outs[k][i] != 0) { ok = 0; break; }
			}
		}
		if (ok) pass("zero-length shared operand -> all-zero outputs");
		else fail("zero-length shared operand -> all-zero outputs");
		free(scratch);
	}

	printf("\n=== Summary ===\n");
	printf("Passed: %d\n", g_passed);
	printf("Failed: %d\n", g_failed);

	return g_failed > 0 ? 1 : 0;
}
