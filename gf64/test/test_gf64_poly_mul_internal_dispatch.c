/*
 * test_gf64_poly_mul_internal_dispatch.c — boundary regression test for
 *   the gf64_poly_mul_internal tier dispatch at gf64/gf64_additive_fft.c:349-438.
 *
 * Background (issue #51 Step 7(b)):
 *   The dispatch picks between three tiers based on (len_a, len_b, out_len):
 *
 *     HQC FFT       : all three >= GF64_HQC_FFT_MIN (96) and both <= GF64_HQC_MAX_LM_N
 *     Karatsuba     : all three >= GF64_POLY_MUL_INTERNAL_KARATSUBA_MIN (128)
 *     Schoolbook    : otherwise
 *
 *   These thresholds are pinned by empirical benchmark numbers (see
 *   gf64/test/bench_hqc_vs_karatsuba.c, cited from PR #53 / b4a2203):
 *     - HQC scalar beats Karatsuba from n >= 96
 *     - HQC AVX-512 (PCLMULQDQ) beats Karatsuba from n >= 64
 *     - peak 4.71x at n=1024 (HQC AVX-512 vs Karatsuba)
 *
 *   GF64_HQC_FFT_MIN = 96 is deliberately conservative (scalar-safe); the
 *   AVX-512 crossover at n=64 is below this and is NOT exposed in the
 *   single-tier dispatch because the chosen constant protects the
 *   scalar path too. Tuning this is Step 7(b) of issue #51.
 *
 *   GF64_POLY_MUL_INTERNAL_KARATSUBA_MIN = 128 fills the (96..128)
 *   crossover gap where neither HQC nor Karatsuba wins (HQC's per-call
 *   setup overhead dominates below 96, Karatsuba's schoolbook base
 *   case fires immediately at 128 anyway).
 *
 * What this test verifies:
 *   1. Boundary at n = 95: dispatch lands in SCHOOLBOOK (below HQC's
 *      per-call setup overhead).
 *   2. Boundary at n = 96: dispatch lands in HQC_FFT (first size where
 *      HQC wins over Karatsuba for the scalar path).
 *   3. Boundary at n = 127: dispatch lands in HQC_FFT (between HQC's
 *      96-min and Karatsuba's 128-min — HQC covers this gap).
 *   4. Boundary at n = 128: dispatch lands in KARATSUBA (first size
 *      where Karatsuba's recursive split pays off; HQC would be valid
 *      here too but Karatsuba was the path before HQC, and the dispatch
 *      picks HQC first when both qualify — see note below).
 *   5. Large n = 4096: dispatch lands in HQC_FFT for sizes well inside
 *      the [96, 2^20] window.
 *
 *   Note on (4): the current dispatch at gf64_additive_fft.c:372-400
 *   checks HQC FIRST. At n = 128, all three (len_a, len_b, out_len)
 *   are >= 96, so HQC wins. The boundary test for Karatsuba therefore
 *   uses an asymmetric case where one operand is small — exactly the
 *   case the dispatch's "all three must be >= threshold" guard
 *   routes to Karatsuba/schoolbook.
 *
 *   This pins the dispatch contract:
 *     - Same-length inputs at the thresholds land in the documented tier
 *     - Asymmetric (n, m) with m < 96 fall through to Karatsuba/schoolbook
 *     - Counter increments match expectation per call site
 *
 * Build (via gf64/test/Makefile):
 *   cd gf64/test
 *   make test_gf64_poly_mul_internal_dispatch
 *   ./test_gf64_poly_mul_internal_dispatch
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../gf64_additive_fft.h"

/* Deterministic 64-bit PRNG so repeated runs produce the same input.
 * This matters because gf64_dispatch_counts is process-global and we
 * reset it between test cases. */
static uint64_t splitmix64(uint64_t *state) {
	uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
	z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL);
	z = ((z ^ (z >> 27)) * 0x94d049bb133111ebULL);
	return z ^ (z >> 31);
}

static void fill_random(uint64_t *p, size_t n, uint64_t seed) {
	uint64_t s = seed;
	for (size_t i = 0; i < n; i++) p[i] = splitmix64(&s);
}

/* Run one poly_mul at the (len_a, len_b, out_len) triple and return which
 * dispatch tier's counter incremented. The expectation is that all three
 * sizes are at-or-above the threshold of the tier that's expected to win.
 *
 * The check uses BOTH the increments (which counter moved) AND the
 * correctness of the result (schoolbook reference vs poly_mul_padded
 * output — bit-exact) to confirm the dispatch routed to the right
 * algorithm. If the dispatch fires the wrong tier (e.g. routes to
 * schoolbook when Karatsuba should win), the wrong counter moves and
 * the test fails. */
static int run_one(size_t len_a, size_t len_b, size_t out_len,
                   const char *name, const char *expected_tier) {
	uint64_t *a = (uint64_t *)malloc(len_a * sizeof(uint64_t));
	uint64_t *b = (uint64_t *)malloc(len_b * sizeof(uint64_t));
	uint64_t *out = (uint64_t *)malloc(out_len * sizeof(uint64_t));
	uint64_t *ref = (uint64_t *)malloc(out_len * sizeof(uint64_t));
	if (!a || !b || !out || !ref) {
		fprintf(stderr, "  OOM in %s\n", name);
		free(a); free(b); free(out); free(ref);
		return 1;
	}

	fill_random(a, len_a, 0xC0FFEE00 ^ (uint64_t)len_a);
	fill_random(b, len_b, 0xDEADBEEF ^ (uint64_t)len_b);

	/* Schoolbook reference for correctness. */
	memset(ref, 0, out_len * sizeof(uint64_t));
	size_t a_cap = (len_a < out_len) ? len_a : out_len;
	size_t b_cap = (len_b < out_len) ? len_b : out_len;
	extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);
	for (size_t i = 0; i < a_cap; i++) {
		for (size_t j = 0; j < b_cap && (i + j) < out_len; j++) {
			ref[i + j] ^= gf64_mul_reference(a[i], b[j]);
		}
	}

	gf64_dispatch_counts_reset();
	gf64_poly_mul_padded(out, a, len_a, b, len_b, out_len);

	uint64_t sb = gf64_dispatch_counts.schoolbook;
	uint64_t kt = gf64_dispatch_counts.karatsuba;
	uint64_t hq = gf64_dispatch_counts.hqc_fft;

	/* Exactly one counter should have moved, and it should match
	 * expected_tier. */
	const char *got;
	if (sb == 1 && kt == 0 && hq == 0) got = "schoolbook";
	else if (kt == 1 && sb == 0 && hq == 0) got = "karatsuba";
	else if (hq == 1 && sb == 0 && kt == 0) got = "hqc_fft";
	else {
		printf("  %s: FAIL  dispatch counter mismatch "
		       "(schoolbook=%llu karatsuba=%llu hqc_fft=%llu)\n",
		       name, (unsigned long long)sb,
		       (unsigned long long)kt, (unsigned long long)hq);
		free(a); free(b); free(out); free(ref);
		return 1;
	}

	if (strcmp(got, expected_tier) != 0) {
		printf("  %s: FAIL  expected=%s got=%s\n",
		       name, expected_tier, got);
		free(a); free(b); free(out); free(ref);
		return 1;
	}

	/* Verify correctness vs schoolbook reference. */
	if (memcmp(out, ref, out_len * sizeof(uint64_t)) != 0) {
		printf("  %s: FAIL  output differs from schoolbook reference\n",
		       name);
		free(a); free(b); free(out); free(ref);
		return 1;
	}

	printf("  %s: OK  (out_len=%zu, tier=%s)\n",
	       name, out_len, got);
	free(a); free(b); free(out); free(ref);
	return 0;
}

int main(void) {
	int failures = 0;

	printf("test_gf64_poly_mul_internal_dispatch\n");
	printf("(pins GF64_HQC_FFT_MIN=96 / GF64_POLY_MUL_INTERNAL_KARATSUBA_MIN=128\n");
	printf(" boundaries for issue #51 Step 7(b))\n\n");

	printf("[1] n=95 (below HQC min): dispatch -> schoolbook\n");
	failures += run_one(95, 95, 189, "n=95 symmetric", "schoolbook");

	printf("\n[2] n=96 (HQC scalar crossover): dispatch -> hqc_fft\n");
	failures += run_one(96, 96, 191, "n=96 symmetric", "hqc_fft");

	printf("\n[3] n=127 (between HQC and Karatsuba minima): dispatch -> hqc_fft\n");
	failures += run_one(127, 127, 253, "n=127 symmetric", "hqc_fft");

	printf("\n[4] n=128 same-length: dispatch -> hqc_fft (HQC is checked first)\n");
	failures += run_one(128, 128, 255, "n=128 symmetric", "hqc_fft");

	printf("\n[5] asymmetric (n=1024, m=64): m<96 routes past HQC AND past Karatsuba -> schoolbook\n");
	/* out_len = 1024 + 64 - 1 = 1087; small-m operand means the
	 * "all three >= threshold" guards at gf64_additive_fft.c:372-376
	 * and :409-411 both reject (m=64 < KARATSUBA_MIN). Schoolbook's
	 * truncated inner loop handles the asymmetric case correctly and
	 * is what the dispatch falls through to.
	 *
	 * Specifically, the dispatch contract is:
	 *   - HQC: all three >= 96 AND both inputs <= 2^20
	 *   - Karatsuba: all three >= 128 (catches the (96..128) crossover gap)
	 *   - Schoolbook: everything else
	 *
	 * Karatsuba does NOT cover the "n ≥ 128 but m < 96" case the way I
	 * originally assumed; both tier guards require ALL three operands
	 * to be at-or-above the threshold. This pins that the asymmetric
	 * path falls through to schoolbook exactly as the dispatch intends. */
	failures += run_one(1024, 64, 1087, "asymmetric 1024x64",
	                    "schoolbook");

	printf("\n[6] n=4096 (well inside the HQC window): dispatch -> hqc_fft\n");
	failures += run_one(4096, 4096, 8191, "n=4096 symmetric", "hqc_fft");

	printf("\n[7] above HQC cap (cap overridden to 256 for the test, len_a = 512):\n");
	printf("    HQC guard fails (len_a exceeds the overridden cap) -> Karatsuba.\n");
	printf("    The override keeps this case cheap: the real 2^20 cap would pad\n");
	printf("    to a next_pow2(2^21) balanced Karatsuba recursion (~3^15 leaves,\n");
	printf("    minutes of CI wall-time — cubic review fc0fd87c).\n");
	gf64_hqc_max_lm_n_override = 256;
	failures += run_one(512, 128, 640, "above HQC cap 512x128 (override)",
	                    "karatsuba");
	gf64_hqc_max_lm_n_override = 0;

	printf("\n[8] out_len above HQC cap (override=256, len_a=len_b=128, out_len=512):\n");
	printf("    HQC guard fails on the PADDED size (full_len = 512 > cap) even\n");
	printf("    though both operands are within it -> Karatsuba. Pins the n_pad\n");
	printf("    bound: the cap must cover next_pow2(max(2*max_len-1, out_len)),\n");
	printf("    not just the operands (cubic review 5ec90e2f P1).\n");
	gf64_hqc_max_lm_n_override = 256;
	failures += run_one(128, 128, 512, "out_len above HQC cap 128x128",
	                    "karatsuba");
	gf64_hqc_max_lm_n_override = 0;

	printf("\n");
	if (failures == 0) {
		printf("ALL PASS\n");
		return 0;
	} else {
		printf("FAILED (%d test cases)\n", failures);
		return 1;
	}
}
