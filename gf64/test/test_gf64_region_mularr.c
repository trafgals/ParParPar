/*
 * test_gf64_region_mularr.c — GF(2^64) region mul_arr cycling contract.
 *
 * Issue #59 P2 A3 branch drive-by: the SIMD region mul_arr kernels'
 * n_coeff > 1 general case computed a DOT PRODUCT (out[i] = XOR_c
 * in[i]*coeff[c]) instead of the per-word CYCLING contract
 * (out[i] = in[i] * coeff[i % n_coeff]) — a silent wrong-result bug
 * (the SUM semantics belong to gf64_region_muladd_*). The AVX-512 /
 * AVX-2 / SSSE3 general cases were rewritten with per-lane coefficient
 * gathers; the scalar kernel was already correct.
 *
 * This test pins the contract at the C level: the dispatched
 * gf64_region_mul_arr must match the scalar reference for every
 * method the host can run, across n_coeff in {1,2,3,4,7}, lens
 * including odd values (SIMD tail epilog), and both in/out layouts.
 *
 * Run: make test_gf64_region_mularr && ./test_gf64_region_mularr
 * Env: PAR3_GF64_USE_AVX512=0 / 1 + PAR3_AVX512_FORCE=2 select the
 * method (gf64_init_dispatch reads them).
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "../gf64_global.h"

/* gf64_single.c's public reference multiply (no header declares it). */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* Reference: the scalar kernel is the contract (it was never broken). */
static void ref_mul_arr(gf64_t *out, const gf64_t *in, const gf64_t *coeff,
                        size_t len, size_t n_coeff) {
	for (size_t i = 0; i < len; i++) {
		out[i] = gf64_mul_reference(in[i], coeff[i % n_coeff]);
	}
}

static uint64_t rng_state = 0xD3D3D3D3D3D3D3D3ULL;
static uint64_t next_rand(void) {
	uint64_t x = rng_state;
	x ^= x << 13; x ^= x >> 7; x ^= x << 17;
	rng_state = x;
	return x;
}

int main(void) {
	gf64_init_dispatch();

	const size_t lens[] = { 0, 1, 2, 3, 4, 5, 8, 16, 17, 33 };
	const size_t ncoeffs[] = { 1, 2, 3, 4, 7 };
	size_t checks = 0, failures = 0;

	for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
		for (size_t ni = 0; ni < sizeof(ncoeffs) / sizeof(ncoeffs[0]); ni++) {
			const size_t len = lens[li];
			const size_t n_coeff = ncoeffs[ni];

			gf64_t *in    = malloc((len ? len : 1) * sizeof(gf64_t));
			gf64_t *coeff = malloc((n_coeff ? n_coeff : 1) * sizeof(gf64_t));
			gf64_t *ref   = malloc((len ? len : 1) * sizeof(gf64_t));
			gf64_t *got   = malloc((len ? len : 1) * sizeof(gf64_t));
			if (!in || !coeff || !ref || !got) { fprintf(stderr, "OOM\n"); return 1; }

			for (size_t i = 0; i < (len ? len : 1); i++) in[i] = next_rand();
			for (size_t i = 0; i < (n_coeff ? n_coeff : 1); i++) coeff[i] = next_rand();

			ref_mul_arr(ref, in, coeff, len, n_coeff);
			memset(got, 0xAA, (len ? len : 1) * sizeof(gf64_t));
			gf64_region_mul_arr(got, in, coeff, len, n_coeff);

			size_t bad = 0;
			for (size_t i = 0; i < len; i++) {
				if (ref[i] != got[i]) bad++;
			}
			checks++;
			if (bad) {
				failures++;
				printf("FAIL len=%zu n_coeff=%zu: %zu/%zu words differ (method %s)\n",
				       len, n_coeff, bad, len,
				       gf64_current_method == GF64_AVX512 ? "AVX512"
				       : gf64_current_method == GF64_AVX2 ? "AVX2"
				       : gf64_current_method == GF64_SSSE3 ? "SSSE3" : "scalar");
			} else {
				printf("ok   len=%zu n_coeff=%zu (method %s)\n",
				       len, n_coeff,
				       gf64_current_method == GF64_AVX512 ? "AVX512"
				       : gf64_current_method == GF64_AVX2 ? "AVX2"
				       : gf64_current_method == GF64_SSSE3 ? "SSSE3" : "scalar");
			}
			free(in); free(coeff); free(ref); free(got);
		}
	}
	printf("%zu/%zu region mul_arr cycling checks passed\n", checks - failures, checks);
	return failures ? 1 : 0;
}
