/*
 * ============================================================================
 * gf64/test/bench_gf64_poly_mul_karatsuba.c
 *
 * Phase 2a microbench: compare Karatsuba polynomial multiplication against
 * the schoolbook baseline at the sizes the Newton iteration in
 * gf64_poly_invmod actually uses. Times a single (a, b, out_len) triple
 * averaged over multiple runs.
 *
 * Targets the WSL Ubuntu + Zen4 8-core build with -march=native. The
 * Karatsuba win at large sizes should be obvious; at small sizes the
 * schoolbook baseline (no malloc, no recursion) wins on constant overhead.
 *
 * Build & run:
 *   $(CC) -O2 -march=native -I.. bench_gf64_poly_mul_karatsuba.c \
 *         ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c \
 *         ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c \
 *         ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c \
 *         -o bench_gf64_poly_mul_karatsuba && ./bench_gf64_poly_mul_karatsuba
 *
 * Output (per size): schoolbook µs/call, karatsuba µs/call, ratio.
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../gf64_additive_fft.h"
#include "../gf64_poly_mul_karatsuba.h"
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

static uint64_t g_rng = 0xFEEDFACE12345678ULL;
static uint64_t splitmix64_next(void) {
	uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

/* Reference schoolbook (no dispatch, always schoolbook). Used as the
 * baseline for "what Karatsuba has to beat". */
static void schoolbook(
	gf64_t *out,
	const gf64_t *a, size_t len_a,
	const gf64_t *b, size_t len_b,
	size_t out_len
) {
	memset(out, 0, out_len * sizeof(gf64_t));
	size_t a_cap = (len_a < out_len) ? len_a : out_len;
	size_t b_cap = (len_b < out_len) ? len_b : out_len;
	for (size_t i = 0; i < a_cap; i++) {
		gf64_t ai = a[i];
		if (ai == 0) continue;
		size_t j_max = (b_cap < out_len - i) ? b_cap : (out_len - i);
		for (size_t j = 0; j < j_max; j++) {
			out[i + j] ^= gf64_mul_reference(ai, b[j]);
		}
	}
}

static double seconds_now(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

typedef struct {
	const char *name;
	size_t la, lb, ol;
} case_t;

int main(void) {
	printf("Phase 2a microbench: schoolbook vs Karatsuba poly_mul\n");
	printf("======================================================\n\n");
	printf("%-32s %14s %14s %8s\n", "case", "schoolbook µs", "karatsuba µs", "ratio");

	/* Sizes chosen to match the Newton-iteration doubling schedule in
	 * gf64_poly_invmod for n = 64, 256, 1024, 4096. The squaring at each
	 * iteration is m x m -> 2m-1 (balanced); the cross-product is
	 * n x (2m-1) -> m_new (essentially balanced after Karatsuba's
	 * power-of-2 padding). */
	case_t cases[] = {
		{"Newton n=64 m=64 squaring",     64,   64,   127},
		{"Newton n=64 final cross",       64,   127,  64},
		{"Newton n=256 m=128 squaring",   128,  128,  255},
		{"Newton n=256 m=256 squaring",   256,  256,  511},
		{"Newton n=256 final cross",      256,  511,  256},
		{"Newton n=1024 m=512 squaring",  512,  512,  1023},
		{"Newton n=1024 m=1024 squaring", 1024, 1024, 2047},
		{"Newton n=1024 final cross",     1024, 2047, 1024},
		{"Newton n=4096 m=2048 squaring", 2048, 2048, 4095},
		{"Newton n=4096 m=4096 squaring", 4096, 4096, 8191},
		{"Newton n=4096 final cross",     4096, 8191, 4096},
	};

	int n_cases = sizeof(cases)/sizeof(cases[0]);
	int iters_per_case = 5;

	for (int c = 0; c < n_cases; c++) {
		size_t la = cases[c].la, lb = cases[c].lb, ol = cases[c].ol;
		gf64_t *a = (gf64_t *)malloc(la * sizeof(gf64_t));
		gf64_t *b = (gf64_t *)malloc(lb * sizeof(gf64_t));
		gf64_t *out = (gf64_t *)malloc(ol * sizeof(gf64_t));
		for (size_t i = 0; i < la; i++) a[i] = splitmix64_next();
		for (size_t i = 0; i < lb; i++) b[i] = splitmix64_next();

		/* Warmup. */
		schoolbook(out, a, la, b, lb, ol);
		gf64_poly_mul_karatsuba(out, a, la, b, lb, ol);

		double t0 = seconds_now();
		for (int it = 0; it < iters_per_case; it++) {
			schoolbook(out, a, la, b, lb, ol);
		}
		double t1 = seconds_now();
		for (int it = 0; it < iters_per_case; it++) {
			gf64_poly_mul_karatsuba(out, a, la, b, lb, ol);
		}
		double t2 = seconds_now();

		double sb_us = (t1 - t0) / iters_per_case * 1e6;
		double kt_us = (t2 - t1) / iters_per_case * 1e6;
		double ratio = sb_us / kt_us;

		printf("%-32s %14.2f %14.2f %7.2fx\n",
		       cases[c].name, sb_us, kt_us, ratio);

		free(a); free(b); free(out);
	}
	return 0;
}