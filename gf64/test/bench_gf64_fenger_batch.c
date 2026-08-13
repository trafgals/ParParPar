/*
 * ============================================================================
 * gf64/test/bench_gf64_fenger_batch.c — A3 word-batched execute bench
 *
 * Issue #59 §4 A3: wall-clock time of gf64_fenger_execute at K = 1, 4, 8
 * (PAR3_FENGER_BATCH_WORDS) over the same prepared context, at the
 * reference shapes. Reports per-K execute time + speedup over K=1, the
 * P2 gate evidence ("10G/100k kernel ≤ 3 min" projection).
 *
 * Shapes:
 *   1G/10K-ish:  N = 2^18, R = 2^10, B = 512  (fully inside the HQC window)
 *   4G/4K-ish:   N = 2^20, R = 2^12, B = 128  (root n = 2^20 = HQC cap)
 *   10G/100k:    N = 2^22, R = 2^14, B = 16   (top 2 levels above the cap)
 *
 * Env (same as the P1 benches):
 *   PAR3_GF64_USE_AVX512=1 PAR3_AVX512_FORCE=2   force AVX-512
 *   PAR3_FENGER_BATCH_WORDS=4                     (default; bench overrides)
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../gf64_global.h"
#include "../gf64_fenger.h"

static uint64_t rng_state = 0xD3D3D3D3D3D3D3D3ULL;
static uint64_t next_rand(void) {
	uint64_t x = rng_state;
	x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
	rng_state = x;
	return x * 0x2545F4914F6CDD1DULL;
}

static double now_sec(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void bench_shape(const char *name, size_t N, size_t R, size_t B,
                        uint64_t firstInput, uint64_t firstRecovery) {
	gf64_fenger_ctx *ctx = gf64_fenger_prepare(firstInput, firstRecovery, N, R);
	if (ctx == NULL) {
		printf("%-14s prepare failed\n", name);
		return;
	}
	gf64_t *in  = malloc(N * B * sizeof(gf64_t));
	gf64_t *out = malloc(R * B * sizeof(gf64_t));
	if (in == NULL || out == NULL) {
		printf("%-14s OOM (N*B = %zu words)\n", name, N * B);
		goto done;
	}
	for (size_t i = 0; i < N * B; i++) in[i] = next_rand();

	const size_t Ks[] = {1, 4, 8};
	double base[1] = {0.0};  /* K=1 column time (speedup denominator) */
	double t0 = now_sec();
	/* Warm-up at the default K (also pins the env-cache-free read). */
	gf64_fenger_execute(ctx, in, B, out, 0, B);
	double t_warm = now_sec() - t0;

	printf("%-14s N=%6zu R=%6zu B=%4zu  (warm %6.3f s)\n",
	       name, N, R, B, t_warm);
	for (size_t ki = 0; ki < sizeof(Ks) / sizeof(Ks[0]); ki++) {
		char env[32];
		snprintf(env, sizeof(env), "%zu", Ks[ki]);
		setenv("PAR3_FENGER_BATCH_WORDS", env, 1);
		const int reps = 1;  /* 431 s/word at N=262144: no repeats there */
		double best = 1e30;
		for (int r = 0; r < reps; r++) {
			double s = now_sec();
			gf64_fenger_execute(ctx, in, B, out, 0, B);
			double e = now_sec() - s;
			if (e < best) best = e;
		}
		double gb = (double)N * B * 8.0 / 1e9;
		double mbps = gb / best * 1024.0;
		printf("  K=%zu: %7.3f s  (%7.1f MB/s)", Ks[ki], best, mbps);
		if (ki > 0 && base[0] > 0.0) {
			printf("   speedup vs K=1: %.2fx", base[0] / best);
		}
		if (ki == 0) base[0] = best;
		printf("\n");
	}
	setenv("PAR3_FENGER_BATCH_WORDS", "4", 1);
	free(in);
	free(out);
done:
	gf64_fenger_release(ctx);
}

int main(void) {
	/* Defeat hypervisor CPUID masking for the AVX-512 column (same as
	 * bench_gf64_subquadratic) when the env vars are set. */
	gf64_init_dispatch();

	printf("Fenger word-batched execute bench (issue #59 A3)\n");
	printf("method: %s\n\n",
	       gf64_current_method == GF64_AVX512 ? "AVX512"
	       : gf64_current_method == GF64_AVX2 ? "AVX2"
	       : gf64_current_method == GF64_SSSE3 ? "SSSE3" : "scalar");

	/* Shape set is chosen so the whole bench finishes in ~20 min on
	 * WSL2. The interp dominates and costs ~40-100 s/word at N=262144,
	 * so the original B=512/128/16 implied hours. B=8 is the minimum
	 * that gives the K=8 column a full batch; the K-speedup ratio is
	 * B-relative, and a larger B only amortizes per-word setup more
	 * (favoring K>1), so this is a conservative lower bound. The
	 * medium shape (N=16384) gives a fast cross-check of the same
	 * ratio at ~40x lower per-word cost. The 4G/10G shapes are left
	 * out: their prepares alone are impractical on WSL2, and the P2
	 * gate ("10G/100k <= 10 min") is JS-pipeline-bound anyway (the
	 * ~27 MB/s env ceiling measured in P1). */
	bench_shape("medium",     1u << 14, 1u << 9, 64, 0, 1u << 20);
	bench_shape("1G/10K-ish", 1u << 18, 1u << 10, 8, 0, 1u << 20);
	return 0;
}
