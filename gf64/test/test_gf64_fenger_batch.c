/*
 * ============================================================================
 * gf64/test/test_gf64_fenger_batch.c — A3 word-batched execute parity tests
 *
 * Issue #59 §4 A3: the per-word interp/eval passes are independent, so the
 * Fenger execute processes K words per batch, sharing per-node transforms
 * (interp combine) and Newton reciprocals (MPE divmods) across the K words.
 *
 * The batched pipeline performs EXACTLY the same field operations per word
 * as the single-word path (shared work is a pure hoist), so the outputs
 * must be bit-identical. This test runs the same prepared context through
 * gf64_fenger_execute with PAR3_FENGER_BATCH_WORDS=1 (single-word
 * reference) vs 4/8 (batched) and compares every word of every real
 * recovery row.
 *
 * Shapes cover: tiny/all-fallback levels (N=1,2,8), mixed (1024, 16384),
 * HQC-heavy (262144 ≈ the 1G/10K reference shape), a B < K tail-clamping
 * case, and both padded shapes (1000→1024, and N=5/R=2 — the e2e golden
 * shape).
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../gf64_fenger.h"
#include "../gf64_global.h"

static int failures = 0, checks = 0;

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t next_rand(void) {
	uint64_t x = rng_state;
	x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
	rng_state = x;
	return x * 0x2545F4914F6CDD1DULL;
}

static void run_case(const char *name,
	uint64_t firstInput, uint64_t firstRecovery,
	size_t numInputs, size_t numRecovery,
	size_t numInputsPadded, size_t numRecoveryPadded,
	uint64_t syntheticBase,
	size_t B, const size_t *Ks, size_t numKs)
{
	gf64_fenger_ctx *ctx;
	if (syntheticBase == 0) {
		ctx = gf64_fenger_prepare(firstInput, firstRecovery, numInputs, numRecovery);
	} else {
		ctx = gf64_fenger_prepare_padded(firstInput, firstRecovery,
			numInputs, numRecovery, numInputsPadded, numRecoveryPadded,
			syntheticBase);
	}
	if (ctx == NULL) {
		printf("FAIL %s: prepare returned NULL\n", name);
		failures++;
		return;
	}

	const size_t N_pad = numInputsPadded ? numInputsPadded : numInputs;
	const size_t R_pad = numRecoveryPadded ? numRecoveryPadded : numRecovery;
	gf64_t *in = malloc(N_pad * B * sizeof(gf64_t));
	gf64_t *out_ref = malloc(R_pad * B * sizeof(gf64_t));
	gf64_t *out_batch = malloc(R_pad * B * sizeof(gf64_t));
	if (in == NULL || out_ref == NULL || out_batch == NULL) {
		printf("FAIL %s: OOM\n", name);
		failures++;
		goto done;
	}
	for (size_t i = 0; i < N_pad * B; i++) {
		in[i] = next_rand();
	}

	for (size_t ki = 0; ki < numKs; ki++) {
		char env[32];
		snprintf(env, sizeof(env), "%zu", Ks[ki]);
		setenv("PAR3_FENGER_BATCH_WORDS", env, 1);
		memset(out_batch, 0, R_pad * B * sizeof(gf64_t));
		gf64_fenger_execute(ctx, in, B, out_batch, 0, B);
		if (ki == 0) {
			memcpy(out_ref, out_batch, R_pad * B * sizeof(gf64_t));
			continue;
		}
		int bad = 0;
		for (size_t r = 0; r < numRecovery && !bad; r++) {
			for (size_t w = 0; w < B && !bad; w++) {
				if (out_ref[r * B + w] != out_batch[r * B + w]) {
					bad = 1;
				}
			}
		}
		checks++;
		if (bad) {
			printf("FAIL %s (K=%zu): mismatch vs K=1\n", name, Ks[ki]);
			failures++;
		} else {
			printf("ok   %s (K=%zu)\n", name, Ks[ki]);
		}
	}

	free(in);
	free(out_ref);
	free(out_batch);
done:
	gf64_fenger_release(ctx);
}

int main(void) {
	const size_t K4[]  = {1, 4};
	const size_t K484[] = {1, 4, 8};
	/* B1-1.4: the K cap was lifted 16 -> 32 (the transform sharing grows
	 * with K); the medium shapes pin K=16/32 bit-exactness. */
	const size_t K1632[] = {1, 16, 32};

	/* Pick the best available SIMD method (standalone binaries default to
	 * scalar, which would make the N=262144 case take minutes). */
	gf64_init_dispatch();

	run_case("N=1/R=2 (crash shape)", 0, 2, 1, 2, 0, 0, 0, 16, K4, 2);
	run_case("N=1/R=1", 0, 1, 1, 1, 0, 0, 0, 8, K4, 2);
	run_case("N=2/R=2", 0, 4, 2, 2, 0, 0, 0, 8, K4, 2);
	/* Walk-only shapes: R > N so deg_p < R and the tree walk runs (not
	 * the Horner fallback). N=4/R=1024 exercises the per-word divmod
	 * fallback; N=512/R=65536 reaches an HQC-eligible level with
	 * deg_fs[k] >= child_deg (the shared-reciprocal branch). */
	run_case("N=4/R=1024 (walk fallback)", 0, 4096, 4, 1024, 0, 0, 0, 8, K4, 2);
	run_case("N=512/R=65536 (walk HQC)", 0, 1048576, 512, 65536, 0, 0, 0, 8, K4, 2);
	run_case("N=8/R=4", 0, 16, 8, 4, 0, 0, 0, 16, K4, 2);
	run_case("N=1024/R=32", 0, 4096, 1024, 32, 0, 0, 0, 16, K1632, 3);
	run_case("N=16384/R=512", 0, 65536, 16384, 512, 0, 0, 0, 16, K1632, 3);
	run_case("N=262144/R=1024 (1G/10K-ish)", 0, 1000000, 262144, 1024, 0, 0, 0, 8, K4, 2);
	run_case("N=1024/R=32 B=5 (K_eff tail)", 0, 4096, 1024, 32, 0, 0, 0, 5, K484, 3);
	run_case("padded N=1000/R=512", 0, 65536, 1000, 512, 1024, 512, 1000000, 16, K4, 2);
	run_case("padded N=5/R=2 (golden shape)", 0, 1024, 5, 2, 8, 2, 2000000, 16, K4, 2);

	printf("%d/%d batched-execute parity checks passed\n",
		checks - failures, checks);
	return failures ? 1 : 0;
}
