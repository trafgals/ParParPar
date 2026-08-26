/*
 * ============================================================================
 * test_gf64_fenger_prepare_batch.c — T2 (issue #59) parity + bench test.
 *
 * Issue #59 P2.2 / T2: the Fenger Toeplitz pipeline's prepare phase
 *   (gf64_fenger_prepare_core) previously called gf64_invert_ita_one
 *   scalar-style N times to build V_prime (line ~158-160 in the original)
 *   and R times to build V_at_y_inv (line ~174-176). Each call is 63
 *   squarings + 62 multiplications; at N=262144/R=32768 (1 GiB / 262144 /
 *   4 KiB bench target shape) that is ~295k scalar inversions per prepare.
 *
 * The fix routes both loops through the existing 8-lane ZMM batch
 * primitive gf64_invert_ita_batch (gf64/gf64_invert_ita_avx512.c:76-125)
 * when the host has a fully functional AVX-512F + VPCLMULQDQ pipeline
 * (gf64_has_vpclmulqdq, the CPUID+XCR0+ZMM-probe-ANDed flag). On hosts
 * without that ISA the existing scalar loop is taken — the bit-exact
 * output is identical (this test's primary gate). The batch primitive
 * handles the count % 8 tail internally (gf64_invert_ita_avx512.c:121-124),
 * so no round-down happens at the call site.
 *
 * What this test asserts
 * ----------------------
 * Two layers of parity:
 *   1. PRIMITIVE LAYER: gf64_invert_ita_batch(out, in, N) is bit-equal to
 *      element-wise gf64_invert_ita_one(in[i]) at N ∈ {8, 1024, 4099 (odd
 *      tail — exercises the count % 8 scalar fallback at 4099 % 8 == 3),
 *      65536}.
 *   2. PIPELINE LAYER: full gf64_fenger_matvec (which now uses the
 *      batched prepare internally) is bit-equal to the explicit Cauchy
 *      reference for N ∈ {8, 1024, 4099 padded to 8192, 65536}. The
 *      Cauchy reference is an independent computation that does NOT use
 *      V_prime / V_at_y_inv at all — so this gate is end-to-end on the
 *      prepare-phase bit-exactness.
 *
 * Bench (printed, not asserted)
 * -----------------------------
 * Wall-time of the inversion section at N=131072 — the inversion loop
 * alone, both the scalar reference and the batch primitive. Records
 * per-call ns and the speedup factor. Records "no AVX-512" if the host
 * lacks working VPCLMULQDQ.
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../gf64_fenger.h"
#include "../gf64_global.h"
#include "../gf64_invert_ita.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);
extern gf64_t gf64_inverse(gf64_t a);
extern gf64_t gf64_invert_ita_one(gf64_t a);
extern void   gf64_invert_ita_batch(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT in,
	size_t N);

static uint64_t g_rng = 0x123456789ABCDEFULL;
static uint64_t splitmix64_next(void) {
	uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

/* ----------------------------------------------------------------------------
 * Layer 1: primitive parity — gf64_invert_ita_batch vs scalar loop.
 *
 * Asserts out_batch[i] == gf64_invert_ita_one(in[i]) for every i in [0, N).
 * Covers the odd-tail shape N=4099 (4099 % 8 == 3) to lock in the
 * gf64_invert_ita_batch's internal scalar tail epilog.
 * --------------------------------------------------------------------------*/
static int test_primitive_parity(size_t N)
{
	gf64_t *in        = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *scalar    = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *batch_out = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (!in || !scalar || !batch_out) {
		fprintf(stderr, "  OOM at N=%zu\n", N);
		free(in); free(scalar); free(batch_out);
		return 1;
	}

	/* Fixed seed per shape so the test is deterministic across runs. */
	g_rng = (uint64_t)0xABCDEF0123456789ULL ^ (uint64_t)N;
	for (size_t i = 0; i < N; i++) in[i] = splitmix64_next();

	for (size_t i = 0; i < N; i++) scalar[i] = gf64_invert_ita_one(in[i]);

	/* Always call the batch primitive — on non-AVX-512 hosts the call
	 * body is non-empty scalar code under the TU_BODY guard (or a no-op
	 * stub that still leaves the array as 0); either way the loop
	 * succeeds. We compare against the scalar reference in both
	 * branches. */
	gf64_invert_ita_batch(batch_out, in, N);

	int bad = 0;
	for (size_t i = 0; i < N; i++) {
		if (scalar[i] != batch_out[i]) {
			if (bad < 5) {
				fprintf(stderr,
					"  PRIMITIVE MISMATCH N=%zu i=%zu: "
					"scalar=0x%016llx batch=0x%016llx\n",
					N, i,
					(unsigned long long)scalar[i],
					(unsigned long long)batch_out[i]);
			}
			bad++;
		}
	}

	free(in); free(scalar); free(batch_out);

	if (bad == 0) {
		printf("  PASS primitive parity N=%zu (N%%8=%zu)\n",
		       N, N & 7);
		return 0;
	}
	printf("  FAIL primitive parity N=%zu (%d mismatches)\n", N, bad);
	return 1;
}

/* ----------------------------------------------------------------------------
 * Layer 2: pipeline parity — gf64_fenger_matvec (uses batched prepare) vs
 * explicit Cauchy reference.
 *
 * The Cauchy reference is independent of V_prime / V_at_y_inv (it sums
 * in[c] / (y_r XOR x_c) directly), so a passing comparison proves the
 * prepare outputs are bit-equal to what a scalar-prepared ctx would
 * have produced — any drift in the batch inversion section would
 * surface as a Cauchy mismatch on the recovery output.
 *
 * For the N=4099 odd-tail shape we use gf64_fenger_prepare_padded with
 * numInputsPadded=8192 (next power of 2). The padded prepare inverts
 * V'(x_c) at 8192-element granularity (8192 % 8 == 0, so the AVX-512
 * fast path processes all 8192 with no scalar tail); the test still
 * exercises the prepare end-to-end at a non-trivial padded size.
 *
 * The "odd tail" is exercised separately by Layer 1 at the primitive
 * boundary — the only place in the pipeline where count % 8 != 0 can
 * occur.
 * --------------------------------------------------------------------------*/
static size_t next_pow2_ge(size_t n)
{
	size_t p = 1;
	while (p < n) p <<= 1;
	return p;
}

static void cauchy_reference(
	const gf64_t *in, size_t N, size_t B,
	gf64_t *out, size_t R,
	uint64_t firstInput, uint64_t firstRecovery)
{
	for (size_t r = 0; r < R; r++) {
		gf64_t yr = (gf64_t)(firstRecovery + (uint64_t)r);
		for (size_t w = 0; w < B; w++) {
			gf64_t acc = 0;
			for (size_t c = 0; c < N; c++) {
				gf64_t xc = (gf64_t)(firstInput + (uint64_t)c);
				gf64_t denom = xc ^ yr;
				if (denom == 0) denom = 1;
				acc ^= gf64_mul_reference(in[c * B + w],
				                          gf64_inverse(denom));
			}
			out[r * B + w] = acc;
		}
	}
}

/* Pipeline parity at one (N, R, B) — uses gf64_fenger_matvec (prepares
 * internally with the batch path). Compares against Cauchy. */
static int test_pipeline_parity_pow2(
	size_t N, size_t R, size_t B,
	uint64_t firstInput, uint64_t firstRecovery)
{
	gf64_t *in  = (gf64_t *)malloc(N * B * sizeof(gf64_t));
	gf64_t *ref = (gf64_t *)malloc(R * B * sizeof(gf64_t));
	gf64_t *got = (gf64_t *)malloc(R * B * sizeof(gf64_t));
	if (!in || !ref || !got) {
		fprintf(stderr, "  OOM at N=%zu R=%zu B=%zu\n", N, R, B);
		free(in); free(ref); free(got);
		return 1;
	}

	/* Distinct seed per shape for reproducibility. */
	g_rng = (uint64_t)0x5A5A5A5A5A5A5A5AULL
	       ^ ((uint64_t)N * 0x9E3779B97F4A7C15ULL)
	       ^ ((uint64_t)R * 0xBF58476D1CE4E5B9ULL)
	       ^ ((uint64_t)B * 0x94D049BB133111EBULL);
	for (size_t i = 0; i < (size_t)N * B; i++) in[i] = splitmix64_next();

	cauchy_reference(in, N, B, ref, R, firstInput, firstRecovery);
	gf64_fenger_matvec(in, N, B, got, R, firstInput, firstRecovery);

	int bad = 0;
	for (size_t i = 0; i < R * B; i++) {
		if (got[i] != ref[i]) {
			if (bad < 5) {
				fprintf(stderr,
					"  PIPELINE MISMATCH N=%zu R=%zu B=%zu "
					"[%zu]: got=0x%016llx want=0x%016llx\n",
					N, R, B, i,
					(unsigned long long)got[i],
					(unsigned long long)ref[i]);
			}
			bad++;
		}
	}

	free(in); free(ref); free(got);

	if (bad == 0) {
		printf("  PASS pipeline parity N=%zu R=%zu B=%zu "
		       "(firstInput=%llu firstRecovery=%llu)\n",
		       N, R, B,
		       (unsigned long long)firstInput,
		       (unsigned long long)firstRecovery);
		return 0;
	}
	printf("  FAIL pipeline parity N=%zu R=%zu B=%zu (%d mismatches)\n",
	       N, R, B, bad);
	return 1;
}

/* Pipeline parity at a NON-power-of-2 N — uses gf64_fenger_prepare_padded
 * with N_padded = next_pow2(N). The recovery rows are compared against
 * Cauchy on the real (N, R) range. */
static int test_pipeline_parity_padded(
	size_t N_real, size_t R, size_t B,
	uint64_t firstInput, uint64_t firstRecovery)
{
	const size_t N_pad = next_pow2_ge(N_real);
	const size_t R_pad = next_pow2_ge(R);
	const uint64_t syntheticBase = firstRecovery + (uint64_t)R_pad;

	gf64_t *in  = (gf64_t *)malloc(N_real * B * sizeof(gf64_t));
	gf64_t *ref = (gf64_t *)malloc(R * B * sizeof(gf64_t));
	gf64_t *got = (gf64_t *)malloc(R * B * sizeof(gf64_t));
	if (!in || !ref || !got) {
		fprintf(stderr, "  OOM at N_real=%zu R=%zu B=%zu\n",
		        N_real, R, B);
		free(in); free(ref); free(got);
		return 1;
	}

	g_rng = (uint64_t)0xF00DBABEF00DBABEULL ^ (uint64_t)N_real
	      ^ ((uint64_t)R * 0x9E3779B97F4A7C15ULL);
	for (size_t i = 0; i < N_real * B; i++) in[i] = splitmix64_next();

	cauchy_reference(in, N_real, B, ref, R, firstInput, firstRecovery);

	gf64_fenger_ctx *ctx = gf64_fenger_prepare_padded(
		firstInput, firstRecovery,
		N_real, R, N_pad, R_pad, syntheticBase);
	if (!ctx) {
		fprintf(stderr, "  prepare_padded returned NULL for "
		        "N_real=%zu R=%zu\n", N_real, R);
		free(in); free(ref); free(got);
		return 1;
	}
	gf64_fenger_execute(ctx, in, B, got, 0, B);
	gf64_fenger_release(ctx);

	int bad = 0;
	for (size_t i = 0; i < R * B; i++) {
		if (got[i] != ref[i]) {
			if (bad < 5) {
				fprintf(stderr,
					"  PADDED PIPELINE MISMATCH N_real=%zu "
					"R=%zu B=%zu [%zu]: got=0x%016llx "
					"want=0x%016llx\n",
					N_real, R, B, i,
					(unsigned long long)got[i],
					(unsigned long long)ref[i]);
			}
			bad++;
		}
	}

	free(in); free(ref); free(got);

	if (bad == 0) {
		printf("  PASS pipeline parity (padded) N_real=%zu R=%zu B=%zu "
		       "(pad %zux%zu)\n",
		       N_real, R, B, N_pad, R_pad);
		return 0;
	}
	printf("  FAIL pipeline parity (padded) N_real=%zu R=%zu B=%zu "
	       "(%d mismatches)\n", N_real, R, B, bad);
	return 1;
}

/* ----------------------------------------------------------------------------
 * Bench: inversion-section wall time at N=131072 — scalar loop vs batch.
 * Prints ns/inversion and the speedup factor. Records "no AVX-512" if
 * the host's gf64_has_vpclmulqdq flag is 0 (the batch primitive is then
 * a stub TU body or a non-ZMM scalar path; still prints both numbers
 * for comparison but the ratio is meaningless).
 *
 * Repeated calls + median to absorb jitter; warmup discarded.
 * --------------------------------------------------------------------------*/
static void bench_inversion_section(size_t N, int reps)
{
	gf64_t *in    = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *out   = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (!in || !out) {
		fprintf(stderr, "  OOM bench\n");
		free(in); free(out);
		return;
	}

	g_rng = 0xDEADBEEFCAFEBABEULL;
	for (size_t i = 0; i < N; i++) in[i] = splitmix64_next();

	/* Warmup (1 run each — JIT/branch-predictor state). */
	{
		for (size_t i = 0; i < N; i++) out[i] = gf64_invert_ita_one(in[i]);
		gf64_invert_ita_batch(out, in, N);
	}

	double *ts = (double *)malloc(reps * sizeof(double));
	double *tb = (double *)malloc(reps * sizeof(double));
	if (!ts || !tb) {
		fprintf(stderr, "  OOM bench arrays\n");
		free(ts); free(tb); free(in); free(out);
		return;
	}

	for (int rep = 0; rep < reps; rep++) {
		struct timespec t0, t1;
		clock_gettime(CLOCK_MONOTONIC, &t0);
		for (size_t i = 0; i < N; i++) {
			out[i] = gf64_invert_ita_one(in[i]);
		}
		clock_gettime(CLOCK_MONOTONIC, &t1);
		ts[rep] = (t1.tv_sec - t0.tv_sec)
		        + (t1.tv_nsec - t0.tv_nsec) / 1e9;

		clock_gettime(CLOCK_MONOTONIC, &t0);
		gf64_invert_ita_batch(out, in, N);
		clock_gettime(CLOCK_MONOTONIC, &t1);
		tb[rep] = (t1.tv_sec - t0.tv_sec)
		        + (t1.tv_nsec - t0.tv_nsec) / 1e9;
	}

	/* Median sort. */
	for (int i = 1; i < reps; i++) {
		double v = ts[i]; int j = i - 1;
		while (j >= 0 && ts[j] > v) { ts[j+1] = ts[j]; j--; }
		ts[j+1] = v;
		v = tb[i]; j = i - 1;
		while (j >= 0 && tb[j] > v) { tb[j+1] = tb[j]; j--; }
		tb[j+1] = v;
	}
	double med_s = ts[reps / 2];
	double med_b = tb[reps / 2];

	double ns_per_inv_s = med_s * 1e9 / (double)N;
	double ns_per_inv_b = med_b * 1e9 / (double)N;
	printf("\n--- Inversion-section bench (N=%zu, reps=%d, median) ---\n",
	       N, reps);
	printf("  AVX-512 status: gf64_has_vpclmulqdq = %d\n",
	       gf64_has_vpclmulqdq);
	printf("  scalar loop:    %8.3f ms total  %8.1f ns/inversion\n",
	       med_s * 1e3, ns_per_inv_s);
	printf("  batch (AVX-512):%8.3f ms total  %8.1f ns/inversion\n",
	       med_b * 1e3, ns_per_inv_b);
	if (gf64_has_vpclmulqdq && med_b > 0.0) {
		printf("  speedup:        %.2fx (scalar/batch)\n",
		       med_s / med_b);
	} else {
		printf("  speedup:        n/a (no working AVX-512 host)\n");
	}

	free(ts); free(tb); free(in); free(out);
}

int main(void) {
	/* Bring up dispatch so gf64_has_vpclmulqdq is meaningful for the
	 * bench section. */
	gf64_init_dispatch();

	printf("Fenger prepare-phase inversion batching — T2 parity (issue #59)\n");
	printf("Primitive layer (gf64_invert_ita_batch vs scalar loop)\n");

	int fails = 0;
	fails += test_primitive_parity(8);
	fails += test_primitive_parity(1024);
	fails += test_primitive_parity(4099);   /* odd-tail: 4099 % 8 == 3 */
	fails += test_primitive_parity(65536);

	printf("\nPipeline layer (gf64_fenger_matvec vs explicit Cauchy)\n");

	/* Power-of-2 shapes — the standard Fenger pipeline. */
	fails += test_pipeline_parity_pow2(8,     4,   4,  0,   100);
	fails += test_pipeline_parity_pow2(1024,  128, 8,  0,   10000);
	fails += test_pipeline_parity_pow2(65536, 64, 4, 0,   100000000ULL);

	/* Non-power-of-2 N — exercises gf64_fenger_prepare_padded end-to-end.
	 * The prepare still inverts at the padded count (next pow2), but the
	 * end-to-end Cauchy comparison pins the prepare outputs to the
	 * bit-exact reference. The "odd-tail" scalar-fallback epilog is
	 * covered at the primitive layer above (test_primitive_parity(4099))
	 * since that is the only place a non-multiple-of-8 count enters the
	 * pipeline. */
	fails += test_pipeline_parity_padded(4099, 64, 4, 0, 1000000ULL);

	printf("\nBench section\n");
	bench_inversion_section(131072, 7);

	printf("\n%s: %d test(s) failed\n", fails ? "FAIL" : "OK", fails);
	return fails ? 1 : 0;
}
