/*
 * ============================================================================
 * gf64/test/bench_gf64_subquadratic.c — subquadratic prep microbench (A1)
 *
 * Issue #59 A1 evidence bench. Measures the three phases that were O(N^2)
 * before A1 (schoolbook divmod MPE + half-EGCD inverse cache at tree
 * build) and are O(N log^2 N) / O(M(N) log N) after:
 *
 *   1. gf64_subproduct_tree_build  (now pure FFT-mul levels)
 *   2. gf64_multi_point_eval       (Newton-reciprocal divmod walk)
 *   3. gf64_multi_point_interp_internal (derivative-based Lagrange form)
 *
 * Also runs the P1 gate shape — bit-exact Fenger vs explicit Cauchy at
 * N=131072 / R=4096 (the issue #59 §8 P1 acceptance shape) — and prints
 * the tree's resident storage.
 *
 * Build & run from gf64/test/:
 *   make bench_gf64_subquadratic && ./bench_gf64_subquadratic
 *
 * Sizes: tree+MPE+interp at N = 4096..131072 (interp included up to
 * 131072); tree+MPE only at 262144 and 1048576 (interp at 2^20 costs
 * ~1.4 GB transient scratch and is dominated by the same muls — run
 * separately if needed).
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../gf64_global.h"
#include "../gf64_subproduct.h"
#include "../gf64_mpe.h"
#include "../gf64_fenger.h"
#include "../gf64_invert_ita.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);
extern gf64_t gf64_inverse(gf64_t a);

static uint64_t g_rng = 0x9E3779B97F4A7C15ULL;

static uint64_t splitmix64_next(void) {
	uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static double now_sec(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* ----------------------------------------------------------------------------
 * Phase bench: tree build + MPE + interp at N.
 * ---------------------------------------------------------------------------- */
static void bench_phase(size_t N) {
	gf64_t *points = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *values = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *f      = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *out    = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *poly   = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (!points || !values || !f || !out || !poly) abort();

	for (size_t i = 0; i < N; i++) {
		gf64_t v;
		do { v = splitmix64_next(); } while (v == 0);
		points[i] = v;
		values[i] = splitmix64_next();
		f[i]      = splitmix64_next();
	}
	if (f[N - 1] == 0) f[N - 1] = 1ULL;

	SubproductTree tree;
	double t0 = now_sec();
	gf64_subproduct_tree_build(points, N, &tree);
	double t_build = now_sec() - t0;

	t0 = now_sec();
	gf64_multi_point_eval(f, N - 1, &tree, out);
	double t_mpe = now_sec() - t0;

	t0 = now_sec();
	gf64_multi_point_interp_internal(&tree, values, poly);
	double t_interp = now_sec() - t0;

	/* Tree resident storage: polynomial storage + level metadata. */
	double tree_mb = ((double)(tree.num_points * 2 + 2 * tree.num_points) * 8
	                  + (double)tree.num_levels * 3 * 8) / 1e6;

	printf("  N=%-8zu tree=%.3fs mpe=%.3fs interp=%.3fs  (tree ~%.1f MB)\n",
	       N, t_build, t_mpe, t_interp, tree_mb);

	gf64_subproduct_tree_free(&tree);
	free(points); free(values); free(f); free(out); free(poly);
}

/* Tree build + MPE only (no interp) at large N. */
static void bench_tree_mpe_only(size_t N) {
	gf64_t *points = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *f      = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *out    = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (!points || !f || !out) abort();

	for (size_t i = 0; i < N; i++) {
		gf64_t v;
		do { v = splitmix64_next(); } while (v == 0);
		points[i] = v;
		f[i]      = splitmix64_next();
	}
	if (f[N - 1] == 0) f[N - 1] = 1ULL;

	SubproductTree tree;
	double t0 = now_sec();
	gf64_subproduct_tree_build(points, N, &tree);
	double t_build = now_sec() - t0;

	t0 = now_sec();
	gf64_multi_point_eval(f, N - 1, &tree, out);
	double t_mpe = now_sec() - t0;

	double tree_mb = ((double)(tree.num_points * 2 + 2 * tree.num_points) * 8
	                  + (double)tree.num_levels * 3 * 8) / 1e6;
	printf("  N=%-8zu tree=%.3fs mpe=%.3fs interp=(skipped)  (tree ~%.1f MB)\n",
	       N, t_build, t_mpe, tree_mb);

	gf64_subproduct_tree_free(&tree);
	free(points); free(f); free(out);
}

/* ----------------------------------------------------------------------------
 * Explicit Cauchy matrix-vector product (reference).
 * ---------------------------------------------------------------------------- */
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

/* ----------------------------------------------------------------------------
 * P1 gate shape: bit-exact Fenger vs Cauchy at N=131072 / R=4096.
 * Returns nonzero on divergence — the P1 acceptance gate must fail the
 * run (the CI badge workflow publishes from this bench's output, so a
 * MISMATCH must never produce a green badge).
 * ---------------------------------------------------------------------------- */
static int gate_shape(size_t N, size_t R, size_t B) {
	gf64_t *in         = (gf64_t *)malloc(N * B * sizeof(gf64_t));
	gf64_t *cauchy_out = (gf64_t *)calloc(R * B, sizeof(gf64_t));
	gf64_t *fenger_out = (gf64_t *)calloc(R * B, sizeof(gf64_t));
	if (!in || !cauchy_out || !fenger_out) abort();

	g_rng = 0xDEADBEEF;
	for (size_t i = 0; i < N * B; i++) in[i] = splitmix64_next();

	const uint64_t fi = 0x10000;
	const uint64_t fr = 0x1000000;

	double t0 = now_sec();
	cauchy_reference(in, N, B, cauchy_out, R, fi, fr);
	double t_cauchy = now_sec() - t0;

	t0 = now_sec();
	gf64_fenger_matvec(in, N, B, fenger_out, R, fi, fr);
	double t_fenger = now_sec() - t0;

	int ok = (memcmp(cauchy_out, fenger_out, R * B * sizeof(gf64_t)) == 0);
	printf("  N=%-6zu R=%-6zu B=%-3zu cauchy=%.2fs fenger=%.2fs %s\n",
	       N, R, B, t_cauchy, t_fenger, ok ? "BIT-EXACT" : "MISMATCH");

	free(in); free(cauchy_out); free(fenger_out);
	return ok ? 0 : 1;
}

int main(void) {
	/* Line-buffer stdout so progress is visible when piped to a file. */
	setvbuf(stdout, NULL, _IOLBF, 0);

	/* Engage the host's best GF method: the HQC mul dispatch gates its
	 * AVX-512 butterfly on gf64_current_method == GF64_AVX512, which
	 * defaults to GF64_SCALAR in a standalone binary. Run with
	 * PAR3_AVX512_FORCE=2 (and/or PAR3_GF64_USE_AVX512=1) on WSL2 to
	 * defeat the hypervisor CPUID masking; the AVX-512 column then
	 * matches bare-metal Windows. */
	gf64_init_dispatch();

	printf("Subquadratic prep microbench (issue #59 A1)\n");
	printf("==========================================\n\n");

	printf("Phases (tree build / MPE / interp):\n");
	if (getenv("PAR3_CI_BENCH")) {
		/* CI mode: bounded sizes only (a full 2^20 run is minutes on a
		 * shared runner and adds nothing the badges read). */
		bench_phase(4096);
		bench_phase(16384);
		bench_phase(65536);
		bench_phase(131072);
	} else {
		bench_phase(4096);
		bench_phase(16384);
		bench_phase(65536);
		bench_phase(131072);
		bench_phase(262144);
		/* Tree + MPE only at 2^20 (the interp there is dominated by the same
		 * muls; keeps the run bounded). */
		bench_tree_mpe_only(1048576);
	}

	printf("\nP1 gate shape (Fenger vs explicit Cauchy):\n");
	int gate_fail = gate_shape(131072, 4096, 4);

	printf("\nDone.\n");
	return gate_fail;
}
