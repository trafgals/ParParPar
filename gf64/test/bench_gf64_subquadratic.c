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
 * the tree's resident storage (level-derived: sum of
 * level_lens[l] * (level_degs[l] + 1) coefficients, which is exactly the
 * build's storage allocation; cubic review f44ead49 P3).
 *
 * Points are generated pairwise-distinct via an open-addressing dedup
 * table (cubic review f44ead49 P3): the derivative-form interpolant
 * aborts on P'(x_j) == 0, so a duplicate point would crash the bench
 * instead of reporting a clean result.
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

/* Generate N pairwise-distinct non-zero points (open-addressing dedup
 * table, load factor <= 0.5 — a linear scan would be O(N^2) at the
 * bench's largest N = 2^20). Duplicate points would make P'(x_j) == 0
 * and abort the derivative-form interpolant. */
static void gen_distinct_points(gf64_t *points, size_t N) {
	size_t cap = 1;
	while (cap < 2 * N) cap <<= 1;
	uint64_t *seen = (uint64_t *)calloc(cap, sizeof(uint64_t));
	if (seen == NULL) abort();

	for (size_t i = 0; i < N; i++) {
		for (;;) {
			gf64_t v = splitmix64_next();
			if (v == 0) continue;
			size_t slot = (size_t)(v * 0x9E3779B97F4A7C15ULL) & (cap - 1);
			while (seen[slot] != 0 && seen[slot] != v) {
				slot = (slot + 1) & (cap - 1);
			}
			if (seen[slot] == 0) {
				seen[slot] = v;
				points[i] = v;
				break;
			}
			/* duplicate — redraw */
		}
	}
	free(seen);
}

/* Tree resident storage: the build allocates exactly
 * sum_l level_lens[l] * (level_degs[l] + 1) coefficients (plus the
 * per-level metadata arrays). */
static double tree_resident_mb(const SubproductTree *tree) {
	size_t total_coeffs = 0;
	for (size_t lev = 0; lev < tree->num_levels; lev++) {
		total_coeffs += tree->level_lens[lev] * (tree->level_degs[lev] + 1);
	}
	return ((double)total_coeffs * 8 + (double)tree->num_levels * 3 * 8) / 1e6;
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

	gen_distinct_points(points, N);
	for (size_t i = 0; i < N; i++) {
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

	printf("  N=%-8zu tree=%.3fs mpe=%.3fs interp=%.3fs  (tree ~%.1f MB)\n",
	       N, t_build, t_mpe, t_interp, tree_resident_mb(&tree));

	gf64_subproduct_tree_free(&tree);
	free(points); free(values); free(f); free(out); free(poly);
}

/* Tree build + MPE only (no interp) at large N. */
static void bench_tree_mpe_only(size_t N) {
	gf64_t *points = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *f      = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *out    = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (!points || !f || !out) abort();

	gen_distinct_points(points, N);
	for (size_t i = 0; i < N; i++) {
		f[i] = splitmix64_next();
	}
	if (f[N - 1] == 0) f[N - 1] = 1ULL;

	SubproductTree tree;
	double t0 = now_sec();
	gf64_subproduct_tree_build(points, N, &tree);
	double t_build = now_sec() - t0;

	t0 = now_sec();
	gf64_multi_point_eval(f, N - 1, &tree, out);
	double t_mpe = now_sec() - t0;

	printf("  N=%-8zu tree=%.3fs mpe=%.3fs interp=(skipped)  (tree ~%.1f MB)\n",
	       N, t_build, t_mpe, tree_resident_mb(&tree));

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
 * Returns 1 on bit-exact match, 0 on mismatch — main() turns a MISMATCH
 * into a nonzero exit code so the P1 acceptance gate fails the run
 * instead of just printing (cubic review 50f46d24 P2).
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
	return ok;
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
	bench_phase(4096);
	bench_phase(16384);
	bench_phase(65536);
	bench_phase(131072);
	/* Tree + MPE only at 2^18 and 2^20 (the interp there is dominated
	 * by the same muls; keeps the run bounded — cubic review f70a81ef
	 * P3: the header documents interp only up to 2^17). */
	bench_tree_mpe_only(262144);
	bench_tree_mpe_only(1048576);

	printf("\nP1 gate shape (Fenger vs explicit Cauchy):\n");
	const int gate_ok = gate_shape(131072, 4096, 4);

	printf("\nDone.\n");
	/* Exit nonzero on a Fenger-vs-Cauchy divergence: the P1 acceptance
	 * gate must fail the run, not just print MISMATCH. */
	return gate_ok ? 0 : 1;
}
