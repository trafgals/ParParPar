/*
 * ============================================================================
 * gf64/test/test_gf64_mpe_interp.c — Tests for multi-point INTERPOLATION
 *
 * T8b of the par3-cauchy-fft-kernel plan. Companion to test_gf64_mpe.c
 * (which covers the EVAL direction).
 *
 * The interpolation function being tested is
 *     gf64_multi_point_interp(SubproductTree *tree, gf64_t *values, N, gf64_t *out)
 * which, given N input points (encoded in the subproduct tree) and N values,
 * returns the unique polynomial `out` of degree < N such that
 *     out(x_j) = values[j]   for every j in [0..N).
 *
 * Tests:
 *
 *   1. Lagrangian reference (small N). For N up to 12 we compute the
 *      expected polynomial directly via the Lagrange formula:
 *          L_j(x) = ∏_{k != j} (x + x_k) / ∏_{k != j} (x_j + x_k)
 *          f(x)   = ⊕_j values[j] · L_j(x)
 *      and compare bit-for-bit.
 *
 *   2. round-trip for larger N. We pick random (x_j, values[j]), interpolate
 *      to recover f, then evaluate f at the same points via the existing
 *      gf64_multi_point_eval. The eval output must match the input values
 *      bit-for-bit. This is a stronger test than the Lagrange reference
 *      because it crosses two PR-independent code paths.
 *
 *   3. deg_ltn_N contract. Since the function returns a polynomial of degree
 *      strictly less than N by construction, when values are consistent with
 *      a polynomial of lower degree, the leading coefficient should be zero.
 *      We pick a known polynomial f_small of degree d with d < N, evaluate
 *      it at the tree's leaves via Horner to get values[j] = f_small(x_j),
 *      and assert that the recovered polynomial's leading coefficient
 *      beyond degree d is zero (more precisely: any coefficient i > d has
 *      out[i] == 0).
 *
 *   4. empty / NULL tree returns immediately. Mirrors the contract tested
 *      by test_gf64_mpe Test 6 in the eval direction.
 *
 * Build & run from gf64/test/:
 *   $(CC) -O2 -march=native -I.. test_gf64_mpe_interp.c \
 *         ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c \
 *         ../gf64_additive_fft.c ../gf64_mul_avx512.c \
 *         ../gf64_square.c ../gf64_single.c \
 *         -o test_gf64_mpe_interp && ./test_gf64_mpe_interp
 *
 * Output is plain text so it can be piped to
 * .omo/evidence/task-8b-par3-cauchy-fft-kernel-interp.log.
 *
 * Exits 0 on full pass; non-zero on any failure.
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../gf64_mpe.h"
#include "../gf64_subproduct.h"
#include "../gf64_invert_ita.h"
#include "../gf64_additive_fft.h"
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* ----------------------------------------------------------------------------
 * splitmix64 PRNG (independent stream from test_gf64_mpe.c — different seeds
 * here).
 * ---------------------------------------------------------------------------- */
static uint64_t g_splitmix_state = 0;

static uint64_t splitmix64_next(void) {
	uint64_t z = (g_splitmix_state += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static void put_seed(uint64_t s) {
	g_splitmix_state = s;
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
 * Reference Horner evaluation (constant-first coefficients).
 * Independent of gf64_mpe.c.
 * ---------------------------------------------------------------------------- */
static gf64_t horner_eval_reference(const gf64_t *c, size_t deg, gf64_t r) {
	if (deg == (size_t)(-1)) {
		return 0;
	}
	gf64_t acc = c[deg];
	for (size_t i = deg; i > 0; i--) {
		acc = gf64_mul_reference(acc, r) ^ c[i - 1];
	}
	return acc;
}

/* ----------------------------------------------------------------------------
 * Naive Lagrange interpolation (O(N^3) schoolbook). Returns a polynomial
 * `out` of degree < N satisfying out(x_j) = values[j] for every j.
 *
 * Algorithm per j:
 *   numer_j(x) = ∏_{k != j} (x + x_k)                  (degree N - 1)
 *   denom_j    = ∏_{k != j} (x_j + x_k)                 (scalar)
 *   L_j(x)     = numer_j(x) * (1 / denom_j)
 *
 * Then f(x) = ⊕_j values[j] * L_j(x).
 *
 * We use gf64_poly_mul / gf64_poly_mul_padded for the per-k product and
 * truncate correctly so the result has exactly N coefficients (degree < N).
 * The reference is O(N^3) field ops and is only used for N <= 12 in tests
 * 1/3 below. Reference exists to catch any structural bug in the Bostan-
 * Schost interpolation; the round-trip test (Test 2) crosses both code
 * paths and provides the bit-exactness gate for larger N.
 * ---------------------------------------------------------------------------- */
static int lagrange_interp_reference(
	const gf64_t *points, const gf64_t *values, size_t N,
	gf64_t *out
) {
	if (N == 0) {
		return 0; /* success, nothing to do */
	}
	if (N == 1) {
		out[0] = values[0];
		return 0;
	}

	gf64_t *acc    = (gf64_t *)calloc(N, sizeof(gf64_t));
	gf64_t *numer  = (gf64_t *)calloc(N, sizeof(gf64_t));
	gf64_t *tmp    = (gf64_t *)calloc(2 * N, sizeof(gf64_t));
	gf64_t leaf[2];

	if (!acc || !numer || !tmp) {
		free(acc); free(numer); free(tmp);
		return -1;
	}

	for (size_t j = 0; j < N; j++) {
		/* numer_j(x) = ∏_{k != j} (x + x_k).
		 * Start with 1 (degree 0, coeffs [1, 0, ...]).
		 * At each k != j, multiply by (x + x_k) (degree 1, coeffs [x_k, 1]). */
		memset(numer, 0, N * sizeof(gf64_t));
		numer[0] = 1ULL;

		size_t numer_deg = 0; /* current highest-degree index with nonzero coeff */
		for (size_t k = 0; k < N; k++) {
			if (k == j) continue;
			leaf[0] = points[k];
			leaf[1] = 1ULL;
			/* numer * leaf = tmp truncated to N coefficients.
			 * `leaf` represents the polynomial (x + x_k) of degree 1;
			 * len_b = 2 captures BOTH coefficients. Passing len_b = 1
			 * would silently drop the x^k term, collapsing the
			 * iteration into a constant product — wrong for the
			 * Lagrange reference. */
			gf64_poly_mul_padded(tmp, numer, numer_deg + 1, leaf, 2, N);
			memcpy(numer, tmp, N * sizeof(gf64_t));
			numer_deg = (numer_deg + 1 < N - 1) ? (numer_deg + 1) : (N - 1);
		}

		/* denom_j = ∏_{k != j} (points[j] + points[k]). */
		gf64_t denom = 1ULL;
		for (size_t k = 0; k < N; k++) {
			if (k == j) continue;
			denom = gf64_mul_reference(denom, points[j] ^ points[k]);
		}
		gf64_t denom_inv = gf64_invert_ita_one(denom);

		/* f_acc += values[j] * numer * denom_inv. */
		for (size_t i = 0; i < N; i++) {
			gf64_t term = gf64_mul_reference(values[j],
			            gf64_mul_reference(numer[i], denom_inv));
			acc[i] ^= term;
		}
	}

	memcpy(out, acc, N * sizeof(gf64_t));
	free(acc);
	free(numer);
	free(tmp);
	return 0;
}

/* ----------------------------------------------------------------------------
 * Test 1: Lagrange reference parity at N = {4, 8, 12}.
 * ---------------------------------------------------------------------------- */
static void test_lagrange_reference(void) {
	printf("Test 1: multi_point_interp == Lagrange reference (small N)\n");

	const size_t Ns[] = { 4, 8, 16 };
	const size_t num_N = sizeof(Ns) / sizeof(Ns[0]);
	int all_ok = 1;

	for (size_t ni = 0; ni < num_N; ni++) {
		size_t N = Ns[ni];
		gf64_t *points = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *values = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *ref    = (gf64_t *)calloc(N, sizeof(gf64_t));
		gf64_t *got    = (gf64_t *)calloc(N, sizeof(gf64_t));
		if (!points || !values || !ref || !got) {
			printf("    N=%zu alloc failed\n", N);
			all_ok = 0;
			free(points); free(values); free(ref); free(got);
			continue;
		}

		put_seed(0x100000001ULL + (uint64_t)ni);
		for (size_t i = 0; i < N; i++) {
			gf64_t v;
			do {
				v = splitmix64_next();
			} while (v == 0);
			points[i] = v;
		}
		for (size_t i = 0; i < N; i++) {
			values[i] = splitmix64_next();
		}

		SubproductTree tree;
		gf64_subproduct_tree_build(points, N, &tree);

		if (lagrange_interp_reference(points, values, N, ref) != 0) {
			printf("    N=%zu lagrange ref alloc failed\n", N);
			all_ok = 0;
			gf64_subproduct_tree_free(&tree);
			free(points); free(values); free(ref); free(got);
			continue;
		}

		gf64_multi_point_interp(&tree, values, got);

		int okN = 1;
		int first_mismatch = -1;
		for (size_t i = 0; i < N; i++) {
			if (got[i] != ref[i]) {
				if (first_mismatch < 0) first_mismatch = (int)i;
				okN = 0;
			}
		}
		if (okN) {
			printf("    N=%zu: bit-exact Lagrange match (%zu/%zu)\n", N, N, N);
			g_passed++;
		} else {
			printf("    N=%zu: first mismatch i=%d got=0x%016llx ref=0x%016llx\n",
			       N, first_mismatch,
			       (unsigned long long)got[first_mismatch],
			       (unsigned long long)ref[first_mismatch]);
			g_failed++;
			all_ok = 0;
		}

		gf64_subproduct_tree_free(&tree);
		free(points); free(values); free(ref); free(got);
	}

	if (all_ok) {
		pass("interp == Lagrange reference at N in {4, 8, 12}");
	} else {
		fail("interp == Lagrange reference");
	}
}

/* ----------------------------------------------------------------------------
 * Test 2: round-trip parity at larger N. Pick random (x_j, values[j]),
 * interpolate to get f, then evaluate f at the same points via the
 * existing gf64_multi_point_eval and assert the eval output equals the
 * original values bit-for-bit.
 * ---------------------------------------------------------------------------- */
static void test_round_trip(void) {
	printf("Test 2: round-trip — interp then eval recovers input values\n");

	const size_t Ns[] = { 4, 16, 64, 256 };
	const size_t num_N = sizeof(Ns) / sizeof(Ns[0]);
	int all_ok = 1;

	for (size_t ni = 0; ni < num_N; ni++) {
		size_t N = Ns[ni];
		gf64_t *points = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *values = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *poly   = (gf64_t *)calloc(N, sizeof(gf64_t));
		gf64_t *eval   = (gf64_t *)calloc(N, sizeof(gf64_t));
		if (!points || !values || !poly || !eval) {
			printf("    N=%zu alloc failed\n", N);
			all_ok = 0;
			free(points); free(values); free(poly); free(eval);
			continue;
		}

		put_seed(0x200000002ULL + (uint64_t)ni);
		for (size_t i = 0; i < N; i++) {
			gf64_t v;
			do {
				v = splitmix64_next();
			} while (v == 0);
			points[i] = v;
		}
		for (size_t i = 0; i < N; i++) {
			values[i] = splitmix64_next();
		}

		SubproductTree tree;
		gf64_subproduct_tree_build(points, N, &tree);

		gf64_multi_point_interp(&tree, values, poly);

		/* Evaluate at the tree leaves. */
		gf64_multi_point_eval(poly, N - 1, &tree, eval);

		int okN = 1;
		int first_mismatch = -1;
		for (size_t i = 0; i < N; i++) {
			if (eval[i] != values[i]) {
				if (first_mismatch < 0) first_mismatch = (int)i;
				okN = 0;
			}
		}
		if (okN) {
			printf("    N=%zu: round-trip bit-exact (%zu/%zu)\n", N, N, N);
			g_passed++;
		} else {
			printf("    N=%zu: first mismatch i=%d eval=0x%016llx values[i]=0x%016llx\n",
			       N, first_mismatch,
			       (unsigned long long)eval[first_mismatch],
			       (unsigned long long)values[first_mismatch]);
			printf("        values[0]=0x%016llx values[%zu]=0x%016llx ...\n",
			       (unsigned long long)values[0],
			       first_mismatch,
			       (unsigned long long)values[first_mismatch]);
			g_failed++;
			all_ok = 0;
		}

		gf64_subproduct_tree_free(&tree);
		free(points); free(values); free(poly); free(eval);
	}

	if (all_ok) {
		pass("interp(round-trip) at N in {4, 16, 64, 256}");
	} else {
		fail("interp round-trip");
	}
}

/* ----------------------------------------------------------------------------
 * Test 3: low-degree polynomial recovery. Pick a polynomial f_small of
 * degree d (d < N). Evaluate at the tree points via Horner (one specific
 * independent code path) to get values[j]. Interpolate, and assert that
 * the recovered polynomial's coefficients match f_small bit-for-bit from
 * index 0 through d, and that coefficients i > d are zero.
 * ---------------------------------------------------------------------------- */
static void test_low_degree_recovery(void) {
	printf("Test 3: low-degree recovery — interp recovers f when deg(f) < N\n");

	const size_t N    = 32;
	const size_t deg  = 7;
	gf64_t *points   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *f_small  = (gf64_t *)calloc(deg + 1, sizeof(gf64_t));
	gf64_t *values   = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *recovered = (gf64_t *)calloc(N, sizeof(gf64_t));
	if (!points || !f_small || !values || !recovered) {
		printf("    alloc failed\n");
		fail("low-deg alloc");
		free(points); free(f_small); free(values); free(recovered);
		return;
	}

	put_seed(0x300000003ULL);
	for (size_t i = 0; i < N; i++) {
		gf64_t v;
		do {
			v = splitmix64_next();
		} while (v == 0);
		points[i] = v;
	}
	for (size_t i = 0; i <= deg; i++) {
		f_small[i] = splitmix64_next();
	}
	if (f_small[deg] == 0) f_small[deg] = 1ULL;

	/* values[j] = horner(f_small, deg, points[j]). */
	gf64_t *last_level = NULL;
	{
		SubproductTree tree_pre;
		gf64_subproduct_tree_build(points, N, &tree_pre);
		last_level = tree_pre.level_data[tree_pre.num_levels - 1];
		for (size_t j = 0; j < N; j++) {
			values[j] = horner_eval_reference(f_small, deg, last_level[2 * j]);
		}
		gf64_subproduct_tree_free(&tree_pre);
	}

	/* Build a fresh tree (since we free'd tree_pre above). */
	SubproductTree tree;
	gf64_subproduct_tree_build(points, N, &tree);

	gf64_multi_point_interp(&tree, values, recovered);

	int ok = 1;
	for (size_t i = 0; i <= deg; i++) {
		if (recovered[i] != f_small[i]) {
			printf("    i=%zu recovered=0x%016llx f=0x%016llx\n",
			       i, (unsigned long long)recovered[i],
			       (unsigned long long)f_small[i]);
			ok = 0;
			break;
		}
	}
	for (size_t i = deg + 1; i < N; i++) {
		if (recovered[i] != 0) {
			printf("    i=%zu recovered=0x%016llx, want 0 (deg < N contract)\n",
			       i, (unsigned long long)recovered[i]);
			ok = 0;
			break;
		}
	}
	if (ok) {
		pass("low-deg recovery: deg(f)=7, N=32, recovered bit-exact");
	} else {
		fail("low-deg recovery");
	}

	gf64_subproduct_tree_free(&tree);
	free(points); free(f_small); free(values); free(recovered);
}

/* ----------------------------------------------------------------------------
 * Test 4: empty / NULL tree no-op contract. Mirrors test_gf64_mpe Test 6.
 * ---------------------------------------------------------------------------- */
static void test_empty_tree(void) {
	printf("Test 4: empty / NULL tree no-op contract\n");

	gf64_t values[4] = { 0x11, 0x22, 0x33, 0x44 };
	gf64_t out[4]    = { 0xAA, 0xBB, 0xCC, 0xDD };

	gf64_multi_point_interp(NULL, values, out);
	if (out[0] == 0xAA && out[3] == 0xDD) {
		pass("interp(NULL tree) leaves out untouched");
	} else {
		fail("interp(NULL tree) no-op");
	}

	SubproductTree empty;
	gf64_subproduct_tree_build(NULL, 0, &empty);
	gf64_multi_point_interp(&empty, values, out);
	gf64_subproduct_tree_free(&empty);
	if (out[0] == 0xAA && out[3] == 0xDD) {
		pass("interp(empty tree) leaves out untouched");
	} else {
		fail("interp(empty tree) no-op");
	}
}

/* ----------------------------------------------------------------------------
 * Test 5 (boundary): N == 1. A single point yields a constant polynomial
 * whose sole coefficient equals values[0].
 * ---------------------------------------------------------------------------- */
static void test_n1(void) {
	printf("Test 5: N == 1 boundary — interp returns constant values[0]\n");

	gf64_t point = 0xCAFEBABEULL;
	gf64_t value = 0xDEADBEEFULL;
	gf64_t out[1] = { 0 };

	SubproductTree tree;
	gf64_subproduct_tree_build(&point, 1, &tree);

	gf64_multi_point_interp(&tree, &value, out);

	gf64_subproduct_tree_free(&tree);

	if (out[0] == value) {
		pass("interp(N=1) returns values[0]");
	} else {
		printf("    out[0]=0x%016llx, want 0x%016llx\n",
		       (unsigned long long)out[0], (unsigned long long)value);
		fail("interp(N=1) constant polynomial");
	}
}

/*
 * Test 5b (boundary diagnostic): N == 2. Isolates whether the N=4+ failure
 * comes from the EGCD at higher degree or from the multi-level recursion.
 * A degree-1 EGCD is straightforward to verify by hand; if N=2 round-trips
 * but N=4 round-trip fails, the EGCD or the multi-level recursion
 * scratch layout is at fault.
 */
static void test_n2(void) {
	printf("Test 5b: N == 2 boundary diagnostic\n");

	gf64_t pts[2]   = { 0x0102030405060708ULL, 0x1112131415161718ULL };
	gf64_t vals[2]  = { 0xAABBCCDDEEFF0011ULL, 0x1122334455667788ULL };
	gf64_t got[2]   = { 0, 0 };
	gf64_t eval[2]  = { 0, 0 };

	SubproductTree tree;
	gf64_subproduct_tree_build(pts, 2, &tree);
	gf64_multi_point_interp(&tree, vals, got);
	gf64_multi_point_eval(got, 1, &tree, eval);
	gf64_subproduct_tree_free(&tree);

	int ok = 1;
	for (size_t i = 0; i < 2; i++) {
		if (eval[i] != vals[i]) {
			printf("    i=%zu eval=0x%016llx values=0x%016llx (got[0]=0x%016llx got[1]=0x%016llx)\n",
			       i,
			       (unsigned long long)eval[i],
			       (unsigned long long)vals[i],
			       (unsigned long long)got[0],
			       (unsigned long long)got[1]);
			ok = 0;
		}
	}
	if (ok) {
		pass("interp(N=2) round-trips at both points");
	} else {
		fail("interp(N=2) round-trip");
	}
}

/* ----------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------------- */
int main(void) {
	printf("GF64 multi-point interpolation tests (T8b of par3-cauchy-fft-kernel)\n");
	printf("========================================================================\n\n");

	test_lagrange_reference();
	test_round_trip();
	test_low_degree_recovery();
	test_empty_tree();
	test_n2();
	test_n1();

	printf("\n=== Summary ===\n");
	printf("Passed: %d\n", g_passed);
	printf("Failed: %d\n", g_failed);

	return g_failed > 0 ? 1 : 0;
}
