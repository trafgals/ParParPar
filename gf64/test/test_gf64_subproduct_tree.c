/*
 * ============================================================================
 * gf64/test/test_gf64_subproduct_tree.c
 *
 * T6 (subproduct tree) correctness test. Validates that the FFT-based
 * subproduct tree actually represents the subproduct of (x + x_i) over the
 * input points at the root level.
 *
 *   - N = 0           -> empty tree (no allocations, zero-initialized struct)
 *   - N = 1           -> root polynomial is exactly x + x_0
 *   - N = 1024 (random) -> Horner-evaluating the root polynomial at 100
 *                          random points yields the same values as computing
 *                          ∏_i (r XOR x_i) directly in the field.
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "../gf64_global.h"
#include "../gf64_subproduct.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

static int g_passed = 0;
static int g_failed = 0;
static uint64_t g_rng = 0xACEDFACE5EEDC0DEULL;

static uint64_t splitmix64_next(void) {
	uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static void pass(const char *name) {
	printf("  PASS: %s\n", name);
	g_passed++;
}

static void fail(const char *name) {
	printf("  FAIL: %s\n", name);
	g_failed++;
}

/*
 * Horner evaluation of poly c[0..deg] at x = r. The polynomial convention is
 * constant-first: c[0] is the constant term, c[k] is the coefficient of x^k.
 * In GF(2^64) addition is XOR.
 */
static gf64_t horner_eval(const gf64_t *c, size_t deg, gf64_t r) {
	gf64_t acc = c[deg];
	for (size_t i = deg; i > 0; i--) {
		acc = gf64_mul_reference(acc, r) ^ c[i - 1];
	}
	return acc;
}

/*
 * Compute the product of (r + x_i) = (r XOR x_i) for i = 0..N-1 by direct
 * multiplication in GF(2^64). This is the "naive" reference that the tree's
 * root polynomial must match at every evaluation point.
 */
static gf64_t product_eval(const gf64_t *points, size_t N, gf64_t r) {
	gf64_t acc = 1;
	for (size_t i = 0; i < N; i++) {
		acc = gf64_mul_reference(acc, r ^ points[i]);
	}
	return acc;
}

static int tree_struct_is_zero(const SubproductTree *t) {
	return t->num_points == 0 && t->num_levels == 0 &&
	       t->level_data == NULL && t->level_lens == NULL &&
	       t->level_degs == NULL && t->storage == NULL;
}

static void test_empty_tree(void) {
	printf("Test: empty tree (N=0)...\n");
	SubproductTree t;
	gf64_subproduct_tree_build(NULL, 0, &t);

	int ok = tree_struct_is_zero(&t);
	gf64_subproduct_tree_free(&t);
	/* After free, all pointers must still be NULL (zero-initialized). */
	ok = ok && tree_struct_is_zero(&t);

	if (ok) {
		pass("empty tree (N=0) build + free leaves struct zero");
	} else {
		fail("empty tree (N=0) build + free leaves struct zero");
	}
}

static void test_singleton(void) {
	printf("Test: singleton (N=1)...\n");
	SubproductTree t;
	gf64_t x0 = 0xDEADBEEFCAFEBABEULL;
	gf64_subproduct_tree_build(&x0, 1, &t);

	int ok = 1;
	if (t.num_points != 1) {
		printf("    num_points: got %zu, want 1\n", t.num_points);
		ok = 0;
	}
	if (t.num_levels != 1) {
		printf("    num_levels: got %zu, want 1\n", t.num_levels);
		ok = 0;
	}
	if (t.level_lens == NULL || t.level_lens[0] != 1) {
		printf("    level_lens[0]: got %zu (or NULL), want 1\n",
		       t.level_lens ? t.level_lens[0] : 0);
		ok = 0;
	}
	if (t.level_degs == NULL || t.level_degs[0] != 1) {
		printf("    level_degs[0]: got %zu (or NULL), want 1\n",
		       t.level_degs ? t.level_degs[0] : 0);
		ok = 0;
	}
	if (t.level_data == NULL || t.level_data[0] == NULL) {
		printf("    level_data[0] is NULL\n");
		ok = 0;
	} else {
		if (t.level_data[0][0] != x0) {
			printf("    c[0]: got 0x%016llx, want 0x%016llx\n",
			       (unsigned long long)t.level_data[0][0],
			       (unsigned long long)x0);
			ok = 0;
		}
		if (t.level_data[0][1] != 1ULL) {
			printf("    c[1]: got 0x%016llx, want 0x1\n",
			       (unsigned long long)t.level_data[0][1]);
			ok = 0;
		}
	}

	gf64_subproduct_tree_free(&t);
	if (ok) {
		pass("singleton (N=1) root poly is x + x_0");
	} else {
		fail("singleton (N=1) root poly is x + x_0");
	}
}

static void test_full_tree_1024(void) {
	printf("Test: full tree (N=1024 random points), root Horner = product at 100 random r...\n");

	const size_t N = 1024;
	gf64_t *points = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (points == NULL) {
		fail("test_full_tree_1024: malloc points failed");
		return;
	}

	g_rng = 0x5EEDC0DECAFEF00DULL;
	for (size_t i = 0; i < N; i++) {
		points[i] = splitmix64_next();
	}

	SubproductTree tree;
	gf64_subproduct_tree_build(points, N, &tree);

	int ok = 1;
	if (tree.num_points != N || tree.num_levels != 11) {
		printf("    num_points=%zu num_levels=%zu (want %zu / 11)\n",
		       tree.num_points, tree.num_levels, N);
		ok = 0;
	}
	if (tree.level_lens[0] != 1 || tree.level_degs[0] != (size_t)N) {
		printf("    root: level_lens[0]=%zu level_degs[0]=%zu (want 1 / %zu)\n",
		       tree.level_lens[0], tree.level_degs[0], N);
		ok = 0;
	}
	if (tree.level_lens[10] != N || tree.level_degs[10] != 1) {
		printf("    leaves: level_lens[10]=%zu level_degs[10]=%zu (want %zu / 1)\n",
		       tree.level_lens[10], tree.level_degs[10], N);
		ok = 0;
	}
	for (size_t i = 0; i < N && ok; i++) {
		gf64_t *leaf_i = tree.level_data[10] + 2 * i;
		if (leaf_i[0] != points[i] || leaf_i[1] != 1ULL) {
			printf("    leaf[%zu] = [0x%016llx, 0x%016llx] want [0x%016llx, 0x1]\n",
			       i, (unsigned long long)leaf_i[0],
			       (unsigned long long)leaf_i[1],
			       (unsigned long long)points[i]);
			ok = 0;
		}
	}

	/* Root polynomial: level_data[0][0..N] = ∏ (x + x_i). */
	gf64_t *root = tree.level_data[0];
	size_t   root_deg = tree.level_degs[0];

	g_rng = 0xFACEFEEDFACEFEEDULL;
	for (size_t trial = 0; trial < 100 && ok; trial++) {
		gf64_t r = splitmix64_next();
		if (r == 0) {
			/* Avoid trivial short-circuit in verification (r = 0 might
			 * coincidentally give a zero result; not a correctness
			 * issue but we want a nonzero r to stress the test). */
			r = (gf64_t)1;
		}
		gf64_t v_horner  = horner_eval(root, root_deg, r);
		gf64_t v_product = product_eval(points, N, r);
		if (v_horner != v_product) {
			printf("    trial=%zu r=0x%016llx horner=0x%016llx product=0x%016llx\n",
			       trial, (unsigned long long)r,
			       (unsigned long long)v_horner,
			       (unsigned long long)v_product);
			ok = 0;
		}
	}

	gf64_subproduct_tree_free(&tree);
	free(points);

	if (ok) {
		pass("1024-point tree root Horner == product (100 trials)");
	} else {
		fail("1024-point tree root Horner == product (100 trials)");
	}
}

/* Issue #59 T8: multi-threaded tree build parity. Builds the same tree
 * twice (once with OMP_NUM_THREADS=1, once with >1) at N=1024 random
 * points and asserts the two trees are bit-identical. gf64_poly_mul is
 * deterministic on the same inputs, so any divergence between the
 * serial and parallel paths indicates a real race or non-deterministic
 * write in the OpenMP-annotated loop in gf64_subproduct.c. Repeated
 * 10 times to catch transient races. */
static int tree_storage_size(const SubproductTree *t) {
	size_t total = 0;
	for (size_t lev = 0; lev < t->num_levels; lev++) {
		total += t->level_lens[lev] * (t->level_degs[lev] + 1);
	}
	return total;
}

/* Cubic P2 review: with N=1024 the OMP gate threshold
 *   parent_count * child_deg^2 >= 2^20
 * never fires (at the deepest levels child_deg is at most 2^9 so the
 * product tops out at ~10^6 < 2^20, and the leaf level has child_deg=1),
 * so the serial/parallel comparison at N=1024 was two identical
 * serial builds and could not detect an OMP-race regression. Add a
 * second sub-test at N=8192 (13 levels, child_deg up to 2^12 = 4096,
 * so the level at child_deg=2^10 already has 8 parents × 2^20 ops =
 * 8M >= 2^20 — the gate fires for ~3 levels). */
static void test_mt_build_parity_large(void) {
	const size_t N = 8192;
	const int num_trials = 5;
	int all_ok = 1;
	const int saved_nthreads = omp_get_max_threads();
	for (int trial = 0; trial < num_trials; trial++) {
		g_rng = 0xC0DEC0DE00000000ULL ^ (uint64_t)trial ^ 0xA5A50000ULL;

		gf64_t *points = (gf64_t *)malloc(N * sizeof(gf64_t));
		if (!points) {
			printf("    LARGE trial=%d malloc failed\n", trial);
			all_ok = 0;
			continue;
		}
		for (size_t i = 0; i < N; i++) points[i] = splitmix64_next();

		omp_set_num_threads(1);
		SubproductTree t_serial;
		gf64_subproduct_tree_build(points, N, &t_serial);

		omp_set_num_threads(4);
		SubproductTree t_parallel;
		gf64_subproduct_tree_build(points, N, &t_parallel);

		const size_t sz_serial   = tree_storage_size(&t_serial);
		const size_t sz_parallel = tree_storage_size(&t_parallel);
		int ok = (sz_serial == sz_parallel) &&
		         (memcmp(t_serial.storage, t_parallel.storage,
		                 sz_serial * sizeof(gf64_t)) == 0);
		if (!ok) {
			printf("    LARGE trial=%d sz_serial=%zu sz_parallel=%zu\n",
			       trial, sz_serial, sz_parallel);
			all_ok = 0;
		}

		gf64_subproduct_tree_free(&t_serial);
		gf64_subproduct_tree_free(&t_parallel);
		free(points);
	}

	omp_set_num_threads(saved_nthreads);

	if (all_ok) {
		pass("multi-threaded tree build (N=8192, OMP gate engages) bit-equal to serial across 5 trials");
	} else {
		fail("multi-threaded tree build (N=8192) diverged from serial in at least one trial");
	}
}

static void test_mt_build_parity(void) {
	const size_t N = 1024;
	const int num_trials = 10;
	int all_ok = 1;
	/* cubic P3: capture the pre-test thread count so the reset at the
	 * end restores whatever OMP_NUM_THREADS/environment value was in
	 * effect before this test pinned it to 1 and 4. omp_get_max_threads()
	 * inside the loop would read the post-pinned value and re-pin the
	 * process at 4 threads for any later multi-threaded tests in this
	 * process. */
	const int saved_nthreads = omp_get_max_threads();
	for (int trial = 0; trial < num_trials; trial++) {
		/* Deterministic per-trial seed. */
		g_rng = 0xC0DEC0DE00000000ULL ^ (uint64_t)trial;

		/* Generate the SAME input points for both builds. */
		gf64_t *points = (gf64_t *)malloc(N * sizeof(gf64_t));
		for (size_t i = 0; i < N; i++) {
			points[i] = splitmix64_next();
		}

		/* Build with OMP_NUM_THREADS=1 (serial path). */
		omp_set_num_threads(1);
		SubproductTree t_serial;
		gf64_subproduct_tree_build(points, N, &t_serial);

		/* Build with OMP_NUM_THREADS=4 (parallel path, 4-way race). */
		omp_set_num_threads(4);
		SubproductTree t_parallel;
		gf64_subproduct_tree_build(points, N, &t_parallel);

		/* Compare the entire storage buffer. The two trees have
		 * identical metadata (level_lens, level_degs) by construction
		 * — only the computed polynomial coefficients could diverge. */
		const size_t sz_serial   = tree_storage_size(&t_serial);
		const size_t sz_parallel = tree_storage_size(&t_parallel);
		int ok = (sz_serial == sz_parallel) &&
		         (memcmp(t_serial.storage, t_parallel.storage,
		                 sz_serial * sizeof(gf64_t)) == 0);
		if (!ok) {
			printf("    trial=%d sz_serial=%zu sz_parallel=%zu\n",
			       trial, sz_serial, sz_parallel);
			all_ok = 0;
		}

		gf64_subproduct_tree_free(&t_serial);
		gf64_subproduct_tree_free(&t_parallel);
		free(points);
	}

	/* Reset to default to avoid leaking the value to other tests. */
	omp_set_num_threads(saved_nthreads);

	if (all_ok) {
		pass("multi-threaded tree build bit-equal to serial across 10 trials (OMP_NUM_THREADS=1 vs 4, N=1024)");
	} else {
		fail("multi-threaded tree build diverged from serial in at least one trial");
	}
}

int main(void) {
	printf("GF64 subproduct tree tests (T6 of par3-cauchy-fft-kernel)\n\n");
	test_empty_tree();
	test_singleton();
	test_full_tree_1024();
	test_mt_build_parity();
	test_mt_build_parity_large();
	printf("\nSummary: %d passed, %d failed\n", g_passed, g_failed);
	return g_failed == 0 ? 0 : 1;
}
