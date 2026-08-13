/*
 * ============================================================================
 * gf64/test/test_gf64_interp_parity.c — derivative-form interpolation tests
 *
 * Issue #59 A1: gf64_multi_point_interp_internal was rewritten from the
 * CRT+inverse-cache combine to the derivative-based Lagrange form
 * (z_j = y_j / P'(x_j), f_parent = f_L·P_R + f_R·P_L). This test pins the
 * new contract:
 *
 *   1. Round-trip: interp(tree, values) -> f, then MPE(f, tree) == values
 *      at every point, for N in {2, 4, 8, 64, 256, 1024}.
 *   2. Direct Lagrange reference at small N (N=8): the interpolant
 *      coefficients must match a from-scratch construction of
 *      f = Σ_j y_j · ∏_{k≠j} (x + x_k)/(x_j + x_k) — independent of
 *      both the tree walk and the derivative.
 *   3. N == 1 edge: out = values[0].
 *   4. Repeated-call determinism (two identical calls, bit-identical
 *      outputs) — guards against scratch aliasing bugs.
 *
 * The interpolation output polynomial of degree < N is UNIQUE, so any
 * correct algorithm is bit-identical; these checks therefore pin the
 * exact contract the Fenger pipeline (gf64_fenger.c) relies on.
 *
 * Build & run from gf64/test/:
 *   make test_gf64_interp_parity && ./test_gf64_interp_parity
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
#include "../gf64_barycentric.h"
#include "../gf64_invert_ita.h"
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* ----------------------------------------------------------------------------
 * splitmix64 PRNG.
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
 * Test 1: round-trip at several N.
 *
 * interp(values) -> f, then MPE(f) must reproduce values at every leaf.
 * ---------------------------------------------------------------------------- */
static void test_roundtrip(void) {
	printf("Test 1: interp -> MPE round-trip (f(x_j) == values[j])\n");

	const size_t sizes[] = { 2, 4, 8, 64, 256, 1024 };
	const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

	for (size_t si = 0; si < num_sizes; si++) {
		const size_t N = sizes[si];

		gf64_t *points = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *values = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *f      = (gf64_t *)calloc(N, sizeof(gf64_t));
		gf64_t *back   = (gf64_t *)calloc(N, sizeof(gf64_t));
		if (!points || !values || !f || !back) {
			printf("    N=%zu alloc failed\n", N);
			fail("roundtrip alloc");
			free(points); free(values); free(f); free(back);
			continue;
		}

		/* Distinct non-zero points (tree-build requirement). */
		put_seed(0x0BADF00D00000000ULL ^ (uint64_t)N);
		for (size_t i = 0; i < N; i++) {
			gf64_t v;
			do {
				v = splitmix64_next();
			} while (v == 0);
			points[i] = v;
		}
		put_seed(0xFEEDFACE00000000ULL ^ (uint64_t)N);
		for (size_t i = 0; i < N; i++) {
			values[i] = splitmix64_next();
		}

		SubproductTree tree;
		gf64_subproduct_tree_build(points, N, &tree);

		gf64_multi_point_interp_internal(&tree, values, f);

		/* Evaluate the interpolant back at the points (deg f < N, so the
		 * Bostan-Schost walk runs, not the Horner fallback). */
		gf64_multi_point_eval(f, N - 1, &tree, back);

		int ok = 1;
		int first_mismatch = -1;
		for (size_t j = 0; j < N; j++) {
			if (back[j] != values[j]) {
				if (first_mismatch < 0) first_mismatch = (int)j;
				ok = 0;
			}
		}

		char label[128];
		snprintf(label, sizeof(label),
		         "round-trip N=%zu: MPE(interp(values)) == values (%zu/%zu)",
		         N, ok ? N : 0, N);
		if (ok) {
			pass(label);
		} else {
			printf("    first mismatch at j=%d: got=0x%016llx want=0x%016llx\n",
			       first_mismatch,
			       (unsigned long long)back[first_mismatch],
			       (unsigned long long)values[first_mismatch]);
			fail(label);
		}

		gf64_subproduct_tree_free(&tree);
		free(points); free(values); free(f); free(back);
	}
}

/* ----------------------------------------------------------------------------
 * Test 2: direct Lagrange reference at N = 8.
 *
 * f(x) = Σ_j y_j · L_j(x),  L_j(x) = ∏_{k≠j} (x + x_k) / (x_j + x_k)
 * (char 2: subtraction == addition). Constructed here with independent
 * code: multiply the linear factors (x + x_k) in a local buffer, scale by
 * the inverse of the pairwise-sum product, accumulate. Compare
 * coefficient-by-coefficient against the interpolation output.
 * ---------------------------------------------------------------------------- */
static void test_lagrange_reference(void) {
	printf("Test 2: interp == direct Lagrange basis construction (N=8)\n");

	const size_t N = 8;

	gf64_t points[8], values[8];
	put_seed(0x5EEDC0DE5EEDC0DEULL);
	for (size_t i = 0; i < N; i++) {
		gf64_t v;
		do {
			v = splitmix64_next();
		} while (v == 0);
		points[i] = v;
	}
	put_seed(0xC0FFEE00C0FFEE00ULL);
	for (size_t i = 0; i < N; i++) {
		values[i] = splitmix64_next();
	}

	SubproductTree tree;
	gf64_subproduct_tree_build(points, N, &tree);

	gf64_t *f = (gf64_t *)calloc(N, sizeof(gf64_t));
	if (f == NULL) {
		printf("    alloc failed\n");
		fail("lagrange alloc");
		gf64_subproduct_tree_free(&tree);
		return;
	}
	gf64_multi_point_interp_internal(&tree, values, f);

	/* Direct Lagrange construction: ref = Σ_j y_j · L_j. */
	gf64_t *L   = (gf64_t *)calloc(N, sizeof(gf64_t));
	gf64_t *ref = (gf64_t *)calloc(N, sizeof(gf64_t));
	int ok = (L != NULL && ref != NULL);
	if (!ok) {
		printf("    L/ref alloc failed\n");
		fail("lagrange alloc 2");
	} else {
		for (size_t j = 0; j < N; j++) {
			/* L_j = ∏_{k≠j} (x + x_k), starting from the constant 1. */
			memset(L, 0, N * sizeof(gf64_t));
			L[0] = 1ULL;
			size_t cur_deg = 0;
			for (size_t k = 0; k < N; k++) {
				if (k == j) continue;
				/* Multiply the running product (degree cur_deg) by
				 * (x + x_k): new[i] = old[i-1] ^ x_k * old[i]. In-place
				 * descending update; the constant term is updated
				 * separately (new[0] = x_k * old[0]). */
				for (size_t i = cur_deg + 1; i > 0; i--) {
					L[i] = L[i - 1] ^ gf64_mul_reference(L[i], points[k]);
				}
				L[0] = gf64_mul_reference(L[0], points[k]);
				cur_deg++;
			}
			/* Normalize: L_j = L_j / ∏_{k≠j} (x_j + x_k). */
			gf64_t denom = 1ULL;
			for (size_t k = 0; k < N; k++) {
				if (k == j) continue;
				denom = gf64_mul_reference(denom, points[j] ^ points[k]);
			}
			gf64_t scale = gf64_mul_reference(values[j],
			                                  gf64_invert_ita_one(denom));
			for (size_t i = 0; i < N; i++) {
				ref[i] ^= gf64_mul_reference(scale, L[i]);
			}
		}

		/* Coefficient-by-coefficient comparison. */
		for (size_t i = 0; i < N; i++) {
			if (ref[i] != f[i]) {
				printf("    coeff[%zu]: interp=0x%016llx lagrange=0x%016llx\n",
				       i, (unsigned long long)f[i],
				       (unsigned long long)ref[i]);
				ok = 0;
				break;
			}
		}
	}

	if (ok) {
		pass("interp == direct Lagrange basis (N=8), all 8 coeffs bit-exact");
	} else {
		fail("interp == direct Lagrange basis (N=8)");
	}

	gf64_subproduct_tree_free(&tree);
	free(f);
	free(L);
	free(ref);
}

/* ----------------------------------------------------------------------------
 * Test 3: N == 1 edge — the interpolant of a single point is the constant
 * values[0].
 * ---------------------------------------------------------------------------- */
static void test_n1(void) {
	printf("Test 3: interp with N=1 returns the constant values[0]\n");

	gf64_t points[1] = { 0x1234ULL };
	gf64_t values[1] = { 0xDEADBEEFCAFEBABEULL };
	gf64_t out[1]    = { 0 };

	SubproductTree tree;
	gf64_subproduct_tree_build(points, 1, &tree);
	gf64_multi_point_interp_internal(&tree, values, out);
	gf64_subproduct_tree_free(&tree);

	if (out[0] == values[0]) {
		pass("interp(N=1) == values[0]");
	} else {
		printf("    out=0x%016llx want=0x%016llx\n",
		       (unsigned long long)out[0], (unsigned long long)values[0]);
		fail("interp(N=1) == values[0]");
	}
}

/* ----------------------------------------------------------------------------
 * Test 4: repeated-call determinism — two identical calls must produce
 * bit-identical output (guards against scratch aliasing / uninitialized
 * reads in the combine recursion).
 * ---------------------------------------------------------------------------- */
static void test_determinism(void) {
	printf("Test 4: repeated interp calls are bit-identical\n");

	const size_t N = 64;
	gf64_t *points = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *values = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *out1   = (gf64_t *)calloc(N, sizeof(gf64_t));
	gf64_t *out2   = (gf64_t *)calloc(N, sizeof(gf64_t));
	if (!points || !values || !out1 || !out2) {
		printf("    alloc failed\n");
		fail("determinism alloc");
		free(points); free(values); free(out1); free(out2);
		return;
	}

	put_seed(0xD3D3D3D3D3D3D3D3ULL);
	for (size_t i = 0; i < N; i++) {
		gf64_t v;
		do {
			v = splitmix64_next();
		} while (v == 0);
		points[i] = v;
	}
	put_seed(0x0DDBA11AA1100001ULL);
	for (size_t i = 0; i < N; i++) {
		values[i] = splitmix64_next();
	}

	SubproductTree tree;
	gf64_subproduct_tree_build(points, N, &tree);
	gf64_multi_point_interp_internal(&tree, values, out1);
	gf64_multi_point_interp_internal(&tree, values, out2);

	int ok = 1;
	for (size_t i = 0; i < N; i++) {
		if (out1[i] != out2[i]) {
			printf("    coeff[%zu]: call1=0x%016llx call2=0x%016llx\n",
			       i, (unsigned long long)out1[i],
			       (unsigned long long)out2[i]);
			ok = 0;
			break;
		}
	}

	if (ok) {
		pass("two identical interp calls produce identical output (N=64)");
	} else {
		fail("interp determinism (N=64)");
	}

	gf64_subproduct_tree_free(&tree);
	free(points); free(values); free(out1); free(out2);
}

/* ----------------------------------------------------------------------------
 * Test 5: weights-aware interp parity — interp_weights(tree, values,
 * barycentric_weights) must be bit-identical to interp_internal(tree,
 * values). Pins the A1 optimization contract that the Fenger pipeline
 * relies on (prepare amortizes the weights; each word skips the MPE +
 * inversions).
 * ---------------------------------------------------------------------------- */
static void test_weights_parity(void) {
	printf("Test 5: interp_weights == interp_internal (N=64 and N=256)\n");

	const size_t sizes[] = { 64, 256 };
	const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

	for (size_t si = 0; si < num_sizes; si++) {
		const size_t N = sizes[si];

		gf64_t *points  = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *values  = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *weights = (gf64_t *)malloc(N * sizeof(gf64_t));
		gf64_t *out_ref = (gf64_t *)calloc(N, sizeof(gf64_t));
		gf64_t *out_w   = (gf64_t *)calloc(N, sizeof(gf64_t));
		if (!points || !values || !weights || !out_ref || !out_w) {
			printf("    N=%zu alloc failed\n", N);
			fail("weights parity alloc");
			free(points); free(values); free(weights);
			free(out_ref); free(out_w);
			continue;
		}

		put_seed(0xBADBEEF000000000ULL ^ (uint64_t)N);
		for (size_t i = 0; i < N; i++) {
			gf64_t v;
			do {
				v = splitmix64_next();
			} while (v == 0);
			points[i] = v;
		}
		put_seed(0xC0FFEE0000000000ULL ^ (uint64_t)N);
		for (size_t i = 0; i < N; i++) {
			values[i] = splitmix64_next();
		}

		SubproductTree tree;
		gf64_subproduct_tree_build(points, N, &tree);
		gf64_barycentric_weights(&tree, weights);

		gf64_multi_point_interp_internal(&tree, values, out_ref);
		gf64_multi_point_interp_weights(&tree, values, weights, out_w);

		int ok = 1;
		for (size_t i = 0; i < N; i++) {
			if (out_ref[i] != out_w[i]) {
				printf("    N=%zu coeff[%zu]: internal=0x%016llx weights=0x%016llx\n",
				       N, i, (unsigned long long)out_ref[i],
				       (unsigned long long)out_w[i]);
				ok = 0;
				break;
			}
		}

		char label[128];
		snprintf(label, sizeof(label),
		         "interp_weights == interp_internal (N=%zu)", N);
		if (ok) {
			pass(label);
		} else {
			fail(label);
		}

		gf64_subproduct_tree_free(&tree);
		free(points); free(values); free(weights);
		free(out_ref); free(out_w);
	}
}

/* ----------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------------- */
int main(void) {
	printf("GF64 derivative-form interpolation tests (issue #59 A1)\n");
	printf("=======================================================\n\n");

	test_roundtrip();
	test_lagrange_reference();
	test_n1();
	test_determinism();
	test_weights_parity();

	printf("\n=== Summary ===\n");
	printf("Passed: %d\n", g_passed);
	printf("Failed: %d\n", g_failed);

	return g_failed > 0 ? 1 : 0;
}
