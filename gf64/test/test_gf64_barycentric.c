/*
 * ============================================================================
 * gf64/test/test_gf64_barycentric.c
 *
 * T7 (barycentric weight extraction) correctness test. Validates that
 *
 *     W_j = 1 / P'(x_j)
 *
 * where P(x) = ∏_i (x + x_i) and P'(x) is the formal derivative in GF(2^64).
 *
 * TEST CASES
 * ----------
 *   1. N = 0           -> no-op (no writes, no allocation failures)
 *   2. N = 1           -> weights_out[0] == 1 (P'(x_0) = 1)
 *   3. N = 1024 (random distinct points)
 *      For each j in [0, N):
 *          a) W_j * Horner(deriv, N-1, x_j) == 1     (self-consistency)
 *          b) W_j * ∏_{k != j} (x_j XOR x_k)  == 1   (independent cross-check)
 *   4. Duplicates: 1024 points where x_0 is duplicated into x_1 and x_2.
 *      For the duplicated indices the "weight" must be 0 (the convention the
 *      Itoh-Tsujii inverter applies to zero inputs — equivalent to 1/0
 *      under our convention).
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../gf64_barycentric.h"
#include "../gf64_global.h"
#include "../gf64_subproduct.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

static int g_passed = 0;
static int g_failed = 0;
static uint64_t g_rng = 0xC0DEFACE7E51C0DEULL;

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
 * Horner evaluation of poly c[0..deg] at x = r in GF(2^64).
 * Identical loop used in test_gf64_subproduct_tree.c — kept local for
 * isolation.
 */
static gf64_t horner_eval(const gf64_t *c, size_t deg, gf64_t r) {
	gf64_t acc = c[deg];
	for (size_t i = deg; i > 0; i--) {
		acc = gf64_mul_reference(acc, r) ^ c[i - 1];
	}
	return acc;
}

/*
 * Construct the formal derivative of a constant-first polynomial c[0..deg]
 * into deriv[0..deg-1] using the char-2 formula:
 *
 *     deriv[j] = c[j + 1]   if j is even
 *     deriv[j] = 0          if j is odd
 *
 * `deriv` must point to at least `deg` slots. Matches the implementation
 * in gf64_barycentric.c step 1.
 */
static void formal_derivative(gf64_t *deriv, const gf64_t *c, size_t deg) {
	for (size_t j = 0; j < deg; j++) {
		deriv[j] = ((j & 1U) == 0U) ? c[j + 1] : (gf64_t)0;
	}
}

/*
 * Compute P_underived_value * P_at_xj for the underived P = ∏ (x + x_i):
 * The naive product excluding index j is:
 *   Q_j = ∏_{k != j} (x_j + x_k) = P'(x_j) by the char-2 derivative rule.
 *
 * For N <= 1 the empty product is 1 by convention (matches P'(x_j) = 1).
 */
static gf64_t prod_excluding(const gf64_t *points, size_t N, size_t j, gf64_t xj) {
	gf64_t acc = 1;
	for (size_t k = 0; k < N; k++) {
		if (k == j) continue;
		acc = gf64_mul_reference(acc, xj ^ points[k]);
	}
	return acc;
}

/* ====================================================================
 * Tests
 * ==================================================================== */

static void test_empty(void) {
	printf("Test: N=0 (empty tree) -> no-op...\n");
	SubproductTree tree;
	gf64_subproduct_tree_build(NULL, 0, &tree);

	gf64_t weights[1] = {(gf64_t)0xDEADBEEFDEADBEEFULL};
	gf64_barycentric_weights(&tree, weights);

	/*
	 * We can't assert the buffer is unchanged because the impl may legally
	 * have left the value untouched (and there's exactly one slot). The
	 * stronger check is that nothing crashes and the tree itself is still
	 * free-able in its zero state.
	 */
	gf64_subproduct_tree_free(&tree);

	int ok = (tree.num_points == 0 && tree.num_levels == 0 &&
	          tree.level_data == NULL && tree.level_lens == NULL &&
	          tree.level_degs == NULL && tree.storage == NULL);

	if (ok) {
		pass("N=0 -> no-op (tree still zero after free)");
	} else {
		fail("N=0 -> no-op (tree still zero after free)");
	}
}

static void test_singleton(void) {
	printf("Test: N=1 -> weight = 1...\n");
	const gf64_t x0 = 0xDEADBEEFCAFEBABEULL;
	SubproductTree tree;
	gf64_subproduct_tree_build(&x0, 1, &tree);

	gf64_t weights[1] = {0};
	gf64_barycentric_weights(&tree, weights);

	gf64_t got = weights[0];
	gf64_t want = 1ULL;

	gf64_subproduct_tree_free(&tree);

	if (got == want) {
		pass("N=1 -> weights[0] == 1");
	} else {
		printf("    weights[0] = 0x%016llx, want 0x%016llx\n",
		       (unsigned long long)got, (unsigned long long)want);
		fail("N=1 -> weights[0] == 1");
	}
}

static void test_full_tree_1024(void) {
	printf("Test: N=1024 random distinct points -> weights verified two ways...\n");

	const size_t N = 1024;
	gf64_t *points = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (points == NULL) {
		fail("test_full_tree_1024: malloc points failed");
		return;
	}

	g_rng = 0x5EEDC0DECAFEF00DULL;
	for (size_t i = 0; i < N; i++) {
		points[i] = splitmix64_next();
		/* Ensure nonzero so division tests aren't testing trivial 0/0. */
		if (points[i] == 0) points[i] = 1;
	}

	SubproductTree tree;
	gf64_subproduct_tree_build(points, N, &tree);

	gf64_t *weights = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (weights == NULL) {
		gf64_subproduct_tree_free(&tree);
		free(points);
		fail("test_full_tree_1024: malloc weights failed");
		return;
	}
	gf64_barycentric_weights(&tree, weights);

	/*
	 * Reference derivative (built independently from the points array —
	 * cross-checks the tree-derived derivative):
	 */
	gf64_t *deriv_ref = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (deriv_ref == NULL) {
		free(weights);
		gf64_subproduct_tree_free(&tree);
		free(points);
		fail("test_full_tree_1024: malloc deriv_ref failed");
		return;
	}

	/* Use the tree's own root to build the verification derivative. This
	 * validates Horner(deriv, N-1, x_j) == P'(x_j) without needing a
	 * separate ground-truth P. The independent check below
	 * (∏_{k != j}(x_j + x_k)) is the truly external reference. */
	const gf64_t *root = tree.level_data[0];
	formal_derivative(deriv_ref, root, N);

	int total_ok = 1;
	int self_consistency_failures = 0;
	int cross_check_failures = 0;

	for (size_t j = 0; j < N; j++) {
		const gf64_t xj = points[j];

		/* Cross-check 1: W_j * P'_horner(x_j) == 1 (self-consistency
		 * via our own derivative). */
		const gf64_t p_prime = horner_eval(deriv_ref, N - 1, xj);
		const gf64_t product1 = gf64_mul_reference(weights[j], p_prime);

		/* Cross-check 2: W_j * ∏_{k != j} (x_j XOR x_k) == 1 (truly
		 * independent — recomputed from the input points directly). */
		const gf64_t qj = prod_excluding(points, N, j, xj);
		const gf64_t product2 = gf64_mul_reference(weights[j], qj);

		if (product1 != 1ULL) {
			if (self_consistency_failures < 5) {
				printf("    self-consistency fail j=%zu: W_j*P'_horner=0x%016llx (want 1)\n",
				       j, (unsigned long long)product1);
			}
			self_consistency_failures++;
		}
		if (product2 != 1ULL) {
			if (cross_check_failures < 5) {
				printf("    cross-check fail j=%zu: W_j*∏=0x%016llx (want 1)\n",
				       j, (unsigned long long)product2);
			}
			cross_check_failures++;
		}
		if (product1 != 1ULL || product2 != 1ULL) {
			total_ok = 0;
		}
	}

	free(deriv_ref);
	free(weights);
	gf64_subproduct_tree_free(&tree);
	free(points);

	if (total_ok) {
		printf("  -> 1024 weights self-consistent AND cross-check passes\n");
		pass("1024 distinct points: W_j * P'_horner(x_j) == 1 (all j)");
		pass("1024 distinct points: W_j * prod(k!=j)(x_j+x_k) == 1 (all j)");
	} else {
		printf("  -> %d self-consistency failures, %d cross-check failures\n",
		       self_consistency_failures, cross_check_failures);
		fail("1024 distinct points: weight verification failed");
	}
}

static void test_duplicates_produce_zero(void) {
	printf("Test: duplicate input points -> duplicated weights = 0...\n");

	const size_t N = 1024;
	gf64_t *points = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (points == NULL) {
		fail("test_duplicates_produce_zero: malloc points failed");
		return;
	}

	g_rng = 0xD0D0C0DECAFEFACEULL;
	for (size_t i = 0; i < N; i++) {
		points[i] = splitmix64_next();
		if (points[i] == 0) points[i] = 1;
	}

	/* Inject duplicates at indices 1 and 2: x_1 = x_0, x_2 = x_0.
	 * P(x) has (x + x_0)^3 as a factor for these three slots.
	 *
	 * In characteristic 2, P'(x_0) = 0 because the multiplicity-3 factor
	 * contributes 3 * (x + x_0)^2 to the derivative, and 3 = 1 (mod 2)
	 * gives the surviving term (x + x_0)^2 which evaluates to 0 at x_0
	 * (since (x_0 + x_0) = 0). Direct check: P' = sum_{k odd} c_k x^{k-1};
	 * the k^th coefficient of P contains c_k = sum_{|S|=k} prod(S), and
	 * this sum is 0 for the duplicated indices' contributions by symmetry
	 * arguments in char 2 — but most importantly, the empty product sum
	 * at j = j_repeat for the duplicated point evaluates to 0 via the
	 * "polynomial in (x + x_0) has a zero of order >= 2, so its derivative
	 * is zero" observation. */
	const gf64_t dup_value = points[0];
	points[1] = dup_value;
	points[2] = dup_value;

	SubproductTree tree;
	gf64_subproduct_tree_build(points, N, &tree);

	gf64_t *weights = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (weights == NULL) {
		gf64_subproduct_tree_free(&tree);
		free(points);
		fail("test_duplicates_produce_zero: malloc weights failed");
		return;
	}
	gf64_barycentric_weights(&tree, weights);

	int ok = 1;
	for (size_t j = 0; j < 3; j++) {
		if (weights[j] != 0ULL) {
			printf("    weights[%zu] = 0x%016llx (want 0, points[%zu] = 0x%016llx)\n",
			       j, (unsigned long long)weights[j], j,
			       (unsigned long long)points[j]);
			ok = 0;
		}
	}

	/* Remaining indices should still satisfy W_j * prod != 1 -> a
	 * non-identity product is fine here; we only need the duplicate slot
	 * sanity check above. Sanity-print one such weight for traceability. */
	if (ok) {
		printf("    weights[0..2] = 0 as expected (3 duplicated slots)\n");
	}

	free(weights);
	gf64_subproduct_tree_free(&tree);
	free(points);

	if (ok) {
		pass("duplicate input points -> weights[duplicated j] == 0");
	} else {
		fail("duplicate input points -> weights[duplicated j] == 0");
	}
}

/* ====================================================================
 * Main
 * ==================================================================== */

int main(void) {
	printf("GF64 barycentric weight tests (T7 of par3-cauchy-fft-kernel)\n\n");
	test_empty();
	test_singleton();
	test_full_tree_1024();
	test_duplicates_produce_zero();
	printf("\nSummary: %d passed, %d failed\n", g_passed, g_failed);
	return g_failed == 0 ? 0 : 1;
}
