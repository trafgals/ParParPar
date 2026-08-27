/*
 * ============================================================================
 * gf64/test/test_gf64_mpe.c — Tests for the multi-point evaluation stack
 *
 * T8 of the par3-cauchy-fft-kernel plan. The eight tests below exercise:
 *
 *   1. multi_point_eval (random deg-100 poly, N=1024 random points):
 *      outputs match independent Horner eval at every point.
 *   2. multi_point_eval (random deg-10 poly, N=64 random points):
 *      same correctness contract on a smaller workload (verifies the
 *      function doesn't degenerate for small N or small degrees).
 *   3. poly_divmod: f = g*q + r reconstruction. Pick random deg_f > deg_g
 *      (and one deg_f < deg_g case), divide, reconstruct, and verify
 *      equivalence bit-for-bit.
 *   4. poly_invmod: g * (1/g) ≡ 1 mod x^n. Multiply g (truncated to n)
 *      with the n-coefficient inverse, then assert the product's first n
 *      coefficients are [1, 0, 0, ..., 0].
 *   5. poly_invmod zero-n edge: n==0 must produce a zero output, not abort.
 *   6. multi_point_eval empty tree: N==0 must return early, not abort.
 *   7. arena scratch-path heap-alloc count (issue #59 T4): counter must
 *      stay at 0 across 16 divmods; legacy path bumps it.
 *   8. gf64_arena_init overflow rejection (cubic P1, pr100 T4):
 *      `gf64_arena_init(&a, SIZE_MAX / 4)` must return 1 with
 *      `a.data == NULL` and `a.cap == 0`.
 *
 * Build & run from gf64/test/:
 *   $(CC) -O2 -march=native -I.. test_gf64_mpe.c \
 *         ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c \
 *         ../gf64_additive_fft.c ../gf64_mul_avx512.c \
 *         ../gf64_square.c ../gf64_single.c \
 *         -o test_gf64_mpe && ./test_gf64_mpe
 *
 * Output is plain text so it can be piped to
 * .omo/evidence/task-8-par3-cauchy-fft-kernel.log.
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
 * splitmix64 PRNG. Single source of randomness for all four tests; the seed
 * is set per-test with put_seed() so different tests get independent streams
 * without sharing state.
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

/*
 * Reference Horner evaluation. Independent of gf64_mpe.c (so the test
 * catches a Horner bug that the implementation copies). Constant-first
 * coefficients [c_0, ..., c_deg], constant 1 is the multiplicative identity
 * in the field.
 */
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
 * Test 1: random deg-100 poly, N=1024 random points.
 *
 * Build a subproduct tree on 1024 random points, evaluate the random
 * polynomial f at every leaf using gf64_multi_point_eval, and compare each
 * output to a from-scratch Horner evaluation of f at the same point.
 * ---------------------------------------------------------------------------- */
static void test_mpe_deg100_n1024(void) {
	printf("Test 1: multi_point_eval — random deg=100, N=1024 random points\n");

	const size_t deg_f = 100;
	const size_t N     = 1024;
	const size_t f_len = deg_f + 1;

	gf64_t *f      = (gf64_t *)malloc(f_len * sizeof(gf64_t));
	gf64_t *points = (gf64_t *)malloc(N     * sizeof(gf64_t));
	gf64_t *out    = (gf64_t *)calloc(N,    sizeof(gf64_t));
	gf64_t *ref    = (gf64_t *)calloc(N,    sizeof(gf64_t));

	if (!f || !points || !out || !ref) {
		printf("    alloc failed\n");
		fail("alloc test 1");
		free(f); free(points); free(out); free(ref);
		return;
	}

	/* Build f (random deg_f polynomial). */
	put_seed(0xDEADBEEFFEEDCAFEULL);
	for (size_t i = 0; i < f_len; i++) {
		f[i] = splitmix64_next();
	}
	/* Ensure leading coefficient is non-zero so deg(f) == deg_f. */
	if (f[deg_f] == 0) {
		f[deg_f] = 1ULL;
	}

	/* Build the 1024 random points. */
	put_seed(0xCAFEBABE5EED1234ULL);
	for (size_t i = 0; i < N; i++) {
		/* Points MUST be non-zero for the subproduct tree's leaves
		 * (x + x_i with x_i == 0 collapses to x, not x_i; this is a
		 * non-trivial structural requirement of Bostan-Schost, not a
		 * limitation of gf64_subproduct_tree_build). The engine
		 * guarantees x_i = firstInput + c >= 1, so we do the same. */
		gf64_t v;
		do {
			v = splitmix64_next();
		} while (v == 0);
		points[i] = v;
	}

	/* Build the subproduct tree. */
	SubproductTree tree;
	gf64_subproduct_tree_build(points, N, &tree);

	/* MPE call. */
	gf64_multi_point_eval(f, deg_f, &tree, out);

	/* Independent reference. */
	gf64_t *last_level = tree.level_data[tree.num_levels - 1];
	int ok = 1;
	int first_mismatch = -1;
	for (size_t j = 0; j < N; j++) {
		gf64_t xj = last_level[2 * j];
		ref[j] = horner_eval_reference(f, deg_f, xj);
		if (out[j] != ref[j]) {
			if (first_mismatch < 0) first_mismatch = (int)j;
			ok = 0;
		}
	}

	if (ok) {
		pass("MPE(deg=100, N=1024) == independent Horner (1024/1024)");
	} else {
		printf("    first mismatch at j=%d: out=0x%016llx ref=0x%016llx\n",
		       first_mismatch,
		       (unsigned long long)out[first_mismatch],
		       (unsigned long long)ref[first_mismatch]);
		fail("MPE(deg=100, N=1024) bit-exact match");
	}

	gf64_subproduct_tree_free(&tree);
	free(f); free(points); free(out); free(ref);
}

/* ----------------------------------------------------------------------------
 * Test 2: random deg-10 poly, N=64 random points.
 *
 * Same contract on a small workload to catch any off-by-one in the leaf
 * indexing that a larger N might mask.
 * ---------------------------------------------------------------------------- */
static void test_mpe_deg10_n64(void) {
	printf("Test 2: multi_point_eval — random deg=10, N=64 random points\n");

	const size_t deg_f = 10;
	const size_t N     = 64;
	const size_t f_len = deg_f + 1;

	gf64_t *f      = (gf64_t *)malloc(f_len * sizeof(gf64_t));
	gf64_t *points = (gf64_t *)malloc(N     * sizeof(gf64_t));
	gf64_t *out    = (gf64_t *)calloc(N,    sizeof(gf64_t));

	if (!f || !points || !out) {
		printf("    alloc failed\n");
		fail("alloc test 2");
		free(f); free(points); free(out);
		return;
	}

	put_seed(0x12345678ABCDEF01ULL);
	for (size_t i = 0; i < f_len; i++) {
		f[i] = splitmix64_next();
	}
	if (f[deg_f] == 0) f[deg_f] = 1ULL;

	put_seed(0x9876543210FEDCBAULL);
	for (size_t i = 0; i < N; i++) {
		gf64_t v;
		do {
			v = splitmix64_next();
		} while (v == 0);
		points[i] = v;
	}

	SubproductTree tree;
	gf64_subproduct_tree_build(points, N, &tree);

	gf64_multi_point_eval(f, deg_f, &tree, out);

	gf64_t *last_level = tree.level_data[tree.num_levels - 1];
	int ok = 1;
	int first_mismatch = -1;
	for (size_t j = 0; j < N; j++) {
		gf64_t xj = last_level[2 * j];
		gf64_t ref = horner_eval_reference(f, deg_f, xj);
		if (out[j] != ref) {
			if (first_mismatch < 0) first_mismatch = (int)j;
			ok = 0;
		}
	}

	if (ok) {
		pass("MPE(deg=10, N=64) == independent Horner (64/64)");
	} else {
		printf("    first mismatch at j=%d: out=0x%016llx\n",
		       first_mismatch, (unsigned long long)out[first_mismatch]);
		fail("MPE(deg=10, N=64) bit-exact match");
	}

	gf64_subproduct_tree_free(&tree);
	free(f); free(points); free(out);
}

/* ----------------------------------------------------------------------------
 * Test 3: gf64_poly_divmod f = g*q + r reconstruction.
 *
 * Pick random f (deg_f=50), random g (deg_g=10), divide; reconstruct
 * (g*q + r) and assert element-by-element equality with f.
 *
 * Also exercise the deg_f < deg_g edge case: r = f, q = 0.
 * ---------------------------------------------------------------------------- */
static void test_divmod_reconstruction(void) {
	printf("Test 3: gf64_poly_divmod — f = g*q + r reconstruction\n");

	/* Case A: deg_f > deg_g. */
	const size_t deg_f = 50;
	const size_t deg_g = 10;
	const size_t f_len = deg_f + 1;
	const size_t g_len = deg_g + 1;
	const size_t q_size = deg_f - deg_g + 1;

	gf64_t *f      = (gf64_t *)malloc(f_len * sizeof(gf64_t));
	gf64_t *g      = (gf64_t *)malloc(g_len * sizeof(gf64_t));
	gf64_t *q      = (gf64_t *)calloc(q_size, sizeof(gf64_t));
	/* r must hold at least deg_f + 1 coefficients (working buffer). */
	gf64_t *r      = (gf64_t *)calloc(deg_f + 1, sizeof(gf64_t));
	gf64_t *recon  = (gf64_t *)calloc(f_len,    sizeof(gf64_t));

	if (!f || !g || !q || !r || !recon) {
		printf("    alloc failed (case A)\n");
		fail("alloc test 3A");
		goto case_b;
	}

	put_seed(0xA1A2A3A4A5A6A7A8ULL);
	for (size_t i = 0; i < f_len; i++) f[i] = splitmix64_next();
	for (size_t i = 0; i < g_len; i++) g[i] = splitmix64_next();
	if (g[deg_g] == 0) g[deg_g] = 1ULL; /* required */

	gf64_poly_divmod(f, deg_f, g, deg_g, q, r);

	/* Reconstruct: g*q + r.
	 *
	 * Note: r only has deg_g coefficients valid (degree < deg_g). r was
	 * overwritten in place by gf64_poly_divmod, so the "extension"
	 * coefficients r[deg_g..deg_f] are unspecified by the contract; we
	 * ignore them.
	 */
	gf64_poly_mul(recon, g, deg_g, q, deg_f - deg_g);
	/* Add r via element-wise XOR for the first deg_g coefficients; the
	 * deg_g slot (if q has nonzero there) is not used because deg(r) <
	 * deg_g. */
	for (size_t i = 0; i < deg_g; i++) {
		recon[i] ^= r[i];
	}

	int ok = 1;
	for (size_t i = 0; i < f_len; i++) {
		if (recon[i] != f[i]) {
			printf("    case A: i=%zu recon=0x%016llx f=0x%016llx\n",
			       i, (unsigned long long)recon[i],
			       (unsigned long long)f[i]);
			ok = 0;
			break;
		}
	}
	if (ok) {
		pass("divmod (deg_f=50, deg_g=10): g*q + r == f bit-exact");
	} else {
		fail("divmod (deg_f=50, deg_g=10) reconstruction");
	}

	free(f); free(g); free(q); free(r); free(recon);

	/* Case B: deg_f < deg_g. r = f, q = 0. */
case_b: {
	const size_t fB_deg = 5;
	const size_t gB_deg = 10;
	gf64_t *fB   = (gf64_t *)malloc((fB_deg + 1) * sizeof(gf64_t));
	gf64_t *gB   = (gf64_t *)malloc((gB_deg + 1) * sizeof(gf64_t));
	gf64_t *qB   = (gf64_t *)calloc(1, sizeof(gf64_t));
	gf64_t *rB   = (gf64_t *)calloc(gB_deg, sizeof(gf64_t));
	if (!fB || !gB || !qB || !rB) {
		printf("    alloc failed (case B)\n");
		fail("alloc test 3B");
		goto case_c;
	}
	put_seed(0xB1B2B3B4B5B6B7B8ULL);
	for (size_t i = 0; i <= fB_deg; i++) fB[i] = splitmix64_next();
	for (size_t i = 0; i <= gB_deg; i++) gB[i] = splitmix64_next();
	if (gB[gB_deg] == 0) gB[gB_deg] = 1ULL;

	gf64_poly_divmod(fB, fB_deg, gB, gB_deg, qB, rB);

	int okB = 1;
	if (qB[0] != 0) {
		printf("    case B: q[0] = 0x%016llx, want 0\n",
		       (unsigned long long)qB[0]);
		okB = 0;
	}
	for (size_t i = 0; i <= fB_deg; i++) {
		if (rB[i] != fB[i]) {
			printf("    case B: r[%zu] = 0x%016llx, want f[%zu] = 0x%016llx\n",
			       i, (unsigned long long)rB[i], i,
			       (unsigned long long)fB[i]);
			okB = 0;
			break;
		}
	}
	for (size_t i = fB_deg + 1; i <= gB_deg - 1; i++) {
		if (rB[i] != 0) {
			printf("    case B: r[%zu] = 0x%016llx, want 0\n",
			       i, (unsigned long long)rB[i]);
			okB = 0;
			break;
		}
	}

	if (okB) {
		pass("divmod (deg_f < deg_g): r == f, q == 0");
	} else {
		fail("divmod (deg_f < deg_g) edge case");
	}

	free(fB); free(gB); free(qB); free(rB);
}

	/* Case C: deg_f == deg_g (just slightly over to ensure single-step
	 * divmod). q has 1 nonzero coefficient. */
case_c: {
	const size_t degC = 20;
	gf64_t *fC  = (gf64_t *)malloc((degC + 1) * sizeof(gf64_t));
	gf64_t *gC  = (gf64_t *)malloc((degC + 1) * sizeof(gf64_t));
	gf64_t *qC  = (gf64_t *)calloc(1, sizeof(gf64_t));
	gf64_t *rC  = (gf64_t *)calloc(degC + 1, sizeof(gf64_t));
	gf64_t *rec = (gf64_t *)calloc(degC + 1, sizeof(gf64_t));
	if (!fC || !gC || !qC || !rC || !rec) {
		printf("    alloc failed (case C)\n");
		fail("alloc test 3C");
		free(fC); free(gC); free(qC); free(rC); free(rec);
		return;
	}
	put_seed(0xC1C2C3C4C5C6C7C8ULL);
	for (size_t i = 0; i <= degC; i++) fC[i] = splitmix64_next();
	for (size_t i = 0; i <= degC; i++) gC[i] = splitmix64_next();
	if (gC[degC] == 0) gC[degC] = 1ULL;

	gf64_poly_divmod(fC, degC, gC, degC, qC, rC);

	gf64_poly_mul(rec, gC, degC, qC, 0); /* degree of qC is 0 */
	for (size_t i = 0; i < degC; i++) rec[i] ^= rC[i];

	int okC = 1;
	for (size_t i = 0; i <= degC; i++) {
		if (rec[i] != fC[i]) {
			printf("    case C: i=%zu rec=0x%016llx f=0x%016llx\n",
			       i, (unsigned long long)rec[i],
			       (unsigned long long)fC[i]);
			okC = 0;
			break;
		}
	}
	if (okC) {
		pass("divmod (deg_f == deg_g): g*q + r == f bit-exact");
	} else {
		fail("divmod (deg_f == deg_g) reconstruction");
	}

	free(fC); free(gC); free(qC); free(rC); free(rec);
}
}

/* ----------------------------------------------------------------------------
 * Test 4: gf64_poly_invmod g * (1/g) ≡ 1 mod x^n.
 *
 * Strategy: pick a random polynomial g with g[0] != 0. Compute
 * `inv_g = gf64_poly_invmod(g, deg_g, n)`. Multiply g (truncated to n
 * coefficients) by inv_g and verify the product's first n coefficients
 * are [1, 0, 0, ..., 0].
 *
 * The check is element-by-element so it does not rely on Horner at x=0;
 * any coefficient x^k for k >= 1 of (g * inv_g mod x^n) must be 0.
 * ---------------------------------------------------------------------------- */
static void test_invmod(void) {
	printf("Test 4: gf64_poly_invmod — g * (1/g) ≡ 1 mod x^n (multiple n)\n");

	/* n_cases includes NON-power-of-2 sizes (96, 192, 255): the final
	 * Newton step then has m_new = n with m = 2^floor(log2(n-1)) > n/2,
	 * so the full product g * r^2 is up to 3n - 2 coefficients long —
	 * regression pin for the issue #59 A1 prod-buffer sizing fix (the
	 * old 2n sizing overflowed the heap, e.g. n = 96 writes 222
	 * coefficients into a 192-slot buffer). */
	const size_t n_cases[] = { 1, 2, 3, 4, 7, 8, 16, 32, 64, 96, 128, 192, 255 };
	const size_t num_n = sizeof(n_cases) / sizeof(n_cases[0]);
	int all_ok = 1;

	for (size_t ci = 0; ci < num_n; ci++) {
		size_t n = n_cases[ci];
		const size_t deg_g = (n < 8) ? (n - 1) : 7;

		gf64_t *g     = (gf64_t *)malloc((deg_g + 1) * sizeof(gf64_t));
		gf64_t *inv   = (gf64_t *)calloc(n,           sizeof(gf64_t));
		gf64_t *prod  = (gf64_t *)calloc(2 * n - 1,   sizeof(gf64_t));
		if (!g || !inv || !prod) {
			printf("    n=%zu alloc failed\n", n);
			all_ok = 0;
			free(g); free(inv); free(prod);
			continue;
		}

		put_seed(0xD0D0D0D0D0D0D0D0ULL ^ (uint64_t)ci);
		for (size_t i = 0; i <= deg_g; i++) g[i] = splitmix64_next();
		/* g[0] must be non-zero for the inverse to exist. */
		if (g[0] == 0) g[0] = 1ULL;

		gf64_poly_invmod(g, deg_g, n, inv);

		/* prod = g * inv. Use gf64_poly_mul; the result has up to
		 * (deg_g + (n - 1)) coefficients. The product is exactly
		 * 1 + x * (something) when restricted mod x^n. */
		gf64_poly_mul(prod, g, deg_g, inv, n - 1);

		int okN = 1;
		/* Check the first n coefficients of prod. */
		if (prod[0] != 1ULL) {
			printf("    n=%zu: prod[0] = 0x%016llx, want 1\n",
			       n, (unsigned long long)prod[0]);
			okN = 0;
		}
		for (size_t k = 1; k < n; k++) {
			if (prod[k] != 0) {
				printf("    n=%zu: prod[%zu] = 0x%016llx, want 0\n",
				       n, k, (unsigned long long)prod[k]);
				okN = 0;
				break;
			}
		}

		if (okN) {
			pass("invmod: g * inv_g == 1 mod x^n (n varying)");
		} else {
			fail("invmod product check");
			all_ok = 0;
		}

		free(g); free(inv); free(prod);
	}

	(void)all_ok; /* The per-n pass/fail already accumulates. */
}

/* ----------------------------------------------------------------------------
 * Test 5 (bonus, not in the spec but cheap): n == 0 must be a no-op.
 * ---------------------------------------------------------------------------- */
static void test_invmod_zero_n(void) {
	printf("Test 5: gf64_poly_invmod(n=0) is a no-op\n");
	gf64_t g = 0xCAFEBABE;
	gf64_t dst = 0xDEADBEEF;
	gf64_poly_invmod(&g, 0, 0, &dst);
	if (dst == 0xDEADBEEF) {
		pass("invmod(n=0) leaves dst untouched");
	} else {
		printf("    dst=0x%016llx, want 0xDEADBEEF\n",
		       (unsigned long long)dst);
		fail("invmod(n=0) no-op contract");
	}

	/* Buffer-arithmetic overflow guard (cubic review c509dd2b P1): n
	 * beyond SIZE_MAX/3 must be refused without writing — the Newton
	 * buffers are sized 2n/3n and 3n - 2 coefficients are written, so
	 * a wrapped calloc would under-allocate and the loop would write
	 * out of bounds. On unfixed code this call aborts (calloc fails). */
	{
		gf64_t huge_dst[2] = { 0x1111111111111111ULL, 0x2222222222222222ULL };
		gf64_t huge_g = 1ULL;
		const size_t huge_n = SIZE_MAX / 3 + 1;
		gf64_poly_invmod(&huge_g, 0, huge_n, huge_dst);
		if (huge_dst[0] == 0x1111111111111111ULL &&
		    huge_dst[1] == 0x2222222222222222ULL) {
			pass("invmod(n > SIZE_MAX/3) refused without writing (overflow guard)");
		} else {
			printf("    dst modified by refused invmod call\n");
			fail("invmod overflow guard");
		}
	}
}

/* ----------------------------------------------------------------------------
 * Test 6 (bonus): gf64_multi_point_eval on an empty / NULL tree returns
 * without writing.
 * ---------------------------------------------------------------------------- */
static void test_mpe_empty(void) {
	printf("Test 6: multi_point_eval on empty / NULL tree returns immediately\n");

	gf64_t f[2] = {0x1ULL, 0x2ULL};
	gf64_t out[4] = {0xAA, 0xBB, 0xCC, 0xDD};

	/* NULL tree */
	gf64_multi_point_eval(f, 1, NULL, out);
	if (out[0] == 0xAA && out[3] == 0xDD) {
		pass("MPE(NULL tree) leaves out untouched");
	} else {
		fail("MPE(NULL tree) no-op contract");
	}

	/* Empty tree. */
	SubproductTree empty;
	gf64_subproduct_tree_build(NULL, 0, &empty);
	gf64_multi_point_eval(f, 1, &empty, out);
	gf64_subproduct_tree_free(&empty);
	if (out[0] == 0xAA && out[3] == 0xDD) {
		pass("MPE(empty tree) leaves out untouched");
	} else {
		fail("MPE(empty tree) no-op contract");
	}
}

/* ----------------------------------------------------------------------------
 * Test 7 (issue #59 T4): scratch-path heap alloc count.
 *
 * The T4 arena routes the divmod/invmod working buffers through a caller-
 * owned bump allocator instead of malloc/free. Bit-exact parity is
 * covered by test_gf64_divmod_parity Test 6; this test pins the heap
 * side of the contract:
 *
 *   - scratch path:  gf64_poly_divmod_scratch / _invmod_scratch add zero
 *                   heap allocations (the working buffers come from the
 *                   arena; the counter is only bumped in the malloc branch
 *                   which the scratch code skips).
 *   - legacy path:  the malloc branch bumps the counter per call (5 per
 *                   divmod + 3 per invmod), so the counter must MOVE
 *                   after even one legacy call. Sanity check on the
 *                   instrumentation itself.
 *
 * The assertions are independent of the specific arithmetic result — they
 * care only about the heap-alloc count contract.
 * ------------------------------------------------------------------------- */
static void test_alloc_count(void) {
	printf("Test 7: scratch-path heap alloc count (issue #59 T4)\n");

	/* Build (deg_f=200, deg_g=100) so m=101 (> GF64_DIVMOD_NEWTON_MIN)
	 * exercises the Newton path's 5+3 buffers per call. */
	const size_t df = 200, dg = 100;
	gf64_t *f   = (gf64_t *)malloc((df + 1) * sizeof(gf64_t));
	gf64_t *g   = (gf64_t *)malloc((dg + 1) * sizeof(gf64_t));
	size_t qw = df - dg + 1;
	size_t rw = (dg > df) ? (dg + 1) : (df + 1);
	gf64_t *q   = (gf64_t *)calloc(qw, sizeof(gf64_t));
	gf64_t *r   = (gf64_t *)calloc(rw, sizeof(gf64_t));
	gf64_t *inv = (gf64_t *)calloc(df + 1, sizeof(gf64_t));
	if (!f || !g || !q || !r || !inv) {
		fail("alloc-count setup: out of memory");
		goto cleanup_setup;
	}
	put_seed(0xCAFEBABEDEADBEEFULL);
	for (size_t i = 0; i <= df; i++) f[i] = splitmix64_next();
	for (size_t i = 0; i <= dg; i++) g[i] = splitmix64_next();
	/* Keep the leading divisor coefficient nonzero so the legacy path
	 * doesn't abort on the synthetic-input edge case. */
	if (g[dg] == 0) g[dg] = 1ULL;

	/* --- Scratch path: zero heap allocations expected. --- */
	gf64_arena_t arena;
	if (gf64_arena_init(&arena, 16 * (df + dg + 4)) != 0) {
		fail("alloc-count: arena init");
		goto cleanup_setup;
	}
	gf64_mpe_heap_alloc_count = 0;
	for (int iter = 0; iter < 16; iter++) {
		gf64_poly_divmod_scratch(f, df, g, dg, q, r, &arena);
	}
	if (gf64_mpe_heap_alloc_count == 0) {
		pass("scratch path: 16 divmods add 0 heap allocations");
	} else {
		char msg[128];
		snprintf(msg, sizeof(msg),
		         "scratch path: 16 divmods added %zu heap allocs (want 0)",
		         gf64_mpe_heap_alloc_count);
		fail(msg);
	}
	gf64_arena_release(&arena, 0);
	gf64_arena_free(&arena);

	/* --- Legacy path: counter MUST move. --- */
	gf64_mpe_heap_alloc_count = 0;
	gf64_poly_divmod(f, df, g, dg, q, r);
	/* One divmod call bumps the counter by 5 (rev_f + rev_g + inv +
	 * rev_q + gq); the invmod inside also adds 3 (g_buf + r_sq +
	 * prod). Total = 8 per legacy call. Assert >= 5 (loose floor). */
	if (gf64_mpe_heap_alloc_count >= 5) {
		char msg[128];
		snprintf(msg, sizeof(msg),
		         "legacy path: 1 divmod bumped counter by %zu (>=5 OK)",
		         gf64_mpe_heap_alloc_count);
		pass(msg);
	} else {
		fail("legacy path: counter did not move (instrumentation broken?)");
	}

cleanup_setup:
	free(f); free(g); free(q); free(r); free(inv);
}

/* ----------------------------------------------------------------------------
 * Test 8 (cubic P1, pr100 task 4): gf64_arena_init size_t overflow rejection.
 *
 * Pre-fix bug: gf64_arena_init() called malloc(words * sizeof(gf64_t)) without
 * first rejecting words that would cause the multiplication to wrap. With
 * adversarial input the multiplication wraps modulo 2^64, malloc sees a
 * (possibly small) size and returns a (possibly small) buffer, but the
 * public surface promises "nonzero on allocation failure (arena left
 * zeroed)" without an explicit overflow guard. On allocators that accept
 * the overcommit (Linux overcommit_memory=1 with permissive mmap), the
 * function returns 0 with a tiny buffer but a huge `cap` field, and any
 * subsequent gf64_arena_push() writes OOB.
 *
 * The fix (cubic P1): a single guard at the top of gf64_arena_init rejects
 * `words > SIZE_MAX / sizeof(gf64_t)` before the multiplication, treating
 * the overflow as an allocation failure with the same zero-state contract
 * the rest of the function already provides.
 *
 * This test pins the new contract with TWO adversarial inputs:
 *
 *   (a) SIZE_MAX / 4 (= 2^62 - 1)  → multiplication wraps to SIZE_MAX-7,
 *       which most allocators reject with NULL even without the guard.
 *       Asserts the guard fires and returns the zero-state contract.
 *
 *   (b) SIZE_MAX / 8 + 2           → multiplication wraps to 8 bytes,
 *       so malloc(8) succeeds and the unfixed code returns 0 with
 *       cap = 2^61 + 1 — the actual exploitable OOB. On the unfixed
 *       code (a) succeeds, (b) fails. On the fixed code both succeed.
 *       This is the test that *actually* proves the guard is needed.
 *
 * Both inputs share the assertions:
 *   - rc == 1 (failure)
 *   - a.data == NULL
 *   - a.cap  == 0
 * ------------------------------------------------------------------------- */
static void test_arena_init_overflow_rejected(void) {
	printf("Test 8: gf64_arena_init overflow rejection (cubic P1, pr100 T4)\n");

	const size_t cap_limit = SIZE_MAX / sizeof(gf64_t);

	struct {
		const char *name;
		size_t      words;
	} cases[] = {
		{ "SIZE_MAX/4",      SIZE_MAX / 4     }, /* wraps to SIZE_MAX-7 */
		{ "SIZE_MAX/8 + 2",  SIZE_MAX / 8 + 2 }, /* wraps to 8 (OOB exploit) */
	};

	for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		gf64_arena_t a;
		a.data = (gf64_t *)0xDEADBEEFDEADBEEFULL;  /* sentinel */
		a.cap  = 0xCAFEBABEDEADBEEFULL;            /* sentinel */
		a.used = 0xFEEDFACE12345678ULL;            /* sentinel */

		if (cases[i].words <= cap_limit) {
			printf("  FAIL: %s: misconfigured (words=%zu must exceed cap_limit=%zu)\n",
			       cases[i].name, cases[i].words, cap_limit);
			fail("arena overflow: test misconfigured");
			continue;
		}

		size_t wrapped = cases[i].words * sizeof(gf64_t);
		printf("  case %s: words=%zu wrapped_size=%zu cap_limit=%zu\n",
		       cases[i].name, cases[i].words, wrapped, cap_limit);

		int rc = gf64_arena_init(&a, cases[i].words);
		char msg[160];
		if (rc == 1 && a.data == NULL && a.cap == 0) {
			snprintf(msg, sizeof(msg),
			         "arena_init(%s) returns 1 with data=NULL cap=0",
			         cases[i].name);
			pass(msg);
		} else {
			snprintf(msg, sizeof(msg),
			         "arena_init(%s): rc=%d data=%p cap=%zu (want 1, NULL, 0)",
			         cases[i].name, rc, (void *)a.data, a.cap);
			fail(msg);
		}
	}
}

/* ----------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------------- */
int main(void) {
	printf("GF64 multi-point evaluation tests (T8 of par3-cauchy-fft-kernel)\n");
	printf("=================================================================\n\n");

	test_mpe_deg100_n1024();
	test_mpe_deg10_n64();
	test_divmod_reconstruction();
	test_invmod();
	test_invmod_zero_n();
	test_mpe_empty();
	test_alloc_count();
	test_arena_init_overflow_rejected();

	printf("\n=== Summary ===\n");
	printf("Passed: %d\n", g_passed);
	printf("Failed: %d\n", g_failed);

	return g_failed > 0 ? 1 : 0;
}
