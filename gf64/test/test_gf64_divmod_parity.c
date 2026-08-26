/*
 * ============================================================================
 * gf64/test/test_gf64_divmod_parity.c — Newton-reciprocal divmod parity test
 *
 * Issue #59 A1 (subquadratic prep): gf64_poly_divmod now dispatches to a
 * Newton-reciprocal fast path for quotients of length m = deg_f - deg_g + 1
 * >= GF64_DIVMOD_NEWTON_MIN (96), delegating to the schoolbook reference
 * below it. This test pins the contract that BOTH paths are bit-exact to
 * the schoolbook reference (gf64_poly_divmod_schoolbook) across the
 * dispatch boundary and at unbalanced / edge shapes:
 *
 *   - m < 96   (schoolbook path taken by the dispatcher)
 *   - m == 95, 96, 97  (the dispatch boundary itself)
 *   - m large  (Newton path; the inner muls hit the HQC-FFT tier)
 *   - deg_f < deg_g, deg_f == deg_g, deg_g == 0 (edge cases)
 *   - sparse divisors (divide by a linear factor)
 *
 * Contract checked per case: fast q and fast r[0..deg_g-1] are
 * bit-identical to the schoolbook q and r[0..deg_g-1], AND the schoolbook
 * result itself satisfies f = g*q + r (reconstruction), so a bug shared
 * by both implementations cannot hide.
 *
 * Build & run from gf64/test/:
 *   make test_gf64_divmod_parity && ./test_gf64_divmod_parity
 *
 * Exits 0 on full pass; non-zero on any failure.
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../gf64_mpe.h"
#include "../gf64_additive_fft.h"
#include "../gf64_global.h"

/* ----------------------------------------------------------------------------
 * splitmix64 PRNG (same family as the other gf64 tests).
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
 * One parity case: random f (deg_f), random g (deg_g, nonzero leading
 * coeff); run both implementations; compare q and the valid part of r.
 * ---------------------------------------------------------------------------- */
static void run_case(size_t deg_f, size_t deg_g, uint64_t seed) {
	const size_t q_size = (deg_f >= deg_g) ? (deg_f - deg_g + 1) : 1;
	/* r working-buffer contract: deg_f + 1 normally; for deg_f < deg_g the
	 * implementation zeroes r[deg_f+1 .. deg_g-1], so the buffer must hold
	 * deg_g coefficients. */
	const size_t r_size = (deg_f >= deg_g) ? (deg_f + 1) : deg_g;

	gf64_t *f   = (gf64_t *)malloc((deg_f + 1) * sizeof(gf64_t));
	gf64_t *g   = (gf64_t *)malloc((deg_g + 1) * sizeof(gf64_t));
	gf64_t *q_fast = (gf64_t *)calloc(q_size, sizeof(gf64_t));
	gf64_t *r_fast = (gf64_t *)calloc(r_size, sizeof(gf64_t));
	gf64_t *q_ref  = (gf64_t *)calloc(q_size, sizeof(gf64_t));
	gf64_t *r_ref  = (gf64_t *)calloc(r_size, sizeof(gf64_t));

	if (!f || !g || !q_fast || !r_fast || !q_ref || !r_ref) {
		printf("    (deg_f=%zu, deg_g=%zu) alloc failed\n", deg_f, deg_g);
		fail("alloc");
		free(f); free(g); free(q_fast); free(r_fast); free(q_ref); free(r_ref);
		return;
	}

	put_seed(seed);
	for (size_t i = 0; i <= deg_f; i++) f[i] = splitmix64_next();
	for (size_t i = 0; i <= deg_g; i++) g[i] = splitmix64_next();
	if (deg_g == 0) {
		/* Constant divisor: must be nonzero. */
		if (g[0] == 0) g[0] = 1ULL;
	} else if (g[deg_g] == 0) {
		g[deg_g] = 1ULL; /* required nonzero leading coefficient */
	}

	gf64_poly_divmod_schoolbook(f, deg_f, g, deg_g, q_ref, r_ref);
	gf64_poly_divmod(f, deg_f, g, deg_g, q_fast, r_fast);

	/* Compare quotient. */
	int ok = 1;
	for (size_t i = 0; i < q_size; i++) {
		if (q_fast[i] != q_ref[i]) {
			printf("    (deg_f=%zu, deg_g=%zu) q[%zu] fast=0x%016llx ref=0x%016llx\n",
			       deg_f, deg_g, i,
			       (unsigned long long)q_fast[i],
			       (unsigned long long)q_ref[i]);
			ok = 0;
			break;
		}
	}
	/* Compare the FULL remainder working buffer r[0..r_size). The
	 * contract documents r[deg_g..] as "unspecified", but both
	 * implementations must leave it identically zeroed: the half-EGCD
	 * (gf64_poly_invmod_mod) memcpys the whole buffer and scans it for
	 * the true degree, so a nonzero high region is observable. */
	if (ok) {
		for (size_t i = 0; i < r_size; i++) {
			if (r_fast[i] != r_ref[i]) {
				printf("    (deg_f=%zu, deg_g=%zu) r[%zu] fast=0x%016llx ref=0x%016llx\n",
				       deg_f, deg_g, i,
				       (unsigned long long)r_fast[i],
				       (unsigned long long)r_ref[i]);
				ok = 0;
				break;
			}
		}
	}
	/* Reconstruction check on the SCHOOLBOOK result (independent of the
	 * fast path): f == g*q_ref + r_ref, deg r_ref < deg_g. */
	if (ok && deg_f >= deg_g) {
		gf64_t *recon = (gf64_t *)calloc(deg_f + 1, sizeof(gf64_t));
		if (recon == NULL) {
			printf("    (deg_f=%zu, deg_g=%zu) recon alloc failed\n", deg_f, deg_g);
			fail("recon alloc");
			ok = 0;
		} else {
			gf64_poly_mul(recon, g, deg_g, q_ref, deg_f - deg_g);
			for (size_t i = 0; i < deg_g; i++) {
				recon[i] ^= r_ref[i];
			}
			for (size_t i = 0; i <= deg_f; i++) {
				if (recon[i] != f[i]) {
					printf("    (deg_f=%zu, deg_g=%zu) recon[%zu]=0x%016llx f[%zu]=0x%016llx\n",
					       deg_f, deg_g, i,
					       (unsigned long long)recon[i], i,
					       (unsigned long long)f[i]);
					ok = 0;
					break;
				}
			}
			free(recon);
		}
	}

	if (ok) {
		char label[128];
		snprintf(label, sizeof(label),
		         "divmod parity (deg_f=%zu, deg_g=%zu): fast == schoolbook, f == g*q + r",
		         deg_f, deg_g);
		pass(label);
	} else {
		char label[128];
		snprintf(label, sizeof(label),
		         "divmod parity (deg_f=%zu, deg_g=%zu)", deg_f, deg_g);
		fail(label);
	}

	free(f); free(g); free(q_fast); free(r_fast); free(q_ref); free(r_ref);
}

/* ----------------------------------------------------------------------------
 * Test 5: dividend-buffer alias — callers may legally reuse the dividend
 * buffer as r (r == f). The Newton path copies f into r with memmove; a
 * memcpy here would be undefined behavior on overlapping objects (cubic
 * review c509dd2b P2). Reference runs first on the pristine f, then the
 * fast path runs with f doubling as the r working buffer.
 * ---------------------------------------------------------------------------- */
static void run_alias_case(size_t deg_f, size_t deg_g, uint64_t seed) {
	const size_t q_size = (deg_f >= deg_g) ? (deg_f - deg_g + 1) : 1;
	const size_t r_size = (deg_f >= deg_g) ? (deg_f + 1) : deg_g;

	gf64_t *f       = (gf64_t *)malloc((deg_f + 1) * sizeof(gf64_t));
	gf64_t *g       = (gf64_t *)malloc((deg_g + 1) * sizeof(gf64_t));
	gf64_t *q_alias = (gf64_t *)calloc(q_size, sizeof(gf64_t));
	gf64_t *q_ref   = (gf64_t *)calloc(q_size, sizeof(gf64_t));
	gf64_t *r_ref   = (gf64_t *)calloc(r_size, sizeof(gf64_t));
	if (!f || !g || !q_alias || !q_ref || !r_ref) {
		printf("    (deg_f=%zu, deg_g=%zu) alias alloc failed\n", deg_f, deg_g);
		fail("alias alloc");
		free(f); free(g); free(q_alias); free(q_ref); free(r_ref);
		return;
	}

	put_seed(seed);
	for (size_t i = 0; i <= deg_f; i++) f[i] = splitmix64_next();
	for (size_t i = 0; i <= deg_g; i++) g[i] = splitmix64_next();
	if (deg_g == 0) {
		if (g[0] == 0) g[0] = 1ULL;
	} else if (g[deg_g] == 0) {
		g[deg_g] = 1ULL; /* required nonzero leading coefficient */
	}

	/* Reference first (f must still be pristine when the fast path runs). */
	gf64_poly_divmod_schoolbook(f, deg_f, g, deg_g, q_ref, r_ref);
	/* Fast path with r aliasing f: f doubles as the r working buffer. */
	gf64_poly_divmod(f, deg_f, g, deg_g, q_alias, f);

	int ok = 1;
	for (size_t i = 0; i < q_size; i++) {
		if (q_alias[i] != q_ref[i]) {
			printf("    (alias deg_f=%zu, deg_g=%zu) q[%zu] alias=0x%016llx ref=0x%016llx\n",
			       deg_f, deg_g, i,
			       (unsigned long long)q_alias[i], (unsigned long long)q_ref[i]);
			ok = 0;
			break;
		}
	}
	if (ok) {
		for (size_t i = 0; i < r_size; i++) {
			if (f[i] != r_ref[i]) {
				printf("    (alias deg_f=%zu, deg_g=%zu) r[%zu] alias=0x%016llx ref=0x%016llx\n",
				       deg_f, deg_g, i,
				       (unsigned long long)f[i], (unsigned long long)r_ref[i]);
				ok = 0;
				break;
			}
		}
	}

	if (ok) {
		pass("divmod with r aliasing f (r == f) matches schoolbook (Newton path)");
	} else {
		fail("divmod r-aliases-f parity");
	}

	free(f); free(g); free(q_alias); free(q_ref); free(r_ref);
}

int main(void) {
	printf("GF64 Newton-reciprocal divmod parity tests (issue #59 A1)\n");
	printf("=========================================================\n\n");

	printf("Test 1: dispatch-boundary sweep (m = 95/96/97 straddles "
	       "GF64_DIVMOD_NEWTON_MIN)\n");
	run_case(150, 55,  0x1111111111111111ULL); /* m = 96  */
	run_case(150, 56,  0x2222222222222222ULL); /* m = 95  */
	run_case(151, 55,  0x3333333333333333ULL); /* m = 97  */
	run_case(100, 20,  0x4444444444444444ULL); /* m = 81  (schoolbook path) */
	run_case(96,  1,   0x5555555555555555ULL); /* m = 96  */

	printf("\nTest 2: Newton path at FFT-mul sizes\n");
	run_case(300,  100, 0xAAAAAAAA11111111ULL);
	run_case(511,  255, 0xBBBBBBBB22222222ULL);
	run_case(512,  256, 0xCCCCCCCC33333333ULL);
	run_case(1024, 512, 0xDDDDDDDD44444444ULL);
	run_case(2048, 1024, 0xEEEEEEEE55555555ULL);
	run_case(4096, 2048, 0xFFFFFFFF66666666ULL);

	printf("\nTest 3: unbalanced shapes\n");
	run_case(300, 1,   0x123456789ABCDEF0ULL); /* divide by linear factor */
	run_case(100, 99,  0x23456789ABCDEF01ULL); /* tiny quotient, huge divisor */
	run_case(1024, 1000, 0x3456789ABCDEF012ULL); /* m = 25 (schoolbook path) */
	run_case(2048, 100, 0x456789ABCDEF0123ULL); /* m = 1949 (Newton path) */

	printf("\nTest 4: edge shapes\n");
	run_case(20, 20, 0x56789ABCDEF01234ULL); /* deg_f == deg_g: single-step q */
	run_case(5,  10, 0x6789ABCDEF012345ULL); /* deg_f <  deg_g: r = f, q = 0 */
	run_case(0,  0,  0x789ABCDEF0123456ULL); /* constants */
	run_case(1,  0,  0x89ABCDEF01234567ULL); /* deg_g == 0 constant divisor */

	printf("\nTest 5: dividend-buffer alias (r == f)\n");
	run_alias_case(300, 100, 0xAAAA000011110000ULL); /* Newton path, m = 201 */
	run_alias_case(1024, 512, 0xBBBB000022220000ULL);

	/* -------------------------------------------------------------------------
	 * Test 6 (issue #59 T4): scratch-vs-malloc parity for gf64_poly_divmod.
	 * The _scratch variant routes its working buffers through a caller-owned
	 * arena instead of malloc/free; the bit-exact contract must hold
	 * across every dispatch boundary, the Newton/FFT tier, and edge
	 * shapes (deg_f < deg_g, deg_f == deg_g, deg_g == 0).
	 * ------------------------------------------------------------------------- */
	{
		struct scratch_case {
			size_t deg_f, deg_g;
		};
		static const struct scratch_case s_cases[] = {
			/* Schoolbook-path edges (m < GF64_DIVMOD_NEWTON_MIN). */
			{  10,   3 },
			{ 100,  20 },
			{  95,  50 }, /* m = 46 (schoolbook) */
			/* Dispatch-boundary (m == 96). */
			{ 150,  55 }, /* m = 96 */
			{ 200,  99 }, /* m = 102 */
			/* Newton path through the HQC-FFT tier. */
			{ 300, 100 },
			{1024, 512 },
			{4096,2048 },
			{4099,   1 },
			/* Edge shapes. */
			{ 100, 200 }, /* deg_f < deg_g */
			{  50,  50 }, /* deg_f == deg_g */
			{   7,   0 }, /* deg_g == 0 constant divisor */
		};
		printf("\nTest 6: scratch variant parity (issue #59 T4)\n");
		for (size_t i = 0; i < sizeof(s_cases) / sizeof(s_cases[0]); i++) {
			const size_t df = s_cases[i].deg_f;
			const size_t dg = s_cases[i].deg_g;
			gf64_t *f = (gf64_t *)malloc((df + 1) * sizeof(gf64_t));
			gf64_t *g = (gf64_t *)malloc((dg + 1) * sizeof(gf64_t));
			/* q/r sized for the worst case: deg_f+1 wide, dg wide. */
			size_t qw = (df >= dg) ? (df - dg + 1) : 1;
			size_t rw = (dg > df) ? (dg + 1) : (df + 1);
			gf64_t *q_ref = (gf64_t *)malloc(qw * sizeof(gf64_t));
			gf64_t *r_ref = (gf64_t *)malloc(rw * sizeof(gf64_t));
			gf64_t *q_scr = (gf64_t *)malloc(qw * sizeof(gf64_t));
			gf64_t *r_scr = (gf64_t *)malloc(rw * sizeof(gf64_t));
			for (size_t j = 0; j <= df; j++) {
				f[j] = splitmix64_next();
			}
			for (size_t j = 0; j <= dg; j++) { g[j] = splitmix64_next(); }
			if (dg == 0) g[0] = splitmix64_next() | 1ULL; /* keep div nonzero */

			gf64_poly_divmod(f, df, g, dg, q_ref, r_ref);

			/* Scratch path: arena sized to the worst single divmod demand
			 * (root divmod with Newton reciprocal) plus safety margin. */
			gf64_arena_t arena;
			if (gf64_arena_init(&arena, 16 * (df + dg + 4)) != 0) {
				fail("arena init");
			}
			gf64_poly_divmod_scratch(f, df, g, dg, q_scr, r_scr, &arena);
			gf64_arena_release(&arena, 0);
			gf64_arena_free(&arena);

			int mismatch = memcmp(q_ref, q_scr, qw * sizeof(gf64_t)) ||
			               memcmp(r_ref, r_scr, rw * sizeof(gf64_t));
			if (mismatch) {
				fail("scratch-vs-malloc bit-exactness");
			} else {
				char msg[128];
				snprintf(msg, sizeof(msg),
				         "scratch parity deg_f=%zu deg_g=%zu m=%zu",
				         df, dg, (df >= dg) ? df - dg + 1 : 1);
				pass(msg);
			}
			free(f); free(g); free(q_ref); free(r_ref); free(q_scr); free(r_scr);
		}
	}

	printf("\n=== Summary ===\n");
	printf("Passed: %d\n", g_passed);
	printf("Failed: %d\n", g_failed);

	return g_failed > 0 ? 1 : 0;
}
