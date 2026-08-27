/*
 * ============================================================================
 * gf64/gf64_mpe.c — Multi-point evaluation and interpolation in GF(2^64)[x]
 *
 * T8 (eval + interp) of the par3-cauchy-fft-kernel plan.
 *
 * Public functions:
 *
 *   1. gf64_poly_divmod      -- fast polynomial division via the
 *                               Newton-reciprocal trick (issue #59 A1),
 *                               O(M(m)) for quotient length m; small
 *                               quotients delegate to the schoolbook
 *                               reference gf64_poly_divmod_schoolbook.
 *
 *   2. gf64_poly_invmod      -- Newton-iteration modular inverse 1/g(x) mod
 *                               x^n. Every multiplication routes through
 *                               gf64_poly_mul's FFT dispatch.
 *
 *   3. gf64_multi_point_eval -- Bostan-Schost top-down multi-point
 *                               evaluation. Each node does two fast
 *                               divmods; total O(M(N) log N).
 *
 *   4. gf64_poly_invmod_mod  -- polynomial modular inverse 1/g(x) mod f(x)
 *                               via iterative half-extended GCD. Kept as a
 *                               standalone library primitive (pinned by
 *                               test_gf64_invmod_mod_parity.c); the
 *                               subproduct-tree inverse cache that used it
 *                               was removed in A1.
 *
 *   5. gf64_multi_point_interp -- Bostan-Schost top-down multi-point
 *                               INTERPOLATION. Given N subproduct-tree
 *                               points and N values, returns the unique
 *                               polynomial of degree < N that matches.
 *                               Derivative-based Lagrange form (issue #59
 *                               A1): P'(x_j) via one MPE of the root
 *                               derivative, z_j = y_j/P'(x_j), top-down
 *                               f_L·P_R + f_R·P_L combine — no tree-level
 *                               inverse cache needed. GATED behind the
 *                               PAR3_GF64_USE_INTERP env var (default
 *                               OFF): the function only enters the
 *                               Bostan-Schost body when the env var is
 *                               "1"/"true"/"yes"/"on".
 *
 * The two interpolation helpers co-located here for now:
 *
 *   - gf64_multi_point_interp_internal: the Bostan-Schost body (always
 *                                       called, never the dispatch entry).
 *   - gf64_interp_dispatch_reset,
 *     gf64_interp_dispatch_probe_count,
 *     gf64_interp_dispatch_reset_probe: explicit test hooks so the
 *                                       opt-in dispatch can be verified
 *                                       from outside this TU.
 *
 * None of T8 is on the engine hot path yet — this file exists to (a)
 * provide the correctness baseline that the final FFT-based variant
 * will be cross-checked against, and (b) gate the interpolation path
 * behind an opt-in so the engine remains BitIdentical until the
 * dispatch wiring lands.
 * ============================================================================
 */

#include "gf64_mpe.h"
#include "gf64_additive_fft.h"
#include "gf64_invert_ita.h"
#include "gf64_subproduct.h"

#include <assert.h>
#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _MSC_VER
/* MSVC shim: case-insensitive compare. */
#define strcasecmp _stricmp
#endif

HEDLEY_BEGIN_C_DECLS

/*
 * Scalar SSE2 reference multiplication, declared in gf64_single.c. The
 * function bodies here are deliberately written against the scalar
 * reference rather than the AVX-512 vectorized gf64_mul so that this TU
 * compiles and runs on hosts without AVX-512 (matching the parity test
 * environment below where -march=native may not advertise AVX-512).
 */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* ----------------------------------------------------------------------------
 * 1. gf64_poly_divmod_schoolbook: schoolbook long division in GF(2^64)[x]
 *
 * REFERENCE implementation. Kept public so the fast Newton-reciprocal
 * dispatcher below (and the parity test test_gf64_divmod_parity.c) can
 * cross-check bit-exactness. Also used directly for small quotients where
 * the Newton setup would dominate.
 *
 * Long-division algorithm (descending-coefficient order):
 *
 *   r <- f (deg_f + 1 coeffs)
 *   for i = deg_f down to deg_g:
 *     coeff = r[i] / g[deg_g]    (in GF(2^64) division is * inverse)
 *     q[i - deg_g] = coeff
 *     if coeff != 0:
 *       for j = 0 .. deg_g:
 *         r[i - deg_g + j] ^= coeff * g[j]
 *
 * On exit r holds the remainder with degree strictly less than deg_g,
 * and q holds the quotient with degree <= deg_f - deg_g.
 *
 * Edge cases:
 *   - deg_g == 0 and g[0] == 0: undefined -> abort()
 *   - deg_f <  deg_g:           r = f, q = 0
 * ---------------------------------------------------------------------------- */
void gf64_poly_divmod_schoolbook(
	const gf64_t *f, size_t deg_f,
	const gf64_t *g, size_t deg_g,
	gf64_t *q,    gf64_t *r
) {
	assert(f != NULL);
	assert(g != NULL);
	assert(q != NULL);
	assert(r != NULL);

	/* Zero-divisor guard. The plan's pseudocode aborts here; preserving that
	 * explicit fail-fast behavior over silently returning a junk result. */
	if (deg_g == 0 && g[0] == 0) {
		abort();
	}

	if (deg_f < deg_g) {
		/* r = f (constant-first copy) */
		for (size_t i = 0; i <= deg_f; i++) {
			r[i] = f[i];
		}
		/* Zero out the trailing slots of r (no-op for Horner, but keeps
		 * the buffer deterministic for the test harness). */
		for (size_t i = deg_f + 1; i <= deg_g - 1; i++) {
			r[i] = 0;
		}
		q[0] = 0;
		return;
	}

	/*
	 * Copy f into r (working buffer). r will be mutated in place; q is
	 * written as we go.
	 */
	for (size_t i = 0; i <= deg_f; i++) {
		r[i] = f[i];
	}

	/* Initialize q to zeros. q_size = deg_f - deg_g + 1 nonzero entries
	 * (degrees 0..deg_f - deg_g). */
	size_t q_size = deg_f - deg_g + 1;
	for (size_t i = 0; i < q_size; i++) {
		q[i] = 0;
	}

	/* Constant leading-coefficient inverse. */
	gf64_t lead_inv = gf64_invert_ita_one(g[deg_g]);

	/*
	 * Standard long division: eliminate r[i] for i = deg_f down to deg_g.
	 * The `i != (size_t)(-1)` guard protects against the underflow when
	 * i == deg_g == 0 (which would underflow the unsigned to all 1s and
	 * then re-enter the loop with a bogus high `i`). With deg_g > 0 the
	 * loop simply halts at i == deg_g.
	 */
	for (size_t i = deg_f; i >= deg_g && i != (size_t)(-1); i--) {
		gf64_t coeff = gf64_mul_reference(r[i], lead_inv);
		q[i - deg_g] = coeff;
		if (coeff != 0) {
			for (size_t j = 0; j <= deg_g; j++) {
				r[i - deg_g + j] ^= gf64_mul_reference(coeff, g[j]);
			}
		}
	}

	/*
	 * r now has degree < deg_g (entries 0..deg_g - 1 are valid; entries
	 * deg_g..deg_f may be zero but are unspecified by the contract).
	 */
}

/* ----------------------------------------------------------------------------
 * 2. gf64_poly_divmod: Newton-reciprocal fast division in GF(2^64)[x]
 *
 * Issue #59 A1: replaces the schoolbook inner loop of the multi-point
 * evaluation / interpolation tree walks. For quotients of length
 * m = deg_f - deg_g + 1 >= GF64_DIVMOD_NEWTON_MIN the quotient is computed
 * via the reverse-polynomial Newton trick (von zur Gathen & Gerhard §9.1):
 *
 *   k   = deg_f - deg_g;   m = k + 1
 *   rev_f[i] = f[deg_f - i]       (i in [0, k])     -- top k+1 coeffs, reversed
 *   rev_g[i] = g[deg_g - i]       (i in [0, deg_g]) -- all of g, reversed
 *   inv      = rev_g^{-1} mod x^m                     (Newton, gf64_poly_invmod)
 *   rev_q    = rev_f * inv mod x^m                    (gf64_poly_mul_padded)
 *   q[i]     = rev_q[k - i]
 *   r        = f ^ (g * q)   (low deg_g coefficients)
 *
 * Identity: f = g*q + r with deg r < deg g  <=>  rev_f ≡ rev_g * rev_q
 * (mod x^m), where rev_q = x^k * q(1/x). Proof sketch: reverse f = g*q + r
 * with respect to x^{deg_f}: rev_f = rev_g * rev_q + x^{deg_f - deg_g + 1} * ...
 * (the reversed remainder is divisible by x^{k+1} because deg r < deg g).
 *
 * Cost: O(M(m)) field ops (one modular inverse via log m FFT muls plus one
 * product), vs O((deg_f+1)(deg_g+1)) schoolbook. The muls route through
 * gf64_poly_mul_internal, which dispatches HQC FFT (n >= 96), Karatsuba
 * (n >= 128) or schoolbook.
 *
 * The r buffer contract is unchanged: r is a working buffer holding a copy
 * of f (deg_f + 1 slots); on return r[0..deg_g-1] is the remainder and the
 * rest is unspecified.
 * ---------------------------------------------------------------------------- */

/* Quotients shorter than this delegate to the schoolbook reference: the
 * Newton setup (three allocations + ~log m FFT muls) would dominate. 96
 * matches GF64_HQC_FFT_MIN so every mul inside the Newton path is an FFT
 * mul, never a quadratic fallback. */
#define GF64_DIVMOD_NEWTON_MIN ((size_t)96)

/* T4 (issue #59): heap-allocation instrumentation for tests. Incremented
 * at every malloc site inside the divmod/invmod implementations. */
size_t gf64_mpe_heap_alloc_count = 0;

/* ---------------------------------------------------------------------------
 * T4 arena: bump allocator over one caller-owned block. The _scratch
 * variants carve their working buffers from here instead of malloc/free
 * per call; callers take a mark before a call and release it after, so
 * peak usage is the deepest single call's demand rather than the sum.
 * ------------------------------------------------------------------------- */
int gf64_arena_init(gf64_arena_t *a, size_t words) {
	a->data = NULL;
	a->cap = 0;
	a->used = 0;
	if (words == 0) return 1;
	if (words > SIZE_MAX / sizeof(gf64_t)) return 1;
	a->data = (gf64_t *)malloc(words * sizeof(gf64_t));
	if (a->data == NULL) return 1;
	a->cap = words;
	return 0;
}

void gf64_arena_free(gf64_arena_t *a) {
	free(a->data);
	a->data = NULL;
	a->cap = 0;
	a->used = 0;
}

size_t gf64_arena_mark(const gf64_arena_t *a) {
	return a->used;
}

void gf64_arena_release(gf64_arena_t *a, size_t mark) {
	a->used = mark;
}

gf64_t *gf64_arena_push(gf64_arena_t *a, size_t words) {
	if (a->used + words > a->cap || words > SIZE_MAX - a->used) {
		abort();
	}
	gf64_t *p = a->data + a->used;
	a->used += words;
	return p;
}

void gf64_poly_divmod(
	const gf64_t *f, size_t deg_f,
	const gf64_t *g, size_t deg_g,
	gf64_t *q,    gf64_t *r
) {
	gf64_poly_divmod_scratch(f, deg_f, g, deg_g, q, r, NULL);
}

/* T4 (issue #59): arena-backed core. arena == NULL selects the legacy
 * per-call malloc path (bit-identical arithmetic either way). */
void gf64_poly_divmod_scratch(
	const gf64_t *f, size_t deg_f,
	const gf64_t *g, size_t deg_g,
	gf64_t *q,    gf64_t *r,
	gf64_arena_t *arena
) {
	assert(f != NULL);
	assert(g != NULL);
	assert(q != NULL);
	assert(r != NULL);

	if (deg_g == 0 && g[0] == 0) {
		abort();
	}

	if (deg_f < deg_g) {
		/* r = f, q = 0 — same contract as the schoolbook reference. */
		for (size_t i = 0; i <= deg_f; i++) {
			r[i] = f[i];
		}
		for (size_t i = deg_f + 1; i <= deg_g - 1; i++) {
			r[i] = 0;
		}
		q[0] = 0;
		return;
	}

	const size_t k = deg_f - deg_g;
	const size_t m = k + 1;

	if (m < GF64_DIVMOD_NEWTON_MIN) {
		gf64_poly_divmod_schoolbook(f, deg_f, g, deg_g, q, r);
		return;
	}

	size_t mark = 0;
	gf64_t *rev_f;
	gf64_t *rev_g;
	gf64_t *inv;
	gf64_t *rev_q;
	gf64_t *gq;
	if (arena != NULL) {
		mark = gf64_arena_mark(arena);
		rev_f = gf64_arena_push(arena, m);
		rev_g = gf64_arena_push(arena, deg_g + 1);
		inv   = gf64_arena_push(arena, m);
		rev_q = gf64_arena_push(arena, m);
		gq    = gf64_arena_push(arena, deg_f + 1);
	} else {
		gf64_mpe_heap_alloc_count += 5;
		rev_f = (gf64_t *)malloc(m * sizeof(gf64_t));
		rev_g = (gf64_t *)malloc((deg_g + 1) * sizeof(gf64_t));
		inv   = (gf64_t *)malloc(m * sizeof(gf64_t));
		rev_q = (gf64_t *)malloc(m * sizeof(gf64_t));
		gq    = (gf64_t *)malloc((deg_f + 1) * sizeof(gf64_t));
	}
	if (rev_f == NULL || rev_g == NULL || inv == NULL ||
	    rev_q == NULL || gq == NULL) {
		if (arena == NULL) {
			free(rev_f); free(rev_g); free(inv); free(rev_q); free(gq);
		}
		abort();
	}

	/* Reverse the top k+1 coefficients of f and all of g. */
	for (size_t i = 0; i < m; i++) {
		rev_f[i] = f[deg_f - i];
	}
	for (size_t i = 0; i <= deg_g; i++) {
		rev_g[i] = g[deg_g - i];
	}

	/* inv = rev_g^{-1} mod x^m. rev_g[0] = g[deg_g] != 0 by the divisor
	 * contract, so the inverse exists. */
	gf64_poly_invmod_scratch(rev_g, deg_g, m, inv, arena);

	/* rev_q = rev_f * inv mod x^m (low m coefficients). */
	gf64_poly_mul_padded(rev_q, rev_f, m, inv, m, m);

	/* Un-reverse the quotient. */
	for (size_t i = 0; i < m; i++) {
		q[i] = rev_q[k - i];
	}

	/* r = f ^ (g * q), keeping only the deg_g low coefficients. The
	 * region r[deg_g..deg_f] is zeroed to match the schoolbook path's
	 * end state: the half-EGCD (gf64_poly_invmod_mod) memcpys the whole
	 * r buffer as the next remainder and scans it for the true degree,
	 * so a nonzero high region would corrupt the EGCD. (The divmod
	 * contract calls this region "unspecified"; both implementations
	 * leave it zero.) memmove, not memcpy: callers may legally reuse
	 * the dividend buffer as r (r == f), and memcpy on overlapping
	 * objects is undefined behavior (cubic review c509dd2b P2). */
	memmove(r, f, (deg_f + 1) * sizeof(gf64_t));
	gf64_poly_mul(gq, g, deg_g, q, k);
	for (size_t i = 0; i < deg_g; i++) {
		r[i] ^= gq[i];
	}
	memset(r + deg_g, 0, (deg_f - deg_g + 1) * sizeof(gf64_t));

	if (arena != NULL) {
		gf64_arena_release(arena, mark);
	} else {
		free(rev_f);
		free(rev_g);
		free(inv);
		free(rev_q);
		free(gq);
	}
}

/* ----------------------------------------------------------------------------
 * 2. gf64_poly_invmod: 1/g(x) mod x^n via Newton iteration (STUB)
 *
 * The Newton iteration for modular polynomial inverse in characteristic 2:
 *
 *   r_{k+1} = g * r_k^2  (mod x^m_new)
 *
 * with m_new = min(2*m, n) and a seed r_0 = g[0]^-1 (so r_0 = 1 mod x).
 * Each doubling step squares the working precision, so after ceil(log2(n))
 * iterations we have a full n-coefficient inverse.
 *
 * STUB note: the multiplication is delegated to gf64_poly_mul, which on
 * T3's AVX-512 host uses an additive FFT internally (still O(M log M)
 * per multiplication). The TODO in the engine integration task (T9) is to
 * skip the per-iteration buffereing and call the FFT path directly.
 * ---------------------------------------------------------------------------- */
void gf64_poly_invmod(
	const gf64_t *g, size_t deg_g,
	size_t n,
	gf64_t *result
) {
	gf64_poly_invmod_scratch(g, deg_g, n, result, NULL);
}

/* T4 (issue #59): arena-backed core. arena == NULL selects the legacy
 * per-call malloc path (bit-identical arithmetic either way; only the
 * scratch provenance differs). */
void gf64_poly_invmod_scratch(
	const gf64_t *g, size_t deg_g,
	size_t n,
	gf64_t *result,
	gf64_arena_t *arena
) {
	if (n == 0) {
		return;
	}
	/* Buffer-arithmetic overflow guard (cubic review c509dd2b P1): the
	 * Newton loop materializes a full product of up to 3n - 2
	 * coefficients in `prod`, so n must satisfy 3n <= SIZE_MAX or the
	 * calloc sizes wrap and the loop writes out of bounds. Refuse (like
	 * the n == 0 no-op above) rather than wrap; no realistic caller is
	 * within an order of magnitude of this bound. This also covers the
	 * 2n-sized r_sq scratch (SIZE_MAX/3 < SIZE_MAX/2). */
	if (n > SIZE_MAX / 3) {
		return;
	}

	assert(g != NULL);
	assert(result != NULL);

	/*
	 * Scratch strategy (T4):
	 *   g_buf  -- truncated copy of g (n entries)
	 *   r_sq   -- scratch for r^2 (up to 2n - 2 coefficients)
	 *   prod   -- scratch for g * r^2
	 *
	 * prod sizing (issue #59 A1 fix): the product written is the FULL
	 * product of (deg m_new - 1) by (deg 2m - 2), i.e. m_new + 2m - 2
	 * coefficients. When n is NOT a power of 2, the final Newton step has
	 * m_new = n and m = 2^floor(log2(n-1)) >= n/2, so the worst case is
	 * n + 2n - 2 = 3n - 2 coefficients (e.g. n = 96: m = 64, m_new = 96,
	 * product length 222 > 2n = 192 — the old 2n sizing overflowed the
	 * heap, caught by ASan via the Newton-reciprocal divmod path). Only
	 * the first m_new coefficients are read back, but the full product is
	 * materialized.
	 */
	size_t mark = 0;
	gf64_t *g_buf;
	gf64_t *r_sq;
	gf64_t *prod;
	if (arena != NULL) {
		mark = gf64_arena_mark(arena);
		g_buf = gf64_arena_push(arena, n);
		r_sq  = gf64_arena_push(arena, 2 * n);
		prod  = gf64_arena_push(arena, 3 * n);
		/* Match the legacy calloc semantics exactly: the Newton loop's
		 * reads stay within freshly written ranges, but zeroing keeps the
		 * buffer contents deterministic and the parity proof trivial. */
		memset(g_buf, 0, n * sizeof(gf64_t));
		memset(r_sq, 0, 2 * n * sizeof(gf64_t));
		memset(prod, 0, 3 * n * sizeof(gf64_t));
	} else {
		gf64_mpe_heap_alloc_count += 3;
		g_buf = (gf64_t *)calloc(n, sizeof(gf64_t));
		r_sq  = (gf64_t *)calloc(2 * n, sizeof(gf64_t));
		prod  = (gf64_t *)calloc(3 * n, sizeof(gf64_t));
	}
	if (g_buf == NULL || r_sq == NULL || prod == NULL) {
		if (arena == NULL) {
			free(g_buf); free(r_sq); free(prod);
		}
		abort();
	}

	/* Truncate g (only deg_g + 1 coefficients matter, padded to n). */
	size_t g_use = (deg_g + 1 < n) ? (deg_g + 1) : n;
	for (size_t i = 0; i < g_use; i++) {
		g_buf[i] = g[i];
	}
	for (size_t i = g_use; i < n; i++) {
		g_buf[i] = 0;
	}

	/* Initialize result: r_0 = 1/g(0) = g[0]^-1, expressed mod x^1. */
	for (size_t i = 0; i < n; i++) {
		result[i] = 0;
	}
	result[0] = gf64_invert_ita_one(g[0]);

	/* Newton iteration. */
	size_t m = 1;
	while (m < n) {
		size_t m_new = (2 * m < n) ? (2 * m) : n;

		/*
		 * r_sq = result^2 (schoolbook path via gf64_poly_mul). The
		 * polynomial has at most (m - 1) coefficients that are nonzero,
		 * so its degree is at most (m - 1). The result has at most
		 * (2*m - 2) nonzero coefficients.
		 */
		gf64_poly_mul(r_sq, result, m - 1, result, m - 1);

		/*
		 * prod = g_buf * r_sq truncated to m_new entries. g_buf has
		 * length n (but is logically truncated to m_new - 1 here for
		 * the next-step precision; using m_new - 1 keeps the result
		 * polynomial length bounded and is the standard
		 * "Newton-truncated multiplication" idiom).
		 */
		gf64_poly_mul(prod, g_buf, m_new - 1, r_sq, 2 * m - 2);

		/* Copy the first m_new coefficients into result; pad the
		 * rest with zeros so the buffer remains deterministic. */
		for (size_t i = 0; i < m_new; i++) {
			result[i] = prod[i];
		}
		for (size_t i = m_new; i < n; i++) {
			result[i] = 0;
		}

		m = m_new;
	}

	if (arena != NULL) {
		gf64_arena_release(arena, mark);
	} else {
		free(g_buf);
		free(r_sq);
		free(prod);
	}
}

/* ----------------------------------------------------------------------------
 * 3. gf64_multi_point_eval: Bostan-Schost top-down multi-point evaluation
 *
 * Reads the N leaf entries from the last level of `tree`, evaluates f at
 * each one, and stores the results in `out`.
 *
 * RECURSION (the plan's TODO(replace-with-bostan-schost) contract):
 *
 *   eval_rem(f, deg_f, node):
 *     if node is leaf i:
 *       out[i] = horner(f, deg_f, x_i)
 *       return
 *     let P_L, P_R be the two child polynomials
 *     r_L = divmod(f, P_R)            # remainder of deg < deg(P_R)
 *     r_R = divmod(f, P_L)            # remainder of deg < deg(P_L)
 *     eval_rem(r_L, deg(r_L)-1, left)
 *     eval_rem(r_R, deg(r_R)-1, right)
 *
 * Correctness: f ≡ r_L (mod P_R), and P_R vanishes on the right subtree's
 * leaves (each leaf is a factor of P_R), so f and r_L agree on those
 * leaves. Total cost per level is O(deg(f) · (#nodes)), i.e. O(D·N/2^ℓ)
 * at level ℓ; summed over levels this is O((D + N) log²(D + N)) with the
 * schoolbook divmod — the contract the Fenger plan promises, replacing the
 * O(N · D) naive Horner the pipeline was actually using.
 * ---------------------------------------------------------------------------- */

static gf64_t horner_eval_poly(const gf64_t *c, size_t deg, gf64_t r) {
	/*
	 * Edge case: deg == (size_t)(-1) means the polynomial is the zero
	 * polynomial; Horner evaluates it to 0 trivially. This lets
	 * multi_point_eval pass a constant-zero remainder into recursive
	 * calls without a separate guard.
	 */
	if (deg == (size_t)(-1)) {
		return 0;
	}

	gf64_t acc = c[deg];
	for (size_t i = deg; i > 0; i--) {
		acc = gf64_mul_reference(acc, r) ^ c[i - 1];
	}
	return acc;
}

/* Top-down Bostan-Schost evaluation. `f`/`deg_f` are the current
 * remainder; `lev`/`node_idx` identify the current tree node; `out`
 * receives the leaf values. Scratch layout (mirrors gf64_mpi_recurse):
 * this frame uses the first 4·slot words for r_L/q/r_R/q2; the
 * recursion passes child_scratch = scratch + 4·slot so each depth has
 * its own region (the parent's r_L/r_R must survive the left subtree
 * recursion before the right subtree reads them). */
static void gf64_mp_eval_recurse(
	const gf64_t *f, size_t deg_f,
	const SubproductTree *tree,
	size_t lev, size_t node_idx,
	gf64_t *out,
	gf64_t *HEDLEY_RESTRICT scratch,
	gf64_arena_t *arena
) {
	const size_t num_levels = tree->num_levels;

	if (lev + 1 == num_levels) {
		/* Leaf: evaluate via Horner. */
		gf64_t *leaves = tree->level_data[lev];
		gf64_t xj = leaves[2 * node_idx]; /* leaf [x_j, 1] */
		out[node_idx] = horner_eval_poly(f, deg_f, xj);
		return;
	}

	/* This node covers N_at_lev leaves; its children each have degree
	 * child_deg = N_at_lev / 2. The remainder r has degree < child_deg,
	 * but gf64_poly_divmod copies f into r first — r must be
	 * deg_f + 1 wide, and deg_f < N_at_lev (interpolation degree bound:
	 * a degree-<N poly over N points), so N_at_lev-wide slots bound both
	 * r and q (= deg_f - child_deg + 1 <= N_at_lev). Same convention as
	 * gf64_mpi_recurse. q must NOT alias r (divmod mutates r in place
	 * while writing q — the interp separates them via distinct slots). */
	const size_t N_at_lev = tree->num_points >> lev;
	const size_t slot = N_at_lev;
	const size_t child_deg = tree->level_degs[lev + 1];
	const size_t child_stride = child_deg + 1;

	gf64_t *children = tree->level_data[lev + 1];
	const gf64_t *P_L = children + (2 * node_idx)     * child_stride;
	const gf64_t *P_R = children + (2 * node_idx + 1) * child_stride;

	/* r_L = f mod P_L (left subtree), r_R = f mod P_R (right). The two
	 * divisions share this frame's scratch; the recursion below uses a
	 * fresh region so the parent's r_L/r_R survive. */
	gf64_t *r_L = scratch;
	gf64_t *q   = scratch + slot;
	gf64_t *r_R = scratch + 2 * slot;
	gf64_t *q2  = scratch + 3 * slot;
	gf64_t *child_scratch = scratch + 4 * slot;

	gf64_poly_divmod_scratch(f, deg_f, P_L, child_deg, q, r_L, arena);
	gf64_poly_divmod_scratch(f, deg_f, P_R, child_deg, q2, r_R, arena);

	/* True remainder degree: the divmod contract leaves r[deg_g..deg_f]
	 * unspecified, so scan for the actual highest set coefficient
	 * (deg_f < child_deg => r == f, degree deg_f). */
	size_t rem_deg = 0;
	for (size_t i = child_deg; i > 0; i--) {
		if (r_L[i - 1] != 0) { rem_deg = i - 1; break; }
	}
	if (deg_f < child_deg && rem_deg > deg_f) rem_deg = deg_f;

	gf64_mp_eval_recurse(r_L, rem_deg, tree, lev + 1, 2 * node_idx,     out, child_scratch, arena);

	rem_deg = 0;
	for (size_t i = child_deg; i > 0; i--) {
		if (r_R[i - 1] != 0) { rem_deg = i - 1; break; }
	}
	if (deg_f < child_deg && rem_deg > deg_f) rem_deg = deg_f;

	gf64_mp_eval_recurse(r_R, rem_deg, tree, lev + 1, 2 * node_idx + 1, out, child_scratch, arena);
}

void gf64_multi_point_eval(
	const gf64_t *f, size_t deg_f,
	const SubproductTree *tree,
	gf64_t *out
) {
	if (tree == NULL || tree->num_points == 0 || f == NULL) {
		return;
	}

	size_t N = tree->num_points;

	/* The Bostan-Schost tree walk requires deg_f < N (the interpolation
	 * degree bound: a degree-<N poly over N points). The Fenger pipeline
	 * can evaluate a degree-(N_in - 1) polynomial at R recovery points
	 * with R < N_in, so fall back to Horner when the bound is violated
	 * (e.g. the prepare step's V(y_r) and the per-word p_w eval). */
	if (N == 1 || deg_f >= N) {
		gf64_t *last_level = tree->level_data[tree->num_levels - 1];
		for (size_t j = 0; j < N; j++) {
			gf64_t xj = last_level[2 * j]; /* leaf j is [x_j, 1] */
			out[j] = horner_eval_poly(f, deg_f, xj);
		}
		return;
	}

	/* Scratch: 4 slots of N_at_lev gf64_t per frame (r_L/q/r_R/q2);
	 * each recursion depth consumes 4·N_at_lev and N_at_lev halves per
	 * level, so the total is 4N·(1 + 1/2 + 1/4 + …) ≈ 8N — the same
	 * worst-case bound as gf64_mpi_recurse's interp scratch.
	 *
	 * T4 (issue #59): the divmod working buffers come from a bump arena
	 * sized to the worst single-call demand (root divmod ≈ 6N words with
	 * its Newton reciprocal) rather than from O(N log N) per-node heap
	 * allocations. */
	gf64_t *scratch = (gf64_t *)malloc(8 * N * sizeof(gf64_t));
	if (scratch == NULL) {
		abort();
	}
	gf64_arena_t arena;
	if (gf64_arena_init(&arena, 8 * N) != 0) {
		abort();
	}

	gf64_mp_eval_recurse(f, deg_f, tree, 0, 0, out, scratch, &arena);

	gf64_arena_free(&arena);
	free(scratch);
}

/* ----------------------------------------------------------------------------
 * 4. gf64_poly_invmod_mod: polynomial modular inverse (1/g mod f via EGCD)
 *
 * The Newton-iteration gf64_poly_invmod from §2 above computes 1/g(x)
 * in the formal power series ring GF(2^k)[[x]] truncated to x^n — i.e.
 * an expansion that satisfies g(x)·h(x) ≡ 1 (mod x^n). That is NOT the
 * same as the polynomial modular inverse 1/g(x) (mod f(x)) demanded by
 * the Bostan-Schost interpolation (T8b), which requires
 *
 *     g(x)·u(x) ≡ 1 (mod f(x)),   deg u < deg f.
 *
 * This implementation is the classical iterative half-extended GCD.
 *
 * Inputs:
 *     g:  polynomial coefficients (constant-first), deg = deg_g
 *     f:  polynomial coefficients (constant-first), deg = deg_f
 * Assumptions: f and g are coprime, f's leading coefficient is nonzero
 * (required by gf64_poly_divmod). The caller is responsible for sizing.
 * Output:
 *     inv_out: at least deg_f coefficients; on success the polynomial
 *     of degree < deg_f satisfying g·u ≡ 1 (mod f). Returns 0 on
 *     success, -1 on failure (non-coprime inputs).
 *
 * Cost: schoolbook O((deg_g+1)·(deg_f+1)) field ops per EGCD step;
 * total per-inverse O((deg_g)² + (deg_f)²).
 *
 * At tree-build time, deg_g == deg_f == deg_child here, so each inverse
 * costs O(deg_child²). Tree-build amortizes to O(N²) total cached inverses.
 * The asymptotic upgrade to O(N log² N) via the Newton / reverse-polynomial
 * trick is deferred (Phase A follow-up).
 * ---------------------------------------------------------------------------- */
int gf64_poly_invmod_mod(
	const gf64_t *g, size_t deg_g,
	const gf64_t *f, size_t deg_f,
	gf64_t *inv_out
) {
	if (deg_f == 0) {
		/* f is a scalar; the modular inverse u is a scalar with
		 * g·u = 1 + k·f, but reducing mod the constant f in the
		 * scalar sense leaves g·u ≡ 1, so u = 1/g[0]. */
		inv_out[0] = gf64_invert_ita_one(g[0]);
		return 0;
	}

	/*
	 * Half-extended Euclidean algorithm with Bezout coefficients.
	 *
	 * Invariant: r_k = s_k·g + t_k·f.
	 *
	 * Initial: (r_0, s_0, t_0) = (g, 1, 0)
	 *          (r_1, s_1, t_1) = (f, 0, 1)
	 *
	 * Update: q_k, _ = divmod(r_{k-1}, r_k)
	 *          r_{k+1} = r_{k-1} - q_k·r_k
	 *          s_{k+1} = s_{k-1} - q_k·s_k
	 *
	 * (We don't need t_k since the answer is in s_k; t_k is the
	 * corresponding Bezout coefficient for f and is not used.)
	 *
	 * Stop when r_curr is the zero polynomial; then r_prev is a
	 * constant c with s_prev·g ≡ c (mod f), so u = s_prev / c is
	 * the desired polynomial modular inverse, of degree < deg_f.
	 */

	const size_t r_buf_size  = (deg_f > deg_g ? deg_f : deg_g) + 1;
	const size_t st_buf_size = deg_f + deg_g + 2;
	const size_t q_buf_size  = (r_buf_size > st_buf_size) ? r_buf_size
	                                                       : st_buf_size;

	gf64_t *r_prev = (gf64_t *)calloc(r_buf_size,  sizeof(gf64_t));
	gf64_t *r_curr = (gf64_t *)calloc(r_buf_size,  sizeof(gf64_t));
	gf64_t *r_next = (gf64_t *)calloc(r_buf_size,  sizeof(gf64_t));
	gf64_t *s_prev = (gf64_t *)calloc(st_buf_size, sizeof(gf64_t));
	gf64_t *s_curr = (gf64_t *)calloc(st_buf_size, sizeof(gf64_t));
	gf64_t *s_next = (gf64_t *)calloc(st_buf_size, sizeof(gf64_t));
	gf64_t *q_buf  = (gf64_t *)calloc(q_buf_size,  sizeof(gf64_t));
	gf64_t *qr_buf = (gf64_t *)calloc(r_buf_size,  sizeof(gf64_t));
	gf64_t *tmp    = (gf64_t *)calloc(st_buf_size, sizeof(gf64_t));

	if (!r_prev || !r_curr || !r_next || !s_prev || !s_curr ||
	    !s_next || !q_buf || !qr_buf || !tmp) {
		free(r_prev); free(r_curr); free(r_next);
		free(s_prev); free(s_curr); free(s_next);
		free(q_buf); free(qr_buf); free(tmp);
		return -1;
	}

	/* r_0 = g, s_0 = 1. */
	memcpy(r_prev, g, (deg_g + 1) * sizeof(gf64_t));
	memset(s_prev, 0, st_buf_size * sizeof(gf64_t));
	s_prev[0] = 1ULL;

	/* r_1 = f, s_1 = 0. */
	memcpy(r_curr, f, (deg_f + 1) * sizeof(gf64_t));
	memset(s_curr, 0, st_buf_size * sizeof(gf64_t));

	/*
	 * Use bool flags for zero-poly state so we don't conflate "degree
	 * 0" (constant) with "the zero polynomial" (size_t max sentinel)
	 * in degree arithmetic.
	 */
	int r_curr_is_zero = 0; /* r_curr starts non-zero (modulus f = input) */
	int s_curr_is_zero = 1; /* s_curr starts as the zero poly (s_1 = 0) */

	size_t deg_r_prev = deg_g;
	size_t deg_r_curr = deg_f;
	size_t deg_s_prev = 0;
	size_t deg_s_curr = 0; /* unused while s_curr_is_zero */

	(void)deg_s_curr; /* silenced warning */

	while (!r_curr_is_zero) {
		gf64_poly_divmod(r_prev, deg_r_prev, r_curr, deg_r_curr,
		                 q_buf, qr_buf);
		size_t deg_q = (deg_r_prev >= deg_r_curr) ? (deg_r_prev - deg_r_curr) : 0;

		memcpy(r_next, qr_buf, r_buf_size * sizeof(gf64_t));
		size_t deg_r_next = deg_r_curr;
		while (deg_r_next > 0 && r_next[deg_r_next] == 0) {
			deg_r_next--;
		}
		int r_next_is_zero = (deg_r_next == 0 && r_next[0] == 0);

		size_t deg_s_next;
		int s_next_is_zero;
		if (s_curr_is_zero) {
			memcpy(s_next, s_prev, st_buf_size * sizeof(gf64_t));
			deg_s_next = deg_s_prev;
			s_next_is_zero = (deg_s_next == 0 && s_next[0] == 0);
		} else {
			memset(tmp, 0, st_buf_size * sizeof(gf64_t));
			for (size_t i = 0; i <= deg_q; i++) {
				if (q_buf[i] == 0) continue;
				for (size_t j = 0; j <= deg_s_curr; j++) {
					tmp[i + j] ^= gf64_mul_reference(q_buf[i], s_curr[j]);
				}
			}
			memcpy(s_next, s_prev, st_buf_size * sizeof(gf64_t));
			for (size_t i = 0; i < st_buf_size; i++) {
				s_next[i] ^= tmp[i];
			}
			deg_s_next = (deg_s_prev > deg_q + deg_s_curr)
			             ? deg_s_prev : (deg_q + deg_s_curr);
			while (deg_s_next > 0 && s_next[deg_s_next] == 0) {
				deg_s_next--;
			}
			s_next_is_zero = (deg_s_next == 0 && s_next[0] == 0);
		}

		/* Rotate the (r, s) windows. */
		gf64_t *t;
		t = r_prev; r_prev = r_curr; r_curr = r_next; r_next = t;
		t = s_prev; s_prev = s_curr; s_curr = s_next; s_next = t;

		deg_r_prev = deg_r_curr;
		deg_r_curr = deg_r_next;
		deg_s_prev = deg_s_curr;
		deg_s_curr = deg_s_next;
		r_curr_is_zero = r_next_is_zero;
		s_curr_is_zero = s_next_is_zero;
	}

	int rc;
	if (deg_s_prev == (size_t)(-1)) {
		/* s_prev is the zero poly: g and f had a non-trivial GCD,
		 * so a modular inverse of g mod f does not exist. */
		rc = -1;
	} else {
		gf64_t c     = r_prev[0];
		gf64_t c_inv = gf64_invert_ita_one(c);
		for (size_t i = 0; i <= deg_s_prev; i++) {
			inv_out[i] = gf64_mul_reference(s_prev[i], c_inv);
		}
		for (size_t i = deg_s_prev + 1; i < deg_f; i++) {
			inv_out[i] = 0;
		}
		rc = 0;
	}

	free(r_prev); free(r_curr); free(r_next);
	free(s_prev); free(s_curr); free(s_next);
	free(q_buf); free(qr_buf); free(tmp);
	return rc;
}

/* ----------------------------------------------------------------------------
 * 5. gf64_multi_point_interp: Bostan-Schost top-down INTERPOLATION
 *
 * DUAL of gf64_multi_point_eval. Recursion contract:
 *
 *   At an internal node (lev, node_idx) covering leaves at tree offsets
 *   [out_offset, out_offset + (N/2^lev)), we have already (via the two
 *   recursive calls into the children) computed f_L and f_R, the unique
 *   polynomials of degree < (N/2^(lev+1)) with f_L(x_j) = values[j] for
 *   every j on the left side and analogously for f_R on the right.
 *
 *   We want f_parent of degree < (N/2^lev) such that
 *       f_parent ≡ f_L    (mod P_left)
 *       f_parent ≡ f_R    (mod P_right)
 *   in GF(2^k)[x], where P_left and P_right are the children's root
 *   polynomials (left and right subproducts). Since they are coprime, the
 *   CRT gives a unique f_parent with
 *
 *       f_parent = f_L + P_left · k       (mod P_parent = P_left · P_right)
 *
 *   where k satisfies  P_left · k ≡ f_R - f_L = f_R + f_L  (mod P_right).
 *
 *   Substituting the cached modular inverse inv = P_left^(-1) mod P_right
 *   (one of the polynomial inverses precomputed at tree-build time and
 *   stored in tree->inv_mod_data; see gf64_subproduct.h):
 *
 *       k = (f_R + f_L) · inv   mod P_right.
 *
 *   Then f_parent = f_L + P_left · k, truncated to < (N/2^lev).
 *
 *   CORRECTNESS:
 *     f_parent ≡ f_L (mod P_left)         — by construction
 *     f_parent = f_L + P_left · k
 *             ≡ f_L + P_left · k (mod P_right)
 *             ≡ f_L + P_left · (f_R + f_L) · P_left^(-1)
 *             ≡ f_L + (f_R + f_L)
 *             ≡ f_R    (mod P_right)      ✓
 *
 *   DEGREE:
 *     deg f_L  < N/2^(lev+1)    = f_size
 *     deg P_left = f_size
 *     deg k    < f_size
 *     => deg f_parent ≤ deg f_L + deg(P_left · k) ≤ (f_size - 1) + 2*(f_size) - 1
 *                  = 2*f_size - 1 = (N/2^lev) - 1                ✓
 *
 *   SCRATCH:
 *     4 slots of size N/2^lev (parent frame), children's frames start at
 *     scratch + 4·(N/2^lev). Each child uses 4 slots of size N/2^(lev+1),
 *     so a frame at depth ℓ uses 4·(N/2^ℓ); total 4·(N + N/2 + N/4 + …)
 *     = 8·N coefficients are upper-bounded by a single 8·N allocation.
 *
 *   CONSUMER:
 *     The per-node polynomial multiplications consume gf64_poly_mul_padded
 *     (the T3/T4 padded-output variant — pre-allocated output, len_a/len_b
 *     separated from degree-based API). The full FFT-based flavor of the
 *     consumer lands via Todo 4's additive-FFT dispatch; this entry is
 *     already opt-in-only (see the dispatch gate below), so the O(n²)
 *     schoolbook primitive inside gf64_poly_mul_padded is acceptable
 *     today and free-of-charge tomorrow.
 *
 *   DISPATCH GATE:
 *     The PUBLIC entry gf64_multi_point_interp checks
 *     PAR3_GF64_USE_INTERP (setenv / unsetenv / "1"/"true"/"yes"/"on"
 *     vs unset/"0"/"false"/"no"/"off"). When the env var is OFF (the
 *     default), the function returns immediately WITHOUT entering the
 *     Bostan-Schost body, does NOT advance the dispatch probe counter,
 *     and leaves `out` undefined — production callers must NOT invoke
 *     this function without the opt-in. The dispatch probe counter
 *     (read via gf64_interp_dispatch_probe_count) is incremented once
 *     per invocation that actually enters the body. Tests use
 *     gf64_interp_dispatch_reset (clears the env-var cache so a fresh
 *     setenv/unsetenv takes effect on the next call) and
 *     gf64_interp_dispatch_reset_probe (zeroes the probe counter).
 * ---------------------------------------------------------------------------- */

/* ---- dispatch gate: probe counter + env-var cache + opt-in check ----
 *
 * Default OFF. Reading convention (case-insensitive):
 *   unset, "", "0", "false", "no", "off"           → OFF
 *   "1", "true", "yes", "on"                        → ON
 *   anything else                                   → OFF (fail-safe default)
 *
 * The env-var decision is cached after the FIRST call to keep the hot
 * path cheap. gf64_interp_dispatch_reset clears the cache so tests can
 * toggle the env var mid-process without restarting.
 */
static int gf64_interp_env_cached = -1; /* -1 = not yet read; 0/1 = decision */
static int gf64_interp_probe_count = 0; /* number of calls that entered the BS body */

static int gf64_interp_read_env(void) {
	const char *e = getenv("PAR3_GF64_USE_INTERP");
	if (e == NULL || e[0] == '\0') return 0;
	if (e[0] == '1') return 1;
	if (strcasecmp(e, "true") == 0) return 1;
	if (strcasecmp(e, "yes")  == 0) return 1;
	if (strcasecmp(e, "on")   == 0) return 1;
	if (e[0] == '0') return 0;
	if (strcasecmp(e, "false") == 0) return 0;
	if (strcasecmp(e, "no")    == 0) return 0;
	if (strcasecmp(e, "off")   == 0) return 0;
	return 0;
}

static int gf64_interp_enabled(void) {
	if (gf64_interp_env_cached < 0) {
		gf64_interp_env_cached = gf64_interp_read_env();
	}
	return gf64_interp_env_cached;
}

void gf64_interp_dispatch_reset(void) {
	/* Force the next call to gf64_interp_enabled to re-read getenv. */
	gf64_interp_env_cached = -1;
}

int gf64_interp_dispatch_probe_count(void) {
	return gf64_interp_probe_count;
}

void gf64_interp_dispatch_reset_probe(void) {
	gf64_interp_probe_count = 0;
}

/* ---- top-down recursive interpolation helper (derivative form) ----
 *
 * Issue #59 A1: the previous Bostan-Schost CRT combine required the
 * per-pair modular inverses P_left^{-1} mod P_right cached at tree-build
 * time (schoolbook half-EGCD — O(N²) total). That cache is gone; the
 * combine below uses the derivative-based Lagrange form instead:
 *
 *   f(x) = Σ_j z_j · P(x)/(x - x_j),   z_j = y_j / P'(x_j)
 *
 * where P = ∏(x - x_k) is the root subproduct and P' its formal
 * derivative. In characteristic 2 the derivative of c_i·x^i is c_i·x^{i-1}
 * when i is odd and 0 when i is even, so P' is computed in O(N) XORs.
 * The z_j are computed with ONE multi-point evaluation of P' (the fast
 * divmod walk) plus N scalar field inversions. The top-down combine then
 * only needs polynomial multiplications:
 *
 *   f_parent = f_L · P_R + f_R · P_L,   deg < N_at_lev
 *
 * Correctness: by induction f_L = Σ_{j∈L} z_j·P_L/(x−x_j) on the left leaf
 * set; multiplying by P_R telescopes to Σ_{j∈L} z_j·P_parent/(x−x_j), and
 * the right child contributes the L↔R-symmetric half. P'(x_j) =
 * ∏_{k≠j}(x_j + x_k) ≠ 0 for distinct points (guaranteed by the Cauchy
 * point-placement contract; the code aborts on 0 defensively).
 *
 * Cost: O(N) derivative + O(M(N) log N) MPE + N scalar inversions +
 * O(M(N) log N) combine muls — subquadratic end to end, matching the
 * tree build. The interpolation output polynomial of degree < N is
 * UNIQUE, so the result is bit-identical to the removed CRT form.
 */
static void gf64_mpi_recurse_deriv(
	const gf64_t *z,
	const SubproductTree *tree,
	size_t lev, size_t node_idx, size_t out_offset,
	gf64_t *out_poly,
	gf64_t *HEDLEY_RESTRICT scratch
) {
	const size_t num_levels = tree->num_levels;
	const size_t N_at_lev   = tree->num_points >> lev; /* leaves this node covers */
	const size_t f_size     = N_at_lev >> 1;            /* child poly size    */

	if (lev + 1 == num_levels) {
		/* Leaf: f is the constant z[out_offset]. */
		out_poly[0] = z[out_offset];
		return;
	}

	/*
	 * Four slots of size N_at_lev. slot0/slot1 hold the children's
	 * results fL/fR (written by the recursion); slot2/slot3 hold the
	 * two products P_R·fL and P_L·fR. The products must not alias their
	 * inputs, so they live in their own slots.
	 */
	gf64_t *slot0         = scratch;
	gf64_t *slot1         = scratch + N_at_lev;
	gf64_t *slot2         = scratch + 2 * N_at_lev;
	gf64_t *slot3         = scratch + 3 * N_at_lev;
	gf64_t *child_scratch = scratch + 4 * N_at_lev;

	gf64_mpi_recurse_deriv(z, tree, lev + 1,
	                       /* node_idx */ 2 * node_idx,
	                       out_offset,
	                       slot0, child_scratch);
	gf64_mpi_recurse_deriv(z, tree, lev + 1,
	                       /* node_idx */ 2 * node_idx + 1,
	                       out_offset + f_size,
	                       slot1, child_scratch);

	/*
	 * Sibling polynomials. P_left/P_right each have degree f_size
	 * (f_size + 1 stored coefficients).
	 */
	const gf64_t *P_left  = tree->level_data[lev + 1]
	                        + (2 * node_idx)     * (f_size + 1);
	const gf64_t *P_right = tree->level_data[lev + 1]
	                        + (2 * node_idx + 1) * (f_size + 1);

	/*
	 * f_parent = fL·P_R + fR·P_L, truncated to N_at_lev coefficients.
	 * deg(fL) < f_size and deg(P_R) = f_size, so deg(fL·P_R) <=
	 * 2·f_size - 1 = N_at_lev - 1 — the truncation is exact, no
	 * information is lost. P_left is passed with length f_size + 1
	 * (its x^f_size term is real).
	 */
	gf64_poly_mul_padded(slot2, P_right, f_size + 1, slot0, f_size, N_at_lev);
	gf64_poly_mul_padded(slot3, P_left,  f_size + 1, slot1, f_size, N_at_lev);

	for (size_t i = 0; i < N_at_lev; i++) {
		out_poly[i] = slot2[i] ^ slot3[i];
	}
}

/*
 * Ungated Bostan-Schost body (derivative form). Exposed (non-static) so
 * pipeline consumers with a hard dependency on interpolation — currently
 * the Fenger Toeplitz path (gf64_fenger.c) — can call it directly without
 * tripping the PAR3_GF64_USE_INTERP opt-in gate. The public
 * gf64_multi_point_interp wrapper still enforces the gate for everyone
 * else.
 *
 * Algorithm (issue #59 A1):
 *   1. P' = derivative of the root subproduct (O(N) XORs; char-2 rule:
 *      keep coefficient i+1 of P only when i+1 is odd).
 *   2. d_j = P'(x_j) for all j via gf64_multi_point_eval (fast divmod
 *      walk, O(M(N) log N)).
 *   3. z_j = values[j] / d_j (N scalar field inversions).
 *   4. Top-down combine f_parent = f_L·P_R + f_R·P_L (2 FFT muls per
 *      node, O(M(N) log N) total).
 *
 * Steps 1-3 are TREE-ONLY (independent of `values`): their result is
 * exactly the barycentric weights 1/P'(x_j). Callers that already hold
 * those weights (the Fenger prepare phase computes them via
 * gf64_barycentric_weights) should use gf64_multi_point_interp_weights
 * instead — it skips the MPE + N inversions entirely and is the
 * per-word hot path of the Fenger pipeline.
 *
 * Scratch: P' (N), d (N), z (N, reuses P''s buffer), recursion scratch
 * (<= 8N) => <= 11N gf64_t words transiently.
 */
void gf64_multi_point_interp_internal(
	const SubproductTree *tree,
	const gf64_t *values,
	gf64_t *out
) {
	if (tree == NULL || values == NULL || out == NULL) {
		/* NULL-safe direct-call contract (cubic review f44ead49 P2):
		 * same guarantee as gf64_multi_point_eval — return immediately
		 * without writing. */
		return;
	}
	const size_t N = tree->num_points;

	if (N == 0) {
		return;
	}
	if (N == 1) {
		/* P = x + x_0, P' = 1: the unique interpolant of one point is
		 * the constant values[0]. */
		out[0] = values[0];
		return;
	}

	/* 1. Derivative of the root (degree N, so N coefficients). */
	const gf64_t *P = tree->level_data[0];
	gf64_t *z = (gf64_t *)malloc(N * sizeof(gf64_t));
	gf64_t *d = (gf64_t *)malloc(N * sizeof(gf64_t));
	if (z == NULL || d == NULL) {
		free(z); free(d);
		abort();
	}
	for (size_t k = 0; k < N; k++) {
		const size_t i = k + 1;          /* P coefficient index */
		z[k] = (i & 1) ? P[i] : 0;       /* char-2 derivative */
	}

	/* 2. d_j = P'(x_j). deg(P') <= N - 1 < N, so the Bostan-Schost walk
	 * (not the Horner fallback) is taken. */
	gf64_multi_point_eval(z, N - 1, tree, d);

	/* 3. weights_j = 1 / d_j, then z_j = values[j] * weights_j. Distinct
	 * points => P'(x_j) != 0; abort loudly on violation (duplicate point
	 * in the tree's point set). */
	for (size_t j = 0; j < N; j++) {
		if (d[j] == 0) {
			abort();
		}
		z[j] = gf64_mul_reference(values[j], gf64_invert_ita_one(d[j]));
	}

	gf64_multi_point_interp_weights(tree, z, NULL, out);

	free(z);
	free(d);
}

/*
 * Weights-aware Bostan-Schost interpolation body (issue #59 A1).
 *
 * Same derivative-form combine as gf64_multi_point_interp_internal, but
 * takes the barycentric weights w_j = 1/P'(x_j) as an explicit input
 * instead of computing them (steps 1-3 above): z_j = values[j] * w_j is
 * a single mul per point, so the per-call cost is just the top-down
 * combine — O(M(N) log N) with NO MPE walk and NO scalar inversions.
 *
 * This is the per-word hot path of the Fenger pipeline: the prepare
 * phase computes the weights once per tree (amortized across all B
 * words) and each word's interpolation reuses them.
 *
 * @param weights  N precomputed barycentric weights w_j = 1/P'(x_j), or
 *                 NULL to fall back to computing them internally (the
 *                 interp_internal entry passes its already-computed
 *                 z-array and NULL here — see below).
 *
 * NOTE ON THE NULL CONVENTION: when `weights` is NULL, `values` is
 * interpreted as the PRE-SCALED z-array (z_j = values[j]/P'(x_j)) and
 * the combine runs directly. gf64_multi_point_interp_internal uses this
 * to avoid a second allocation; external callers always pass real
 * weights.
 */
void gf64_multi_point_interp_weights(
	const SubproductTree *tree,
	const gf64_t *values,
	const gf64_t *weights,
	gf64_t *out
) {
	if (tree == NULL || values == NULL || out == NULL) {
		/* NULL-safe direct-call contract (cubic review f44ead49 P2):
		 * same guarantee as gf64_multi_point_eval — return immediately
		 * without writing. */
		return;
	}
	const size_t N = tree->num_points;

	if (N == 0) {
		return;
	}
	if (N == 1) {
		/* P = x + x_0, P' = 1: the unique interpolant of one point is
		 * the constant values[0] (times the weight, which is always 1
		 * here — kept explicit for the weights entry). */
		out[0] = (weights != NULL)
		         ? gf64_mul_reference(values[0], weights[0])
		         : values[0];
		return;
	}

	gf64_t *z;
	if (weights != NULL) {
		z = (gf64_t *)malloc(N * sizeof(gf64_t));
		if (z == NULL) abort();
		for (size_t j = 0; j < N; j++) {
			z[j] = gf64_mul_reference(values[j], weights[j]);
		}
	} else {
		/* Pre-scaled z-array (internal convention). */
		z = (gf64_t *)values;
	}

	/* Top-down combine. Worst-case scratch at the root: 4 slots of N,
	 * plus the children's scratch (4·N/2 + 4·N/4 + … = 4·N), giving
	 * <= 8·N total. */
	const size_t scratch_words = 8 * N;
	gf64_t *scratch = (gf64_t *)malloc(scratch_words * sizeof(gf64_t));
	if (scratch == NULL) {
		if (weights != NULL) free(z);
		abort();
	}

	gf64_mpi_recurse_deriv(z, tree,
	                       /* lev */ 0, /* node_idx */ 0, /* out_offset */ 0,
	                       out, scratch);

	if (weights != NULL) free(z);
	free(scratch);
}

/*
 * Public entry: opt-in dispatch gate.
 *
 *   - NULL tree / empty tree / NULL values:  return immediately (matches
 *     the gf64_multi_point_eval contract; "out" is left untouched so
 *     callers can pre-fill it with sentinels and detect no-ops).
 *
 *   - PAR3_GF64_USE_INTERP unset / 0 / off:  the function returns
 *     immediately WITHOUT entering the Bostan-Schost body and WITHOUT
 *     bumping the dispatch probe counter. Production dispatch is
 *     untouched in this state — no engine wiring calls this function.
 *
 *   - PAR3_GF64_USE_INTERP=1 / true / yes / on:  enters the Bostan-Schost
 *     body, bumps the dispatch probe counter once, and writes `out`.
 */
void gf64_multi_point_interp(
	const SubproductTree *tree,
	const gf64_t *values,
	gf64_t *out
) {
	if (tree == NULL || tree->num_points == 0 || values == NULL) {
		return;
	}

	if (!gf64_interp_enabled()) {
		/* Default OFF: this is the bit-exact contract — when the env
		 * var is not set, the MPE/CRT body is NOT entered and the
		 * probe counter is NOT advanced. Tests verify both halves. */
		return;
	}

	gf64_interp_probe_count++;
	gf64_multi_point_interp_internal(tree, values, out);
}

HEDLEY_END_C_DECLS
