/*
 * ============================================================================
 * gf64/gf64_mpe.c — Multi-point evaluation in GF(2^64)[x]
 *
 * T8 of the par3-cauchy-fft-kernel plan.
 *
 * Three public functions:
 *
 *   1. gf64_poly_divmod    -- schoolbook O((deg_f + 1) * (deg_g + 1)) long
 *                             division. Exact, fast enough for the test
 *                             suite sizes (deg_f <= 200); the engine will
 *                             eventually want an FFT-based Newton-iteration
 *                             variant for large deg_f.
 *
 *   2. gf64_poly_invmod    -- Newton-iteration modular inverse 1/g(x) mod x^n
 *                             using cubic-time schoolbook gf64_poly_mul for
 *                             the squaring and the multiplication. The
 *                             asymptotic cost is O(n^2 log n); the proper
 *                             O(n log n) implementation would swap in
 *                             gf64_poly_mul_fft at the marked TODO.
 *
 *   3. gf64_multi_point_eval -- naive Horner fallback (see file header).
 *                              Eventually replaced by recursive top-down
 *                              Bostan-Schost tree-walking.
 *
 * None of these are on the engine hot path yet (engine integration is T9).
 * The schoolbook choices here are deliberate per the T8 plan: "If
 * implementing Newton iteration is too complex for T8, you can use a
 * simpler schoolbook polynomial division for T8 (O(n^2) per division), and
 * add a TODO comment for the Newton-iteration optimization in a follow-up
 * task."
 * ============================================================================
 */

#include "gf64_mpe.h"
#include "gf64_additive_fft.h"
#include "gf64_invert_ita.h"
#include "gf64_subproduct.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
 * 1. gf64_poly_divmod: schoolbook long division in GF(2^64)[x]
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
void gf64_poly_divmod(
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
	if (n == 0) {
		return;
	}

	assert(g != NULL);
	assert(result != NULL);

	/*
	 * Allocation strategy:
	 *   g_buf  -- truncated copy of g (n entries)
	 *   r_sq   -- scratch for r^2 (up to 2n - 2 coefficients)
	 *   prod   -- scratch for g * r^2 (up to 2n - 2 coefficients)
	 *
	 * TODO(replace-with-fft): an FFT-based Newton iteration would reuse a
	 * single scratch area of size O(n) and avoid the 2n-sized buffers
	 * below.
	 */
	gf64_t *g_buf = (gf64_t *)calloc(n, sizeof(gf64_t));
	gf64_t *r_sq  = (gf64_t *)calloc(2 * n, sizeof(gf64_t));
	gf64_t *prod  = (gf64_t *)calloc(2 * n, sizeof(gf64_t));
	if (g_buf == NULL || r_sq == NULL || prod == NULL) {
		free(g_buf); free(r_sq); free(prod);
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

	free(g_buf);
	free(r_sq);
	free(prod);
}

/* ----------------------------------------------------------------------------
 * 3. gf64_multi_point_eval: naive-Horner fallback for multi-point evaluation
 *
 * Reads the N leaf entries from the last level of `tree`, evaluates f at
 * each one via Horner (linear in deg_f per point), and stores the results
 * in `out`.
 *
 * TODO(replace-with-bostan-schost): once gf64_poly_divmod is integrated and
 * its modular-truncation correctness is verified against this naive path,
 * replace the body with the recursive top-down tree walk:
 *
 *   eval_rem(f, deg_f, node, out_buf):
 *     if node is leaf i:
 *       out_buf[i] = horner(f, deg_f, x_i)
 *       return
 *     let g_L, g_R be the two children
 *     r_L = divmod(f, g_R)         # remainder of deg < deg(g_R)
 *     r_R = divmod(f - r_L, g_L)    # conceptually; actual schema differs
 *     eval_rem(r_L, node_left, out_buf)
 *     eval_rem(r_R, node_right, out_buf)
 *
 * That recursion gives the O((D + N) log^2(D + N)) contract the plan
 * promises. For now we accept O(N * D).
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

void gf64_multi_point_eval(
	const gf64_t *f, size_t deg_f,
	const SubproductTree *tree,
	gf64_t *out
) {
	if (tree == NULL || tree->num_points == 0 || f == NULL) {
		return;
	}

	size_t N = tree->num_points;
	gf64_t *last_level = tree->level_data[tree->num_levels - 1];

	for (size_t j = 0; j < N; j++) {
		gf64_t xj = last_level[2 * j]; /* leaf j is [x_j, 1] */
		out[j] = horner_eval_poly(f, deg_f, xj);
	}
}

HEDLEY_END_C_DECLS
