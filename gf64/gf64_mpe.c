/*
 * ============================================================================
 * gf64/gf64_mpe.c — Multi-point evaluation and interpolation in GF(2^64)[x]
 *
 * T8 (eval + interp) of the par3-cauchy-fft-kernel plan.
 *
 * Public functions:
 *
 *   1. gf64_poly_divmod      -- schoolbook O((deg_f + 1) * (deg_g + 1)) long
 *                               division. Exact, fast enough for the test
 *                               suite sizes (deg_f <= 200).
 *
 *   2. gf64_poly_invmod      -- Newton-iteration modular inverse 1/g(x) mod
 *                               x^n. Cubic-time schoolbook today; future
 *                               FFT follow-up at the marked TODO.
 *
 *   3. gf64_multi_point_eval -- naive Horner fallback. O(N * deg_f).
 *
 *   4. gf64_poly_invmod_mod  -- polynomial modular inverse 1/g(x) mod f(x)
 *                               via iterative half-extended GCD. Used at
 *                               subproduct tree-build time to cache
 *                               P_left^(-1) mod P_right (needed by the
 *                               Bostan-Schost interpolation below).
 *
 *   5. gf64_multi_point_interp -- Bostan-Schost top-down multi-point
 *                               INTERPOLATION. Given N subproduct-tree
 *                               points and N values, returns the unique
 *                               polynomial of degree < N that matches.
 *                               GATED behind the PAR3_GF64_USE_INTERP env
 *                               var (default OFF): the function only
 *                               enters the Bostan-Schost body when the
 *                               env var is "1"/"true"/"yes"/"on".
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

/* ---- top-down recursive Bostan-Schost interpolation helper ---- */
static void gf64_mpi_recurse(
	const gf64_t *values,
	const SubproductTree *tree,
	size_t lev, size_t node_idx, size_t out_offset,
	gf64_t *out_poly,
	gf64_t *HEDLEY_RESTRICT scratch
) {
	const size_t num_levels = tree->num_levels;
	const size_t N_at_lev   = tree->num_points >> lev; /* leaves this node covers */
	const size_t f_size     = N_at_lev >> 1;            /* child poly size    */

	if (lev + 1 == num_levels) {
		/* Leaf: f is the constant values[out_offset]. */
		out_poly[0] = values[out_offset];
		return;
	}

	/*
	 * Four slots of size N_at_lev. The fourth (slot3) is needed because
	 * the leaf recursion writes fL into out_poly (== slot0 at this frame's
	 * address), and the combine step's divmod later overwrites slot0
	 * with k; we need a stable place to keep the saved fL until step F.
	 */
	gf64_t *slot0         = scratch;
	gf64_t *slot1         = scratch + N_at_lev;
	gf64_t *slot2         = scratch + 2 * N_at_lev;
	gf64_t *slot3         = scratch + 3 * N_at_lev;
	gf64_t *child_scratch = scratch + 4 * N_at_lev;

	/* Recurse on children. fL is written to slot0, fR to slot1.
	 *
	 * Note that at the leaves, slot0 == out_poly at that recursion depth,
	 * so the leaf writes its constant into both addresses — but that's
	 * fine since the combine steps treat slot0 as fL. */
	gf64_mpi_recurse(values, tree, lev + 1,
	                 /* node_idx */ 2 * node_idx,
	                 out_offset,
	                 slot0, child_scratch);
	gf64_mpi_recurse(values, tree, lev + 1,
	                 /* node_idx */ 2 * node_idx + 1,
	                 out_offset + f_size,
	                 slot1, child_scratch);

	/*
	 * Step A: save fL (slot0) into slot3 so step D's divmod can safely
	 *         overwrite slot0 with k. Slot0 is fL's canonical home for
	 *         the parent's contract; we cannot store fL elsewhere within
	 *         this frame's four slots because slot1 holds fR (still
	 *         needed for d) and slot2 is the d/prod scratch.
	 */
	memcpy(slot3, slot0, f_size * sizeof(gf64_t));

	/*
	 * Step B: d = fR + fL (XOR over char 2) into slot2; pad to N_at_lev.
	 */
	for (size_t i = 0; i < f_size; i++) {
		slot2[i] = slot1[i] ^ slot0[i];
	}
	memset(slot2 + f_size, 0, (N_at_lev - f_size) * sizeof(gf64_t));

	/*
	 * Sibling polynomials and cached inverse. The tree builder must
	 * have populated inv_mod_data[lev] for this (lev, node_idx) pair.
	 */
	const gf64_t *P_left     = tree->level_data[lev + 1]
	                           + (2 * node_idx)     * (f_size + 1);
	const gf64_t *P_right    = tree->level_data[lev + 1]
	                           + (2 * node_idx + 1) * (f_size + 1);
	const gf64_t *inv_LmodR  = tree->inv_mod_data[lev]
	                           + node_idx * (f_size + 1);

	/*
	 * Step C: tmp = d · inv_LmodR truncated to 2·f_size coefficients.
	 *         slot1 (fR region, no longer needed) becomes tmp.
	 */
	gf64_poly_mul_padded(slot1, slot2, f_size, inv_LmodR, f_size,
	                     2 * f_size);

	/*
	 * Step D: k = tmp mod P_right (degree < f_size).
	 *         q_buf lives in slot2 (overwriting d, no longer needed); r
	 *         in slot0. The divmod is OK with q_buf aliasing tmp or fL
	 *         because it does not read those positions after f is copied
	 *         into r.
	 */
	(void)gf64_poly_divmod(slot1, 2 * f_size - 1, P_right, f_size,
	                       slot2 /* q_buf */, slot0 /* r -> k */);

	/*
	 * Step E: prod = P_left · k, truncated to N_at_lev coefficients.
	 *         slot2 (last free region) becomes prod.
	 *
	 * P_left has degree f_size (== num_points / 2^(lev+1)), so it occupies
	 * f_size + 1 storage coefficients; passing the length as f_size would
	 * drop the x^(f_size) term and collapse the polynomial.
	 */
	gf64_poly_mul_padded(slot2, P_left, f_size + 1, slot0, f_size, N_at_lev);

	/*
	 * Step F: f_parent = saved_fL + prod, with high-order coefficients
	 *         beyond f_size coming from prod alone. Write into out_poly.
	 */
	for (size_t i = 0; i < f_size; i++) {
		out_poly[i] = slot3[i] ^ slot2[i];
	}
	for (size_t i = f_size; i < N_at_lev; i++) {
		out_poly[i] = slot2[i];
	}
}

/*
 * Ungated Bostan-Schost body. Exposed (non-static) so pipeline consumers
 * with a hard dependency on interpolation — currently the Fenger Toeplitz
 * path (gf64_fenger.c) — can call it directly without tripping the
 * PAR3_GF64_USE_INTERP opt-in gate. The public gf64_multi_point_interp
 * wrapper still enforces the gate for everyone else.
 */
void gf64_multi_point_interp_internal(
	const SubproductTree *tree,
	const gf64_t *values,
	gf64_t *out
) {
	const size_t N = tree->num_points;

	/*
	 * Worst-case scratch at the root: 4 slots of N, plus the children's
	 * scratch (4·N/2 + 4·N/4 + … = 4·N), giving ≤ 8·N total.
	 */
	const size_t scratch_words = 8 * N;
	gf64_t *scratch = (gf64_t *)malloc(scratch_words * sizeof(gf64_t));
	if (scratch == NULL) {
		abort();
	}

	gf64_mpi_recurse(values, tree,
	                 /* lev */ 0, /* node_idx */ 0, /* out_offset */ 0,
	                 out, scratch);

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
