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
 *   2. gf64_poly_invmod    -- Newton-iteration modular inverse 1/g(x) mod x^n.
 *                             The squaring and the multiplication per
 *                             doubling step delegate to gf64_poly_mul_padded
 *                             (PR-1: pre-allocated-output wrapper around the
 *                             schoolbook core, O(len_a * len_b) per call).
 *                             The asymptotic gain comes from a future
 *                             FFT-based poly_mul_padded — see the doc on
 *                             gf64_poly_mul_internal in gf64_additive_fft.c
 *                             for why the additive FFT does not directly
 *                             implement the convolution theorem in GF(2^64).
 *
 *   3. gf64_multi_point_eval -- naive Horner fallback (see file header).
 *                              Eventually replaced by recursive top-down
 *                              Bostan-Schost tree-walking.
 *
 * The gf64_poly_invmod path is independent of gf64_poly_divmod; together
 * they are the building blocks gf64_multi_point_eval will use once its
 * Bostan-Schost body is written. Until then, gf64_multi_point_eval's naive
 * Horner is correct but slow.
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
 * 2. gf64_poly_invmod: 1/g(x) mod x^n via Newton iteration over GF(2^64)
 *
 * The Newton iteration for modular polynomial inverse in characteristic 2:
 *
 *   r_{k+1} = g * r_k^2  (mod x^m_new)
 *
 * with m_new = min(2*m, n) and a seed r_0 = g[0]^-1 (so r_0 = 1 mod x).
 * Each doubling step squares the working precision, so after ceil(log2(n))
 * iterations we have a full n-coefficient inverse.
 *
 * Implementation note: each per-iteration multiplication is delegated to
 * gf64_poly_mul_padded (T3/T4 additively FFT-based polynomial multiply
 * from gf64_additive_fft.h). This avoids the cubic-time schoolbook cost
 * of the earlier stub; overall cost is O(M(n) log n) where M is the FFT-
 * based polynomial multiply. Per-iteration working set is one n-slot
 * scratch buffer (r_sq), reused across iterations.
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
	 *   g_buf   -- truncated copy of g (n entries, padded to zero beyond
	 *              deg_g + 1)
	 *   r_sq    -- scratch for r^2; size grows with m per iteration but
	 *              is bounded by 2*n - 2 (the worst case at m = n/2)
	 *
	 * Padded Newton multiplication reads only what it needs from `g_buf`
	 * (the early-iteration prefix m_new - 1) so the same backing buffer
	 * is safe at every step.
	 */
	gf64_t *g_buf = (gf64_t *)calloc(n,            sizeof(gf64_t));
	gf64_t *r_sq  = (gf64_t *)calloc((2 * n) - 1,  sizeof(gf64_t));
	if (g_buf == NULL || r_sq == NULL) {
		free(g_buf);
		free(r_sq);
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
		 * r_sq = result^2 truncated to 2*m_new - 1 coefficients.  The
		 * current `result` has at most m nonzero coefficients (degree
		 * < m by Newton's invariant: r_k ≡ g^-1 mod x^m), so result^2
		 * has degree < 2*m. We compute 2*m_new - 1 coefficients so the
		 * subsequent g * r_sq multiplication has enough precision to
		 * drive the next iteration correctly.
		 */
		size_t r_sq_len = 2 * m - 1;
		if (r_sq_len > (2 * n) - 1) {
			r_sq_len = (2 * n) - 1;
		}
		gf64_poly_mul_padded(r_sq, result, m, result, m, r_sq_len);

		/*
		 * prod = g_buf * r_sq truncated to m_new entries. The Newton-
		 * truncated multiplication idiom: only the low-order m_new
		 * coefficients of g * r_sq matter for the next iteration.
		 *
		 * - g_buf uses the first m_new entries (truncated); we cap
		 *   the read at n but the precision g needs is m_new.
		 * - r_sq uses all 2*m - 1 coefficients the squaring produced.
		 *
		 * Since the padding math bounds both inputs below 2 * m_new,
		 * the result is bit-exact to the original cubic-time version.
		 */
		size_t g_use_len = (m_new < n) ? m_new : n;
		gf64_poly_mul_padded(result, g_buf, g_use_len, r_sq, r_sq_len, m_new);

		/*
		 * Zero the high-order coefficients of result so the buffer
		 * stays deterministic for any code that reads past m_new (the
		 * Newton invariant only guarantees correctness up to m_new).
		 */
		if (m_new < n) {
			memset(result + m_new, 0, (n - m_new) * sizeof(gf64_t));
		}

		m = m_new;
	}

	free(g_buf);
	free(r_sq);
}

/* ----------------------------------------------------------------------------
 * 3. gf64_multi_point_eval: Bostan-Schost top-down multi-point evaluation
 *
 * Tree layout: the SubproductTree stores levels in BFS order.
 *   level_data[0]              = root polynomial of degree N
 *   level_data[1]              = 2 polynomials of degree N/2
 *   ...
 *   level_data[num_levels - 1] = N leaves, each [x_j, 1] (degree 1)
 *
 * Each level_data[lev] holds `level_lens[lev]` polynomials in BFS order,
 * each polynomial of degree `level_degs[lev]`.  Children of the polynomial
 * at level `lev` index `i` sit at level `lev + 1` indices `2i` and `2i + 1`.
 *
 * Recursion (Bostan-Schost 2003, top-down):
 *
 *   eval_at_subtree(f, deg_f, lev, out_offset):
 *     if lev == num_levels - 1:           # leaf
 *       x = leaf[0]
 *       out[out_offset] = horner(f, deg_f, x)
 *       return
 *     P_parent = level_data[lev][0 .. deg_parent]
 *     q, r = divmod(f, P_parent)          # f = P_parent * q + r
 *     eval_at_subtree(r,           lev + 1, out_offset)        # left half
 *     q_left, _ = divmod(q, P_left)       # q mod P_left
 *     eval_at_subtree(q_left,     lev + 1, out_offset + N/2^(lev+1))
 *
 * Correctness: at every recursive step the inputs to the children are
 * polynomials whose values at the children's leaves coincide with f at
 * those leaves (because r ≡ f (mod P_parent), and q_left ≡ q ≡ f*P_parent^-1
 * ≡ f (mod P_left) — since P_parent = P_left * P_right, q mod P_left =
 * f / P_parent mod P_left).
 *
 * Cost: O(M(n) log n) per level, summed across log N levels = O((D + N)
 * log^2(D + N)) when paired with an FFT-based divmod / multiply.  Current
 * gf64_poly_divmod is O(n^2) schoolbook, so this implementation has the
 * right CALL SHAPE but the same overall cost as the naive-Horner fallback
 * — the asymptotic win comes when divmod is upgraded to FFT (the same
 * FFT-multiply primitive PR-1 establishes the call shape for). Once that
 * FFT lands, no signature change here is needed.
 * ---------------------------------------------------------------------------- */

/*
 * Horner evaluation of a constant-first polynomial c[0..deg] at x = r.
 * Returns 0 when the polynomial is empty (deg == (size_t)(-1) sentinel).
 */
static gf64_t horner_eval_poly(const gf64_t *c, size_t deg, gf64_t r) {
	if (deg == (size_t)(-1)) {
		return 0;
	}
	gf64_t acc = c[deg];
	for (size_t i = deg; i > 0; i--) {
		acc = gf64_mul_reference(acc, r) ^ c[i - 1];
	}
	return acc;
}

/*
 * Top-down Bostan-Schost walker.
 *
 * Recursion contract: at the current node (level `lev`, index `node_idx`
 * within that level, covering leaves [out_offset, out_offset +
 * 2^(num_levels - 1 - lev))), the polynomial whose leaves we are
 * evaluating is `f`. We compute
 *
 *     f_L = f mod P_left
 *     f_R = f mod P_right
 *
 * and recurse on each side with the matching subtree. At the leaves,
 * f(x_j) = f_L(x_j) (resp. f_R) by construction, and a single Horner
 * per leaf writes the result.
 *
 * Subproduct tree indexing (BFS): at level_data[lev], polynomial i lives
 * at offset i * (level_degs[lev] + 1). Children of i at level lev + 1 are
 * at indices 2i and 2i + 1. So for a non-root call we MUST pass the
 * node_idx so the function can locate the right P_left / P_right at the
 * children's level.
 *
 * Scratch discipline: every recursive call needs (r_L, r_R, q) backed by
 * three disjoint regions, AND r_R must remain valid until the right
 * recursion consumes it. The simplest correct layout is a single per-call
 * scratch of size (deg_f + 1) for f itself plus three slots of size
 * (deg_parent + 1) for r_L, r_R, q; cross-frame aliasing is benign because
 * we copy `f` into scratch up front, so left-recursion's writes never
 * touch the parent's f_in.
 */
static void gf64_mpe_recurse(
	const gf64_t *f_in, size_t deg_f,
	const SubproductTree *tree,
	size_t lev, size_t node_idx, size_t out_offset,
	gf64_t *out, gf64_t *HEDLEY_RESTRICT scratch
) {
	const size_t last_lev = tree->num_levels - 1;

	if (lev == last_lev) {
		/* Leaf: node_idx is the leaf index directly. */
		const gf64_t *leaf = tree->level_data[last_lev] + 2 * node_idx;
		out[out_offset] = horner_eval_poly(f_in, deg_f, leaf[0]);
		return;
	}

	const size_t deg_parent = tree->level_degs[lev];
	const size_t deg_child  = tree->level_degs[lev + 1];
	const size_t slot       = (deg_f + 1 > deg_parent + 1) ? (deg_f + 1) : (deg_parent + 1);

	/*
	 * Copy f_in into the first slot of scratch so the two divmod calls
	 * below operate on owned memory. divmod may freely overwrite its
	 * own scratch buffers without aliasing the caller's `f_in`.
	 */
	gf64_t *f_copy = scratch;
	gf64_t *rL     = scratch + slot;
	gf64_t *rR     = scratch + 2 * slot;
	gf64_t *q      = scratch + 3 * slot;
	(void)q;

	for (size_t i = 0; i <= deg_f; i++) {
		f_copy[i] = f_in[i];
	}

	/*
	 * Children of (lev, node_idx) sit at level_data[lev + 1] indices
	 * 2*node_idx and 2*node_idx + 1.
	 */
	const gf64_t *P_left  = tree->level_data[lev + 1]
	                        + (2 * node_idx)     * (deg_child + 1);
	const gf64_t *P_right = tree->level_data[lev + 1]
	                        + (2 * node_idx + 1) * (deg_child + 1);

	gf64_poly_divmod(f_copy, deg_f, P_left,  deg_child, q, rL);
	gf64_poly_divmod(f_copy, deg_f, P_right, deg_child, q, rR);

	const size_t child_count = tree->num_points >> (lev + 1);

	gf64_mpe_recurse(rL, deg_child - 1, tree, lev + 1,
	                 2 * node_idx,     out_offset,
	                 out, scratch);
	gf64_mpe_recurse(rR, deg_child - 1, tree, lev + 1,
	                 2 * node_idx + 1, out_offset + child_count,
	                 out, scratch);
}

void gf64_multi_point_eval(
	const gf64_t *f, size_t deg_f,
	const SubproductTree *tree,
	gf64_t *out
) {
	if (tree == NULL || tree->num_points == 0 || f == NULL) {
		return;
	}

	const size_t N = tree->num_points;

	/*
	 * Worst-case scratch at the root: f has deg_f coefficients; the
	 * remainder slots need deg_parent + 1 = N + 1 entries at the root
	 * (and shrink at every deeper level, so once-size-fits-all at the
	 * root is correct). Four slots, each of size max(deg_f + 1, N + 1).
	 */
	const size_t slot =
		(deg_f + 1 > N + 1) ? (deg_f + 1) : (N + 1);
	const size_t scratch_words = 4 * slot;
	gf64_t *scratch = (gf64_t *)malloc(scratch_words * sizeof(gf64_t));
	if (scratch == NULL) {
		abort();
	}

	gf64_mpe_recurse(f, deg_f, tree,
	                 /* lev */ 0, /* node_idx */ 0, /* out_offset */ 0,
	                 out, scratch);

	free(scratch);
}

HEDLEY_END_C_DECLS
