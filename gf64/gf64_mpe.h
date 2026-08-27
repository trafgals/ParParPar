/*
 * ============================================================================
 * gf64/gf64_mpe.h — Multi-point evaluation in GF(2^64)[x]
 *
 * T8 of the par3-cauchy-fft-kernel plan.
 *
 * Provides three functions used by the Bostan-Schost top-down multi-point
 * evaluation:
 *
 *   - gf64_poly_divmod:    polynomial division f = g*q + r (schoolbook O(n^2)).
 *   - gf64_poly_invmod:    1/g(x) mod x^n via Newton iteration (single seed +
 *                          doubling). This is a STUB using cubic-time
 *                          schoolbook multiplication; an FFT-based follow-up
 *                          is the deferred optimization.
 *   - gf64_multi_point_eval: top-down Bostan-Schost tree-walking evaluator
 *                          (currently falling back to naive Horner per point,
 *                          O(N * deg_f) total). Correct but not optimal.
 *
 * Background on the Bostan-Schost algorithm (the pure-math contract this
 * module implements) can be found in Bostan & Schost (2004), "Polynomial
 * evaluation and interpolation on special sets of points". The asymptotic
 * target is O((D + N) log^2(D + N)) field operations; the current naive
 * Horner fallback is O(N * D) and exists to (a) keep the engine compilable
 * and testable in isolation, and (b) provide a correctness baseline that
 * the faster Bostan-Schost implementation will be cross-checked against.
 *
 * Polynomial convention: coefficients are constant-first. For a polynomial
 *   f(x) = c0 + c1*x + c2*x^2 + ... + c_deg*x^deg
 * f is stored as {c0, c1, ..., c_deg}. The leading coefficient is the last
 * element, so deg = number_of_coeffs - 1.
 *
 * Buffers passed in / out:
 *   - q (output): quotient buffer; must hold at least (deg_f - deg_g + 1)
 *                  coefficients when deg_f >= deg_g. The caller is
 *                  responsible for sizing.
 *   - r (output): remainder buffer; must hold at least (deg_g) coefficients
 *                  (the remainder degree is strictly less than deg_g).
 *   - result (output, gf64_poly_invmod): inverse buffer; must hold n
 *                  coefficients.
 *   - out (output, gf64_multi_point_eval): evaluation results, indexed [0..N).
 * ============================================================================
 */

#ifndef GF64_MPE_H
#define GF64_MPE_H

#include "gf64_global.h"
#include "gf64_subproduct.h"

#include <stddef.h>

HEDLEY_BEGIN_C_DECLS

/*
 * Fast polynomial long division in GF(2^64)[x]:
 *
 *   f(x) = g(x) * q(x) + r(x),  deg(r) < deg(g)
 *
 * Dispatches on quotient length m = deg_f - deg_g + 1:
 *   - m < 96:  schoolbook O((deg_f + 1) * (deg_g + 1)) long division
 *              (delegates to gf64_poly_divmod_schoolbook)
 *   - m >= 96: Newton-reciprocal division (issue #59 A1) — the
 *              quotient is recovered in O(M(m)) from rev(f) * rev(g)^{-1}
 *              mod x^m, with the modular inverse computed by Newton
 *              iteration and every multiplication routed through the
 *              HQC-FFT / Karatsuba / schoolbook mul dispatch
 *              (gf64_poly_mul_internal).
 *
 * The FULL division is O(M(m) + M(deg_f + 1)): the remainder is
 * reconstructed as r = f - g*q — one (deg_g + 1)-by-m product whose
 * output spans deg_f + 1 coefficients. When deg_g is large relative to
 * the quotient (m << deg_g), the reconstruction term dominates and the
 * "O(M(m))" quotient bound alone would overstate the guarantee
 * (cubic review f70a81ef P2 / f44ead49 P3).
 *
 * @param f      Dividend coefficients [c_0, ..., c_deg_f].
 * @param deg_f  Degree of the dividend.
 * @param g      Divisor coefficients [c_0, ..., c_deg_g].
 * @param deg_g  Degree of the divisor. Must satisfy deg_g == 0 ==> g[0] != 0.
 * @param q      Output quotient buffer of size (deg_f - deg_g + 1).
 *               When deg_f < deg_g, the buffer receives a single 0 at q[0]
 *               and r receives (deg_g - 1) + 1 = deg_g coefficients.
 * @param r      Output remainder buffer of size max(deg_f + 1, deg_g).
 *               Used as a working buffer that initially holds a copy of f
 *               and is reduced in place; on return, the coefficients
 *               r[0..deg_g - 1] are the remainder. For deg_f < deg_g the
 *               region r[deg_f + 1 .. deg_g - 1] is ZEROED (the half-EGCD
 *               consumer memcpys the whole buffer), so callers must size r
 *               to hold max(deg_f + 1, deg_g) coefficients — the deg_f + 1
 *               sizing alone is an out-of-bounds write in that edge case
 *               (cubic review f70a81ef P2; test_gf64_divmod_parity sizes r
 *               to deg_g for deg_f < deg_g).
 *
 * Aborts on a zero divisor (g[0] == 0 when deg_g == 0). For deg_g > 0 the
 * leading coefficient g[deg_g] is required to be nonzero (the function
 * silently produces garbage if not).
 *
 * Correct for any GF(2^64) coefficients in {0, ..., 2^64 - 1}. Addition
 * in the inner loop is XOR, matching the field's characteristic-2
 * addition. Both paths are bit-exact to the schoolbook reference
 * (gf64_poly_divmod_schoolbook); test_gf64_divmod_parity.c pins this
 * across the dispatch boundary.
 */
void gf64_poly_divmod(
	const gf64_t *f, size_t deg_f,
	const gf64_t *g, size_t deg_g,
	gf64_t *q,    gf64_t *r
);

/*
 * Schoolbook O((deg_f + 1) * (deg_g + 1)) polynomial long division in
 * GF(2^64)[x]. REFERENCE implementation — kept public so the fast
 * Newton-reciprocal gf64_poly_divmod and its parity test can cross-check
 * bit-exactness, and used directly by the dispatcher for short quotients.
 *
 * Same contract as gf64_poly_divmod (identical buffer sizing, same
 * zero-divisor abort, same deg_f < deg_g edge case).
 */
void gf64_poly_divmod_schoolbook(
	const gf64_t *f, size_t deg_f,
	const gf64_t *g, size_t deg_g,
	gf64_t *q,    gf64_t *r
);

/*
 * Modular inverse of a polynomial: result = 1/g(x) mod x^n in GF(2^64)[x].
 *
 * Uses a Newton-iteration scheme that doubles precision each step:
 *
 *   r_0   = 1/g(0) = g[0]^-1
 *   r_{k+1} = g * r_k^2 (mod x^m_new)   with m_new = min(2*m, n)
 *
 * At every doubling step the truncation to m_new coefficients ensures the
 * lower-order coefficients are exact. The final result satisfies
 *
 *   g(x) * result(x) ≡ 1 (mod x^n)
 *
 * @param g      Polynomial coefficients [c_0, ..., c_deg_g]. Must satisfy
 *               g[0] != 0 (otherwise the inverse does not exist).
 * @param deg_g  Degree of g.
 * @param n      Number of coefficients to compute.
 * @param result Output buffer of size n. On return, result(x) holds the
 *               unique polynomial in GF(2^64)[x] of degree < n satisfying
 *               g(x) * result(x) ≡ 1 (mod x^n).
 *
 * Aborts on n == 0 (no-op would be a cleaner contract but the function
 * explicitly returns instead for n == 0; see the .c file).
 *
 * Implementation note: this is a STUB that uses cubic-time schoolbook
 * gf64_poly_mul for both `result^2` and `g * result^2`. The asymptotic cost
 * is O(n^2 log n) instead of the proper O(n log n) of an FFT-based
 * Newton iteration. The TODO is to swap the schoolbook poly_mul call for
 * the existing gf64_poly_mul_fft (T3/T4) once the optimized path is
 * needed in the engine.
 */
void gf64_poly_invmod(
	const gf64_t *g, size_t deg_g,
	size_t n,
	gf64_t *result
);

/*
 * Multi-point evaluation: given a polynomial f of degree <= deg_f and a
 * subproduct tree built from N input points, compute
 *
 *   out[j] = f(tree.points[j])   for j = 0 .. N-1
 *
 * Implementation dispatches on the interpolation degree bound:
 *
 *   - deg_f < N: Bostan-Schost top-down recursive tree walk
 *     (`gf64_mp_eval_recurse`). Each non-leaf node computes
 *     f mod P_left and f mod P_right via gf64_poly_divmod; P_left/right
 *     vanish on their respective leaf sets so the remainders agree with
 *     f at the leaves. Cost O((deg_f + N) log(deg_f + N) log N) field
 *     operations in practice.
 *
 *   - deg_f >= N or N == 1: HORNER FALLBACK. The Bostan-Schost walk
 *     requires the interpolation degree bound deg_f < N; the Fenger
 *     pipeline evaluates degree-(N_in - 1) polynomials at R recovery
 *     points where R < N_in (the prepare-time V(y_r) eval and the
 *     per-word p_w eval). When the bound is violated, evaluate f at
 *     each leaf independently via Horner's method. Cost O(N * deg_f).
 *
 * @param f      Polynomial coefficients of the function to evaluate,
 *               [c_0, ..., c_deg_f].
 * @param deg_f  Degree of f.
 * @param tree   Subproduct tree built by gf64_subproduct_tree_build (T6).
 *               Must have leaf entries [x_i, 1] at the last level.
 * @param out    Output buffer of size tree->num_points.
 *
 * On a NULL/empty tree, returns immediately without writing to out. The
 * caller must size out to at least tree->num_points gf64_t slots.
 */
void gf64_multi_point_eval(
	const gf64_t *f, size_t deg_f,
	const SubproductTree *tree,
	gf64_t *out
);

/*
 * Polynomial modular inverse (1/g mod f via half-extended GCD). T8b
 * building block. Returns 0 on success and writes `deg_f` coefficients
 * to `inv_out`; -1 if g and f share a non-trivial GCD (modular inverse
 * does not exist).
 *
 * NOTE (issue #59 A1): no longer called from gf64_subproduct_tree_build —
 * the per-pair inverse cache was removed and the interpolation switched
 * to the derivative-based Lagrange form (see gf64_multi_point_interp_internal).
 * This function is kept as a standalone library primitive, pinned by
 * test_gf64_invmod_mod_parity.c. Its internal EGCD steps now benefit from
 * the fast Newton-reciprocal gf64_poly_divmod.
 *
 * Inputs `g` (deg_g) and `f` (deg_f) must be coprime; `f`'s leading
 * coefficient must be non-zero (required by gf64_poly_divmod). Caller
 * is responsible for sizing `inv_out` to hold at least `max(1, deg_f)`
 * slots — the deg_f == 0 case (constant modulus) writes the scalar
 * inverse inv_out[0] = 1/g[0] (cubic review 5a3b44c9 P1).
 *
 * Cost: O(deg_f · M(max(deg_f, deg_g))) field ops per inverse with the
 * fast divmod — the EGCD runs O(deg_f) steps but carries degree-deg_g
 * state (r_0 = g), so the O(deg_f · M(deg_f)) bound holds as stated
 * only when deg_g = O(deg_f) (cubic review 5a3b44c9 P2).
 */
int gf64_poly_invmod_mod(
	const gf64_t *g, size_t deg_g,
	const gf64_t *f, size_t deg_f,
	gf64_t *inv_out
);

/*
 * Top-down Bostan-Schost multi-point INTERPOLATION (T8b). Given N
 * subproduct-tree points and N values at those points, returns the
 * unique polynomial of degree < N satisfying out(x_j) = values[j]
 * for every j in [0..N).
 *
 * Algorithm (issue #59 A1, derivative-based Lagrange form): the tree
 * needs NO inverse cache — one MPE of the root derivative P' gives
 * P'(x_j), the values are scaled to z_j = y_j / P'(x_j), and the
 * top-down combine f_parent = f_L·P_R + f_R·P_L routes through the tiered
 * mul dispatch (gf64_poly_mul_padded: schoolbook for small/asymmetric
 * operands, Karatsuba outside the HQC range, HQC-FFT at scale).
 * O(M(N) log N) end to end.
 *
 * GATED behind the PAR3_GF64_USE_INTERP env var. The function checks
 * the env var at runtime; if unset/0/false/no/off, it returns
 * immediately without modifying `out`. Set
 *     PAR3_GF64_USE_INTERP=1   (or true/yes/on)
 * to enable. Default OFF — production dispatch is NOT wired; this is
 * opt-in only via the env var.
 *
 * @param tree    Subproduct tree built by gf64_subproduct_tree_build
 *                (T6). Points must be distinct (P'(x_j) != 0 is
 *                required by the derivative form; duplicates abort).
 * @param values  N evaluation results at the subproduct-tree leaves
 *                [values[0], ..., values[N-1]] = [f(x_0), ..., f(x_{N-1})].
 * @param out     Output buffer of N coefficients. On success,
 *                out[0..N-1] is the unique polynomial of degree < N
 *                satisfying out(x_j) = values[j]. On disabled-by-gate
 *                the buffer is left untouched.
 *
 * Test hooks (declared here for the unit tests; production code
 * should not touch these):
 *
 *   gf64_interp_dispatch_reset():       clear the env-var cache so a
 *                                       fresh setenv()/unsetenv() takes
 *                                       effect on the next call.
 *   gf64_interp_dispatch_probe_count(): number of invocations that
 *                                       actually entered the Bostan-
 *                                       Schost body. Used to verify
 *                                       "default OFF" semantics.
 *   gf64_interp_dispatch_reset_probe(): zero the probe counter.
 */
void gf64_multi_point_interp(
	const SubproductTree *tree,
	const gf64_t *values,
	gf64_t *out
);

/*
 * Ungated Bostan-Schost interpolation body. Same contract as
 * gf64_multi_point_interp but does NOT check the PAR3_GF64_USE_INTERP
 * env-var gate — it always enters the Bostan-Schost body and writes
 * `out`. Intended for pipeline consumers with a hard dependency on
 * interpolation (the Fenger Toeplitz path, gf64_fenger.c). NULL
 * tree/values/empty-tree still return immediately per the shared
 * gf64_multi_point_eval contract.
 */
void gf64_multi_point_interp_internal(
	const SubproductTree *tree,
	const gf64_t *values,
	gf64_t *out
);

/*
 * Weights-aware Bostan-Schost interpolation body (issue #59 A1) — the
 * per-word hot path of the Fenger pipeline.
 *
 * Same unique-polynomial contract as gf64_multi_point_interp_internal,
 * but takes the barycentric weights w_j = 1/P'(x_j) (as produced by
 * gf64_barycentric_weights) as an explicit input, so the per-call cost
 * drops to the top-down combine only: z_j = values[j] * w_j is one mul
 * per point — NO MPE walk, NO scalar inversions. The Fenger prepare
 * phase computes the weights once per tree and amortizes them across
 * all B words.
 *
 * @param tree    Subproduct tree built by gf64_subproduct_tree_build
 *                (T6). Points must be distinct.
 * @param values  N evaluation results at the subproduct-tree leaves.
 * @param weights N barycentric weights w_j = 1/P'(x_j); MUST NOT be
 *                NULL for external callers (NULL is the internal
 *                pre-scaled-z convention used by interp_internal).
 * @param out     Output buffer of N coefficients: the unique polynomial
 *                of degree < N satisfying out(x_j) = values[j].
 *
 * NULL-safe contract: on NULL tree / values / out (or an empty tree,
 * num_points == 0), returns immediately without writing — the same
 * guarantee as gf64_multi_point_interp_internal (cubic review f44ead49
 * P2). A NULL `weights` is NOT an error: it selects the internal
 * pre-scaled-z convention.
 */
void gf64_multi_point_interp_weights(
	const SubproductTree *tree,
	const gf64_t *values,
	const gf64_t *weights,
	gf64_t *out
);

void gf64_interp_dispatch_reset(void);
int  gf64_interp_dispatch_probe_count(void);
void gf64_interp_dispatch_reset_probe(void);

/* ============================================================================
 * T4 (issue #59): arena-backed scratch for the polynomial division path.
 *
 * The Newton-reciprocal gf64_poly_divmod mallocs 5 working buffers per
 * call (m>=96) and gf64_poly_invmod mallocs 3 more; the Bostan-Schost
 * MPE walk calls divmod twice per internal node, so one evaluation used
 * to perform O(N log N) heap allocations. The _scratch variants below
 * take a caller-owned bump arena instead: identical arithmetic, zero
 * per-call heap traffic. Bit-exact to the malloc variants (pinned by
 * test_gf64_divmod_parity / test_gf64_mpe).
 * ============================================================================ */

typedef struct {
	gf64_t *data;
	size_t cap;   /* capacity in gf64_t words */
	size_t used;
} gf64_arena_t;

/* Allocate `words` gf64_t words for the arena. Returns 0 on success,
 * nonzero on allocation failure (arena left zeroed). Words exceeding
 * SIZE_MAX/sizeof(gf64_t) is treated as failure (returns 1) so the
 * size_t multiplication never wraps. */
int  gf64_arena_init(gf64_arena_t *a, size_t words);
void gf64_arena_free(gf64_arena_t *a);
size_t gf64_arena_mark(const gf64_arena_t *a);
void gf64_arena_release(gf64_arena_t *a, size_t mark);
gf64_t *gf64_arena_push(gf64_arena_t *a, size_t words);

/* Same contracts as gf64_poly_divmod / gf64_poly_invmod. `arena` must be
 * initialised and have sufficient capacity; scratch is reclaimed when the
 * caller releases the arena mark taken before the call. */
void gf64_poly_divmod_scratch(
	const gf64_t *f, size_t deg_f,
	const gf64_t *g, size_t deg_g,
	gf64_t *q,    gf64_t *r,
	gf64_arena_t *arena
);

void gf64_poly_invmod_scratch(
	const gf64_t *g, size_t deg_g,
	size_t n,
	gf64_t *result,
	gf64_arena_t *arena
);

/* Test instrumentation: number of heap allocations performed inside the
 * divmod/invmod implementations since the last reset. Zero when every
 * call is routed through an arena. */
extern size_t gf64_mpe_heap_alloc_count;

HEDLEY_END_C_DECLS

#endif /* GF64_MPE_H */
