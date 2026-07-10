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
 * Schoolbook O((deg_f + 1) * (deg_g + 1)) polynomial long division in
 * GF(2^64)[x]:
 *
 *   f(x) = g(x) * q(x) + r(x),  deg(r) < deg(g)
 *
 * @param f      Dividend coefficients [c_0, ..., c_deg_f].
 * @param deg_f  Degree of the dividend.
 * @param g      Divisor coefficients [c_0, ..., c_deg_g].
 * @param deg_g  Degree of the divisor. Must satisfy deg_g == 0 ==> g[0] != 0.
 * @param q      Output quotient buffer of size (deg_f - deg_g + 1).
 *               When deg_f < deg_g, the buffer receives a single 0 at q[0]
 *               and r receives (deg_g - 1) + 1 = deg_g coefficients.
 * @param r      Output remainder buffer of size (deg_f + 1). Used as a
 *               working buffer that initially holds a copy of f and is
 *               reduced in place; on return, the coefficients
 *               r[0..deg_g - 1] are the remainder (with the rest of the
 *               buffer being scratch / unspecified). The caller is
 *               responsible for sizing r to hold at least deg_f + 1
 *               coefficients so the working buffer can be expanded in
 *               place during the long-division elimination loop.
 *
 * Aborts on a zero divisor (g[0] == 0 when deg_g == 0). For deg_g > 0 the
 * leading coefficient g[deg_g] is required to be nonzero (the function
 * silently produces garbage if not).
 *
 * Correct for any GF(2^64) coefficients in {0, ..., 2^64 - 1}. Addition
 * in the inner loop is XOR, matching the field's characteristic-2
 * addition. Uses gf64_mul_reference as the field multiplication primitive.
 */
void gf64_poly_divmod(
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
 * The current implementation is a NAIVE FALLBACK that evaluates f at every
 * leaf of the tree independently using Horner's method. This is correct
 * but O(N * deg_f) field operations; the plan's deferred Bostan-Schost
 * top-down recursive form would be O((deg_f + N) log^2(deg_f + N)).
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
 *
 * TODO(replace-with-Bostan-Schost): once gf64_poly_divmod's correctness is
 * confirmed against the tree, replace the naive Horner loop with a
 * recursive top-down traversal:
 *   eval(f, node) -> r (degree < deg(node))
 *     if leaf: out[i] = horner(f, points[i])
 *     else:    eval(f mod child_L, node->left)
 *              eval(f mod child_R, node->right)
 * The recursive form requires deg_f modular reductions at each internal
 * node, each O(M log M) via the FFT-based gf64_poly_divmod.
 */
void gf64_multi_point_eval(
	const gf64_t *f, size_t deg_f,
	const SubproductTree *tree,
	gf64_t *out
);

HEDLEY_END_C_DECLS

#endif /* GF64_MPE_H */
