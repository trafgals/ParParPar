/*
 * ============================================================================
 * gf64/gf64_barycentric.h — Barycentric weight extraction over GF(2^64).
 *
 * T7 of the par3-cauchy-fft-kernel plan.
 *
 * The barycentric weights for a set of distinct points {x_0, ..., x_{N-1}}
 * in GF(2^64) are defined as:
 *
 *     W_j = 1 / P'(x_j)
 *
 * where P(x) = ∏_{i=0}^{N-1} (x + x_i) is the subproduct-tree root polynomial
 * (degree N) and P'(x) is its formal derivative.
 *
 * In characteristic 2, subtraction equals addition (XOR), so
 *     P(x) = ∏_i (x + x_i) = ∏_i (x XOR x_i).
 *
 * In characteristic 2 the formal derivative has a special form: even-powered
 * coefficients vanish (the chain rule gives a factor of k = 0 in GF(2) for
 * even k) and odd-powered coefficients stay but shift down by ONE degree.
 * Concretely, with P having constant-first coefficients c[0..N]:
 *
 *     P(x)  = c[0] + c[1]*x + c[2]*x^2 + c[3]*x^3 + c[4]*x^4 + c[5]*x^5 + ...
 *     P'(x) = c[1]    + 0*x^1  + c[3]*x^2 + 0*x^3 + c[5]*x^4 + ...
 *           = sum_{j even, j >= 0} c[j+1] * x^j
 *
 * So if we represent P' as a constant-first array `deriv` with the same
 * convention (deriv[j] is the coefficient of x^j in P'), then
 *
 *     deriv[j] = c[j + 1]    if j is even
 *     deriv[j] = 0           if j is odd
 *
 * Only the even-indexed slots of `deriv` carry P's odd-indexed coefficients;
 * the odd-indexed slots are zero. The degree of P' is therefore
 *
 *     - N - 1   when N is odd  (the largest odd position k = N contributes x^{N-1})
 *     - N - 2   when N is even (the largest odd position k = N - 1 contributes x^{N-2})
 *
 * Concretely, for the common case N = 1024 we have a length-1024 `deriv`
 * array with 512 nonzero entries (positions 0, 2, 4, ..., 1022) and an
 * actual degree of 1022.
 *
 * The barycentric weights are the multiplicative inverses of P'(x_j)
 * (equivalently, of ∏_{k != j} (x_j + x_k) — see chain-rule check below).
 * T5's
 * `gf64_invert_ita_batch` performs the inversion in O(N) batches per
 * element (63 squarings + 62 multiplies per element, AVX-512 vectorized).
 *
 * CURRENT EVALUATION STRATEGY
 * ---------------------------
 * Multi-point evaluation of P'(x_j) for every input point uses the
 * Bostan-Schost top-down tree walk (gf64_multi_point_eval, T8) with the
 * Newton-reciprocal divmod from issue #59 A1 — O(M(N) log N) instead of
 * the old O(N^2) Horner stop-gap. The derivative array is built in O(N)
 * XORs (char-2 rule above) and the N values are inverted in one shot via
 * T5's Itoh-Tsujii batch inverter (gf64_invert_ita_batch).
 * ============================================================================
 */

#ifndef GF64_BARYCENTRIC_H
#define GF64_BARYCENTRIC_H

#include "gf64_global.h"
#include "gf64_subproduct.h"

#include <stddef.h>

HEDLEY_BEGIN_C_DECLS

/*
 * Compute barycentric weights for the input points of a subproduct tree.
 *
 * Given the subproduct tree built for input points {x_0, ..., x_{N-1}},
 * write
 *
 *     weights_out[j] = 1 / P'(x_j)   for j = 0..N-1
 *
 * where P(x) = ∏_i (x + x_i) is the tree's root polynomial and P'(x) is
 * its formal derivative in GF(2^64).
 *
 * Inputs:
 *   tree        - A successfully built subproduct tree from
 *                 gf64_subproduct_tree_build (T6). The tree is READ-ONLY.
 *                 Its root (level_data[0]) must contain the P(x)
 *                 coefficients, and its leaves (level_data[num_levels-1])
 *                 must contain the input points as the constant term of each
 *                 leaf polynomial (constant-first layout: leaf_i = [x_i, 1]).
 *   weights_out - Caller-owned buffer of N gf64_t values. Must not alias
 *                 any storage reachable from `tree`.
 *
 * Output:
 *   weights_out[j] receives 1 / P'(x_j). In edge cases:
 *     - N == 0                    -> writes nothing, returns.
 *     - N == 1                    -> weights_out[0] = 1
 *                                   (because for P(x) = x + x_0, P'(x) = 1).
 *     - duplicate x_j == x_k      -> P'(x_j) = 0 and the weight
 *                                   conventionally becomes 0 (the T5
 *                                   inverter returns 0 for zero input).
 *
 * Complexity:
 *   - Step 1 (formal derivative):  O(N) memory + O(N) work.
 *   - Step 2 (multi-point evaluation): O(M(N) log N) via the Bostan-Schost
 *                                  tree walk (issue #59 A1; the O(N^2)
 *                                  Horner stop-gap and its T8 TODO are
 *                                  gone).
 *   - Step 3 (ITA inversion):      O(N log k) field ops per element, vectorized.
 *
 * On allocation failure: calls abort().
 */
void gf64_barycentric_weights(const SubproductTree *tree, gf64_t *weights_out);

HEDLEY_END_C_DECLS

#endif /* GF64_BARYCENTRIC_H */
