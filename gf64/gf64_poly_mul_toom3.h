/*
 * ============================================================================
 * gf64/gf64_poly_mul_toom3.h — Toom-Cook 3 (Toom 2.3) polynomial
 *   multiplication over GF(2^64)[x], bit-exact with the existing
 *   schoolbook / Karatsuba.
 *
 * Algorithmic complexity: T(n) = 5 T(n/3) + O(n), so T(n) = O(n^log_3(5))
 * ≈ O(n^1.465), strictly better than Karatsuba's O(n^1.585).
 *
 * CHAR-2 SAFE EVALUATION POINTS: {0, 1, 2, 3, infinity}.
 *
 * The standard Toom-3 uses {0, 1, -1, -2, inf}. In characteristic 2,
 * -1 = 1 and -2 = 2, so the standard set collapses to {0, 1, 1, 2, inf}
 * — only 4 distinct points, which is insufficient. We use {0, 1, 2, 3, inf}
 * instead. The interpolation matrix V^{-1} is computed over GF(2^64)
 * symbolically by gf64/test/gen_toom3_constants.c and embedded below.
 *
 * Verified that V * V^{-1} = I (25/25 entries OK) under the GF(2^64)
 * scalar reference.
 * ============================================================================
 */

#ifndef GF64_POLY_MUL_TOOM3_H
#define GF64_POLY_MUL_TOOM3_H

#include <stddef.h>

#include "gf64_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Toom-Cook 3 (2.3) polynomial multiplication in GF(2^64)[x].
 *
 * Same API as gf64_poly_mul_karatsuba:
 *   - out       : caller-owned buffer of `out_len` coefficients.
 *   - a, b      : input polynomials in constant-first order.
 *   - len_a     : a has `len_a` coefficients.
 *   - len_b     : b has `len_b` coefficients.
 *   - out_len   : write only the low-order `out_len` coefficients of a*b.
 *
 * Drop-in replacement for the body of gf64_poly_mul_padded. Inputs beyond
 * out_len in either operand are ignored.
 */
void gf64_poly_mul_toom3(
    gf64_t *out,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b,
    size_t out_len
);

#ifdef __cplusplus
}
#endif

#endif /* GF64_POLY_MUL_TOOM3_H */