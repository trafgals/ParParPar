/*
 * ============================================================================
 * gf64/gf64_poly_mul_karatsuba.h
 *
 * Karatsuba polynomial multiplication over GF(2^64)[x], bit-exact with the
 * existing schoolbook in gf64_additive_fft.c::gf64_poly_mul_internal.
 *
 * This is the Phase 2a concrete-progress step toward the "beat PAR2 state of
 * the art" plan (see PR comment on commit 23e9411). It is a real algorithmic
 * speedup (O(n^2) → O(n^1.585)) but does NOT yet deliver asymptotic gains on
 * the canonical bench, which is dominated by the legacy 2D-muladd Cauchy
 * recovery kernel — that kernel does not call gf64_poly_mul at all. Karatsuba
 * here makes the polynomial-multiplication primitive cheap enough that the
 * T6 subproduct tree, T7 barycentric weights, and T8 Bostan-Schost MPE all
 * become usable as the underlying engine for the Barycentric Cauchy recovery
 * path (Phase 2c).
 *
 * The actual asymptotic win (Phase 2b: real FFT) requires a convolution-
 * theorem-correct transform and is research-grade work; Karatsuba is the
 * pragmatic, immediately useful intermediate primitive.
 *
 * API
 * ---
 *   void gf64_poly_mul_karatsuba(out, a, len_a, b, len_b, out_len)
 *
 *   - out       : caller-owned buffer of `out_len` coefficients.
 *   - a, b      : input polynomials in constant-first order.
 *   - len_a     : a has `len_a` coefficients (indices [0, len_a)).
 *   - len_b     : b has `len_b` coefficients.
 *   - out_len   : write only the low-order `out_len` coefficients of a*b.
 *
 * Drop-in replacement for the body of gf64_poly_mul_padded. Inputs beyond
 * out_len in either operand are ignored (consistent with the existing
 * schoolbook's `a_cap`/`b_cap` discipline).
 *
 * Cost: O(n^1.585) for balanced sizes above GF64_KARATSUBA_THRESHOLD (64);
 *       falls through to schoolbook below that.
 * ============================================================================
 */

#ifndef GF64_POLY_MUL_KARATSUBA_H
#define GF64_POLY_MUL_KARATSUBA_H

#include <stddef.h>

#include "gf64_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Karatsuba polynomial multiplication in GF(2^64)[x]. See file header.
 */
void gf64_poly_mul_karatsuba(
	gf64_t *out,
	const gf64_t *a, size_t len_a,
	const gf64_t *b, size_t len_b,
	size_t out_len
);

#ifdef __cplusplus
}
#endif

#endif /* GF64_POLY_MUL_KARATSUBA_H */