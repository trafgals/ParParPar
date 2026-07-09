#ifndef GF64_ADDITIVE_FFT_H
#define GF64_ADDITIVE_FFT_H

/*
 * Scalar Gao-Mateer-style additive FFT over GF(2^64).
 *
 * This T3 implementation uses the simplified monomial-basis recursion from
 * the par3-cauchy-fft-kernel plan. The public inverse intentionally preserves
 * the plan's non-normalized scaling: for n = 2^k, k > 0,
 *
 *   gf64_fft_inverse(gf64_fft_forward(p, n), n) = p * 2^(k - 1)
 *
 * where 2^(k - 1) is interpreted as the GF(2^64) element with that bit set.
 * gf64_poly_mul compensates internally and returns ordinary coefficient-basis
 * polynomial multiplication.
 */

#include "gf64_global.h"

#include <stddef.h>

HEDLEY_BEGIN_C_DECLS

void gf64_fft_forward(gf64_t *poly, size_t n);
void gf64_fft_inverse(gf64_t *poly, size_t n);

void gf64_poly_mul(
	gf64_t *out,
	const gf64_t *a,
	size_t deg_a,
	const gf64_t *b,
	size_t deg_b
);

HEDLEY_END_C_DECLS

#endif /* GF64_ADDITIVE_FFT_H */
