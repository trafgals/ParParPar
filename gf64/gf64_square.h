#ifndef GF64_SQUARE_H
#define GF64_SQUARE_H

/*
 * ============================================================================
 * gf64/gf64_square.h — Public prototypes for GF(2^64) Frobenius squaring
 *
 * T2 of the par3-cauchy-fft-kernel plan. Provides:
 *   void gf64_square       (gf64_t* out, const gf64_t* in, size_t N)
 *   void gf64_square_avx512(gf64_t* out, const gf64_t* in, size_t N)
 *
 * BACKGROUND
 *   Squaring in GF(2^k) is a *linear* map (Frobenius endomorphism):
 *     f(x) -> f(x)^2
 *   For each input bit i, the output bit sits at position 2i, then the
 *   128-bit result is reduced modulo the irreducible polynomial
 *     x^64 + x^4 + x^3 + x + 1          (low 64 bits = 0x1B)
 *   matching the reduction in gf64/gf64_solve.c:7-37 and the scalar SSE2
 *   reference gf64_mul_reference in gf64/gf64_single.c.
 *
 *   The output is bit-exact to `gf64_mul_reference(a[i], a[i])` for every
 *   element — verified by gf64/test/test_gf64_square.c.
 *
 * ISA / DISPATCH
 *   The scalar version is portable C; the AVX-512 version is compiled with
 *   __attribute__((target("avx512f"))) (no VPCLMULQDQ needed — squaring
 *   reduces to shifts/xors, no carry-less multiplies). Caller (the eventual
 *   FFT kernel) is responsible for ensuring the host supports AVX-512F
 *   before binding the AVX-512 entry; the scalar version is always
 *   available as a fallback.
 *
 * CONCURRENCY
 *   Both functions are re-entrant and stateless. They write N elements to
 *   `out` from `in[i]^2` for i in [0, N). `out` MAY alias `in`.
 * ============================================================================
 */

#include "gf64_global.h"

HEDLEY_BEGIN_C_DECLS

/*
 * Scalar element-by-element GF(2^64) squaring of a length-N vector.
 *
 *   out[i] = in[i]^2   for i in [0, N)
 *
 * Bit-exact to gf64_mul_reference(in[i], in[i]).
 *
 * @param out  Output vector (N elements). May alias `in`.
 * @param in   Input vector (N elements).
 * @param N    Number of 64-bit GF elements (NOT byte count).
 */
void gf64_square(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT in,
	size_t N
);

/*
 * AVX-512 vectorized element-by-element GF(2^64) squaring of a length-N
 * vector. 8 lanes per iteration; the N % 8 tail is handled by the scalar
 * gf64_square.
 *
 *   out[i] = in[i]^2   for i in [0, N)
 *
 * Bit-exact to gf64_mul_reference(in[i], in[i]).
 *
 * Host: requires AVX-512F (no VPCLMULQDQ / IFMA / BMI2 needed). Caller is
 * responsible for dispatching here only when the host supports AVX-512F.
 */
void gf64_square_avx512(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT in,
	size_t N
);

HEDLEY_END_C_DECLS

#endif /* GF64_SQUARE_H */