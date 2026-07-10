#ifndef GF64_MUL_H
#define GF64_MUL_H

/*
 * ============================================================================
 * gf64/gf64_mul.h — Public prototype for the AVX-512 vectorized gf64_mul
 *
 * T1 of the par3-cauchy-fft-kernel plan: this header is the new public surface
 * for the Barycentric path's element-by-element GF(2^64) multiplication.
 *
 * The scalar gf64_mul (gf64/gf64_solve.c:7-37) is preserved untouched; the
 * implementation here is a separate translation unit (gf64/gf64_mul_avx512.c)
 * that processes 8 GF(2^64) elements per iteration using VPCLMULQDQ +
 * AVX-512F integer ops. The reduction arithmetic (x^64 + x^4 + x^3 + x + 1,
 * i.e. 0x1B) is bit-exact to the scalar reference, so the SIMD path is
 * semantically interchangeable with the scalar one whenever AVX-512 is
 * available.
 *
 * HOST REQUIREMENTS
 *   The function is compiled with __attribute__((target("avx512f,vpclmulqdq")))
 *   (see gf64/gf64_mul_avx512.c). On non-AVX-512 hosts, the Barycentric path
 *   is dispatched to the scalar gf64_mul via the existing GF64Method dispatch
 *   (gf64/gf64_dispatch.c) — gf64_mul_avx512 is only ever called when the
 *   dispatch selects GF64_AVX512. No runtime CPU detection is required inside
 *   the function itself; the contract is enforced at the dispatch site, not
 *   here.
 *
 * CONCURRENCY
 *   The function is re-entrant and stateless. It writes `N` elements to `out`
 *   from `a[i] * b[i]` for i in [0, N). `out` MAY alias `a` or `b`.
 * ============================================================================
 */

#include "gf64_global.h"

HEDLEY_BEGIN_C_DECLS

/*
 * Element-by-element GF(2^64) multiplication of two length-N vectors.
 *
 *   out[i] = gf64_mul(a[i], b[i])   for i in [0, N)
 *
 * The reduction polynomial is x^64 + x^4 + x^3 + x + 1 (0x100000000000001B).
 * Output is bit-exact to the scalar gf64_mul in gf64/gf64_solve.c.
 *
 * @param out  Output vector (N elements). May alias `a` or `b`.
 * @param a    First input vector (N elements).
 * @param b    Second input vector (N elements).
 * @param N    Number of 64-bit GF elements (NOT byte count).
 *
 * Host: requires AVX-512F + VPCLMULQDQ. Caller is responsible for dispatching
 * to this only when the host supports those features.
 */
void gf64_mul_avx512(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT a,
	const gf64_t *HEDLEY_RESTRICT b,
	size_t N
);

HEDLEY_END_C_DECLS

#endif /* GF64_MUL_H */