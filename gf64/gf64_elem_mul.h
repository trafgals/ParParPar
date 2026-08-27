#ifndef __GF64_ELEM_MUL_H
#define __GF64_ELEM_MUL_H

/*
 * ============================================================================
 * gf64/gf64_elem_mul.h — ISA-dispatched element-wise GF(2^64) multiply
 *
 * Issue #59 Wave 1 / T3. Public surface for the vectorized element-wise
 * multiply used by the Fenger Toeplitz pipeline's execute stages 4a/4d
 * (gf64_fenger.c:269-296 and 706-803).
 *
 * Replaces the per-element scalar `gf64_mul_reference` calls in 4a
 * (`weighted[c] = in[c]*V_prime[c]`) and 4d (`out[r] = p_at_y[r]*V(y_r)^-1`)
 * with a single contiguous element-wise multiply, dispatched via the
 * standard GF64Method dispatch table (`gf64_current_method`). On AVX-512
 * hosts the existing clmul-Barrett lane primitive (gf64_mul_avx512.c)
 * processes 8 GF(2^64) elements per VPCLMULQDQ pair; AVX-2 / SSSE3 /
 * scalar hosts fall back to a scalar loop using the public SSE2 reference.
 *
 * CONCURRENCY
 *   The function is re-entrant and stateless. The scalar fallback
 *   delegates to gf64_mul_reference (gf64_single.c); the AVX-512 path
 *   delegates to gf64_mul_avx512 (gf64_mul_avx512.c, aliasing-safe).
 *   Output MAY alias either input.
 *
 * BIT-EXACTNESS
 *   The scalar fallback is bit-exact to gf64_mul_reference. The AVX-512
 *   path is bit-exact to the scalar reference (verified by
 * test_gf64_mul_avx512.c, mirrored by test_gf64_elem_mul.c).
 * ============================================================================
 */

#include "gf64_global.h"

HEDLEY_BEGIN_C_DECLS

/*
 * Element-by-element GF(2^64) multiplication with ISA dispatch.
 *
 *   out[i] = gf64_mul(a[i], b[i])   for i in [0, N)
 *
 * Dispatch (per call):
 *   - gf64_current_method == GF64_AVX512 AND gf64_has_vpclmulqdq:
 *     delegates to gf64_mul_avx512 (8 elements per VPCLMULQDQ pair).
 *   - otherwise (GF64_AVX2 / GF64_SSSE3 / GF64_SCALAR, or AVX-512 host
 *     lacking the SIGILL-tested ZMM probe): scalar loop using
 *     gf64_mul_reference.
 *
 * The function is safe to call with N == 0 (no-op). Aliasing: out MAY
 * alias `a` or `b`.
 *
 * @param out  Output vector (N elements). May alias `a` or `b`.
 * @param a    First input vector (N elements).
 * @param b    Second input vector (N elements).
 * @param N    Number of 64-bit GF elements (NOT byte count).
 */
void gf64_elem_mul(
	gf64_t *out,
	const gf64_t *a,
	const gf64_t *b,
	size_t N
);

HEDLEY_END_C_DECLS

#endif /* __GF64_ELEM_MUL_H */