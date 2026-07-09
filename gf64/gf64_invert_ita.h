#ifndef GF64_INVERT_ITA_H
#define GF64_INVERT_ITA_H

/*
 * ============================================================================
 * gf64/gf64_invert_ita.h — Itoh-Tsujii batched inversion in GF(2^64)
 *
 * T5 of the par3-cauchy-fft-kernel plan. Provides:
 *   void gf64_invert_ita_batch(gf64_t* out, const gf64_t* in, size_t N)
 *   gf64_t gf64_invert_ita_one(gf64_t a)
 *
 * BACKGROUND
 *   The multiplicative inverse in GF(2^k) is a^(2^k - 2) by Fermat's
 *   Little Theorem (a^(2^k - 1) = 1 for any non-zero a, so a^(-1) = a^(2^k - 2)).
 *   For GF(2^64) we compute a^(2^64 - 2) using the Itoh-Tsujii addition chain
 *   derived from the recurrence t_1 = a, t_k = t_{k-1}^2 * a:
 *       t_2   = a^(2^2 - 1)   = a^3
 *       t_3   = a^(2^3 - 1)   = a^7
 *       ...
 *       t_63  = a^(2^63 - 1)
 *       final = t_63^2        = a^(2^64 - 2) = a^(-1)
 *   Each iteration is 1 squaring + 1 multiplication; 62 iterations total
 *   (k=2..63), followed by 1 final squaring. Total per-element cost:
 *       63 squarings + 62 multiplications
 *   (since k=1 starts with t = a, the loop runs from k=2 to 63 inclusive).
 *
 *   Squarings use the Frobenius map (linear, O(1) ops per element); multiplies
 *   reuse the existing gf64_mul_reference (gf64/gf64_single.c:6). All
 *   operations are branchless; zero input returns zero (convention).
 *
 * ISA / DISPATCH
 *   gf64_invert_ita_one is portable C. gf64_invert_ita_batch is AVX-512
 *   vectorized (8 lanes per inner iteration) via __attribute__((target(...)))
 *   on the function definition; the N % 8 tail falls back to the scalar
 *   gf64_invert_ita_one. Caller is responsible for ensuring AVX-512F is
 *   available before binding the AVX-512 path; the scalar one is always
 *   available. Mirrors the dispatch model of gf64/gf64_mul_avx512.c and
 *   gf64/gf64_square.c.
 *
 * CONCURRENCY
 *   Both functions are re-entrant and stateless. `out` MAY alias `in`
 *   (the AVX-512 batch loads all 8 inputs into a ZMM at the start, so
 *   aliasing is safe).
 * ============================================================================
 */

#include "gf64_global.h"

HEDLEY_BEGIN_C_DECLS

/*
 * Scalar Itoh-Tsujii inversion of a single GF(2^64) element.
 *
 *   out = gf64_invert_ita_one(a) == a^(-1) in GF(2^64)
 *
 * Returns 0 when a == 0 (the multiplicative inverse of 0 is undefined;
 * returning 0 is the convention used throughout this codebase).
 *
 * Per-element cost: 63 squarings + 62 multiplications.
 * Bit-exact to the inverse produced by the AVX-512 batched path; verified by
 * gf64/test/test_gf64_invert_ita.c.
 *
 * @param a  Input element (any 64-bit value).
 * @return   The multiplicative inverse of `a` (or 0 if a == 0).
 */
gf64_t gf64_invert_ita_one(gf64_t a);

/*
 * AVX-512 vectorized Itoh-Tsujii batched inversion of a length-N vector.
 *
 *   out[i] = gf64_invert_ita_one(in[i])   for i in [0, N)
 *
 * 8 lanes per inner iteration; the N % 8 tail is handled by the scalar
 * gf64_invert_ita_one. `out` MAY alias `in`.
 *
 * Host: requires AVX-512F (no VPCLMULQDQ needed — multiplies go through
 * the existing scalar path that calls the AVX-512 scalar reduction
 * gf64_region_scalar; see "tail strategy" in gf64/gf64_mul_avx512.c
 * which uses gf64_mul_reference for the tail). Caller is responsible
 * for ensuring AVX-512F is available before binding this entry; on
 * non-AVX-512 hosts the GF64Method dispatch in gf64_dispatch.c falls
 * back to the scalar gf64_inverse_batch_scalar (EEA-based, unchanged).
 */
void gf64_invert_ita_batch(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT in,
	size_t N
);

HEDLEY_END_C_DECLS

#endif /* GF64_INVERT_ITA_H */
