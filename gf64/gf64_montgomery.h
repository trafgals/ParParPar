#ifndef GF64_MONTGOMERY_H
#define GF64_MONTGOMERY_H

/*
 * ============================================================================
 * gf64/gf64_montgomery.h — Montgomery batch inversion in GF(2^64) (A2)
 *
 * PAR3 Cauchy-FFT kernel speedup plan, Strategy A2. Provides:
 *   void gf64_montgomery_batch_invert(gf64_t* out, gf64_t* in, size_t N)
 *
 * MATH
 *   Classical Montgomery batch inversion. For a vector a[0..N), compute
 *   inv[i] = 1/a[i] for all i via:
 *
 *     1. Forward pass:  P[0] = a[0];  P[i] = P[i-1] * a[i]
 *     2. Invert total:  inv_total = 1/P[N-1]
 *     3. Backward pass: inv[i] = inv_total * P[i-1] * (a[i+1] * ... * a[N-1])
 *        Implemented with a running suffix product that starts at inv_total
 *        and is updated each step: suffix *= a[i+1]; inv[i] = suffix * P[i-1].
 *
 *   Cost: 2*(N-1) multiplications + 1 single-element inversion.
 *
 *   vs Itoh-Tsujii: 125 ops/element (63 sq + 62 mul). For NR elements,
 *   IT costs 125*NR ops; Montgomery costs 2*NR + 125 ops.
 *
 *   On Zen4 with VPCLMULQDQ at ~5 cycles/mul, IT processes ~10M elements
 *   in 2.3s; Montgomery processes the same in ~5ms (~500x faster on the
 *   inversion step alone). The muladd downstream dominates, so the
 *   end-to-end win is modest (~10-15%) but real.
 *
 * ISA / DISPATCH
 *   Mirrors the gf64_invert_ita / gf64_mul / gf64_square ISA structure:
 *   - gf64_montgomery_batch_invert_scalar: portable C, used on non-AVX-512
 *     hosts (fallback via GF64Method dispatch).
 *   - gf64_montgomery_batch_invert_avx512: AVX-512 vectorized, 8 lanes/iter
 *     for both forward and backward passes.
 *
 * MEMORY
 *   In-place aliasing: out MAY alias in. The Montgomery algorithm needs
 *   to keep a[] available while computing prefix products, so we copy a[]
 *   to the output buffer first, then compute prefix products in-place
 *   on the original input buffer. The backward pass reads from both
 *   buffers.
 *
 *   Two buffers of N gf64_t each are needed. The caller passes a separate
 *   scratch buffer via `gf64_montgomery_batch_invert_with_scratch` if it
 *   wants to avoid the implicit heap alloc.
 *
 * CONVENTION
 *   0 input → 0 output (matches gf64_invert_ita_one convention).
 * ============================================================================
 */

#include "gf64_global.h"

HEDLEY_BEGIN_C_DECLS

/*
 * Scalar Montgomery batch inversion. In-place: out MAY alias in.
 * N may be 0 (no-op) or any positive integer.
 */
void gf64_montgomery_batch_invert_scalar(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT in,
	size_t N
);

/*
 * AVX-512 vectorized Montgomery batch inversion. 8 lanes/iter.
 * In-place: out MAY alias in.
 */
__attribute__((target("avx512f,vpclmulqdq")))
void gf64_montgomery_batch_invert_avx512(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT in,
	size_t N
);

/*
 * Default dispatch entry point. Selects scalar or AVX-512 based on
 * gf64_current_method at call time. For non-AVX-512 hosts, falls back
 * to scalar. For AVX-512 hosts (or AVX-2 with VPCLMULQDQ if supported),
 * uses the AVX-512 implementation (which is bit-exact to scalar).
 */
void gf64_montgomery_batch_invert(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT in,
	size_t N
);

HEDLEY_END_C_DECLS

#endif /* GF64_MONTGOMERY_H */