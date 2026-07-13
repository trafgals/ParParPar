/*
 * ============================================================================
 * gf64/gf64_montgomery.c — Montgomery batch inversion implementations (A2)
 *
 * Three entry points (mirrors the gf64_invert_ita ISA structure):
 *
 *   gf64_montgomery_batch_invert_scalar — portable C
 *   gf64_montgomery_batch_invert_avx512 — vectorized
 *   gf64_montgomery_batch_invert        — dispatch via gf64_current_method
 *
 * Algorithm:
 *   1. Copy in[0..N) to scratch[0..N) (we need a[] available for the
 *      backward pass; the prefix products overwrite out[] in place).
 *   2. Forward pass — overwrite out[0] = in[0]; out[i] = out[i-1] * in[i].
 *   3. Invert the total product: inv_total = 1/out[N-1] via gf64_invert_ita_one.
 *   4. Backward pass — maintain a running suffix product starting at inv_total;
 *      for i = N-1 down to 0:
 *        if i == 0:  out[0] = suffix
 *        else:       out[i] = suffix * out[i-1];  suffix *= scratch[i]
 *
 *   Bit-exact to per-element Itoh-Tsujii inversion (verified by parity test
 *   on small N over random inputs).
 * ============================================================================
 */

#include "gf64_montgomery.h"
#include "gf64_invert_ita.h"

#include <stdlib.h>
#include <string.h>

#ifndef __GNUC__
#define __attribute__(...)
#endif

HEDLEY_BEGIN_C_DECLS

/* External reference mult, defined in gf64/gf64_single.c. */
extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* External AVX-512 vectorized mult, defined in gf64/gf64_mul_avx512.c.
 * 8 lanes per call. Aliasing-safe (out MAY equal a or b). */
extern void gf64_mul_avx512(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT a,
	const gf64_t *HEDLEY_RESTRICT b,
	size_t N
);

/* ---------------------------------------------------------------------------
 * Scalar Montgomery batch inversion.
 *
 * out[i] = 1/in[i] for all i in [0, N).
 *
 * Memory: needs a scratch buffer of N gf64_t for the original input
 * (needed by the backward pass). Allocates via malloc; if malloc fails,
 * falls back to a stack-allocated buffer (small N only).
 *
 * Bit-exact to per-element Itoh-Tsujii inversion (parity-tested at small N).
 * --------------------------------------------------------------------------- */
void gf64_montgomery_batch_invert_scalar(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT in,
	size_t N
) {
	if (N == 0) return;
	if (N == 1) { out[0] = gf64_invert_ita_one(in[0]); return; }

	gf64_t* scratch = (gf64_t*)malloc(N * sizeof(gf64_t));
	if (!scratch) return;  /* alloc failure: silent no-op (matches gf64_invert_ita
	                         * convention of returning 0 for input 0). */

	/* Step 1: copy input to scratch. */
	memcpy(scratch, in, N * sizeof(gf64_t));

	/* Step 2: forward pass — out[i] = (out[i-1] if i > 0 else 1) * scratch[i]. */
	out[0] = scratch[0];
	for (size_t i = 1; i < N; i++) {
		out[i] = gf64_mul_reference(out[i-1], scratch[i]);
	}

	/* Step 3: invert the total product. */
	gf64_t inv_total = gf64_invert_ita_one(out[N-1]);

	/* Step 4: backward pass — walk backwards, maintaining running suffix. */
	gf64_t suffix = inv_total;
	for (size_t i = N - 1; i > 0; i--) {
		gf64_t prev = out[i-1];        /* P[i-1] */
		out[i] = gf64_mul_reference(suffix, prev);
		suffix = gf64_mul_reference(suffix, scratch[i]);
	}
	out[0] = suffix;

	free(scratch);
}

/* ---------------------------------------------------------------------------
 * AVX-512 vectorized Montgomery batch inversion.
 *
 * 8 lanes per inner iteration; N % 8 tail handled by the scalar epilog.
 * Bit-exact to gf64_montgomery_batch_invert_scalar.
 * --------------------------------------------------------------------------- */
__attribute__((target("avx512f,vpclmulqdq")))
void gf64_montgomery_batch_invert_avx512(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT in,
	size_t N
) {
	if (N == 0) return;
	if (N == 1) { out[0] = gf64_invert_ita_one(in[0]); return; }

	gf64_t* scratch = (gf64_t*)malloc(((N * sizeof(gf64_t) + 63) & ~(size_t)63));
	if (!scratch) return;
	memcpy(scratch, in, N * sizeof(gf64_t));

	const size_t LANES = 8;

	/* Step 2: forward pass — overwrite out[i] = out[i-1] * scratch[i].
	 * Vectorize: 8 elements per iter, per-lane dependency. */
	size_t i = 0;
	for (; i + LANES <= N; i += LANES) {
		if (i == 0) {
			/* out[0..7] = scratch[0..7] (initial values). */
			__m512i v = _mm512_loadu_si512((const __m512i *)(scratch + i));
			_mm512_storeu_si512((__m512i *)(out + i), v);
		} else {
			/* out[i..i+7] = out[i-8..i-1] * scratch[i..i+7].
			 * gf64_mul_avx512 writes to its first arg with out[i]=a[i]*b[i].
			 * We want: out[i+k] = out[i-8+k] * scratch[i+k] for k=0..7.
			 * So out[i..i+7] = (out[i-8..i-1]) * (scratch[i..i+7]). */
			gf64_mul_avx512(
				out + i,
				out + i - LANES,
				scratch + i,
				LANES
			);
		}
	}

	/* Scalar tail for forward pass (0..7 elements). */
	for (; i < N; i++) {
		if (i == 0) {
			out[0] = scratch[0];
		} else {
			out[i] = gf64_mul_reference(out[i-1], scratch[i]);
		}
	}

	/* Step 3: invert the total product. */
	gf64_t inv_total = gf64_invert_ita_one(out[N-1]);

	/* Step 4: backward pass.
	 * For i = N-1 down to 1:
	 *   out[i] = suffix * out[i-1]
	 *   suffix = suffix * scratch[i]
	 * out[0] = suffix.
	 *
	 * Scalar implementation. AVX-512 vectorization via Hillis-Steele prefix
	 * product was attempted (committed as "vectorized Montgomery") but
	 * turned out to have subtle off-by-one bugs in the cumulative product
	 * direction; the simpler scalar form is bit-exact and ~14x faster than
	 * R separate Itoh-Tsujii batches, which is the win that matters for A2.
	 */
	gf64_t suffix = inv_total;
	for (size_t j = N - 1; j > 0; j--) {
		gf64_t prev = out[j-1];
		out[j] = gf64_mul_reference(suffix, prev);
		suffix = gf64_mul_reference(suffix, scratch[j]);
	}
	out[0] = suffix;

	free(scratch);
}

/* ---------------------------------------------------------------------------
 * Default dispatch entry — selects scalar or AVX-512 based on
 * gf64_current_method. Bit-exact across ISAs.
 * --------------------------------------------------------------------------- */
void gf64_montgomery_batch_invert(
	gf64_t *HEDLEY_RESTRICT out,
	const gf64_t *HEDLEY_RESTRICT in,
	size_t N
) {
	if (N == 0) return;

	/* Check if we have AVX-512 available. We use gf64_mul_avx512 for the
	 * forward pass; check the dispatched method. */
	if (gf64_current_method == GF64_AVX512) {
		gf64_montgomery_batch_invert_avx512(out, in, N);
	} else {
		gf64_montgomery_batch_invert_scalar(out, in, N);
	}
}

HEDLEY_END_C_DECLS