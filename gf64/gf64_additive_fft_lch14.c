/*
 * gf64/gf64_additive_fft_lch14.c — additive FFT over GF(2^64) using the
 *   Lin-Chung-Han (LCH14) novel polynomial basis.
 *
 * Algorithm (verified against the convolution-theorem probe at
 * gf64/test/test_gf64_fft_poly_mul.c):
 *
 *   Forward (level i, stride 2^i, pairs (j, j+2^i)):
 *     mu = s_i(v_j) / s_i(v_i) = s_i(v_j)   (since s_i(v_i) = 1 by definition)
 *     even = arr[j], odd = arr[j+2^i]
 *     arr[j]        = even ^ mu * odd
 *     arr[j + 2^i]  = arr[j] ^ odd
 *
 *   Inverse is the same network traversed in reverse:
 *     [Recover odd:]
 *       arr[j + 2^i] = arr[j] ^ arr[j + 2^i]      (one XOR)
 *     [Recover even:]
 *       arr[j]       = arr[j] ^ mu * arr[j + 2^i]  (one mul + one XOR)
 *
 * Convolution theorem (LCH14):
 *   For polynomials a, b in GF(2^64)[x] of degree < n = 2^m, with
 *   deg(a) + deg(b) < n (i.e., degree-2n product fits in n slots),
 *   AFT(a) * AFT(b) = AFT(a*b) pointwise.
 *
 *   The transform is plain (non-cyclic) convolution, so deg(a*b) must
 *   stay < n; the caller pads to n = 2^ceil(log2(deg(a) + deg(b) + 1)).
 *
 * Multiplier precomputation:
 *   s_i(v_j) for j = 0..2^i-1 is precomputed once per transform. Total
 *   work across all m levels: O(n) field squarings + XORs, dominated
 *   by the O(n log n) butterfly work.
 */

#include "gf64_additive_fft.h"
#include "gf64_mul.h"
#include "gf64_square.h"
#include "gf64_cantor_basis.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

HEDLEY_BEGIN_C_DECLS

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/*
 * Compute the multiplier mu_j = s_i(v_j) for j = 0..2^i-1.
 *
 * s_0(x) = x.
 * s_{i+1}(x) = s_i(x)^2 + s_i(x) (Frobenius composition).
 *
 * For each j, we compute s_i(v_j) by iterating s_0, s_1, ..., s_i
 * applied to v_j. Each step is one squaring + one XOR (cheap).
 *
 * Output: an array `mu` of size 2^i, where mu[j] = s_i(v_j).
 */
static void lch14_compute_multipliers(int i, gf64_t *mu, size_t stride) {
    size_t n = (size_t)1 << i;
    for (size_t j = 0; j < n; j++) {
        gf64_t v_j = 0;
        for (int k = 0; k < 20; k++) {
            if ((j >> k) & 1) v_j ^= GF64_CANTOR_BASIS[k];
        }
        gf64_t s = v_j;
        for (int step = 0; step < i; step++) {
            s = gf64_mul_reference(s, s) ^ s;  /* sigma(s) */
        }
        mu[j * stride] = s;
    }
}

/*
 * Forward LCH14 additive FFT. In-place.
 *
 * Pre: arr has n = 2^m elements representing the polynomial
 *   f(x) = sum_{k=0}^{n-1} arr[k] * X^k
 * Post: arr[k] = f(v_k) where v_k = sum_i bit_i(k) * GF64_CANTOR_BASIS[i]
 *       for k = 0..n-1.
 */
void gf64_fft_forward_lch14(gf64_t *arr, size_t n) {
    if (n <= 1) return;
    assert((n & (n - 1)) == 0);  /* n is a power of 2 */

    int m = 0;
    while (((size_t)1 << m) < n) m++;

    gf64_t *mu = (gf64_t *)malloc(((size_t)1 << (m - 1)) * sizeof(gf64_t));
    if (mu == NULL) abort();

    for (int i = 0; i < m; i++) {
        size_t stride = (size_t)1 << i;
        size_t chunk_count = n >> (i + 1);
        lch14_compute_multipliers(i, mu, 1);
        /* Each chunk has 2^(i+1) elements; within each chunk, pair
         * (chunk_base + j, chunk_base + j + 2^i) for j in [0, 2^i). */
        for (size_t chunk = 0; chunk < chunk_count; chunk++) {
            size_t chunk_base = chunk << (i + 1);
            for (size_t j = 0; j < stride; j++) {
                size_t lo = chunk_base + j;
                size_t hi = lo + stride;
                gf64_t even_v = arr[lo];
                gf64_t odd_v  = arr[hi];
                gf64_t gamma = gf64_mul_reference(mu[j], odd_v);
                arr[lo] = even_v ^ gamma;
                arr[hi] = arr[lo] ^ odd_v;
            }
        }
    }

    free(mu);
}

/*
 * Inverse LCH14 additive FFT. In-place.
 *
 * Pre: arr has n = 2^m elements representing evaluations
 *   f(v_0), f(v_1), ..., f(v_{n-1}).
 * Post: arr[k] = coefficient of X^k in f(x) (mod X^n, since the
 *       LCH14 basis X_i has degree < n).
 *
 * Note: with the plain (non-cyclic) transform, f must have degree
 *       < n. The caller is responsible for ensuring this; the
 *       polynomial-multiplication primitive handles padding.
 */
void gf64_fft_inverse_lch14(gf64_t *arr, size_t n) {
    if (n <= 1) return;
    assert((n & (n - 1)) == 0);  /* n is a power of 2 */

    int m = 0;
    while (((size_t)1 << m) < n) m++;

    gf64_t *mu = (gf64_t *)malloc(((size_t)1 << (m - 1)) * sizeof(gf64_t));
    if (mu == NULL) abort();

    /* Process levels in reverse. */
    for (int i = m - 1; i >= 0; i--) {
        size_t stride = (size_t)1 << i;
        size_t chunk_count = n >> (i + 1);
        lch14_compute_multipliers(i, mu, 1);
        for (size_t chunk = 0; chunk < chunk_count; chunk++) {
            size_t chunk_base = chunk << (i + 1);
            for (size_t j = 0; j < stride; j++) {
                size_t lo = chunk_base + j;
                size_t hi = lo + stride;
                /* Inverse: recover even first, then odd. */
                arr[hi] = arr[lo] ^ arr[hi];
                arr[lo] = arr[lo] ^ gf64_mul_reference(mu[j], arr[hi]);
            }
        }
    }

    free(mu);
}

HEDLEY_END_C_DECLS