/*
 * gf64/gf64_poly_mul_toom3.c — Toom-Cook 3 (Toom 2.3) over GF(2^64)[x].
 *
 * Algorithm
 * ---------
 * Split A and B into 3 limbs each (size k = ceil(n/3)):
 *   A(x) = A0(x) + x^k * A1(x) + x^(2k) * A2(x)
 *   B(x) = B0(x) + x^k * B1(x) + x^(2k) * B2(x)
 * where Ai(x), Bi(x) are polynomials of degree < k.
 *
 * For 5 evaluation "points" s in {0, 1, 2, 3, infinity}, define the
 * POLYNOMIALS A_s and B_s (each of degree < k):
 *   A_0  = A0
 *   A_1  = A0 + A1 + A2
 *   A_2  = A0 + 2*A1 + 4*A2      (with 2, 4 as GF(2^64) elements)
 *   A_3  = A0 + 3*A1 + 9*A2
 *   A_inf = A2                   (leading limb; s^2 dominates as s -> inf)
 *
 * Recursively compute the 5 POLYNOMIAL sub-products C_s = A_s * B_s, each
 * of degree < 2k.
 *
 * Recover the 5 limb-polynomials T_0..T_4 of C(x) by interpolating the
 * 5 C_s polynomials using the precomputed Vandermonde inverse V^{-1}:
 *   T_p(x) = sum_q V^{-1}[p][q] * C_{s_q}(x)
 *
 * Assemble: C(x) = T_0(x) + x^k * T_1(x) + ... + x^(4k) * T_4(x).
 *
 * Complexity: T(n) = 5 T(n/3) + O(n) = O(n^log_3(5)) ~ O(n^1.465),
 * strictly better than Karatsuba's O(n^1.585).
 *
 * Char-2 safety: standard Toom-3 uses {0,1,-1,-2,inf}; in char 2 this
 * collapses because -1 = 1, -2 = 2. We use {0,1,2,3,inf} instead, all
 * distinct elements of GF(2^64). Interpolation constants computed by
 * gf64/test/gen_toom3_constants.c and verified V*V^{-1} = I.
 */

#include "gf64_poly_mul_toom3.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

HEDLEY_BEGIN_C_DECLS

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* Forward-declare Karatsuba as a fallback for Toom-3's base case so the
 * 5 sub-products of size <= TOOM3_KARATSUBA_FALLBACK use the O(n^1.585)
 * path instead of schoolbook. */
void gf64_poly_mul_karatsuba(
    gf64_t *out,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b,
    size_t out_len
);

/*
 * Vandermonde inverse for points {0, 1, 2, 3, inf} over GF(2^64).
 * Computed by gf64/test/gen_toom3_constants.c and verified bit-exact.
 * T_p = sum_q V_inv[p][q] * C_{s_q}.
 */
static const gf64_t GF64_TOOM3_VINV[5][5] = {
    { 0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL },
    { 0x7ffffffffffffffaULL, 0x0000000000000001ULL, 0x800000000000000dULL, 0xfffffffffffffff6ULL, 0x0000000000000006ULL },
    { 0x0000000000000000ULL, 0x7ffffffffffffffbULL, 0xfffffffffffffff6ULL, 0x800000000000000dULL, 0x0000000000000007ULL },
    { 0x7ffffffffffffffbULL, 0x7ffffffffffffffbULL, 0x7ffffffffffffffbULL, 0x7ffffffffffffffbULL, 0x0000000000000000ULL },
    { 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000001ULL }
};

/* Schoolbook fallback (unused — Karatsuba is now the base case via
 * gf64_poly_mul_karatsuba). Kept here in case future tunings want a
 * schoolbook base case directly. */
#if 0
static void toom3_schoolbook(
    gf64_t *out,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b,
    size_t out_len
) {
    memset(out, 0, out_len * sizeof(gf64_t));
    size_t a_cap = (len_a < out_len) ? len_a : out_len;
    size_t b_cap = (len_b < out_len) ? len_b : out_len;
    for (size_t i = 0; i < a_cap; i++) {
        gf64_t ai = a[i];
        if (ai == 0) continue;
        size_t j_max = (b_cap < out_len - i) ? b_cap : (out_len - i);
        for (size_t j = 0; j < j_max; j++) {
            out[i + j] ^= gf64_mul_reference(ai, b[j]);
        }
    }
}
#endif

static void toom3_recursive(
    gf64_t *out, size_t out_len,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b
);

void gf64_poly_mul_toom3(
    gf64_t *out,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b,
    size_t out_len
) {
    if (out_len == 0) return;

    /* Truncate inputs at out_len. */
    if (len_a > out_len) len_a = out_len;
    if (len_b > out_len) len_b = out_len;

    if (len_a == 0 || len_b == 0) {
        memset(out, 0, out_len * sizeof(gf64_t));
        return;
    }

    toom3_recursive(out, out_len, a, len_a, b, len_b);
}

/*
 * Build the polynomial A_s in As_buf (size k):
 *   A_s = A0 + s*A1 + s^2*A2
 * c0, c1, c2 are GF(2^64) scalar coefficients; result is per-limb
 * linear combination.
 */
static void toom3_build_polynomial(
    gf64_t *As_buf, size_t k,
    const gf64_t *A0, const gf64_t *A1, const gf64_t *A2,
    gf64_t c0, gf64_t c1, gf64_t c2
) {
    /* Avoid VLAs (not portable to MSVC). Use a fixed cap of 1024 limbs. */
    gf64_t tmp[1024];
    if (k > 1024) {
        /* For very large k, fall back to a heap allocation. */
        gf64_t *t = (gf64_t *)malloc(k * sizeof(gf64_t));
        if (t == NULL) abort();
        for (size_t i = 0; i < k; i++) {
            t[i] = A0[i] ^ gf64_mul_reference(c1, A1[i]) ^ gf64_mul_reference(c2, A2[i]);
        }
        memcpy(As_buf, t, k * sizeof(gf64_t));
        free(t);
        return;
    }
    for (size_t i = 0; i < k; i++) {
        tmp[i] = A0[i] ^ gf64_mul_reference(c1, A1[i]) ^ gf64_mul_reference(c2, A2[i]);
    }
    memcpy(As_buf, tmp, k * sizeof(gf64_t));
    (void)c0; /* c0 is currently always 1; kept in API for future use */
}

static void toom3_recursive(
    gf64_t *out, size_t out_len,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b
) {
    /* Base case: below threshold, use Karatsuba (which itself falls back
     * to schoolbook at 64). Karatsuba's lower overhead wins below this
     * size on microbench. */
    const size_t TOOM3_BASE = 768;
    size_t n_raw = (len_a > len_b) ? len_a : len_b;
    if (n_raw <= TOOM3_BASE) {
        gf64_poly_mul_karatsuba(out, a, len_a, b, len_b, out_len);
        return;
    }

    size_t k = (n_raw + 2) / 3;

    /* Allocate zero-padded k-sized limb buffers. */
    gf64_t *A0 = (gf64_t *)calloc(k, sizeof(gf64_t));
    gf64_t *A1 = (gf64_t *)calloc(k, sizeof(gf64_t));
    gf64_t *A2 = (gf64_t *)calloc(k, sizeof(gf64_t));
    gf64_t *B0 = (gf64_t *)calloc(k, sizeof(gf64_t));
    gf64_t *B1 = (gf64_t *)calloc(k, sizeof(gf64_t));
    gf64_t *B2 = (gf64_t *)calloc(k, sizeof(gf64_t));
    if (!A0 || !A1 || !A2 || !B0 || !B1 || !B2) abort();

    size_t copy_a0 = (k < len_a) ? k : len_a;
    size_t copy_a1 = (len_a > k) ? ((len_a - k < k) ? len_a - k : k) : 0;
    size_t copy_a2 = (len_a > 2 * k) ? len_a - 2 * k : 0;
    size_t copy_b0 = (k < len_b) ? k : len_b;
    size_t copy_b1 = (len_b > k) ? ((len_b - k < k) ? len_b - k : k) : 0;
    size_t copy_b2 = (len_b > 2 * k) ? len_b - 2 * k : 0;

    if (copy_a0) memcpy(A0, a, copy_a0 * sizeof(gf64_t));
    if (copy_a1) memcpy(A1, a + k, copy_a1 * sizeof(gf64_t));
    if (copy_a2) memcpy(A2, a + 2 * k, copy_a2 * sizeof(gf64_t));
    if (copy_b0) memcpy(B0, b, copy_b0 * sizeof(gf64_t));
    if (copy_b1) memcpy(B1, b + k, copy_b1 * sizeof(gf64_t));
    if (copy_b2) memcpy(B2, b + 2 * k, copy_b2 * sizeof(gf64_t));

    /* Allocate 5 sub-product buffers (polynomials of degree < 2k each)
     * plus 2 build buffers. */
    gf64_t *C0 = (gf64_t *)calloc(2 * k, sizeof(gf64_t));
    gf64_t *C1 = (gf64_t *)calloc(2 * k, sizeof(gf64_t));
    gf64_t *C2 = (gf64_t *)calloc(2 * k, sizeof(gf64_t));
    gf64_t *C3 = (gf64_t *)calloc(2 * k, sizeof(gf64_t));
    gf64_t *Cinf = (gf64_t *)calloc(2 * k, sizeof(gf64_t));
    gf64_t *As_buf = (gf64_t *)malloc(k * sizeof(gf64_t));
    gf64_t *Bs_buf = (gf64_t *)malloc(k * sizeof(gf64_t));
    if (!C0 || !C1 || !C2 || !C3 || !Cinf || !As_buf || !Bs_buf) abort();

    gf64_t one = 1;
    gf64_t two = 2;
    gf64_t three = 3;
    gf64_t four = gf64_mul_reference(two, two);
    gf64_t nine = gf64_mul_reference(three, three);

    /* s = 0: A_0 = A0, B_0 = B0. */
    toom3_recursive(C0, 2 * k, A0, k, B0, k);

    /* s = 1: A_1 = A0 + A1 + A2. */
    toom3_build_polynomial(As_buf, k, A0, A1, A2, one, one, one);
    toom3_build_polynomial(Bs_buf, k, B0, B1, B2, one, one, one);
    toom3_recursive(C1, 2 * k, As_buf, k, Bs_buf, k);

    /* s = 2: A_2 = A0 + 2*A1 + 4*A2. */
    toom3_build_polynomial(As_buf, k, A0, A1, A2, one, two, four);
    toom3_build_polynomial(Bs_buf, k, B0, B1, B2, one, two, four);
    toom3_recursive(C2, 2 * k, As_buf, k, Bs_buf, k);

    /* s = 3: A_3 = A0 + 3*A1 + 9*A2. */
    toom3_build_polynomial(As_buf, k, A0, A1, A2, one, three, nine);
    toom3_build_polynomial(Bs_buf, k, B0, B1, B2, one, three, nine);
    toom3_recursive(C3, 2 * k, As_buf, k, Bs_buf, k);

    /* s = inf: A_inf = A2 (leading limb). */
    toom3_recursive(Cinf, 2 * k, A2, k, B2, k);

    /* Interpolate the 5 polynomial sub-products via V^{-1} to recover the
     * 5 limb polynomials T_0..T_4 of C(x), then assemble. */
    gf64_t *Cs[5] = { C0, C1, C2, C3, Cinf };
    memset(out, 0, out_len * sizeof(gf64_t));

    for (int i = 0; i < 5; i++) {
        size_t pos_base = i * k;
        if (pos_base >= out_len) break;
        size_t avail = out_len - pos_base;
        size_t write_count = (2 * k <= avail) ? 2 * k : avail;
        for (size_t ip = 0; ip < write_count; ip++) {
            gf64_t acc = 0;
            for (int j = 0; j < 5; j++) {
                acc ^= gf64_mul_reference(GF64_TOOM3_VINV[i][j], Cs[j][ip]);
            }
            out[pos_base + ip] ^= acc;
        }
    }

    free(C0); free(C1); free(C2); free(C3); free(Cinf);
    free(As_buf); free(Bs_buf);
    free(A0); free(A1); free(A2);
    free(B0); free(B1); free(B2);
}

HEDLEY_END_C_DECLS