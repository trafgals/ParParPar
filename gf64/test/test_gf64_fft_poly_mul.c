/*
 * ============================================================================
 * gf64/test/test_gf64_fft_poly_mul.c
 *
 * Phase 2b FFT convolution-theorem probe.
 *
 * Tests whether the LCH14 additive FFT in gf64_additive_fft_lch14.c
 * satisfies the convolution theorem: forward-FFT + pointwise-multiply +
 * inverse-FFT should compute polynomial convolution in GF(2^64)[x].
 *
 * The convolution theorem (LCH14) holds when deg(a) + deg(b) < n, where
 * n is the transform length. So this test uses inputs of length n/2
 * each (so deg sum < n).
 *
 * Build & run from gf64/test/:
 *   $(CC) -O2 -march=native -I.. test_gf64_fft_poly_mul.c \
 *         ../gf64_additive_fft.c ../gf64_additive_fft_lch14.c \
 *         ../gf64_poly_mul_karatsuba.c ../gf64_mpe.c ../gf64_subproduct.c \
 *         ../gf64_invert_ita.c ../gf64_mul_avx512.c ../gf64_square.c \
 *         ../gf64_single.c -o test_gf64_fft_poly_mul && ./test_gf64_fft_poly_mul
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../gf64_additive_fft.h"
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

extern void gf64_fft_forward_lch14(gf64_t *arr, size_t n);
extern void gf64_fft_inverse_lch14(gf64_t *arr, size_t n);

static uint64_t g_rng = 0xC0DEBA5EFACE0001ULL;
static uint64_t splitmix64_next(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int g_passed = 0;
static int g_failed = 0;
static void pass(const char *name) { printf("  PASS: %s\n", name); g_passed++; }
static void fail(const char *name) { printf("  FAIL: %s\n", name); g_failed++; }

/* Schoolbook reference. */
static void schoolbook(
    gf64_t *out, const gf64_t *a, size_t la, const gf64_t *b, size_t lb
) {
    memset(out, 0, (la + lb - 1) * sizeof(gf64_t));
    for (size_t i = 0; i < la; i++) {
        for (size_t j = 0; j < lb; j++) {
            out[i + j] ^= gf64_mul_reference(a[i], b[j]);
        }
    }
}

static int test_one(const char *name, size_t n, uint64_t seed) {
    /* Use input lengths n/2 to satisfy the LCH14 convolution theorem
     * (deg(a) + deg(b) < n). */
    size_t a_len = n / 2;
    size_t b_len = n / 2;
    if (a_len == 0) a_len = 1;
    if (b_len == 0) b_len = 1;

    g_rng = seed;
    gf64_t *a = (gf64_t *)malloc(a_len * sizeof(gf64_t));
    gf64_t *b = (gf64_t *)malloc(b_len * sizeof(gf64_t));
    for (size_t i = 0; i < a_len; i++) a[i] = splitmix64_next();
    for (size_t i = 0; i < b_len; i++) b[i] = splitmix64_next();

    gf64_t *ab_ref = (gf64_t *)calloc(a_len + b_len - 1, sizeof(gf64_t));
    schoolbook(ab_ref, a, a_len, b, b_len);

    /* Build zero-padded length-n buffers, forward-FFT each, pointwise multiply, inverse-FFT. */
    gf64_t *A = (gf64_t *)calloc(n, sizeof(gf64_t));
    gf64_t *B = (gf64_t *)calloc(n, sizeof(gf64_t));
    memcpy(A, a, a_len * sizeof(gf64_t));
    memcpy(B, b, b_len * sizeof(gf64_t));

    gf64_fft_forward_lch14(A, n);
    gf64_fft_forward_lch14(B, n);
    for (size_t i = 0; i < n; i++) {
        A[i] = gf64_mul_reference(A[i], B[i]);
    }
    gf64_fft_inverse_lch14(A, n);

    int mismatches = 0;
    size_t check = a_len + b_len - 1;
    for (size_t i = 0; i < check; i++) {
        if (A[i] != ab_ref[i]) {
            if (mismatches < 3) {
                printf("    %s @ n=%zu: a*b[%zu] fft=0x%016llx ref=0x%016llx\n",
                       name, n, i,
                       (unsigned long long)A[i], (unsigned long long)ab_ref[i]);
            }
            mismatches++;
        }
    }

    free(a); free(b); free(ab_ref); free(A); free(B);

    if (mismatches == 0) {
        pass(name);
        return 0;
    } else {
        printf("    %s @ n=%zu: %d mismatches\n", name, n, mismatches);
        fail(name);
        return 1;
    }
}

int main(void) {
    printf("Phase 2b FFT convolution-theorem probe (LCH14)\n");
    printf("================================================\n\n");
    printf("Inputs of length n/2 each (so deg sum < n, satisfying the\n");
    printf("LCH14 convolution theorem). If the additive FFT satisfies the\n");
    printf("convolution theorem, all PASS.\n\n");

    test_one("n=2",   2,   0xA1A1A1A1A1A1A1A1ULL);
    test_one("n=4",   4,   0xA2A2A2A2A2A2A2A2ULL);
    test_one("n=8",   8,   0xA3A3A3A3A3A3A3A3ULL);
    test_one("n=16",  16,  0xA4A4A4A4A4A4A4A4ULL);
    test_one("n=32",  32,  0xA5A5A5A5A5A5A5A5ULL);
    test_one("n=64",  64,  0xA6A6A6A6A6A6A6A6ULL);
    test_one("n=128", 128, 0xA7A7A7A7A7A7A7A7ULL);
    test_one("n=256", 256, 0xA8A8A8A8A8A8A8A8ULL);

    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", g_passed);
    printf("Failed: %d\n", g_failed);
    printf("\nInterpretation:\n");
    printf("  - All PASS: the LCH14 additive FFT does implement the convolution\n");
    printf("    theorem (FFT poly_mul is a drop-in), giving Phase 2b O(n log n)\n");
    printf("    for free. Proceed to Fenger Toeplitz (Phase 3).\n");
    printf("  - Some FAIL: the LCH14 transform is not yet correct; debug the\n");
    printf("    butterfly multipliers s_i(v_j)/s_i(v_i).\n");
    return g_failed > 0 ? 1 : 0;
}