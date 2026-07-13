/*
 * test_gf64_poly_mul_toom3.c — parity test for Toom-Cook 3 poly_mul.
 *
 * Tests gf64_poly_mul_toom3 against schoolbook reference across a grid
 * of sizes that exercise both the schoolbook base case and the recursive
 * Toom-3 path, with multiple polynomials per size for confidence.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);
void gf64_poly_mul_toom3(
    gf64_t *out,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b,
    size_t out_len
);

static uint64_t g_rng = 0xDEADBEEFCAFEBABEULL;
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

static void schoolbook(
    gf64_t *out,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b
) {
    memset(out, 0, (len_a + len_b - 1) * sizeof(gf64_t));
    for (size_t i = 0; i < len_a; i++) {
        for (size_t j = 0; j < len_b; j++) {
            out[i + j] ^= gf64_mul_reference(a[i], b[j]);
        }
    }
}

static int test_one(const char *name, size_t len_a, size_t len_b, size_t out_len, uint64_t seed) {
    g_rng = seed;
    gf64_t *a = (gf64_t *)malloc(len_a * sizeof(gf64_t));
    gf64_t *b = (gf64_t *)malloc(len_b * sizeof(gf64_t));
    for (size_t i = 0; i < len_a; i++) a[i] = splitmix64_next();
    for (size_t i = 0; i < len_b; i++) b[i] = splitmix64_next();

    gf64_t *ref = (gf64_t *)calloc(len_a + len_b, sizeof(gf64_t));
    schoolbook(ref, a, len_a, b, len_b);

    gf64_t *got = (gf64_t *)calloc(out_len, sizeof(gf64_t));
    gf64_poly_mul_toom3(got, a, len_a, b, len_b, out_len);

    int mismatches = 0;
    size_t check = (out_len < len_a + len_b - 1) ? out_len : len_a + len_b - 1;
    for (size_t i = 0; i < check; i++) {
        if (got[i] != ref[i]) {
            if (mismatches < 3) {
                printf("    %s: got[%zu]=0x%016llx ref[%zu]=0x%016llx\n",
                       name, i,
                       (unsigned long long)got[i], i,
                       (unsigned long long)ref[i]);
            }
            mismatches++;
        }
    }

    free(a); free(b); free(ref); free(got);

    if (mismatches == 0) {
        pass(name);
        return 0;
    } else {
        printf("    %s: %d mismatches\n", name, mismatches);
        fail(name);
        return 1;
    }
}

int main(void) {
    printf("Toom-Cook 3 parity test (gf64_poly_mul_toom3)\n");
    printf("==============================================\n\n");

    /* Schoolbook base case (n <= 192). */
    test_one("n=1",   1,   1,   1,   0xAAAA);
    test_one("n=10",  10,  10,  19,  0xAAAB);
    test_one("n=64",  64,  64,  127, 0xAAAC);
    test_one("n=128", 128, 128, 255, 0xAAAD);
    test_one("n=192", 192, 192, 383, 0xAAAE);

    /* Recursive Toom-3 path (n > 192). */
    test_one("n=193", 193, 193, 385, 0xBBB0);
    test_one("n=256", 256, 256, 511, 0xBBB1);
    test_one("n=384", 384, 384, 767, 0xBBB2);
    test_one("n=512", 512, 512, 1023, 0xBBB3);
    test_one("n=1024", 1024, 1024, 2047, 0xBBB4);

    /* Asymmetric sizes. */
    test_one("n_a=300, n_b=100", 300, 100, 399, 0xCCC0);
    test_one("n_a=500, n_b=200", 500, 200, 699, 0xCCC1);
    test_one("n_a=1000, n_b=300", 1000, 300, 1299, 0xCCC2);

    /* Edge: out_len smaller than full convolution. */
    test_one("n=500 truncated to out=100", 500, 500, 100, 0xDDD0);
    test_one("n=500 truncated to out=500", 500, 500, 500, 0xDDD1);

    /* Edge: zero in a or b. */
    gf64_t zero_buf[100] = {0};
    gf64_t one_a[100];
    for (size_t i = 0; i < 100; i++) one_a[i] = (gf64_t)(i + 1);
    gf64_t *got = (gf64_t *)calloc(199, sizeof(gf64_t));
    gf64_poly_mul_toom3(got, zero_buf, 100, one_a, 100, 199);
    int all_zero = 1;
    for (size_t i = 0; i < 199; i++) if (got[i] != 0) all_zero = 0;
    if (all_zero) pass("a=0 -> all zero"); else fail("a=0 -> all zero");
    free(got);

    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", g_passed);
    printf("Failed: %d\n", g_failed);
    return g_failed > 0 ? 1 : 0;
}