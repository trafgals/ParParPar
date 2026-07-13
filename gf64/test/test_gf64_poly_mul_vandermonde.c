/*
 * test_gf64_poly_mul_vandermonde.c — bit-exact parity test for the
 * Vandermonde-FFT-based gf64_poly_mul_vandermonde against schoolbook
 * reference multiplication.
 *
 * Build & run from gf64/test/:
 *   gcc -O2 -march=native -I.. test_gf64_poly_mul_vandermonde.c \
 *       ../gf64_additive_fft.c ../gf64_additive_fft_vandermonde.c \
 *       ../gf64_poly_mul_karatsuba.c ../gf64_mpe.c \
 *       ../gf64_subproduct.c ../gf64_invert_ita.c \
 *       ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c \
 *       -o test_gf64_poly_mul_vandermonde \
 *       && ./test_gf64_poly_mul_vandermonde
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../gf64_additive_fft.h"
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

static uint64_t g_rng = 0xBA5EBA5EBA5EBA5EULL;
static uint64_t splitmix64_next(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

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

static int g_pass = 0, g_fail = 0;
static void pass(const char *t) { printf("  PASS: %s\n", t); g_pass++; }
static void fail(const char *t) { printf("  FAIL: %s\n", t); g_fail++; }

static int test_one(const char *t, size_t da, size_t db, uint64_t seed) {
    g_rng = seed;
    gf64_t *a = (gf64_t *)malloc((da + 1) * sizeof(gf64_t));
    gf64_t *b = (gf64_t *)malloc((db + 1) * sizeof(gf64_t));
    for (size_t i = 0; i <= da; i++) a[i] = splitmix64_next();
    for (size_t i = 0; i <= db; i++) b[i] = splitmix64_next();

    gf64_t *ab_ref = (gf64_t *)malloc((da + db + 1) * sizeof(gf64_t));
    schoolbook(ab_ref, a, da + 1, b, db + 1);

    gf64_t *ab_fft = (gf64_t *)malloc((da + db + 1) * sizeof(gf64_t));
    memset(ab_fft, 0, (da + db + 1) * sizeof(gf64_t));
    gf64_poly_mul_vandermonde(ab_fft, a, da, b, db);

    int mismatches = 0;
    for (size_t i = 0; i <= da + db; i++) {
        if (ab_fft[i] != ab_ref[i]) {
            if (mismatches < 3) {
                printf("    [%s] a*b[%zu] fft=0x%016llx ref=0x%016llx\n",
                       t, i, (unsigned long long)ab_fft[i],
                       (unsigned long long)ab_ref[i]);
            }
            mismatches++;
        }
    }

    free(a); free(b); free(ab_ref); free(ab_fft);
    if (mismatches == 0) { pass(t); return 0; }
    printf("    %s: %d mismatches\n", t, mismatches);
    fail(t);
    return 1;
}

int main(void) {
    printf("Vandermonde-FFT poly_mul parity test\n");
    printf("====================================\n\n");
    test_one("deg_a=0 deg_b=0",   0,   0,   0x0000000000000001ULL);
    test_one("deg_a=3 deg_b=3",   3,   3,   0x0000000000000002ULL);
    test_one("deg_a=5 deg_b=2",   5,   2,   0x0000000000000003ULL);
    test_one("deg_a=15 deg_b=15", 15,  15,  0x0000000000000004ULL);
    test_one("deg_a=31 deg_b=31", 31,  31,  0x0000000000000005ULL);
    test_one("deg_a=63 deg_b=63", 63,  63,  0x0000000000000006ULL);
    test_one("deg_a=127 deg_b=127", 127, 127, 0x0000000000000007ULL);
    test_one("deg_a=255 deg_b=255", 255, 255, 0x0000000000000008ULL);
    test_one("deg_a=1 deg_b=512", 1,   512, 0x0000000000000009ULL);
    test_one("deg_a=512 deg_b=1", 512, 1,   0x00000000000000A0ULL);
    test_one("deg_a=200 deg_b=300", 200, 300, 0x00000000000000B0ULL);

    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", g_pass);
    printf("Failed: %d\n", g_fail);
    return g_fail > 0 ? 1 : 0;
}
