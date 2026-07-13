/*
 * bench_poly_mul_toom3_vs_karatsuba.c — microbench Karatsuba vs Toom-3.
 *
 * Compares wall-clock time of gf64_poly_mul_karatsuba vs
 * gf64_poly_mul_toom3 over a grid of sizes. Toom-3 has complexity
 * O(n^1.465) vs Karatsuba's O(n^1.585) — should win at large n.
 *
 * Build & run:
 *   cd gf64/test
 *   gcc -O3 -march=native -I.. bench_poly_mul_toom3_vs_karatsuba.c \
 *       ../gf64_poly_mul_toom3.c ../gf64_poly_mul_karatsuba.c \
 *       ../gf64_single.c -o bench_poly_mul_toom3_vs_karatsuba
 *   ./bench_poly_mul_toom3_vs_karatsuba
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

void gf64_poly_mul_karatsuba(gf64_t *out, const gf64_t *a, size_t len_a,
                             const gf64_t *b, size_t len_b, size_t out_len);
void gf64_poly_mul_toom3(gf64_t *out, const gf64_t *a, size_t len_a,
                         const gf64_t *b, size_t len_b, size_t out_len);

static uint64_t g_rng = 0x1234567890ABCDEFULL;
static uint64_t splitmix64_next(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static double bench(int which, size_t n, int iters) {
    gf64_t *a = (gf64_t *)malloc(n * sizeof(gf64_t));
    gf64_t *b = (gf64_t *)malloc(n * sizeof(gf64_t));
    g_rng = 0xDEADBEEF;
    for (size_t i = 0; i < n; i++) { a[i] = splitmix64_next(); b[i] = splitmix64_next(); }
    gf64_t *out = (gf64_t *)calloc(2 * n - 1, sizeof(gf64_t));

    /* Warmup. */
    for (int i = 0; i < 2; i++) {
        if (which == 0) gf64_poly_mul_karatsuba(out, a, n, b, n, 2 * n - 1);
        else            gf64_poly_mul_toom3(out, a, n, b, n, 2 * n - 1);
    }

    double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        if (which == 0) gf64_poly_mul_karatsuba(out, a, n, b, n, 2 * n - 1);
        else            gf64_poly_mul_toom3(out, a, n, b, n, 2 * n - 1);
    }
    double t1 = now_sec();
    free(a); free(b); free(out);
    return (t1 - t0) / iters;
}

int main(void) {
    printf("Toom-3 vs Karatsuba poly_mul microbench\n");
    printf("========================================\n\n");
    printf("%8s | %12s | %12s | %8s\n", "n", "karatsuba_s", "toom3_s", "speedup");
    printf("---------+--------------+--------------+----------\n");

    size_t sizes[] = { 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096 };
    int iters[] = { 50000, 20000, 10000, 5000, 2000, 1000, 500, 200, 100, 50, 25, 10 };

    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    for (int i = 0; i < nsizes; i++) {
        size_t n = sizes[i];
        double tk = bench(0, n, iters[i]);
        double t3 = bench(1, n, iters[i]);
        printf("%8zu | %12.4e | %12.4e | %7.2fx\n",
               n, tk, t3, tk / t3);
    }

    printf("\nToom-3 should win above ~256 (5 sub-products vs 3, but smaller sub-products).\n");
    return 0;
}