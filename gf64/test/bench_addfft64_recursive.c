/*
 * bench_addfft64_recursive.c — measure wall-clock throughput of
 *   gf64_addfft64_fwd_recursive / _inv_recursive (FIX-3a option (c) hybrid)
 * and gf64_addfft64_poly_mul_recursive (the realistic Fenger pipeline
 * primitive). Sizes 1024..16384.
 *
 * Reference for the matrix-form path is bench_basis_cvt_quick.c which
 * measures 0.44 MB/s at n=4096 (250x gap to gate).
 *
 * Build & run (gcc inside WSL):
 *   cd gf64/test && gcc -O2 -mpclmul -mssse3 -mavx2 -mavx512f -mvpclmulqdq \
 *       -I.. bench_addfft64_recursive.c ../gf64_additive_fft_hqc2026.c \
 *       ../gf64_invert.c ../gf64_invert_ita.c ../gf64_poly_mul_karatsuba.c \
 *       ../gf64_poly_mul_toom3.c ../gf64_mul_avx512.c ../gf64_square.c \
 *       ../gf64_single.c -o bench_addfft64_recursive
 *   ./bench_addfft64_recursive
 */

#include "../gf64_global.h"
#include "../gf64_additive_fft.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

static uint64_t g_rng = 0xCAFEBABE12345ULL;
static uint64_t splitmix64_next(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void fill_random(gf64_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = splitmix64_next();
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(void) {
    int sizes[] = {1024, 2048, 4096, 8192, 16384};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int iters = 20;

    printf("Throughput: gf64_addfft64_fwd_recursive / _inv_recursive / _poly_mul_recursive\n");
    printf("size      fwd_recur   fwd_MB/s   inv_recur   inv_MB/s   mul_recur  mul_MB/s\n");
    printf("--------  ----------  ---------  ----------  ---------  ---------  ---------\n");

    for (int si = 0; si < n_sizes; si++) {
        int n = sizes[si];
        gf64_t *fwd_buf_r = (gf64_t *)malloc((size_t)n * sizeof(gf64_t));
        gf64_t *a_buf = (gf64_t *)malloc(((size_t)n / 2) * sizeof(gf64_t));
        gf64_t *b_buf = (gf64_t *)malloc(((size_t)n / 2) * sizeof(gf64_t));
        gf64_t *out_buf = (gf64_t *)malloc(((size_t)n - 1) * sizeof(gf64_t));
        if (!fwd_buf_r || !a_buf || !b_buf || !out_buf) abort();

        g_rng = 0xDEADBEEF ^ (uint64_t)n;
        fill_random(fwd_buf_r, n);
        fill_random(a_buf, n / 2);
        fill_random(b_buf, n / 2);

        double bytes_addfft = (double)n * sizeof(gf64_t);
        double bytes_mul = (double)(n - 1) * sizeof(gf64_t);

        /* Recursive forward */
        double t0 = now_sec();
        for (int it = 0; it < iters; it++) gf64_addfft64_fwd_recursive(fwd_buf_r, (size_t)n);
        double t_fwd_r = now_sec() - t0;
        /* Recursive inverse */
        t0 = now_sec();
        for (int it = 0; it < iters; it++) gf64_addfft64_inv_recursive(fwd_buf_r, (size_t)n);
        double t_inv_r = now_sec() - t0;
        /* Polynomial multiplication via recursive addFFT
         * (closest primitive to what Fenger uses) */
        t0 = now_sec();
        for (int it = 0; it < iters; it++)
            gf64_addfft64_poly_mul_recursive(out_buf,
                a_buf, (size_t)(n/2), b_buf, (size_t)(n/2), (size_t)(n - 1));
        double t_mul_r = now_sec() - t0;

        double mb_fwd_r = bytes_addfft / 1e6 / (t_fwd_r / iters);
        double mb_inv_r = bytes_addfft / 1e6 / (t_inv_r / iters);
        double mb_mul_r = bytes_mul / 1e6 / (t_mul_r / iters);
        printf("%-8d  %9.4fs   %9.0f   %9.4fs   %9.0f   %8.4fs   %8.0f\n",
               n, t_fwd_r / iters, mb_fwd_r, t_inv_r / iters, mb_inv_r,
               t_mul_r / iters, mb_mul_r);

        free(fwd_buf_r); free(a_buf); free(b_buf); free(out_buf);
    }
    printf("\nReference: matrix-form BasisCvt at n=4096 measured 0.44 MB/s in\n");
    printf("bench_basis_cvt_quick.c. Recursive Algorithm 1 is now the\n");
    printf("hot-path; matrix-form path is preserved for bit-exact parity.\n");
    return 0;
}