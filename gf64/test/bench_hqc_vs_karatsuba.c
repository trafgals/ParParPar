/*
 * /tmp/bench_hqc_vs_karatsuba.c — benchmark HQC FFT (scalar + AVX-512)
 *   vs Karatsuba for polynomial multiplication at canonical sizes.
 *
 * Sizes tested: 16, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096,
 * 8192, 16384.
 * (Smaller sizes use schoolbook inside Karatsuba; sizes in
 *  {16, 64, ..., 8192, 16384} fall back to matrix-form in HQC FFT.)
 *
 * Run: /tmp/bench_hqc_vs_karatsuba
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "/mnt/c/code/trafgals/parparpar/gf64/gf64_additive_fft.h"

extern void gf64_poly_mul_karatsuba(
    uint64_t *out, const uint64_t *a, size_t la,
    const uint64_t *b, size_t lb, size_t out_len);

static uint64_t state = 0xc0ffee01;
static uint64_t splitmix64(void) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double time_one_poly_mul_recursive(size_t n_actual, uint64_t *a, uint64_t *b,
                                          uint64_t *out, uint64_t *scratch,
                                          int use_avx512, int iters) {
    size_t n_pad = 1;
    while (n_pad < 2 * n_actual - 1) n_pad <<= 1;
    size_t sw = gf64_addfft64_poly_mul_recursive_scratch_words(n_pad);
    for (int i = 0; i < 3; i++) {
        if (use_avx512)
            gf64_addfft64_poly_mul_recursive_scratch_avx512(
                out, a, n_actual, b, n_actual, n_actual, scratch, sw);
        else
            gf64_addfft64_poly_mul_recursive_scratch(
                out, a, n_actual, b, n_actual, n_actual, scratch, sw);
    }
    clock_t t0 = clock();
    for (int i = 0; i < iters; i++) {
        if (use_avx512)
            gf64_addfft64_poly_mul_recursive_scratch_avx512(
                out, a, n_actual, b, n_actual, n_actual, scratch, sw);
        else
            gf64_addfft64_poly_mul_recursive_scratch(
                out, a, n_actual, b, n_actual, n_actual, scratch, sw);
    }
    clock_t t1 = clock();
    return (double)(t1 - t0) * 1000.0 / CLOCKS_PER_SEC / iters;
}

static double time_one_karatsuba(size_t n, uint64_t *a, uint64_t *b,
                                 uint64_t *out, int iters) {
    for (int i = 0; i < 3; i++)
        gf64_poly_mul_karatsuba(out, a, n, b, n, n);
    clock_t t0 = clock();
    for (int i = 0; i < iters; i++)
        gf64_poly_mul_karatsuba(out, a, n, b, n, n);
    clock_t t1 = clock();
    return (double)(t1 - t0) * 1000.0 / CLOCKS_PER_SEC / iters;
}

static double bench_one(size_t n, int iters) {
    uint64_t *a = malloc(n * sizeof(uint64_t));
    uint64_t *b = malloc(n * sizeof(uint64_t));
    uint64_t *out = malloc((2*n) * sizeof(uint64_t));
    for (size_t i = 0; i < n; i++) { a[i] = splitmix64(); b[i] = splitmix64(); }
    double ms_k = time_one_karatsuba(n, a, b, out, iters);
    size_t n_pad = 1;
    while (n_pad < 2 * n - 1) n_pad <<= 1;
    size_t sw = gf64_addfft64_poly_mul_recursive_scratch_words(n_pad);
    uint64_t *scratch = malloc(sw * sizeof(uint64_t));
    double ms_hs = time_one_poly_mul_recursive(n, a, b, out, scratch, 0, iters);
    double ms_ha = time_one_poly_mul_recursive(n, a, b, out, scratch, 1, iters);
    printf("%-7zu %7.3f ms %7.3f ms %7.3f ms %5.2fx  %5.2fx\n",
           n, ms_k, ms_hs, ms_ha, ms_k / ms_ha, ms_hs / ms_ha);
    free(a); free(b); free(out); free(scratch);
    return ms_k;
}

int main(void) {
    printf("%-7s %-10s %-10s %-10s %-7s %-7s\n",
           "n", "karatsuba", "hqc-scalar", "hqc-avx512", "vs-kara", "avx-gain");
    printf("%-7s %-10s %-10s %-10s %-7s %-7s\n",
           "-------", "----------", "----------", "----------", "-------", "-------");

    /* Small crossover sweep */
    size_t small[] = {16, 32, 48, 64, 96, 128, 192};
    for (size_t i = 0; i < sizeof(small)/sizeof(small[0]); i++)
        bench_one(small[i], 10000);

    printf("\n--- canonical sizes ---\n");
    /* Canonical sizes */
    size_t big[] = {256, 512, 1024, 2048, 4096, 8192, 16384};
    for (size_t i = 0; i < sizeof(big)/sizeof(big[0]); i++)
        bench_one(big[i], (big[i] <= 512) ? 5000 : (big[i] <= 4096) ? 500 : 100);

    return 0;
}
