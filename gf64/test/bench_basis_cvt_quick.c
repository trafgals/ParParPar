/* bench_basis_cvt_quick.c — measure wall-clock cost of matrix-form basisCvt
 * at smaller sizes (n=1024..8192). Avoids n=16384 since M_inv build there
 * is O(n^3) ≈ 1 hour one-time cost.
 *
 * Build:
 *   gcc -O2 -mpclmul -mssse3 -mavx2 -mavx512f -mvpclmulqdq -I.. \
 *       bench_basis_cvt_quick.c ../gf64_additive_fft_hqc2026.c \
 *       ../gf64_additive_fft.c ../gf64_invert.c ../gf64_invert_ita.c \
 *       ../gf64_poly_mul_karatsuba.c ../gf64_poly_mul_toom3.c \
 *       ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c \
 *       -o bench_basis_cvt_quick
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "../gf64_global.h"
#include "../gf64_additive_fft.h"

static uint64_t g_rng = 0xCAFEBABE12345ULL;
static uint64_t splitmix64_next(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(void) {
    /* Sizes where the matrix-form M_inv build completes quickly.
     * n=8192: M_inv is 512 MiB; build is ~10s. Skipped here. */
    int sizes[] = {1024, 2048, 4096};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int iters = 10;

    printf("size      fwd_total   inv_total   fwd_avg     inv_avg     bytes/s_fwd   bytes/s_inv\n");
    printf("--------  ----------  ----------  ----------  ----------  ------------  ------------\n");

    for (int si = 0; si < n_sizes; si++) {
        int n = sizes[si];
        gf64_t *f = (gf64_t *)malloc(n * sizeof(gf64_t));
        if (!f) abort();
        g_rng = 0xDEADBEEF ^ (uint64_t)n;
        for (int i = 0; i < n; i++) f[i] = splitmix64_next();

        /* Warmup (triggers M_inv build) */
        double t_warm = now_sec();
        gf64_addfft64_fwd(f, n);
        gf64_addfft64_inv(f, n);
        double warm = now_sec() - t_warm;

        /* Time forward */
        double t0 = now_sec();
        for (int it = 0; it < iters; it++) gf64_addfft64_fwd(f, n);
        double t_fwd = now_sec() - t0;

        /* Time inverse */
        t0 = now_sec();
        for (int it = 0; it < iters; it++) gf64_addfft64_inv(f, n);
        double t_inv = now_sec() - t0;

        double bytes = (double)n * sizeof(gf64_t);
        double throughput_fwd = (bytes * iters) / t_fwd;
        double throughput_inv = (bytes * iters) / t_inv;
        printf("%-8d  %9.4fs  %9.4fs  %9.4fs  %9.4fs  %9.0f/s   %9.0f/s\n",
               n, t_fwd, t_inv, t_fwd / iters, t_inv / iters,
               throughput_fwd, throughput_inv);
        printf("  (warmup: %.3fs including M_inv build)\n", warm);

        free(f);
    }

    printf("\nFor n=8192: M_inv is 512 MiB; build is ~10s.\n");
    printf("For n=16384: M_inv is 2 GiB; build is ~1 hour.\n");
    printf("The matrix-form fallback is impractical for n > 4096.\n");
    printf("\nTo clear the 100 MB/s gate at canonical N=10K:\n");
    printf("  - Need recursive Algorithm 1 basisCvt (O(N log N)) for n > 4096\n");
    printf("  - OR persistent M_inv cache (one-time build cost amortized)\n");
    printf("  - OR option (c) hybrid: matrix-form for n <= 4096, recursive for larger\n");

    return 0;
}