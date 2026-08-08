/*
 * bench_gf64_fenger_vs_cauchy.c — microbench: Fenger pipeline vs explicit
 *   Cauchy matvec at canonical PAR3 shapes (issue #28).
 *
 * Compares wall-clock throughput of:
 *
 *   cauchy_reference: out[r][w] = XOR_c in[c][w] / (y_r XOR x_c)
 *                     (explicit O(N) per output)
 *
 *   gf64_fenger_matvec: Bostan-Schost top-down Fenger pipeline (issue #28)
 *
 * Also exercises the multi-threaded execute path via OpenMP when B is
 * large enough to benefit.
 *
 * Build & run:
 *   cd gf64/test
 *   gcc -O3 -march=native -fopenmp -I.. bench_gf64_fenger_vs_cauchy.c \
 *       ../gf64_fenger.c ../gf64_mpe.c ../gf64_subproduct.c \
 *       ../gf64_barycentric.c ../gf64_invert_ita.c ../gf64_invert.c \
 *       ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c \
 *       ../gf64_poly_mul_toom3.c ../gf64_mul_avx512.c ../gf64_square.c \
 *       ../gf64_single.c -o bench_gf64_fenger_vs_cauchy
 *   ./bench_gf64_fenger_vs_cauchy
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "../gf64_global.h"
#include "../gf64_fenger.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);
extern gf64_t gf64_inverse(gf64_t a);

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

/* ----- Explicit Cauchy matrix-vector product. */
static void cauchy_reference(
    const gf64_t *in, size_t N, size_t B,
    gf64_t *out, size_t R,
    uint64_t firstInput, uint64_t firstRecovery)
{
    for (size_t r = 0; r < R; r++) {
        gf64_t yr = (gf64_t)(firstRecovery + (uint64_t)r);
        gf64_t *out_row = out + r * B;
        for (size_t w = 0; w < B; w++) out_row[w] = 0;
        for (size_t c = 0; c < N; c++) {
            gf64_t xc = (gf64_t)(firstInput + (uint64_t)c);
            gf64_t denom = gf64_inverse(yr ^ xc);
            const gf64_t *in_row = in + c * B;
            for (size_t w = 0; w < B; w++) {
                out_row[w] ^= gf64_mul_reference(in_row[w], denom);
            }
        }
    }
}

static void bench_config(size_t N, size_t R, size_t B, int iters, int num_threads) {
    gf64_t *in         = (gf64_t *)malloc(N * B * sizeof(gf64_t));
    gf64_t *cauchy_out = (gf64_t *)calloc(R * B, sizeof(gf64_t));
    gf64_t *fenger_out = (gf64_t *)calloc(R * B, sizeof(gf64_t));
    if (!in || !cauchy_out || !fenger_out) abort();

    g_rng = 0xDEADBEEF;
    for (size_t i = 0; i < N * B; i++) in[i] = splitmix64_next();

    const uint64_t fi = 0x10000;
    const uint64_t fr = 0x1000000;

    /* Warmup. */
    cauchy_reference(in, N, B, cauchy_out, R, fi, fr);
    gf64_fenger_matvec(in, N, B, fenger_out, R, fi, fr);

    double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        cauchy_reference(in, N, B, cauchy_out, R, fi, fr);
    }
    double t_cauchy = now_sec() - t0;

    /* Fenger single-thread. */
    t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        gf64_fenger_matvec(in, N, B, fenger_out, R, fi, fr);
    }
    double t_fenger_1t = now_sec() - t0;

    /* Fenger multi-thread (OpenMP-sharded over B). */
    double t_fenger_mt = -1.0;
    if (num_threads > 1) {
        gf64_fenger_ctx *ctx = gf64_fenger_prepare(fi, fr, N, R);
        omp_set_num_threads(num_threads);
        const size_t chunk = (B + (size_t)num_threads - 1) / (size_t)num_threads;

        t0 = now_sec();
        for (int i = 0; i < iters; i++) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                size_t w_start = (size_t)tid * chunk;
                size_t w_end   = w_start + chunk;
                if (w_end > B) w_end = B;
                if (w_start < B) {
                    gf64_fenger_execute(ctx, in, B, fenger_out, w_start, w_end);
                }
            }
        }
        t_fenger_mt = now_sec() - t0;
        gf64_fenger_release(ctx);
    }

    const double bytes_total = (double)(N * B + R * B) * sizeof(gf64_t);
    const double bw_cauchy = bytes_total / t_cauchy / 1e6;
    const double bw_fenger_1t = bytes_total / t_fenger_1t / 1e6;
    double bw_fenger_mt = 0.0;
    if (t_fenger_mt > 0) bw_fenger_mt = bytes_total / t_fenger_mt / 1e6;

    if (t_fenger_mt > 0) {
        printf("  N=%-5zu R=%-5zu B=%-6zu  cauchy=%7.1f  fenger1t=%7.1f  fenger%dt=%7.1f MB/s\n",
               N, R, B, bw_cauchy, bw_fenger_1t, num_threads, bw_fenger_mt);
    } else {
        printf("  N=%-5zu R=%-5zu B=%-6zu  cauchy=%7.1f  fenger1t=%7.1f MB/s  speedup=%5.2fx\n",
               N, R, B, bw_cauchy, bw_fenger_1t, t_cauchy / t_fenger_1t);
    }

    free(in); free(cauchy_out); free(fenger_out);
}

int main(void) {
    int ncpu = omp_get_num_procs();
    printf("Fenger Toeplitz pipeline — microbench (issue #28)\n");
    printf("Wall-clock throughput; CPU count = %d\n\n", ncpu);

    bench_config(8,   8,   4096, 20, ncpu);
    bench_config(16,  16,  4096, 20, ncpu);
    bench_config(32,  32,  4096, 20, ncpu);
    bench_config(64,  64,  4096, 10, ncpu);
    bench_config(128, 128, 4096,  5, ncpu);
    bench_config(256, 64,  4096,  5, ncpu);
    bench_config(512, 256, 2048,  3, ncpu);
    bench_config(1024, 256, 1024, 3, ncpu);

    /* Larger-N sweep to expose the asymptotic regime. */
    bench_config(256, 128, 256, 10, ncpu);
    bench_config(512, 128, 256, 10, ncpu);
    bench_config(1024, 256, 512, 5, ncpu);
    bench_config(2048, 256, 256, 3, ncpu);
    bench_config(2048, 512, 256, 3, ncpu);

    /* Larger-N sweep to expose the additive FFT win. */
    bench_config(2048, 1024, 64, 3, ncpu);
    bench_config(2048, 2048, 32, 2, ncpu);

    /* Canonical-ish workload: N=8K data slices (closest power-of-2 to the
     * canonical 10K that the subproduct tree builder accepts), R=1K recovery,
     * scaled-down B for microbench RAM. */
    bench_config(8192,  1024, 32,  2, ncpu);
    bench_config(16384, 1024, 16,  1, ncpu);
    bench_config(16384, 4096, 16,  1, ncpu);
    bench_config(65536, 1024, 8,   1, ncpu);
    bench_config(65536, 4096, 4,   1, ncpu);

    printf("\nDone.\n");
    return 0;
}
