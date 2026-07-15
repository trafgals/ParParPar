/*
 * bench_fenger_large_n.c — focused microbench for the Fenger Toeplitz
 *   pipeline at canonical-ish N values (issue #28 / #26).
 *
 * The general bench_gf64_fenger_vs_cauchy.c sweeps many small sizes; this
 * file isolates the large-N regime where the additive-FFT poly_mul win
 * matters, with line-buffered output and reduced iteration counts so the
 * wall-clock stays bounded.
 *
 * Build:
 *   cd gf64/test
 *   gcc -O3 -march=native -fopenmp -I.. bench_fenger_large_n.c \
 *       ../gf64_fenger.c ../gf64_mpe.c ../gf64_subproduct.c \
 *       ../gf64_barycentric.c ../gf64_invert_ita.c ../gf64_invert.c \
 *       ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c \
 *       ../gf64_poly_mul_toom3.c ../gf64_mul_avx512.c ../gf64_square.c \
 *       ../gf64_single.c -o bench_fenger_large_n
 *
 * Run:
 *   ./bench_fenger_large_n
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
    setvbuf(stdout, NULL, _IOLBF, 0);
    fprintf(stdout, ">>> N=%zu R=%zu B=%zu iters=%d threads=%d  alloc...\n",
            N, R, B, iters, num_threads);

    gf64_t *in         = (gf64_t *)malloc(N * B * sizeof(gf64_t));
    gf64_t *cauchy_out = (gf64_t *)calloc(R * B, sizeof(gf64_t));
    gf64_t *fenger_out = (gf64_t *)calloc(R * B, sizeof(gf64_t));
    if (!in || !cauchy_out || !fenger_out) abort();

    g_rng = 0xDEADBEEF;
    for (size_t i = 0; i < N * B; i++) in[i] = splitmix64_next();

    const uint64_t fi = 0x10000;
    const uint64_t fr = 0x1000000;

    fprintf(stdout, "    warmup...\n");
    cauchy_reference(in, N, B, cauchy_out, R, fi, fr);
    gf64_fenger_matvec(in, N, B, fenger_out, R, fi, fr);

    fprintf(stdout, "    timing cauchy...\n");
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        cauchy_reference(in, N, B, cauchy_out, R, fi, fr);
    }
    double t_cauchy = now_sec() - t0;

    fprintf(stdout, "    timing fenger-1t...\n");
    t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        gf64_fenger_matvec(in, N, B, fenger_out, R, fi, fr);
    }
    double t_fenger_1t = now_sec() - t0;

    double t_fenger_mt = -1.0;
    if (num_threads > 1) {
        fprintf(stdout, "    timing fenger-%dt...\n", num_threads);
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
        fprintf(stdout, "  N=%-5zu R=%-5zu B=%-6zu  cauchy=%7.1f  fenger1t=%7.1f  fenger%dt=%7.1f MB/s\n",
               N, R, B, bw_cauchy, bw_fenger_1t, num_threads, bw_fenger_mt);
    } else {
        fprintf(stdout, "  N=%-5zu R=%-5zu B=%-6zu  cauchy=%7.1f  fenger1t=%7.1f MB/s  speedup=%5.2fx\n",
               N, R, B, bw_cauchy, bw_fenger_1t, t_cauchy / t_fenger_1t);
    }

    free(in); free(cauchy_out); free(fenger_out);
}

int main(void) {
    int ncpu = omp_get_num_procs();
    fprintf(stdout, "Fenger Toeplitz pipeline — large-N focused microbench (issue #28)\n");
    fprintf(stdout, "CPU count = %d\n\n", ncpu);

    /* The canonical-ish workload sizes that the original bench's later
     * entries stalled on; isolated here with line-buffered output so
     * progress is visible. */
    bench_config(8192,  1024, 32,  2, ncpu);
    bench_config(16384, 1024, 16,  1, ncpu);

    fprintf(stdout, "\nDone.\n");
    return 0;
}
