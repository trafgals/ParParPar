/*
 * probe_hqc2026_timing.c — isolate where the time goes at large N.
 *
 * Measures wall-clock for each phase:
 *   1. gf64_addfft64_fwd / inv  (the HQC Alg 2 butterflies) at various N
 *   2. gf64_addfft64_poly_mul  (forward + inv + pointwise + inv)
 *   3. gf64_subproduct_tree_build (the Fenger prep)
 *
 * Each fprintf is followed by fflush so output appears live in the
 * background-task capture.
 *
 * Build:
 *   gcc -O3 -march=native -I.. probe_hqc2026_timing.c \
 *       ../gf64_additive_fft.c ../gf64_additive_fft_hqc2026.c \
 *       ../gf64_invert.c ../gf64_invert_ita.c ../gf64_poly_mul_karatsuba.c \
 *       ../gf64_poly_mul_toom3.c ../gf64_mul_avx512.c ../gf64_square.c \
 *       ../gf64_single.c ../gf64_subproduct.c ../gf64_mpe.c \
 *       -o probe_hqc2026_timing
 */

#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../gf64_global.h"

extern void gf64_addfft64_fwd(gf64_t *arr, size_t n);
extern void gf64_addfft64_inv(gf64_t *arr, size_t n);
extern void gf64_addfft64_poly_mul(
    gf64_t *out, const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b, size_t out_len
);

extern void gf64_subproduct_tree_build(
    gf64_t *tree[], size_t *tlen, size_t *sub_r_at,
    const gf64_t *points, size_t n,
    gf64_t **scratch, size_t *scratch_len,
    gf64_t **rev_scratch, size_t *rev_scratch_len
);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void say(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
    fputc('\n', stdout);
    fflush(stdout);
}

static uint64_t g_rng = 0x1234567890ABCDEFULL;
static uint64_t splitmix64_next(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void time_fft(int n, int iters) {
    gf64_t *in  = (gf64_t *)malloc((size_t)n * sizeof(gf64_t));
    gf64_t *out = (gf64_t *)malloc((size_t)n * sizeof(gf64_t));
    if (!in || !out) abort();
    for (int i = 0; i < n; i++) in[i] = splitmix64_next();

    say("  fft N=%d warmup...", n);
    memcpy(out, in, (size_t)n * sizeof(gf64_t));
    gf64_addfft64_fwd(out, n);

    say("  fft N=%d timing %d iters...", n, iters);
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        memcpy(out, in, (size_t)n * sizeof(gf64_t));
        gf64_addfft64_fwd(out, n);
    }
    double t = now_sec() - t0;
    say("    gf64_addfft64_fwd  N=%-5d  %8.4f ms/call  (%d iters)",
        n, t * 1000.0 / iters, iters);

    t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        memcpy(out, in, (size_t)n * sizeof(gf64_t));
        gf64_addfft64_inv(out, n);
    }
    t = now_sec() - t0;
    say("    gf64_addfft64_inv  N=%-5d  %8.4f ms/call  (%d iters)",
        n, t * 1000.0 / iters, iters);

    free(in); free(out);
}

static void time_poly_mul(int n, int iters) {
    gf64_t *a = (gf64_t *)malloc((size_t)n * sizeof(gf64_t));
    gf64_t *b = (gf64_t *)malloc((size_t)n * sizeof(gf64_t));
    gf64_t *o = (gf64_t *)malloc((size_t)(2*n - 1) * sizeof(gf64_t));
    if (!a || !b || !o) abort();
    for (int i = 0; i < n; i++) { a[i] = splitmix64_next(); b[i] = splitmix64_next(); }

    say("  mul N=%d warmup...", n);
    gf64_addfft64_poly_mul(o, a, n, b, n, 2*n-1);

    say("  mul N=%d timing %d iters...", n, iters);
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) gf64_addfft64_poly_mul(o, a, n, b, n, 2*n-1);
    double t = now_sec() - t0;
    say("    gf64_addfft64_poly_mul N=%-5d  %8.4f ms/call  (%d iters)",
        n, t * 1000.0 / iters, iters);

    free(a); free(b); free(o);
}

static void time_subproduct(size_t n, int iters) {
    gf64_t *pts = (gf64_t *)malloc(n * sizeof(gf64_t));
    if (!pts) abort();
    for (size_t i = 0; i < n; i++) pts[i] = (gf64_t)(0x10000ULL + (uint64_t)i);

    gf64_t **tree = NULL;
    size_t  *tlen = NULL;
    size_t  *sub_r_at = NULL;
    gf64_t  *scratch = NULL, *rev_scratch = NULL;
    size_t   scratch_len = 0, rev_scratch_len = 0;

    say("  subprod N=%zu warmup...", n);
    double t0 = now_sec();
    gf64_subproduct_tree_build(
        tree, tlen, sub_r_at, pts, n,
        &scratch, &scratch_len, &rev_scratch, &rev_scratch_len
    );
    double t = now_sec() - t0;
    say("    gf64_subproduct_tree_build  N=%-5zu  %8.4f ms (cold)", n, t * 1000.0);

    if (iters > 1) {
        t0 = now_sec();
        for (int i = 1; i < iters; i++) {
            gf64_subproduct_tree_build(
                tree, tlen, sub_r_at, pts, n,
                &scratch, &scratch_len, &rev_scratch, &rev_scratch_len
            );
        }
        t = now_sec() - t0;
        say("    gf64_subproduct_tree_build  N=%-5zu  %8.4f ms/call  (%d iters)",
            n, t * 1000.0 / (iters - 1), iters - 1);
    }

    free(pts);
}

int main(void) {
    say("=== Additive FFT micro-timing (HQC Alg 2) ===");

    say("");
    say("Forward/inverse FFT:");
    time_fft(2,    100000);
    time_fft(4,    100000);
    time_fft(8,    10000);
    time_fft(16,   10000);
    time_fft(32,   10000);
    time_fft(64,   10000);
    time_fft(128,  10000);
    time_fft(256,  5000);
    time_fft(512,  2000);
    time_fft(1024, 1000);
    time_fft(2048, 500);
    time_fft(4096, 100);
    time_fft(8192, 20);

    say("");
    say("Poly mul:");
    time_poly_mul(2,    100000);
    time_poly_mul(4,    100000);
    time_poly_mul(8,    10000);
    time_poly_mul(16,   10000);
    time_poly_mul(32,   10000);
    time_poly_mul(64,   10000);
    time_poly_mul(128,  10000);
    time_poly_mul(256,  5000);
    time_poly_mul(512,  2000);
    time_poly_mul(1024, 1000);
    time_poly_mul(2048, 500);
    time_poly_mul(4096, 100);
    time_poly_mul(8192, 20);

    say("");
    say("Subproduct tree:");
    time_subproduct(2,    100);
    time_subproduct(4,    100);
    time_subproduct(8,    100);
    time_subproduct(16,   50);
    time_subproduct(32,   20);
    time_subproduct(64,   10);
    time_subproduct(128,  10);
    time_subproduct(256,  5);
    time_subproduct(512,  3);
    time_subproduct(1024, 3);
    time_subproduct(2048, 2);
    time_subproduct(4096, 2);
    time_subproduct(8192, 1);

    say("");
    say("Done.");
    return 0;
}