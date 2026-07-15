/*
 * probe_addfft64_recursive.c — verify the FIX-3a recursive HQC 2026 addFFT
 *   entry points (gf64_addfft64_fwd_recursive / _inv_recursive /
 *   _poly_mul_recursive).
 *
 * Two probes:
 *
 *   1) ROUND-TRIP at sizes 2..16384 (powers of 2):
 *        inv_recursive(fwd_recursive(f, n), n) == f
 *      The recursive BasisCvt is the polyeval cvt port; round-trip PASS
 *      was already verified at 2..4096 in
 *      gf64/test/probe_basis_cvt_polyeval.c. This extends the range to
 *      16384 to cover the canonical PAR3 workload (n = 10K..16K).
 *
 *   2) CONVOLUTION THEOREM at sizes 4, 8, 16, 32, ..., 4096:
 *        inv_recursive(fwd_recursive(a, n) . fwd_recursive(b, n), n) == a * b
 *      (pointwise-multiply in the recursive-novelpoly basis = poly
 *      convolution in the monomial basis, for deg(a), deg(b) < n/2).
 *      Note: the *recursive* family has its own BasisCvt convention; the
 *      resulting poly_mul is bit-exact to the schoolbook poly_mul up to a
 *      (well-defined) constant offset documented in
 *      PHASE_2c_FINDINGS_2026-07-15.md.
 *
 * Build & run (gcc inside WSL — see CLAUDE.md):
 *   cd gf64/test && gcc -O2 -mpclmul -mssse3 -mavx2 -mavx512f -mvpclmulqdq \
 *       -I.. probe_addfft64_recursive.c ../gf64_additive_fft_hqc2026.c \
 *       ../gf64_invert.c ../gf64_invert_ita.c ../gf64_poly_mul_karatsuba.c \
 *       ../gf64_poly_mul_toom3.c ../gf64_mul_avx512.c ../gf64_square.c \
 *       ../gf64_single.c -o probe_addfft64_recursive
 *   ./probe_addfft64_recursive
 */

#include "../gf64_global.h"
#include "../gf64_additive_fft.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

static int check_eq(const char *tag, const gf64_t *a, const gf64_t *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            fprintf(stderr, "%s mismatch at i=%d: got %016llx, want %016llx\n",
                    tag, i, (unsigned long long)a[i], (unsigned long long)b[i]);
            return 0;
        }
    }
    return 1;
}

/* Probe 1: round-trip across the full canonical grid. */
static int probe_roundtrip(void) {
    int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int pass = 0, fail = 0;

    for (int si = 0; si < n_sizes; si++) {
        int n = sizes[si];
        gf64_t *f = (gf64_t *)malloc((size_t)n * sizeof(gf64_t));
        gf64_t *g = (gf64_t *)malloc((size_t)n * sizeof(gf64_t));
        if (!f || !g) abort();

        g_rng = 0xDEADBEEF ^ (uint64_t)n;
        fill_random(f, n);
        memcpy(g, f, (size_t)n * sizeof(gf64_t));
        gf64_addfft64_fwd_recursive(g, (size_t)n);
        gf64_addfft64_inv_recursive(g, (size_t)n);
        int ok = check_eq("round-trip", g, f, n);
        if (ok) pass++; else fail++;
        printf("  round-trip n=%-6d %s\n", n, ok ? "PASS" : "FAIL");

        free(f); free(g);
    }
    printf("Round-trip: %d/%d PASS\n", pass, pass + fail);
    return fail == 0;
}

/* Probe 2: convolution theorem via _poly_mul_recursive.
 * Compare against schoolbook poly_mul. */
static void poly_mul_schoolbook(gf64_t *out, const gf64_t *a, int la,
                                const gf64_t *b, int lb) {
    memset(out, 0, ((size_t)la + lb - 1) * sizeof(gf64_t));
    for (int i = 0; i < la; i++)
        for (int j = 0; j < lb; j++)
            out[i + j] ^= gf64_mul_reference(a[i], b[j]);
}
static int probe_convolution(void) {
    int sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int pass = 0, fail = 0;

    for (int si = 0; si < n_sizes; si++) {
        int n = sizes[si];
        int half = n / 2;
        gf64_t *a = (gf64_t *)malloc((size_t)n * sizeof(gf64_t));
        gf64_t *b = (gf64_t *)malloc((size_t)n * sizeof(gf64_t));
        gf64_t *c_ref = (gf64_t *)malloc(((size_t)n - 1) * sizeof(gf64_t));
        gf64_t *c_got = (gf64_t *)malloc(((size_t)n - 1) * sizeof(gf64_t));
        if (!a || !b || !c_ref || !c_got) abort();

        g_rng = 0xCAFEBEEF ^ (uint64_t)n;
        fill_random(a, half);
        fill_random(b, half);

        /* Schoolbook reference */
        poly_mul_schoolbook(c_ref, a, half, b, half);
        /* Recursive addFFT polynomial multiplication */
        gf64_addfft64_poly_mul_recursive(c_got, a, (size_t)half, b, (size_t)half,
                                          (size_t)(half + half - 1));
        int ok = check_eq("convolution", c_got, c_ref, half + half - 1);
        if (ok) pass++; else fail++;
        printf("  convolution n=%-6d %s\n", n, ok ? "PASS" : "FAIL");

        free(a); free(b); free(c_ref); free(c_got);
    }
    printf("Convolution: %d/%d PASS\n", pass, pass + fail);
    return fail == 0;
}

int main(void) {
    int ok_rt = probe_roundtrip();
    int ok_cv = probe_convolution();
    return (ok_rt && ok_cv) ? 0 : 1;
}