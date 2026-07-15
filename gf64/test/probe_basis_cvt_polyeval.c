/* probe_basis_cvt_polyeval.c — port polyeval's bc_64 (recursive basis
 * conversion at the bit level) to our GF(2^64) abstraction.
 *
 * Polyeval's actual structure uses Algorithm 1 with general k via the cvt
 * function (in bc_256.c ref). It computes:
 *
 *   cvt(data, datalen, logsize_blk, polyloglen_blk):
 *     si = max pow of 2 < polyloglen_blk
 *     rep_in_si(data, datalen, logsize_blk, polyloglen_blk, si)  # decompose via s_si at scale (logsize_blk + polyloglen_blk - 1)
 *     cvt(data, datalen, logsize_blk, si)                          # recurse on "low" half
 *     cvt(data, datalen, logsize_blk+si, polyloglen_blk-si)        # recurse on "high" half
 *
 * GF(2^64) port: datalen = n (poly length in elements), logsize_blk = 0
 * (no inner block structure since gf64_t IS our atomic unit), polyloglen_blk
 * = log2(n). At each recursion level, decompose via s_k where k = max pow
 * of 2 < polyloglen_blk, then recurse on the two halves.
 *
 * Build:
 *   gcc -O2 -mpclmul -mssse3 -mavx2 -mavx512f -mvpclmulqdq -I.. probe_basis_cvt_polyeval.c \
 *       ../gf64_additive_fft_hqc2026.c ../gf64_additive_fft.c ../gf64_invert.c \
 *       ../gf64_invert_ita.c ../gf64_poly_mul_karatsuba.c ../gf64_poly_mul_toom3.c \
 *       ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c \
 *       -o probe_basis_cvt_polyeval
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "../gf64_global.h"
#include "../gf64_additive_fft.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);
extern gf64_t gf64_inverse(gf64_t a);

/* ===== polyeval port: decomposition primitives ===== */

/* Largest power of 2 strictly less than n. (Mirrors choose_si.) */
static int choose_si(int n) {
    int si = 1;
    for (int i = 1; (1 << i) < n; i++) {
        si = 1 << i;
    }
    return si;
}

/* 2-term division: divide poly (in place) by x^{si_h} - x^{si_l}.
 * Mirrors polyeval's div_blk. */
static void div_blk_gf64(gf64_t *poly, int si_h, int si_l, int polylen) {
    int deg_diff = si_h - si_l;
    for (int i = polylen - 1; i >= si_h; i--) {
        poly[i - deg_diff] ^= poly[i];
    }
}

/* Inverse 2-term division. */
static void idiv_blk_gf64(gf64_t *poly, int si_h, int si_l, int polylen) {
    int deg_diff = si_h - si_l;
    for (int i = si_h; i < polylen; i++) {
        poly[i - deg_diff] ^= poly[i];
    }
}

/* rep_in_si: decompose via s_si at multiple scales within the poly.
 * Mirrors polyeval's rep_in_si. */
static void rep_in_si_gf64(gf64_t *data, int datalen, int logsize_blk,
                            int polyloglen_blk, int si) {
    for (int i = polyloglen_blk - 1; i >= si; i--) {
        int polylen = 1 << (i + logsize_blk + 1);
        int si_h = 1 << (i + logsize_blk);
        int si_l = 1 << (i + logsize_blk - si);
        for (int j = 0; j < datalen; j += polylen) {
            div_blk_gf64(data + j, si_h, si_l, polylen);
        }
    }
}

static void irep_in_si_gf64(gf64_t *data, int datalen, int logsize_blk,
                             int polyloglen_blk, int si) {
    for (int i = si; i < polyloglen_blk; i++) {
        int polylen = 1 << (i + logsize_blk + 1);
        int si_h = 1 << (i + logsize_blk);
        int si_l = 1 << (i + logsize_blk - si);
        for (int j = 0; j < datalen; j += polylen) {
            idiv_blk_gf64(data + j, si_h, si_l, polylen);
        }
    }
}

/* cvt: Algorithm 1 recursive basisCvt (Chen 2018 / HQC 2026).
 * Mirrors polyeval's cvt in bc_256.c. */
static void cvt_gf64(gf64_t *data, int datalen, int logsize_blk, int polyloglen_blk) {
    if (polyloglen_blk <= 1) return;
    int si = choose_si(polyloglen_blk);
    rep_in_si_gf64(data, datalen, logsize_blk, polyloglen_blk, si);
    cvt_gf64(data, datalen, logsize_blk, si);
    cvt_gf64(data, datalen, logsize_blk + si, polyloglen_blk - si);
}

static void icvt_gf64(gf64_t *data, int datalen, int logsize_blk, int polyloglen_blk) {
    if (polyloglen_blk <= 1) return;
    int si = choose_si(polyloglen_blk);
    icvt_gf64(data, datalen, logsize_blk, si);
    icvt_gf64(data, datalen, logsize_blk + si, polyloglen_blk - si);
    irep_in_si_gf64(data, datalen, logsize_blk, polyloglen_blk, si);
}

/* Top-level: cvt_gf64 with default params. */
static void basisCvt_polyeval(gf64_t *poly, unsigned n) {
    if (n <= 2) return;
    int log_n = 0;
    while ((1U << log_n) < n) log_n++;
    cvt_gf64(poly, n, 0, log_n);
}

static void ibasisCvt_polyeval(gf64_t *poly, unsigned n) {
    if (n <= 2) return;
    int log_n = 0;
    while ((1U << log_n) < n) log_n++;
    icvt_gf64(poly, n, 0, log_n);
}

/* ===== Test helpers ===== */

static uint64_t g_rng = 0xCAFEBABE12345ULL;
static uint64_t splitmix64_next(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
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

int main(void) {
    int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    int rt_pass = 0, rt_fail = 0;
    int xv_pass = 0, xv_fail = 0;

    for (int si = 0; si < n_sizes; si++) {
        int n = sizes[si];
        gf64_t *f     = (gf64_t *)malloc(n * sizeof(gf64_t));
        gf64_t *g_rt  = (gf64_t *)malloc(n * sizeof(gf64_t));
        gf64_t *g_xv  = (gf64_t *)malloc(n * sizeof(gf64_t));
        if (!f || !g_rt || !g_xv) abort();

        /* Random input */
        g_rng = 0xDEADBEEF ^ (uint64_t)n;
        for (int i = 0; i < n; i++) f[i] = splitmix64_next();

        /* Test 1: round-trip via polyeval-style basisCvt/ibasisCvt */
        memcpy(g_rt, f, n * sizeof(gf64_t));
        basisCvt_polyeval(g_rt, n);
        ibasisCvt_polyeval(g_rt, n);
        int ok_rt = check_eq("round-trip", g_rt, f, n);
        if (ok_rt) rt_pass++; else rt_fail++;

        /* Test 2: cross-validate against gf64_addfft64_inv.
         * If basisCvt_polyeval == basisCvt_matrix, then
         *   gf64_addfft64_inv(basisCvt_polyeval(f)) == f
         * because gf64_addfft64_inv(gf64_addfft64_fwd(f)) == f and
         * gf64_addfft64_inv = ibasisCvt_matrix . butterfly_inv .
         */
        memcpy(g_xv, f, n * sizeof(gf64_t));
        basisCvt_polyeval(g_xv, n);
        gf64_addfft64_inv(g_xv, n);
        int ok_xv = check_eq("cross-val", g_xv, f, n);
        if (ok_xv) xv_pass++; else xv_fail++;

        printf("%-4s  N=%-5d round-trip   %-4s  cross-val\n",
               ok_rt ? "PASS" : "FAIL", n,
               ok_xv ? "PASS" : "FAIL");

        free(f); free(g_rt); free(g_xv);
    }

    printf("\nSummary: %d/%d round-trip, %d/%d cross-val\n",
           rt_pass, rt_pass + rt_fail,
           xv_pass, xv_pass + xv_fail);

    return (rt_fail || xv_fail) ? 1 : 0;
}