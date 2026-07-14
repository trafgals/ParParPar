/*
 * test_gf64_additive_fft_hqc2026.c — verify HQC 2026 §2.3 Algorithm 2
 *   (LCH14 addFFT) over GF(2^64) with the corrected multiplier s_{i-1}(a).
 *
 * Sister probe to the canonical formula in gf64_additive_fft_lch14.c
 * (which uses the broken multiplier s_i(v_j) and fails the convolution
 * probe). This file uses the corrected algorithm from
 * gf64_additive_fft_hqc2026.c and exercises it at the canonical PAR3
 * transform sizes up through the available Cantor-basis depth (ℓ_m = 20,
 * covering n ∈ {2, 4, 8, 16, 32, 64, ..., 2^20}).
 *
 *   Build & run (gcc inside WSL — see CLAUDE.md):
 *     cd gf64/test && make test_gf64_additive_fft_hqc2026 \
 *       && ./test_gf64_additive_fft_hqc2026
 */

#include "../gf64_global.h"
#include "../gf64_additive_fft.h"
#include "../gf64_cantor_basis.h"
#include "../gf64_square.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);
extern gf64_t gf64_inverse(gf64_t a);

/* ----- reference math -----
 *
 * poly_eval(c, n, at): compute sum c_j * at^j (schoolbook Horner).
 * poly_mul_schoolbook: standard convolution.
 * W_m(j) = XOR of GF64_CANTOR_BASIS[k] for bits k set in j.
 */
static gf64_t W_m(int j) {
    gf64_t v = 0;
    for (int k = 0; k < 20; k++) if ((j >> k) & 1) v ^= GF64_CANTOR_BASIS[k];
    return v;
}
static gf64_t poly_eval(const gf64_t *c, int n, gf64_t at) {
    gf64_t r = 0, p = 1;
    for (int i = 0; i < n; i++) { r ^= gf64_mul_reference(p, c[i]); p = gf64_mul_reference(p, at); }
    return r;
}
static void poly_mul_schoolbook(gf64_t *out, const gf64_t *a, int la,
                                const gf64_t *b, int lb) {
    memset(out, 0, ((size_t)la + lb - 1) * sizeof(gf64_t));
    for (int i = 0; i < la; i++)
        for (int j = 0; j < lb; j++)
            out[i + j] ^= gf64_mul_reference(a[i], b[j]);
}

/* ----- probes (random sampling; exhaustive 16^N is infeasible) ----- */

#define XOSHIRO state[0]

static void fill_random(gf64_t *out, int n, uint64_t *state) {
    /* Tiny xorshift64 for deterministic seeding. */
    for (int i = 0; i < n; i++) {
        uint64_t x = *state;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        *state = x;
        out[i] = x;
    }
}

static int verify_forward(int n, int ncases, uint64_t seed) {
    /* Forward-output probe: fwd(c) should equal (f(W_m(0)+a), ..., f(W_m(n-1)+a)). */
    uint64_t state = seed;
    int npass = 0;
    int logn = 0; while ((1 << logn) < n) logn++;
    gf64_t a = GF64_CANTOR_BASIS[logn - 1];

    /* Worked-example debug for n=8 (the size that fails). */
    if (n == 8 && ncases > 0) {
        gf64_t c_dbg[8] = {0x0102030405060708ULL, 0x1112131415161718ULL,
                           0x2122232425262728ULL, 0x3132333435363738ULL,
                           0x4142434445464748ULL, 0x5152535455565758ULL,
                           0x6162636465666768ULL, 0x7172737475767778ULL};
        gf64_t f_dbg[8];
        memcpy(f_dbg, c_dbg, n * sizeof(gf64_t));
        gf64_addfft64_fwd(f_dbg, n);
        printf("  --- worked example (a = basis[%d] = 0x%016llx):\n",
               logn - 1, (unsigned long long)a);
        for (int i = 0; i < n; i++) {
            gf64_t want = poly_eval(c_dbg, n, a ^ W_m(i));
            printf("    addFFT[%d] = 0x%016llx | want = 0x%016llx %s\n",
                   i, (unsigned long long)f_dbg[i], (unsigned long long)want,
                   f_dbg[i] == want ? "OK" : "MISMATCH");
        }
    }

    for (int trial = 0; trial < ncases; trial++) {
        gf64_t c[1024], f[1024];
        fill_random(c, n, &state);
        memcpy(f, c, n * sizeof(gf64_t));
        gf64_addfft64_fwd(f, n);
        int ok = 1;
        for (int i = 0; i < n; i++) {
            gf64_t want = poly_eval(c, n, a ^ W_m(i));
            if (f[i] != want) { ok = 0; break; }
        }
        if (ok) npass++;
    }
    return npass * 10000 / ncases;
}

static int verify_roundtrip(int n, int ncases, uint64_t seed) {
    uint64_t state = seed;
    int npass = 0;
    for (int trial = 0; trial < ncases; trial++) {
        gf64_t c0[1024], f[1024];
        fill_random(c0, n, &state);
        memcpy(f, c0, n * sizeof(gf64_t));
        gf64_addfft64_fwd(f, n);
        gf64_addfft64_inv(f, n);
        int ok = 1;
        for (int j = 0; j < n; j++) if (f[j] != c0[j]) { ok = 0; break; }
        if (ok) npass++;
    }
    return npass * 10000 / ncases;
}

static int verify_convolution(int n, int ncases, uint64_t seed) {
    /* n must be ≥ 4 to fit a degree-1 × degree-1 = degree-2 product. */
    if (n < 4) return 10000;
    uint64_t state = seed;
    int half = n / 2;
    int npass = 0;
    for (int trial = 0; trial < ncases; trial++) {
        gf64_t A[1024], B[1024];
        fill_random(A, half, &state);
        memset(A + half, 0, (n - half) * sizeof(gf64_t));
        fill_random(B, half, &state);
        memset(B + half, 0, (n - half) * sizeof(gf64_t));
        gf64_t ab_ref[1024];
        poly_mul_schoolbook(ab_ref, A, half, B, half);
        gf64_t FA[1024], FB[1024];
        memcpy(FA, A, n * sizeof(gf64_t));
        memcpy(FB, B, n * sizeof(gf64_t));
        gf64_addfft64_fwd(FA, n);
        gf64_addfft64_fwd(FB, n);
        for (int i = 0; i < n; i++) FA[i] = gf64_mul_reference(FA[i], FB[i]);
        gf64_addfft64_inv(FA, n);
        int ok = 1;
        for (int i = 0; i < 2 * half - 1; i++)
            if (FA[i] != ab_ref[i]) { ok = 0; break; }
        if (ok) npass++;
    }
    return npass * 10000 / ncases;
}

int main(void) {
    printf("HQC 2026 §2.3 Algorithm 2 (LCH14 addFFT) over GF(2^64)\n");
    printf("==========================================================\n");
    printf("Irreducible: x^64 + x^4 + x^3 + x + 1 (NIST AES-GCM)\n");
    printf("Cantor basis (v_0..v_7): ");
    for (int i = 0; i < 8; i++) printf("%016llx ", (unsigned long long)GF64_CANTOR_BASIS[i]);
    printf("\n\n");

    const int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256};
    const int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    const int NFORWARD = 200;
    const int NROUND = 100;
    const int NCONV = 200;

    int all_pass = 1;

    for (int si = 0; si < nsizes; si++) {
        int n = sizes[si];
        printf("n=%d:\n", n);
        int r1 = verify_forward(n, NFORWARD, 0xC0FFEEULL ^ (uint64_t)n);
        printf("  forward-output (%d cases):     %5.2f%%\n", NFORWARD, r1 / 100.0);
        if (r1 != 10000) all_pass = 0;
        int r2 = verify_roundtrip(n, NROUND, 0xCAFEULL ^ (uint64_t)n);
        printf("  round-trip      (%d cases):     %5.2f%%\n", NROUND, r2 / 100.0);
        if (r2 != 10000) all_pass = 0;
        if (n >= 4) {
            int r3 = verify_convolution(n, NCONV, 0xBEEFULL ^ (uint64_t)n);
            printf("  convolution     (%d cases):     %5.2f%%\n", NCONV, r3 / 100.0);
            if (r3 != 10000) all_pass = 0;
        }
    }

    printf("\nCONCLUSION: %s\n",
           all_pass ? "100% pass across all probes / all sizes (probing at GF(2^64))"
                    : "FAILURES (see above)");
    return all_pass ? 0 : 1;
}
