/*
 * test_gf64_gao_mateer.c — Gao-Mateer evaluation-basis FFT over GF(2^4).
 *
 * DIF (decimation-in-frequency) form: pairs output slots i and i+half
 * using the same sub-result, with multipliers sqrt(wm(i)) and
 * sqrt(wm(i+half)) (inverse Frobenius of the W_m evaluation points).
 *
 * Recursion: f(x) = f_even(x^2) + x · f_odd(x^2). Evaluating at v:
 *   f(v) = F_even(v^2) + v · F_odd(v^2).
 * F_even and F_odd have degree < n/2.
 *
 * DIF structure (input in natural order, output in natural order):
 *   For each i in [0, half):
 *     even[i] = poly[i], odd[i] = poly[i + half]
 *     square each (Frobenius applied to coefficients).
 *     recurse on each half (size half).
 *     a[i]        = even[i] + sqrt(wm(i)) * odd[i]
 *     a[i + half] = even[i] + sqrt(wm(i + half)) * odd[i]
 *
 * Build & run from gf64/test/:
 *   gcc -O2 test_gf64_gao_mateer.c -o test_gf64_gao_mateer && ./test_gf64_gao_mateer
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef uint8_t gf16_t;
#define GF16_MOD_POLY 0x13
static uint8_t gf16_exp[16], gf16_log[16];
static void gf16_init(void) {
    int x = 1;
    for (int i = 0; i < 15; i++) { gf16_exp[i] = (uint8_t)x; x <<= 1; if (x & 0x10) x ^= GF16_MOD_POLY; x &= 0xF; }
    gf16_exp[15] = gf16_exp[0];
    for (int i = 0; i < 15; i++) gf16_log[gf16_exp[i]] = (uint8_t)i;
}
static inline gf16_t gf16_mul(gf16_t a, gf16_t b) {
    if (!a || !b) return 0;
    return gf16_exp[(gf16_log[a] + gf16_log[b]) % 15];
}
static inline gf16_t gf16_sq(gf16_t a) {
    if (!a) return 0;
    return gf16_exp[(2 * gf16_log[a]) % 15];
}
static inline gf16_t gf16_sqrt(gf16_t a) {
    if (!a) return 0;
    gf16_t r = a;
    for (int i = 0; i < 3; i++) r = gf16_sq(r);  /* x^8 in GF(2^4) */
    return r;
}

static gf16_t basis[4];
static int compute_basis(void) {
    basis[0] = 1;
    for (int i = 0; i < 3; i++) {
        gf16_t target = basis[i];
        for (int c = 2; c < 16; c++) {
            if ((gf16_sq(c) ^ c) != target) continue;
            int indep = 1;
            for (int m = 0; m < (1 << i); m++) {
                gf16_t s = 0;
                for (int k = 0; k <= i; k++) if ((m >> k) & 1) s ^= basis[k];
                if (s == c) { indep = 0; break; }
            }
            if (indep) { basis[i+1] = c; goto next_i; }
        }
        return -1;
        next_i:;
    }
    return 0;
}

static gf16_t wm(int j) {
    gf16_t r = 0;
    for (int k = 0; k < 4; k++) if ((j >> k) & 1) r ^= basis[k];
    return r;
}

/* DIF Gao-Mateer forward.
 * poly: input monomial coefficients a_0..a_{n-1}, OUTPUT evaluations at wm(0..n-1).
 * Scratch: caller-owned buffer of size 2*n. */
static void gao_fwd(gf16_t *poly, int n, gf16_t *scratch) {
    if (n <= 1) return;
    int half = n / 2;
    /* Deinterleave: even[i] = poly[i], odd[i] = poly[i+half]. */
    gf16_t *even = scratch;
    gf16_t *odd  = scratch + half;
    for (int i = 0; i < half; i++) { even[i] = poly[i]; odd[i] = poly[i + half]; }
    /* Frobenius: square each coefficient in place. */
    for (int i = 0; i < half; i++) { even[i] = gf16_sq(even[i]); odd[i] = gf16_sq(odd[i]); }
    /* Recurse on each half. */
    gao_fwd(even, half, scratch + n);
    gao_fwd(odd,  half, scratch + n);
    /* Butterfly. */
    for (int i = 0; i < half; i++) {
        gf16_t sqrt_lo = gf16_sqrt(wm(i));
        gf16_t sqrt_hi = gf16_sqrt(wm(i + half));
        poly[i]        = even[i] ^ gf16_mul(sqrt_lo, odd[i]);
        poly[i + half] = even[i] ^ gf16_mul(sqrt_hi, odd[i]);
    }
}

/* Inverse: recover monomial coefficients from evaluations.
 * Inverse butterfly:
 *   from poly[i] = even[i] ^ sqrt_lo * odd[i]
 *         poly[i+half] = even[i] ^ sqrt_hi * odd[i]
 *   we have poly[i] ^ poly[i+half] = (sqrt_lo ^ sqrt_hi) * odd[i].
 *   So odd[i] = mul(inv(sqrt_lo ^ sqrt_hi), poly[i] ^ poly[i+half]).
 *   And even[i] = poly[i] ^ mul(sqrt_lo, odd[i]).
 * Then UN-Frobenius (square root each coefficient) to get the pre-Frobenius values,
 * then recurse on each half. */
static void gao_inv(gf16_t *poly, int n, gf16_t *scratch) {
    if (n <= 1) return;
    int half = n / 2;
    gf16_t *even = scratch;
    gf16_t *odd  = scratch + half;
    /* Inverse butterfly. */
    for (int i = 0; i < half; i++) {
        gf16_t sqrt_lo = gf16_sqrt(wm(i));
        gf16_t sqrt_hi = gf16_sqrt(wm(i + half));
        gf16_t diff = sqrt_lo ^ sqrt_hi;
        if (diff == 0) {
            /* sqrt_lo == sqrt_hi: poly[i] == poly[i+half], can't recover odd[i].
             * This happens when wm(i)^2 == wm(i+half)^2, i.e., they're Frobenius-twins.
             * For our V={0,1,6,7}: sqrt(0)=0, sqrt(1)=1, sqrt(6)=7, sqrt(7)=6.
             * sqrt_lo^sqrt_hi:
             *   i=0: sqrt(0)^sqrt(2) = 0^7 = 7 (non-zero). OK.
             *   i=1: sqrt(1)^sqrt(3) = 1^6 = 7 (non-zero). OK.
             * Good — for GF(2^4), the difference is always non-zero at this level. */
            fprintf(stderr, "diff=0 at i=%d (sqrt_lo=%X sqrt_hi=%X)\n", i, sqrt_lo, sqrt_hi);
            abort();
        }
        /* odd[i] = poly[i] ^ poly[i+half] divided by (sqrt_lo ^ sqrt_hi). */
        gf16_t inv_diff = gf16_exp[(15 - gf16_log[diff]) % 15];
        odd[i] = gf16_mul(inv_diff, poly[i] ^ poly[i + half]);
        even[i] = poly[i] ^ gf16_mul(sqrt_lo, odd[i]);
    }
    /* Un-Frobenius: square root (inverse of squaring). */
    for (int i = 0; i < half; i++) { even[i] = gf16_sqrt(even[i]); odd[i] = gf16_sqrt(odd[i]); }
    /* Recurse on each half. */
    gao_inv(even, half, scratch + n);
    gao_inv(odd,  half, scratch + n);
    /* Interleave back. */
    for (int i = 0; i < half; i++) { poly[i] = even[i]; poly[i + half] = odd[i]; }
}

static void poly_mul_schoolbook(gf16_t *out, const gf16_t *a, int la, const gf16_t *b, int lb) {
    memset(out, 0, la + lb - 1);
    for (int i = 0; i < la; i++) for (int j = 0; j < lb; j++) out[i+j] ^= gf16_mul(a[i], b[j]);
}

/* Convolution-theorem probe at size n: forward, pointwise-mul, inverse, compare to schoolbook. */
static int probe(int n) {
    int npass = 0, ncases = 0;
    /* Use small random inputs. */
    for (int a0 = 1; a0 < 16; a0++)
    for (int a1 = 0; a1 < 16; a1++)
    for (int b0 = 1; b0 < 16; b0++)
    for (int b1 = 0; b1 < 16; b1++) {
        gf16_t a[2] = {(gf16_t)a0, (gf16_t)a1};
        gf16_t b[2] = {(gf16_t)b0, (gf16_t)b1};
        gf16_t ab_ref[3] = {0};
        poly_mul_schoolbook(ab_ref, a, 2, b, 2);

        gf16_t A[16] = {a[0], a[1], 0, 0};  /* padded to n */
        gf16_t B[16] = {b[0], b[1], 0, 0};
        gf16_t scratch[256];
        gao_fwd(A, n, scratch);
        gao_fwd(B, n, scratch);
        for (int i = 0; i < n; i++) A[i] = gf16_mul(A[i], B[i]);
        gao_inv(A, n, scratch);

        int ok = 1;
        for (int i = 0; i < 3; i++) if (A[i] != ab_ref[i]) { ok = 0; break; }
        ncases++;
        if (ok) npass++;
    }
    return npass * 10000 / ncases;
}

int main(void) {
    gf16_init();
    if (compute_basis()) { printf("basis FAIL\n"); return 1; }
    printf("Basis: v_0=0x%X v_1=0x%X v_2=0x%X v_3=0x%X\n\n",
           basis[0], basis[1], basis[2], basis[3]);
    printf("W_m points: wm(0)=0x%X wm(1)=0x%X wm(2)=0x%X wm(3)=0x%X\n",
           wm(0), wm(1), wm(2), wm(3));
    printf("sqrt: sqrt(0)=0x%X sqrt(1)=0x%X sqrt(6)=0x%X sqrt(7)=0x%X\n\n",
           gf16_sqrt(0), gf16_sqrt(1), gf16_sqrt(6), gf16_sqrt(7));

    /* Round-trip sanity at n=2. */
    {
        gf16_t A[2] = {0xa, 0xb};
        gf16_t scratch[32];
        gao_fwd(A, 2, scratch);
        printf("n=2 fwd [0x%X 0x%X] -> [0x%X 0x%X]  (want f(0)=0x%X f(1)=0x%X)\n",
               0xa, 0xb, A[0], A[1], 0xa, 0xa ^ 0xb);
        gao_inv(A, 2, scratch);
        printf("n=2 inv back to [0x%X 0x%X]  (want [0x%X 0x%X])\n",
               A[0], A[1], 0xa, 0xb);
    }

    /* Round-trip at n=4. */
    {
        gf16_t A[4] = {0xa, 0xb, 0xc, 0xd};
        gf16_t scratch[128];
        printf("\nn=4 input [0x%X 0x%X 0x%X 0x%X]\n", A[0], A[1], A[2], A[3]);
        gao_fwd(A, 4, scratch);
        printf("n=4 fwd   [0x%X 0x%X 0x%X 0x%X]\n", A[0], A[1], A[2], A[3]);
        printf("        want f(0)=0x%X f(1)=0x%X f(wm(2))=0x%X f(wm(3))=0x%X\n",
               0xa, 0xa^0xb^0xc^0xd,
               (gf16_mul(gf16_mul(gf16_mul(0xd, wm(2)) ^ 0xc, wm(2)) ^ 0xb, wm(2)) ^ 0xa),
               (gf16_mul(gf16_mul(gf16_mul(0xd, wm(3)) ^ 0xc, wm(3)) ^ 0xb, wm(3)) ^ 0xa));
        gao_inv(A, 4, scratch);
        printf("n=4 inv   [0x%X 0x%X 0x%X 0x%X]  (want [0x%X 0x%X 0x%X 0x%X])\n",
               A[0], A[1], A[2], A[3], 0xa, 0xb, 0xc, 0xd);
    }

    /* Convolution-theorem probe at n=4. */
    printf("\nConv-theorem probe at n=4 (a_len=b_len=2):\n");
    int rate = probe(4);
    printf("Pass rate: %.2f%%\n", rate / 100.0);
    return 0;
}