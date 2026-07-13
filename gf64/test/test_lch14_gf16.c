/*
 * test_lch14_gf16.c — standalone LCH14 additive FFT over GF(2^4)
 *
 * Debug scaffold for Phase 2b. Every operation fits in a uint8_t and
 * can be verified by hand. If the algorithm is correct here, port
 * back to gf64_additive_fft_lch14.c.
 *
 * Irreducible polynomial: x^4 + x + 1 (16-element field).
 * Field element: 4-bit unsigned integer; arithmetic is polynomial
 * modulo x^4 + x + 1 with GF(2) coefficients.
 *
 * Build & run:
 *   gcc -O0 -g test_lch14_gf16.c -o test_lch14_gf16 && ./test_lch14_gf16
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef uint8_t gf16_t;

#define GF16_MOD_POLY 0x13  /* x^4 + x + 1 (high-bit implicit) */

/* Precomputed log and exp tables for GF(2^4) with x^4 + x + 1. */
static uint8_t gf16_exp[16], gf16_log[16];

static void gf16_init_tables(void) {
    /* Generator: g = 0x02 (= x), with multiplication modulo x^4 + x + 1. */
    int i, x = 1;
    for (i = 0; i < 15; i++) {
        gf16_exp[i] = (uint8_t)x;
        x <<= 1;
        if (x & 0x10) x ^= GF16_MOD_POLY;
        x &= 0xF;
    }
    gf16_exp[15] = gf16_exp[0];  /* wrap */
    for (i = 0; i < 15; i++) gf16_log[gf16_exp[i]] = (uint8_t)i;
    /* gf16_log[0] = 0 (undefined; we won't query it). */
}

static gf16_t gf16_mul(gf16_t a, gf16_t b) {
    if (!a || !b) return 0;
    return gf16_exp[(gf16_log[a] + gf16_log[b]) % 15];
}

static gf16_t gf16_sq(gf16_t a) {
    if (!a) return 0;
    return gf16_exp[(2 * gf16_log[a]) % 15];
}

static gf16_t gf16_div(gf16_t a, gf16_t b) {
    if (!a) return 0;
    assert(b != 0);
    return gf16_exp[((int)gf16_log[a] - (int)gf16_log[b] + 15) % 15];
}

/* sigma(x) = x^2 + x. */
static gf16_t gf16_sigma(gf16_t x) {
    return gf16_sq(x) ^ x;
}

/* s_i(x) = sigma composed i times. */
static gf16_t gf16_si(int i, gf16_t x) {
    for (int s = 0; s < i; s++) x = gf16_sigma(x);
    return x;
}

/* Find Cantor basis v_0..v_3 for GF(2^4) such that
 *   v_0 = 1
 *   v_{i+1}^2 + v_{i+1} = v_i
 *   and v_i ∉ span(v_0..v_{i-1}).
 *
 * Returns 0 on success, populates `out`. */
static int compute_cantor_basis(gf16_t *out, int n) {
    out[0] = 1;
    for (int i = 0; i < n - 1; i++) {
        gf16_t v_i = out[i];
        gf16_t found = 0;
        for (gf16_t cand = 2; cand != 0; cand++) {
            gf16_t f = gf16_sigma(cand);
            if (f != v_i) continue;
            /* Verify independence: cand ∉ span(out[0..i]). */
            int indep = 1;
            for (int m = 0; m < (1 << i); m++) {
                gf16_t s = 0;
                for (int k = 0; k <= i; k++) {
                    if ((m >> k) & 1) s ^= out[k];
                }
                if (s == cand) { indep = 0; break; }
            }
            if (indep) { found = cand; break; }
        }
        if (!found) return -1;
        out[i + 1] = found;
    }
    return 0;
}

static gf16_t cantor_bas_dbg[4];
static int gf16_cantor_n = 4;

/* Multiplier computation.
 *
 * Two variants tried in this debug scaffold:
 *   (A) mu_j = s_i(W_m[j])             (vanishes on the low half)
 *   (B) mu_j = s_i(W_m[j + 2^i])       (uses the coset "high" element)
 *
 * Brute-force conv-theorem probe at n=4 found BOTH variants give the
 * same 6975/57600 pass rate (the algorithm is degenerate for high
 * coefficients, and only works when the high-half coefficients vanish).
 * The brute-force output below documents this for both.
 *
 * To switch variants, change `j_high` to either `j` (variant A) or
 * `j | (1 << level)` (variant B, current default).
 *
 * W_m[j] = sum_k bit_k(j) * basis[k]. */
static gf16_t compute_mu(int level, int j) {
    int j_high = j | (1 << level);   /* W_m element index for the "high" slot */
    gf16_t v_j = 0;
    for (int k = 0; k < gf16_cantor_n; k++) {
        if ((j_high >> k) & 1) v_j ^= (gf16_t)cantor_bas_dbg[k];
    }
    return gf16_si(level, v_j);
}

/* --- LCH14 forward (multiplier = s_i(W_m[j])) --- */
static void lch14_forward(gf16_t *arr, int n) {
    if (n <= 1) return;
    int m = 0; while ((1 << m) < n) m++;
    for (int i = 0; i < m; i++) {
        int stride = 1 << i;
        int chunk_count = n >> (i + 1);
        for (int chunk = 0; chunk < chunk_count; chunk++) {
            int base = chunk << (i + 1);
            for (int j = 0; j < stride; j++) {
                int lo = base + j;
                int hi = lo + stride;
                gf16_t mA = compute_mu(i, j);
                gf16_t even_v = arr[lo];
                gf16_t odd_v  = arr[hi];
                gf16_t gamma = gf16_mul(mA, odd_v);
                arr[lo] = even_v ^ gamma;
                arr[hi] = arr[lo] ^ odd_v;
            }
        }
    }
}

/* --- inverse --- */
static void lch14_inverse(gf16_t *arr, int n) {
    if (n <= 1) return;
    int m = 0; while ((1 << m) < n) m++;
    for (int i = m - 1; i >= 0; i--) {
        int stride = 1 << i;
        int chunk_count = n >> (i + 1);
        for (int chunk = 0; chunk < chunk_count; chunk++) {
            int base = chunk << (i + 1);
            for (int j = 0; j < stride; j++) {
                int lo = base + j;
                int hi = lo + stride;
                gf16_t mA = compute_mu(i, j);
                arr[hi] = arr[lo] ^ arr[hi];
                arr[lo] = arr[lo] ^ gf16_mul(mA, arr[hi]);
            }
        }
    }
}

/* Polynomial evaluation: f(x) = a[0] + a[1]*x + ... + a[deg]*x^deg at
 * point v. Uses Horner: r = a[deg]; for i = deg-1 .. 0: r = r*v + a[i]. */
static gf16_t poly_eval(const gf16_t *a, int deg, gf16_t v) {
    if (deg < 0) return 0;
    gf16_t r = a[deg];
    for (int i = deg - 1; i >= 0; i--) {
        r = gf16_mul(r, v) ^ a[i];
    }
    return r;
}

static void poly_mul_schoolbook(gf16_t *out, const gf16_t *a, int la,
                                const gf16_t *b, int lb) {
    memset(out, 0, (la + lb - 1) * sizeof(gf16_t));
    for (int i = 0; i < la; i++) {
        for (int j = 0; j < lb; j++) {
            out[i + j] ^= gf16_mul(a[i], b[j]);
        }
    }
}

static void print_arr(const char *tag, const gf16_t *a, int n) {
    printf("%s: [", tag);
    for (int i = 0; i < n; i++) printf(" %X", a[i]);
    printf(" ]\n");
}

int main(void) {
    printf("GF(2^4) LCH14 reference (x^4 + x + 1)\n");
    printf("=====================================\n\n");

    gf16_init_tables();

    int rc = compute_cantor_basis(cantor_bas_dbg, gf16_cantor_n);
    if (rc) { printf("Cantor-basis generation FAILED (%d)\n", rc); return rc; }

    /* Print basis. */
    printf("Computed Cantor basis: ");
    for (int i = 0; i < gf16_cantor_n; i++) printf("v_%d=0x%X ", i, cantor_bas_dbg[i]);
    printf("\n");

    /* Verify properties. */
    for (int i = 0; i < gf16_cantor_n - 1; i++) {
        gf16_t v_next = cantor_bas_dbg[i + 1];
        gf16_t chk = gf16_sq(v_next) ^ v_next;
        if (chk != cantor_bas_dbg[i]) {
            printf("FAIL: sigma(v_%d) = 0x%X, expected v_%d = 0x%X\n",
                   i + 1, chk, i, cantor_bas_dbg[i]);
            return 2;
        }
    }
    printf("Cantor recurrence OK (sigma(v_{i+1}) = v_i)\n");

    /* Verify independence. */
    for (int i = 0; i < gf16_cantor_n; i++) {
        for (int m = 0; m < (1 << i); m++) {
            gf16_t s = 0;
            for (int k = 0; k < i; k++) {
                if ((m >> k) & 1) s ^= (gf16_t)cantor_bas_dbg[k];
            }
            if (s == cantor_bas_dbg[i]) {
                printf("FAIL: v_%d in span(v_0..v_%d)\n", i, i - 1);
                return 3;
            }
        }
    }
    printf("Independence OK (each v_i not in span of previous)\n\n");

    /* Print multiplier table. */
    printf("Multiplier table mu[level][j] = s_i(W_m[j]):\n");
    for (int i = 0; i < 4; i++) {
        printf("  level %d: ", i);
        for (int j = 0; j < (1 << i); j++) {
            printf("0x%X ", compute_mu(i, j));
        }
        printf("\n");
    }
    printf("\n");

    /* Brute-force conv-theorem probe at n=4. */
    printf("Brute-force conv-theorem probe at n=4 (a_len=b_len=2):\n");
    int npass = 0, nfail = 0, ncases = 0;
    gf16_t win_a[2] = {0}, win_b[2] = {0};
    for (int a0 = 1; a0 < 16; a0++) {
        for (int a1 = 0; a1 < 16; a1++) {
            for (int b0 = 1; b0 < 16; b0++) {
                for (int b1 = 0; b1 < 16; b1++) {
                    gf16_t a[2] = {(gf16_t)a0, (gf16_t)a1};
                    gf16_t b[2] = {(gf16_t)b0, (gf16_t)b1};
                    gf16_t ab_ref[3] = {0};
                    poly_mul_schoolbook(ab_ref, a, 2, b, 2);

                    gf16_t A[4] = {a[0], a[1], 0, 0};
                    gf16_t B[4] = {b[0], b[1], 0, 0};
                    lch14_forward(A, 4);
                    lch14_forward(B, 4);
                    for (int i = 0; i < 4; i++) A[i] = gf16_mul(A[i], B[i]);
                    lch14_inverse(A, 4);

                    ncases++;
                    int ok = 1;
                    for (int i = 0; i < 3; i++) if (A[i] != ab_ref[i]) { ok = 0; break; }
                    if (ok) {
                        npass++;
                        if (npass == 1) {
                            win_a[0] = a[0]; win_a[1] = a[1];
                            win_b[0] = b[0]; win_b[1] = b[1];
                            memcpy(A, win_a, 4);  /* re-forward to show */
                        }
                    } else {
                        nfail++;
                    }
                }
            }
        }
    }
    printf("Result: %d / %d pass, %d fail\n", npass, ncases, nfail);

    if (npass == 0) {
        printf("\nNo inputs satisfy conv-theorem; the multiplier formula\n");
        printf("  mu_j = s_i(W_m[j])\n");
        printf("does not implement LCH14 for this field. Variants to test:\n");
        printf("  (B) mu_j = s_i(W_m[j + 2^i])\n");
        printf("  (C) mu_j = (s_i(W_m[j + 2^i]))^-1 * s_i(basis[i+1])\n");
        printf("  (D) swap mu/pre-mul: arr[hi] = even^mu*odd; arr[lo] = arr[hi]^odd\n");
    } else if (npass < (ncases * 95) / 100) {
        /* Less than 95% pass rate: algorithm is wrong but maybe close.
         * Dump a single failing input to inspect. */
        printf("\nAlgorithm wrong; sample failing input:\n");
        npass = 0;  /* reset so the search loop runs */
        for (int a0 = 1; a0 < 16 && npass == 0; a0++) {
            for (int a1 = 0; a1 < 16 && npass == 0; a1++) {
                for (int b0 = 1; b0 < 16 && npass == 0; b0++) {
                    for (int b1 = 0; b1 < 16 && npass == 0; b1++) {
                        gf16_t a[2] = {(gf16_t)a0, (gf16_t)a1};
                        gf16_t b[2] = {(gf16_t)b0, (gf16_t)b1};
                        gf16_t ab_ref[3] = {0};
                        poly_mul_schoolbook(ab_ref, a, 2, b, 2);

                        gf16_t A[4] = {a[0], a[1], 0, 0};
                        gf16_t B[4] = {b[0], b[1], 0, 0};
                        lch14_forward(A, 4);
                        lch14_forward(B, 4);
                        for (int i = 0; i < 4; i++) A[i] = gf16_mul(A[i], B[i]);
                        lch14_inverse(A, 4);

                        int ok = 1;
                        for (int i = 0; i < 3; i++) if (A[i] != ab_ref[i]) { ok = 0; break; }
                        if (!ok) {
                            /* display immediately */
                            printf("FAIL a=[%X %X] b=[%X %X]\n", a0, a1, b0, b1);
                            printf("  ab_ref   = [%X %X %X]\n", ab_ref[0], ab_ref[1], ab_ref[2]);
                            gf16_t Ae[4] = {a[0], a[1], 0, 0};
                            lch14_forward(Ae, 4);
                            printf("  fwd(a)   = [%X %X %X %X]\n", Ae[0], Ae[1], Ae[2], Ae[3]);
                            gf16_t Be[4] = {b[0], b[1], 0, 0};
                            lch14_forward(Be, 4);
                            printf("  fwd(b)   = [%X %X %X %X]\n", Be[0], Be[1], Be[2], Be[3]);
                            gf16_t pp[4] = {gf16_mul(Ae[0],Be[0]), gf16_mul(Ae[1],Be[1]),
                                            gf16_mul(Ae[2],Be[2]), gf16_mul(Ae[3],Be[3])};
                            lch14_inverse(pp, 4);
                            printf("  inv(pp)  = [%X %X %X %X] (want ab_ref = [%X %X %X])\n",
                                   pp[0], pp[1], pp[2], pp[3], ab_ref[0], ab_ref[1], ab_ref[2]);
                            printf("  eval(a) at 0/1/v_1/1+v_1 = %X/%X/%X/%X\n",
                                   poly_eval(a, 1, 0), poly_eval(a, 1, 1),
                                   poly_eval(a, 1, (gf16_t)cantor_bas_dbg[1]),
                                   poly_eval(a, 1, (gf16_t)(1 ^ cantor_bas_dbg[1])));
                            printf("  eval(b) at 0/1/v_1/1+v_1 = %X/%X/%X/%X\n",
                                   poly_eval(b, 1, 0), poly_eval(b, 1, 1),
                                   poly_eval(b, 1, (gf16_t)cantor_bas_dbg[1]),
                                   poly_eval(b, 1, (gf16_t)(1 ^ cantor_bas_dbg[1])));
                            npass = 1;  /* exit condition */
                        }
                    }
                }
            }
        }
    } else {
        printf("\nAlgorithm appears to pass (>=95%%).\n");
        gf16_t a[2] = {1, 2};
        gf16_t A[4] = {1, 2, 0, 0};
        print_arr("input a", A, 4);
        lch14_forward(A, 4);
        print_arr("forward(a)", A, 4);
        printf("explicit eval: f(0)=0x%X  f(1)=0x%X  f(v_1)=0x%X  f(1+v_1)=0x%X\n",
               poly_eval(a, 1, 0),
               poly_eval(a, 1, 1),
               poly_eval(a, 1, (gf16_t)cantor_bas_dbg[1]),
               poly_eval(a, 1, (gf16_t)(1 ^ cantor_bas_dbg[1])));
    }
    return 0;
}
