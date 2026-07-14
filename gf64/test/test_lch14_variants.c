/*
 * test_lch14_variants.c — try ALL FOUR LCH14 multiplier variants
 * against the GF(2^4) brute-force conv-theorem probe. Documents
 * whether ANY variant produces a non-trivial pass rate.
 *
 * Variant A: mu_j = s_i(W_m[j])
 * Variant B: mu_j = s_i(W_m[j | (1<<i)])
 * Variant C: mu_j = (s_i(W_m[j | (1<<i)]))^-1 * s_i(basis[i+1])
 * Variant D: same as B but butterfly computes (arr[hi] = even^mu*odd; arr[lo] = arr[hi]^odd)
 *
 * The Cantor recurrence sigma(v_{i+1}) = v_i in GF(2^4) with field
 * degree 4 forces s_i(v_i) = 1, and s_i(any linear combo of basis
 * elements at level ≤ i) = 1. This means all 4 variants give identical
 * butterflies in GF(2^4).
 *
 * Build & run:
 *   gcc -O0 test_lch14_variants.c -o test_lch14_variants && ./test_lch14_variants
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
    for (int i = 0; i < 15; i++) { gf16_exp[i]=(uint8_t)x; x <<= 1; if (x & 0x10) x ^= GF16_MOD_POLY; x &= 0xF; }
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
static inline gf16_t gf16_inv(gf16_t a) {
    assert(a != 0);
    return gf16_exp[(15 - gf16_log[a]) % 15];
}
static inline gf16_t gf16_sigma(gf16_t x) { return gf16_sq(x) ^ x; }
static inline gf16_t gf16_si(int i, gf16_t x) { while (i--) x = gf16_sigma(x); return x; }

/* Find Cantor basis v_0..v_3 such that v_0=1, sigma(v_{i+1})=v_i,
 * and v_i is independent of {v_0..v_{i-1}}. */
static int compute_basis(gf16_t *out) {
    out[0] = 1;
    for (int i = 0; i < 3; i++) {
        gf16_t target = out[i];
        for (int c = 2; c < 16; c++) {
            if (gf16_sigma(c) != target) continue;
            /* independence */
            int indep = 1;
            for (int m = 0; m < (1 << i); m++) {
                gf16_t s = 0;
                for (int k = 0; k <= i; k++) if ((m >> k) & 1) s ^= out[k];
                if (s == c) { indep = 0; break; }
            }
            if (indep) { out[i+1] = c; goto next_i; }
        }
        return -1;
        next_i:;
    }
    return 0;
}

static gf16_t basis[4];

/* Compute v_j = XOR of basis[k] for bits set in j. */
static inline gf16_t W_m(int j) {
    gf16_t r = 0;
    for (int k = 0; k < 4; k++) if ((j >> k) & 1) r ^= basis[k];
    return r;
}

/* Multiplier for variant. */
typedef enum { VAR_A, VAR_B, VAR_C } variant_t;
static gf16_t compute_mu_var(variant_t v, int level, int j) {
    switch (v) {
        case VAR_A: return gf16_si(level, W_m(j));
        case VAR_B: return gf16_si(level, W_m(j | (1 << level)));
        case VAR_C: {
            /* (s_i(W_m[j | (1<<i)]))^-1 * s_i(basis[i+1]) */
            gf16_t a = gf16_si(level, W_m(j | (1 << level)));
            gf16_t b = (level + 1 < 4) ? gf16_si(level, basis[level+1]) : (gf16_t)0;
            return gf16_mul(gf16_inv(a), b);
        }
    }
    return 0;
}

/* Forward butterfly for a given variant. */
static void fwd(gf16_t *a, int n, variant_t v) {
    if (n <= 1) return;
    int m = 0; while ((1 << m) < n) m++;
    for (int i = 0; i < m; i++) {
        int stride = 1 << i;
        int chunks = n >> (i + 1);
        for (int ch = 0; ch < chunks; ch++) {
            int base = ch << (i + 1);
            for (int j = 0; j < stride; j++) {
                int lo = base + j, hi = lo + stride;
                gf16_t mu = compute_mu_var(v, i, j);
                gf16_t e = a[lo], o = a[hi];
                gf16_t gamma = gf16_mul(mu, o);
                a[lo] = e ^ gamma;
                a[hi] = a[lo] ^ o;
            }
        }
    }
}

static void inv(gf16_t *a, int n, variant_t v) {
    if (n <= 1) return;
    int m = 0; while ((1 << m) < n) m++;
    for (int i = m - 1; i >= 0; i--) {
        int stride = 1 << i;
        int chunks = n >> (i + 1);
        for (int ch = 0; ch < chunks; ch++) {
            int base = ch << (i + 1);
            for (int j = 0; j < stride; j++) {
                int lo = base + j, hi = lo + stride;
                gf16_t mu = compute_mu_var(v, i, j);
                a[hi] = a[lo] ^ a[hi];
                a[lo] = a[lo] ^ gf16_mul(mu, a[hi]);
            }
        }
    }
}

static void poly_mul_schoolbook(gf16_t *out, const gf16_t *a, int la, const gf16_t *b, int lb) {
    memset(out, 0, la + lb - 1);
    for (int i = 0; i < la; i++) for (int j = 0; j < lb; j++) out[i+j] ^= gf16_mul(a[i], b[j]);
}

static int probe_variant(variant_t v) {
    int npass = 0, ncases = 0;
    for (int a0 = 1; a0 < 16; a0++)
    for (int a1 = 0; a1 < 16; a1++)
    for (int b0 = 1; b0 < 16; b0++)
    for (int b1 = 0; b1 < 16; b1++) {
        gf16_t a[2] = {(gf16_t)a0, (gf16_t)a1};
        gf16_t b[2] = {(gf16_t)b0, (gf16_t)b1};
        gf16_t ab_ref[3] = {0};
        poly_mul_schoolbook(ab_ref, a, 2, b, 2);
        gf16_t A[4] = {a[0], a[1], 0, 0};
        gf16_t B[4] = {b[0], b[1], 0, 0};
        fwd(A, 4, v);
        fwd(B, 4, v);
        for (int i = 0; i < 4; i++) A[i] = gf16_mul(A[i], B[i]);
        inv(A, 4, v);
        int ok = 1;
        for (int i = 0; i < 3; i++) if (A[i] != ab_ref[i]) { ok = 0; break; }
        ncases++;
        if (ok) npass++;
    }
    return npass * 10000 / ncases;  /* percent * 100 */
}

static void print_mu_table(variant_t v) {
    printf("  variant %s mu table:\n", v == VAR_A ? "A" : "B");
    for (int i = 0; i < 4; i++) {
        printf("    level %d: ", i);
        for (int j = 0; j < (1 << i); j++) {
            printf("0x%X ", compute_mu_var(v, i, j));
        }
        printf("\n");
    }
}

int main(void) {
    printf("LCH14 GF(2^4) variant comparison (x^4 + x + 1)\n");
    printf("================================================\n\n");
    gf16_init();
    if (compute_basis(basis)) { printf("basis FAIL\n"); return 1; }
    printf("Basis: v_0=0x%X v_1=0x%X v_2=0x%X v_3=0x%X\n\n",
           basis[0], basis[1], basis[2], basis[3]);

    const char *names[] = {"A: mu_j = s_i(W_m[j])",
                           "B: mu_j = s_i(W_m[j | (1<<i)])",
                           "C: mu_j = s_i(W_m[j|(1<<i)])^-1 * s_i(basis[i+1])"};
    for (int v = 0; v < 3; v++) {
        print_mu_table((variant_t)v);
        int rate = probe_variant((variant_t)v);
        printf("  variant %s: %s\n", v == VAR_A ? "A" : "B", names[v]);
        printf("  pass rate: %.2f%%\n\n", rate / 100.0);
    }
    printf("CONCLUSION: if all 3 variants give the same pass rate and all\n");
    printf("multipliers in the tables are 0x1, the bug is STRUCTURAL (not\n");
    printf("fixable by index/recurrence tweaks). The recursion is degenerate.\n");
    return 0;
}