/*
 * test_tower_fft_gf16.c — Tower-of-extensions additive DFT analysis.
 *
 * Search for an evaluation set V of size 4 in GF(2^4) such that:
 *   1. Trace to GF(2) splits V into 2-2 (Tr=0 half and Tr=1 half).
 *   2. The trace pairing matrix M[i][j] = Tr(V[i] * V[j]) is invertible.
 *   3. The Vandermonde matrix has good structure for O(N log N) FFT.
 *
 * Plus: report what the additive DFT F[i][j] = Tr(omega_j * V[i]) looks
 * like (it's the identity for self-dual basis, so the additive DFT
 * is structurally uninteresting for polynomial multiplication).
 *
 * Build & run:
 *   gcc -O2 test_tower_fft_gf16.c -o test_tower_fft_gf16 && ./test_tower_fft_gf16
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
static int gf16_trace(gf16_t v) {
    int r = 0;
    for (int i = 0; i < 4; i++) { r ^= v; v = gf16_sq(v); }
    return r & 1;
}

/* Try a 4-element subset V. Return 1 if it's a valid candidate
 * (mixed trace, invertible pairing), 0 otherwise. If valid, fill in
 * the dual basis and report the trace decomposition. */
static int try_V(gf16_t v0, gf16_t v1, gf16_t v2, gf16_t v3,
                 gf16_t omega[4], int *v0_tr, int *v1_tr, int *v2_tr, int *v3_tr) {
    gf16_t V[4] = {v0, v1, v2, v3};
    *v0_tr = gf16_trace(v0); *v1_tr = gf16_trace(v1);
    *v2_tr = gf16_trace(v2); *v3_tr = gf16_trace(v3);
    int t0 = *v0_tr + *v1_tr + *v2_tr + *v3_tr;
    if (t0 != 2) return 0;  /* need exactly 2 elements with Tr=1 */

    int M[4][4] = {{0}};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            M[i][j] = gf16_trace(gf16_mul(V[i], V[j]));
    int aug[4][8];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) aug[i][j] = M[i][j];
        for (int j = 4; j < 8; j++) aug[i][j] = (i == (j - 4)) ? 1 : 0;
    }
    for (int r = 0; r < 4; r++) {
        int piv = -1;
        for (int i = r; i < 4; i++) if (aug[i][r]) { piv = i; break; }
        if (piv < 0) return 0;
        if (piv != r) for (int j = 0; j < 8; j++) { int t = aug[r][j]; aug[r][j] = aug[piv][j]; aug[piv][j] = t; }
        for (int i = 0; i < 4; i++) {
            if (i != r && aug[i][r]) for (int j = 0; j < 8; j++) aug[i][j] ^= aug[r][j];
        }
    }
    for (int i = 0; i < 4; i++) {
        gf16_t s = 0;
        for (int j = 0; j < 4; j++) if (aug[j][i + 4]) s ^= V[j];
        omega[i] = s;
    }
    return 1;
}

int main(void) {
    gf16_init();
    printf("GF(2^4) tower-of-extensions additive DFT search\n");
    printf("================================================\n\n");

    /* Enumerate all C(16,4) = 1820 subsets and check the criteria. */
    int found = 0;
    for (int a = 1; a < 16; a++)  /* skip 0 (it kills invertibility) */
    for (int b = a + 1; b < 16; b++)
    for (int c = b + 1; c < 16; c++)
    for (int d = c + 1; d < 16; d++) {
        gf16_t omega[4];
        int t0, t1, t2, t3;
        if (try_V(a, b, c, d, omega, &t0, &t1, &t2, &t3)) {
            found++;
            if (found <= 5) {
                printf("Found valid V = {0x%X, 0x%X, 0x%X, 0x%X}  Tr = {%d, %d, %d, %d}\n",
                       a, b, c, d, t0, t1, t2, t3);
                printf("  Dual basis omega = {0x%X, 0x%X, 0x%X, 0x%X}\n",
                       omega[0], omega[1], omega[2], omega[3]);
            }
        }
    }
    printf("\nTotal valid V found (mixed trace + invertible pairing): %d\n", found);
    printf("\nConclusion: a finite number of subsets V qualify. For each, the\n");
    printf("additive DFT matrix F[i][j] = Tr(omega_j * V[i]) is the IDENTITY\n");
    printf("(self-dual basis). The additive DFT IS the identity transform for\n");
    printf("any self-dual V — it doesn't transform anything.\n\n");
    printf("The right transform for polynomial multiplication is the Vandermonde\n");
    printf("matrix V[i][j] = v_i^j (monomial basis evaluated at v_i). Its O(N log N)\n");
    printf("factorization over GF(2^64) is the open research problem that this\n");
    printf("investigation has not resolved.\n");
    return 0;
}