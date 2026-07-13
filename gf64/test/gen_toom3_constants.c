/*
 * gen_toom3_constants.c — compute Toom-Cook 3 interpolation constants for
 *   evaluation points {0, 1, 2, 3, infinity} over GF(2^64).
 *
 * Run as a one-shot generator to embed in gf64_poly_mul_toom3.c.
 *
 * Build & run (Linux/WSL):
 *   cd gf64/test
 *   gcc -O2 -march=native -I.. gen_toom3_constants.c \
 *       ../gf64_single.c -o gen_toom3_constants && ./gen_toom3_constants
 */

#include <stdio.h>
#include <stdint.h>
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

static gf64_t gf64_pow_u64(gf64_t base, uint64_t exponent) {
    gf64_t result = 1;
    while (exponent != 0) {
        if ((exponent & 1U) != 0) {
            result = gf64_mul_reference(result, base);
        }
        exponent >>= 1;
        if (exponent != 0) {
            base = gf64_mul_reference(base, base);
        }
    }
    return result;
}

static void mat_inv5(gf64_t inv[5][5]) {
    gf64_t s[5] = { 0, 1, 2, 3, 0 };
    gf64_t V[5][5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (i == 4) {
                V[i][j] = (j == 4) ? 1 : 0;
            } else {
                gf64_t v = 1;
                for (int k = 0; k < j; k++) {
                    v = gf64_mul_reference(v, s[i]);
                }
                V[i][j] = v;
            }
        }
    }

    gf64_t M[5][10];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) M[i][j] = V[i][j];
        for (int j = 5; j < 10; j++) M[i][j] = (i == (j - 5)) ? 1 : 0;
    }

    for (int col = 0; col < 5; col++) {
        int pivot = -1;
        for (int r = col; r < 5; r++) {
            if (M[r][col] != 0) { pivot = r; break; }
        }
        if (pivot < 0) {
            printf("Singular matrix at col %d!\n", col);
            return;
        }
        if (pivot != col) {
            for (int j = 0; j < 10; j++) {
                gf64_t t = M[col][j];
                M[col][j] = M[pivot][j];
                M[pivot][j] = t;
            }
        }
        gf64_t pv = M[col][col];
        gf64_t pv_inv = gf64_pow_u64(pv, UINT64_MAX - 1ULL);
        for (int j = 0; j < 10; j++) {
            M[col][j] = gf64_mul_reference(M[col][j], pv_inv);
        }
        for (int r = 0; r < 5; r++) {
            if (r == col) continue;
            gf64_t factor = M[r][col];
            if (factor == 0) continue;
            for (int j = 0; j < 10; j++) {
                M[r][j] ^= gf64_mul_reference(factor, M[col][j]);
            }
        }
    }

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            inv[i][j] = M[i][j + 5];
        }
    }

    printf("\n=== Verification: V * V^{-1} ===\n");
    int ok = 1;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            gf64_t s = 0;
            for (int k = 0; k < 5; k++) {
                s ^= gf64_mul_reference(V[i][k], inv[k][j]);
            }
            gf64_t expected = (i == j) ? 1 : 0;
            if (s != expected) {
                printf("  MISMATCH at [%d][%d]\n", i, j);
                ok = 0;
            }
        }
    }
    if (ok) printf("  All 25 entries OK.\n");
}

int main(void) {
    gf64_t inv[5][5];
    mat_inv5(inv);

    printf("\n=== Toom-Cook 3 interpolation matrix V^{-1} ===\n");
    printf("static const gf64_t GF64_TOOM3_VINV[5][5] = {\n");
    for (int i = 0; i < 5; i++) {
        printf("    { ");
        for (int j = 0; j < 5; j++) {
            printf("0x%016llxULL%s",
                   (unsigned long long)inv[i][j],
                   (j < 4) ? ", " : "");
        }
        printf(" }%s\n", (i < 4) ? "," : "");
    }
    printf("};\n");

    return 0;
}