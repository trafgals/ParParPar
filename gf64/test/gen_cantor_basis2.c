/*
 * gen_cantor_basis2.c — find Cantor basis v_0..v_{m-1} for GF(2^64)
 *   (NIST polynomial x^64 + x^4 + x^3 + x + 1).
 *
 * Solves y^2 + y = c via linear algebra over GF(2) and picks each v_i
 * to be the solution NOT in the F_2-span of v_0..v_{i-1} (preserves
 * linear independence of the resulting basis).
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

static gf64_t poly_mul(gf64_t a, gf64_t b) {
    /* Use 128-bit accumulator and reduce fully. For each set bit at
     * position p in the accumulator where p >= 64, substitute
     * X^p = X^(p-64) * (X^4+X^3+X+1) and accumulate. */
    __uint128_t acc = 0;
    for (int i = 0; i < 64; i++) {
        if ((a >> i) & 1) {
            for (int j = 0; j < 64; j++) {
                if ((b >> j) & 1) {
                    int p = i + j;
                    if (p < 64) {
                        acc ^= ((__uint128_t)1) << p;
                    } else {
                        int q = p - 64;
                        acc ^= ((__uint128_t)1) << (q + 4);
                        acc ^= ((__uint128_t)1) << (q + 3);
                        acc ^= ((__uint128_t)1) << (q + 1);
                        acc ^= ((__uint128_t)1) << q;
                    }
                }
            }
        }
    }
    /* Now reduce bits at positions 64..127. Each iteration handles the
     * highest set bit, replacing X^(64+k) with X^k * (X^4+X^3+X+1).
     * Iterate until no bits above 63 remain. */
    while (acc >> 64) {
        int p = 127;
        while (p >= 64 && !((acc >> p) & 1)) p--;
        if (p < 64) break;
        int q = p - 64;
        acc ^= ((__uint128_t)1) << p;  /* clear the high bit */
        acc ^= ((__uint128_t)1) << (q + 4);
        acc ^= ((__uint128_t)1) << (q + 3);
        acc ^= ((__uint128_t)1) << (q + 1);
        acc ^= ((__uint128_t)1) << q;
    }
    return (gf64_t)acc;
}

static gf64_t apply_sq_plus_id(gf64_t a_bits) {
    return poly_mul(a_bits, a_bits) ^ a_bits;
}

static gf64_t solve_artinschreier(gf64_t c) {
    uint64_t M[64][2];
    memset(M, 0, sizeof(M));
    for (int j = 0; j < 64; j++) {
        gf64_t ej = (gf64_t)1 << j;
        gf64_t col = apply_sq_plus_id(ej);
        for (int i = 0; i < 64; i++) {
            M[i][0] |= (((uint64_t)(col >> i)) & 1) << j;
        }
    }
    for (int i = 0; i < 64; i++) {
        M[i][1] = ((uint64_t)(c >> i)) & 1;
    }
    int row = 0;
    int pivot_col[64];
    for (int i = 0; i < 64; i++) pivot_col[i] = -1;

    for (int col = 0; col < 64 && row < 64; col++) {
        int pr = -1;
        for (int r = row; r < 64; r++) {
            if ((M[r][0] >> col) & 1) { pr = r; break; }
        }
        if (pr < 0) continue;
        if (pr != row) {
            uint64_t t0 = M[pr][0], t1 = M[pr][1];
            M[pr][0] = M[row][0]; M[pr][1] = M[row][1];
            M[row][0] = t0; M[row][1] = t1;
        }
        pivot_col[row] = col;
        for (int r = 0; r < 64; r++) {
            if (r == row) continue;
            if ((M[r][0] >> col) & 1) {
                M[r][0] ^= M[row][0];
                M[r][1] ^= M[row][1];
            }
        }
        row++;
    }
    /* Debug: dump pivot columns and check rank. */
    int has_pivot[64] = {0};
    for (int r = 0; r < row; r++) if (pivot_col[r] >= 0) has_pivot[pivot_col[r]] = 1;
    int missing = -1;
    for (int c = 0; c < 64; c++) if (!has_pivot[c]) { missing = c; break; }

    int failed = 0;
    for (int r = row; r < 64; r++) {
        if (M[r][0] == 0 && M[r][1] != 0) failed = 1;
    }
    if (failed) {
        printf("    [debug] row=%d, c=0x%016llx, free_col=%d, M[r][0]=%016llx, M[r][1]=%llx\n",
               row, (unsigned long long)c, missing,
               (unsigned long long)M[row][0], (unsigned long long)M[row][1]);
        return 0;
    }
    gf64_t sol = 0;
    for (int r = 0; r < row; r++) {
        if (M[r][1] & 1) {
            sol |= (gf64_t)1 << pivot_col[r];
        }
    }
    return sol;
}

static int in_span(gf64_t y, gf64_t *v, int n) {
    uint64_t M[64][2];
    memset(M, 0, sizeof(M));
    for (int j = 0; j < n && j < 64; j++) {
        for (int i = 0; i < 64; i++) {
            M[i][0] |= (((uint64_t)(v[j] >> i)) & 1) << j;
        }
    }
    for (int i = 0; i < 64; i++) {
        M[i][1] = ((uint64_t)(y >> i)) & 1;
    }
    int row = 0;
    for (int col = 0; col < n && row < 64; col++) {
        int pr = -1;
        for (int r = row; r < 64; r++) {
            if ((M[r][0] >> col) & 1) { pr = r; break; }
        }
        if (pr < 0) continue;
        if (pr != row) {
            uint64_t t0 = M[pr][0], t1 = M[pr][1];
            M[pr][0] = M[row][0]; M[pr][1] = M[row][1];
            M[row][0] = t0; M[row][1] = t1;
        }
        for (int r = 0; r < 64; r++) {
            if (r == row) continue;
            if ((M[r][0] >> col) & 1) {
                M[r][0] ^= M[row][0];
                M[r][1] ^= M[row][1];
            }
        }
        row++;
    }
    for (int r = row; r < 64; r++) {
        if (M[r][0] == 0 && M[r][1] != 0) return 0;
    }
    return 1;
}

static gf64_t trace(gf64_t c) {
    gf64_t t = 0;
    gf64_t s = c;
    for (int i = 0; i < 64; i++) {
        t ^= s;
        s = poly_mul(s, s);
    }
    return t;
}

int main(void) {
    printf("Cantor basis for GF(2^64) via Artin-Schreier linear algebra\n");
    printf("Irreducible: x^64 + x^4 + x^3 + x + 1\n\n");

    gf64_t y = solve_artinschreier(1);
    if (y == 0) { printf("No solution to y^2 + y = 1!\n"); return 1; }
    printf("v[1] = 0x%016llx  Tr=0x%016llx\n",
           (unsigned long long)y, (unsigned long long)trace(y));

    gf64_t v[20];
    v[0] = 1;
    v[1] = y;
    for (int i = 2; i < 20; i++) {
        gf64_t cand = solve_artinschreier(v[i-1]);
        gf64_t tr_prev = trace(v[i-1]);
        printf("  Tr(v[%d]) = 0x%016llx (must be 0 for next step)\n",
               i-1, (unsigned long long)tr_prev);
        if (cand == 0) {
            printf("FAILED: no solution to v[%d]^2 + v[%d] = v[%d]\n", i, i, i-1);
            return 1;
        }
        if (in_span(cand, v, i)) cand ^= 1;
        v[i] = cand;
        gf64_t chk = poly_mul(v[i], v[i]) ^ v[i];
        if (chk != v[i-1]) {
            printf("v[%d]^2 + v[%d] verification failed!\n", i, i);
            return 1;
        }
        printf("v[%2d] = 0x%016llx  Tr=0x%016llx\n",
               i, (unsigned long long)v[i], (unsigned long long)trace(v[i]));
    }

    /* s_i(v_i) verification and W = span(v_0..v_{m-1}) enumeration. */
    printf("\ns_i(v_i) (should all be 1):\n");
    for (int i = 0; i < 14; i++) {
        gf64_t s_val = v[i];
        for (int k = 0; k < i; k++) s_val = poly_mul(s_val, s_val) ^ s_val;
        printf("s_%d(v_%d) = 0x%016llx\n", i, i, (unsigned long long)s_val);
    }

    /* Compute all elements of W_m = span(v_0..v_{m-1}) for m=4 (sanity check). */
    printf("\nW_4 = span(v_0..v_3):\n");
    int m = 4;
    int n_pts = 1 << m;
    for (int idx = 0; idx < n_pts; idx++) {
        gf64_t w = 0;
        for (int k = 0; k < m; k++) {
            if ((idx >> k) & 1) w ^= v[k];
        }
        printf("  W[%2d] = 0x%016llx\n", idx, (unsigned long long)w);
    }

    return 0;
}