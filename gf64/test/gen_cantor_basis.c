/*
 * gen_cantor_basis.c — compute Cantor basis vectors v_0, ..., v_{m-1}
 *   for GF(2^64) with irreducible polynomial x^64 + x^4 + x^3 + x + 1.
 *
 * The Cantor basis satisfies:
 *   v_0 = 1
 *   v_{i+1}^2 + v_{i+1} = v_i
 *
 * So v_1 is a solution to y^2 + y = 1 in GF(2^64), and so on.
 *
 * Run:
 *   cd gf64/test
 *   gcc -O2 -march=native -I.. gen_cantor_basis.c ../gf64_single.c \
 *       -o gen_cantor_basis && ./gen_cantor_basis
 */

#include <stdio.h>
#include <stdint.h>
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* square in GF(2^64): x -> x^2 via the polynomial reduction.
 * Use 128-bit accumulator to avoid undefined behavior on shifts >= 64. */
static gf64_t gf64_sq(gf64_t x) {
    __uint128_t r = 0;
    for (int i = 0; i < 64; i++) {
        if ((x >> i) & 1ULL) {
            int j = 2 * i;
            if (j < 64) {
                r ^= ((__uint128_t)1) << j;
            } else {
                /* X^64 = X^4 + X^3 + X + 1 (irreducible). */
                int k = j - 64;
                r ^= ((__uint128_t)1) << (k + 4);
                r ^= ((__uint128_t)1) << (k + 3);
                r ^= ((__uint128_t)1) << (k + 1);
                r ^= ((__uint128_t)1) << k;
            }
        }
    }
    /* Truncate to 64 bits (the upper 64 bits should be 0 since we
     * reduced all terms via X^64 = X^4 + X^3 + X + 1, giving degree < 64). */
    return (gf64_t)(r & 0xFFFFFFFFFFFFFFFFULL);
}

/* Find v such that v^2 + v = c in GF(2^64) by random search.
 * Returns 0 if no solution found. */
static gf64_t find_artinschreier(gf64_t c) {
    /* Tr(c) must be 0 for solutions to exist. */
    gf64_t tr = c;
    for (int i = 0; i < 6; i++) tr ^= gf64_mul_reference(tr, tr);  /* Tr = sum of squarings */
    if (tr != 0) {
        printf("Tr(c)=1, no solution exists!\n");
        return 0;
    }

    uint64_t seed = 0x1234567890ABCDEFULL;
    for (int trial = 0; trial < 10000; trial++) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        gf64_t a = seed;
        /* Use gf64_mul_reference for squaring to ensure agreement with
         * the verification step. */
        gf64_t sq = gf64_mul_reference(a, a);
        if ((sq ^ a) == c) return a;
        gf64_t a2 = a ^ 1;
        gf64_t sq2 = gf64_mul_reference(a2, a2);
        if ((sq2 ^ a2) == c) return a2;
    }
    return 0;
}

int main(void) {
    printf("Computing Cantor basis for GF(2^64)\n");
    printf("Irreducible: x^64 + x^4 + x^3 + x + 1\n\n");

    gf64_t v[20];
    v[0] = 1;
    printf("v[0] = 0x%016llx\n", (unsigned long long)v[0]);

    for (int i = 1; i < 20; i++) {
        v[i] = find_artinschreier(v[i-1]);
        if (v[i] == 0) {
            printf("FAILED to find v[%d]\n", i);
            return 1;
        }
        gf64_t check = gf64_mul_reference(v[i], v[i]) ^ v[i];
        if (check != v[i-1]) {
            printf("v[%d]^2 + v[%d] = 0x%016llx != v[%d] = 0x%016llx\n",
                   i, i, (unsigned long long)check, i-1, (unsigned long long)v[i-1]);
            return 1;
        }
        printf("v[%2d] = 0x%016llx  (verified)\n", i, (unsigned long long)v[i]);
    }

    /* Also compute s_i(2^i) for i = 0..13 (denominators). */
    printf("\ns_i(2^i) values:\n");
    for (int i = 0; i < 14; i++) {
        gf64_t two_to_i = (i == 0) ? 1 : (1ULL << i);
        gf64_t s_val = two_to_i;
        for (int k = 0; k < i; k++) {
            s_val = gf64_mul_reference(s_val, s_val) ^ s_val;
        }
        printf("s_%d(2^%d) = 0x%016llx\n",
               i, i, (unsigned long long)s_val);
    }

    return 0;
}