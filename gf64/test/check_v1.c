#include <stdio.h>
#include <stdint.h>
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

int main(void) {
    gf64_t v0 = 0x0000000000000001ULL;
    gf64_t v1 = 0x19c9369f278adc02ULL;
    gf64_t v2 = 0xa181e7d66f5ff794ULL;
    gf64_t v3 = 0x5db84357ce785d08ULL;
    gf64_t v4 = 0xb973d466f5c9d0caULL;
    gf64_t v5 = 0x521ac889831a075eULL;

    gf64_t v1_sq = gf64_mul_reference(v1, v1);
    printf("v1^2 + v1 = 0x%016llx (should be 1)\n",
           (unsigned long long)(v1_sq ^ v1));

    gf64_t v1_cubed = gf64_mul_reference(v1, v1_sq);
    printf("v1^3 = 0x%016llx (should be 1 if v1 is a primitive cube root of 1)\n",
           (unsigned long long)v1_cubed);

    /* Test if recurrence matches product formula for s_3, s_4, s_5.
     * For s_3 = s_2 * s_2(x + v_2), need v_2^4 + v_2 = 1.
     * For s_4 = s_3 * s_3(x + v_3), need v_3^4 + v_3 = (something).
     * The condition is: v_k^2 + v_k = v_{k-1} (by definition).
     * Also: v_{k+1}^4 + v_{k+1} = v_k^4 + v_k (???) — actually I need to derive.
     *
     * For s_{k+1} = s_k * s_k(x + v_{k+1}), need s_k(x + v_{k+1}) = s_k(x) + c_k
     * for some constant c_k. Since s_k(x + v) = (x + v)^{2^k} + ... = x^{2^k} + v^{2^k} + ...,
     * the "shift" is v^{2^k} + v (the Frobenius-linear part).
     *
     * For the recurrence s_{k+1} = s_k^2 + s_k to match, we need
     * v_{k+1}^{2^k} + v_{k+1} = 1 (the constant term of s_k).
     *
     * For k=2 (s_2 has constant term 0 since s_2(0) = 0):
     *   need v_3^{4} + v_3 = 0, i.e., v_3^4 = v_3.
     * For k=3 (s_3 has constant term 0):
     *   need v_4^8 + v_4 = 0, i.e., v_4^8 = v_4.
     *
     * In general: v_{k+1}^{2^k} = v_{k+1} (Frobenius-fixed element).
     * This means v_{k+1} ∈ GF(2^k).
     */
    printf("\nFor s_{k+1} = s_k * s_k(x + v_{k+1}) to equal s_k^2 + s_k (recurrence),\n");
    printf("we need v_{k+1}^{2^k} = v_{k+1}, i.e., v_{k+1} in GF(2^k).\n\n");

    /* Check v_2: in GF(2^2)? v_2^4 should equal v_2. */
    gf64_t v2_sq = gf64_mul_reference(v2, v2);
    gf64_t v2_sq_sq = gf64_mul_reference(v2_sq, v2_sq);
    printf("v_2    = 0x%016llx\n", (unsigned long long)v2);
    printf("v_2^2  = 0x%016llx\n", (unsigned long long)v2_sq);
    printf("v_2^4  = 0x%016llx\n", (unsigned long long)v2_sq_sq);
    printf("v_2^4 == v_2? %s\n\n", v2_sq_sq == v2 ? "YES (v_2 in GF(2^2))" : "NO");

    /* Check v_3: in GF(2^3)? v_3^8 should equal v_3. */
    gf64_t v3_sq = gf64_mul_reference(v3, v3);
    gf64_t v3_4 = gf64_mul_reference(v3_sq, v3_sq);
    gf64_t v3_8 = gf64_mul_reference(v3_4, v3_4);
    printf("v_3    = 0x%016llx\n", (unsigned long long)v3);
    printf("v_3^4  = 0x%016llx\n", (unsigned long long)v3_4);
    printf("v_3^8  = 0x%016llx\n", (unsigned long long)v3_8);
    printf("v_3^8 == v_3? %s\n\n", v3_8 == v3 ? "YES (v_3 in GF(2^3))" : "NO");

    /* Check v_4: in GF(2^4)? v_4^16 should equal v_4. */
    gf64_t v4_sq = gf64_mul_reference(v4, v4);
    gf64_t v4_4 = gf64_mul_reference(v4_sq, v4_sq);
    gf64_t v4_8 = gf64_mul_reference(v4_4, v4_4);
    gf64_t v4_16 = gf64_mul_reference(v4_8, v4_8);
    printf("v_4    = 0x%016llx\n", (unsigned long long)v4);
    printf("v_4^16 = 0x%016llx\n", (unsigned long long)v4_16);
    printf("v_4^16 == v_4? %s\n\n", v4_16 == v4 ? "YES (v_4 in GF(2^4))" : "NO");

    /* Check v_5: in GF(2^5)? v_5^32 should equal v_5. */
    gf64_t v5_sq = gf64_mul_reference(v5, v5);
    gf64_t v5_4 = gf64_mul_reference(v5_sq, v5_sq);
    gf64_t v5_8 = gf64_mul_reference(v5_4, v5_4);
    gf64_t v5_16 = gf64_mul_reference(v5_8, v5_8);
    gf64_t v5_32 = gf64_mul_reference(v5_16, v5_16);
    printf("v_5    = 0x%016llx\n", (unsigned long long)v5);
    printf("v_5^32 = 0x%016llx\n", (unsigned long long)v5_32);
    printf("v_5^32 == v_5? %s\n\n", v5_32 == v5 ? "YES (v_5 in GF(2^5))" : "NO");

    printf("If all v_i are in their respective GF(2^i), then the recurrence\n");
    printf("s_i = s_{i-1}^2 + s_{i-1} matches the product formula. This means\n");
    printf("the polyeval cvt would match our matrix-form basisCvt.\n");

    return 0;
}