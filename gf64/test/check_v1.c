#include <stdio.h>
#include <stdint.h>
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

int main(void) {
    gf64_t v0 = 0x0000000000000001ULL;
    gf64_t v1 = 0x19c9369f278adc02ULL;
    gf64_t v1_sq = gf64_mul_reference(v1, v1);
    gf64_t v1_sq_plus_v1 = v1_sq ^ v1;
    printf("v1^2 + v1 = 0x%016llx (should be v0 = 1)\n", (unsigned long long)v1_sq_plus_v1);

    gf64_t v1_cubed = gf64_mul_reference(v1, v1_sq);
    printf("v1^3 = 0x%016llx (should be 1 if v1 is a primitive cube root of 1)\n", (unsigned long long)v1_cubed);

    return 0;
}