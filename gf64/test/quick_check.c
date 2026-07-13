#include <stdio.h>
#include <stdint.h>
#define GF16_MOD_POLY 0x13
static uint8_t gf16_exp[16], gf16_log[16];
static void gf16_init_tables(void) {
    int i, x = 1;
    for (i = 0; i < 15; i++) {
        gf16_exp[i] = (uint8_t)x;
        x <<= 1;
        if (x & 0x10) x ^= GF16_MOD_POLY;
        x &= 0xF;
    }
    gf16_exp[15] = gf16_exp[0];
    for (i = 0; i < 15; i++) gf16_log[gf16_exp[i]] = (uint8_t)i;
}
int main(void) {
    gf16_init_tables();
    printf("exp table: ");
    for (int i = 0; i < 15; i++) printf("e[%d]=%X ", i, gf16_exp[i]);
    printf("\nlog table: ");
    for (int i = 1; i < 16; i++) printf("l[%X]=%d ", i, gf16_log[i]);
    printf("\n");
    int basis[4] = {1, 6, 2, 10};
    for (int i = 0; i < 4; i++) {
        int a = basis[i];
        int a2 = gf16_exp[(2 * gf16_log[a]) % 15];
        int sig = a2 ^ a;
        printf("sigma(v_%d) = v_%d^2 + v_%d = %X + %X = %X ; expected v_%d = %X ; match=%d\n",
               i, i, i, a2, a, sig, (i+1) % 4, basis[(i+1) % 4], sig == basis[(i+1) % 4]);
    }
    /* What is s_1(basis[i]) for each? */
    for (int i = 0; i < 4; i++) {
        int a = basis[i];
        int a2 = gf16_exp[(2 * gf16_log[a]) % 15];
        int s1 = a2 ^ a;
        int s2 = gf16_exp[(2 * gf16_log[s1]) % 15] ^ s1;
        int s3 = gf16_exp[(2 * gf16_log[s2]) % 15] ^ s2;
        printf("s_1(v_%d) = %X ; s_2(v_%d) = %X ; s_3(v_%d) = %X\n", i, s1, i, s2, i, s3);
    }
    return 0;
}
