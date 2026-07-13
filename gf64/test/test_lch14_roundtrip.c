/* Quick round-trip test for LCH14 FFT */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../gf64_global.h"

extern void gf64_fft_forward_lch14(gf64_t *arr, size_t n);
extern void gf64_fft_inverse_lch14(gf64_t *arr, size_t n);

int main(void) {
    /* Test: forward then inverse should recover input. */
    size_t n = 4;
    gf64_t *arr = (gf64_t *)calloc(n, sizeof(gf64_t));
    arr[0] = 0xDEADBEEF;
    arr[1] = 0xCAFEBABE;
    arr[2] = 0x12345678;
    arr[3] = 0x9ABCDEF0;

    gf64_t *orig = (gf64_t *)malloc(n * sizeof(gf64_t));
    memcpy(orig, arr, n * sizeof(gf64_t));

    printf("Input:  ");
    for (size_t i = 0; i < n; i++) printf("0x%016llx ", (unsigned long long)arr[i]);
    printf("\n");

    gf64_fft_forward_lch14(arr, n);
    printf("After forward: ");
    for (size_t i = 0; i < n; i++) printf("0x%016llx ", (unsigned long long)arr[i]);
    printf("\n");

    gf64_fft_inverse_lch14(arr, n);
    printf("After inverse: ");
    for (size_t i = 0; i < n; i++) printf("0x%016llx ", (unsigned long long)arr[i]);
    printf("\n");

    printf("Expected:     ");
    for (size_t i = 0; i < n; i++) printf("0x%016llx ", (unsigned long long)orig[i]);
    printf("\n");

    int diffs = 0;
    for (size_t i = 0; i < n; i++) {
        if (arr[i] != orig[i]) diffs++;
    }
    printf("Round-trip diffs: %d / %zu\n", diffs, n);

    free(arr); free(orig);
    return diffs;
}