/*
 * test_gf64_fenger_padded.c — K5 padding parity test (issue #46).
 *
 * Verifies that gf64_fenger_prepare_padded with NON-power-of-2 real
 * counts (padded up to the next power of 2 with synthetic zero-data
 * inputs) produces bit-exact output vs the explicit Cauchy reference:
 *
 *   out[r][w] = XOR_c in[c][w] / (y_r XOR x_c)
 *
 * for N_real / R_real in {3, 5, 6, 10, 17, 100} — shapes that the
 * original power-of-2-only Fenger path could NOT handle (it fell back
 * to the legacy Cauchy kernel). The synthetic input points are placed
 * at syntheticBase = firstRecovery + R_padded (disjoint from both the
 * real inputs and the recovery range, as the engine does).
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../gf64_global.h"
#include "../gf64_fenger.h"
#include "../gf64_mpe.h"
#include "../gf64_subproduct.h"
#include "../gf64_barycentric.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);
extern gf64_t gf64_inverse(gf64_t a);

static uint64_t g_rng = 0x123456789ABCDEFULL;
static uint64_t splitmix64_next(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static size_t next_pow2_ge(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void cauchy_reference(
    const gf64_t *in, size_t N, size_t B,
    gf64_t *out, size_t R,
    uint64_t firstInput, uint64_t firstRecovery)
{
    for (size_t r = 0; r < R; r++) {
        gf64_t yr = (gf64_t)(firstRecovery + (uint64_t)r);
        for (size_t w = 0; w < B; w++) {
            gf64_t acc = 0;
            for (size_t c = 0; c < N; c++) {
                gf64_t xc = (gf64_t)(firstInput + (uint64_t)c);
                gf64_t denom = xc ^ yr;
                if (denom == 0) denom = 1;
                acc ^= gf64_mul_reference(in[c * B + w], gf64_inverse(denom));
            }
            out[r * B + w] = acc;
        }
    }
}

static int run_case(size_t N, size_t R, size_t B,
                    uint64_t firstInput, uint64_t firstRecovery)
{
    const size_t N_pad = next_pow2_ge(N);
    const size_t R_pad = next_pow2_ge(R);
    const uint64_t syntheticBase = firstRecovery + (uint64_t)R_pad;

    gf64_t *in  = (gf64_t *)malloc(N * B * sizeof(gf64_t));
    gf64_t *ref = (gf64_t *)malloc(R * B * sizeof(gf64_t));
    gf64_t *got = (gf64_t *)malloc(R * B * sizeof(gf64_t));
    if (!in || !ref || !got) { fprintf(stderr, "OOM\n"); exit(1); }

    for (size_t i = 0; i < N * B; i++) in[i] = splitmix64_next();

    cauchy_reference(in, N, B, ref, R, firstInput, firstRecovery);

    gf64_fenger_ctx *ctx = gf64_fenger_prepare_padded(
        firstInput, firstRecovery, N, R, N_pad, R_pad, syntheticBase);
    if (!ctx) { fprintf(stderr, "prepare_padded returned NULL\n"); exit(1); }
    gf64_fenger_execute(ctx, in, B, got, 0, B);
    gf64_fenger_release(ctx);

    int bad = 0;
    for (size_t i = 0; i < R * B; i++) {
        if (got[i] != ref[i]) {
            if (bad < 5) {
                fprintf(stderr,
                    "  MISMATCH N=%zu R=%zu B=%zu [%zu]: got %016llx want %016llx\n",
                    N, R, B, i,
                    (unsigned long long)got[i], (unsigned long long)ref[i]);
            }
            bad++;
        }
    }

    free(in); free(ref); free(got);

    if (bad == 0) {
        printf("  PASS: N=%zu R=%zu B=%zu firstInput=%llu firstRecovery=%llu (pad %zux%zu)\n",
               N, R, B, (unsigned long long)firstInput,
               (unsigned long long)firstRecovery, N_pad, R_pad);
        return 0;
    }
    printf("  FAIL: N=%zu R=%zu B=%zu (%d mismatches)\n", N, R, B, bad);
    return 1;
}

int main(void) {
    printf("Fenger padded (K5) parity test — non-power-of-2 vs Cauchy\n\n");

    int fails = 0;
    /* Non-power-of-2 input AND recovery counts. firstInput=0,
     * firstRecovery=N keeps real inputs and recovery disjoint
     * (canonical create layout). */
    fails += run_case(3,   2,   4,  0,   3);
    fails += run_case(5,   3,   8,  0,   5);
    fails += run_case(6,   4,   8,  0,   6);
    fails += run_case(10,  6,  16,  0,  10);
    fails += run_case(17,  9,  16,  0,  17);
    fails += run_case(100, 12, 32,  0, 100);
    fails += run_case(1000, 48, 64, 0, 1000);
    /* Padded recovery only (inputs already power-of-2). */
    fails += run_case(8,  100, 64, 0, 8);
    fails += run_case(16, 48,  64, 0, 16);

    printf("\n%s: %d case(s) failed\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
