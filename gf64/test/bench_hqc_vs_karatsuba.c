/*
 * /tmp/bench_hqc_vs_karatsuba.c — benchmark HQC FFT (scalar + AVX-512)
 *   vs Karatsuba for polynomial multiplication at canonical sizes.
 *
 * Sizes tested: 16, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096,
 * 8192, 16384.
 * (Smaller sizes use schoolbook inside Karatsuba; sizes in
 *  {16, 64, ..., 8192, 16384} fall back to matrix-form in HQC FFT.)
 *
 * Run: /tmp/bench_hqc_vs_karatsuba
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <setjmp.h>

#include "gf64_additive_fft.h"

extern void gf64_poly_mul_karatsuba(
    uint64_t *out, const uint64_t *a, size_t la,
    const uint64_t *b, size_t lb, size_t out_len);

/* Host-availability probe for VPCLMULQDQ (cpuid 7.0 ECX bit 10 + XCR0
 * mask 0xE6 — see gf64/cpu_detect.c). The bench's AVX-512 path calls
 * gf64_addfft64_poly_mul_recursive_scratch_avx512 (compiled with
 * __attribute__((target("avx512f,vpclmulqdq")))), which SIGILLs on hosts
 * that lack VPCLMULQDQ even when CPUID reports AVX-512F. Probe directly
 * rather than going through gf64_init_dispatch() — the latter pulls in
 * gf64_region_*_arr TUs the bench doesn't otherwise need.
 *
 * NOTE (cubic review 4914681432 P2): the previous in-file
 * `probe_zmm_executes()` used hardcoded bytes
 *     .byte 0xc4, 0xe3, 0x7d, 0x44, 0xc0, 0x00
 * that actually decoded (per objdump) to `vpclmulqdq ymm0, ymm0, ymm0`
 * — a VEX.256 YMM instruction, not the ZMM/EVEX path the AVX-512
 * kernel actually exercises. The probe therefore could not catch the
 * WSL2/Hyper-V observer effect (YMM path works, ZMM SIGILLs).
 * Replaced with the public one-shot ZMM SIGILL probe from
 * gf64/cpu_detect.c, which runs a real EVEX vpaddd zmm0 instruction
 * (matching `try_zmm_insn()` in cpu_detect.c). The result is cached
 * inside gf64_zmm_probe() so the probe runs once per process. */
extern int gf64_has_vpclmulqdq_probe(void);
extern int gf64_zmm_probe(void);

static uint64_t state = 0xc0ffee01;
static uint64_t splitmix64(void) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double time_one_poly_mul_recursive(size_t n_actual, uint64_t *a, uint64_t *b,
                                          uint64_t *out, uint64_t *scratch,
                                          int use_avx512, int iters,
                                          int host_has_vpclmulqdq) {
    /* Force the AVX-512 path off on hosts that lack VPCLMULQDQ. The
     * host_has_vpclmulqdq flag is computed once in main() and passed
     * in — calling gf64_has_vpclmulqdq_probe() here would re-evaluate
     * CPUID+XCR0 and miss the ZMM SIGILL probe result (cubic review
     * 4914681432 P2). Without this guard, calling the _avx512 entry
     * on a non-AVX-512 host SIGILLs the bench process. */
    if (use_avx512 && !host_has_vpclmulqdq) use_avx512 = 0;

    size_t n_pad = 1;
    while (n_pad < 2 * n_actual - 1) n_pad <<= 1;
    size_t sw = gf64_addfft64_poly_mul_recursive_scratch_words(n_pad);
    for (int i = 0; i < 3; i++) {
        if (use_avx512)
            gf64_addfft64_poly_mul_recursive_scratch_avx512(
                out, a, n_actual, b, n_actual, n_actual, scratch, sw);
        else
            gf64_addfft64_poly_mul_recursive_scratch(
                out, a, n_actual, b, n_actual, n_actual, scratch, sw);
    }
    clock_t t0 = clock();
    for (int i = 0; i < iters; i++) {
        if (use_avx512)
            gf64_addfft64_poly_mul_recursive_scratch_avx512(
                out, a, n_actual, b, n_actual, n_actual, scratch, sw);
        else
            gf64_addfft64_poly_mul_recursive_scratch(
                out, a, n_actual, b, n_actual, n_actual, scratch, sw);
    }
    clock_t t1 = clock();
    return (double)(t1 - t0) * 1000.0 / CLOCKS_PER_SEC / iters;
}

static double time_one_karatsuba(size_t n, uint64_t *a, uint64_t *b,
                                 uint64_t *out, int iters) {
    for (int i = 0; i < 3; i++)
        gf64_poly_mul_karatsuba(out, a, n, b, n, n);
    clock_t t0 = clock();
    for (int i = 0; i < iters; i++)
        gf64_poly_mul_karatsuba(out, a, n, b, n, n);
    clock_t t1 = clock();
    return (double)(t1 - t0) * 1000.0 / CLOCKS_PER_SEC / iters;
}

static double bench_one(size_t n, int iters, int host_has_vpclmulqdq) {
    uint64_t *a = malloc(n * sizeof(uint64_t));
    uint64_t *b = malloc(n * sizeof(uint64_t));
    uint64_t *out = malloc((2*n) * sizeof(uint64_t));
    for (size_t i = 0; i < n; i++) { a[i] = splitmix64(); b[i] = splitmix64(); }
    double ms_k = time_one_karatsuba(n, a, b, out, iters);
    size_t n_pad = 1;
    while (n_pad < 2 * n - 1) n_pad <<= 1;
    size_t sw = gf64_addfft64_poly_mul_recursive_scratch_words(n_pad);
    uint64_t *scratch = malloc(sw * sizeof(uint64_t));
    double ms_hs = time_one_poly_mul_recursive(n, a, b, out, scratch, 0, iters, host_has_vpclmulqdq);
    double ms_ha = time_one_poly_mul_recursive(n, a, b, out, scratch, 1, iters, host_has_vpclmulqdq);
    printf("%-7zu %7.3f ms %7.3f ms %7.3f ms %5.2fx  %5.2fx\n",
           n, ms_k, ms_hs, ms_ha, ms_k / ms_ha, ms_hs / ms_ha);
    free(a); free(b); free(out); free(scratch);
    return ms_k;
}

int main(void) {
    /* Two-stage probe so we can warn up front:
     *   1. CPUID+XCR0 (cheap, ~50 cycles): the historical probe.
     *   2. SIGILL probe (one-shot, catches WSL2/Hyper-V lying CPUID).
     *      Both must succeed for the AVX-512 column to be exercised.
     *      The SIGILL probe runs through gf64_zmm_probe() which
     *      caches its result in gf64_zmm_works; subsequent calls
     *      from elsewhere in the process see the same cached value.
     *      The ZMM probe is gated behind the CPUID probe — if CPUID
     *      already reports no VPCLMULQDQ, skip the SIGILL test. */
    int has_vpclmulqdq_cpuid = gf64_has_vpclmulqdq_probe();
    int zmm_works = has_vpclmulqdq_cpuid ? gf64_zmm_probe() : 0;
    int has_vpclmulqdq = has_vpclmulqdq_cpuid && zmm_works;
    if (has_vpclmulqdq_cpuid && !zmm_works) {
        printf("Note: CPUID+XCR0 advertise VPCLMULQDQ but the actual ZMM "
               "probe SIGILL'd (WSL2/Hyper-V observer effect). Forcing "
               "AVX-512 column to mirror the scalar column.\n\n");
    }

    printf("%-7s %-10s %-10s %-10s %-7s %-7s\n",
           "n", "karatsuba", "hqc-scalar", "hqc-avx512", "vs-kara", "avx-gain");
    printf("%-7s %-10s %-10s %-10s %-7s %-7s\n",
           "-------", "----------", "----------", "----------", "-------", "-------");

    if (!has_vpclmulqdq) {
        printf("\nNote: VPCLMULQDQ not available on this host; AVX-512 "
               "column will mirror the scalar column.\n");
    }

    /* Small crossover sweep */
    size_t small[] = {16, 32, 48, 64, 96, 128, 192};
    for (size_t i = 0; i < sizeof(small)/sizeof(small[0]); i++)
        bench_one(small[i], 10000, has_vpclmulqdq);

    printf("\n--- canonical sizes ---\n");
    /* Canonical sizes */
    size_t big[] = {256, 512, 1024, 2048, 4096, 8192, 16384};
    for (size_t i = 0; i < sizeof(big)/sizeof(big[0]); i++)
        bench_one(big[i], (big[i] <= 512) ? 5000 : (big[i] <= 4096) ? 500 : 100, has_vpclmulqdq);

    return 0;
}
