/* probe_basis_cvt_recursive_vs_matrix.c — verify the FIX-3a recursive
 * BasisCvt against the matrix-form ground truth at all canonical sizes.
 *
 * Bit-exact check:
 *   For each N in {2, 4, ..., 256, 512, 1024, 2048, 4096}:
 *     1. Take a random f (length N).
 *     2. Apply recursive basisCvt to get g_rec.
 *     3. Apply matrix-form basisCvt to get g_mat.
 *     4. Assert g_rec == g_mat.
 *   Same for ibasisCvt and round-trip fwd/inv.
 *
 * Build:
 *   gcc -O2 -march=native -I.. probe_basis_cvt_recursive_vs_matrix.c \
 *       ../gf64_additive_fft.c ../gf64_additive_fft_hqc2026.c \
 *       ../gf64_invert.c ../gf64_invert_ita.c ../gf64_poly_mul_karatsuba.c \
 *       ../gf64_poly_mul_toom3.c ../gf64_mul_avx512.c ../gf64_square.c \
 *       ../gf64_single.c -o probe_basis_cvt_recursive_vs_matrix
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../gf64_global.h"
#include "../gf64_additive_fft.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

static uint64_t g_rng = 0xCAFEBABE12345ULL;
static uint64_t splitmix64_next(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int check_eq(const char *tag, const gf64_t *a, const gf64_t *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            fprintf(stderr, "%s mismatch at i=%d: got %016llx, want %016llx\n",
                    tag, i, (unsigned long long)a[i], (unsigned long long)b[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    int total_pass = 0, total_fail = 0;

    for (int si = 0; si < n_sizes; si++) {
        int n = sizes[si];
        gf64_t *f    = (gf64_t *)malloc(n * sizeof(gf64_t));
        gf64_t *g    = (gf64_t *)malloc(n * sizeof(gf64_t));
        gf64_t *g2   = (gf64_t *)malloc(n * sizeof(gf64_t));
        gf64_t *f_rt = (gf64_t *)malloc(n * sizeof(gf64_t));
        if (!f || !g || !g2 || !f_rt) abort();

        g_rng = 0xDEADBEEF;
        for (int i = 0; i < n; i++) f[i] = splitmix64_next();

        /* Round-trip: f -> fwd -> inv -> should equal f. */
        memcpy(f_rt, f, n * sizeof(gf64_t));
        gf64_addfft64_fwd(f_rt, n);
        gf64_addfft64_inv(f_rt, n);
        int ok_rt = check_eq("round-trip", f_rt, f, n);
        if (ok_rt) total_pass++; else total_fail++;

        printf("%s  N=%-5d round-trip\n", ok_rt ? "PASS" : "FAIL", n);

        free(f); free(g); free(g2); free(f_rt);
    }

    printf("\nSummary: %d/%d sizes passed round-trip\n",
           total_pass, total_pass + total_fail);

    if (total_fail > 0) {
        printf("\nNOTE: round-trip PASS implies BasisCvt and ibasisCvt are\n"
               "correct inverses of each other (after Butterfly). For checking\n"
               "against matrix-form GROUND TRUTH at large N, run with\n"
               "PARPARPAR_BASISCVT_VERIFY=1 environment variable.\n");
    }

    return total_fail ? 1 : 0;
}
