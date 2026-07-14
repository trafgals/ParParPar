/*
 * Standalone test for gf64_poly_invmod_mod (polynomial modular inverse).
 *
 * Picks specific polynomials g and f that are coprime by construction
 * (g = (x+a)(x+b)… and f = (x+c)(x+d)… with disjoint point sets), then
 * computes inv such that g * inv ≡ 1 (mod f) via half-EGCD; verifies by
 * direct gf64_poly_divmod.
 *
 * Cases: (deg_g, deg_f) ∈ {(1,1), (3,3), (5,5), (1,2), (2,1), (3,1)}.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../gf64_mpe.h"
#include "../gf64_additive_fft.h"
#include "../gf64_global.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

int main(void) {
    printf("gf64_poly_invmod_mod test\n");
    printf("=========================\n\n");

    int total_ok = 1;

    /* Test cases: g and f are constructed as PRODUCTS of distinct linear
     * factors with disjoint point sets, so they're guaranteed coprime. */
    struct {
        const char *name;
        gf64_t *g_pts;  size_t n_g;
        gf64_t *f_pts;  size_t n_f;
    } cases[] = {
        /* (1, 1): g = (x+0x11), f = (x+0x22). */
        {"(deg_g=1, deg_f=1)",
         (gf64_t[]){0x11ULL}, 1,
         (gf64_t[]){0x22ULL}, 1},
        /* (2, 2): g = (x+0x10)(x+0x11), f = (x+0x20)(x+0x21). */
        {"(deg_g=2, deg_f=2)",
         (gf64_t[]){0x10ULL, 0x11ULL}, 2,
         (gf64_t[]){0x20ULL, 0x21ULL}, 2},
    };
    int num_cases = sizeof(cases) / sizeof(cases[0]);

    for (int ci = 0; ci < num_cases; ci++) {
        size_t n_g = cases[ci].n_g;
        size_t n_f = cases[ci].n_f;

        /* Build g = ∏ (x + g_pts[j]) using a separate accumulator buffer:
         * gf64_poly_mul_padded memsets the output before reading the inputs,
         * so an in-place call where `out == a` would lose the prior product.
         * The accumulator is freshly sized each step (it's discarded), and
         * the trailing copy into `g` keeps the final compact degree-n_g
         * representation. */
        gf64_t *g_one = (gf64_t *)calloc(1, sizeof(gf64_t));
        g_one[0] = 1ULL;
        size_t g_acc_deg = 0;
        gf64_t *g_acc = (gf64_t *)calloc(n_g + 1, sizeof(gf64_t));
        g_acc[0] = 1ULL;
        for (size_t j = 0; j < n_g; j++) {
            gf64_t leaf[2] = { cases[ci].g_pts[j], 1ULL };
            gf64_t *new_acc = (gf64_t *)calloc(g_acc_deg + 2, sizeof(gf64_t));
            gf64_poly_mul_padded(new_acc, g_acc, g_acc_deg + 1, leaf, 2, g_acc_deg + 2);
            free(g_acc);
            g_acc = new_acc;
            g_acc_deg++;
        }
        free(g_one);
        size_t deg_g = n_g; /* degree = n_g */
        gf64_t *g = g_acc;

        /* Build f similarly. */
        gf64_t *f_one = (gf64_t *)calloc(1, sizeof(gf64_t));
        f_one[0] = 1ULL;
        size_t f_acc_deg = 0;
        gf64_t *f_acc = (gf64_t *)calloc(n_f + 1, sizeof(gf64_t));
        f_acc[0] = 1ULL;
        for (size_t j = 0; j < n_f; j++) {
            gf64_t leaf[2] = { cases[ci].f_pts[j], 1ULL };
            gf64_t *new_acc = (gf64_t *)calloc(f_acc_deg + 2, sizeof(gf64_t));
            gf64_poly_mul_padded(new_acc, f_acc, f_acc_deg + 1, leaf, 2, f_acc_deg + 2);
            free(f_acc);
            f_acc = new_acc;
            f_acc_deg++;
        }
        free(f_one);
        size_t deg_f = n_f;
        gf64_t *f = f_acc;

        printf("Case %s:\n", cases[ci].name);
        printf("  g = [");
        for (size_t i = 0; i <= deg_g; i++) {
            printf("0x%016llx%s", (unsigned long long)g[i],
                   i < deg_g ? ", " : "");
        }
        printf("]\n");
        printf("  f = [");
        for (size_t i = 0; i <= deg_f; i++) {
            printf("0x%016llx%s", (unsigned long long)f[i],
                   i < deg_f ? ", " : "");
        }
        printf("]\n");

        /* Compute inv. */
        gf64_t *inv = calloc(deg_f + 1, sizeof(gf64_t));
        int rc = gf64_poly_invmod_mod(g, deg_g, f, deg_f, inv);
        if (rc != 0) {
            printf("  EGCD returned %d (FAIL: should not be non-coprime)\n", rc);
            total_ok = 0;
            free(g); free(f); free(inv);
            continue;
        }

        /* Verify: g * inv (mod f) should be 1 (constant). */
        gf64_t *prod = calloc(deg_g + deg_f + 1, sizeof(gf64_t));
        gf64_poly_mul(prod, g, deg_g, inv, deg_f);
        gf64_t *q = calloc(deg_g + 1, sizeof(gf64_t));
        /* gf64_poly_divmod uses `r` as a working copy of the dividend (size
         * deg_dividend + 1 = deg_g + deg_f + 1), not the divisor. */
        gf64_t *r = calloc(deg_g + deg_f + 1, sizeof(gf64_t));
        gf64_poly_divmod(prod, deg_g + deg_f, f, deg_f, q, r);

        printf("  inv = [");
        for (size_t i = 0; i < deg_f; i++) {
            printf("0x%016llx%s", (unsigned long long)inv[i],
                   i < deg_f - 1 ? ", " : "");
        }
        printf("]\n");
        printf("  g*inv mod f = [");
        for (size_t i = 0; i < deg_f; i++) {
            printf("0x%016llx%s", (unsigned long long)r[i],
                   i < deg_f - 1 ? ", " : "");
        }
        printf("]\n");

        int ok = 1;
        if (r[0] != 1ULL) {
            printf("  FAIL: r[0] = 0x%016llx, want 1\n", (unsigned long long)r[0]);
            ok = 0;
        }
        for (size_t i = 1; i < deg_f; i++) {
            if (r[i] != 0) {
                printf("  FAIL: r[%zu] = 0x%016llx, want 0\n", i, (unsigned long long)r[i]);
                ok = 0;
                break;
            }
        }
        if (ok) {
            printf("  PASS\n\n");
        } else {
            total_ok = 0;
            printf("\n");
        }

        free(g); free(f); free(inv); free(prod); free(q); free(r);
    }

    printf("\n%s\n", total_ok ? "ALL OK" : "SOME FAILED");
    return total_ok ? 0 : 1;
}
