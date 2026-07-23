/*
 * ============================================================================
 * test_gf64_fenger_kernel.c — Phase 0 sanity check for the Fenger Toeplitz
 *   pipeline wiring (issue #28).
 *
 * Compares two implementations of the Cauchy matrix-vector product over
 * GF(2^64)[x]:
 *
 *   out[r] = XOR_{c=0..N-1} in[c] / (y_r XOR x_c)      (Cauchy matrix form)
 *
 * where x_c = firstInput + c, y_r = firstRecovery + r.
 *
 * The two implementations are:
 *
 *   1. CAUCHY (reference): explicit O(N) per output. For each (r, c),
 *      out[r] ^= in[c] / (y_r XOR x_c). Uses gf64_inverse (T5) and
 *      gf64_mul_reference (T1) for the field ops.
 *
 *   2. FENGER NAIVE: the standard Fenger decomposition
 *
 *        f_w(x) = sum_c in[c] · L_c(x)       (Lagrange interpolation at {x_c})
 *        out[r] = f_w(y_r) / V(y_r)            (V(x) = prod_c (x XOR x_c))
 *
 *      For Phase 0 we compute f_w explicitly via Lagrange basis polynomial
 *      (O(N²) per word) and evaluate at each y_r (O(N) per (w, r)). Then
 *      V(y_r) is computed via Horner (O(N) per r).
 *
 *   The Fenger naive is asymptotically worse than the Cauchy reference
 *   (O(B · N²) vs O(B · N·R)) but is BIT-EXACT for any (N, R) with N ≤ ~64.
 *   For Phase 0 the test runs at small N to verify the Fenger decomposition
 *   is correct; the asymptotic O((D+R)·log²(D+R)) tree-walking form is
 *   Phase 1.
 *
 * Bit-exact parity is checked at multiple (N, R, B) configurations, with
 * random inputs seeded from a fixed PRNG for reproducibility.
 * ============================================================================
 */

#include "../gf64_global.h"
#include "../gf64_invert.h"
#include "../gf64_invert_ita.h"
#include "../gf64_mul.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);
extern gf64_t gf64_inverse(gf64_t a);
extern gf64_t gf64_invert_ita_one(gf64_t a);

/* ----- Reference: explicit Cauchy matrix-vector product -----
 * out[r] = XOR_c in[c] / (y_r XOR x_c) for r in [0, numRecovery).
 * Single block of B gf64_t values, no LRU cache. */
static void cauchy_reference(
    const gf64_t *in, size_t N, size_t B,
    gf64_t *out, size_t R,
    uint64_t firstInput, uint64_t firstRecovery)
{
    /* Per row, accumulate c * in[c][w] / (y_r XOR x_c). For each (r, c),
     * the denominator is gf64_inverse(y_r XOR x_c). */
    for (size_t r = 0; r < R; r++) {
        gf64_t yr = (gf64_t)(firstRecovery + (uint64_t)r);
        gf64_t *out_row = out + r * B;
        /* Initialize out[r][w] = 0 (in char 2, zero is 0). */
        for (size_t w = 0; w < B; w++) out_row[w] = 0;
        for (size_t c = 0; c < N; c++) {
            gf64_t xc = (gf64_t)(firstInput + (uint64_t)c);
            gf64_t denom = gf64_inverse(yr ^ xc);
            const gf64_t *in_row = in + c * B;
            for (size_t w = 0; w < B; w++) {
                out_row[w] ^= gf64_mul_reference(in_row[w], denom);
            }
        }
    }
}

/* ----- V(y_r) = product_{c=0..N-1} (y_r XOR x_c) via Horner.
 * For each r, build V(r) directly via iterated GF(2^64) mul. */
static gf64_t V_eval(uint64_t y_r, uint64_t firstInput, size_t N)
{
    gf64_t v = 1;
    for (size_t c = 0; c < N; c++) {
        uint64_t xc = firstInput + (uint64_t)c;
        v = gf64_mul_reference(v, (gf64_t)(y_r ^ xc));
    }
    return v;
}

/* ----- Lagrange basis polynomial L_c(x) for c=0..N-1 over GF(2^64).
 * Returns the (N+1)-coefficient constant-first array for L_c. Caller owns
 * the buffer. Implements L_c(x) = prod_{k != c} (x XOR x_k) / (x_c XOR x_k).
 * O(N²) per c. */
static void lagrange_basis(
    size_t c, size_t N, uint64_t firstInput, gf64_t *L_out)
{
    /* Numerator polynomial: num(x) = prod_{k != c} (x XOR x_k).
     * This is a degree-(N-1) polynomial; constant-first. */
    gf64_t num[64];
    for (size_t i = 0; i < N; i++) num[i] = 0;
    /* num starts as the constant polynomial 1. Multiply by (x XOR x_k)
     * for each k != c. */
    num[0] = 1;
    int cur_deg = 0;
    for (size_t k = 0; k < N; k++) {
        if (k == c) continue;
        uint64_t xk = firstInput + (uint64_t)k;
        /* Multiply num(x) by (x XOR x_k). Result has degree cur_deg + 1.
         * In char 2: "x XOR xk" = "x + xk" = "x - xk", so
         *   (x + xk) * sum_i num[i] * x^i   accumulates to
         *     coef[i+1] = num[i]                    (from x * num[i] * x^i)
         *     coef[i]   = xk * num[i]               (from xk * num[i] * x^i)
         */
        gf64_t new_num[64] = {0};
        for (int i = 0; i <= cur_deg; i++) {
            new_num[i+1] ^= num[i];                                       /* from x · num[i] */
            new_num[i]   ^= gf64_mul_reference(num[i], (gf64_t)xk);       /* from xk · num[i] */
        }
        for (int i = 0; i < N; i++) num[i] = new_num[i];
        cur_deg++;
    }
    /* Now num is the numerator. Denominator: denom = num(x_c). */
    gf64_t xc = (gf64_t)(firstInput + (uint64_t)c);
    gf64_t denom = 0;
    gf64_t xc_pow = 1;
    for (size_t i = 0; i < N; i++) {
        denom ^= gf64_mul_reference(num[i], xc_pow);
        xc_pow = gf64_mul_reference(xc_pow, xc);
    }
    /* L_c = num / denom. In char 2, division = multiply by inverse. */
    gf64_t denom_inv = gf64_inverse(denom);
    for (size_t i = 0; i < N; i++) {
        L_out[i] = gf64_mul_reference(num[i], denom_inv);
    }
}

/* ----- Fenger naive: f_w(x) = sum_c in[c][w] · V'(x_c) · L_c(x), evaluated
 * at y_r; out[r][w] = f_w(y_r) / V(y_r). This is the correct Fenger 2009
 * decomposition of the Cauchy matrix-vector product:
 *
 *   sum_c in[c] / (y - x_c)  =  p(y) / V(y)
 *
 * where p(x) is the unique polynomial of degree < N satisfying
 *   p(x_c) = in[c] * V'(x_c)
 * and V(x) = prod_c (x - x_c). The V'(x_c) weighting is what turns the
 * Lagrange-basis partial-fraction identity into the bare 1/(y - x_c)
 * summand (the residue of p/V at the pole x_c is in[c]). */
static void fenger_naive(
    const gf64_t *in, size_t N, size_t B,
    gf64_t *out, size_t R,
    uint64_t firstInput, uint64_t firstRecovery)
{
    /* Precompute L_c and V'(x_c) for all c. O(N²) per c, total O(N³). */
    gf64_t *L_basis = (gf64_t *)calloc(N * N, sizeof(gf64_t));
    gf64_t *V_prime = (gf64_t *)calloc(N, sizeof(gf64_t));
    if (L_basis == NULL || V_prime == NULL) abort();
    for (size_t c = 0; c < N; c++) {
        lagrange_basis(c, N, firstInput, L_basis + c * N);
        /* V'(x_c) = prod_{k != c} (x_c - x_k). The denominator computed by
         * lagrange_basis is exactly this quantity (the value at x_c of the
         * numerator polynomial prod_{k != c} (x - x_k)). We saved it via the
         * `denom` local inside lagrange_basis; recompute it here for
         * clarity. */
        gf64_t xc = (gf64_t)(firstInput + (uint64_t)c);
        gf64_t vp = 1;
        for (size_t k = 0; k < N; k++) {
            if (k == c) continue;
            uint64_t xk = firstInput + (uint64_t)k;
            vp = gf64_mul_reference(vp, (gf64_t)(xc ^ xk));
        }
        V_prime[c] = vp;
    }
    /* For each recovery point r, compute V(y_r) once. */
    gf64_t *v_at = (gf64_t *)calloc(R, sizeof(gf64_t));
    if (v_at == NULL) abort();
    for (size_t r = 0; r < R; r++) {
        v_at[r] = V_eval(firstRecovery + (uint64_t)r, firstInput, N);
    }
    /* For each word w: out[r][w] = (sum_c in[c][w] * V'(x_c) * L_c(y_r)) / V(y_r).
     * O(N · R) per word via per-(r,c) Horner on L_c. */
    for (size_t w = 0; w < B; w++) {
        for (size_t r = 0; r < R; r++) {
            gf64_t yr = (gf64_t)(firstRecovery + (uint64_t)r);
            gf64_t acc = 0;
            for (size_t c = 0; c < N; c++) {
                /* Evaluate L_c at yr. L_c is a degree-(N-1) polynomial. */
                gf64_t val = 0;
                gf64_t yr_pow = 1;
                const gf64_t *Lc = L_basis + c * N;
                for (size_t i = 0; i < N; i++) {
                    val ^= gf64_mul_reference(Lc[i], yr_pow);
                    yr_pow = gf64_mul_reference(yr_pow, yr);
                }
                /* acc += in[c][w] * V'(x_c) * val. */
                gf64_t weighted = gf64_mul_reference(in[c * B + w], V_prime[c]);
                acc ^= gf64_mul_reference(weighted, val);
            }
            out[r * B + w] = gf64_mul_reference(acc, gf64_inverse(v_at[r]));
        }
    }
    free(L_basis);
    free(V_prime);
    free(v_at);
}

/* ----- xorshift64 PRNG for reproducible random inputs. */
static uint64_t xorshift_state = 0xDEADBEEFCAFEBABEULL;
static uint64_t xorshift64(void) {
    uint64_t x = xorshift_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    xorshift_state = x;
    return x;
}

static int test_config(size_t N, size_t R, size_t B, uint64_t firstInput, uint64_t firstRecovery)
{
    gf64_t *in  = (gf64_t *)malloc(N * B * sizeof(gf64_t));
    gf64_t *cauchy_out = (gf64_t *)calloc(R * B, sizeof(gf64_t));
    gf64_t *fenger_out = (gf64_t *)calloc(R * B, sizeof(gf64_t));
    if (!in || !cauchy_out || !fenger_out) {
        fprintf(stderr, "  alloc fail N=%zu R=%zu B=%zu\n", N, R, B);
        return 1;
    }
    /* Random inputs. */
    for (size_t i = 0; i < N * B; i++) in[i] = xorshift64();

    cauchy_reference(in, N, B, cauchy_out, R, firstInput, firstRecovery);
    fenger_naive (in, N, B, fenger_out,  R, firstInput, firstRecovery);

    /* Bit-exact compare. */
    int ok = 1;
    for (size_t i = 0; i < R * B; i++) {
        if (cauchy_out[i] != fenger_out[i]) {
            fprintf(stderr, "  MISMATCH at N=%zu R=%zu B=%zu idx=%zu: cauchy=0x%016llx fenger=0x%016llx\n",
                    N, R, B, i, (unsigned long long)cauchy_out[i], (unsigned long long)fenger_out[i]);
            ok = 0;
            break;
        }
    }
    if (ok) printf("  PASS: N=%zu R=%zu B=%zu firstInput=0x%llx firstRecovery=0x%llx\n",
                   N, R, B, (unsigned long long)firstInput, (unsigned long long)firstRecovery);
    free(in); free(cauchy_out); free(fenger_out);
    return ok ? 0 : 1;
}

int main(void)
{
    printf("Fenger Toeplitz kernel — Phase 0 sanity (issue #28)\n");
    printf("Verifying Fenger decomposition (naive) bit-exactly equals\n");
    printf("explicit Cauchy matrix-vector product over GF(2^64)[x].\n");
    printf("\n");

    int failures = 0;
    /* Small N, R, B at various firstInput/firstRecovery offsets. */
    failures += test_config(2, 2, 4, 0, 100);
    failures += test_config(4, 2, 8, 0, 100);
    failures += test_config(4, 4, 8, 5, 200);
    failures += test_config(8, 4, 16, 100, 1000);
    /* N=8, R=8 with disjoint (firstInput=1) vs (firstRecovery=20) so no y_r
     * coincides with an x_c. The earlier firstInput=firstRecovery=1 case
     * is degenerate (V(y_r)=0 makes gf64_inverse undefined). */
    failures += test_config(8, 8, 16, 1, 20);
    /* Edge: degenerate. */
    failures += test_config(1, 1, 1, 0, 0);
    /* Larger (still O(N^3 * B) for fenger_naive but fast enough at this size). */
    failures += test_config(16, 4, 32, 0x100, 0x200);

    printf("\n%s: %d test(s) failed\n", failures == 0 ? "OK" : "FAIL", failures);
    return failures;
}
