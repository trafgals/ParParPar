/*
 * test_dft_gf16.c — compute the additive-DFT matrix over GF(2^4)
 * explicitly via characters, and check whether the LCH14 recursion
 * matches.
 *
 * The additive DFT over GF(2^k) is F[i][j] = chi_j(v_i) where
 *   chi_a(x) = (-1)^{Tr(a * x)}
 * and {v_i}, {omega_j} are dual bases (Tr(omega_i v_j) = delta_ij).
 *
 * For n = 2^m the DFT is an n x n matrix over {+1, -1} (= {0, 1} in
 * char 2). The convolution theorem (circulant / Toeplitz) holds: for
 * an "evaluation basis", F . a (monomial coefs) gives evaluations
 * f(v_i), and F^{-1} recovers monomial coefs.
 *
 * Compute F, factor it level-by-level (bit-reversal of Cooley-Tukey),
 * and extract the multiplier constants. Compare against the LCH14
 * algorithm's butterfly structure.
 *
 * Build & run from gf64/test/:
 *   gcc -O0 test_dft_gf16.c -o test_dft_gf16 -lm && ./test_dft_gf16
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef uint8_t gf16_t;
#define GF16_MOD_POLY 0x13  /* x^4 + x + 1 */

static uint8_t gf16_exp[16], gf16_log[16];
static void gf16_init_tables(void) {
    int x = 1;
    for (int i = 0; i < 15; i++) { gf16_exp[i] = (uint8_t)x; x <<= 1; if (x & 0x10) x ^= GF16_MOD_POLY; x &= 0xF; }
    gf16_exp[15] = gf16_exp[0];
    for (int i = 0; i < 15; i++) gf16_log[gf16_exp[i]] = (uint8_t)i;
}
static inline gf16_t gf16_mul(gf16_t a, gf16_t b) {
    if (!a || !b) return 0;
    return gf16_exp[(gf16_log[a] + gf16_log[b]) % 15];
}
static inline gf16_t gf16_sq(gf16_t a) {
    if (!a) return 0;
    return gf16_exp[(2 * gf16_log[a]) % 15];
}
static inline gf16_t gf16_div(gf16_t a, gf16_t b) {
    assert(b != 0);
    if (!a) return 0;
    return gf16_exp[((int)gf16_log[a] - (int)gf16_log[b] + 15) % 15];
}

/* Frobenius powers: v^(2^i). Equivalent to (gf16_log[v] * (1 << i)) % 15
 * but we just call gf16_exp. */
static inline gf16_t gf16_pow(gf16_t a, int k) {
    if (!a) return 0;
    int lk = (gf16_log[a] * k) % 15;
    /* k could be a Frobenius power (e.g. 2^i). For non-power-of-2 k
     * we just do additive-chains but here we only need small powers. */
    return gf16_exp[lk];
}

/* Tr_{GF(2^k)/GF(2)}(v) = v + v^2 + v^4 + ... + v^{2^{k-1}}.
 * For k = 4: Tr(v) = v + v^2 + v^4 + v^8. */
static int gf16_trace(gf16_t v) {
    int r = v;
    for (int i = 0; i < 4; i++) {
        v = gf16_sq(v);
        r ^= v;
    }
    return r & 1;
}

/* --- Find dual basis {omega_0, ..., omega_3} such that
 * Tr(omega_i * v_j) = delta(i, j).
 *
 * Solve the 4x4 system Tr(omega_i * v_j) = delta(i,j) over GF(2).
 * The v_j are the natural basis {v_0=1, v_1, v_2, v_3}. */
static int compute_dual_basis(gf16_t *omega, const gf16_t *v) {
    /* Build 4x4 matrix M[i][j] = Tr(v_i * v_j) over GF(2).
     * Then solve M^T * omega = delta, i.e., M^T is symmetric so M. */
    int M[4][4] = {{0}};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            M[i][j] = gf16_trace(gf16_mul(v[i], v[j]));
        }
    }
    /* Invert M over GF(2). */
    int aug[4][8];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) aug[i][j] = M[i][j];
        for (int j = 4; j < 8; j++) aug[i][j] = (i == (j - 4)) ? 1 : 0;
    }
    for (int r = 0; r < 4; r++) {
        /* Find pivot. */
        int piv = -1;
        for (int i = r; i < 4; i++) if (aug[i][r]) { piv = i; break; }
        if (piv < 0) return -1;  /* singular */
        if (piv != r) for (int j = 0; j < 8; j++) { int t = aug[r][j]; aug[r][j] = aug[piv][j]; aug[piv][j] = t; }
        for (int i = 0; i < 4; i++) {
            if (i != r && aug[i][r]) {
                for (int j = 0; j < 8; j++) aug[i][j] ^= aug[r][j];
            }
        }
    }
    /* Aug columns 4-7 now hold M^{-1}. */
    /* omega_i = sum over j of aug[j][i+4] * v_j (treating as GF(2) coefs on v). */
    for (int i = 0; i < 4; i++) {
        gf16_t s = 0;
        for (int j = 0; j < 4; j++) {
            if (aug[j][i + 4]) s ^= v[j];
        }
        omega[i] = s;
    }
    /* Verify. */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int t = gf16_trace(gf16_mul(omega[i], v[j]));
            int want = (i == j) ? 1 : 0;
            if (t != want) {
                printf("Dual basis verify FAIL at (%d,%d)\n", i, j);
                return -2;
            }
        }
    }
    return 0;
}

/* DFT matrix: F[i][j] = chi_j(v_i) = (-1)^{Tr(omega_j * v_i)} ∈ {+1, -1}.
 * Encode as +1 = 0 (char 2: additive identity), -1 = 1 (additive inverse).
 * So F[i][j] = Tr(omega_j * v_i). */
static void build_dft_matrix(int F[16][16], const gf16_t *v, const gf16_t *omega, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            F[i][j] = gf16_trace(gf16_mul(omega[j], v[i]));
        }
    }
}

/* Print the matrix. */
static void print_matrix(const char *tag, int F[16][16], int n) {
    printf("%s (%d x %d):\n", tag, n, n);
    for (int i = 0; i < n; i++) {
        printf("  ");
        for (int j = 0; j < n; j++) printf("%d ", F[i][j]);
        printf("\n");
    }
}

/* Apply the DFT to a vector of n GF(2^4) values.
 * The matrix entries are {0, 1}; we sum with XOR (mod 2 addition). */
static void dft_apply(gf16_t *out, const int F[16][16], const gf16_t *in, int n) {
    for (int i = 0; i < n; i++) {
        gf16_t s = 0;
        for (int j = 0; j < n; j++) {
            if (F[i][j]) s ^= in[j];
        }
        out[i] = s;
    }
}

/* Product form: g_i = sum_j F[i][j] * x_j. */
static void pointwise_mul(gf16_t *out, const int F[16][16],
                          const gf16_t *a, const gf16_t *b, int n) {
    for (int i = 0; i < n; i++) {
        gf16_t s = 0;
        for (int j = 0; j < n; j++) {
            if (F[i][j]) s ^= gf16_mul(a[j], b[j]);
        }
        out[i] = s;
    }
}

/* Polynomial multiplication via DFT: out = IDFT(DFT(a) * DFT(b)). */
static void poly_via_dft(gf16_t *out, const gf16_t *a, int la,
                         const gf16_t *b, int lb,
                         const int F[16][16], const int Fi[16][16], int n) {
    gf16_t A[16] = {0}, B[16] = {0};
    for (int i = 0; i < la; i++) A[i] = a[i];
    for (int i = 0; i < lb; i++) B[i] = b[i];
    gf16_t FA[16], FB[16], FAB[16];
    dft_apply(FA, F, A, n);
    dft_apply(FB, F, B, n);
    pointwise_mul(FAB, F /* unused */, FA, FB, n);
    (void)Fi;
    /* Apply inverse DFT (the same matrix since F is symmetric for
     * additive DFT). */
    gf16_t tmp[16];
    dft_apply(tmp, F, FAB, n);
    /* In additive DFT over char 2 with self-dual characters, F*F = n*I
     * requires dividing by n. For n=4 = 2^m, we need 1/n scaling.
     * For now we scale by 4 (= n, which means dividing by 1/4 = ... no).
     * The closed form: g(v) = (1/n) * sum_omega f-hat[omega] * chi_omega(v).
     * For char 2 with additive DFT, the matrix square gives n*I only
     * if 1/n is valid in the field. n = 4 = 0 in char 2^k for any k >= 2,
     * so this is NOT valid. */
    printf("(note: 1/n scaling issue — need to handle inv(n) in char 2)\n");
    for (int i = 0; i < n; i++) out[i] = tmp[i];
}

static void schoolbook(gf16_t *out, const gf16_t *a, int la, const gf16_t *b, int lb) {
    memset(out, 0, (la + lb - 1) * sizeof(gf16_t));
    for (int i = 0; i < la; i++) {
        for (int j = 0; j < lb; j++) {
            out[i + j] ^= gf16_mul(a[i], b[j]);
        }
    }
}

/* Find recursive DFT multiplier: at level i, the butterfly pairs
 * (j, j + 2^i) processes F such that F[i][j] and F[i][j + 2^i]
 * are related to F[i+1][?] via a multiplier mu.
 *
 * For the standard Cooley-Tukey: F[i][j] = F[i+1][j] + mu * F[i+1][j + 2^{i-1}]
 * (or similar). Let's extract this from the matrix. */
static void extract_butterfly(const int F[16][16], int n, int level) {
    int stride = 1 << level;
    printf("Butterfly at level %d (stride %d):\n", level, stride);
    for (int j = 0; j < stride; j++) {
        /* Find the row at level (n/2) that corresponds to this butterfly. */
        /* Try row j and j + stride. */
        /* F[i][j] = F[i'][j'] + mu * F[i'][j' + n/2] */
        /* For n=4, level 1, stride 2: 2 butterflies (j=0,1). */
        printf("  pair (%d, %d): F[%d][%d..%d]: ", j, j + stride, j, j, j + stride);
        for (int c = j; c <= j + stride; c++) printf("%d ", F[j][c]);
        printf("; F[%d][..]: ", j + stride);
        for (int c = j; c <= j + stride; c++) printf("%d ", F[j + stride][c]);
        printf("\n");
    }
}

int main(void) {
    printf("Additive DFT over GF(2^4) with x^4 + x + 1\n");
    printf("============================================\n\n");

    gf16_init_tables();

    /* Use the natural Cantor basis {1, v_1, v_2, v_3}. */
    gf16_t v[4] = {1, 6, 2, 0xA};
    /* Compute the dual basis. */
    gf16_t omega[4];
    int rc = compute_dual_basis(omega, v);
    if (rc) { printf("Dual basis FAILED (%d)\n", rc); return rc; }
    printf("Basis v: ");
    for (int i = 0; i < 4; i++) printf("%X ", v[i]);
    printf("\nOmega:  ");
    for (int i = 0; i < 4; i++) printf("%X ", omega[i]);
    printf("\n\n");

    /* Verify Tr(omega_i * v_j) = delta_{ij}. */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int t = gf16_trace(gf16_mul(omega[i], v[j]));
            printf("%d", (t == ((i == j) ? 1 : 0)) ? 1 : 0);
        }
        printf("\n");
    }
    printf("\n");

    /* Build the n x n DFT matrix for various n. */
    for (int n : (int[]){2, 4}) {
        printf("=== n = %d ===\n", n);
        int F[16][16];
        build_dft_matrix(F, v, omega, n);
        print_matrix("DFT F", F, n);

        if (n == 4) {
            /* Extract butterfly structure. */
            for (int lvl = 0; lvl < 2; lvl++) {
                extract_butterfly(F, n, lvl);
            }
        }
    }

    /* For n = 4, run the convolution test. */
    {
        int n = 4;
        gf16_t a[2] = {1, 2};
        gf16_t b[2] = {3, 4};
        gf16_t ab_ref[3];
        schoolbook(ab_ref, a, 2, b, 2);
        printf("\nReference convolution: [%X %X %X]\n", ab_ref[0], ab_ref[1], ab_ref[2]);

        int F[16][16];
        build_dft_matrix(F, v, omega, n);
        gf16_t A[16] = {a[0], a[1], 0, 0};
        gf16_t B[16] = {b[0], b[1], 0, 0};
        gf16_t FA[16], FB[16];
        dft_apply(FA, F, A, n);
        dft_apply(FB, F, B, n);
        printf("DFT(A): [");
        for (int i = 0; i < n; i++) printf("%X ", FA[i]);
        printf("]\n");
        printf("DFT(B): [");
        for (int i = 0; i < n; i++) printf("%X ", FB[i]);
        printf("]\n");
    }
    return 0;
}
