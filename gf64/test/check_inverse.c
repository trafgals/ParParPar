/* check_inverse.c — verify the Gauss-Jordan inversion in
 * gf64_additive_fft_hqc2026.c by checking M · M_inv = I at n=8. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t gf64_t;
static const gf64_t GF64_CANTOR_BASIS[20] = {
    0x0000000000000001ULL, 0x19c9369f278adc02ULL, 0xa181e7d66f5ff794ULL,
    0x5db84357ce785d08ULL, 0xb973d466f5c9d0caULL, 0x521ac889831a075eULL,
    0x033ce8beddc8a656ULL, 0xb5846c4e07b91010ULL, 0x4087b8cbb37a32ecULL,
    0x00d0d3888c0ae17cULL, 0xafd5ac70237f2222ULL, 0xe3f5af99cc3aaaf8ULL,
    0x5a1db3b16a0b58b8ULL, 0x09947c54fe7ee248ULL, 0x0e8eaf0e0068f544ULL,
    0xa2a113500b4b4f5aULL, 0xe96f9805d6ce0bb0ULL, 0x53496f8b5c9edd4cULL,
    0xad325cb6f4ac2a9eULL, 0x4a8dcf8bd7ede826ULL,
};

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);
extern gf64_t gf64_inverse(gf64_t a);

static void compute_sj(int j, int n, gf64_t *s_out) {
    gf64_t cur[4096], sq_part[4096];
    for (int i = 0; i < n; i++) cur[i] = 0;
    cur[1] = 1;
    for (int iter = 1; iter <= j; iter++) {
        for (int i = 0; i < n; i++) sq_part[i] = 0;
        for (int i = 0; i < n; i++) {
            if (cur[i] == 0) continue;
            int two_i = 2 * i;
            if (two_i < n) sq_part[two_i] ^= 1;
        }
        for (int i = 0; i < n; i++) cur[i] = sq_part[i] ^ cur[i];
    }
    for (int i = 0; i < n; i++) s_out[i] = cur[i];
}
static void poly_mul_trunc(int n, gf64_t *a, gf64_t *b, gf64_t *out) {
    for (int i = 0; i < n; i++) out[i] = 0;
    for (int ja = 0; ja < n; ja++) {
        if (a[ja] == 0) continue;
        for (int jb = 0; jb < n - ja; jb++) {
            if (b[jb] == 0) continue;
            out[ja + jb] ^= gf64_mul_reference(a[ja], b[jb]);
        }
    }
}
static gf64_t compute_v_j(int j) {
    gf64_t v = 0;
    for (int k = 0; k < 20; k++) if ((j >> k) & 1) v ^= GF64_CANTOR_BASIS[k];
    return v;
}

int main(void) {
    int n = 8;
    gf64_t *X = calloc(n * n, sizeof(gf64_t));
    gf64_t cur[4096], s_poly[4096], new_cur[4096];
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) cur[i] = 0;
        cur[0] = 1;
        for (int bit_pos = 0; bit_pos < 20; bit_pos++) {
            if (!((k >> bit_pos) & 1)) continue;
            compute_sj(bit_pos, n, s_poly);
            poly_mul_trunc(n, cur, s_poly, new_cur);
            memcpy(cur, new_cur, n * sizeof(gf64_t));
        }
        for (int j = 0; j < n; j++) X[k * n + j] = cur[j];
    }
    /* Print X basis. */
    printf("X (col k = monomial coeffs of X_k), n=8:\n");
    for (int j = 0; j < n; j++) {
        printf("  degree x^%d:", j);
        for (int k = 0; k < n; k++) printf(" %016llx", (unsigned long long)X[k * n + j]);
        printf("\n");
    }
    /* Print V (cantors at this depth). */
    printf("v_table:\n");
    for (int j = 0; j < n; j++)
        printf("  [%d] = 0x%016llx\n", j, (unsigned long long)compute_v_j(j));

    /* Build M (col k = row j = X_k[j]). */
    gf64_t *M = calloc(n * n, sizeof(gf64_t));
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++) M[j * n + k] = X[k * n + j];

    /* Gauss-Jordan to compute M_inv = M^{-1} on [M | I]. */
    size_t aug_size = 2 * n;
    gf64_t *aug = calloc(n * aug_size, sizeof(gf64_t));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i * aug_size + j] = M[i * n + j];
        for (int j = 0; j < n; j++) aug[i * aug_size + (n + j)] = (i == j) ? 1 : 0;
    }
    for (int col = 0; col < n; col++) {
        int pivot = -1;
        for (int r = col; r < n; r++)
            if (aug[r * aug_size + col] != 0) { pivot = r; break; }
        if (pivot < 0) { printf("SINGULAR at col %d\n", col); return 1; }
        if (pivot != col) {
            for (size_t j = 0; j < aug_size; j++) {
                gf64_t t = aug[col * aug_size + j];
                aug[col * aug_size + j] = aug[pivot * aug_size + j];
                aug[pivot * aug_size + j] = t;
            }
        }
        gf64_t pv_inv = gf64_inverse(aug[col * aug_size + col]);
        for (size_t j = 0; j < aug_size; j++) aug[col * aug_size + j] = gf64_mul_reference(aug[col * aug_size + j], pv_inv);
        for (int r = 0; r < n; r++) {
            if (r == col) continue;
            gf64_t factor = aug[r * aug_size + col];
            if (factor == 0) continue;
            for (size_t j = 0; j < aug_size; j++)
                aug[r * aug_size + j] ^= gf64_mul_reference(factor, aug[col * aug_size + j]);
        }
    }
    gf64_t *M_inv = calloc(n * n, sizeof(gf64_t));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) M_inv[i * n + j] = aug[i * aug_size + (n + j)];

    /* Check M * M_inv = I. */
    int ok = 1;
    printf("M * M_inv:\n");
    for (int i = 0; i < n; i++) {
        printf("  row %d:", i);
        for (int j = 0; j < n; j++) {
            gf64_t acc = 0;
            for (int k = 0; k < n; k++)
                acc ^= gf64_mul_reference(M[i * n + k], M_inv[k * n + j]);
            printf(" %016llx%s", (unsigned long long)acc,
                   acc == (i == j ? 1 : 0) ? "" : " *");
            if (acc != (i == j ? 1 : 0)) ok = 0;
        }
        printf("\n");
    }
    printf("%s\n", ok ? "M*M_inv = I (Gauss-Jordan correct)" : "M*M_inv != I — INVERSION BUG");
    return ok ? 0 : 1;
}
