/*
 * gf64/gf64_additive_fft_vandermonde.c — additive FFT over GF(2^64) using
 *   an explicit Vandermonde matrix via the Cantor basis.
 *
 * The forward transform is the EVALUATION matrix F[i][j] = v_i^j where
 * v_i are the W_m elements indexed by i (sum of GF64_CANTOR_BASIS[k] over
 * bits set in i, with v_0 = 0). This is the "obvious" correct additive
 * DFT, not the canonical O(N log N) recursive form — which we still do
 * not have due to ongoing Phase 2b debugging.
 *
 * Why this is still useful:
 *   - It satisfies the convolution theorem (pointwise mul in this basis =
 *     poly convolution in monomial basis).
 *   - It's an O(N²) mat-vec, but the matrix is pre-computed once per
 *     call site (depends on N, not on the input polynomial).
 *   - As a drop-in replacement for gf64_fft_forward / gf64_fft_inverse,
 *     it unblocks downstream consumers (Phase 3 Fenger Toeplitz).
 *
 * What it does NOT do:
 *   - It's O(N²) per call, not O(N log N) like a real additive FFT.
 *   - The Fenger Toeplitz pipeline consuming this FFT gets only the
 *     constant-factor win of the structural decomposition, not the
 *     asymptotic FFT win. The combined cost matches the current
 *     Cauchy 2D-muladd throughput within a small factor.
 *
 * Build & run:
 *   gcc -O2 -I.. test_gf64_fft_poly_mul.c \
 *       ../gf64_additive_fft.c ../gf64_additive_fft_vandermonde.c \
 *       ../gf64_poly_mul_karatsuba.c ../gf64_invert_ita.c \
 *       ../gf64_mpe.c ../gf64_subproduct.c ../gf64_mul_avx512.c \
 *       ../gf64_square.c ../gf64_single.c -o test_gf64_fft_poly_mul \
 *       && ./test_gf64_fft_poly_mul
 */

#include "gf64_additive_fft.h"
#include "gf64_mul.h"
#include "gf64_cantor_basis.h"
#include "gf64_invert_ita.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

HEDLEY_BEGIN_C_DECLS

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/*
 * Compute v_j = the j-th W_m element (sum of GF64_CANTOR_BASIS[k] for bits
 * set in j). v_0 is the zero element of W_m.
 */
static gf64_t compute_v_j(int j) {
    gf64_t v = 0;
    for (int k = 0; k < 20; k++) {
        if ((j >> k) & 1) v ^= GF64_CANTOR_BASIS[k];
    }
    return v;
}

/*
 * Build the n x n Vandermonde matrix V[i][j] = v_i^j (monomial basis).
 *
 * The matrix is stored row-major in `out` (out[i * n + j]). Since v_0 = 0,
 * the first column is [1, 0, 0, ..., 0] regardless of basis.
 *
 * The matrix entries are GF(2^64) values, not bits: v_i^j for j > 0 may
 * be non-trivial (since v_i ∈ GF(2^64), not just {0, 1}).
 */
static void build_vandermonde(int n, gf64_t *out) {
    /* Compute v_i for each i, then v_i^j for j = 0..n-1. */
    gf64_t *v_arr = (gf64_t *)malloc(n * sizeof(gf64_t));
    if (v_arr == NULL) abort();
    for (int i = 0; i < n; i++) v_arr[i] = compute_v_j(i);

    for (int i = 0; i < n; i++) {
        gf64_t v = v_arr[i];
        out[i * n + 0] = 1;          /* v^0 = 1 for all v (including v_0=0). */
        gf64_t vj = 1;
        for (int j = 1; j < n; j++) {
            vj = gf64_mul_reference(vj, v);
            out[i * n + j] = vj;
        }
    }
    free(v_arr);
}

/*
 * Invert the n x n Vandermonde matrix via Gaussian elimination over GF(2^64).
 *
 * Output: out[i * n + j] holds the (i,j) entry of V^{-1}.
 *
 * Each entry is a GF(2^64) value. The matrix has full rank (since the
 * evaluation points v_0, ..., v_{n-1} are distinct), so Gaussian elimination
 * succeeds.
 */
static void invert_vandermonde(int n, gf64_t *out) {
    /* Build V in column-major working buffer for easier pivoting. */
    gf64_t *M = (gf64_t *)malloc(n * n * sizeof(gf64_t));
    if (M == NULL) abort();
    build_vandermonde(n, M);

    /* Augment with identity. */
    gf64_t *aug = (gf64_t *)malloc(n * 2 * n * sizeof(gf64_t));
    if (aug == NULL) { free(M); abort(); }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * 2 * n + j] = M[i * n + j];
            aug[i * 2 * n + n + j] = (i == j) ? 1 : 0;
        }
    }

    /* Forward elimination with partial pivoting. */
    for (int col = 0; col < n; col++) {
        /* Find row with non-zero pivot in this column at or below `col`. */
        int piv = -1;
        for (int r = col; r < n; r++) {
            if (aug[r * 2 * n + col] != 0) { piv = r; break; }
        }
        if (piv < 0) abort();  /* Singular: should never happen. */
        if (piv != col) {
            for (int j = 0; j < 2 * n; j++) {
                gf64_t tmp = aug[col * 2 * n + j];
                aug[col * 2 * n + j] = aug[piv * 2 * n + j];
                aug[piv * 2 * n + j] = tmp;
            }
        }
        /* Scale pivot row so the leading entry is 1. */
        gf64_t inv = gf64_invert_ita_one(aug[col * 2 * n + col]);
        for (int j = 0; j < 2 * n; j++) {
            aug[col * 2 * n + j] = gf64_mul_reference(inv, aug[col * 2 * n + j]);
        }
        /* Eliminate below and above. */
        for (int r = 0; r < n; r++) {
            if (r == col) continue;
            gf64_t factor = aug[r * 2 * n + col];
            if (factor == 0) continue;
            for (int j = 0; j < 2 * n; j++) {
                aug[r * 2 * n + j] ^= gf64_mul_reference(factor, aug[col * 2 * n + j]);
            }
        }
    }

    /* Copy inverse columns from augmented. */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            out[i * n + j] = aug[i * 2 * n + n + j];
        }
    }

    free(aug);
    free(M);
}

/* Caches for V and V^{-1} per n. The matrices depend only on n, not on the
 * input data. They're built lazily on first use. */
typedef struct VCacheEntry {
    int n;
    gf64_t *V;
    gf64_t *Vi;
    int refcount;
    struct VCacheEntry *next;
} VCacheEntry;

static VCacheEntry *s_vcache = NULL;

static void get_or_build_matrices(int n, gf64_t **V_out, gf64_t **Vi_out) {
    for (VCacheEntry *e = s_vcache; e != NULL; e = e->next) {
        if (e->n == n) {
            e->refcount++;
            *V_out = e->V;
            *Vi_out = e->Vi;
            return;
        }
    }
    VCacheEntry *e = (VCacheEntry *)malloc(sizeof(VCacheEntry));
    if (e == NULL) abort();
    e->n = n;
    e->V = (gf64_t *)malloc(n * n * sizeof(gf64_t));
    e->Vi = (gf64_t *)malloc(n * n * sizeof(gf64_t));
    if (e->V == NULL || e->Vi == NULL) abort();
    build_vandermonde(n, e->V);
    invert_vandermonde(n, e->Vi);
    e->refcount = 1;
    e->next = s_vcache;
    s_vcache = e;
    *V_out = e->V;
    *Vi_out = e->Vi;
}

static void release_matrices(int n) {
    for (VCacheEntry *e = s_vcache; e != NULL; e = e->next) {
        if (e->n == n) {
            e->refcount--;
            return;
        }
    }
}

/*
 * Forward Vandermonde FFT: out = V * in.
 *
 *   Pre:  in holds n coefficients of the polynomial
 *              f(x) = sum_j in[j] * x^j
 *   Post: out[i] = f(v_i) = sum_j in[j] * v_i^j
 */
void gf64_fft_forward_vandermonde(gf64_t *out, const gf64_t *in, size_t n) {
    assert(n >= 1 && (n & (n - 1)) == 0);  /* power of 2 */
    if (n == 1) { out[0] = in[0]; return; }

    gf64_t *V = NULL, *Vi = NULL;
    get_or_build_matrices((int)n, &V, &Vi);

    for (size_t i = 0; i < n; i++) {
        gf64_t s = 0;
        for (size_t j = 0; j < n; j++) {
            s ^= gf64_mul_reference(V[i * n + j], in[j]);
        }
        out[i] = s;
    }
    release_matrices((int)n);
}

/*
 * Inverse Vandermonde FFT: out = V^{-1} * in.
 *
 *   Pre:  in holds evaluations in[i] = f(v_i) for i = 0..n-1.
 *   Post: out holds the n monomial coefficients of f.
 */
void gf64_fft_inverse_vandermonde(gf64_t *out, const gf64_t *in, size_t n) {
    assert(n >= 1 && (n & (n - 1)) == 0);
    if (n == 1) { out[0] = in[0]; return; }

    gf64_t *V = NULL, *Vi = NULL;
    get_or_build_matrices((int)n, &V, &Vi);

    for (size_t i = 0; i < n; i++) {
        gf64_t s = 0;
        for (size_t j = 0; j < n; j++) {
            s ^= gf64_mul_reference(Vi[i * n + j], in[j]);
        }
        out[i] = s;
    }
    release_matrices((int)n);
}

/*
 * Vandermonde-FFT-based poly_mul. Forward + pointwise + inverse.
 *
 * Computes out = a * b in GF(2^64)[x]. a has deg_a + 1 coefficients,
 * b has deg_b + 1 coefficients; out must hold deg_a + deg_b + 1
 * coefficients.
 */
static size_t next_pow2_size_t(size_t x) {
    size_t r = 1;
    while (r < x) r <<= 1;
    return r;
}

void gf64_poly_mul_vandermonde(
    gf64_t *out,
    const gf64_t *a, size_t deg_a,
    const gf64_t *b, size_t deg_b
) {
    const size_t out_len = deg_a + deg_b + 1;
    if (out_len <= 1) {
        /* Deg_a = 0 and deg_b = 0: a * b is a single element. */
        if (deg_a == 0 && deg_b == 0) {
            out[0] = gf64_mul_reference(a[0], b[0]);
        } else {
            out[0] = 0;
        }
        return;
    }

    const size_t n = next_pow2_size_t(out_len);
    if (n < 2) {
        /* Deg_a = 0, deg_b = 0 (handled above); or out_len = 1 (deg = 0). */
        if (deg_a == 0 && deg_b == 0) {
            out[0] = gf64_mul_reference(a[0], b[0]);
        } else {
            out[0] = 0;
        }
        return;
    }

    gf64_t *A = (gf64_t *)calloc(n, sizeof(gf64_t));
    gf64_t *B = (gf64_t *)calloc(n, sizeof(gf64_t));
    gf64_t *FA = (gf64_t *)calloc(n, sizeof(gf64_t));
    gf64_t *FB = (gf64_t *)calloc(n, sizeof(gf64_t));
    if (A == NULL || B == NULL || FA == NULL || FB == NULL) abort();

    for (size_t i = 0; i <= deg_a; i++) A[i] = a[i];
    for (size_t i = 0; i <= deg_b; i++) B[i] = b[i];

    gf64_fft_forward_vandermonde(FA, A, n);
    gf64_fft_forward_vandermonde(FB, B, n);
    for (size_t i = 0; i < n; i++) {
        FA[i] = gf64_mul_reference(FA[i], FB[i]);
    }

    gf64_t *src = (gf64_t *)malloc(n * sizeof(gf64_t));
    if (src == NULL) abort();
    memcpy(src, FA, n * sizeof(gf64_t));
    /* Write inverse output into a stack n-slot buffer, then copy
     * out_len slots to the caller's out. Avoids any aliasing or
     * out-of-bounds writes past `out_len`. */
    gf64_t *inv_out = (gf64_t *)malloc(n * sizeof(gf64_t));
    if (inv_out == NULL) abort();
    gf64_fft_inverse_vandermonde(inv_out, src, n);
    memcpy(out, inv_out, out_len * sizeof(gf64_t));
    free(inv_out);
    free(src);
    free(A); free(B); free(FA); free(FB);
}

HEDLEY_END_C_DECLS
