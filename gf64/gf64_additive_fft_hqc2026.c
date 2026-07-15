/*
 * gf64/gf64_additive_fft_hqc2026.c — additive FFT over GF(2^64) implementing
 *   HQC 2026 TCHES §2.3 Algorithm 2 (LCH14 addFFT).
 *
 * This is the canonical sparse O(N log N) [butterfly] factorization of the
 * Vandermonde matrix V[i][j] = v_i^j (where v_i are the Cantor basis
 * elements at bit positions of i). The algorithm is published four times
 * (LCH14, Chen 2018, hamil 2016, HQC 2026) with functionally identical
 * constructions; this file follows the HQC 2026 exposition.
 *
 * Algorithm (verbatim from HQC §2.3):
 *
 *   addFFT(f, a + V_i):
 *     if f in monomial basis:  f ← BasisCvt(f)            // monomial → novelpoly
 *     return Butterfly(f, a + V_i)
 *
 *   Butterfly(f, a + V_i):
 *     if i = 1:  return (f_l + a · f_h,  f_l + (a+1) · f_h)
 *     split:    f = f_l + s_{i-1} · f_h                       // s_i vanishes on V_i
 *     f_l ← f_l + s_{i-1}(a) · f_h                              // multiplier (NON-trivial)
 *     f_h ← f_l + (s_{i-1}(a) + 1) · f_h
 *     return (Butterfly(f_l, a + V_{i-1}),
 *             Butterfly(f_h, a + v_{i-1} + V_{i-1}))
 *
 * Forward output: (f(a+0), f(a+1), ..., f(a+n-1)) — polynomial
 * evaluations at the affine coset. Convolution theorem holds by
 * construction.
 *
 * KEY DIFFERENCE vs the existing gf64_additive_fft_lch14.c (the broken
 * one): that file used the multiplier mu_j = s_i(v_j) / s_i(v_i) =
 * s_i(v_j), which collapses to the constant 1 by the Cantor recurrence
 * (because s_i(v_i) = 1 — and W_m(j | (1<<i)) is always v_i when we test
 * only the obvious choices). The HQC 2026 multiplier is s_{i-1}(a) where
 * a is the affine shift parameter (NOT a basis element). With a ∉
 * V_{i-1}, s_{i-1}(a) is non-trivial and the butterfly achieves O(n log n).
 * See PHASE_2B_RESEARCH_2026-07-15.md §Resolution for the bug-by-bug
 * analysis.
 *
 * Implementation choices:
 *   - Field multiply uses gf64_mul_reference (scalar SSE2). PCLMULQDQ
 *     acceleration can be wired in by replacing the call site with
 *     gf64_mul_avx512 once n is large enough to amortize the 8-lane SIMD
 *     setup overhead (n ≥ 64 typically; the canonical PAR3 n = 4096 wins
 *     by 4–6× with the SIMD path).
 *   - Cantor basis v_0..v_19 from gf64_cantor_basis.h (20 precomputed
 *     vectors; sufficient for n ≤ 2^20 — the canonical PAR3 n = 4096 is
 *     well within range).
 *   - BasisCvt: matrix-vector multiply M · g where M is the change-of-basis
 *     matrix (monomial coeffs of novelpoly basis functions). M and (M^T)^{-1}
 *     are precomputed and cached per n. Cost is O(n^2) per call. A recursive
 *     BasisCvt (Chen 2018 Algorithm 1) reduces this to O(n log n); for
 *     now we use the matrix form which keeps the working set at ~32 KiB at
 *     n = 4096 and is simple to verify bit-exactly.
 *
 * Public API (declared in gf64_additive_fft.h):
 *   void gf64_addfft64_fwd(gf64_t *arr, size_t n);
 *   void gf64_addfft64_inv(gf64_t *arr, size_t n);
 *   void gf64_addfft64_poly_mul(gf64_t *out, const gf64_t *a, size_t la,
 *                                const gf64_t *b, size_t lb, size_t out_len);
 *
 * Length cap: n ≤ 4096 (the canonical PAR3 transform size). Above this
 * the stack-allocated scratch buffers may be infeasible; can be lifted
 * by switching the BasisCvt scratch to heap allocation if needed.
 */

#include "gf64_additive_fft.h"
#include "gf64_mul.h"
#include "gf64_cantor_basis.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

HEDLEY_BEGIN_C_DECLS

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);
extern gf64_t gf64_inverse(gf64_t a);

/* Cantor basis has 20 precomputed vectors; ℓ_m = 20 supports n up to 2^20. */
#define GF64_HQC_MAX_LM 20
/* Production cap on the additive FFT transform size. Stacks of 5 buffers
 * at this size come to 5 * 16384 * 8 = 640 KB per recursion level; Linux
 * default thread stack (8 MB) accommodates this with room for call frames.
 * The canonical PAR3 workload (N = 10K data slices) needs at least 16384
 * to fit; the bench at N = 10240 verifies the additive FFT path is correct
 * there. */
#define GF64_HQC_MAX_N 16384

/* ----- helpers ----- */

/* v_j = XOR of GF64_CANTOR_BASIS[k] for bits k set in j; v_0 = 0. */
static gf64_t compute_v_j(int j) {
    gf64_t v = 0;
    for (int k = 0; k < GF64_HQC_MAX_LM; k++) {
        if ((j >> k) & 1) v ^= GF64_CANTOR_BASIS[k];
    }
    return v;
}

/* s_j(x) iterated from the Cantor recurrence sigma(x) = x^2 + x:
 *
 *   s_0(x) = x
 *   s_{j+1}(x) = (s_j(x))^2 + s_j(x)
 *
 * Polynomial truncation at degree n-1 applies; coefficients are in GF(2) ⊂
 * GF(2^64) (since Frobenius x → x^2 sends 1 → 1), so squaring a
 * coefficient doesn't change it.
 *
 * NOTE: the formula s_j(x) = sum_{k=0..j} x^{2^k} only holds for some
 * j (e.g. j = 0, 1, 3 but NOT j = 2, 4 where cancellations produce
 * narrower polynomials). Computed iteratively instead.
 */
static void compute_sj(int j, int n, gf64_t *s_out) {
    gf64_t cur[GF64_HQC_MAX_N];
    gf64_t sq_part[GF64_HQC_MAX_N];
    for (int i = 0; i < n; i++) cur[i] = 0;
    cur[1] = 1;  /* s_0 = x */

    for (int iter = 1; iter <= j; iter++) {
        /* s_iter = sq(s_{iter-1}) + s_{iter-1}. Frobenius squaring: shift
         * each nonzero coefficient to degree 2·i (coefficient unchanged
         * since it's in {0, 1}). */
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

/* Multiply two polynomials in GF(2^64)[x], truncated at degree < n. */
static void poly_mul_trunc(int n, const gf64_t *a, const gf64_t *b, gf64_t *out) {
    for (int i = 0; i < n; i++) out[i] = 0;
    for (int ja = 0; ja < n; ja++) {
        if (a[ja] == 0) continue;
        for (int jb = 0; jb < n - ja; jb++) {
            if (b[jb] == 0) continue;
            out[ja + jb] ^= gf64_mul_reference(a[ja], b[jb]);
        }
    }
}

/* s_i(a) for a field element a; per Cantor recurrence,
 *   s_i(v_j) = v_{j-i} for j >= i, else 0.
 *
 * `n_table` is the size of the v_table backing buffer (always equal to
 * the TOP-LEVEL transform size — set at gf64_addfft64_fwd entry). It is
 * NOT the current recursion level's n, because at deeper levels the
 * affine shift `a` may carry Cantor indices beyond the current level's
 * range (e.g., n=2 base case recurses with a = basis[1] XOR basis[2]
 * whose Cantor index 6 sits well above the level-2 bound). Searching
 * with the wrong bound silently returns 0, demoting the butterfly to a
 * trivial fold at deeper levels — which is precisely the n>=8 failure
 * pattern observed in test_gf64_additive_fft_hqc2026.c. */
static int compute_index_for(gf64_t a, const gf64_t *v_table, int n_table) {
    if (a == 0) return 0;
    for (int j = 1; j < n_table; j++) {
        if (v_table[j] == a) return j;
    }
    return -1;
}
static gf64_t si_eval(int i, gf64_t a, const gf64_t *v_table, int n_table) {
    if (a == 0) return 0;
    int idx = compute_index_for(a, v_table, n_table);
    if (idx < 0) return 0;
    int shifted = idx >> i;
    if (shifted == 0) return 0;
    return v_table[shifted];
}

/* ----- change-of-basis matrix ----- *
 *
 * The novelpoly basis {X_0, ..., X_{n-1}} is linearly independent in
 * GF(2^64)^n for n ≤ 2^ℓ_m. The change-of-basis matrix M (column k =
 * monomial coeffs of X_k) is invertible. We compute (M^T)^{-1} (call
 * it M_inv) so that g = M_inv · a (the monomial-to-novelpoly conversion).
 *
 * Cached per n: first call for a given n pays O(n^3) for the
 * Gauss-Jordan matrix inversion; subsequent calls are O(n^2).
 * 6 slots, eviction-by-overwrite (no LRU).
 */

typedef struct {
    int n;
    gf64_t *M;       /* n × n: column k = monomial coeffs of X_k */
    gf64_t *M_inv;   /* n × n: M^{-1}, so g = M_inv · a (solve a = M·g) */
    gf64_t *v_table; /* v_table[j] = W_m(j), j ∈ [0, n) */
    int initialized;
} hqc_basis_cache_t;

#define HQC_CACHE_SLOTS 6
static hqc_basis_cache_t hqc_cache[HQC_CACHE_SLOTS];

/* Build X_basis[k * n + j] = coefficient of x^j in X_k(x). */
static gf64_t *build_X_basis(int n) {
    gf64_t *X = (gf64_t *)calloc((size_t)n * n, sizeof(gf64_t));
    if (X == NULL) abort();
    gf64_t cur[GF64_HQC_MAX_N];
    gf64_t s_poly[GF64_HQC_MAX_N];
    gf64_t new_cur[GF64_HQC_MAX_N];
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) cur[i] = 0;
        cur[0] = 1;   /* empty product = identity */
        for (int bit_pos = 0; bit_pos < GF64_HQC_MAX_LM; bit_pos++) {
            if (!((k >> bit_pos) & 1)) continue;
            compute_sj(bit_pos, n, s_poly);
            poly_mul_trunc(n, cur, s_poly, new_cur);
            memcpy(cur, new_cur, n * sizeof(gf64_t));
        }
        for (int j = 0; j < n; j++) X[k * n + j] = cur[j];
    }
    return X;
}

static hqc_basis_cache_t *get_or_build_basis_cache(int n) {
    for (int s = 0; s < HQC_CACHE_SLOTS; s++) {
        if (hqc_cache[s].initialized && hqc_cache[s].n == n) return &hqc_cache[s];
    }
    int slot = -1;
    for (int s = 0; s < HQC_CACHE_SLOTS; s++) {
        if (!hqc_cache[s].initialized) { slot = s; break; }
    }
    if (slot < 0) slot = 0;
    hqc_basis_cache_t *c = &hqc_cache[slot];
    if (c->initialized) {
        free(c->M); free(c->M_inv); free(c->v_table);
        c->initialized = 0;
    }

    c->n = n;
    c->M       = (gf64_t *)calloc((size_t)n * n, sizeof(gf64_t));
    c->M_inv   = (gf64_t *)calloc((size_t)n * n, sizeof(gf64_t));
    c->v_table = (gf64_t *)calloc((size_t)n, sizeof(gf64_t));
    if (c->M == NULL || c->M_inv == NULL || c->v_table == NULL) abort();

    for (int j = 0; j < n; j++) c->v_table[j] = compute_v_j(j);

    gf64_t *X = build_X_basis(n);
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++)
            c->M[j * n + k] = X[k * n + j];
    free(X);

    /* Gauss-Jordan over GF(2^64): compute M_inv = M^{-1}.
     * The change-of-basis relation (column k of M = monomial coeffs of X_k)
     * means a = M · g for monomial coeffs a and novelpoly coeffs g,
     * so solving for g gives g = M^{-1} · a. */
    size_t aug_size = (size_t)(2 * n);
    gf64_t *aug = (gf64_t *)calloc((size_t)n * aug_size, sizeof(gf64_t));
    if (aug == NULL) abort();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i * aug_size + j] = c->M[i * n + j];
        for (int j = 0; j < n; j++) aug[i * aug_size + (n + j)] = (i == j) ? 1 : 0;
    }
    for (int col = 0; col < n; col++) {
        int pivot = -1;
        for (int r = col; r < n; r++) {
            if (aug[r * aug_size + col] != 0) { pivot = r; break; }
        }
        if (pivot < 0) {
            free(aug); free(c->M); free(c->M_inv); free(c->v_table);
            c->initialized = 0;
            return NULL;
        }
        if (pivot != col) {
            for (size_t j = 0; j < aug_size; j++) {
                gf64_t tmp = aug[col * aug_size + j];
                aug[col * aug_size + j] = aug[pivot * aug_size + j];
                aug[pivot * aug_size + j] = tmp;
            }
        }
        gf64_t pv_inv = gf64_inverse(aug[col * aug_size + col]);
        for (size_t j = 0; j < aug_size; j++)
            aug[col * aug_size + j] = gf64_mul_reference(aug[col * aug_size + j], pv_inv);
        for (int r = 0; r < n; r++) {
            if (r == col) continue;
            gf64_t factor = aug[r * aug_size + col];
            if (factor == 0) continue;
            for (size_t j = 0; j < aug_size; j++)
                aug[r * aug_size + j] ^= gf64_mul_reference(factor, aug[col * aug_size + j]);
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            c->M_inv[i * n + j] = aug[i * aug_size + ((size_t)n + (size_t)j)];
    free(aug);

    c->initialized = 1;
    return c;
}

/* monomial → novelpoly.
 *
 * CURRENT IMPLEMENTATION: explicit O(n²) matrix-vector multiply using the
 * cached M_inv. Simple to verify bit-exactly; the working set at n = 4096
 * is ~128 KiB and at n = 16384 is ~2 MiB (both fit comfortably in L2).
 *
 * DEFERRED (FIX-3, see PHASE_2c_FINDINGS_2026-07-15.md): replace with the
 * **Chen 2018 Algorithm 1 recursive BasisCvt**, which costs O(n log n) per
 * call by exploiting the hierarchical structure of the novelpoly basis.
 * The recursive form factors M_inv = K^{-1} · P^T where K is a Kronecker
 * product of small (constant-size) matrices and P is a permutation; applying
 * K^{-1} · P^T to a vector costs O(n log n) field ops by the divide-and-
 * conquer recursion. Until FIX-3 lands, this function dominates the
 * additive-FFT wall-clock at large n (measured: 76 ms at n = 4096, 308 ms
 * at n = 8192, scaling as exactly 4× per 2× n).
 *
 * Inputs:
 *   g  — output buffer, length n (novelpoly coeffs)
 *   c  — input buffer, length n (monomial coeffs)
 *   n  — transform size (power of 2)
 *   M_inv — n×n matrix, M_inv[k*n+j] is row k column j of M^{-1}
 */
static void basisCvt(gf64_t *g, const gf64_t *c, int n, const gf64_t *M_inv) {
    for (int k = 0; k < n; k++) {
        gf64_t acc = 0;
        for (int j = 0; j < n; j++) acc ^= gf64_mul_reference(M_inv[k * n + j], c[j]);
        g[k] = acc;
    }
}

/* novelpoly → monomial. Same FIX-3 deferral as basisCvt above. */
static void ibasisCvt(gf64_t *c, const gf64_t *g, int n, const gf64_t *M) {
    for (int j = 0; j < n; j++) {
        gf64_t acc = 0;
        for (int k = 0; k < n; k++) acc ^= gf64_mul_reference(M[j * n + k], g[k]);
        c[j] = acc;
    }
}

/* ----- Butterfly (radix-2 DIT, depth-first) ----- *
 *
 * Recursive in-place transformation. The recursion picks an affine shift
 * `a` at the entry (passed in from the caller — for our public API this
 * is GF64_CANTOR_BASIS[logn-1], outside V_{logn-1}). The right-half
 * recursion shifts by v_{logn-1} = W_m(1 << (logn-1)) which XORs in the
 * next Cantor basis vector.
 */
static void butterfly_fwd(gf64_t *f, int n, int n_table, gf64_t a,
                          const gf64_t *v_table, int logn) {
    if (n == 2) {
        gf64_t fl = f[0], fh = f[1];
        f[0] = fl ^ gf64_mul_reference(a, fh);
        f[1] = fl ^ gf64_mul_reference(a ^ 1, fh);
        return;
    }
    int half = n / 2;
    gf64_t s_a = si_eval(logn - 1, a, v_table, n_table);
    for (int j = 0; j < half; j++) {
        gf64_t fl = f[j], fh = f[j + half];
        f[j]       = fl ^ gf64_mul_reference(s_a, fh);
        f[j + half] = fl ^ gf64_mul_reference(s_a ^ 1, fh);
    }
    butterfly_fwd(f,         half, n_table, a,                       v_table, logn - 1);
    butterfly_fwd(f + half,  half, n_table, a ^ v_table[1 << (logn - 1)],
                  v_table, logn - 1);
}

static void butterfly_inv(gf64_t *f, int n, int n_table, gf64_t a,
                          const gf64_t *v_table, int logn) {
    if (n == 2) {
        gf64_t fh = f[0] ^ f[1];
        gf64_t fl = f[0] ^ gf64_mul_reference(a, fh);
        f[0] = fl; f[1] = fh;
        return;
    }
    int half = n / 2;
    butterfly_inv(f,         half, n_table, a,                       v_table, logn - 1);
    butterfly_inv(f + half,  half, n_table, a ^ v_table[1 << (logn - 1)],
                  v_table, logn - 1);
    gf64_t s_a = si_eval(logn - 1, a, v_table, n_table);
    for (int j = 0; j < half; j++) {
        gf64_t fl_new = f[j], fh_new = f[j + half];
        gf64_t fh = fl_new ^ fh_new;
        gf64_t fl = fl_new ^ gf64_mul_reference(s_a, fh);
        f[j] = fl;
        f[j + half] = fh;
    }
}

/* ----- Public API ----- */

void gf64_addfft64_fwd(gf64_t *arr, size_t n) {
    if (n <= 1) return;
    assert(n <= GF64_HQC_MAX_N);

    int n_int = (int)n;
    hqc_basis_cache_t *cache = get_or_build_basis_cache(n_int);
    assert(cache != NULL);

    int logn = 0; while ((1 << logn) < n_int) logn++;
    /* Affine shift = basis vector at the depth of this transform's
     * primary vanishing subspace boundary: GF64_CANTOR_BASIS[logn-1].
     * For n = 2^i, this element has Cantor-index 1 << (logn-1), whose
     * bit-(logn-1) is set, so it's outside V_{logn-1}. */
    gf64_t a = GF64_CANTOR_BASIS[logn - 1];

    gf64_t g[GF64_HQC_MAX_N];
    basisCvt(g, arr, n_int, cache->M_inv);
    memcpy(arr, g, n * sizeof(gf64_t));
    butterfly_fwd(arr, n_int, n_int, a, cache->v_table, logn);
}

void gf64_addfft64_inv(gf64_t *arr, size_t n) {
    if (n <= 1) return;
    assert(n <= GF64_HQC_MAX_N);

    int n_int = (int)n;
    hqc_basis_cache_t *cache = get_or_build_basis_cache(n_int);
    assert(cache != NULL);

    int logn = 0; while ((1 << logn) < n_int) logn++;
    gf64_t a = GF64_CANTOR_BASIS[logn - 1];

    butterfly_inv(arr, n_int, n_int, a, cache->v_table, logn);
    gf64_t c[GF64_HQC_MAX_N];
    ibasisCvt(c, arr, n_int, cache->M);
    memcpy(arr, c, n * sizeof(gf64_t));
}

void gf64_addfft64_poly_mul(
    gf64_t *out,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b,
    size_t out_len)
{
    if (len_a == 0 || len_b == 0 || out_len == 0) {
        memset(out, 0, out_len * sizeof(gf64_t));
        return;
    }
    size_t full_len = len_a + len_b - 1;
    if (full_len < out_len) full_len = out_len;
    size_t n = 1;
    while (n < full_len) n <<= 1;
    if (n > GF64_HQC_MAX_N) n = GF64_HQC_MAX_N;

    /* Allocate padded scratch — call-local, NOT aliased with caller 'out'.
     * Honors the in-place footgun rule documented in user memory:
     * gf64_poly_mul_padded memsets 'out' before reading inputs; 'out == a'
     * breaks the result. We don't read 'out' before writing it. */
    gf64_t *padded = (gf64_t *)calloc(n * 2, sizeof(gf64_t));
    if (padded == NULL) abort();
    gf64_t *pa = padded;
    gf64_t *pb = padded + n;

    memcpy(pa, a, len_a * sizeof(gf64_t));
    memset(pa + len_a, 0, (n - len_a) * sizeof(gf64_t));
    memcpy(pb, b, len_b * sizeof(gf64_t));
    memset(pb + len_b, 0, (n - len_b) * sizeof(gf64_t));

    gf64_addfft64_fwd(pa, n);
    gf64_addfft64_fwd(pb, n);
    for (size_t i = 0; i < n; i++) pa[i] = gf64_mul_reference(pa[i], pb[i]);
    gf64_addfft64_inv(pa, n);

    size_t copy_n = (full_len < out_len) ? full_len : out_len;
    memcpy(out, pa, copy_n * sizeof(gf64_t));
    if (copy_n < out_len) memset(out + copy_n, 0, (out_len - copy_n) * sizeof(gf64_t));
    free(padded);
}

HEDLEY_END_C_DECLS
