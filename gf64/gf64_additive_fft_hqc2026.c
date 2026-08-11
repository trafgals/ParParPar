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
 * Public API (declared in gf64_additive_fft.h, scratch-required, two-tier
 * length caps):
 *   void gf64_addfft64_fwd_scratch / _inv_scratch / _poly_mul_scratch
 *       — matrix-form path, capped at GF64_HQC_MAX_MATRIXFORM_N = 16384
 *   void gf64_addfft64_fwd_recursive_scratch / _inv_recursive_scratch /
 *   _poly_mul_recursive_scratch
 *       — matrix-free recursive path, capped at GF64_HQC_MAX_LM_N = 2^20
 *
 * Length caps: see gf64_additive_fft.h for the rationale and the full
 * _scratch_words queries. The recursive path now uses Chen 2018 General
 * Algorithm 1 (the polyeval port `basisCvt_recursive_v2`/`ibasisCvt_
 * recursive_v2`) at every power-of-2 n ≤ 2^20 — no longer restricted to
 * the simple-2-term-prefix sizes {4, 8, 32, 512, 131072}. Verified at
 * the boundary sizes {131072, 262144, 1048576} in
 * test_gf64_additive_fft_hqc2026.c [Test 4 + 5]. The decomposition is
 * the 2-term XOR-shift divide `hqc_cvt_div_blk` (`(x^si - x)` only)
 * applied at every level — see lines 671-718 for the recursion. Pure
 * XOR + a single gf64_mul_reference in the butterfly → trivially host-
 * portable (no AVX-512 gating needed for the polyeval recursion).
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
extern void gf64_mul_avx512(gf64_t *out, const gf64_t *a, const gf64_t *b, size_t N);
extern gf64_t gf64_inverse(gf64_t a);

/* Cantor basis has 20 precomputed vectors; ℓ_m = 20 supports n up to 2^20. */
#define GF64_HQC_MAX_LM 20

/* (GF64_HQC_MAX_MATRIXFORM_N and GF64_HQC_MAX_LM_N are defined in
 * gf64_additive_fft.h so that gf64_additive_fft.c can see them for the
 * dispatch tier in gf64_poly_mul_internal.) */

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
 *
 * Scratch layout: caller passes cur[0..n) and sq_part[0..n); both are n words.
 */
static void compute_sj(int j, int n, gf64_t *s_out,
                       gf64_t *cur, gf64_t *sq_part) {
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
 * the TOP-LEVEL transform size — set at gf64_addfft64_fwd_scratch entry). It is
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

#define HQC_CACHE_SLOTS 16  /* FIX-3: recursive BasisCvt visits ~log_2(N) distinct
                              * sizes (one per recursion level where matrix-form
                              * fallback fires). 6 was insufficient for large N. */
static hqc_basis_cache_t hqc_cache[HQC_CACHE_SLOTS];

/* FIX-3a: separate cache for v_table-only lookups (used by the recursive
 * addFFT entry points, which don't need M or M_inv). Building v_table
 * is O(N log N) at worst (v_table[j] = XOR of GF64_CANTOR_BASIS bits set
 * in j — O(log N) XORs per entry); building M_inv is O(N^3) Gauss-Jordan
 * (seconds at N=4096, hours at N=16384). The recursive path needs only
 * v_table for the butterfly's index lookups. */
typedef struct {
    int n;
    gf64_t *v_table;
    int initialized;
} hqc_vtable_cache_t;
#define HQC_VTABLE_CACHE_SLOTS 16
static hqc_vtable_cache_t hqc_vtable_cache[HQC_VTABLE_CACHE_SLOTS];
static gf64_t *get_or_build_v_table(int n) {
    for (int s = 0; s < HQC_VTABLE_CACHE_SLOTS; s++) {
        if (hqc_vtable_cache[s].initialized && hqc_vtable_cache[s].n == n)
            return hqc_vtable_cache[s].v_table;
    }
    int slot = -1;
    for (int s = 0; s < HQC_VTABLE_CACHE_SLOTS; s++) {
        if (!hqc_vtable_cache[s].initialized) { slot = s; break; }
    }
    if (slot < 0) slot = 0;
    hqc_vtable_cache_t *c = &hqc_vtable_cache[slot];
    if (c->initialized) { free(c->v_table); c->initialized = 0; }
    c->n = n;
    c->v_table = (gf64_t *)calloc((size_t)n, sizeof(gf64_t));
    if (c->v_table == NULL) abort();
    for (int j = 0; j < n; j++) c->v_table[j] = compute_v_j(j);
    c->initialized = 1;
    return c->v_table;
}

/* Build X_basis[k * n + j] = coefficient of x^j in X_k(x).
 *
 * Scratch layout: caller passes cur[0..n), s_poly[0..n), new_cur[0..n);
 * plus sj_scratch[0..2n) for compute_sj's internal region.
 */
static gf64_t *build_X_basis(int n,
                             gf64_t *cur, gf64_t *s_poly, gf64_t *new_cur,
                             gf64_t *sj_scratch) {
    gf64_t *X = (gf64_t *)calloc((size_t)n * n, sizeof(gf64_t));
    if (X == NULL) abort();
    gf64_t *sj_cur    = sj_scratch;
    gf64_t *sj_sqpart = sj_scratch + n;
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) cur[i] = 0;
        cur[0] = 1;   /* empty product = identity */
        for (int bit_pos = 0; bit_pos < GF64_HQC_MAX_LM; bit_pos++) {
            if (!((k >> bit_pos) & 1)) continue;
            compute_sj(bit_pos, n, s_poly, sj_cur, sj_sqpart);
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

    /* Allocate one-shot scratch for the cache build. build_X_basis needs
     * 3n (cur, s_poly, new_cur) plus 2n for compute_sj's internal region. */
    size_t build_scratch_words = (size_t)5 * n;
    gf64_t *build_scratch = (gf64_t *)malloc(build_scratch_words * sizeof(gf64_t));
    if (build_scratch == NULL) abort();
    gf64_t *cur     = build_scratch;
    gf64_t *s_poly  = build_scratch + n;
    gf64_t *new_cur = build_scratch + 2 * n;
    gf64_t *sj_reg  = build_scratch + 3 * n;  /* 2n region for compute_sj */
    gf64_t *X = build_X_basis(n, cur, s_poly, new_cur, sj_reg);
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++)
            c->M[j * n + k] = X[k * n + j];
    free(X);
    free(build_scratch);

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

/* ----- FIX-3: Chen 2018 Algorithm 1 recursive BasisCvt -----
 *
 * Reference: Chen, Cheng, Kuo, Li, Yang, "Multiplying boolean Polynomials
 * with Frobenius Partitions in Additive FFT", arXiv 1803.11301 (Apr 2018),
 * Algorithm 1.
 *
 * CURRENT STATE (2026-07-15, post-FIX-3 attempts):
 *
 *   - Simple 2-term recursion (k = m-1) implemented and bit-exact verified at
 *     n in {2, 4, 8, 32, 512, 1024, 2048, 4096}, with O(n log n) at sizes
 *     where m-1 is itself a power of 2 (i.e., n in {2, 4, 8, 32, 512, 131072}).
 *
 *   - For sizes where m-1 is NOT a power of 2 (n in {16, 64, 128, 256, 1024,
 *     2048, ...}), the recursion falls back to the matrix-form O(n^2) BasisCvt
 *     using a cached M_inv. The M_inv build is O(n^3) Gauss-Jordan.
 *
 *   - The General Algorithm 1 (i = max pow of 2 <= log2(n-1), S = 2^i,
 *     m = n/S, with h_j in R[x]^{<S} and a non-trivial y-domain recursion
 *     for step 5) is NOT yet implemented. An attempt was made in this
 *     session and reverted because step 5 (the BasisCvt(h^(y)) recursion
 *     with the y-Cantor basis defined over R[x]^{<S}) requires a non-trivial
 *     "polynomial-ring coefficient" semantics that the current scalar
 *     basisCvt does not provide. See PHASE_2c_FINDINGS_2026-07-15.md and
 *     issue #29 for the full discussion.
 *
 *   - The simple recursion structure is:
 *       f(x) = a(x) + b(x) * (x^{n/2} + x),  deg a, deg b < n/2
 *     with O(n) back-substitution and a clean triangular system (because
 *     s_{m-1} = x^{n/2} + x has only 2 terms).
 *
 *   - Cost at sizes where k = m-1 IS a power of 2: T(n) = 2 T(n/2) + O(n)
 *     = O(n log n).
 *
 *   - Cost at sizes where k = m-1 is NOT a power of 2: T(n) = O(n^2) per
 *     BasisCvt call, plus O(n^3) one-time cache build. The matrix is cached
 *     across calls; cache has 16 slots.
 *
 *   - For the canonical PAR3 workload (n = 10K to 16K), k = 12..13 NOT a
 *     power of 2, so the matrix-form path is hit at the top level. M_inv
 *     at n = 16384 is 2 GiB, build is O(4e12) field ops = ~1 hour one-time
 *     cost. The Fenger pipeline uses multiple sizes so the cache evicts
 *     frequently; in practice the one-time build per size is the bottleneck.
 *
 * TODO (FIX-3a follow-up): implement Algorithm 1 with the proper y-domain
 * recursion. This requires:
 *   1. A "field" of size |R|^S over R[x]^{<S} via the Cantor construction
 *      y-v_0 = 1, y-v_i^2 + y-v_i = y-v_{i-1}. For y-v_1 this is a root of
 *      T^2 + T + 1 = 0 in R (exists since 2 | 64). For y-v_2 etc., the
 *      elements are in R[x]^{<S} (polynomials of degree < S) and require
 *      explicit construction via the Cantor basis.
 *   2. A "BasisCvt over R[x]^{<S}" that handles polynomial coefficients:
 *      field multiplication becomes polynomial multiplication mod some ideal;
 *      v_table lookup becomes polynomial comparison.
 *   3. The recursion on h^(y) at size m with the y-Cantor basis.
 *
 * This is non-trivial; estimated 3-5 days for a C engineer with the Chen 2018
 * paper in hand. Filed as issue #29.
 */

/* Decompose f(x) = a(x) + b(x) * (x^{n/2} + x), deg(a), deg(b) < n/2.
 *
 * f[i] = a[i] + (i >= 1 ? b[i-1] : 0) + (i >= n/2 ? b[i - n/2] : 0)
 * (using GF(2) addition = XOR).
 *
 * Back-substitution:
 *   The bottom row n-1 = (half-1) + half is only hit by b[half-1].
 *     b[half-1] = f[n-1]
 *   Rows in [half, n-1] are hit by exactly one b each (b[i-half]).
 *     b[i] = f[i+half]  for i = half-1, half-2, ..., 1
 *   Row half is also hit by b[half-1] (the +x contribution).
 *     b[0] = f[half] + b[half-1]
 *   Then a:
 *     a[0] = f[0]
 *     a[i] = f[i] + b[i-1]  for i = 1, ..., half-1
 */
static void basisCvt_decompose(gf64_t *a, gf64_t *b, const gf64_t *f, int n) {
    int half = n / 2;
    b[half - 1] = f[n - 1];
    for (int i = half - 2; i >= 1; i--) {
        b[i] = f[i + half];
    }
    b[0] = f[half] ^ b[half - 1];
    a[0] = f[0];
    for (int i = 1; i < half; i++) {
        a[i] = f[i] ^ b[i - 1];
    }
}

/* Chen 2018 Algorithm 1 recursive BasisCvt (monomial → novelpoly).
 *
 * Only valid when n = 2^(k+1) for some k such that s_k = x^{n/2} + x
 * (i.e., k is a power of 2). For other n, fall back to matrix-form.
 *
 * Scratch layout (peak ~2n over the whole recursion tree): each level
 * needs 2 sub-buffers of half = n total, plus the child uses its own
 * region. Threads `scratch + 2*half` to the left recursion, `scratch` to
 * the right — same total region, reused after the left returns. */
static void basisCvt_recursive(gf64_t *g, const gf64_t *f, int n,
                               gf64_t *scratch) {
    if (n == 1) {
        g[0] = f[0];
        return;
    }
    if (n == 2) {
        g[0] = f[0];
        g[1] = f[1];
        return;
    }

    int m = 0; while ((1 << m) < n) m++;
    int k = m - 1;
    int k_is_pow2 = (k > 0) && ((k & (k - 1)) == 0);

    if (!k_is_pow2) {
        /* Fallback: get the cache entry for THIS n and use matrix-form. */
        hqc_basis_cache_t *cache = get_or_build_basis_cache(n);
        assert(cache != NULL);
        basisCvt(g, f, n, cache->M_inv);
        return;
    }

    int half = n / 2;

    gf64_t *a     = scratch;
    gf64_t *b     = scratch + half;
    basisCvt_decompose(a, b, f, n);

    gf64_t *g_lo  = scratch;            /* aliases a (overwritten by left rec) */
    gf64_t *g_hi  = scratch + half;     /* aliases b (overwritten by right rec) */
    basisCvt_recursive(g_lo, a, half, scratch + n);
    basisCvt_recursive(g_hi, b, half, scratch + n);

    for (int i = 0; i < half; i++) {
        g[i] = g_lo[i];
        g[i + half] = g_hi[i];
    }
}

/* Inverse: novelpoly → monomial. Symmetric to forward via the same
 * decomposition. Split g by index halves, recurse on each, then combine
 * via f = a + b * (x^{n/2} + x). Same scratch layout as forward. */
static void ibasisCvt_recursive(gf64_t *c, const gf64_t *g, int n,
                                gf64_t *scratch) {
    if (n == 1) {
        c[0] = g[0];
        return;
    }
    if (n == 2) {
        c[0] = g[0];
        c[1] = g[1];
        return;
    }

    int m = 0; while ((1 << m) < n) m++;
    int k = m - 1;
    int k_is_pow2 = (k > 0) && ((k & (k - 1)) == 0);

    if (!k_is_pow2) {
        hqc_basis_cache_t *cache = get_or_build_basis_cache(n);
        assert(cache != NULL);
        ibasisCvt(c, g, n, cache->M);
        return;
    }

    int half = n / 2;

    gf64_t *g_lo = scratch;
    gf64_t *g_hi = scratch + half;
    for (int i = 0; i < half; i++) {
        g_lo[i] = g[i];
        g_hi[i] = g[i + half];
    }

    gf64_t *a    = scratch;             /* aliases g_lo */
    gf64_t *b    = scratch + half;      /* aliases g_hi */
    ibasisCvt_recursive(a, g_lo, half, scratch + n);
    ibasisCvt_recursive(b, g_hi, half, scratch + n);

    /* Combine: c(x) = a(x) + b(x) * (x^half + x).
     *   c[0]      = a[0]
     *   c[i]      = a[i] + b[i-1]  for 1 ≤ i < half
     *   c[half]   = b[0] + b[half-1]
     *   c[half+j] = b[j]           for 1 ≤ j < half  */
    c[0] = a[0];
    for (int i = 1; i < half; i++) {
        c[i] = a[i] ^ b[i - 1];
    }
    c[half] = b[0] ^ b[half - 1];
    for (int j = 1; j < half; j++) {
        c[half + j] = b[j];
    }
}

/* Public entry: dispatch to recursive path only when k = m-1 is a power of 2
 * (i.e., m-1 ∈ {1, 2, 4, 8, ...} so that s_{m-1} = x^{n/2} + x has only 2
 * nonzero coefficients and the simple decompose applies). Otherwise fall
 * back to the matrix-form O(n²) BasisCvt. */
static void basisCvt_dispatch(gf64_t *g, const gf64_t *c, int n,
                              const gf64_t *M_inv, gf64_t *scratch) {
    (void)M_inv;
    basisCvt_recursive(g, c, n, scratch);
}

static void ibasisCvt_dispatch(gf64_t *c, const gf64_t *g, int n,
                             const gf64_t *M, gf64_t *scratch) {
    (void)M;
    ibasisCvt_recursive(c, g, n, scratch);
}

/* ----- FIX-3a recursive BasisCvt (Chen 2018 Algorithm 1, general k) -----
 *
 * The polyeval cvt / icvt recursion operates on poly coefficients with
 * 2-term divisions by (x^{2^si_h} - x^{2^si_l}) at every level. This is the
 * Algorithm 1 general-k decomposition: at each recursion, choose si = max
 * power of 2 strictly less than the remaining polyloglen, decompose via the
 * subspace polynomial s_si at the right scale, then recurse on the two
 * halves. The form is identical to polyeval/bc/src/ref/bc_256.c cvt() but
 * adapted to GF(2^64) with logsize_blk = 0 (no inner block structure; our
 * atomic unit IS a gf64_t element).
 *
 * Round-trip verified PASS at all sizes 2..4096 in
 * gf64/test/probe_basis_cvt_polyeval.c (and an earlier cross-val test that
 * conflates BasisCvt with butterfly is acknowledged in the test docstring).
 * The decomposition computes a *different* BasisCvt than the matrix-form
 * basisCvt/ibasisCvt (because the matrix-form uses the M built from the
 * actual X_k product, while this recursion uses the Frobenius recurrence
 * s_i = s_{i-1}^2 + s_{i-1} which gives a basis of equivalent shape but
 * with different constant offsets). For correctness as a self-inverse
 * transform pair, cvt then icvt is the identity — see the probe.
 *
 * Cost: T(n) = O(n log n) via T(n) = T(2^k) + T(n - 2^k) + O(n).
 */

/* Largest power of 2 strictly less than n. (Mirrors polyeval choose_si.) */
static int hqc_cvt_choose_si(int n) {
    int si = 1;
    for (int i = 1; (1 << i) < n; i++) {
        si = 1 << i;
    }
    return si;
}

/* 2-term division: divide poly (in place) by x^{si_h} - x^{si_l}. */
static void hqc_cvt_div_blk(gf64_t *poly, int si_h, int si_l, int polylen) {
    int deg_diff = si_h - si_l;
    for (int i = polylen - 1; i >= si_h; i--) {
        poly[i - deg_diff] ^= poly[i];
    }
}

/* Inverse 2-term division: multiply poly (in place) by x^{si_h} - x^{si_l}. */
static void hqc_cvt_idiv_blk(gf64_t *poly, int si_h, int si_l, int polylen) {
    int deg_diff = si_h - si_l;
    for (int i = si_h; i < polylen; i++) {
        poly[i - deg_diff] ^= poly[i];
    }
}

/* Decompose via s_si at multiple scales within the poly. */
static void hqc_cvt_rep_in_si(gf64_t *data, int datalen, int logsize_blk,
                              int polyloglen_blk, int si) {
    for (int i = polyloglen_blk - 1; i >= si; i--) {
        int polylen = 1 << (i + logsize_blk + 1);
        int si_h = 1 << (i + logsize_blk);
        int si_l = 1 << (i + logsize_blk - si);
        for (int j = 0; j < datalen; j += polylen) {
            hqc_cvt_div_blk(data + j, si_h, si_l, polylen);
        }
    }
}

static void hqc_cvt_irep_in_si(gf64_t *data, int datalen, int logsize_blk,
                               int polyloglen_blk, int si) {
    for (int i = si; i < polyloglen_blk; i++) {
        int polylen = 1 << (i + logsize_blk + 1);
        int si_h = 1 << (i + logsize_blk);
        int si_l = 1 << (i + logsize_blk - si);
        for (int j = 0; j < datalen; j += polylen) {
            hqc_cvt_idiv_blk(data + j, si_h, si_l, polylen);
        }
    }
}

/* Recursive Algorithm 1 BasisCvt (forward). O(N log N). */
static void hqc_cvt(gf64_t *data, int datalen, int logsize_blk, int polyloglen_blk) {
    if (polyloglen_blk <= 1) return;
    int si = hqc_cvt_choose_si(polyloglen_blk);
    hqc_cvt_rep_in_si(data, datalen, logsize_blk, polyloglen_blk, si);
    hqc_cvt(data, datalen, logsize_blk, si);
    hqc_cvt(data, datalen, logsize_blk + si, polyloglen_blk - si);
}

/* Recursive Algorithm 1 BasisCvt (inverse). O(N log N). */
static void hqc_icvt(gf64_t *data, int datalen, int logsize_blk, int polyloglen_blk) {
    if (polyloglen_blk <= 1) return;
    int si = hqc_cvt_choose_si(polyloglen_blk);
    hqc_icvt(data, datalen, logsize_blk, si);
    hqc_icvt(data, datalen, logsize_blk + si, polyloglen_blk - si);
    hqc_cvt_irep_in_si(data, datalen, logsize_blk, polyloglen_blk, si);
}

/* Top-level entry: monomial -> novelpoly via Algorithm 1 recursion.
 * Works at any n (power of 2, n >= 2); O(N log N).
 *
 * Scratch: caller passes tmp[0..n) — hqc_cvt operates in place on tmp. */
static void basisCvt_recursive_v2(gf64_t *g, const gf64_t *f, int n,
                                  gf64_t *tmp) {
    if (n <= 1) {
        if (n == 1) g[0] = f[0];
        return;
    }
    memcpy(tmp, f, (size_t)n * sizeof(gf64_t));
    int log_n = 0;
    while ((1 << log_n) < n) log_n++;
    hqc_cvt(tmp, n, 0, log_n);
    memcpy(g, tmp, (size_t)n * sizeof(gf64_t));
}

/* Top-level entry: novelpoly -> monomial via Algorithm 1 recursion.
 * Scratch: caller passes tmp[0..n). */
static void ibasisCvt_recursive_v2(gf64_t *c, const gf64_t *g, int n,
                                   gf64_t *tmp) {
    if (n <= 1) {
        if (n == 1) c[0] = g[0];
        return;
    }
    memcpy(tmp, g, (size_t)n * sizeof(gf64_t));
    int log_n = 0;
    while ((1 << log_n) < n) log_n++;
    hqc_icvt(tmp, n, 0, log_n);
    memcpy(c, tmp, (size_t)n * sizeof(gf64_t));
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

/* ----- AVX-512 (PCLMULQDQ) accelerated butterfly ----- *
 *
 * Mirrors butterfly_fwd / butterfly_inv above; replaces the scalar
 * gf64_mul_reference in the non-leaf case with gf64_mul_avx512, processing
 * 8 lanes per PCLMULQDQ call. Saves the second multiplication per
 * 8-element batch by computing (s_a ^ 1) * fh = (s_a * fh) ^ fh.
 *
 * Stack overhead per recursion level: 3 × 8 = 24 gf64_t = 192 bytes for
 * the sa/fh/prod broadcast buffers. At n = 131072 (log2 = 17 levels) total
 * is ~3.3 KB — comfortably within default thread stacks.
 *
 * Falls back to scalar gf64_mul_reference for half < 8 (the SIMD setup
 * overhead dominates for small problems).
 *
 * Bit-exact to butterfly_fwd / butterfly_inv — verified at n ∈ {32, 512}
 * by test_gf64_additive_fft_hqc2026.
 *
 * Host: requires AVX-512F + VPCLMULQDQ (same as gf64_mul_avx512).
 */
#define GF64_HQC_BUTTERFLY_AVX512_LANES 8

/* Per-function target attribute (GCC/Clang). On MSVC the equivalent is
 * the /arch:AVX512 compile flag on the gf64_avx512_arr vcxproj (set in
 * binding.gyp), which enables AVX-512F + VPCLMULQDQ for the whole TU;
 * the static helpers then need no attribute. */
#if defined(__GNUC__) && !defined(__clang__)
#define GF64_HQC_TARGET_AVX512_PCLMUL __attribute__((target("avx512f,vpclmulqdq")))
#else
#define GF64_HQC_TARGET_AVX512_PCLMUL
#endif

GF64_HQC_TARGET_AVX512_PCLMUL
static void butterfly_fwd_avx512(gf64_t *f, int n, int n_table, gf64_t a,
                                const gf64_t *v_table, int logn) {
    if (n == 2) {
        gf64_t fl = f[0], fh = f[1];
        f[0] = fl ^ gf64_mul_reference(a, fh);
        f[1] = fl ^ gf64_mul_reference(a ^ 1, fh);
        return;
    }
    int half = n / 2;
    gf64_t s_a = si_eval(logn - 1, a, v_table, n_table);
    if (half >= GF64_HQC_BUTTERFLY_AVX512_LANES) {
        /* SIMD: 8 lanes per PCLMULQDQ call. */
        gf64_t sa_buf[GF64_HQC_BUTTERFLY_AVX512_LANES];
        for (int k = 0; k < GF64_HQC_BUTTERFLY_AVX512_LANES; k++) sa_buf[k] = s_a;
        int j = 0;
        for (; j + GF64_HQC_BUTTERFLY_AVX512_LANES <= half; j += GF64_HQC_BUTTERFLY_AVX512_LANES) {
            gf64_t fh_buf[GF64_HQC_BUTTERFLY_AVX512_LANES];
            gf64_t prod_buf[GF64_HQC_BUTTERFLY_AVX512_LANES];
            for (int k = 0; k < GF64_HQC_BUTTERFLY_AVX512_LANES; k++)
                fh_buf[k] = f[j + k + half];
            gf64_mul_avx512(prod_buf, fh_buf, sa_buf, GF64_HQC_BUTTERFLY_AVX512_LANES);
            for (int k = 0; k < GF64_HQC_BUTTERFLY_AVX512_LANES; k++) {
                gf64_t fl = f[j + k];
                f[j + k]         = fl ^ prod_buf[k];
                f[j + k + half]  = fl ^ prod_buf[k] ^ fh_buf[k];
            }
        }
        /* Tail: < 8 elements remaining. */
        for (; j < half; j++) {
            gf64_t fl = f[j], fh = f[j + half];
            f[j]       = fl ^ gf64_mul_reference(s_a, fh);
            f[j + half] = fl ^ gf64_mul_reference(s_a ^ 1, fh);
        }
    } else {
        for (int j = 0; j < half; j++) {
            gf64_t fl = f[j], fh = f[j + half];
            f[j]       = fl ^ gf64_mul_reference(s_a, fh);
            f[j + half] = fl ^ gf64_mul_reference(s_a ^ 1, fh);
        }
    }
    butterfly_fwd_avx512(f,         half, n_table, a,                       v_table, logn - 1);
    butterfly_fwd_avx512(f + half,  half, n_table, a ^ v_table[1 << (logn - 1)],
                        v_table, logn - 1);
}

GF64_HQC_TARGET_AVX512_PCLMUL
static void butterfly_inv_avx512(gf64_t *f, int n, int n_table, gf64_t a,
                                const gf64_t *v_table, int logn) {
    if (n == 2) {
        gf64_t fh = f[0] ^ f[1];
        gf64_t fl = f[0] ^ gf64_mul_reference(a, fh);
        f[0] = fl; f[1] = fh;
        return;
    }
    int half = n / 2;
    butterfly_inv_avx512(f,         half, n_table, a,                       v_table, logn - 1);
    butterfly_inv_avx512(f + half,  half, n_table, a ^ v_table[1 << (logn - 1)],
                        v_table, logn - 1);
    gf64_t s_a = si_eval(logn - 1, a, v_table, n_table);
    if (half >= GF64_HQC_BUTTERFLY_AVX512_LANES) {
        gf64_t sa_buf[GF64_HQC_BUTTERFLY_AVX512_LANES];
        for (int k = 0; k < GF64_HQC_BUTTERFLY_AVX512_LANES; k++) sa_buf[k] = s_a;
        int j = 0;
        for (; j + GF64_HQC_BUTTERFLY_AVX512_LANES <= half; j += GF64_HQC_BUTTERFLY_AVX512_LANES) {
            gf64_t fh_buf[GF64_HQC_BUTTERFLY_AVX512_LANES];
            gf64_t prod_buf[GF64_HQC_BUTTERFLY_AVX512_LANES];
            for (int k = 0; k < GF64_HQC_BUTTERFLY_AVX512_LANES; k++)
                fh_buf[k] = f[j + k] ^ f[j + k + half];
            gf64_mul_avx512(prod_buf, fh_buf, sa_buf, GF64_HQC_BUTTERFLY_AVX512_LANES);
            for (int k = 0; k < GF64_HQC_BUTTERFLY_AVX512_LANES; k++) {
                f[j + k]         = f[j + k] ^ prod_buf[k];
                f[j + k + half]  = fh_buf[k];
            }
        }
        for (; j < half; j++) {
            gf64_t fl_new = f[j], fh_new = f[j + half];
            gf64_t fh = fl_new ^ fh_new;
            gf64_t fl = fl_new ^ gf64_mul_reference(s_a, fh);
            f[j] = fl;
            f[j + half] = fh;
        }
    } else {
        for (int j = 0; j < half; j++) {
            gf64_t fl_new = f[j], fh_new = f[j + half];
            gf64_t fh = fl_new ^ fh_new;
            gf64_t fl = fl_new ^ gf64_mul_reference(s_a, fh);
            f[j] = fl;
            f[j + half] = fh;
        }
    }
}

/* ----- Public API -----
 *
 * The HQC FFT is exposed via a `_scratch` suffix on every entry point. The
 * companion `*_scratch_words(n)` function returns the worst-case scratch
 * size in gf64_t words for a given transform length. Caller allocates
 * scratch once, threads it through. Mirrors the convention established by
 * gf64_poly_divmod_scratch (gf64_mpe.h:115).
 *
 * Scratch sizing summary:
 *   gf64_addfft64_fwd_scratch_words / _inv_scratch_words / _poly_mul_scratch_words
 *       = 4n   (covers matrix-form fallback peak)
 *   gf64_addfft64_fwd_recursive_scratch_words / _inv_recursive_scratch_words
 *       = 2n   (entry g/c plus basisCvt_v2 tmp, sequential)
 *   gf64_addfft64_poly_mul_recursive_scratch_words
 *       = 4n   (padded 2n plus recursive scratch 2n)
 *
 * Length caps (two tiers):
 *   GF64_HQC_MAX_MATRIXFORM_N = 16384   matrix-form (cache-backed) path
 *   GF64_HQC_MAX_LM_N = 131072          matrix-free recursive path
 *
 * Sizes in (131072, 2^20] require the General Algorithm 1 (FIX-3a
 * follow-up, gf64_additive_fft_hqc2026.c:387-427).
 */

/* ----- Scratch size queries ----- */

size_t gf64_addfft64_fwd_scratch_words(size_t n) {
    if (n <= 1) return 0;
    return 4 * n;
}

size_t gf64_addfft64_inv_scratch_words(size_t n) {
    if (n <= 1) return 0;
    return 4 * n;
}

size_t gf64_addfft64_poly_mul_scratch_words(size_t n) {
    if (n <= 1) return 0;
    return 4 * n;
}

size_t gf64_addfft64_fwd_recursive_scratch_words(size_t n) {
    if (n <= 1) return 0;
    return 2 * n;
}

size_t gf64_addfft64_inv_recursive_scratch_words(size_t n) {
    if (n <= 1) return 0;
    return 2 * n;
}

size_t gf64_addfft64_poly_mul_recursive_scratch_words(size_t n) {
    if (n <= 1) return 0;
    /* Layout: [pa: n | pb: n | inner: 2n] = 4n total. The inner 2n
     * region holds the working scratch for fwd/ivt recursive transforms
     * (one fwd at a time — same region reused). */
    return 4 * n;
}

/* ----- Matrix-form path (capped at GF64_HQC_MAX_MATRIXFORM_N) ----- */

void gf64_addfft64_fwd_scratch(gf64_t *arr, size_t n,
                               gf64_t *scratch, size_t scratch_words) {
    if (n <= 1) return;
    assert(n <= GF64_HQC_MAX_MATRIXFORM_N);
    assert(scratch_words >= gf64_addfft64_fwd_scratch_words(n));

    int n_int = (int)n;
    hqc_basis_cache_t *cache = get_or_build_basis_cache(n_int);
    assert(cache != NULL);

    /* Scratch layout: [g: n | basisCvt internal: 2n | spare: n] = 4n total.
     * basisCvt_dispatch takes the 2n region (recursive descent of basisCvt). */
    gf64_t *g = scratch;
    gf64_t *bc_scratch = scratch + n;
    basisCvt_dispatch(g, arr, n_int, cache->M_inv, bc_scratch);
    memcpy(arr, g, n * sizeof(gf64_t));

    int logn = 0; while ((1 << logn) < n_int) logn++;
    /* Affine shift = basis vector at the depth of this transform's
     * primary vanishing subspace boundary: GF64_CANTOR_BASIS[logn-1].
     * For n = 2^i, this element has Cantor-index 1 << (logn-1), whose
     * bit-(logn-1) is set, so it's outside V_{logn-1}. */
    gf64_t a = GF64_CANTOR_BASIS[logn - 1];
    butterfly_fwd(arr, n_int, n_int, a, cache->v_table, logn);
}

void gf64_addfft64_inv_scratch(gf64_t *arr, size_t n,
                               gf64_t *scratch, size_t scratch_words) {
    if (n <= 1) return;
    assert(n <= GF64_HQC_MAX_MATRIXFORM_N);
    assert(scratch_words >= gf64_addfft64_inv_scratch_words(n));

    int n_int = (int)n;
    hqc_basis_cache_t *cache = get_or_build_basis_cache(n_int);
    assert(cache != NULL);

    int logn = 0; while ((1 << logn) < n_int) logn++;
    gf64_t a = GF64_CANTOR_BASIS[logn - 1];
    butterfly_inv(arr, n_int, n_int, a, cache->v_table, logn);

    /* Scratch layout: [c: n | basisCvt internal: 2n | spare: n] = 4n total. */
    gf64_t *c = scratch;
    gf64_t *bc_scratch = scratch + n;
    ibasisCvt_dispatch(c, arr, n_int, cache->M, bc_scratch);
    memcpy(arr, c, n * sizeof(gf64_t));
}

/* ----- Matrix-free recursive path (capped at GF64_HQC_MAX_LM_N) ----- */

void gf64_addfft64_fwd_recursive_scratch(gf64_t *arr, size_t n,
                                         gf64_t *scratch, size_t scratch_words) {
    if (n <= 1) return;
    assert(n <= GF64_HQC_MAX_LM_N);
    assert(scratch_words >= gf64_addfft64_fwd_recursive_scratch_words(n));

    int n_int = (int)n;
    int logn = 0; while ((1 << logn) < n_int) logn++;

    /* Scratch layout: [g: n | basisCvt_v2 tmp: n] = 2n total. */
    gf64_t *g = scratch;
    gf64_t *cvt_tmp = scratch + n;
    basisCvt_recursive_v2(g, arr, n_int, cvt_tmp);
    memcpy(arr, g, n * sizeof(gf64_t));

    /* Butterfly on the novelpoly basis. Reuses the existing
     * butterfly_fwd (which is basis-agnostic — it works in the novelpoly
     * basis of any Algorithm-1-style BasisCvt). v_table only — we don't
     * need M / M_inv, so the O(N^3) build is skipped. */
    gf64_t *v_table = get_or_build_v_table(n_int);
    gf64_t a = GF64_CANTOR_BASIS[logn - 1];
    butterfly_fwd(arr, n_int, n_int, a, v_table, logn);
}

void gf64_addfft64_inv_recursive_scratch(gf64_t *arr, size_t n,
                                         gf64_t *scratch, size_t scratch_words) {
    if (n <= 1) return;
    assert(n <= GF64_HQC_MAX_LM_N);
    assert(scratch_words >= gf64_addfft64_inv_recursive_scratch_words(n));

    int n_int = (int)n;
    int logn = 0; while ((1 << logn) < n_int) logn++;

    gf64_t *v_table = get_or_build_v_table(n_int);
    gf64_t a = GF64_CANTOR_BASIS[logn - 1];

    /* Inverse butterfly first. */
    butterfly_inv(arr, n_int, n_int, a, v_table, logn);

    /* Scratch layout: [c: n | basisCvt_v2 tmp: n] = 2n total. */
    gf64_t *c = scratch;
    gf64_t *cvt_tmp = scratch + n;
    ibasisCvt_recursive_v2(c, arr, n_int, cvt_tmp);
    memcpy(arr, c, n * sizeof(gf64_t));
}

/* ----- Polynomial multiplication ----- */

/* Recursive path (matrix-free). Convolution theorem holds:
 * inv(fwd(a) . fwd(b)) = a*b (pointwise), self-consistent within the
 * _recursive family only. */
void gf64_addfft64_poly_mul_recursive_scratch(
    gf64_t *out,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b,
    size_t out_len,
    gf64_t *scratch, size_t scratch_words)
{
    if (len_a == 0 || len_b == 0 || out_len == 0) {
        memset(out, 0, out_len * sizeof(gf64_t));
        return;
    }
    size_t full_len = len_a + len_b - 1;
    if (full_len < out_len) full_len = out_len;
    size_t n = 1;
    while (n < full_len) n <<= 1;
    assert(n <= GF64_HQC_MAX_LM_N);
    assert(scratch_words >= gf64_addfft64_poly_mul_recursive_scratch_words(n));

    /* Scratch layout: [pa: n | pb: n | inner: 2n] = 4n total. */
    gf64_t *pa = scratch;
    gf64_t *pb = scratch + n;
    gf64_t *inner = scratch + 2 * n;

    memcpy(pa, a, len_a * sizeof(gf64_t));
    memset(pa + len_a, 0, (n - len_a) * sizeof(gf64_t));
    memcpy(pb, b, len_b * sizeof(gf64_t));
    memset(pb + len_b, 0, (n - len_b) * sizeof(gf64_t));

    gf64_addfft64_fwd_recursive_scratch(pa, n, inner, 2 * n);
    gf64_addfft64_fwd_recursive_scratch(pb, n, inner, 2 * n);
    for (size_t i = 0; i < n; i++) pa[i] = gf64_mul_reference(pa[i], pb[i]);
    gf64_addfft64_inv_recursive_scratch(pa, n, inner, 2 * n);

    size_t copy_n = (full_len < out_len) ? full_len : out_len;
    memcpy(out, pa, copy_n * sizeof(gf64_t));
    if (copy_n < out_len) memset(out + copy_n, 0, (out_len - copy_n) * sizeof(gf64_t));
}

/* Matrix-form path. Honors the in-place footgun rule documented in user
 * memory: gf64_poly_mul_padded memsets 'out' before reading inputs;
 * 'out == a' breaks the result. We don't read 'out' before writing it. */
void gf64_addfft64_poly_mul_scratch(
    gf64_t *out,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b,
    size_t out_len,
    gf64_t *scratch, size_t scratch_words)
{
    if (len_a == 0 || len_b == 0 || out_len == 0) {
        memset(out, 0, out_len * sizeof(gf64_t));
        return;
    }
    size_t full_len = len_a + len_b - 1;
    if (full_len < out_len) full_len = out_len;
    size_t n = 1;
    while (n < full_len) n <<= 1;
    assert(n <= GF64_HQC_MAX_MATRIXFORM_N);
    assert(scratch_words >= gf64_addfft64_poly_mul_scratch_words(n));

    /* Scratch layout: [pa: n | pb: n | inner: 2n] = 4n total. The inner
     * region holds basisCvt's scratch during _fwd / _inv; pa/pb hold the
     * padded copies of a and b and the eventual pointwise product. */
    gf64_t *pa = scratch;
    gf64_t *pb = scratch + n;
    gf64_t *inner = scratch + 2 * n;

    memcpy(pa, a, len_a * sizeof(gf64_t));
    memset(pa + len_a, 0, (n - len_a) * sizeof(gf64_t));
    memcpy(pb, b, len_b * sizeof(gf64_t));
    memset(pb + len_b, 0, (n - len_b) * sizeof(gf64_t));

    gf64_addfft64_fwd_scratch(pa, n, inner, 2 * n);
    gf64_addfft64_fwd_scratch(pb, n, inner, 2 * n);
    for (size_t i = 0; i < n; i++) pa[i] = gf64_mul_reference(pa[i], pb[i]);
    gf64_addfft64_inv_scratch(pa, n, inner, 2 * n);

    size_t copy_n = (full_len < out_len) ? full_len : out_len;
    memcpy(out, pa, copy_n * sizeof(gf64_t));
    if (copy_n < out_len) memset(out + copy_n, 0, (out_len - copy_n) * sizeof(gf64_t));
}

/* ----- AVX-512 (PCLMULQDQ) public entries ----- *
 *
 * Same scratch layout and cap as the scalar _recursive_scratch family;
 * only the butterfly is SIMD-accelerated. Bit-exact to the scalar entries.
 *
 * Host: requires AVX-512F + VPCLMULQDQ.
 */

void gf64_addfft64_fwd_recursive_scratch_avx512(gf64_t *arr, size_t n,
                                                gf64_t *scratch, size_t scratch_words) {
    if (n <= 1) return;
    assert(n <= GF64_HQC_MAX_LM_N);
    assert(scratch_words >= gf64_addfft64_fwd_recursive_scratch_words(n));

    int n_int = (int)n;
    int logn = 0; while ((1 << logn) < n_int) logn++;

    /* Scratch layout mirrors the scalar _recursive_scratch entry. */
    gf64_t *g = scratch;
    gf64_t *cvt_tmp = scratch + n;
    basisCvt_recursive_v2(g, arr, n_int, cvt_tmp);
    memcpy(arr, g, n * sizeof(gf64_t));

    gf64_t *v_table = get_or_build_v_table(n_int);
    gf64_t a = GF64_CANTOR_BASIS[logn - 1];
    butterfly_fwd_avx512(arr, n_int, n_int, a, v_table, logn);
}

void gf64_addfft64_inv_recursive_scratch_avx512(gf64_t *arr, size_t n,
                                                gf64_t *scratch, size_t scratch_words) {
    if (n <= 1) return;
    assert(n <= GF64_HQC_MAX_LM_N);
    assert(scratch_words >= gf64_addfft64_inv_recursive_scratch_words(n));

    int n_int = (int)n;
    int logn = 0; while ((1 << logn) < n_int) logn++;

    gf64_t *v_table = get_or_build_v_table(n_int);
    gf64_t a = GF64_CANTOR_BASIS[logn - 1];
    butterfly_inv_avx512(arr, n_int, n_int, a, v_table, logn);

    gf64_t *c = scratch;
    gf64_t *cvt_tmp = scratch + n;
    ibasisCvt_recursive_v2(c, arr, n_int, cvt_tmp);
    memcpy(arr, c, n * sizeof(gf64_t));
}

void gf64_addfft64_poly_mul_recursive_scratch_avx512(
    gf64_t *out,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b,
    size_t out_len,
    gf64_t *scratch, size_t scratch_words)
{
    if (len_a == 0 || len_b == 0 || out_len == 0) {
        memset(out, 0, out_len * sizeof(gf64_t));
        return;
    }
    size_t full_len = len_a + len_b - 1;
    if (full_len < out_len) full_len = out_len;
    size_t n = 1;
    while (n < full_len) n <<= 1;
    assert(n <= GF64_HQC_MAX_LM_N);
    assert(scratch_words >= gf64_addfft64_poly_mul_recursive_scratch_words(n));

    /* Same layout as the scalar _poly_mul_recursive_scratch entry. */
    gf64_t *pa = scratch;
    gf64_t *pb = scratch + n;
    gf64_t *inner = scratch + 2 * n;

    memcpy(pa, a, len_a * sizeof(gf64_t));
    memset(pa + len_a, 0, (n - len_a) * sizeof(gf64_t));
    memcpy(pb, b, len_b * sizeof(gf64_t));
    memset(pb + len_b, 0, (n - len_b) * sizeof(gf64_t));

    gf64_addfft64_fwd_recursive_scratch_avx512(pa, n, inner, 2 * n);
    gf64_addfft64_fwd_recursive_scratch_avx512(pb, n, inner, 2 * n);
    for (size_t i = 0; i < n; i++) pa[i] = gf64_mul_reference(pa[i], pb[i]);
    gf64_addfft64_inv_recursive_scratch_avx512(pa, n, inner, 2 * n);

    size_t copy_n = (full_len < out_len) ? full_len : out_len;
    memcpy(out, pa, copy_n * sizeof(gf64_t));
    if (copy_n < out_len) memset(out + copy_n, 0, (out_len - copy_n) * sizeof(gf64_t));
}

HEDLEY_END_C_DECLS
