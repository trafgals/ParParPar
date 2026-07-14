/*
 * gf64/gf64_poly_mul_toom3.c — Toom-Cook 3 (Toom 2.3) over GF(2^64)[x].
 *
 * Algorithm
 * ---------
 * Split A and B into 3 limbs each (size k = ceil(n/3)):
 *   A(x) = A0(x) + x^k * A1(x) + x^(2k) * A2(x)
 *   B(x) = B0(x) + x^k * B1(x) + x^(2k) * B2(x)
 * where Ai(x), Bi(x) are polynomials of degree < k.
 *
 * For 5 evaluation "points" s in {0, 1, 2, 3, infinity}, define the
 * POLYNOMIALS A_s and B_s (each of degree < k):
 *   A_0  = A0
 *   A_1  = A0 + A1 + A2
 *   A_2  = A0 + 2*A1 + 4*A2      (with 2, 4 as GF(2^64) elements)
 *   A_3  = A0 + 3*A1 + 9*A2
 *   A_inf = A2                   (leading limb; s^2 dominates as s -> inf)
 *
 * Recursively compute the 5 POLYNOMIAL sub-products C_s = A_s * B_s, each
 * of degree < 2k.
 *
 * Recover the 5 limb-polynomials T_0..T_4 of C(x) by interpolating the
 * 5 C_s polynomials using the precomputed Vandermonde inverse V^{-1}:
 *   T_p(x) = sum_q V^{-1}[p][q] * C_{s_q}(x)
 *
 * Assemble: C(x) = T_0(x) + x^k * T_1(x) + ... + x^(4k) * T_4(x).
 *
 * Complexity: T(n) = 5 T(n/3) + O(n) = O(n^log_3(5)) ~ O(n^1.465),
 * strictly better than Karatsuba's O(n^1.585).
 *
 * Char-2 safety: standard Toom-3 uses {0,1,-1,-2,inf}; in char 2 this
 * collapses because -1 = 1, -2 = 2. We use {0,1,2,3,inf} instead, all
 * distinct elements of GF(2^64). Interpolation constants computed by
 * gf64/test/gen_toom3_constants.c and verified V*V^{-1} = I.
 *
 * Memory management (issue #27 §auxiliary):
 *   Naive Toom-3 issues ~13 malloc/calloc calls per recursion level
 *   (6 limb buffers + 5 sub-product buffers + 2 build buffers + frees).
 *   For deep recursion this dominates the runtime (per-call allocator
 *   overhead as the perf regression called out in §Phase 2b). The
 *   thread_local arena below replaces these with a single contiguous
 *   buffer per thread that grows lazily and resets at top-level call
 *   entry, so each recursion "borrows" regions from the arena and
 *   releases them implicitly when its parent resets the offset.
 *
 *   The arena stores bytes (gf64_t-aligned), so A0, A1, A2, B0, B1, B2
 *   (each of size k), and the 5 C_s buffers (each of size 2k) all come
 *   from the same bump-allocator. As_buf and Bs_buf (size k each) come
 *   from the same arena. Total memory per call at depth d is bounded by
 *   the SUM over depths of (3k + 3k + 5·2k + k + k) bytes = ~26k bytes
 *   per level — the arena grows to support the deepest pending level,
 *   not the total. After the call returns the threads' arena persists
 *   for the next top-level call (allocator-friendly: same base
 *   ptr across calls, same growth pattern).
 */

#include "gf64_poly_mul_toom3.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

HEDLEY_BEGIN_C_DECLS

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);

/* Forward-declare Karatsuba as a fallback for Toom-3's base case so the
 * 5 sub-products of size <= TOOM3_KARATSUBA_FALLBACK use the O(n^1.585)
 * path instead of schoolbook. */
void gf64_poly_mul_karatsuba(
    gf64_t *out,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b,
    size_t out_len
);

/*
 * Vandermonde inverse for points {0, 1, 2, 3, inf} over GF(2^64).
 * Computed by gf64/test/gen_toom3_constants.c and verified bit-exact.
 * T_p = sum_q V_inv[p][q] * C_{s_q}.
 */
static const gf64_t GF64_TOOM3_VINV[5][5] = {
    { 0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL },
    { 0x7ffffffffffffffaULL, 0x0000000000000001ULL, 0x800000000000000dULL, 0xfffffffffffffff6ULL, 0x0000000000000006ULL },
    { 0x0000000000000000ULL, 0x7ffffffffffffffbULL, 0xfffffffffffffff6ULL, 0x800000000000000dULL, 0x0000000000000007ULL },
    { 0x7ffffffffffffffbULL, 0x7ffffffffffffffbULL, 0x7ffffffffffffffbULL, 0x7ffffffffffffffbULL, 0x0000000000000000ULL },
    { 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000001ULL }
};

/* ---- thread_local scratch arena (issue #27 §auxiliary) ---- *
 *
 * The arena is per-thread (each worker_thread and the main thread have
 * their own). It grows lazily on first use and is reused across calls:
 * the toplevel entry resets `top` to 0, and recursive sub-calls bump
 * `top` by their working-set sizes. After the toplevel returns, the
 * arena's memory stays allocated for the next call.
 *
 * If a single call's total scratch exceeds the arena's `capacity`, the
 * arena reallocates (capacity doubles until it fits; abort() on
 * failure to match the existing malloc-failure convention).
 *
 * On thread exit the OS reclaims the thread_local — no explicit free.
 */
typedef struct {
    unsigned char *base;
    size_t capacity;
    size_t top;
} toom3_arena_t;

#if defined(__GNUC__) || defined(__clang__)
static __thread toom3_arena_t toom3_arena = { NULL, 0, 0 };
#elif defined(_MSC_VER)
static __declspec(thread) toom3_arena_t toom3_arena = { NULL, 0, 0 };
#else
static toom3_arena_t toom3_arena = { NULL, 0, 0 };
#endif

/* Allocate n_coeffs gf64_t from the arena; pointer is valid until the
 * next call that resets the arena's `top`. Coeffs are NOT zeroed; the
 * caller is responsible for zero-padding where needed. */
static gf64_t *toom3_arena_alloc(size_t n_coeffs) {
    /* Round the current top up to gf64_t alignment. */
    size_t bytes = n_coeffs * sizeof(gf64_t);
    size_t aligned_top = (toom3_arena.top + sizeof(gf64_t) - 1) & ~(sizeof(gf64_t) - 1);
    if (aligned_top + bytes > toom3_arena.capacity) {
        size_t new_capacity = (toom3_arena.capacity == 0)
                                  ? (bytes > 65536 ? bytes : 65536)
                                  : toom3_arena.capacity;
        while (new_capacity < aligned_top + bytes) {
            if (new_capacity > (SIZE_MAX / 2)) { new_capacity = aligned_top + bytes; break; }
            new_capacity *= 2;
        }
        unsigned char *new_base = (unsigned char *)realloc(toom3_arena.base, new_capacity);
        if (new_base == NULL) abort();
        toom3_arena.base = new_base;
        toom3_arena.capacity = new_capacity;
    }
    gf64_t *p = (gf64_t *)(toom3_arena.base + aligned_top);
    toom3_arena.top = aligned_top + bytes;
    return p;
}

#define ARENA_RESET_DEPTH 128
static size_t toom3_arena_depths[ARENA_RESET_DEPTH] = {0};
static int toom3_arena_depth_count = 0;

static void toom3_arena_push(void) {
    if (toom3_arena_depth_count < ARENA_RESET_DEPTH) {
        toom3_arena_depths[toom3_arena_depth_count++] = toom3_arena.top;
    }
}
static void toom3_arena_pop(void) {
    if (toom3_arena_depth_count > 0) {
        toom3_arena.top = toom3_arena_depths[--toom3_arena_depth_count];
    }
}

/* Schoolbook fallback (unused — Karatsuba is now the base case via
 * gf64_poly_mul_karatsuba). Kept here in case future tunings want a
 * schoolbook base case directly. */
#if 0
static void toom3_schoolbook(
    gf64_t *out,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b,
    size_t out_len
) {
    memset(out, 0, out_len * sizeof(gf64_t));
    size_t a_cap = (len_a < out_len) ? len_a : out_len;
    size_t b_cap = (len_b < out_len) ? len_b : out_len;
    for (size_t i = 0; i < a_cap; i++) {
        gf64_t ai = a[i];
        if (ai == 0) continue;
        size_t j_max = (b_cap < out_len - i) ? b_cap : (out_len - i);
        for (size_t j = 0; j < j_max; j++) {
            out[i + j] ^= gf64_mul_reference(ai, b[j]);
        }
    }
}
#endif

static void toom3_recursive(
    gf64_t *out, size_t out_len,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b
);

void gf64_poly_mul_toom3(
    gf64_t *out,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b,
    size_t out_len
) {
    if (out_len == 0) return;

    /* Truncate inputs at out_len. */
    if (len_a > out_len) len_a = out_len;
    if (len_b > out_len) len_b = out_len;

    if (len_a == 0 || len_b == 0) {
        memset(out, 0, out_len * sizeof(gf64_t));
        return;
    }

    /* Reset arena depth tracking at the top-level call. The arena grows
     * as needed during the recursion and reclaims space when this entry
     * returns. Subsequent top-level calls reuse the same backing memory
     * without releasing to the OS — the OS reclaims on thread exit. */
    toom3_arena_depth_count = 0;
    toom3_arena_push();  /* mark the entry point */

    toom3_recursive(out, out_len, a, len_a, b, len_b);

    toom3_arena_pop();
    /* After popping, top points to before the recursive call; the next
     * top-level call pushes again and resets the depth counter. The
     * arena's base/ capacity are retained for the next call. */
}

/*
 * Build the polynomial A_s in As_buf (size k):
 *   A_s = A0 + s*A1 + s^2*A2
 * c0, c1, c2 are GF(2^64) scalar coefficients; result is per-limb
 * linear combination.
 */
static void toom3_build_polynomial(
    gf64_t *As_buf, size_t k,
    const gf64_t *A0, const gf64_t *A1, const gf64_t *A2,
    gf64_t c0, gf64_t c1, gf64_t c2
) {
    for (size_t i = 0; i < k; i++) {
        As_buf[i] = A0[i] ^ gf64_mul_reference(c1, A1[i]) ^ gf64_mul_reference(c2, A2[i]);
    }
    (void)c0; /* c0 is currently always 1; kept in API for future use */
}

static void toom3_recursive(
    gf64_t *out, size_t out_len,
    const gf64_t *a, size_t len_a,
    const gf64_t *b, size_t len_b
) {
    /* Base case: below threshold, use Karatsuba (which itself falls back
     * to schoolbook at 64). Karatsuba's lower overhead wins below this
     * size on microbench. */
    const size_t TOOM3_BASE = 768;
    size_t n_raw = (len_a > len_b) ? len_a : len_b;
    if (n_raw <= TOOM3_BASE) {
        gf64_poly_mul_karatsuba(out, a, len_a, b, len_b, out_len);
        return;
    }

    size_t k = (n_raw + 2) / 3;

    /* Allocate zero-padded k-sized limb buffers from the arena. */
    gf64_t *A0 = toom3_arena_alloc(k); memset(A0, 0, k * sizeof(gf64_t));
    gf64_t *A1 = toom3_arena_alloc(k); memset(A1, 0, k * sizeof(gf64_t));
    gf64_t *A2 = toom3_arena_alloc(k); memset(A2, 0, k * sizeof(gf64_t));
    gf64_t *B0 = toom3_arena_alloc(k); memset(B0, 0, k * sizeof(gf64_t));
    gf64_t *B1 = toom3_arena_alloc(k); memset(B1, 0, k * sizeof(gf64_t));
    gf64_t *B2 = toom3_arena_alloc(k); memset(B2, 0, k * sizeof(gf64_t));

    size_t copy_a0 = (k < len_a) ? k : len_a;
    size_t copy_a1 = (len_a > k) ? ((len_a - k < k) ? len_a - k : k) : 0;
    size_t copy_a2 = (len_a > 2 * k) ? len_a - 2 * k : 0;
    size_t copy_b0 = (k < len_b) ? k : len_b;
    size_t copy_b1 = (len_b > k) ? ((len_b - k < k) ? len_b - k : k) : 0;
    size_t copy_b2 = (len_b > 2 * k) ? len_b - 2 * k : 0;

    if (copy_a0) memcpy(A0, a, copy_a0 * sizeof(gf64_t));
    if (copy_a1) memcpy(A1, a + k, copy_a1 * sizeof(gf64_t));
    if (copy_a2) memcpy(A2, a + 2 * k, copy_a2 * sizeof(gf64_t));
    if (copy_b0) memcpy(B0, b, copy_b0 * sizeof(gf64_t));
    if (copy_b1) memcpy(B1, b + k, copy_b1 * sizeof(gf64_t));
    if (copy_b2) memcpy(B2, b + 2 * k, copy_b2 * sizeof(gf64_t));

    /* Mark the recursion entry: capture the arena top for this level.
     * Sub-recursion borrows below this; this level's allocations are
     * reclaimed when we pop. */
    toom3_arena_push();

    /* Allocate 5 sub-product buffers (polynomials of degree < 2k each)
     * plus 2 build buffers. */
    gf64_t *C0    = toom3_arena_alloc(2 * k); memset(C0,    0, 2 * k * sizeof(gf64_t));
    gf64_t *C1    = toom3_arena_alloc(2 * k); memset(C1,    0, 2 * k * sizeof(gf64_t));
    gf64_t *C2    = toom3_arena_alloc(2 * k); memset(C2,    0, 2 * k * sizeof(gf64_t));
    gf64_t *C3    = toom3_arena_alloc(2 * k); memset(C3,    0, 2 * k * sizeof(gf64_t));
    gf64_t *Cinf  = toom3_arena_alloc(2 * k); memset(Cinf,  0, 2 * k * sizeof(gf64_t));
    gf64_t *As_buf = toom3_arena_alloc(k);     /* not zeroed — overwritten below */
    gf64_t *Bs_buf = toom3_arena_alloc(k);     /* not zeroed — overwritten below */

    gf64_t one = 1;
    gf64_t two = 2;
    gf64_t three = 3;
    gf64_t four = gf64_mul_reference(two, two);
    gf64_t nine = gf64_mul_reference(three, three);

    /* s = 0: A_0 = A0, B_0 = B0. */
    toom3_recursive(C0, 2 * k, A0, k, B0, k);

    /* s = 1: A_1 = A0 + A1 + A2. */
    toom3_build_polynomial(As_buf, k, A0, A1, A2, one, one, one);
    toom3_build_polynomial(Bs_buf, k, B0, B1, B2, one, one, one);
    toom3_recursive(C1, 2 * k, As_buf, k, Bs_buf, k);

    /* s = 2: A_2 = A0 + 2*A1 + 4*A2. */
    toom3_build_polynomial(As_buf, k, A0, A1, A2, one, two, four);
    toom3_build_polynomial(Bs_buf, k, B0, B1, B2, one, two, four);
    toom3_recursive(C2, 2 * k, As_buf, k, Bs_buf, k);

    /* s = 3: A_3 = A0 + 3*A1 + 9*A2. */
    toom3_build_polynomial(As_buf, k, A0, A1, A2, one, three, nine);
    toom3_build_polynomial(Bs_buf, k, B0, B1, B2, one, three, nine);
    toom3_recursive(C3, 2 * k, As_buf, k, Bs_buf, k);

    /* s = inf: A_inf = A2 (leading limb). */
    toom3_recursive(Cinf, 2 * k, A2, k, B2, k);

    /* Interpolate the 5 polynomial sub-products via V^{-1} to recover the
     * 5 limb polynomials T_0..T_4 of C(x), then assemble. */
    gf64_t *Cs[5] = { C0, C1, C2, C3, Cinf };
    memset(out, 0, out_len * sizeof(gf64_t));

    for (int i = 0; i < 5; i++) {
        size_t pos_base = i * k;
        if (pos_base >= out_len) break;
        size_t avail = out_len - pos_base;
        size_t write_count = (2 * k <= avail) ? 2 * k : avail;
        for (size_t ip = 0; ip < write_count; ip++) {
            gf64_t acc = 0;
            for (int j = 0; j < 5; j++) {
                acc ^= gf64_mul_reference(GF64_TOOM3_VINV[i][j], Cs[j][ip]);
            }
            out[pos_base + ip] ^= acc;
        }
    }

    toom3_arena_pop();
}

HEDLEY_END_C_DECLS