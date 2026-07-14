/*
 * ============================================================================
 * gf64/gf64_subproduct.h — Bottom-up FFT-based subproduct tree over GF(2^64).
 *
 * T6 of the par3-cauchy-fft-kernel plan.
 *
 * A subproduct tree over N input points {x_0, ..., x_{N-1}} is a complete
 * binary tree of polynomials:
 *
 *   - leaves (level log2(N)):   M_i(x) = x + x_i   (degree 1)
 *   - internal nodes:           product of the two child polynomials
 *   - root    (level 0):        P(x)   = ∏_i (x + x_i)   (degree N)
 *
 * Each polynomial's coefficients are stored constant-first:
 *   c[0] is the constant term, c[k] is the coefficient of x^k.
 *
 * Tree storage layout (BFS order):
 *   - level_data[ℓ]   points to the first coefficient of the ℓ-th level
 *   - level_lens[ℓ]   is the number of polynomials at level ℓ (= N >> ℓ)
 *   - level_degs[ℓ]   is the degree of each polynomial at level ℓ (= 1 << ℓ
 *                     wait — actually roots have max degree, leaves degree 1;
 *                     see level-math below)
 *
 * Level indexing:
 *   - Level 0:             1 polynomial (root),  degree = N
 *   - Level ℓ:             2^ℓ polynomials,       degree = N / 2^ℓ
 *   - Level log2(N):       N leaf polynomials,    degree = 1
 *
 * N must be 0 (returns an empty tree), 1, or a power of 2.
 * ============================================================================
 */

#ifndef GF64_SUBPRODUCT_H
#define GF64_SUBPRODUCT_H

#include "gf64_global.h"

#include <stddef.h>

HEDLEY_BEGIN_C_DECLS

typedef struct {
	gf64_t **level_data;  /* level_data[ℓ] -> start of level ℓ's polynomials    */
	size_t   *level_lens; /* level_lens[ℓ] = #polynomials at level ℓ            */
	size_t   *level_degs; /* level_degs[ℓ] = degree of each poly at level ℓ     */
	size_t    num_levels; /* log2(N) + 1                                       */
	size_t    num_points; /* N                                                 */
	gf64_t   *storage;    /* backing buffer that level_data points into         */

	/*
	 * Cached polynomial inverses for the multi-point INTERPOLATION
	 * direction (T8b of par3-cauchy-fft-kernel, issue #27).
	 *
	 * For each internal node (lev, node_idx) with lev < num_levels - 1:
	 *   P_left  = level_data[lev+1] + (2*node_idx)     * (deg_child+1)
	 *   P_right = level_data[lev+1] + (2*node_idx + 1) * (deg_child+1)
	 *   inv_mod_data[lev][node_idx * (deg_child + 1) .. ]
	 *           = P_left^(-1) mod P_right  (degree < deg(P_right))
	 *
	 * inv_mod_data[lev] mirrors level_data[lev]'s BFS indexing for the
	 * internal-node levels. For lev == num_levels - 1 (the leaves, which
	 * have no children), inv_mod_data[lev] == NULL.
	 *
	 * Computed at tree-build time via gf64_poly_invmod over each internal
	 * node. Cost: O(N log^2 N) schoolbook field ops, dominated by the
	 * Newton iteration's poly_mul cost; ~10^8 ops at N=1024.
	 *
	 * inv_storage is the single contiguous backing buffer that
	 * inv_mod_data points into (same allocation discipline as `storage`).
	 */
	gf64_t **inv_mod_data;
	gf64_t   *inv_storage;
} SubproductTree;

/*
 * Build a subproduct tree for the N input points {points[0..N-1]}.
 *
 * Requirements:
 *   - N == 0            -> empty tree, no allocations beyond the struct
 *   - N == 1            -> single-level tree, root polynomial = (x + x_0)
 *   - N power of 2 >= 2 -> full log2(N)+1 level tree
 *
 * On failure (non power-of-2 N, allocation failure), the struct is left in
 * a zero-initialized, free()-safe state via gf64_subproduct_tree_free.
 */
void gf64_subproduct_tree_build(const gf64_t *points, size_t N, SubproductTree *out);

/*
 * Release all internal buffers held by `tree`. Safe on a zero-initialized
 * struct and idempotent (after free, the struct is zero-initialized).
 */
void gf64_subproduct_tree_free(SubproductTree *tree);

HEDLEY_END_C_DECLS

#endif /* GF64_SUBPRODUCT_H */
