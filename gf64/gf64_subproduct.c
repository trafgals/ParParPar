/*
 * ============================================================================
 * gf64/gf64_subproduct.c — Bottom-up FFT-based subproduct tree over GF(2^64).
 *
 * T6 of the par3-cauchy-fft-kernel plan.
 *
 * Given N input points {x_0, ..., x_{N-1}} in GF(2^64), this file builds a
 * complete binary tree of subproducts:
 *
 *   - leaves (level log2(N)):   M_i(x) = x + x_i   (degree 1, coeffs [x_i, 1])
 *   - internal nodes:           product of the two child polynomials
 *   - root    (level 0):        P(x)   = ∏_i (x + x_i)   (degree N)
 *
 * Each node's polynomial multiplication is done via gf64_poly_mul (T3,
 * gf64_additive_fft.h). T3's polynomial multiplication is internally
 * FFT-based with a Naive fallback that produces identical coefficient
 * output; either way, the result is bit-exact to naive O(MN) convolution.
 * For T6 the priority is correctness over peak performance: the asymptotic
 * complexity is dominated by the per-node multiplications and matches T3.
 *
 * Tree layout — BFS order over a single backing buffer:
 *   - level_data[ℓ]   -> start of level ℓ's polynomial coefficients
 *   - level_lens[ℓ]   = N >> ℓ   (number of polynomials at this level)
 *   - level_degs[ℓ]   = 1 << ? — actually computed dynamically from N; see
 *                       the build function. For level ℓ, deg = N / 2^ℓ.
 *
 * Level 0 holds the root (one polynomial of degree N) and level log2(N)
 * holds the leaves (N polynomials of degree 1). For T6 we require N to be
 * 0, 1, or a power of 2.
 *
 * Polynomial convention: coefficients are constant-first. For
 *   x + x_i = 1 * x + x_i
 * the leaf i is stored as two coefficients: [x_i, 1].
 * ============================================================================
 */

#include "gf64_subproduct.h"
#include "gf64_additive_fft.h"
#include "gf64_mpe.h"
#include "gf64_global.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

HEDLEY_BEGIN_C_DECLS

static int gf64_is_power_of_two(size_t n) {
	return n != 0 && (n & (n - 1)) == 0;
}

/* Integer log2 of a power of two. log2(1) = 0; log2(2) = 1; ...; log2(2^k) = k. */
static size_t gf64_log2_size(size_t n) {
	size_t log_n = 0;
	while (n > 1) {
		n >>= 1;
		log_n++;
	}
	return log_n;
}

void gf64_subproduct_tree_build(const gf64_t *points, size_t N, SubproductTree *out) {
	assert(out != NULL);

	/*
	 * Zero the output so that gf64_subproduct_tree_free is always safe,
	 * including the early-return for N == 0 and on allocation failure.
	 */
	memset(out, 0, sizeof(*out));
	out->num_points = N;

	if (N == 0) {
		/* Empty tree. */
		return;
	}

	/* Level 0 of N=1 has just one leaf-as-root polynomial (degree 1).
	 * For N >= 2 the tree must be a full binary tree, so N must be a
	 * power of 2. */
	if (N > 1 && !gf64_is_power_of_two(N)) {
		assert(gf64_is_power_of_two(N));
		return;
	}

	out->num_levels = gf64_log2_size(N) + 1;

	/* Per-level metadata arrays. */
	out->level_lens = (size_t *)calloc(out->num_levels, sizeof(size_t));
	out->level_degs = (size_t *)calloc(out->num_levels, sizeof(size_t));
	out->level_data = (gf64_t **)calloc(out->num_levels, sizeof(gf64_t *));
	if (out->level_lens == NULL || out->level_degs == NULL || out->level_data == NULL) {
		gf64_subproduct_tree_free(out);
		abort();
	}

	/*
	 * Populate per-level counts and degrees.
	 *   level 0:        1 polynomial of degree N
	 *   level ℓ:        2^ℓ polynomials of degree N / 2^ℓ
	 *   level log2(N):  N leaves of degree 1
	 */
	for (size_t lev = 0; lev < out->num_levels; lev++) {
		out->level_lens[lev] = (size_t)1U << lev;
		out->level_degs[lev] = N >> lev;
	}

	/*
	 * Total coefficients needed across the whole tree:
	 *   sum_{ℓ} level_lens[ℓ] * (level_degs[ℓ] + 1)
	 * = sum_{ℓ} (2^ℓ)        * (N / 2^ℓ + 1)
	 * = sum_{ℓ} (N + 2^ℓ)
	 * = (log2(N) + 1) * N + (2^(log2(N)+1) - 1)
	 * = (log2(N) + 1) * N + 2 * N - 1
	 * For N=1024, log2(N)=10: 11*1024 + 2048 - 1 = 13311 coefficients.
	 */
	size_t total_coeffs = 0;
	for (size_t lev = 0; lev < out->num_levels; lev++) {
		total_coeffs += out->level_lens[lev] * (out->level_degs[lev] + 1);
	}

	out->storage = (gf64_t *)calloc(total_coeffs, sizeof(gf64_t));
	if (out->storage == NULL) {
		gf64_subproduct_tree_free(out);
		abort();
	}

	/* Set per-level pointers into the contiguous storage. */
	{
		size_t offset = 0;
		for (size_t lev = 0; lev < out->num_levels; lev++) {
			out->level_data[lev] = out->storage + offset;
			offset += out->level_lens[lev] * (out->level_degs[lev] + 1);
		}
	}

	/*
	 * Initialize leaves at the highest level. Each leaf has
	 * degree 1 and represents x + x_i with constant coefficient x_i
	 * and x-coefficient 1.
	 */
	{
		size_t last_lev = out->num_levels - 1;
		gf64_t *leaves = out->level_data[last_lev];
		size_t leaf_deg = out->level_degs[last_lev]; /* == 1 */
		assert(leaf_deg == 1);
		for (size_t i = 0; i < N; i++) {
			leaves[2 * i + 0] = points[i];
			leaves[2 * i + 1] = 1ULL;
		}
	}

	/*
	 * Bottom-up: for each child level (from leaves down to level 1),
	 * compute the parent level as the FFT-based product of its two
	 * children. After the loop the root sits at level 0.
	 *
	 * The number of parent polynomials at level (child_lev - 1) is
	 * level_lens[child_lev - 1] = level_lens[child_lev] / 2. Each
	 * parent has degree 2 * child_deg.
	 */
	{
		size_t last_lev = out->num_levels - 1;
		for (size_t child_lev = last_lev; child_lev > 0; child_lev--) {
			size_t parent_lev = child_lev - 1;
			size_t child_deg  = out->level_degs[child_lev];
			size_t parent_deg = out->level_degs[parent_lev];   /* == 2 * child_deg */
			size_t parent_count = out->level_lens[parent_lev]; /* == level_lens[child_lev] / 2 */

			gf64_t *parent_base = out->level_data[parent_lev];
			gf64_t *child_base  = out->level_data[child_lev];

#ifdef GF64_OPENMP_PARALLEL_PREPARE
			/* Work-proportional chunking: gate OMP on multiplication
			 * work (parent_count * child_deg) rather than parent_count
			 * alone — cubic P2 (task 16). The cheapest levels (small
			 * child_deg × many parents) stay serial; the expensive
			 * near-root levels (large child_deg × few parents) get
			 * parallelized. gf64_poly_mul is deterministic on the same
			 * inputs, so the result is bit-equal to the serial path
			 * regardless of execution order. */
			#pragma omp parallel for schedule(dynamic, 16) \
				if(parent_count * child_deg > 4096)
#endif
			for (size_t i = 0; i < parent_count; i++) {
				gf64_t *left   = child_base + (2 * i)     * (child_deg + 1);
				gf64_t *right  = child_base + (2 * i + 1) * (child_deg + 1);
				gf64_t *parent = parent_base + i * (parent_deg + 1);
				gf64_poly_mul(parent, left, child_deg, right, child_deg);
			}
		}
	}

	/*
	 * Issue #59 A1: the per-pair polynomial modular inverse cache
	 * (P_left^{-1} mod P_right per sibling pair, populated via the
	 * schoolbook half-EGCD gf64_poly_invmod_mod) has been REMOVED. It
	 * made tree build O(N²) — at N ~ 2.62M (padded 10G/100k) that alone
	 * was ~10^12 field ops. The Bostan-Schost interpolation no longer
	 * needs the cache: gf64_multi_point_interp_internal (gf64_mpe.c)
	 * uses the derivative-based Lagrange form (P'(x_j) via one MPE of
	 * the root derivative + top-down f_L·P_R + f_R·P_L combine), which
	 * is O(M(N) log N) and needs only the polynomial tree itself.
	 * Tree build is now purely the bottom-up mul levels: O(N log² N).
	 */
}

void gf64_subproduct_tree_free(SubproductTree *tree) {
	if (tree == NULL) {
		return;
	}
	free(tree->storage);
	free(tree->level_data);
	free(tree->level_lens);
	free(tree->level_degs);
	memset(tree, 0, sizeof(*tree));
}

HEDLEY_END_C_DECLS
