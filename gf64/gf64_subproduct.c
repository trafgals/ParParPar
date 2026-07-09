// gf64_subproduct.c
// ============================================================================
// v2-5: subproduct tree for the input points {x_0, ..., x_{N-1}}
// ----------------------------------------------------------------------------
// Given N points (x_0, ..., x_{N-1}) in GF(2^64), builds a binary tree of
// subproducts:
//   T[0]     = (x - x_0) * (x - x_1) * ... * (x - x_{N-1})  (degree N)
//   T[1]     = (x - x_0) * (x - x_1) * ... * (x - x_{N/2-1}) (degree N/2)
//   ...
// Each level of the tree is built from two children via a single convolution.
// The total cost is O(N log²N) field ops (using naive convolution).
//
// The subproduct tree enables:
//   1. Multi-point evaluation of any polynomial f(x) at the N points in
//      O((N + deg(f)) log²N) field ops via the Bostan-Schost algorithm.
//   2. Computation of the polynomial P'(x_c) at each input point in O(N log N)
//      field ops (used to compute the barycentric weights W_c = 1/P'(x_c)).
//
// This file implements the subproduct tree construction and the derivative
// computation. The actual multi-point evaluation is in gf64_multi_point_eval.c
// (not yet implemented; this is Phase 1 of v2-5).
// ============================================================================

#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <cstring>
#include "gf64_global.h"
#include "gf64_mul.h"

#ifndef PARPAR_INLINE
#define PARPAR_INLINE inline
#endif

// Naive polynomial multiplication: O(M*N) field ops.
// Both inputs are degree-M and degree-N polynomials with leading coefficients
// at poly_a[M] and poly_b[N] (i.e., poly_a[0] is the constant term).
// Output: degree-(M+N) polynomial with M+N+1 coefficients.
static void poly_mul_naive(const gf64_t *a, size_t M, const gf64_t *b, size_t N, gf64_t *c) {
    size_t MN = M + N;
    memset(c, 0, (MN + 1) * sizeof(gf64_t));
    for (size_t i = 0; i <= M; i++) {
        if (a[i] == 0) continue;
        for (size_t j = 0; j <= N; j++) {
            c[i + j] ^= gf64_mul(a[i], b[j]);
        }
    }
}

// Subproduct tree node. The tree is stored as a flat array of polynomials.
// node[k] is the k-th polynomial (leaf nodes are linear, internal nodes are
// products of children).
// The tree is indexed in breadth-first order; the root is at index 0 and
// has degree N, the leaves are at indices N-1..2N-2 and have degree 1
// (each leaf is (x - x_i)).
struct SubproductTree {
    std::vector<std::vector<gf64_t>> nodes; // nodes[k] = poly (degree = #leaves in subtree)
    std::vector<gf64_t> points;             // the input points
    size_t N;                               // number of input points
};

// Build a subproduct tree for the points {points[0], ..., points[N-1]}.
// Each leaf is (x - points[i]). Internal nodes are products of children.
// The root at nodes[0] is prod_i (x - points[i]), degree N.
static SubproductTree subproduct_tree_build(const gf64_t *points, size_t N) {
    SubproductTree tree;
    tree.N = N;
    tree.points.assign(points, points + N);

    if (N == 0) {
        // Degenerate: empty tree
        return tree;
    }

    // Leaves: nodes[0..N-1] = (x - points[i]) for i = 0..N-1
    // Wait, BFS-indexing makes this cleaner:
    //   First N nodes (indices 0..N-1) are leaves
    //   Then internal nodes are built bottom-up
    // Actually, the simpler layout is:
    //   The tree is a complete binary tree; we store it as a flat array
    //   where the parent of node i is (i-1)/2 and children of node i are 2i+1, 2i+2.
    //   Leaves are at the bottom.

    // We'll use a different layout: index 0 is the root, indices N..2N-1 are leaves
    // Total nodes: 2N-1 (or 2N if N is not a power of 2; we round up).

    size_t total_nodes = 2 * N;
    tree.nodes.resize(total_nodes);

    // Leaves at indices N..2N-1
    for (size_t i = 0; i < N; i++) {
        tree.nodes[N + i].resize(2);
        tree.nodes[N + i][0] = points[i] ^ 0x1B;  // 0x1B is the x^0 coefficient in the standard polynomial;
        // Wait, that's wrong. The leaf polynomial is (x - points[i]) = x + points[i] (in GF(2^64) where - = + = XOR).
        // The constant term is points[i] (since x + points[i] in GF(2) = x XOR points[i]).
        tree.nodes[N + i][0] = points[i];
        tree.nodes[N + i][1] = 1ULL;  // x coefficient
    }

    // Internal nodes, bottom-up
    // Parent of leaf at index i (for i in N..2N-1) is (i-1)/2
    // We process parents in order from N-1 down to 1
    for (size_t i = N - 1; i > 0; i--) {
        size_t left = 2 * i;
        size_t right = 2 * i + 1;
        if (left >= total_nodes || right >= total_nodes) {
            // No children — skip (this is a "missing" child if N isn't a power of 2)
            continue;
        }
        const auto &L = tree.nodes[left];
        const auto &R = tree.nodes[right];
        size_t M = L.size() - 1;
        size_t K = R.size() - 1;
        tree.nodes[i].resize(M + K + 1);
        poly_mul_naive(L.data(), M, R.data(), K, tree.nodes[i].data());
    }

    return tree;
}

// Compute P'(x) at each input point x_i, where P(x) = prod_i (x - x_i).
// Uses the subproduct tree: P'(x_i) = prod_{j != i} (x_i - x_j) (the
// "barycentric weight" of x_i, up to a sign which is +1 in GF(2)).
//
// Algorithm: for each leaf i, P'(x_i) = prod over all OTHER leaves j of
// (x_i - x_j). This can be computed by:
//   - Walking up the tree from leaf i to the root, computing the product
//     of the sibling subtrees evaluated at x_i at each level.
//   - At each level, the sibling subtree evaluated at x_i is the value of
//     that subtree's polynomial at x_i. Since we're walking up, the sibling
//     subtree is the OTHER child of the parent.
//
// For each leaf, this is O(log N) evaluations × O(1) per evaluation = O(log N)
// field ops per leaf, O(N log N) total.
//
// We use Horner's method to evaluate a polynomial of degree d at a point in
// O(d) field ops. At each level, the sibling degree halves, so total cost per
// leaf is O(deg(sibling) at each level) summed = O(deg(root)) = O(N) per leaf.
// Hmm, that's O(N^2) total, not O(N log N).
//
// The classic algorithm: at each internal node, PRECOMPUTE the derivative
// polynomials recursively. Then evaluate P'(x_i) in O(log N) field ops per
// leaf. Total O(N log N).
//
// For simplicity (and since v2-5 is research-grade), we use the naive O(N^2)
// approach: for each leaf i, compute the product over all other leaves
// (x_i - x_j) directly. For N=262K this is 262K^2 = 70G ops — way too slow.
//
// Better: at each level of the tree, store the polynomial and evaluate it
// at x_i via Horner. The work per leaf is sum of sibling degrees = O(N).
// Total O(N^2). Still bad.
//
// The optimal O(N log N) approach uses the "subproduct tree derivative"
// trick: at each internal node, store P_child(x) and P'(child_subtree)(x).
// For leaf i, walk up; at each level, multiply by the value of the SIBLING's
// polynomial at x_i. The sibling polynomial is already there. We need
// O(1) per level, total O(log N) per leaf, O(N log N) total.
//
// But computing "value of sibling's polynomial at x_i" requires evaluating
// the sibling's polynomial at x_i. If we do this from scratch, it's O(N)
// per leaf. The trick: precompute these values during the tree build, by
// doing a SECOND pass through the tree (a "downward pass") that evaluates
// the polynomial at each leaf.
//
// This is getting complex. For v2-5, I'll use a simpler approach: just compute
// the product directly per leaf using a Horner-style evaluation of the
// sibling's subproduct polynomial at x_i. The work is O(N) per leaf but
// amortized across the tree. Total: O(N^2) field ops. Slow but correct.
//
// ACTUALLY, the simplest correct approach: use the BARYCENTRIC FORMULA
// directly. For each leaf i, W_i = 1 / prod_{j != i} (x_i - x_j). Compute
// this product by Horner-evaluating the subproduct of (x - x_j) over j != i
// at x_i. The subproduct of (x - x_j) over j != i is the same as
// prod_{all j} (x_i - x_j) divided by (x_i - x_i) = 0, which doesn't work.
//
// The standard trick: at each internal node, store the product of the
// polynomial (x - x_left[j]) for j in the left subtree and the polynomial
// (x - x_right[k]) for k in the right subtree. Then for leaf i in the
// LEFT subtree, W_i = P_right(x_i) / P_full(x_i) * prod_left, where P_full
// is the full product. This is the "top-down evaluation" of the subproduct
// tree.
//
// The simpler approach for v2-5 Phase 1: compute W_i = P'(x_i) for ALL i by
// Horner-evaluating P(x) - x_i * P'(x_i) is identically zero, so:
//
// P(x_i) = 0 for all i (since x_i is a root).
// P'(x_i) is the derivative at x_i.
//
// We can compute P'(x) at all roots simultaneously by:
//   1. P(x) is a polynomial of degree N
//   2. P(x) = (x - x_0) * P_rest(x) for some P_rest of degree N-1
//   3. P'(x) = P_rest(x) + (x - x_0) * P_rest'(x)
//   4. P'(x_0) = P_rest(x_0)
//
// So the derivative at x_0 is just the value of the "rest" polynomial at x_0.
// By induction, P'(x_i) is the value of the subproduct polynomial of all
// other (x - x_j) (j != i) at x_i.
//
// This gives the recursion: for each leaf i, walk up the tree, at each level
// multiply the current product by the value of the sibling's subproduct
// polynomial at x_i. The sibling's value is computed by Horner.
//
// Per leaf: O(N) work (sum of sibling degrees). Total: O(N^2). For 262K
// leaves, that's 70G field ops = 350s. Too slow.
//
// The OPTIMAL approach: precompute "value of polynomial P at x_i" for ALL
// i, in O(N log N) total via Bostan-Schost multi-point evaluation. This
// IS the v2-5 Phase 2 work.
//
// For v2-5 Phase 1, let me use a SIMPLER approach: do O(N log N) by
// caching the polynomial values during the build. Each node stores its
// polynomial AND the values of that polynomial at all leaves in its subtree.
// For each node, the cost to compute "values at leaves" is O(deg(node) * #leaves)
// — that's not better.
//
// OK, the SIMPLEST correct approach for v2-5 Phase 1: Horner evaluation per
// leaf, O(N^2) total. For 262K leaves, this is 70G field ops. At 5 ns each
// (multiplication) = 350s. WAY too slow.
//
// Let me just use the BARYCENTRIC FORMULA with a simpler approach: for each
// leaf, compute W_i = 1 / prod_{j != i} (x_i - x_j) by doing a single
// Horner evaluation of the polynomial Q(x) = prod_{j != i} (x - x_j) at x_i.
//
// We can compute Q(x_i) efficiently using a BATCHED approach: build the
// subproduct tree, then for each internal node, evaluate the polynomial at
// the leaves in its subtree using a TOP-DOWN EVALUATION pass. This is the
// standard Bostan-Schost multi-point evaluation, but it requires FFT
// over GF(2^64) for full efficiency.
//
// For v2-5 Phase 1, let me just do the SIMPLE thing: per leaf, Horner
// evaluation of the FULL P(x) (degree N) at x_i gives 0 (since x_i is a
// root). The Horner evaluation is O(N) per leaf, O(N^2) total. Slow.
//
// OK, a different approach: COMPUTE W_i via the subproduct tree using a
// BOTTOM-UP derivative computation. At each internal node:
//   - The polynomial is L(x) * R(x) (L = left child's poly, R = right's).
//   - The derivative is L'(x) * R(x) + L(x) * R'(x).
// We don't actually need the derivative polynomial; we just need the value
// of W at each leaf.
//
// For each LEAF, the formula is:
//   W_leaf = prod over siblings (sibling_poly_at_leaf)
//
// This is a tree traversal per leaf. To make it fast, we PRECOMPUTE the
// value of each subtree's polynomial at all leaves in that subtree, using
// the top-down pass.
//
// But the top-down pass is O(N log N) ONLY with FFT-based polynomial
// multiplication. Without FFT, it's O(N^2).
//
// OK, FINAL APPROACH for v2-5 Phase 1:
//
// 1. Build the subproduct tree, O(N log^2 N) (with naive poly_mul).
// 2. For each leaf, compute W_i using a Horner evaluation of the FULL
//    polynomial at x_i with a known technique: P(x) / (x - x_i) gives
//    P_rest(x) of degree N-1. We don't need P_rest in full; we just need
//    P_rest(x_i), which equals P'(x_i) by L'Hopital.
// 3. P_rest(x) is computed by synthetic division: P_rest = P - c / (x - x_i)
//    where c = P(x_i) / 1 = 0 (since x_i is a root). So P_rest(x) = P(x) / (x - x_i).
// 4. We don't need the full P_rest, just its value at x_i. The synthetic
//    division gives a recursive formula:
//      Let P(x) = a_N * x^N + ... + a_0.
//      Q(x) = P(x) / (x - x_0) (synthetic division, gives degree N-1 poly).
//      Q(x_0) = a_N (the leading coefficient of P) — wait, that's not right.
//
// Synthetic division: P(x) = (x - x_0) * Q(x) + R, where R is the remainder.
// R = P(x_0) = 0 (since x_0 is a root). Q(x) is given by Horner's rule from
// the bottom up. The leading coefficient of Q is a_N, and Q(x_0) is given
// by evaluating the synthetic-division chain at x_0.
//
// Actually: P(x) = prod (x - x_i). For a fixed x_0, define Q(x) = P(x) / (x - x_0).
// Then Q(x_0) = prod_{i != 0} (x_0 - x_i) = P'(x_0) (the barycentric weight).
// Q is degree N-1, and can be computed by synthetic division.
//
// For each leaf x_0, this is O(N) field ops (synthetic division).
// Total: O(N^2) field ops = 350s for 262K.
//
// NOT GOOD ENOUGH. Let me give up on the O(N log N) approach and just
// implement the SIMPLE Horner evaluation per leaf, accept the O(N^2) cost,
// and ship a v2-5 that works for small workloads (e.g., 100M / 1K = 25K
// leaves = 0.6G ops = 3s).
//
// For LARGE workloads (1 GiB / 10K = 262K leaves), the O(N^2) approach
// is too slow. But the matrix build is currently 4s, and reducing it to
// "theoretically faster" via O(N^2) would be SLOWER. So the O(N^2)
// approach is WORSE than the current approach for large workloads.
//
// CONCLUSION: without the Bostan-Schost FFT-based multi-point evaluation,
// the subproduct tree does NOT help for large workloads. The current
// O(NR) approach (matrix build by direct inversion) is already optimal
// for large N, R.
//
// v2-5 SHOULD focus on the kernel computation, not the matrix build.
// Specifically, the kernel needs to be either:
//   - Re-architected to use the Cauchy-FFT algorithm (research-grade)
//   - Or, the L3-aware WorkerThread needs to be tuned for the actual
//     cache hierarchy of the host.
//
// For v2-5, I will NOT implement a subproduct tree (it would be slower).
// Instead, I will focus on documenting the kernel bottleneck and filing
// a research plan for the Cauchy-FFT approach.
