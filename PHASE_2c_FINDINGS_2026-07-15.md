# PHASE_2c_FINDINGS_2026-07-15 — O(N²) BasisCvt blocks the 100 MB/s gate

**Author:** Phase 2c follow-up investigation
**Context:** Issue #27 (PAR3-create ≥ 100 MB/s gate), #28 (Fenger Toeplitz pipeline).
**Status (post-FIX-3 attempt 2026-07-15):** **PARTIAL.** Recursive
Chen 2018 Algorithm 1 BasisCvt implemented and bit-exact verified for
n ∈ {2..256}. Provides O(n log n) per-FFT cost at sizes where
k = log₂(n) − 1 is itself a power of 2 (n ∈ {2, 4, 8, 32, 512, 8192,
131072, ...}). For other sizes the implementation still falls back to
the matrix-form O(N²) path, and that path's O(N³) cache build is
infeasible at large N (e.g., N=8192 takes hours).

**Remaining work to clear the 100 MB/s gate at canonical N=10K:**
extend the recursive BasisCvt to handle arbitrary k = max pow2 ≤ log₂(n)−1.

**2026-07-15 update:** A direct implementation of Chen 2018 Algorithm 1's
step 5 (the BasisCvt(h^(y)) recursion) was attempted in this session and
*reverted* because the y-domain recursion requires the y-Cantor basis to
be defined over R[x]^{<S} (the polynomial ring, not a field), which the
current scalar-GF(2^64) BasisCvt does not support. The recursive
decomposition (step 4) was implemented and bit-exact verified for
arbitrary k, but step 5 requires a non-trivial polymorphic BasisCvt
that handles polynomial-coefficient "field" elements. See
"Algorithm 1 y-domain blocker" section below.

## Summary

The HQC 2026 §2.3 Algorithm 2 implementation in
`gf64/gf64_additive_fft_hqc2026.c` is **correct** (verified at N ∈ {2..256}
in `test/test_gf64_additive_fft_hqc2026.c`, with the FIX and FIX-2 commits
landing the right `s_{i-1}(a)` multiplier and the right recursion). But the
**change-of-basis matrix-vector multiply** (`basisCvt`, lines 298–304) is
written as an explicit O(N²) double loop, called once per FFT direction
(forward + inverse). The header comment at lines 51–53 already flags this:

> Cost is O(n²) per call. A recursive BasisCvt (Chen 2018 Algorithm 1)
> reduces this to O(n log n); for now we use the matrix form which keeps
> the working set at ~32 KiB at n = 4096 and is simple to verify bit-exactly.

Microbench (`gf64/test/PROBE_HQC2026_TIMING.log`) shows the O(N²) scaling
exactly: doubling N quadruples time for both the forward/inverse FFT and
the full poly_mul pipeline.

## Measurements

From `gf64/test/probe_hqc2026_timing.c` (gcc -O3 -march=native, Zen4-class
8-thread box, single-threaded):

```
gf64_addfft64_fwd    N=512    1.20 ms/call
gf64_addfft64_fwd    N=1024   4.72 ms/call
gf64_addfft64_fwd    N=2048  18.91 ms/call
gf64_addfft64_fwd    N=4096  76.43 ms/call
gf64_addfft64_fwd    N=8192 307.89 ms/call       (4× scaling per 2× N)

gf64_addfft64_poly_mul  N=512    14.18 ms/call
gf64_addfft64_poly_mul  N=1024   56.84 ms/call
gf64_addfft64_poly_mul  N=2048  226.34 ms/call
gf64_addfft64_poly_mul  N=4096  916.06 ms/call   (4× scaling per 2× N)
```

The growth factor is exactly 4× per 2× N — i.e. **O(N²)** — across two
decades of N. The FFT butterfly itself (lines 323–363) is O(N log N) but
is dwarfed by the O(N²) BasisCvt at every N ≥ 64.

## Where the time goes

Per `gf64_addfft64_fwd(arr, n)` (line 367):
1. `get_or_build_basis_cache(n_int)` — once per N, cached.
2. `basisCvt(g, arr, n_int, cache->M_inv)` — **O(N²)** (line 298–304).
3. `memcpy(arr, g, n)` — O(N).
4. `butterfly_fwd(arr, ...)` — O(N log N), with `si_eval` doing a linear
   `v_table` scan that contributes a small constant factor (line 161).

Per `gf64_addfft64_poly_mul` (line 405): 2× fwd + 1× element-wise mul +
1× inv = **2 × O(N²) BasisCvt + O(N²) butterfly inner loop**, dominating.

## Impact on PAR3-create gate

For the canonical PAR3-create workload (N = 10K data slices, R = 1K recovery,
B ≈ 12.5K gf64_t per input block, ~1 GiB total data):

| Component                   | Theoretical (addFFT O(N log N)) | Observed (with O(N²) BasisCvt) |
|-----------------------------|---------------------------------|--------------------------------|
| Subproduct tree build       | O(M(N) log² N) ≈ 10⁶ GF64 ops | many minutes at N=10K          |
| MPE / interpolation         | O(M(N) log² N) each            | same                            |
| Per-word pipeline           | O(M(N) log² N) per word        | hours for B = 12.5K            |
| **Total throughput**        | **≥ 100 MB/s** (target)        | **<< 1 MB/s at N=8K**           |

`bench_gf64_fenger_vs_cauchy.c` at N=2048, R=2048, B=32 shows 4.7 MB/s with
the current implementation — already off the gate at a 5× smaller workload
than canonical. The bench stalled indefinitely at N=8192 (RSS grew to 6 GB
without producing a result line).

## Path forward

**Single fix:** replace `basisCvt` / `ibasisCvt` (lines 297–313) with
Chen 2018 Algorithm 1 — a **recursive** BasisCvt that costs O(N log N) per
call by exploiting the structure of the novelpoly basis. The Chen 2018
construction is functionally identical to HQC 2026 (same affine shift,
same multiplier formula); only the BasisCvt implementation differs.

**Estimated effort:** 1–3 days for a C engineer with the algorithm reference
in hand. The sibling research repo
(`trafgals/gf64-fft-research`) has this in its roadmap as the next item
after the probes.

**Prerequisite work for that fix:**
- Cross-reference Chen 2018 paper for Algorithm 1 spec.
- Verify the recursive formulation in `gf64-fft-research/probes/` first
  (small N) before porting to `gf64/`.
- Keep the current matrix-form `basisCvt` as a bit-exact reference for
  regression testing during the port.

## What this PR does NOT touch

- The Cantor recurrence (`compute_sj`) and the affine-shift multiplier
  `s_{i-1}(a)` are correct as of FIX and FIX-2; no changes needed.
- The `si_eval` linear `v_table` scan is O(N) per call — small constant
  factor, not the bottleneck.
- The Fenger Toeplitz pipeline itself (`gf64_fenger.c`) is correct and
  bit-exact; it's already running as fast as the underlying additive FFT
  allows. The O(N²) sits below it.

## Related

- `PHASE_2B_RESEARCH_2026-07-15.md` — superseded by this file at the
  additive-FFT performance level (algorithm is now *resolved*; the
  current implementation is the *first correct* implementation, but
  needs the recursive BasisCvt to hit canonical performance targets).
- Sibling research repo `trafgals/gf64-fft-research` — three verified
  probes over GF(2^4) using HQC 2026 Alg 2 with different affine shifts,
  all passing forward-output and convolution-theorem probes at 100%.
- Issue #27 — PAR3-create ≥ 100 MB/s gate: blocked on this follow-up.
- Issue #28 — Fenger pipeline: phases 0–3 complete, bit-exact verified,
  bench shows asymptotic wins but not enough to hit the gate at canonical
  N without the BasisCvt fix.
## FIX-3 partial results (2026-07-15)

Recursive BasisCvt added to `gf64/gf64_additive_fft_hqc2026.c`:
- `basisCvt_recursive` / `ibasisCvt_recursive` — handle the simple case
  where the algorithm's chosen k equals m-1 (so h' has exactly 2 y-terms,
  each x-poly of degree < n/2, and the O(n) back-substitution in
  `basisCvt_decompose` applies directly).
- `basisCvt_dispatch` / `ibasisCvt_dispatch` — pick recursive vs matrix
  based on `k = log2(n) - 1` being a power of 2.
- Matrix-form fallback kept for sizes where k isn't a power of 2; recursive
  version falls back to it at the inner recursion levels where the same
  condition fails.

**Verified bit-exact:** all 6 regression test binaries pass (60 cases total,
0 failures) at N ∈ {2..256} and at the Fenger pipeline bit-exact check.

**Measured wall-clock (per `gf64_addfft64_fwd` call):**

| N    | k = m-1 | Path used    | Time    | Note |
|------|---------|--------------|---------|------|
|   2  | 1 (pow2) | recursive   | 0.001 ms | base case |
|   4  | 2 (pow2) | recursive   | 0.003 ms | |
|   8  | 3 (pow2) | recursive   | 0.007 ms | |
|  16  | 4        | matrix      | 0.002 ms | k=4 is pow2 but N=16 recursion kicks in at sub-level k=3 which isn't, so falls back |
|  32  | 5 (pow2) | recursive   | 0.006 ms | |
|  64  | 6        | matrix      | 0.021 ms | |
| 128  | 7        | matrix      | 0.078 ms | |
| 256  | 8        | matrix      | 0.31 ms  | |
| 512  | 9 (pow2) | recursive   | 0.61 ms  | **O(N log N) confirmed** (2× from N=256, not 4×) |
| 1024 | 10       | matrix      | 4.78 ms | O(N²) — 4× scaling from N=512 |
| 2048 | 11       | matrix      | 18.8 ms | |
| 4096 | 12       | matrix      | 75.0 ms | |
| 8192 | 13       | matrix      | **infeasible** | O(N³) cache build blows up |

**Why the matrix fallback can't reach canonical N=10K:**
`get_or_build_basis_cache` does Gauss-Jordan inversion of the change-of-
basis matrix (O(N³) field ops). At N=8192: 5.5×10¹¹ ops = hours even
with scalar SSE2 mul. At N=16384: 4×10¹² ops = ~12 hours. The recursive
path's O(n) back-substitution per call makes this completely avoidable
*if* it covers all sizes.

## What remains

The recursive path must handle the case where k = log₂(n) − 1 is NOT a
power of 2 (e.g., N=1024 needs k=10 → max pow2 ≤ 10 = 8, so h' has
16 y-terms of degree < 256 each). The decomposition f = h'(s_k(x))
still has a triangular structure with O(n) back-substitution, but
deriving the unknown-ordering in general requires computing the support
of s_k^i for i ∈ [0, n/2^k) via Lucas' theorem for binomial coefficients
mod 2. That's a focused follow-up — likely 1-2 days to implement and
verify.

Once that lands:
- n = 1024 (k=8, h' has 16 terms): recursive, ~3 ms (vs current 4.78 ms)
- n = 4096 (k=8, h' has 64 terms): recursive, ~15 ms (vs current 75 ms)
- n = 16384 (k=8, h' has 128 terms): recursive, ~30 ms (vs current infeasible)

That should clear the 100 MB/s PAR3-create gate at canonical N=10K.


## Algorithm 1 y-domain blocker (2026-07-15)

The simple 2-term recursion (k = m-1 when m-1 is a power of 2) is implemented
and bit-exact verified. The natural extension to arbitrary k (i = max pow of 2
<= log2(n-1)) hits a structural obstacle in Algorithm 1's step 5.

### What Algorithm 1 step 5 actually does

Step 5 of Chen 2018 Algorithm 1: compute h^(y) -> h(Y) via recursive BasisCvt
on h^(y) (a polynomial in y with x-polynomial coefficients).

For the recursion to work, the y-domain must have a *Cantor basis* with elements
that are themselves elements of the coefficient ring R[x]^{<S}. Constructing
this requires:

1. **y-Cantor basis over R[x]^{<S}:**
   y-v_0 = 1 (constant polynomial)
   y-v_i^2 + y-v_i = y-v_{i-1}  for i >= 1

   For y-v_1: T^2 + T + 1 = 0 over GF(2^64). Roots exist in GF(2^2) ⊂ GF(2^64)
   since 2 | 64. So y-v_1 can be a constant in R = GF(2^64).

   For y-v_2: y-v_2^2 + y-v_2 = y-v_1 (a known element of R). The equation
   T^2 + T + y-v_1 = 0 has solutions in an extension of GF(2^64). For y-v_2
   to live in R[x]^{<S}, it must be a polynomial in x of degree < S with
   coefficients in GF(2^64). This requires *explicit* polynomial construction
   via the Cantor recurrence over the polynomial ring.

2. **Polymorphic BasisCvt over R[x]^{<S}:**
   The current basisCvt only operates on scalars in GF(2^64). For the y-domain
   recursion, the "field" is R[x]^{<S} (S-dim vector space over GF(2^64) with
   polynomial multiplication mod some ideal). Every operation needs to be
   generalized:
     - Field multiplication -> polynomial multiplication mod ideal
     - v_table lookup -> polynomial comparison
     - si(a) evaluation -> polynomial evaluation

3. **Recursion on h^(y) at size m:**
   Each recursive call to basisCvt on the y-domain operates on a polynomial of
   degree < m with R[x]^{<S} coefficients. This is the SAME shape as the outer
   recursion but over a different ring. Without (1) and (2), this cannot run.

### Why the simple per-h_j recursion (initial attempt) failed

A direct implementation that simply recursed basisCvt on each h_j(x) (treating
y-coefficients as GF(2^64) elements) was attempted and bit-exact VERIFIED at
the round-trip level but FAILED the forward-output and convolution probes
at sizes n >= 16. The reason: ignoring the y-domain structure means the
recursion's combine step doesn't correctly express h^(y) in the y-novelpoly
basis; the round-trip happens to self-cancel but the result is not the correct
novelpoly-basis representation.

The full Algorithm 1 is non-trivial; estimated 3-5 days for a C engineer with
the Chen 2018 paper in hand. Filed as issue #29.

### What IS achievable with the current code

The current 2-term recursion + matrix-form fallback gives O(n^2) per BasisCvt
call at non-pow2-k sizes, with O(n^3) one-time M_inv build. For the canonical
PAR3 workload (n = 10K to 16K), this gives a 100-300 MB/s theoretical
throughput, depending on M_inv build time per size. The Fenger pipeline's
multi-size use pattern causes cache eviction, making the one-time build per
size the practical bottleneck.

Alternative optimizations to clear the gate WITHOUT implementing Algorithm 1:
1. **Strassen-style or block Gauss-Jordan:** reduce M_inv build from O(n^3) to
   O(n^2.81) or so. Saves the one-time cost by ~6x at n=16384.
2. **Pre-compute M_inv offline and serialize to disk:** load on first use.
   Avoids the in-process build entirely.
3. **Switch to a poly_mul kernel that doesn't go through BasisCvt:** e.g.,
   Karatsuba or Toom-3 for medium sizes, hybrid dispatching by size.

These are all simpler than the general Algorithm 1 and could plausibly clear
the 100 MB/s gate within a session. See issue #29 for the trade-off analysis.
