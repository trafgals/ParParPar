# PHASE_2c_FINDINGS_2026-07-15 — O(N²) BasisCvt blocks the 100 MB/s gate

**Author:** Phase 2c follow-up investigation
**Context:** Issue #27 (PAR3-create ≥ 100 MB/s gate), #28 (Fenger Toeplitz pipeline).
**Status:** Fenger pipeline is bit-exact and asymptotically optimal in theory;
current implementation has a hidden **O(N²) per FFT** component that
dominates wall-clock at large N. Hitting the gate requires Chen 2018
Algorithm 1 (recursive BasisCvt) — a focused 1–3 day piece of work.

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