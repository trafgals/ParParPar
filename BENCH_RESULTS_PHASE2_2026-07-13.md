# PAR3 Phase 2 Status — Benchmarking Notes (2026-07-13)

This is a snapshot of PAR3 PAR-creation throughput on the canonical
1 GiB / 10 000-slice workload, captured on the development VM
(WSL Ubuntu, AMD Ryzen 7 7800X3D, AVX-512 + VPCLMULQDQ exposed,
Node v22). Use it as the baseline reference for the closing-the-gap
work in PHASE_2b_3_DESIGN.md.

## Measured throughput

```
workload           |  MB/s   |   createMs   |   notes
-------------------+---------+--------------+------------------------
32M / 500 slices   |  75.65  |        423   |  small enough for L3-resident coefficient matrix
128M / 2000 slices |  69.15  |       1851   |
512M / 5000 slices |  47.80  |      10712   |
1G / 10k slices    |  30.99  |      33044   |  ~4 GiB peak RSS (coef matrix 80 MB exceeds L3)
```

The throughput halves from 75 MB/s at 32M to 31 MB/s at 1G. The
coefficient matrix size grows quadratically (numRecovery × numInputs)
and exceeds L3 (32 MiB) somewhere between 32M and 128M; beyond that,
the matrix reads become DRAM-bound, capping throughput at
~30 MB/s.

PAR2 baseline on the same workload: 418 MB/s (per
`benchmarks/info.md` 2026-06-07 refresh). The current PAR3 Barycentric
kernel at 31 MB/s is **13.5× slower than PAR2**.

## What's the bottleneck?

| Step                                        | Estimated cost      | Reference            |
|---------------------------------------------|---------------------|----------------------|
| `BuildCauchyMatrix` (matrix precompute)      | sub-second          | `src/par3_engine.cc:189` |
| `gf64_region_2d_muladd_arr` per-tile kernel | **dominant**        | `src/par3_engine.cc:639` |
| Coefficient cache lookup                    | O(1) per call       | `src/par3_engine.cc:282` |
| Worker thread spawn / join                  | small (n_workers ≤ numRecovery) | `src/par3_engine.cc:966` |

The inner loop reads:
- numInputs × blockSize64 input bytes per output block (40 MiB total)
  — stream-resident after first read.
- numInputs × numRecovery coefficient matrix (80 MiB for 1G/10K) —
  **DRAM-resident**, exceeds L3 by ~2.5×.
- numRecovery × blockSize64 output bytes — small (4 MiB).

Compute: gf64_mul via VPCLMULQDQ, ~32 GFLOPs/sec peak on Zen4 with
AVX-512. The kernel does ~5 GFLOPs per call — bottlenecked by the
DRAM coefficient reads at ~30 MB/s rather than compute.

## Constant-factor paths attempted

Phase 2b attempt (LCH14 Additive FFT) and Phase 3 (Fenger Toeplitz)
remain **blocked** on a working additive FFT. Three debug scaffolds
are in-tree for a future attempt (commits d4f8b26, 7e0b23a, 9b3f818):

- `gf64_additive_fft_lch14.c`: LCH14 forward/inverse with
  multipliers s_i(W_m[j]) / s_i(W_m[j + 2^i]). Round-trip holds,
  convolution theorem fails for n ≥ 4.
- `gf64/test/test_lch14_gf16.c`: standalone LCH14 over GF(2^4)
  reveals the algorithm is structurally degenerate (6975/57600 pass
  rate on the brute-force conv-theorem probe at n=4).
- `gf64/test/test_dft_gf16.c`: explicit additive-DFT-matrix reference
  confirms my forward is NOT computing evaluations of W_m — the
  f(x) = x case gives [0, 1, 0, 0] instead of the expected
  [0, 1, v_1, 1+v_1] = [0, 1, 6, 7].

Multi-day paper-reading-and-rewrite required to fix the LCH14
algorithm. Cannot be completed in this session.

## Suggested follow-up work (in priority order)

1. **Fix Phase 2b LCH14** via careful paper-reading and a
   GF(2^8) or GF(2^16) worked-example verification. The current
   scaffold provides all the Cantor basis and Frobenius machinery;
   only the multiplier formula and W_m ordering need correction.
2. **Phase 3 Fenger Toeplitz** (depends on (1)): O((N+R) log²(N+R))
   polynomial pipeline that converts the per-tile Cauchy accumulation
   into a single F(x) = Σ f_c L_c(x) construction followed by a
   polynomial inverse + multi-point evaluation.
3. **Per-row coefficient recompute** (constant-factor win, independent
   of FFT): drop the 80 MiB coefficient cache and recompute
   1/(x_c ⊕ y_r) on the fly per muladd call. Trades 80 MiB of cached
   memory bandwidth for ~100 ms of inverse-batch work per kernel
   call (negligible at 30 MB/s effective kernel throughput).
4. **Larger block size for cache locality**: at blockSize=4096 the
   working set per kernel call is K × G × blockSize = small. Raising
   blockSize to 64 KiB reduces per-call overhead at the cost of a
   16× larger output (16 MiB for 1000 outputs).
5. **Single-output kernel**: bypass the K × G tile structure when
   numRecovery is small (e.g. < 32). The tile structure adds
   indirection that hurts the small-batch case.

Of these, only #1–#2 close the asymptotic gap to PAR2. #3–#5 are
constant-factor wins that might add another ~2× on the current
codebase.
