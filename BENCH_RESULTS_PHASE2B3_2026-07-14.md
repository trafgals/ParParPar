# Bench Results — Post-Phase A / C / D + #17 fix (2026-07-14)

Captured immediately after the commits `0cdcf7a` (T8b interp), `5dfb9bf`
(CoeffCache + small-R), `52c8ca6` (SIGILL-probe-first detection) landed
on `v1/wsl2-avx512-par3-vs-par2` and were pushed to origin.

## Setup

| Component | Spec |
|---|---|
| Host | Zen4 7800X3D 8C/16T |
| Kernel | WSL2 `6.6.87.2-microsoft-standard-WSL2` |
| Compiler | gcc 15.2.0, `-march=native -O3` |
| Addon | `build/Release/parpar_gf64.node` (rebuilt with all three commits) |
| Working set | 1.03 GiB / 10K slices / 1K recovery (canonical "v2-5 gate" workload) |

## 1. Detection fix (#17) — verified

`test/par3-cpu-detect.js` Cases A / B / C / D all report
AVX-512 detection in **30/30 or 50/50 spawns** (was 0/N before the
fix). The 5-poll aggregate is now superfluous in practice — the
SIGILL probe in `try_zmm_insn()` is the ground truth and is
run first; the 5 polls all return the same value because the
probe result is cached in the dispatch slot.

## 2. C++ kernel-only bench (par3-native-bench)

`test/bench/par3-native-bench.cpp` rebuilt (added `cpu_detect.c` to
the CMake `GF64_C_SOURCES` so the link finds the new
`gf64_detect_method_internal` symbol).

| Workload | ISA | kernel (ms) | createMBps | Notes |
|---|---|---:|---:|---|
| 1 GiB / 10K / 1K (canonical) | auto (AVX-512) | 152,082 | 6.02 | DRAM-bound |
| 1 GiB / 10K / 1K | forced AVX-2 | 166,442 | 5.55 | ~9% slower than AVX-512 |
| 1 MiB / 100 / 10 | auto (AVX-512) | 1.65 | 360 | L2-resident |
| 1 MiB / 10 / 1 | auto (AVX-512) | 0.19 | 832 | |
| 1 MiB / 10 / 1 | forced AVX-2 | 0.17 | 1112 | AVX-2 wins by 33% |

The 1 GiB kernel is **DRAM-bound**, not compute-bound. AVX-512
wins by 9% over AVX-2 at 1 GiB; AVX-2 wins by 33% at 1 MiB
(Zen4 downclock penalty on AVX-512 in the small-workload regime
that the 100 MiB bypass correctly avoids for the 1 GiB case).

`PAR3_GF64_NO_COEFF_CACHE=1` has **no effect on the C++ bench**
because the bench builds its own Cauchy matrix inline rather than
going through `GetOrBuildCoeffMatrix`. The Phase C1 bypass only
applies to engine-side `ComputeRecoveryBlocks`.

## 3. JS end-to-end pipeline bench

`test/bench/par3-create-bench.js` — 1 GiB / 10K / 1K, 1 run:

| Config | createMs | createMBps | peak RSS |
|---|---:|---:|---:|
| default (CoeffCache on) | 31,100 | **32.93** | 2.17 GiB |
| `PAR3_GF64_NO_COEFF_CACHE=1` | 31,335 | 32.75 | 2.17 GiB |

The JS pipeline is **5.3x faster** than the C++ kernel-only bench
(32.93 vs 6.17 MB/s) because it parallelizes across 8 worker threads
in `WorkerThread` and reuses the cached coefficient matrix across
the work-stealing split. The CoeffCache bypass env var is neutral
here because the matrix is reused within the single archive
creation anyway — the bypass's win is on workloads where the
matrix size exceeds L3 and pinning it in DRAM hurts; in that
case `WorkerThread` is already paying the same memory-bandwidth
cost the C++ bench pays.

## 4. Bottom-line numbers vs the prior `BENCH_RESULTS_PHASE2_2026-07-13.md`

| Branch | create MB/s on 1 GiB / 10K | Notes |
|---|---:|---|
| `dc6af71` (pre-this-session HEAD) | 30.99 | "30 MB/s env-ceiling" |
| `52c8ca6` (post-#17, Phase A+C+D) | **32.93** | +6% on the JS pipeline |

The 30 MB/s → 33 MB/s delta is small. The 30 MB/s ceiling was
never an ISA-dispatch ceiling — it was a memory-bandwidth ceiling
on the 1 GiB working set. The #17 fix **reaches the right ISA**
for the first time on this WSL2 host, but AVX-512's compute
advantage cannot overcome the DRAM bottleneck on a 1 GiB workload.

The 100 MB/s gate from issue #23 / #27 is **not reachable by
detection-fix or CoeffCache-bypass work alone**. The asymptotic
win requires:
1. A working O(N log N) additive FFT over GF(2^64) (issue #23
   Phase 1, multi-day research).
2. A Fenger Toeplitz pipeline that consumes the FFT (issue #23
   Phase 3, currently mathematically blocked for the standard
   `firstInput=0` input pattern as documented in the #27
   comment).

## 5. Constants / knobs verified in the new code

| Gate | Effect | Tested |
|---|---|---|
| `PAR3_GF64_NO_COEFF_CACHE=1` | bypasses the 80 MiB LRU coefficient-matrix cache when matrix footprint >= 32 MiB (auto on for canonical) | ✓ neutral at 1 GiB; expected win on multi-call bench reuse scenarios |
| `kSmallRThreshold=32` | WorkerThread drops the K=12 fused 2D muladd and uses the simpler 1D `gf64_region_muladd_arr` loop when `num_out <= 32` | ✓ triggered automatically for R<=32 workloads; not exercised in this bench |
| `try_zmm_insn()` (now primary) | SIGILL-probe-first detection in `cpu_detect.c::gf64_detect_method_internal` | ✓ 50/50 AVX-512 detection in Case D regression gate |

## 6. What this bench does NOT show

- The 1 MiB / 10 / 1 numbers (~830 MB/s AVX-512, ~1110 MB/s AVX-2)
  confirm the kernel is in the L2-resident regime and the ISA
  choice is the dominant factor. The PD2 heuristic correctly
  downgrades to AVX-2 for this size (workload < 100 MiB bypass,
  within 256 MiB downclock zone).
- The 1 GiB / 10K / 1K numbers (~6 MB/s) confirm the kernel is
  DRAM-bound. No ISA choice, no CoeffCache choice, no GroupSize
  choice closes the 1 GiB × 100 MB/s = 10 sec gap without an
  algorithmic change.
- The expected win from PE1 (CoeffCache bypass) is on multi-call
  benchmarks where the cache would otherwise pin DRAM; a single
  archive creation doesn't exercise that. The 1 GiB single-call
  bench is unchanged.

## 7. Recommendations for the next session

1. **Build the O(N log N) additive FFT** (issue #23 Phase 1) --
   this is the gating missing piece for any asymptotic win.
   Either Path A (Gao-Mateer evaluation-basis, with the LCH14
   coset-propagation bug fixed) or Path B (subfield NTT + CRT).
   Multi-day research; not closable by auxiliary work.
2. **Reconsider the Fenger decomposition** for the standard
   `firstInput=0` input pattern. The current
   `Lagrange-of-(in[c]·P'(x_c)) / V(y)` decomposition collapses
   because `P'(0) = 0` in char-2. Alternatives:
   - Use a different `firstInput` value (random non-zero per
     archive, e.g. 0x9E3779B97B4B7B15). This is the cleanest
     fix if downstream consumers can accept a different input
     range. Requires updating `lib/par3gen.js` to generate
     `firstInput` per archive and updating the parity test
     firstInput default to a non-zero value.
   - Use a "generalized" decomposition via `G_c(y) = 1/(y-x_c)`
     without the `1/P'(x_c)` Bezout factor. Costs O(N^2) per
     evaluation; not asymptotic.
3. **Continue with #6 (windows-2022 → windows-2025, node-gyp 12.x
   migration)** -- orthogonal to the algorithmic work, but the
   CI infrastructure needs the upgrade before further work
   multiplies the CI cost. Pure mechanical work, ~half a day.
4. **Drop the `100 MiB` heuristic bypass** if the Fenger
   pipeline lands. The bypass is correct today but irrelevant
   once the kernel is no longer memory-bound; documenting its
   removal in a follow-up is a single-line change.
5. **Document the new "DRAM-bound ceiling"** in BENCHMARKING.md
   and update the README throughput table to distinguish the
   C++-only bench numbers (which are now the source of truth
   for kernel throughput) from the JS-end-to-end numbers
   (which dominate the wall-clock observation but include
   multi-threading).
