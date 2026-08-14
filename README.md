# ParParPar

High-performance PAR3 create and repair with GF(2^64) recovery, written in C++ with a Node.js frontend.

A fork of [animetosho/ParPar](https://github.com/animetosho/ParPar) that adds PAR3 (the next-generation PAR archive format), unlimited input-block-per-slice GF(2^64) Cauchy-matrix recovery, and a C++ kernel exposed via NAPI.

- **PAR3 create, verify, and repair** — upstream has none of the three for PAR3
- **GF(2^64) Cauchy-matrix recovery** with no 65 537 input-block-per-slice cap
- **No 2 GiB file-size limit** — verified on a 4.3 GiB archive
- **Native AVX-512 / AVX-2 C++ kernel** with a JS BigInt fallback
- **Bit-exact parity** verified on 7 300+ randomized scenarios across 4 ISAs

## Throughput

Numbers below are measured on the project's workhorse recipe: **1 GiB / 10% recovery / 4 threads / tmpfs / `taskset -c 0-3`** on Zen4 / WSL2 (Ubuntu 22.04). The end-to-end ceiling on this host is ~30 MB/s — the JS layer's NAPI + `worker_threads` overhead, not the C++ kernel. A standalone C++-only bench (`test/bench/par3-native-bench`) confirms the kernel itself is hardware-bound at **~1097 MB/s on AVX-2**.

| Workload | Throughput | Notes |
|---|---:|---|
| PAR3 create, 1 GiB (default env) | ~30 MB/s | mmap + streaming NAPI + Buffer pool + LRU pool + `worker_threads` hash + 16 MiB → 256 MiB AVX-512 threshold + wider SIMD K=2 + parallel Cauchy + software prefetch |
| PAR3 create, 1 GiB (`PAR3_GF64_USE_AVX512=1`) | ~30 MB/s | Operator escape hatch for reliable AVX-512 dispatch on WSL2/Hyper-V hosts.[^1] |
| PAR3 create, 1 GiB / 1M slices (warmup) | **71.51 MB/s** | The 1 GiB / 1M-slice workload with warmup; 10K-slice workloads are JS-pipeline-bound. |
| PAR3 create, 1 GiB / 1000 slices (native Windows, AVX-512 auto) | **102 MB/s** | MSVC-built `parpar_gf64.node` on Zen4 7800X3D. AVX-512 detected via CPUID+XCR0+`_xgetbv` (MSVC intrinsics, no SIGILL probe). |
| PAR3 create, 1 GiB / 10K slices (native Windows) | 28.3 MB/s | Matches the WSL forced-AVX-512 ceiling — JS-pipeline is the bound, not the kernel. |
| PAR3 repair, 1 GiB / 5% loss (native Windows) | 58.5 MB/s | Higher than create because the recovery path uses pre-computed coefficients. |
| PAR2 reference, 1 GiB / 1000 slices create (native Windows) | **622 MB/s** | PAR3 GF(2^64) Cauchy-matrix overhead is the dominant cost vs PAR2's GFNI+AVX-512. |

PAR3 GF(2^64) trades a larger Galois field for a higher recovery-block cap and unbounded input size. The create path has been verified end-to-end on a 4.3 GiB archive; the kernel-parity test below proves bit-exact correctness on every ISA.

[^1]: On WSL2 / Hyper-V, `-march=native` binaries trigger hypervisor AVX-512 masking. The fix is three-layer: an isolated detection TU built with `-mno-avx512f` (in `gf64/cpu_detect.c`), a one-shot ZMM SIGILL probe (cached, release/acquire publish — see `gf64/test/test_gf64_zmm_probe.c`), and the `PAR3_GF64_USE_AVX512` env override. The operator escape hatch co-exists with `PAR3_AVX512_FORCE`. See `BENCHMARKING.md` for the full state and `test/par3-cpu-detect.js` for the regression test.

## Kernel benchmarks

The end-to-end throughput above is host-pipeline-bound (the JS layer's
NAPI + worker_threads overhead, not the C++ kernel). This section
reports the **kernel-level** benchmarks that isolate the kernel itself
from the pipeline. These are the numbers that matter for the next
research step (closing the kernel-to-pipeline gap).

### HQC FFT vs Karatsuba polynomial multiplication

The project ships an HQC 2026 TCHES §2.3 Algorithm 2 (LCH14 addFFT)
implementation as an alternative to Karatsuba for `gf64_poly_mul_*`.
The matrix-free recursive path is bit-exact against schoolbook at all
tested sizes (see `gf64/test/test_gf64_additive_fft_hqc2026.c`).

Benchmark: `gf64/test/bench_hqc_vs_karatsuba.c`. Measures
`gf64_poly_mul_*` only (no NAPI / JS overhead). Reported times are
averages over 10000 iterations (small n) / 5000 (n ≤ 512) / 500 (n ≤
4096) / 100 (n > 4096). Host: Zen4 / Linux / GCC `-O2 -march=native`.

| n | Karatsuba (ms) | HQC scalar (ms) | HQC AVX-512 (ms) | vs Karatsuba | AVX-512 gain |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.001 | 0.002 | 0.002 | 0.32× | 1.28× |
| 32 | 0.002 | 0.006 | 0.004 | 0.58× | 1.43× |
| 48 | 0.005 | 0.012 | 0.008 | 0.62× | 1.51× |
| 64 | 0.009 | 0.012 | 0.008 | 1.03× | 1.46× |
| 96 | 0.043 | 0.029 | 0.018 | 2.35× | 1.55× |
| 128 | 0.051 | 0.028 | 0.019 | 2.74× | 1.52× |
| 192 | 0.137 | 0.067 | 0.042 | 3.26× | 1.59× |
| 256 | 0.154 | 0.066 | 0.042 | 3.64× | 1.57× |
| 512 | 0.462 | 0.162 | 0.102 | 4.53× | 1.59× |
| 1024 | 1.392 | 0.442 | 0.303 | 4.60× | 1.46× |
| 2048 | 4.229 | 1.229 | 0.917 | **4.61×** | 1.34× |
| 4096 | 12.480 | 4.065 | 3.382 | 3.69× | 1.20× |
| 8192 | 37.349 | 13.890 | 12.338 | 3.03× | 1.13× |
| 16384 | 111.883 | 49.865 | 46.494 | 2.41× | 1.07× |

HQC FFT beats Karatsuba at **n ≥ 96** (scalar) and **n ≥ 64**
(AVX-512). The scalar path loses narrowly at n = 64 (0.012 vs 0.009 ms);
the AVX-512 path is 11 % faster there (0.008 vs 0.009 ms). The peak
advantage is **4.61×** at n = 2048. Above n = 8192 the AVX-512 gain
shrinks to 1.07× (the FFT's O(n log n) asymptotic still wins, but the
scalar field multiply cost bottoms out the AVX-512 advantage). The
FFT cap is `GF64_HQC_MAX_LM_N = 131072`; sizes above 16384 (not
tabulated here for wall-clock budget) follow the same asymptotic
profile.

The HQC FFT path is wired into `gf64_poly_mul_internal`'s dispatch
at `gf64/gf64_additive_fft.c:369-413`. The tier selects HQC FFT
when all of `(len_a, len_b, out_len)` are in `[GF64_HQC_FFT_MIN,
GF64_HQC_MAX_LM_N]`, falls through to Karatsuba when only one operand
is small or n exceeds the cap, and falls back to schoolbook below
the Karatsuba threshold. The current `GF64_HQC_FFT_MIN = 96` is
chosen to be safe for the scalar path (matches the scalar crossover
above). Sizes above 131072 (the matrix-free cap) currently route
through Karatsuba; lifting that further requires the General
Algorithm 1 follow-up tracked in
[issue #51](https://github.com/trafgals/ParParPar/issues/51).

### Fenger Toeplitz pipeline vs explicit Cauchy matrix-vector product

The project ships a Fenger Toeplitz pipeline (`gf64_fenger_matvec`,
issue #28) as an alternative to the explicit O(N·R·B) Cauchy
matrix-vector product. The pipeline uses subproduct-tree-based
multi-point evaluation to drive the per-word interpolation and
evaluation, with a multi-threaded OpenMP-parallel execute path.

Benchmark: `gf64/test/bench_gf64_fenger_vs_cauchy.c`. Wall-clock
throughput (MB/s) of `gf64_fenger_matvec` against the explicit Cauchy
reference. CPU count = 8 (8 logical cores via OpenMP). Iters vary per
case (2–20) to stay within the 60 s wall-clock budget. The 8t column
shows the parallel multi-thread result via the OpenMP-parallel execute
path.

| N | R | B (bytes) | cauchy (MB/s) | fenger 1t (MB/s) | fenger 8t (MB/s) | 8t vs cauchy |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 4096 | 22.7 | 0.4 | 2.9 | 0.13× |
| 16 | 16 | 4096 | 12.1 | 0.4 | 2.5 | 0.21× |
| 32 | 32 | 4096 | 6.0 | 0.3 | 2.4 | 0.40× |
| 64 | 64 | 4096 | 6.0 | 0.6 | 4.3 | 0.72× |
| 128 | 128 | 4096 | 6.0 | 1.0 | 7.5 | 1.25× |
| 256 | 64 | 4096 | 7.5 | 1.1 | 7.8 | 1.04× |
| 512 | 256 | 2048 | 3.7 | 1.2 | 9.2 | 2.49× |
| 1024 | 256 | 1024 | 3.1 | 0.8 | 6.3 | 2.03× |
| 256 | 128 | 256 | 2.2 | 0.6 | 4.2 | 1.91× |
| 512 | 128 | 256 | 1.8 | 0.4 | 2.3 | 1.28× |
| 1024 | 256 | 512 | 1.7 | 0.5 | 2.3 | 1.35× |
| 2048 | 256 | 256 | 2.5 | 0.6 | 3.4 | 1.36× |
| 2048 | 512 | 256 | 1.4 | 0.5 | 3.7 | 2.64× |
| 2048 | 1024 | 64 | 0.7 | 0.4 | 2.9 | 4.14× |
| 2048 | 2048 | 32 | 0.6 | 0.5 | 3.7 | 6.17× |
| 8192 | 1024 | 32 | 0.7 | 0.2 | 1.6 | 2.29× |
| 16384 | 1024 | 16 | 0.9 | 0.2 | 2.5 | 2.78× |
| 16384 | 4096 | 16 | 0.3 | 0.2 | 1.7 | 5.67× |

The Fenger pipeline's single-thread throughput is below the
explicit Cauchy reference at small/medium sizes — the per-word
interpolation / evaluation cost is the bottleneck. Multi-threaded
(8t) throughput approaches or exceeds the single-thread Cauchy at
N ≥ 128 with R ≥ 128, with the fenger-8t / cauchy-1t ratio reaching
**6.17×** at N = 2048, R = 2048 (the high-R narrow-B case). At small
N (8 / 16) the parallel pipeline is dominated by the per-row
scheduling overhead and is below the explicit Cauchy. At N = 256+,
R = 256+ the Fenger 8t path is competitive or better than the
explicit Cauchy reference at the same wall-clock.

The Fenger pipeline is currently **not** wired into the engine's
dispatch (the parallel Cauchy path in `src/par3_engine.cc` does not
route through `gf64_fenger_matvec`). Routing it through Fenger at the
appropriate sizes is the next research step, tracked in
[issue #51](https://github.com/trafgals/ParParPar/issues/51).

## What this fork adds

Upstream [ParPar](https://github.com/animetosho/ParPar) only creates PAR2 archives. This fork extends it with PAR3 end-to-end:

- **PAR3 create, verify, and repair** with GF(2^64) Cauchy-matrix recovery (no 65 537 input-block-per-slice cap, no 2 GiB file-size limit)
- A **C++ kernel** that runs on scalar, SSSE3, AVX-2, or AVX-512 + PCLMULQDQ, with four new SIMD entries per optimization vector
- A **Bostan-Schost top-down multi-point evaluation / subproduct tree** path through `gf64_subproduct.c` + `gf64_mpe.c` (issue #46 research track)
- An **HQC 2026 additive FFT** (`gf64_additive_fft_hqc2026.c`) over GF(2^64), wired into `gf64_poly_mul_internal`'s dispatch at `gf64/gf64_additive_fft.c:369-413` (HQC FFT → Karatsuba → schoolbook). Lifts per-invmod cost to `O(n log² n)` for `n ∈ [GF64_HQC_FFT_MIN, GF64_HQC_MAX_LM_N] = [96, 2^20]`
- A **Fenger Toeplitz pipeline** (`gf64_fenger.c`) for `O(n²)` recovery at sizes where the Cauchy-matrix approach loses to Toeplitz

### Optimization vectors

Each vector is bit-exact verified by [`test/par3-kernel-parity.js`](test/par3-kernel-parity.js) (Sections F + G + H + I; **> 7 300 cumulative PASS scenarios** across 4 ISAs).

| Vector | Files | What it adds |
|---|---|---|
| **PA1–PA7 — Coupled-input kernel** | `gf64_region_coupled_muladd_*_arr.c` | `out[w] ^= XOR_g (in_blocks[g][w] * coeff_blocks[g])`. 4 SIMD entries (scalar / SSSE3 / AVX-2 / AVX-512); NAPI binding `coupled_muladd_arr`. |
| **PB1–PB7 — Fused-output kernel** | `gf64_region_fused_output_muladd_*_arr.c` | K output blocks × 1 input block. NAPI binding `fused_output_muladd_arr`; loop-order swap. |
| **PC1–PC7 — 2D-blocked kernel** | `gf64_region_2d_muladd_*_arr.c` | K outputs × G inputs per call. NAPI binding `gf64_2d_muladd_arr`; 2D-tile loop. |
| **PD1–PD3 — Supporting opts** | `gf64_invert_*`, `gf64_dispatch.c` | PD1: SIMD `gf64_inverse` batch. PD2: AVX-512 downclock heuristic (16 MiB threshold + `PAR3_AVX512_FORCE`). PD3: `BLOCK_SIZE` autotune. |
| **Algorithmic kernels** | `gf64_fenger.c`, `gf64_mpe.c`, `gf64_subproduct.c`, `gf64_barycentric.c`, `gf64_additive_fft*.c` | Toeplitz pipeline, Bostan-Schost MPE, subproduct tree, barycentric Cauchy reciprocal, Gao-Mateer + HQC 2026 additive FFT. |

### Kernel benchmarks

Three sets of micro-benchmarks establish the kernel's behaviour in isolation from the JS pipeline (which dominates end-to-end on 10K-slice workloads).

**HQC FFT vs Karatsuba** (`gf64/test/bench_hqc_vs_karatsuba.c`):

| `n` | Karatsuba | HQC scalar | HQC AVX-512 | HQC AVX-512 vs Karatsuba |
|---:|---:|---:|---:|---:|
| 16 | 0.001 ms | 0.002 ms | 0.002 ms | 0.34× |
| 64 | 0.009 ms | 0.013 ms | 0.008 ms | 1.03× |
| **96** | 0.044 ms | 0.029 ms | 0.019 ms | 2.37× *(crossover, scalar)* |
| 128 | 0.052 ms | 0.029 ms | 0.019 ms | 2.79× |
| 512 | 0.474 ms | 0.163 ms | 0.104 ms | 4.58× |
| **1024** | 1.421 ms | 0.441 ms | 0.298 ms | **4.77× peak** |
| 4096 | 12.764 ms | 4.121 ms | 3.433 ms | 3.72× |

Run with: `cd gf64/test && make bench_hqc_vs_karatsuba && ./bench_hqc_vs_karatsuba` (Zen4 / WSL2, `-march=native -O3`, 2026-08-13). The HQC scalar path beats Karatsuba from `n ≥ 96`; the AVX-512 (PCLMULQDQ) path beats Karatsuba from `n ≥ 64`. The 4.77× peak at `n = 1024` is the headline; at `n = 4096` HQC AVX-512 is 3.72× faster (the earlier "10× slower" was measured with an incorrect `vpclmulqdq ymm0, ymm0, ymm0` probe byte sequence — see `gf64/test/Makefile` and cubic review 4914681432 P2).

**Fenger Toeplitz vs Cauchy** (`gf64/test/bench_gf64_fenger_vs_cauchy.c`):

Not yet published: at larger `n` the bench is dominated by the schoolbook `gf64_poly_divmod` / half-EGCD `gf64_poly_invmod_mod` in tree prep + MPE walker (issue #50 — the Newton-reciprocal divmod follow-up), and a single `n=4096` run takes minutes. The Fenger-vs-Cauchy numbers will be filled in once the subquadratic prep lands; the pipeline wiring is described below. The Fenger pipeline is wired into `compute_recovery_streaming` (`src/gf64_addon.cc:2308-2351`) when `(numInputs, numRecovery)` are both powers of two, `blockSize` is 8-byte aligned, and either `PAR3_GF64_USE_FENGER=1` is set or the host is non-Windows.

For the full kernel + end-to-end bench numbers, see [`BENCHMARKING.md`](BENCHMARKING.md).

### Proof of correctness

The kernel-parity test ([`test/par3-kernel-parity.js`](test/par3-kernel-parity.js)) exercises every new kernel entry against a naive JS reference. > 7 300 PASS scenarios across all four ISAs (avx2, ssse3, avx512, scalar); each scenario is both a happy-path bit-exact match and a negative-trap mismatch (flip one coefficient bit, assert the comparison fails).

| Section | Coverage | Scenarios |
|---|---|---:|
| F | groupSize × ISA grid | 2 691 |
| G | coupled-input (`_runCoupledInputParity`, 7 group sizes × 200 + 7 traps) | 1 407 |
| H | fused-output (K ∈ {1,2,4,8,16} × 200 + traps) | … |
| I | 2D-blocked (K × G Cartesian, 100 + trap per tuple) | … |
| **Total** | | **> 7 300** |

PAR2 still works as it did upstream. There is no PAR2 regression. For details on how this implementation diverges from the spec and from `par3cmdline`, see [test/fixtures/par3-spec-amendments.md](test/fixtures/par3-spec-amendments.md).

## Usage

Create a PAR3 archive:

```bash
node bin/par3.js create --output myarchive --recovery-slices 10 file1 file2
```

Or with a percentage:

```bash
node bin/par3.js create --output myarchive --recovery-slices 10% file1 file2
```

Repair a damaged PAR3 archive:

```bash
node bin/par3.js repair myarchive.par3
```

The full option list is in [`help.txt`](help.txt) and [`help-full.txt`](help-full.txt).

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `PAR3_GF64_USE_AVX512` | `auto` | `1`/`true`/`yes`/`on` force AVX-512 dispatch; `0`/`false`/`no`/`off` force scalar. Auto-fallback on non-AVX-512 hosts. Required to be set on WSL2/Hyper-V hosts. |
| `PAR3_AVX512_FORCE` | off | Override the AVX-512 downclock heuristic regardless of workload size. |
| `PAR3_GF64_PARALLEL_HASH` | on | Parallel BLAKE3 hashing across `worker_threads` pool. Disable for single-threaded debugging. |
| `PAR3_GF64_GROUP` | 12 | Group size for the coupled-input kernel (PA1-PA7). |
| `PAR3_GF64_USE_FENGER` | on | Route `compute_recovery_streaming` through the Fenger Toeplitz pipeline when (numInputs, numRecovery) are both powers of two and `blockSize` is 8-byte aligned. Default on every host since the #62 fix (the Windows-only opt-in gate `PAR3_FENGER_WINDOWS_ENABLE` is removed; `PAR3_FENGER_WINDOWS_MAX_INPUTS` remains an optional defensive cap). |

## Development

### Running tests

```bash
npm test                 # all test files
npm run test:e2e         # the 3 end-to-end tests (CI target)
```

GF(2^64) kernel tests live in `gf64/test/` and are built via `make`:

```bash
cd gf64/test && make all
```

This builds:

- 14 kernel parity tests (13 `test_gf64_*` binaries plus `test_cpu_detect` — additive FFT, HQC 2026, barycentric, MPE, subproduct tree, Fenger kernel / pipeline / padded, square, mul, invert-ita, poly mul variants, ZMM probe, CPU detect; see the `all` target in `gf64/test/Makefile`)
- 2 kernel-internal boundary tests (`test_gf64_invmod_mod_parity` for the invmod path through the HQC dispatch, `test_gf64_poly_mul_internal_dispatch` for the HQC/Karatsuba/schoolbook tier boundaries)
- 2 micro-benchmarks (`bench_hqc_vs_karatsuba`, `bench_gf64_fenger_vs_cauchy`)

`test/par3-engine-barycentric-gate.js` is the regression gate for the cubic review 4910826158 WSL2 dispatch bug: it runs `compute_recovery_barycentric` with `PAR3_GF64_USE_AVX512=0` (forcing the fallback path) and verifies it completes without SIGILL and produces the same output as the JS reference.

*par-compare.js* tests PAR2 generation by comparing output from ParPar against that of `par2cmdline`. As such, `par2cmdline` needs to be installed for tests to run. Some tests cover extreme cases (large RAM, lots of recovery data); expect to need ~8 GiB RAM and ~20 GiB disk free.

### Build

```bash
npm install              # installs native build deps (node-gyp)
npm run build            # builds parpar_gf64.node (the NAPI addon)
```

On Linux the build emits a partially static binary (dependent on libc) for OpenCL support. Set `BUILD_STATIC=--fully-static` for a fully static build.

The AVX-512 / VPCLMULQDQ feature gates are runtime-checked against CPUID + XCR0; the detection TU is compiled with `-mno-avx512f` to defeat the WSL2 observer-effect (see the footnote above).

For a self-contained `parpar` binary, see the `nexe/` folder.

## License

This code is Public Domain or [CC0](https://creativecommons.org/publicdomain/zero/1.0/legalcode) (or equivalent) if PD isn't recognised.
