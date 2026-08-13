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
| `PAR3_GF64_USE_FENGER` | off on Windows, on elsewhere | Route `compute_recovery_streaming` through the Fenger Toeplitz pipeline when (numInputs, numRecovery) are both powers of two and `blockSize` is 8-byte aligned. |

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
