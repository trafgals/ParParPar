# ParParPar

High-performance PAR3 create and repair with GF(2^64) recovery, written in C++ with a Node.js frontend.

## Throughput

End-to-end throughput numbers below are measured on the workhorse
recipe for this project: **1 GiB / 10% recovery / 4 threads / tmpfs
/ taskset -c 0-3** on Zen4 / WSL2 (Ubuntu 22.04). The throughput
ceiling on this host is ~30 MB/s for the end-to-end pipeline — the
JS layer's NAPI + worker_threads overhead, not the C++ kernel. A
standalone C++-only bench (`test/bench/par3-native-bench`) confirms
the kernel itself is hardware-bound at ~1097 MB/s on AVX-2 — the
end-to-end stack still hits the same ~30 MB/s ceiling. The kernel
benchmarks that isolate the kernel from the pipeline are in
[Kernel benchmarks](#kernel-benchmarks) below.

| Config | Throughput | Notes |
|---|---:|---|
| PAR3 create 1 GiB (default env) | ~30 MB/s | mmap + streaming NAPI + Buffer pool + LRU pool + worker_threads hash + AVX-512 threshold (16 MiB → 256 MiB) + wider SIMD K=2 + parallel Cauchy + software prefetch + isolated detection TU + SIGILL probe |
| PAR3 create 1 GiB (`PAR3_GF64_USE_AVX512=1`) | ~30 MB/s | Operator escape hatch for reliable AVX-512 dispatch on WSL2/Hyper-V hosts. See [†] below. |
| PAR3 create 1 GiB / 1M slices (warmup) | **71.51 MB/s** | The 1 GiB / 1M-slice workload with warmup; the 10K-slice workload is JS-pipeline-bound. |
| Native Windows PAR3, AVX-512 auto, 1 GiB / 1000 sl | **102 MB/s** | MSVC-built `parpar_gf64.node`, Zen4 7800X3D. AVX-512 detected via CPUID+XCR0+`_xgetbv` (MSVC intrinsics, no SIGILL probe). |
| Native Windows PAR3, 1 GiB / 10K sl | 28.3 MB/s | Matches WSL forced-AVX-512 ceiling — confirms the JS-pipeline bound. |
| Native Windows PAR3, repair (1 GiB, 5% loss) | 58.5 MB/s | Repair throughput is higher than create because the recovery path uses pre-computed coefficients. |
| PAR2 reference (same host, 1 GiB / 1000 sl create) | 622 MB/s | PAR3 GF(2^64) Cauchy-matrix overhead is the dominant cost vs PAR2's GFNI+AVX-512. |

[†] **WSL2 dispatch bug (issue #17):** on WSL2/Hyper-V hosts, `-march=native`
compiles AVX-512 instructions into the binary, which the hypervisor detects
and uses to mask CPUID's AVX-512 feature bits. The detection layer
(`gf64/cpu_detect.c`) ships a three-layer fix: (1) SIGILL probe for runtime
ZMM execution testing, (2) `binding.gyp` builds the detection TU with
`-mno-avx512f` to remove the architectural trigger, (3) `PAR3_GF64_USE_AVX512`
operator escape hatch with `1/true/yes/on` / `0/false/no/off` / `auto`
semantics. Architectural isolation alone is partial — the operator must set
`PAR3_GF64_USE_AVX512=1` to force reliable AVX-512 dispatch on
WSL2/Hyper-V hosts. See [BENCHMARKING.md §5](BENCHMARKING.md) for the full
state and `test/par3-cpu-detect.js` for the regression test.

PAR3 GF(2^64) trades a larger Galois field for a higher recovery-block cap
and unbounded input size. It lifts the 65 537 input-block-per-slice cap and
the 2 GiB file size limit that constrain PAR2. The create path has been
verified end-to-end on a 4.3 GiB archive.

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

HQC FFT beats Karatsuba at **n ≥ 64** (scalar and AVX-512). The peak
advantage is **4.61×** at n = 2048. Above n = 8192 the AVX-512 gain
shrinks to 1.07× (the FFT's O(n log n) asymptotic still wins, but the
scalar field multiply cost bottoms out the AVX-512 advantage). The
FFT cap is `GF64_HQC_MAX_LM_N = 2^20`; sizes above 16384 (not
tabulated here for wall-clock budget) follow the same asymptotic
profile.

The HQC FFT path is currently **not** wired into
`gf64_poly_mul_internal`'s dispatch (the dispatch tier at
`gf64_additive_fft.c:335-386` still goes schoolbook → Karatsuba).
The cap-aware query `gf64_hqc_supports_size(n)` is in place and
gated on `GF64_HQC_MAX_LM_N = 2^20`. The optimal crossover
(`GF64_HQC_FFT_MIN`) is **n = 64** from this data;
`GF64_HQC_FFT_MIN = 64` would be a safe dispatch choice. This is the
next research step tracked in
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

| N | R | B (bytes) | cauchy (MB/s) | fenger 1t (MB/s) | fenger 8t (MB/s) | 8t speedup |
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
N ≥ 128 with R ≥ 128, with the speedup growing to **6.17×** at
N = 2048, R = 2048 (the high-R narrow-B case). At small N (8 / 16)
the parallel pipeline is dominated by the per-row scheduling overhead
and is below the explicit Cauchy. At N = 256+, R = 256+ the Fenger 8t
path is competitive or better than the explicit Cauchy reference at
the same wall-clock.

The Fenger pipeline is currently **not** wired into the engine's
dispatch (the parallel Cauchy path in `src/par3_engine.cc` does not
route through `gf64_fenger_matvec`). Routing it through Fenger at the
appropriate sizes is the next research step, tracked in
[issue #51](https://github.com/trafgals/ParParPar/issues/51).

## What this fork adds

Upstream [ParPar](https://github.com/animetosho/ParPar) only creates PAR2
archives. This fork extends it with PAR3:

- PAR3 create, verify, and repair (upstream has none of the three for PAR3)
- GF(2^64) Cauchy-matrix recovery, no 65 537 input-block cap
- Streams files larger than 2 GiB (verified on 4.3 GiB)
- Native AVX-512 / AVX-2 C++ kernel exposed via a NAPI binding, with a JS
  BigInt fallback when the kernel is disabled
- A bit-exact parity test that proves the C++ kernel matches the JS path on
  1 000 randomized inputs across every ISA level

### Kernel stack

The kernel ships four stacked optimization vectors on top of the baseline
1D muladd. Each is bit-exact verified by an extended
[`test/par3-kernel-parity.js`](test/par3-kernel-parity.js) (Sections F + G +
H + I; > 7 300 cumulative PASS scenarios across 4 ISAs).

**Coupled-input kernel** (`gf64_region_coupled_muladd_*_arr`):
4 SIMD entries (scalar / SSSE3 / AVX-2 / AVX-512) implementing
`out[w] ^= XOR_g (in_blocks[g][w] * coeff_blocks[g])`. Each coefficient
pairs with its own input block, matching the engine's actual hot-loop
semantics. Dispatch slot + NAPI binding (`coupled_muladd_arr`); WorkerThread
stacks G-sized groups (G = 12 default, env-overridable via
`PAR3_GF64_GROUP`).

**Fused-output kernel** (`gf64_region_fused_output_muladd_*_arr`):
4 SIMD entries processing K output blocks against one input block per
call, batched as `outs[k] ^= in[w] * coeff_block_starts[k]` for k ∈ [0..K).
Dispatch slot + NAPI binding (`fused_output_muladd_arr`); loop-order swap in
WorkerThread (outer j, inner K-batch).

**2D-blocked kernel** (`gf64_region_2d_muladd_*_arr`):
4 SIMD entries combining the previous two into K outputs × G inputs per
call. Dispatch slot + NAPI binding (`gf64_2d_muladd_arr`); WorkerThread
runs a 2D-tile loop.

**Supporting optimizations**:
- SIMD `gf64_inverse` batch (Cauchy matrix construction throughput).
- AVX-512 downclock heuristic (`gf64_method_for_workload()` + 16 MiB
  threshold + `PAR3_AVX512_FORCE` env override) — avoids Zen4's 2×
  frequency drop on small workloads.
- `BLOCK_SIZE` autotune (env-gated).

### Proof of correctness

The kernel-parity test
([`test/par3-kernel-parity.js`](test/par3-kernel-parity.js)) exercises all
three new kernel entries against a naive JS reference. Final pass count is
> 7 300 across all four ISAs (avx2, ssse3, avx512, scalar); each scenario is
both a happy-path bit-exact match and a negative-trap mismatch (flip one
coefficient bit, assert the comparison fails). Sections:

- **F** — groupSize × ISA grid (2 691 cases)
- **G** — coupled-input (`_runCoupledInputParity`, 7 group sizes × 200 + 7
  traps = 1 407 cases)
- **H** — fused-output (K ∈ {1,2,4,8,16} × 200 + traps)
- **I** — 2D-blocked (`K × G` Cartesian product, 100 + trap per tuple)

PAR2 still works as it did upstream. There is no PAR2 regression.

For details on how this implementation diverges from the spec and from
par3cmdline, see
[test/fixtures/par3-spec-amendments.md](test/fixtures/par3-spec-amendments.md).

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

The full option list is in [`help.txt`](help.txt) and
[`help-full.txt`](help-full.txt).

## Development

### Running Tests

Run all 7 test files with:

```bash
npm test
```

Run only the 3 end-to-end tests (CI target) with:

```bash
npm run test:e2e
```

*par-compare.js* tests PAR2 generation by comparing output from ParPar against that of par2cmdline. As such, par2cmdline needs to be installed for tests to be run. Note that tests will cover extreme cases, including those using large amounts of memory, generating large amounts of recovery data and so on. As such, you will likely need a machine with large amounts of RAM available (preferrably at least 8GB) and reasonable amount of free disk space available (20GB or more recommended) to successfully run all tests.
The test will write several files to a temporary location (sourced from `TEMP` or `TMP` environment variables, or the current working directory if none set) and will likely take a while to complete.

### Building Binary

A basic script to compile the ParPar binary is provided in the *nexe* folder. The script has been tested with NodeJS 12.20.0 and may work on other 12.x.x versions.

1. If you haven’t done so already, do an `npm install` in ParPar’s folder to ensure its dependencies are available
2. Enter the *nexe* folder and do an `npm install` to pull down required build packages (note, nexe requires NodeJS 10 or greater)
3. If desired, edit the variables at the top of *nexe/build.js*
4. Run `node build`. If everything worked, there’ll eventually be a *parpar* or *parpar.exe* binary built.
   If it fails during compilation, enter the *nexe/build/12.20.0* (or whatever version of NodeJS you’re using) and get more info by:
   - Linux: build using the `make` command
   - Windows: build using `vcbuild.bat` followed by build options, e.g. `vcbuild nosign x86 noetw intl-none release static no-cctest without-intl ltcg`

On Linux, this will generate a partially static build (dependent on libc) for OpenCL support. Set the `BUILD_STATIC` environment variable to `--fully-static` if you want a fully static build.

See also the Github Actions [build workflows](.github/workflows).

## License

This code is Public Domain or [CC0](https://creativecommons.org/publicdomain/zero/1.0/legalcode) (or equivalent) if PD isn’t recognised.
