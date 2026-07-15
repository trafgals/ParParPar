# ParParPar

High-performance PAR3 create and repair with GF(2^64) recovery, written in C++ with a Node.js frontend.

## Throughput

**PAR3-create 1 GiB throughput (Zen4 7800X3D, AVX-512, 1 GiB / 10K slices / 10% recovery, 1 GiB / 1000 slices / 10% recovery):**

| State | 1 GiB / 10K sl | 1 GiB / 1000 sl | Source |
|---|---:|---:|---|
| **PARPAR baseline** (upstream PAR2 GF(16)+GFNI+AVX512) | n/a | **622 MB/s** create | native Windows host, MSVC, par2cmdline parity |
| **Current state** (this fork, PAR3 GF(2^64)) | 28–30 MB/s | **102 MB/s** | env-ceiling on WSL2 / native Windows; v2-4 (matrix-build / file-read overlap) commit |
| **Target state** (issue #27 acceptance gate) | **≥ 100 MB/s** | n/a | 1 GiB / 10K-slice is the canonical PAR3-create workload |

The 10K-slice workload is held at the ~30 MB/s JS-pipeline ceiling: the
C++-only kernel hits ~1097 MB/s on AVX2 (`test/bench/par3-native-bench`),
so kernel throughput is not the limiter. The bottleneck is the O(N²)
matrix-form basisCvt inside `gf64_addfft64` (250× too slow at n=4096);
`PHASE_2c_FINDINGS_2026-07-15.md` documents the recursive Algorithm 1
(Chen 2018 / HQC 2026 §2.3) port required to clear the gate. Tracking
issue: [trafgals/parparpar#30](https://github.com/trafgals/parparpar/issues/30)
at the only acceptable PR target — never upstream `animetosho/ParPar`.

[†] **WSL2 dispatch bug (issue #17):** on WSL2/Hyper-V hosts, `-march=native`
compiles AVX-512 instructions into the binary, which the hypervisor detects
and uses to mask CPUID's AVX-512 feature bits. The avx512-wsl2-detect work
ships a three-layer fix: (1) isolated detection TU `gf64/cpu_detect.c` with a
SIGILL probe for runtime ZMM execution testing, (2) `binding.gyp` builds the
detection TU with `-mno-avx512f` to remove the architectural trigger,
(3) `PAR3_GF64_USE_AVX512` operator escape hatch with `1/true/yes/on` /
`0/false/no/off` / `auto` semantics. Architectural isolation alone is
partial. The operator must set `PAR3_GF64_USE_AVX512=1` to force reliable
AVX-512 dispatch on WSL2/Hyper-V hosts. See [BENCHMARKING.md §5](BENCHMARKING.md)
for the full state and `test/par3-cpu-detect.js` for the regression test.

PAR3 GF(2^64) trades a larger Galois field for a higher recovery-block cap
and unbounded input size. It lifts the 65 537 input-block-per-slice cap and
the 2 GiB file size limit that constrain PAR2. The create path has been
verified end-to-end on a 4.3 GiB archive.

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

### v2 max-perf kernel stack

The v2 max-perf plan shipped four stacked optimization vectors on top of the
existing kernel. Each vector is bit-exact verified by an extended
[`test/par3-kernel-parity.js`](test/par3-kernel-parity.js) (Sections F + G +
H + I; > 7 300 cumulative PASS scenarios across 4 ISAs).

**PA1–PA7 — Coupled-input kernel** (`gf64_region_coupled_muladd_*_arr`):
4 new SIMD entries (scalar / SSSE3 / AVX-2 / AVX-512) implementing
`out[w] ^= XOR_g (in_blocks[g][w] * coeff_blocks[g])`. Each coefficient
pairs with its own input block, matching the engine's actual hot-loop
semantics. Dispatch slot + NAPI binding (`coupled_muladd_arr`); WorkerThread
refactored to stack G-sized groups (G = 12 default, env-overridable via
`PAR3_GF64_GROUP`).

**PB1–PB7 — Fused-output kernel** (`gf64_region_fused_output_muladd_*_arr`):
4 new SIMD entries processing K output blocks against one input block per
call, batched as `outs[k] ^= in[w] * coeff_block_starts[k]` for k ∈ [0..K).
Dispatch slot + NAPI binding (`fused_output_muladd_arr`); loop-order swap in
WorkerThread (outer j, inner K-batch).

**PC1–PC7 — 2D-blocked kernel** (`gf64_region_2d_muladd_*_arr`):
4 new SIMD entries combining the previous two into K outputs × G inputs per
call. Dispatch slot + NAPI binding (`gf64_2d_muladd_arr`); WorkerThread
refactored to 2D-tile loop.

**PD1–PD3 — Supporting optimizations**:
- **PD1**: SIMD `gf64_inverse` batch (Cauchy matrix construction throughput)
- **PD2**: AVX-512 downclock heuristic (`gf64_method_for_workload()` +
  16 MiB threshold + `PAR3_AVX512_FORCE` env override) — avoids Zen4's
  2× frequency drop on small workloads
- **PD3**: `BLOCK_SIZE` autotune (env-gated)

**PE1 — CoeffCache auto-bypass (`PAR3_GF64_NO_COEFF_CACHE`)**: at canonical
1 GiB / 10K-slice workloads the LRU cache for the (R × N) coefficient matrix
holds up to 8 × ~80 MiB = ~640 MiB of pinned DRAM, competing with the muladd
working set for memory bandwidth. The cache provides zero benefit for the
single-archive-creation use case (one entry per `(N, R, fi, fr)` tuple,
touched once). Default behavior: bypass the cache and allocate the matrix
in a call-local buffer when the matrix footprint is ≥ 32 MiB; otherwise keep
the LRU cache warm for repeated small calls. Env override:
`PAR3_GF64_NO_COEFF_CACHE=0` forces the cache on; `=1` forces bypass. See
issue #27 §auxiliary constant-factor paths.

**PE2 — Small-R single-output kernel shortcut**: when `numRecovery ≤ 32`,
`WorkerThread` switches from the fused K=12 2D muladd kernel to the simpler
`gf64_region_muladd_arr` 1D loop (the same shape `ComputeRepairBlocks`
uses). The 2D kernel's per-call setup (outs_ptr / in_blocks_ptr arrays, the
K-tile outer loop, the Kk clamp) costs more than the work it saves when the
output count is small. Threshold is hardcoded at 32 (matching issue #27
§auxiliary guidance); no env gate.

### Proof of correctness

The kernel-parity test
([`test/par3-kernel-parity.js`](test/par3-kernel-parity.js)) exercises all
three new kernel entries against a naive JS reference. Final pass count is
> 7 300 across all four ISAs (avx2, ssse3, avx512, scalar); each scenario is
both a happy-path bit-exact match and a negative-trap mismatch (flip one
coefficient bit, assert the comparison fails). Sections:

- **F** — groupSize × ISA grid (pre-existing, 2 691 cases)
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
