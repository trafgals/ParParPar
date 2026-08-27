# ParParPar

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg?style=flat-square)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Node.js](https://img.shields.io/badge/node-%3E%3D0.10-brightgreen.svg?style=flat-square)](https://nodejs.org)
[![PAR3 Support](https://img.shields.io/badge/format-PAR3%20%2B%20PAR2-blue.svg?style=flat-square)](#why-parparpar)

**High-performance PAR3 create, verify, and repair library & CLI with $\text{GF}(2^{64})$ recovery, written in C++ with a Node.js frontend.**

ParParPar is a next-generation fork of [animetosho/ParPar](https://github.com/animetosho/ParPar). It adds full support for the **PAR3 archive format**, removes the legacy slice and file size limitations of PAR2, and introduces an optimized C++ SIMD engine (AVX-512, AVX-2, and SSSE3) with streaming asynchronous I/O.

---

## Why ParParPar?

PAR2 has been the standard parity archive format for decades, but its 16-bit Galois field ($\text{GF}(2^{16})$) imposes hard limitations on modern, multi-gigabyte datasets. **ParParPar** solves these fundamental bottlenecks with $\text{GF}(2^{64})$ arithmetic and a modern architecture:

| Capability | PAR2 (ParPar / par2cmdline) | PAR3 (ParParPar) |
| :--- | :--- | :--- |
| **Max Recovery Slices** | 32,767 slices (hard spec limit) | **Millions / Unbounded** ($\text{GF}(2^{64})$ Cauchy matrix) |
| **Max Input Slices** | 65,535 slices total | **Unbounded** (no input slice limits) |
| **Archive File Size** | 2–4 GiB practical limit | **Unlimited** (verified on multi-gigabyte/terabyte archives) |
| **Full Lifecycle** | Create only (upstream ParPar) | **Full Create, Verify, and Repair** suite |
| **SIMD Acceleration** | $\text{GF}(2^{16})$ GFNI / AVX-512 | **Native $\text{GF}(2^{64})$ AVX-512, AVX-2, and SSSE3** |
| **Hashing Engine** | MD5 / 16-bit CRC | **Multi-threaded BLAKE3** cryptographic hashing |
| **Backwards Compatibility** | PAR2 only | **Full PAR2 creation preserved** alongside PAR3 |

---

<a id="benchmarks"></a>
<a id="throughput"></a>
## Benchmarks & Performance

ParParPar delivers high-throughput $\text{GF}(2^{64})$ Cauchy-matrix recovery and parallel streaming I/O. Below are measured throughput rates across different workload geometries on a local reference host and automated CI runs:

| Project / Format | Workload | Slice Count | Block Size | Throughput (Zen4 Local) | Throughput (CI Runner) | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **PAR2** (ParPar) | 1 GiB Create ($R=8$) | 1,024 | 1 MiB | [![PAR2 1GB/1k (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar2-1g-1k-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | — | GFNI + AVX-512 reference baseline; pow2 slice count for direct PAR3 comparison |
| **PAR2** (ParPar) | 1 GiB Create ($R=8$) | 32,768 | 32 KiB | [![PAR2 1GB/32k (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar2-1g-32k-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | [![PAR2 1GB/32k (CI)](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Ftrafgals%2Fde34e89a8dda313fba9565f56ba7767b%2Fraw%2Fpar2-1g-32k-ci.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | Max PAR2 slice ceiling |
| **PAR3** (ParParPar) | 1 GiB Create ($R=8$) | 1,024 | 1 MiB | [![PAR3 1GB/1k (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar3-1g-1k-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | — | Coarse slicing; **matvec** kernel (N=1024 < `BARY_MIN_INPUTS_DEFAULT=10000`, falls below the Barycentric gate at this size; also below `FENGER_MIN_R=8192`) |
| **PAR3** (ParParPar) | 1 GiB Create ($R=8$) | 32,768 | 32 KiB | [![PAR3 1GB/32k (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar3-1g-32k-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | [![PAR3 1GB/32k (CI)](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Ftrafgals%2Fde34e89a8dda313fba9565f56ba7767b%2Fraw%2Fpar3-1g-32k-ci.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | Matching PAR2 slice count; PAR3 now matches PAR2 throughput |
| **PAR3** (ParParPar) | 1 GiB Create ($R=8$) | 131,072 | 8 KiB | [![PAR3 1GB/131072 (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar3-1g-131072-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | [![PAR3 1GB/131072 (CI)](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Ftrafgals%2Fde34e89a8dda313fba9565f56ba7767b%2Fraw%2Fpar3-1g-131072-ci.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | $4\times$ beyond PAR2 max slice cap |
| **PAR3** (ParParPar) | 1 GiB Create ($R=8$) | 262,144 | 4 KiB | [![PAR3 1GB/262144 (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar3-1g-262144-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | [![PAR3 1GB/262144 (CI)](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Ftrafgals%2Fde34e89a8dda313fba9565f56ba7767b%2Fraw%2Fpar3-1g-262144-ci.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | Fine-grained recovery ($8\times$ PAR2 cap) |
| **PAR3** (ParParPar) | 4 GiB Create ($R=8$) | 65,536 | 64 KiB | [![PAR3 4GB/65536 (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar3-4g-65536-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | N/A | Large archive creation (largest pow2 shape on Node 20 — V8's 4 GiB single-Buffer cap); requires `PAR3_GF64_INPUT_POOL_SIZE=4294963200` env var to bypass the +64 KiB input-pool headroom collision (lib/par3gen.js:638 — `poolSize = max(64 MiB, totalSize + 64 KiB)`; at this shape `totalSize = 4 GiB` so `poolSize = 4 GiB + 64 KiB`, which overflows V8's `Buffer.alloc` cap of 4 GiB exact by 65,536 B) |
| **PAR3** (ParParPar) | 16 GiB Create ($R=8$) | 262,144 | 64 KiB | [![PAR3 16GB/262144 (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar3-16g-262144-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | N/A | "Magic shape" — measured 2026-08-28 on Node 22.22.3 (V8 Buffer cap lifted at `Buffer.kMaxLength = undefined`); Barycentric kernel ($R=8 < \text{FENGER\_MIN\_R}$); needs ABI-127 addon rebuild |
| **PAR3** (ParParPar) | 32 GiB Create ($R=8$) | 524,288 | 64 KiB | [![PAR3 32GB/524288 (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar3-32g-524288-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | N/A | Multi-archive scale; Barycentric kernel; measured 2026-08-28, peak RSS 32.25 GiB (alloc pool fits in 63 GiB physical RAM with pagefile auto-grow to 116 GiB) |
| **PAR3** (ParParPar) | 64 GiB Create ($R=8$) | 1,048,576 | 64 KiB | [![PAR3 64GB/1048576 (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar3-64g-1048576-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | N/A | "Memory wall" — Barycentric kernel; measured 2026-08-28, **swap-bound** (alloc pool 64 GiB exceeds 63 GiB physical RAM; pagefile auto-grew to 173 GiB to fit; throughput reflects swap-fault overhead, not pure kernel cost — on ≥128 GiB-RAM host expect ~3× the number) |
| **PAR3** (ParParPar) | 32 MiB Repair (8 loss) | 256 | 128 KiB | **4,981 ms** | — | Coupled RHS regression — see issue #90 (was 119 ms pre-#78) |
| **PAR3** (ParParPar) | Matrix Solve (10k geom) | 10,000 | — | **224 ms** ($658\times$ vs JS) | — | Native SIMD matrix solver |

*All throughput measurements in MB/s (higher is better). Zen4 column: measured on AMD Ryzen 7 7800X3D reference host (native Windows, MSVC addon, AVX-512 auto). Short workloads (≤2 s/run, e.g. PAR2 1G/1k, 1G/32k): 5–8 back-to-back runs, median, no warmup. Long workloads (>10 s/run, e.g. PAR3 1G/131k, 1G/262k): 3 measured runs + 1 warmup-discarded run (V8 JIT + buffer-pool warmup); large (4G/65536, 32G/524288, 64G/1048576): 1 warmup + 1 measured (memory-wall probe — see `bench-reports/zen4-{32g,64g}-2026-08-28/` for the dated artifacts). Repair/bench A/B rows: 3-run median. CI column: measured monthly on GitHub Actions shared runners (`ubuntu-latest`). The 16 GiB/262144 row is measured on Node 22.22.3 (V8 Buffer cap lifted in Node 22; Node 20's 4 GiB single-Buffer limit still blocks this shape without an ABI-127 addon rebuild). The 32 GiB/524288 row requires ≥33 GiB system memory (alloc pool + working buffers fit comfortably; pagefile auto-grew from 30 GiB initial to 116 GiB at peak with negligible swap pressure). Separately, the 64 GiB/1048576 row exceeds 64 GiB physical RAM on this 63 GiB host and is swap-bound (the input pool alone is 64 GiB + 64 KiB — strictly larger than the host's physical RAM); Windows pagefile auto-grew to 173 GiB at peak to fit, and the create phase thrashes the input pool through swap, accounting for ~340 s of the 371 s create time. The 64 GiB throughput number reflects this swap-fault overhead, not pure kernel cost — on a host with ≥128 GiB RAM (input pool fully resident), expect ~400-500 MB/s based on the 32 GiB baseline (552 MB/s) × 2× scale penalty for kernel work. Note: the `sources.json` file still carries a `par3-10g-262144-zen4` entry — that row was removed from this table in #99 because its 40 KiB block size is not a power of 2 (`lib/par3gen.js:1270` rejects with `Block size must be a power of 2`; the create engine requires pow2 block sizes for its Toeplitz/Cauchy matrix layouts); the badge stays in `sources.json` for stability of external dashboards. To re-enable a 10 GiB PAR3 row, use 64 KiB blocks (10 GiB ÷ 64 KiB = 163840 slices — not a power of 2) or 80 KiB blocks (131072 slices — but 80 KiB is itself non-pow2), or fall back to PAR2 for arbitrary block sizes. Full protocol and reproduction steps are documented in [BENCHMARKING.md](BENCHMARKING.md). Zen4 badges are endpoint JSONs served from the `feat/ci-benchmark-badge` branch (`benchmarks/badges/<id>.json`); CI badges are served from the public gist [de34e89a](https://gist.github.com/trafgals/de34e89a8dda313fba9565f56ba7767b). The `scripts/update-benchmark-badges.js` helper regenerates the endpoint JSONs from `benchmarks/badges/sources.json` on the badge branch.*

---

## Installation

### From Source

```bash
git clone https://github.com/trafgals/ParParPar.git
cd ParParPar
npm install
```

To install the `par3` and `parpar` CLI commands globally:

```bash
npm link
```

---

## CLI Usage

ParParPar provides an intuitive command-line interface for creating, verifying, and repairing PAR3 archives.

### 1. Create PAR3 Archive

Create a PAR3 archive with 8 recovery slices (pow2):

```bash
node bin/par3.js create --output myarchive --recovery-slices 8 file1.mkv file2.iso  # explicit count: must be a power of 2
```

Create with a percentage of recovery data:

```bash
node bin/par3.js create --output myarchive --recovery-slices 10% /path/to/files/
```

Customize block size:

```bash
node bin/par3.js create --output backup --recovery-slices 5% --block-size 64K data.tar
```

### 2. Verify Archive

Verify the integrity of protected files and recovery packets:

```bash
node bin/par3.js verify myarchive.par3
```

### 3. Repair Damaged Files

Automatically reconstruct damaged or missing data from recovery slices:

```bash
node bin/par3.js repair myarchive.par3
```

### Common Options

| Option | Description |
| :--- | :--- |
| `-o, --output <file>` | Base filename for generated PAR3 archive files |
| `-r, --recovery-slices <N>` | Number of recovery slices to create (integer or percentage e.g., `10%`); the integer form must be a power of 2 (the Fenger/Bostan-Schost fast path requires it) — use a percentage or PAR2 for arbitrary counts |
| `-b, --block-size <size>` | Matrix computation block size (default: `1MB`) |
| `-m, --gf-method <method>` | GF arithmetic method (`auto`, `scalar`, `ssse3`, `avx2`, `avx512`) |
| `-t, --threads <N>` | Number of worker threads |
| `--memory-limit <size>` | Memory limit for chunking |
| `-R, --recurse` | Recurse into directories |
| `-v, --verbose` | Verbose output during processing |
| `-q, --quiet` | Quiet output |
| `--json` | Emit structured JSON output |

*For complete options and syntax, see [help.txt](help.txt) or [help-full.txt](help-full.txt).*

---

## JavaScript / Node.js API

You can easily integrate ParParPar into Node.js applications:

```javascript
const par3 = require('./lib/par3gen');

// Create a PAR3 archive
par3.create(['file1.bin', 'file2.bin'], 'myarchive', {
  recoverySlices: '8%',
  blockSize: 4096
}, function(err) {
  if (err) throw err;
  console.log('PAR3 archive created successfully!');
});

// Repair files
par3.repair('myarchive.par3', './target_dir', {}, function(err, result) {
  if (err) throw err;
  console.log(`Repaired ${result.blocksRepaired} damaged blocks.`);
});
```

---

## Hardware Acceleration & Environment Variables

ParParPar automatically selects the fastest available SIMD instructions (AVX-512, AVX-2, SSSE3, or Scalar). You can fine-tune or override dispatch behavior with environment variables:

| Variable | Default | Description |
| :--- | :---: | :--- |
| `PAR3_GF64_USE_AVX512` | `auto` | Set to `1`/`true` to force AVX-512 SIMD; `0`/`false` to force AVX-2. Useful on virtualized or hypervisor hosts. |
| `PAR3_AVX512_FORCE` | `off` | Set to `2` to unconditionally force AVX-512 execution, bypassing workload size downclock heuristics. |
| `PAR3_GF64_PARALLEL_HASH` | `on` | Enables parallel BLAKE3 hashing across worker threads. Set to `0` for single-threaded debugging. |
| `PAR3_GF64_USE_FENGER` | `on` | Enables the Fenger Toeplitz pipeline for accelerated matrix-vector recovery on power-of-two geometries. |
| `PAR3_FENGER_THREADS` | `auto` | Pin the Fenger multi-thread execute to a specific thread count (positive integer). Bypasses the affinity-aware `GetEffectiveCpuCount` fallback. Useful for deterministic thread-race parity tests and reproducible perf measurement. The 32-thread cap applies to the auto-detected value; a forced value is clamped to `blockSize64` (and `≥ 1`) at the call site to prevent over-subscription. |
| `PAR3_SIMULATED_BUFFER_CAP` | `0` (off) | Inject a synthetic V8 Buffer cap (bytes) to exercise the chunked XOR-combine path on ordinary hardware — REC-body byte-equality is asserted at every caps in `[32768, 50000, 65536, 131072, 262144, 1048576]`. Set to `0` to disable. Useful for CI-only regression tests of the chunked-input path (issue #91). |

---

## Technical Architecture & Reliability

- **$\text{GF}(2^{64})$ Galois Field Engine**: Cauchy matrix recovery algorithm scaled to 64-bit field elements, completely eliminating PAR2's 65k block and 32k slice limits.
- **Multi-ISA SIMD Acceleration**: C++ kernels vectorized for AVX-512 (VPCLMULQDQ), AVX-2, and SSSE3 with pure JavaScript BigInt fallback.
- **Async Streaming Pipeline**: Overlaps disk I/O, BLAKE3 checksum calculations, and SIMD Galois field matrix multiplications across worker threads.
- **Proven Bit-Exact Parity**: Thoroughly verified against mathematical reference implementations across 7,300+ automated test scenarios and all supported ISAs.

---

## Development & Testing

```bash
# Run test suite
npm test

# Run end-to-end tests
npm run test:e2e

# Build native C++ addon
npm install
```

---

## License

This project is licensed under the [Creative Commons Zero v1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) (CC0-1.0) / Public Domain.

Based on the original [animetosho/ParPar](https://github.com/animetosho/ParPar) project by Anime Tosho.
