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
| **PAR2** (ParPar) | 1 GiB Create ($R=8$) | 1,000 | 1 MiB | **622 MB/s** | — | GFNI + AVX-512 reference baseline |
| **PAR2** (ParPar) | 1 GiB Create ($R=8$) | 32,768 | 32 KiB | [![PAR2 1GB/32k (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar2-1g-32k-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | [![PAR2 1GB/32k (CI)](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Ftrafgals%2Fde34e89a8dda313fba9565f56ba7767b%2Fraw%2Fpar2-1g-32k-ci.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | Max PAR2 slice ceiling |
| **PAR3** (ParParPar) | 1 GiB Create ($R=8$) | 1,000 | 1 MiB | **98.7 MB/s** | — | Coarse slicing / high efficiency |
| **PAR3** (ParParPar) | 1 GiB Create ($R=8$) | 32,768 | 32 KiB | [![PAR3 1GB/32k (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar3-1g-32k-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | [![PAR3 1GB/32k (CI)](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Ftrafgals%2Fde34e89a8dda313fba9565f56ba7767b%2Fraw%2Fpar3-1g-32k-ci.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | Matching PAR2 slice count |
| **PAR3** (ParParPar) | 1 GiB Create ($R=8$) | 131,072 | 8 KiB | [![PAR3 1GB/131072 (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar3-1g-131072-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | [![PAR3 1GB/131072 (CI)](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Ftrafgals%2Fde34e89a8dda313fba9565f56ba7767b%2Fraw%2Fpar3-1g-131072-ci.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | $4\times$ beyond PAR2 max slice cap |
| **PAR3** (ParParPar) | 1 GiB Create ($R=8$) | 262,144 | 4 KiB | [![PAR3 1GB/262144 (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar3-1g-262144-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | [![PAR3 1GB/262144 (CI)](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Ftrafgals%2Fde34e89a8dda313fba9565f56ba7767b%2Fraw%2Fpar3-1g-262144-ci.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | Fine-grained recovery ($8\times$ PAR2 cap) |
| **PAR3** (ParParPar) | 10 GiB Create ($R=8$) | 131,072 | 80 KiB | [![PAR3 10GB/131072 (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar3-10g-131072-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | [![PAR3 10GB/131072 (CI)](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Ftrafgals%2Fde34e89a8dda313fba9565f56ba7767b%2Fraw%2Fpar3-10g-131072-ci.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | Large archive creation |
| **PAR3** (ParParPar) | 10 GiB Create ($R=8$) | 262,144 | 40 KiB | [![PAR3 10GB/262144 (Zen4)](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2Fpar3-10g-262144-zen4.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | [![PAR3 10GB/262144 (CI)](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Ftrafgals%2Fde34e89a8dda313fba9565f56ba7767b%2Fraw%2Fpar3-10g-262144-ci.json&style=flat-square)](https://github.com/trafgals/ParParPar#throughput) | Large archive, high slice count |
| **PAR3** (ParParPar) | 32 MiB Repair (8 loss) | 256 | 128 KiB | **119 ms** | — | Native Cauchy matrix + coupled RHS |
| **PAR3** (ParParPar) | Matrix Solve (10k geom) | 10,000 | — | **161 ms** ($761\times$ vs JS) | — | Native SIMD matrix solver |

*All throughput measurements in MB/s (higher is better). Zen4 column: measured on AMD Ryzen 7 7800X3D reference host (native Windows, MSVC addon, AVX-512 auto, 3-run median). CI column: measured monthly on GitHub Actions shared runners (`ubuntu-latest`). Full benchmarking protocol and reproduction steps are documented in [BENCHMARKING.md](BENCHMARKING.md).*

---

## Installation

### Via npm

```bash
npm install -g @animetosho/parpar
```

### From Source

```bash
git clone https://github.com/trafgals/ParParPar.git
cd ParParPar
npm install
```

---

## CLI Usage

ParParPar provides an intuitive command-line interface for creating, verifying, and repairing PAR3 archives.

### 1. Create PAR3 Archive

Create a PAR3 archive with 10 recovery slices:

```bash
node bin/par3.js create --output myarchive --recovery-slices 10 file1.mkv file2.iso
```

Create with a percentage of recovery data:

```bash
node bin/par3.js create --output myarchive --recovery-slices 10% /path/to/files/
```

Customize block size and slice distribution:

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
| `-o, --output <file>` | Base name for generated PAR3 archive files |
| `-r, --recovery-slices <N>` | Number of recovery slices to create (integer or percentage e.g., `10%`) |
| `-s, --slice-size <bytes>` | Size of each slice in bytes (or suffix `K`, `M`, `G`) |
| `-b, --block-size <bytes>` | Matrix computation block size (default: 4 KiB) |
| `-m, --memory <bytes>` | Maximum memory limit for buffers and caching |
| `-v, --verbose` | Verbose output during processing |

*For complete options and syntax, see [help.txt](help.txt) or [help-full.txt](help-full.txt).*

---

## JavaScript / Node.js API

You can easily integrate ParParPar into Node.js applications:

```javascript
const par3 = require('@animetosho/parpar/lib/par3gen');

// Create a PAR3 archive
par3.create(['file1.bin', 'file2.bin'], 'myarchive', {
  recoverySlices: '10%',
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
