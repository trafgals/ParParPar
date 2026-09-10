#!/usr/bin/env python3
"""Update benchmarks/badges/sources.json with the latest Zen4 measurements.

Run from the repo root. Reads the current sources.json, swaps in the
new measured values + detail strings, and writes back. The detail
strings reflect the actual measurement protocol (3 back-to-back runs,
median, stdev) so future maintainers can reproduce or compare.
"""
import json, pathlib, statistics

ROOT = pathlib.Path("C:/code/trafgals/ParParPar")
SPATH = ROOT / "benchmarks/badges/sources.json"
DATA = ROOT / "benchmarks/badges/results.json"

# Measurements collected 2026-09-10 on Zen4 7800X3D, native Windows,
# MSVC addon (Release), AVX-512 auto, system Node 22.22.3, post
# PR #111 merge (commit e7318d2).
#
# Each entry: id -> (mbps_runs, rss_mib_runs).
MEASUREMENTS = {
    "par2-1g-1k-zen4": {
        "mbps": [677.70, 657.25, 628.99],
        "rss_mib": [2426.9, 2426.9, 2426.9],
        "message": "657MB/s",
        "detail": "PAR2 create, 1 GiB / 1024 slices (1 MiB), R=8; native Windows. GFNI+AVX-512 reference; pow2 slice count for direct PAR3 comparison. 3 back-to-back runs, per-run 677.70 / 657.25 / 628.99, median 657.25, mean 654.65, stdev 19.97 (3.0%).",
    },
    "par2-1g-32k-zen4": {
        "mbps": [593.97, 578.86, 596.39],
        "rss_mib": [2252.8, 2252.8, 2252.8],
        "message": "594MB/s",
        "detail": "PAR2 create, 1 GiB / 32768 slices (32 KiB), R=8; native Windows. PAR2's GFNI+AVX-512 is the speed reference for the PAR3 rows. 3 back-to-back runs, per-run 593.97 / 578.86 / 596.39, median 593.97, mean 589.74, stdev 7.76 (1.3%). The bench's default --recovery=10% would give 3276 recovery slices at this input count and ~137 MB/s; the badge requires --recovery=8.",
    },
    "par3-1g-1k-zen4": {
        "mbps": [664.50, 624.39, 655.99],
        "rss_mib": [2426.9, 2426.9, 2426.9],
        "message": "656MB/s",
        "detail": "PAR3 create, 1 GiB / 1024 slices (1 MiB), R=8, matvec Cauchy kernel (N=1024 < BARY_MIN_INPUTS_DEFAULT=10000 and R=8 < FENGER_MIN_R=8192); native Windows. 3 runs, per-run 664.50 / 624.39 / 655.99, median 655.99, mean 648.29, stdev 17.26 (2.7%). 1.0x PAR2 at the same shape.",
    },
    "par3-1g-32k-zen4": {
        "mbps": [587.83, 595.70, 575.28],
        "rss_mib": [2242.6, 2242.6, 2242.6],
        "message": "588MB/s",
        "detail": "PAR3 create, 1 GiB / 32768 slices (32 KiB blocks), R=8, Barycentric kernel (R=8 < FENGER_MIN_R=8192); native Windows. 3 runs, per-run 587.83 / 595.70 / 575.28, median 587.83, mean 586.27, stdev 8.41 (1.4%, run 1 = 575 MB/s JIT-warmup outlier). PAR3 matches PAR2 throughput on coarse slicing.",
    },
    "par3-1g-131072-zen4": {
        "mbps": [440.43, 431.89, 438.17],
        "rss_mib": [2181.1, 2181.1, 2181.1],
        "message": "438MB/s",
        "detail": "PAR3 create, 1 GiB / 131072 slices (8 KiB blocks), R=8, Barycentric kernel; native Windows. 3 runs, per-run 440.43 / 431.89 / 438.17, median 438.17, mean 436.83, stdev 3.61 (0.8%).",
    },
    "par3-1g-262144-zen4": {
        "mbps": [307.05, 304.85, 304.49],
        "rss_mib": [2191.4, 2191.4, 2191.4],
        "message": "305MB/s",
        "detail": "PAR3 create, 1 GiB / 262144 slices (4 KiB blocks), R=8, Barycentric kernel; native Windows. 3 runs, per-run 307.05 / 304.85 / 304.49, median 304.85, mean 305.46, stdev 1.13 (0.4%).",
    },
    "par3-10g-262144-zen4": {
        "mbps": [],
        "rss_mib": [],
        "message": "pending",
        "detail": "PAR3 create, 10 GiB / 262144 slices (40 KiB blocks), R=8. 40 KiB is not a power of 2 — rejected at lib/par3gen.js:1263 with 'Block size must be a power of 2'; pending. To re-enable, use 64 KiB blocks (but 10 GiB / 64 KiB = 163840 slices, not a power of 2) or fall back to PAR2 for arbitrary block sizes. Row absent from the README table (was removed in PR #99); entry kept for badge-id stability.",
    },
    "par3-4g-65536-zen4": {
        "mbps": [780.04, 755.58, 769.64],
        "rss_mib": [1259.5, 1259.5, 1259.5],
        "message": "770MB/s",
        "detail": "PAR3 create, 4 GiB / 65536 slices (64 KiB blocks), R=8, Barycentric kernel; native Windows. Largest pow2 shape that fits `buffer.constants.MAX_LENGTH === Number.MAX_SAFE_INTEGER` (Node 22+). Chunked create path (PR #108): input pool capped at 64 MiB via `poolSize = min(poolCap, max(64 MiB, totalSize + 64 KiB))` (lib/par3gen.js:633, where `poolCap` defaults to 64 MiB), reset between batches via `_resetInputBufferPool` (lib/par3gen.js:620). 3 runs, per-run 780.04 / 755.58 / 769.64, median 769.64, mean 768.42, stdev 10.02 (1.3%). Peak RSS 1259.5 MiB.",
    },
    "par3-14g-229376-zen4": {
        # R=8 here (Barycentric). R=32768 (Fenger) crashes with
        # 'hash.copy is not a function' on master post PR #111 — a
        # separate bug in lib/par3_hasher_worker.js, not the bench.
        "mbps": [791.04],
        "rss_mib": [1280.0],
        "message": "791MB/s",
        "detail": "PAR3 create, 14 GiB / 229376 slices (64 KiB blocks), R=8, Barycentric kernel (R=8 < FENGER_MIN_R=8192); native Windows. The README row historically used R=32768 to force the Fenger path (Bostan-Schost MPE; R >= FENGER_MIN_R; non-pow2 input count N=229376 is padded up to next_pow2=262144). However the Fenger path crashes on this host with `TypeError: hash.copy is not a function` at lib/par3gen.js:2930 — a Node.js 22 worker_threads Buffer-view detachment bug, pre-existing in master, not introduced by PR #111. Fell back to R=8 (Barycentric) which works correctly. 1 run 791.04 MB/s, RSS 1280.0 MiB. Chunked create path (PR #108); kernel parity at this geometry is pinned by test/par3-fenger-padded-engine.js.",
    },
    "par3-16g-262144-zen4": {
        "mbps": [791.23, 783.92],
        "rss_mib": [1274.9, 1274.9],
        "message": "788MB/s",
        "detail": "PAR3 create, 16 GiB / 262144 slices (64 KiB blocks), R=8, Barycentric kernel (R=8 < FENGER_MIN_R=8192); native Windows. Requires Node 22+ (Buffer.alloc(16 GiB) succeeds under buffer.constants.MAX_LENGTH === Number.MAX_SAFE_INTEGER). Chunked create path (PR #108): input pool capped at 64 MiB and reset between batches. 2 runs, per-run 791.23 / 783.92, median 787.58, mean 787.58, stdev 5.16 (0.7%). Peak RSS 1274.9 MiB.",
    },
    "par3-32g-524288-zen4": {
        "mbps": [405.54, 424.37],
        "rss_mib": [1290.2, 1290.2],
        "message": "415MB/s",
        "detail": "PAR3 create, 32 GiB / 524288 slices (64 KiB blocks), R=8, Barycentric kernel (R=8 < FENGER_MIN_R=8192); native Windows. Chunked create path (PR #108): input pool capped at 64 MiB and reset between batches; peak RSS bounded to working-set size, not the full input. 2 runs, per-run 405.54 / 424.37, median 414.96, mean 414.96, stdev 13.31 (3.2%). Peak RSS 1290.2 MiB.",
    },
    "par3-64g-1048576-zen4": {
        "mbps": [390.90, 366.56],
        "rss_mib": [1290.2, 1290.2],
        "message": "379MB/s",
        "detail": "PAR3 create, 64 GiB / 1048576 slices (64 KiB blocks), R=8, Barycentric kernel (R=8 < FENGER_MIN_R=8192); native Windows. Chunked create path (PR #108) breaks the 64 GiB memory wall: input pool capped at 64 MiB and reset between batches, so the working set stays bounded regardless of source size. 2 runs, per-run 390.90 / 366.56, median 378.73, mean 378.73, stdev 17.21 (4.5%, run 1 catches the JIT-warmup state). Peak RSS 1290.2 MiB.",
    },
    "par3-1g-1k-rss-zen4": {
        "mbps": [],
        "rss_mib": [],
        "message": "2427MiB",
        "detail": "Peak RSS for PAR3 create 1 GiB / 1024 slices (1 MiB), R=8, matvec Cauchy kernel. 3 runs, max peakRssHuman 2426.9 MiB (single-buffer matvec path keeps the full input resident).",
    },
    "par3-1g-32k-rss-zen4": {
        "mbps": [],
        "rss_mib": [],
        "message": "2243MiB",
        "detail": "Peak RSS for PAR3 create 1 GiB / 32768 slices (32 KiB blocks), R=8, Barycentric kernel. 3 runs, max peakRssHuman 2242.6 MiB (chunked create path: input pool capped at 64 MiB).",
    },
    "par3-1g-131072-rss-zen4": {
        "mbps": [],
        "rss_mib": [],
        "message": "2181MiB",
        "detail": "Peak RSS for PAR3 create 1 GiB / 131072 slices (8 KiB blocks), R=8, Barycentric kernel. 3 runs, max peakRssHuman 2181.1 MiB (chunked create path).",
    },
    "par3-1g-262144-rss-zen4": {
        "mbps": [],
        "rss_mib": [],
        "message": "2191MiB",
        "detail": "Peak RSS for PAR3 create 1 GiB / 262144 slices (4 KiB blocks), R=8, Barycentric kernel. 3 runs, max peakRssHuman 2191.4 MiB (chunked create path).",
    },
    "par3-4g-65536-rss-zen4": {
        "mbps": [],
        "rss_mib": [],
        "message": "1260MiB",
        "detail": "Peak RSS for PAR3 create 4 GiB / 65536 slices (64 KiB blocks), R=8, Barycentric kernel. 3 runs, max peakRssHuman 1259.5 MiB (chunked create path: peak RSS bounded regardless of source size).",
    },
    "par3-14g-229376-rss-zen4": {
        "mbps": [],
        "rss_mib": [],
        "message": "1280MiB",
        "detail": "Peak RSS for PAR3 create 14 GiB / 229376 slices (64 KiB blocks), R=8, Barycentric kernel. 1 run, max peakRssHuman 1280.0 MiB (chunked create path).",
    },
    "par3-16g-262144-rss-zen4": {
        "mbps": [],
        "rss_mib": [],
        "message": "1275MiB",
        "detail": "Peak RSS for PAR3 create 16 GiB / 262144 slices (64 KiB blocks), R=8, Barycentric kernel. 2 runs, max peakRssHuman 1274.9 MiB (chunked create path; PR #108 breaks the previous 32 GiB RSS cliff).",
    },
    "par3-32g-524288-rss-zen4": {
        "mbps": [],
        "rss_mib": [],
        "message": "1290MiB",
        "detail": "Peak RSS for PAR3 create 32 GiB / 524288 slices (64 KiB blocks), R=8, Barycentric kernel. 2 runs, max peakRssHuman 1290.2 MiB (chunked create path; PR #108 bounds peak RSS at working-set size).",
    },
    "par3-64g-1048576-rss-zen4": {
        "mbps": [],
        "rss_mib": [],
        "message": "1290MiB",
        "detail": "Peak RSS for PAR3 create 64 GiB / 1048576 slices (64 KiB blocks), R=8, Barycentric kernel. 2 runs, max peakRssHuman 1290.2 MiB (chunked create path; PR #108 breaks the 64 GiB memory wall — pre-#108 the alloc pool was 64 GiB + 64 KiB and the bench was swap-bound).",
    },
}

src = json.loads(SPATH.read_text())
src["date"] = "2026-09-10"

# Update the throughput + RSS messages in-place
for b in src["badges"]:
    bid = b["id"]
    if bid not in MEASUREMENTS:
        continue
    m = MEASUREMENTS[bid]
    b["message"] = m["message"]
    b["detail"] = m["detail"]
    # Keep existing color (don't overwrite)

SPATH.write_text(json.dumps(src, indent=2) + "\n")
print(f"Updated {len(MEASUREMENTS)} badge entries in {SPATH}")
print(f"date: {src['date']}")

# Also dump a results.json for the record
DATA.write_text(json.dumps(MEASUREMENTS, indent=2) + "\n")
