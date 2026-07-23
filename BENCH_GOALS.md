# BENCH_GOALS — locked upstream baseline + per-wave ceiling

> Single source of truth for PAR3-vs-PAR2 throughput claims. Locked before any optimisation PR opens. Locked upstream SHA + exact flags + exact dataset + exact reference-host result. No looser numbers may be cited downstream.

## Workload matrix

| Workload | Slice count | Recovery count | Source | Goal | Failure mode |
|---|---|---|---|---|---|
| 1 GiB / 10k / 1k | 10 000 | 1 000 | issue #27 | PAR3 end-to-end ≥ **100 MB/s** | blocks merge |
| 1 GiB / 32k / 1k | 32 000 | 1 000 | new beat goal | PAR3 end-to-end **>** upstream PAR2 end-to-end | blocks merge |
| 1 GiB / 100k+ / 1k | 100 000+ | 1 000 | maximise | PAR3 kernel-only ≥ `max(prev-wave × 1.05, PAR2 kernel-only + 10 MB/s)` | re-tune within wave, then escalate |

## Locked upstream PAR2 baseline

| Workload | Upstream commit SHA | Build flags | Dataset | Reference host | Result |
|---|---|---|---|---|---|
| 1 GiB / 1k slice / 10% recovery (≈1000 sl) | `6d8ec6aeae0aefc058c283c35d6a10232a374be7` | upstream default (`GF16`+`GFNI`+`AVX512` on x86, `make -j` on the published build rules) | synthetic 1 GiB random | Zen4 7800X3D, native Windows, MSVC, par2cmdline parity | **622 MB/s create** |
| 1 GiB / 10k / 1k | `6d8ec6aeae0aefc058c283c35d6a10232a374be7` | (same) | (same) | (same) | (TBD by locked baseline run — record actual here before any PAR3 PR opens) |
| 1 GiB / 32k / 1k | `6d8ec6aeae0aefc058c283c35d6a10232a374be7` | (same) | (same) | (same) | (TBD) |
| 1 GiB / 100k / 1k | `6d8ec6aeae0aefc058c283c35d6a10232a374be7` | (same) | (same) | (same) | (TBD) |

## Per-wave tuning ceiling (100k+ workload)

```
ceiling = max(previous_wave_kernel_only * 1.05, upstream_PAR2_kernel_only + 10 MB/s)
```

If a wave's PAR3 kernel-only throughput falls below this ceiling:
1. One re-tune within the wave.
2. If still below: escalate to the plan owner before the wave can merge.

## Comparison methodology (strict)

- End-to-end PAR3-create vs end-to-end upstream PAR2-create, strict `>` (not `>=`) for the 32k beat goal.
- Identical files, identical slice count, identical recovery count, identical host, identical `taskset` affinity, identical build flags, identical dataset.
- Median of 5 runs after 1-run warm-up.
- Kernel-only numbers (separate from end-to-end) reported here as supplementary evidence, NOT as the merge gate.

## PR #26-specific carve-out

PR #26 head `030059d` is **frozen** and **not mergeable**. Its results are not a baseline for any of the above rows. The locked upstream baseline is `animetosho/ParPar` at `6d8ec6aeae0aefc058c283c35d6a10232a374be7`, built and run independently of this fork.