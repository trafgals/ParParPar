# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository identity and fork policy

**This is a fork of `animetosho/ParPar`** — see `AGENTS.md` for the full targeting rules. Briefly:

- **`origin`** → `trafgals/parparpar` (the only acceptable PR/push target)
- **`upstream`** → `animetosho/ParPar` (read-only reference; never PR, never push)

Upstream only accepts PAR2 + GF(16) work; this fork's PAR3/GF(2^64)/AVX-512 additions have no upstream home. Every PR must target `trafgals/parparpar` — never `animetosho/ParPar`. `git remote -v` is the verification step before any push.

## Architecture (three layers)

1. **`gf64/`** — native GF(2^64) arithmetic kernel. Single-header `gf64_global.h`; per-ISA TUs for scalar/SSSE3/AVX2/AVX-512 (see `gf64_dispatch.c` and `gf64_region_*.c`).
   - Field representation: NIST polynomial `x^64 + x^4 + x^3 + x + 1`.
   - Core ops: `gf64_mul_avx512` (8-lane), `gf64_square` (Frobenius SWAR), `gf64_invert_ita` (Itoh-Tsujii batched inversion).
   - Polynomial algos: `gf64_additive_fft.c` (Karatsuba + Van-der-monde dispatch in `gf64_poly_mul_internal`), `gf64_subproduct.c` (T6), `gf64_mpe.c` (T8a eval + T8b interp — Bostan-Schost top-down), `gf64_barycentric.c` (T7), `gf64_invert_ita.c` (T5).
   - **The additive FFT primitive (issue #23 Phase 1) has a known fix.** `PHASE_2B_RESEARCH_2026-07-15.md` documents the original "open problem" verdict; the resolution lives in the sibling repo `trafgals/gf64-fft-research` (`C:\code\trafgals\gf64-fft-research\`) — see that repo's `README.md` and `RESEARCH_SYNTHESIS.md`. The canonical O(N log N) algorithm is **HQC 2026 TCHES §2.3 Algorithm 2** (Chen/Chiu/Peng/Yang, "Accelerating HQC with Additive FFT"), which is functionally identical to LCH14, Chen 2018, hamil 2016. The bug in this repo's `gf64_additive_fft_lch14.c` is a wrong multiplier formula (uses `s_i(v_j)` which collapses to 1 by Cantor recurrence); HQC Alg 2 requires `s_{i-1}(a)` where `a` is the affine shift, `a ∉ V_{i-1}`. Three independent retypings in `gf64-fft-research/probes/` verify it at 100% over GF(2^4). Port effort per the sibling README: 2–5 days for a C engineer with PCLMULQDQ. `gf64_poly_mul_vandermonde.c` is the O(N²) ground-truth oracle; `gf64_additive_fft.c` is the dispatch point where the fix plugs in.
   - **Do not alias `out == a` or `out == b` in any new `gf64_poly_mul*` entry.** The `gf64_poly_mul_padded` in-place footgun is documented in user memory; the fix pattern is caller-owned scratch that the transform runs on, with `out` written via memcpy at the end.

2. **`src/`** — NAPI bindings exposing the kernel to Node. `gf64_addon.cc` registers `gf64_*` entries; `par3_engine*.cc` is the C++ recovery engine. `gf64_stub.cc` is the fallback when the kernel is disabled. `cpuid.h` + `platform.h` handle host feature detection.

3. **`lib/`** — Node frontend. `parpar.js` is the entry; `par3gen.js` is the PAR3 create/repair path; `gf64_js.js` is the JS BigInt fallback when the kernel is absent or disabled.

## Build & test commands

- **Native build (Node-gyp, NAPI module):** `npm install` → `node-gyp rebuild`. The `gf64/` TUs are also built standalone via `gf64/test/Makefile` for C-only microbenches. `binding.gyp` is the C++ build config and is the source of truth for which `.cc`/`.c` files compile into the NAPI module.

- **Native tests (gcc-only, run inside WSL — see WSL section below):**
  - `cd gf64/test && make` builds all 14 test binaries.
  - `make <name>` for one binary (e.g. `make test_gf64_mpe`, `make test_gf64_additive_fft`, `make bench_poly_mul_toom3_vs_karatsuba`).
  - The aggregate `GF64_SOURCES` at line 11 of the Makefile is **stale**; per-test rules at lines 36-153 are the source of truth (each one declares its own link set).

- **JS test suite:** `npm test` runs all `*.js` files in `test/`. `npm run test:e2e` runs only the `e2e-*.js` files. Both require `--openssl-legacy-provider` on newer Node (the npm scripts supply it).

- **Build/test escape hatches:** `PAR3_GF64_USE_AVX512=1` forces AVX-512 dispatch on WSL2/Hyper-V (required because `-march=native` triggers CPUID masking on those hosts). `PAR3_AVX512_FORCE`, `PAR3_GF64_NO_COEFF_CACHE`, `PAR3_GF64_PARALLEL_HASH=0`, and `PAR3_GF64_GROUP` are runtime knobs documented in `README.md` §Throughput.

## WSL constraint (verbatim)

Builds and C microbenches inside `gf64/test/` require gcc/make which are not on the host's PATH. Run them via WSL:

```
wsl -e bash -c "cd /mnt/c/code/trafgals/ParParPar/gf64/test && make <target> && ./<binary>"
```

AVX-512 benchmarks **MUST** be run both inside and outside WSL because the WSL2/Hyper-V CPUID-masking behavior diverges from the host. Issue #17 documents the SIGILL probe in `gf64/cpu_detect.c`.

## Research / planning context

- **Issue [#23](https://github.com/trafgals/ParParPar/issues/23)** — PAR3-create asymptotic parity with PAR2. Phase 1 (additive FFT) has a known fix in the sibling research repo (see Architecture §1); Phases 2 (subproduct + MPE) are complete; Phase 3 (Fenger Toeplitz) is blocked on the FFT landing, not on `firstInput=0` algebra.
- **Issue [#27](https://github.com/trafgals/ParParPar/issues/27)** — Phase 2b/3 follow-up; tracks the remaining items needed to close the 13.5× gap. Acceptance gate (canonical 1 GiB / 10K-slice / 1K-recovery PAR3-create on Zen4 + AVX-512): **≥ 100 MB/s**. Items 1–3 cannot ship without the FFT; PE1 / PE2 / D1–D5 are bit-exact verified by `test/par3-kernel-parity.js` and ship independent of it.
- **Staged delivery plan:** `C:\Users\dimit\.claude\plans\in-https-github-com-trafgals-parparpar-i-sleepy-steele.md` — 7-stage plan with Option C strategy; the LCH14 triage phase is expected to pass once the multiplier bug is fixed (do not re-attempt from scratch).
- **Phase 2b research findings:** `PHASE_2B_RESEARCH_2026-07-15.md` — original "open problem" verdict (now stale). Update / supersede with citations to HQC 2026 + Chen 2018 + hamil 2016 before publishing new FFT work.
- **Sibling research repo:** `trafgals/gf64-fft-research` (`C:\code\trafgals\gf64-fft-research\`) — three single-translation-unit C probes (`probes/test_lch14_variants.c`, `probes/test_gf64_gao_mateer.c`, `probes/test_tower_fft_gf16.c`) each implementing HQC 2026 §2.3 Algorithm 2 with a different affine shift, all passing forward-output and convolution-theorem probes at 100% over GF(2^4) under CI. **Cross-reference this repo before any local re-attempt of the FFT.**
- **Phase 2b design:** `PHASE_2b_3_DESIGN.md` — pre-investigation design doc with the original Path A/B effort estimates (now stale; sibling repo's corrected estimate is 2–5 days).
- **Bench history:** `BENCHMARKING.md` (full bench protocol — tmpfs mount, `taskset -c 0-3`, 3-run median) and `BENCH_RESULTS_*.md` (per-session numbers).

## Conventions specific to this repo

- **`abort()` on malloc failure** is the existing convention in `gf64_alloc_halves` (matches `gf64_additive_fft.c` lines 99-100). New code in `gf64/` should match unless there's a documented reason not to.
- **No new `gf64_dispatch.c` slot** required by default — existing FFT entries use explicit caller-picks-symbol (no runtime ISA dispatch). Mirror this when adding new FFT entries.
- **The `Newton-iteration out_len - i` clamp** in `gf64_poly_mul_internal` (lines 277-296) must be preserved in any new `gf64_poly_mul_*` entry — it is what makes `gf64_poly_invmod` correct.
- **Commit messages reference issue numbers and stage IDs** (e.g. "phase 2a", "T3", "issue #27 §auxiliary"). Use these conventions to keep `git log` navigable.
- **Task-ID prefixes** (v2/v3 max-perf plan + Phase 2b follow-ups): `PA1–PA7` (coupled-input kernel), `PB1–PB7` (fused-output), `PC1–PC7` (2D-blocked), `PD1–PD3` (SIMD inv batch, AVX-512 downclock, BLOCK_SIZE autotune), `PE1–PE2` (CoeffCache auto-bypass, small-R shortcut), `D1–D5` (JS-pipeline: affinity pool, parallel hash, 64-block batching, bulk read, batched output writes). Use these in commit subjects to keep history a navigable plan index.
- **Canonical kernel regression gate:** `test/par3-kernel-parity.js` — exhaustive bit-exact parity between the C kernel and the JS BigInt reference across all four ISA variants (avx2 / ssse3 / avx512 / scalar); > 7 300 cumulative scenarios required to pass before any `gf64_region_*` or `gf64_mul_*` change is mergeable. WSL2/Hyper-V AVX-512 detection is gate-tested by `test/par3-cpu-detect.js` (issue #17).
- **Public composing primitives in `gf64/`:**
  `gf64_mul_reference` (scalar SSE2), `gf64_mul_avx512` (8 lanes, AVX-512F + VPCLMULQDQ), `gf64_square` / `gf64_square_avx512` (Frobenius SWAR), `gf64_invert_ita_one` / `gf64_invert_ita_batch`. There is no public `gf64_mul` scalar symbol — `gf64_mul_reference` is the de-facto scalar mul.

## Related user memory (preserve across sessions)

These are in `C:\Users\dimit\.claude\projects\C--Users-dimit\memory\` and constrain how work in this repo is done:

- `parparpar-fork-policy.md` — only ever target `trafgals/parparpar`; never upstream `animetosho/parpar`.
- `parparpar-repo.md` — fork of `animetosho/ParPar`; `gf64/` native + `lib/` Node + `src/` NAPI; issue #23 working branch.
- `wsl-from-windows-host.md` — WSL Ubuntu 2 is installed; gcc/make live there; AVX-512 benchmarks run both inside and outside WSL.
- `gf64-poly-mul-padded-inplace-footgun.md` — `gf64_poly_mul_padded` memsets `out` before reading inputs; `out == a` breaks the result.
- `gf64-additive-fft-research.md` — additive FFT primitive **was an open research problem** in this fork. **Resolution (2026-07-15):** sibling repo `trafgals/gf64-fft-research` proves the algorithm is settled — HQC 2026 TCHES §2.3 Algorithm 2 (LCH14 addFFT). The memory file is stale; cross-reference the sibling repo before any FFT work.
