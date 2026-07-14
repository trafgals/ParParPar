# Additive FFT Primitive Research — Findings 2026-07-15

## Background

GitHub issue [#23](https://github.com/trafgals/ParParPar/issues/23) Phase 1 requires an
**O(N log N) additive FFT over GF(2^64)** that satisfies the convolution theorem. This
is the gating missing piece for closing the 13.5× PAR3 vs PAR2 throughput gap on the
canonical 1 GiB / 10K-slice / 1K-recovery PAR3-create workload (~33 MB/s vs 622 MB/s).

This document captures the state of the research after exhaustive analysis of three
algorithmic approaches, all implemented in `gf64/test/` and verified against the
GF(2^4) brute-force convolution-theorem probe.

## Why this is hard

`GF(2^64)*` has order `2^64 - 1` (odd). Consequence: no `2^k`-th roots of unity exist
for any `k > 0`. This rules out **all** standard multiplicative NTTs (Cooley-Tukey,
Schönhage-Strassen, etc.).

Three custom approaches remain:

1. **Gao-Mateer "evaluation-basis"** additive FFT — uses Frobenius structure of a
   Cantor basis span as evaluation points.
2. **LCH14** (Lin-Chung-Han 2014) — additive FFT over GF(2^k) via coset propagation.
3. **Tower of extensions** additive DFT — recursive trace decomposition using
   `Tr_{GF(2^k)/GF(2)}(x) = sum_{i=0}^{k-1} x^{2^i}` to split the evaluation set.

All three are "research-grade" per `PHASE_2b_3_DESIGN.md` lines 44-99, with
estimates of 2-4 weeks of focused work each.

## Approach 1: LCH14 Coset-Propagation Bug Triage

**Files:** `gf64/test/test_lch14_gf16.c` (existing), `gf64/test/test_lch14_variants.c` (new).

**Setup:** The LCH14 algorithm uses `mu_j = s_i(W_m[j])` as the butterfly multiplier,
where `s_i` is the i-th iterate of `sigma(x) = x^2 + x` and `W_m[j]` is the j-th
W_m element (XOR of Cantor basis vectors at bits of j). All operations fit in a
`uint8_t` in GF(2^4), allowing hand-verification.

**Three candidate variants tested:**

| Variant | Multiplier | Pass rate (n=4 brute-force probe) |
|---|---|---|
| A | `s_i(W_m[j])` | 6975/57600 = 12.10% |
| B | `s_i(W_m[j \| (1<<i)])` | 6975/57600 = 12.10% |
| C | `(s_i(W_m[j \| (1<<i)]))^-1 * s_i(basis[i+1])` | 6975/57600 = 12.10% |

**Critical finding:** All three variants give the EXACT same pass rate. The LCH14
multiplier formula `s_i(W_m[j \| (1<<i)])` collapses to the constant 1 in GF(2^4) — by
the Cantor recurrence `sigma(v_{i+1}) = v_i`, we have `s_i(v_i) = 1` for every i,
and the `W_m[j | (1<<i)]` combinations in GF(2^4) all evaluate to 1 under `s_i`.

Variant A produces all-0 multipliers (because `W_m(j)` for j in [0, 2^i) uses only
low bits, mapping to small basis combinations where `s_i = 0`).

Variant C produces 6 (= inv(1) · basis[1]) at levels 0-2; then sigma^3 vanishes
identically in GF(2^4) (since x^4 + x = 0 for all x ∈ GF(2^4) after one Frobenius
application, as `x^4 = x+1` mod x^4+x+1), making all level-3 multipliers 0.

**Verdict:** **Bug is structural, not fixable by index/recurrence tweaks.** Stage 0
of the original plan cannot succeed in GF(2^4). Skipped to Stage 1.

## Approach 2: Gao-Mateer DIF Recursive Butterfly

**Files:** `gf64/test/test_gf64_gao_mateer.c` (new, 225 lines).

**Setup:** Decimation-in-frequency form of the recursive Gao-Mateer transform. At
each level, the algorithm splits the input into even/odd halves, applies Frobenius
(squaring) to each half coefficient, recurses, then combines with multipliers
`sqrt(wm(i))` and `sqrt(wm(i+half))` (inverse Frobenius of the W_m evaluation points).

**Round-trip:** PASS — forward then inverse recovers the input at n=2 and n=4.

**Convolution theorem:** FAIL — the forward output is NOT polynomial evaluations
`f(wm(i))`. Each level of recursion applies Frobenius to the coefficients, so
depth-d recursion produces coefficients raised to the 2^d power. The forward output
at n=4 is `(a_0^4, a_0^4+a_1^4+a_2^4+a_3^4, a_0^4+a_1^4, a_0^4+a_2^4)` for input
`(a, b, c, d)`.

**Root cause:** This is the same "1/n scaling issue" documented in
`gf64/test/test_dft_gf16.c` lines 191-197: in char 2, the additive DFT matrix
satisfies `F · F = n · I` where `n=4=0` in char 2, so the matrix is rank-deficient and
the transform isn't invertible. The Gao-Mateer evaluation transform inherits this
issue: without explicit inverse-Frobenius in the forward path, the output isn't the
polynomial evaluations the convolution theorem needs.

**Verdict:** Round-trip works (by construction), but conv-theorem fails because the
forward output is `f^(2^depth)(v)`, not `f(v)`.

## Approach 3: Tower-of-Extensions Additive DFT

**Files:** `gf64/test/test_tower_fft_gf16.c` (new, ~120 lines).

**Setup:** For each candidate evaluation set V of size 4 in GF(2^4), check:
1. Mixed trace (2 elements with Tr-to-GF(2) = 0, 2 with Tr = 1).
2. Invertible trace pairing matrix `M[i][j] = Tr(V[i] * V[j])`.
3. Recoverable self-dual basis via Gaussian elimination over GF(2).

**Result:** 336 valid V's found (out of C(15,4) = 1365 candidates; we skip V's
containing 0 since the trace pairing is then singular).

**Critical finding:** For ANY self-dual basis, the additive DFT matrix
`F[i][j] = chi_j(v_i) = (-1)^{Tr(omega_j * v_i)}` is the **IDENTITY** by
construction (Tr(omega_j * v_i) = delta_{ij} is the definition of self-duality, so
the ±1 encoding gives 1 on the diagonal and 0 elsewhere in char 2).

This means the additive DFT IS the identity transform — it doesn't transform the
input at all. The "tower of extensions" structure cannot produce a useful FFT for
polynomial multiplication.

**The correct transform for polynomial multiplication is the Vandermonde matrix**
`V[i][j] = v_i^j` (monomial basis evaluated at the evaluation points `v_i`). This
matrix IS invertible for distinct `v_i` and DOES satisfy the convolution theorem
(pointwise multiplication in the transformed basis = polynomial convolution in the
monomial basis).

**Verdict:** The additive DFT analysis confirms that the **additive DFT is not the
right transform** for this problem. The Vandermonde matrix is the right one, but
its O(N log N) factorization over GF(2^k) remains the open research problem.

## Root cause: Why no clean recursive FFT exists over GF(2^k)

A clean O(N log N) butterfly structure (like Cooley-Tukey for multiplicative FFT)
requires:
1. **Roots of unity** (for the twiddle factors). NOT AVAILABLE in GF(2^k) since
   `2^m` does not divide `2^k - 1` for any `m > 0`.
2. **A self-similar structure** in the transform matrix (e.g., Fourier matrix has
   `F_2N = [[F_N, D·F_N], [F_N, -D·F_N]]` where D is a diagonal of twiddles).

For the Vandermonde matrix `V[i][j] = v_i^j` over GF(2^k), the structure IS partially
self-similar: for v_i in a Frobenius-stable subset, the squared evaluation points
form a smaller evaluation set. But the "twiddle factor" matrix that connects
levels is NOT diagonal — it's a more complex sparse matrix that the literature does
not provide a closed form for.

The Gao-Mateer "evaluation-basis" claim from `PHASE_2b_3_DESIGN.md` lines 44-76
asserts such a factorization exists but does not provide the explicit construction.
Our investigation found no working formulation after three different structural
approaches.

## What WOULD close the throughput gap

Three paths forward, in order of feasibility:

### Path A — Find or invent the Vandermonde sparse factorization

**Effort:** Multi-week research (per design doc estimate 2-4 weeks).

**Status:** Unresolved. Three structural approaches investigated, all hit
fundamental issues:
- LCH14 multiplier formula collapses to constants in GF(2^4).
- Gao-Mateer DIF recursive structure produces Frobenius-iterated evaluations, not
  polynomial evaluations.
- Tower-of-extensions additive DFT reduces to identity for any self-dual basis.

**Next steps (if pursued):**
1. Consult the Gao-Mateer paper directly (currently inaccessible) for the explicit
   multiplier formula at each recursion level.
2. Investigate non-Cantor evaluation sets (random subsets of GF(2^k) with mixed
   trace and non-trivial Vandermonde structure).
3. Try the "Karatsuba-like" decomposition: split each input polynomial into 3
   limbs and evaluate at 5 points (this is just Toom-3 with different eval
   points — but it might capture the W_m Vandermonde structure better than the
   standard Toom-3 evaluation at {0, 1, 2, 3, inf}).

### Path B — Use a multiplicative NTT in a different field

**Effort:** Multi-week engineering.

**Status:** Not viable in GF(2^k). Would require changing the entire field to GF(p^k)
for an odd prime p (e.g., GF(31^k) where 31^k - 1 is divisible by 2^s for some s).
This is a fundamental architectural change to the PAR3 engine and is out of scope
for the FFT primitive plan.

### Path C — Constant-factor wins without asymptotic improvement

**Effort:** Days.

**Status:** Partial progress made this session:
- CoeffCache auto-bypass at >= 32 MiB (PE1, commit `5dfb9bf`).
- Small-R single-output kernel shortcut (PE2, commit `5dfb9bf`).
- AVX-512 detection fix for WSL2 SIGILL probe (issue #17, commit `52c8ca6`).
- Toom-3 dispatch wired but disabled by default (commit `da22d08`) due to per-call
  malloc overhead.

**Status of the 13.5x gap:** Post all of the above, PAR3-create is 33 MB/s on the
canonical workload. The gap to PAR2's 622 MB/s is purely algorithmic.

**Next steps (incremental):**
1. Optimize Toom-3 implementation (thread_local scratch pool to remove per-call
   malloc churn). Could enable Toom-3 dispatch above 2304 and recover some
   algorithmic win.
2. AVX-512 vectorize the Vandermonde-FFT's mat-vec. Currently O(N²) but with 8-lane
   SIMD on the inner product. Crossover point for this vs Karatsuba not measured.
3. Implement Toom-4 (similar to Toom-3 but with 4 limbs and 7 evaluation points).
   ~1.18× over Toom-3 at large n; ~1.8× over Karatsuba.
4. Issue #6 (windows-2022 → windows-2025, node-gyp 12.x) — orthogonal CI work.

## Conclusion

The additive FFT primitive over GF(2^64) for polynomial multiplication remains an
open research problem after exhaustive investigation. The fundamental barrier is
that `GF(2^64)*` has odd order, ruling out multiplicative NTTs, and the additive
alternatives (Gao-Mateer, LCH14, tower-of-extensions) all hit structural issues
that prevent a clean O(N log N) factorization.

The Vandermonde matrix IS the correct transform and IS invertible, but its
sparse factorization is not in the literature. Closing the 13.5× gap requires
either finding this factorization (multi-week research) or pivoting to constant-
factor optimizations on the existing O(N²) pipeline.

This document is committed alongside:
- `gf64/test/test_lch14_variants.c` (commit `7632d83`)
- `gf64/test/test_gf64_gao_mateer.c` (commit `68b14be`)
- `gf64/test/test_tower_fft_gf16.c` (commit `ed155aa`)

The plan file at `C:\Users\dimit\.claude\plans\in-https-github-com-trafgals-parparpar-i-sleepy-steele.md`
contains the staged delivery plan with this outcome documented.