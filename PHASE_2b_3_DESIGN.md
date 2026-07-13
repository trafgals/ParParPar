# Phase 2b and Phase 3 — design notes for closing the PAR2 gap

> Context. PAR2 achieves ~622 MB/s on the canonical 1 GiB / 10 000-slice
> / 1 000-recovery kernel; PAR3-create currently sits at ~270 MB/s (the
> legacy 2D-muladd Cauchy recovery, post-Phase 1A cache integration and
> post-Phase 2a Karatsuba polynomial multiplication). To reach the PAR2
> target we need an asymptotic improvement, not a constant-factor tune —
> i.e. we need to replace O(N·R·B) work with something closer to
> O((N+R)·log²(N+R)) by exploiting polynomial FFTs.
>
> This document is the design reference for the two remaining work items
> in the "beat PAR2" plan: a real FFT for GF(2^64)[x] (Phase 2b), and
> the Fenger Toeplitz decomposition that consumes it (Phase 3).

## Phase 2b — a real FFT for GF(2^64)[x]

### What we have

`gf64_additive_fft.c` (509 lines) implements the Gao-Mateer additive
FFT over GF(2^64). The forward transform splits the polynomial into
even/odd halves, applies the Frobenius map (`x → x²`) to each half,
recurses, and combines with a scalar butterfly. The inverse does 63
Frobenius-un-squarings and applies a round-trip scale. The scalar
path lives alongside an AVX-512 vectorized path that lifts the
element-wise ops into 8-lane SIMD.

The implementation is correct and used by the additive FFT primitives,
but **does not** implement the convolution theorem for arbitrary
GF(2^64) inputs. The probe at `gf64/test/test_gf64_fft_poly_mul.c`
empirically confirms this: forward-FFT + pointwise-multiply + inverse-
FFT against a schoolbook reference produces coefficient-wise junk for
every power-of-two size from 2 to 256. So the existing additive FFT
gives us the right *form* of an O(N log N) transform, but the character
basis is monomial-flavoured rather than evaluation-flavoured — and
without evaluation-flavoured characters, pointwise multiplication does
not commute with the basis change in a way that yields polynomial
convolution.

### What closing the gap requires

There are two viable paths from where the code currently sits to a
working FFT for GF(2^64)[x] polynomial multiplication:

#### Path A — finish the Gao-Mateer evaluation-basis transform

The Gao-Mateer paper proves that an additive FFT over GF(2^k) using a
specific choice of additive characters (one per "sub-coset" of a chosen
subfield GF(2^d) inside GF(2^k)) gives a ring homomorphism between
GF(2^k)[x] / (x^N - 1) and GF(2^k)^N for N = 2^m · 2^d with m+k ≤ some
bound. Implementing this requires:

1. Choose a subfield GF(2^d) of GF(2^64). The most useful sizes are
   d ∈ {8, 16, 32, 64}; smaller subfields give more parallelism but
   less transform size per FFT call.
2. Choose a primitive element β for GF(2^d) / GF(2). Lift to γ in
   GF(2^64) — typically a root of the same primitive polynomial.
3. Define the additive characters χ_u(x) = (-1)^{Tr(x · u)} for u in
   each sub-coset {u_0, u_0 + γ, u_0 + γ², ...}. These are the
   evaluation points.
4. Rewrite the forward recursion to evaluate at these χ_u values
   rather than the current squaring-based basis. The recursion
   structure stays the same; the element-wise operations change to
   Frobenius-on-the-characters, and the butterfly coefficient becomes
   the appropriate γ-power.

The existing `gf64_fft_forward` / `gf64_fft_inverse` are the right
*shape* — a Path A implementation replaces the recursion's element-
wise operations and the butterfly's multiplication constant, keeping
the scalar/AVX-512 code structure intact.

Estimated effort: a careful implementation plus verification against
the existing probe (extended to multiple polynomial degrees and random
seeds) is realistic in **2–4 weeks** of focused work for someone
already familiar with the algebra; longer from cold. The reward is
O(n log n) polynomial multiplication, which cascades through the
T6/T7/T8 pipeline as a drop-in.

#### Path B — subfield NTT via CRT

GF(2^64) admits roots of unity only in GF(2^d) subfields (since
GF(2^64)* has order 2^64 - 1, which is odd, so no 2^k-th roots of unity
exist for k > 0). This rules out multiplicative NTT over GF(2^64).

But GF(2^64) is isomorphic to a tower GF(2) ⊂ GF(2^8) ⊂ GF(2^16) ⊂
... ⊂ GF(2^64). Each sub-extension carries multiplicative roots of
unity of appropriate order (Fermat primes, etc.). A CRT-based approach
decomposes a GF(2^64) element into independent residues under chosen
subfield isomorphisms, runs an NTT in each subfield, recombines via
the CRT basis, and obtains the polynomial convolution result.

The classic Cantor-basis NTT in GF(2)[x] is the simplest instance. For
GF(2^64), the natural place to start is the sub-coset NTT of Lin et al.
or the Cantor basis decomposition specific to GF(2^64).

Estimated effort: comparable to Path A; the algorithmic theory is more
accessible but the constant-factor engineering (managing the subfield
basis tables, the CRT reconstruction steps, the AVX-512 vectorization
across subfields) is heavier.

### Why both paths are research-grade

Both paths require either a custom multiplicative NTT over GF(2^k)
subfields (Path B) or a basis-correct additive FFT over GF(2^k)
(Path A). Neither has a "use the existing code as-is" fast path — the
existing transform is monomial-basis, and the gap to evaluation-basis
is not a constant-factor change but a structural one.

The end-to-end correctness test for either path is the existing probe
(`test_gf64_fft_poly_mul.c`), extended to a wider size grid and to
random seeds.

### Pragmatic sub-targets achievable without full FFT

Even without a working FFT, two concrete algorithmic wins remain on
the polynomial-multiplication primitive:

* **Toom-Cook 3** — extends Karatsuba (O(n^1.585)) to O(n^1.465) by
  evaluating at 5 points instead of 2; ~1.5× speedup over Karatsuba at
  n = 4096. Implementation: ~250 LOC, similar to Karatsuba. Tracked.

* **Toom-Cook 4** — extends to O(n^1.404). ~1.18× over Toom-3, ~1.8×
  over Karatsuba at large n. ~350 LOC. Tracked.

Neither matches FFT but both are concrete algorithmic improvements on
the polynomial-multiplication primitive that would speed up the
T6/T7/T8 invmod-and-MPE work that the engine can use once Phase 2c
becomes algorithmically useful (i.e. once the Barycentric kernel
uses FFT-based MPE for the per-row evaluation).

## Phase 3 — Fenger Toeplitz decomposition

Once Phase 2b is in place, the Cauchy matrix-vector product

    out[r][w] = ⊕_c M[r][c] · in[c][w]       with M[r][c] = 1/(x_c ⊕ y_r)

becomes a polynomial problem. The Fenger 2009 decomposition rewrites
it as two polynomial multiplications and one polynomial division,
all over GF(2^64)[x]. The total cost becomes O(M(N+R) · log(N+R))
where M is the FFT-based poly_mul cost — asymptotically
O((N+R) · log²(N+R)) once you account for the Newton iteration
needed inside the polynomial inverse.

The shape is:

1. Encode the inputs as coefficients of a polynomial F(x) = ⊕_c f_c · L_c(x)
   where L_c are the Lagrange bases for the input points {x_c}.
2. Compute G(x) = F(x) / ∏_c (x ⊕ x_c) via polynomial division in
   GF(2^64)[x] / V(x).
3. Evaluate G at every recovery point y_r, which gives ⊕_c f_c / (y_r ⊕ x_c).
4. Multiply by 1/V(y_r) (a single evaluation of V at each y_r) for
   normalisations.

Steps 1, 3, and 4 use polynomial arithmetic that becomes O((N+R) log(N+R))
once Phase 2b lands. Step 2's division reduces to a multiplication +
truncation under the FFT regime (Newton iteration).

Total asymptotic cost: O((N+R) · log²(N+R)) including the Newton
iteration, vs the legacy O(N·R·B). For N = 10 000 and R = 1 000 this is
about a 10⁵× operations reduction (8·10⁹ → 8·10⁴ at B = 1024),
comfortably matching or beating PAR2's 622 MB/s on the canonical
workload.

### Phase 3 prerequisites

Phase 3 cannot begin in earnest until Phase 2b delivers a working FFT.

## Summary

* **Phase 2b** is the gating work item. Two viable algorithmic paths
  (Gao-Mateer evaluation-basis transform; subfield NTT via CRT). Both
  are multi-week research-grade implementations. The probe at
  `test_gf64_fft_poly_mul.c` documents the precise gap.
* **Phase 2a** (Karatsuba) and **Phase 2c** (Barycentric front-end)
  are completed and shipped; they don't close the parity-vs-PAR2 gap
  on the bench but they keep the polynomial primitives and the engine
  architecture ready for the FFT swap-in.
* **Phase 3** (Fenger Toeplitz) becomes a 1-2 week implementation
  once Phase 2b is in place.

---

## Status update — 2026-07-13

### Phase 2b LCH14 attempt: partial result

A Lin-Chung-Han 2014 (LCH14) additive FFT scaffold was implemented
(commit d4f8b26) against Path A:

* `gf64_cantor_basis.h` — 20 precomputed Cantor basis vectors
  v_i satisfying s_{i+1}^2 + s_{i+1} = s_i (Artin-Schreier recurrence).
  Generated by `gf64/test/gen_cantor_basis2.c` via Gaussian elimination
  over GF(2) with span-checking at each step. Verified s_i(v_i) = 1
  for i = 0..7.
* `gf64_additive_fft_lch14.c` — forward/inverse LCH14 butterflies with
  multipliers mu_j = s_i(v_j) / s_i(v_i) = s_i(v_j).

**Round-trip identity holds:** test_lch14_roundtrip shows
forward+inverse = identity for n = 4 (0/4 diffs).

**Convolution theorem probe (test_gf64_fft_poly_mul) fails for n ≥ 4.**
Only n = 2 passes. The forward transform's output is consistent with
the canonical Gao-Mateer-style recursion but does not evaluate the
polynomial at the basis elements of W_m. Likely culprits:

1. **Multiplier indexing.** The formula
   mu_j = s_i(W_m[j + 2^i]) / s_i(W_m[i]) is plausible (the
   "high-half" element provides the multiplicative constant).
   Confirming this requires symbolic or numerical evaluation against
   the LCH14 paper's worked example.
2. **W_m element correspondence.** My mapping
   `v_j = ⊕_{k: bit_k(j)=1} basis[k]` treats v_0 = 0 (the zero
   element), but s_i(basis[i]) = 1 requires the formula to refer to
   the (i+1)-th Cantor basis vector, which my notation calls
   `basis[i]`. The shifted-index correction is the most likely fix.
3. **Output basis convention.** The forward may produce "novel basis
   coefficients" d_k (where f(x) = ⊕_k d_k X_k(x)) rather than
   evaluations f(v_k). The inverse may then recover them correctly
   (round-trip works), but convolution requires evaluation form.

### Toom-3 reality check

Toom-Cook 3 (commit 860ee66) was implemented as the concrete
algorithmic improvement, but a microbench
(`gf64/test/bench_poly_mul_toom3_vs_karatsuba.c`, scalar reference
multiplier) shows it is **not** a clear win over the existing
in-place Karatsuba:

```
  n   | karatsuba (ms) | toom-3 (ms) | speedup
------+----------------+-------------+--------
  64  |     0.017      |    0.018    |  0.99x
 128  |     0.052      |    0.053    |  1.00x
 256  |     0.157      |    0.157    |  1.00x
 512  |     0.471      |    0.470    |  1.00x
1024  |     1.410      |    2.090    |  0.67x  ← toom-3 LOSES
1536  |     3.817      |    2.523    |  1.51x
2048  |     4.284      |    6.254    |  0.69x  ← toom-3 LOSES
3072  |    11.430      |   10.593    |  1.08x
4096  |    12.969      |   12.190    |  1.06x
```

At the sizes characteristic of the Newton iteration in the T6/T7/T8
pipeline (256 → 512 → 1024), Karatsuba wins or ties. Toom-3's five
sub-products don't dominate Karatsuba's three until the sizes where
the recursion depth is large enough that the lower per-subproduct
arithmetic beats the higher overhead. The implementation is
arithmetically correct (16/16 parity tests pass) and remains useful
as a baseline polynomial-multiplication primitive for future phases
where its crossover kicks in past n = 4096.

### Phase 3: blocked on Phase 2b

The Fenger 2009 Toeplitz decomposition is straightforward once a
working FFT is available; without it, the per-row MPE inside
`gf64_mpe.c` continues to use the in-place Horner scheme, and the
Barycentric engine runs at the post-Phase-2c baseline of ~270 MB/s
on the canonical 1 GiB / 10 000-slice workload (vs PAR2's 622 MB/s).

### Recommended next step

Re-attempt Phase 2b with a known-good reference implementation. Two
options:

1. **Worked-example verification.** Take a small-field instance
   (e.g. GF(2^4)) where the LCH14 transform can be checked by hand,
   verify the multiplier formula and indexing, port the corrected
   algorithm to GF(2^64), and re-run the convolution probe.
2. **Use the Cantor-basis NTT (Path B).** Lift the modulus
   x^{2^64} + x over GF(2)[x] and run an NTT via CRT in a subfield
   like GF(2^8). The smaller field gives a manageable debug target.

Either path is multi-week focused work but is the only algorithmic
route from ~270 MB/s to PAR2's ~622 MB/s on the canonical workload.

### Status update — 2026-07-13 (Vandermonde-FFT milestone, commit ddfdbfd)

A VANDERMONDE-MATRIX additive FFT was shipped as a correctness-grounded
drop-in (gf64_additive_fft_vandermonde.c + gf64_poly_mul_vandermonde
in gf64_additive_fft.h). The forward transform is the evaluation
matrix F[i][j] = v_i^j where v_i are the W_m elements; the inverse is
the matrix inverse via Gaussian elimination. Both F and F^{-1} are
cached per n.

* Conv-theorem probe (n=2..256): **8/8 PASS** (was 1/8 with LCH14).
* Vandermonde poly_mul parity: 11/11 random degrees (up to 511).
* T6/T7/T8 tests still pass after linking the new TU.

The Vandermonde FFT is O(N²) per forward/inverse (matrix-vector multiply
against the cached matrix). This satisfies correctness but NOT the
asymptotic O(N log N) requirement for the Fenger Toeplitz pipeline:

* The Fenger pipeline computes out[r] = F(y_r) / V(y_r) where F is the
  Lagrange interpolant of in[c] * w_c at points x_c.
* T8 (existing) handles the EVALUATE step (F at y_r).
* INTERPOLATE is missing: it requires Bostan-Schost top-down
  reconstruction (or Vandermonde * h, which is O(N²) per slice =
  Cauchy-equivalent).
* With only the Vandermonde FFT, Fenger stays at the same asymptotic
  cost as Cauchy.

Closing the 13.5× gap (per BENCH_RESULTS_PHASE2_2026-07-13) to PAR2's
622 MB/s requires either:
  * an O(N log N) additive FFT (the current Vandermonde is too slow),
  * OR an O(N²) O(N log² N) Bostan-Schost interpolator using
    schoolbook-style divide-and-conquer, OR
  * dropping the FFT path and shipping a constant-factor fix instead.