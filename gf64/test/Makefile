# GF64 Test Makefile
# Builds and runs GF(2^64) arithmetic and SIMD tests

CC := gcc
CXX := g++
CFLAGS := -std=c99 -Wall -Wextra -O2 -march=native -D_POSIX_C_SOURCE=200112L -D_GNU_SOURCE
CXXFLAGS := -std=c++11 -Wall -Wextra -O2 -march=native -D_POSIX_C_SOURCE=200112L -D_GNU_SOURCE
LDFLAGS :=

# Source files
GF64_SOURCES := ../gf64_single.c \
	../gf64_region_scalar.c \
	../gf64_region_ssse3.c \
	../gf64_region_ssse3_arr.c \
	../gf64_region_avx2.c \
	../gf64_region_avx2_arr.c \
	../gf64_region_avx512.c \
	../gf64_region_avx512_arr.c \
	../gf64_mul_avx512.c \
	../gf64_square.c \
	../gf64_additive_fft.c \
	../gf64_additive_fft_hqc2026.c \
	../gf64_subproduct.c \
	../gf64_barycentric.c \
	../gf64_mpe.c \
	../gf64_dispatch.c \
	../cpu_detect.c \
	../gf64_invert.c \
	../gf64_invert_ssse3.c \
	../gf64_invert_avx2.c \
	../gf64_invert_avx512.c \
	../gf64_invert_ita.c \
	../gf64_poly_mul_karatsuba.c
GF64_INCLUDES := -I..

# Test sources
TEST_SOURCES := test.c
TEST_ALL_SOURCES := test_all.c
TEST_GF64_MUL_AVX512_SOURCES := test_gf64_mul_avx512.c
TEST_GF64_SQUARE_SOURCES := test_gf64_square.c
TEST_GF64_INVERT_ITA_SOURCES := test_gf64_invert_ita.c
TEST_GF64_ADDITIVE_FFT_SOURCES := test_gf64_additive_fft.c
TEST_GF64_SUBPRODUCT_TREE_SOURCES := test_gf64_subproduct_tree.c
TEST_GF64_BARYCENTRIC_SOURCES := test_gf64_barycentric.c
TEST_GF64_MPE_SOURCES := test_gf64_mpe.c
TEST_CPU_DETECT_SOURCES := test_cpu_detect.c
TEST_GF64_ZMM_PROBE_SOURCES := test_gf64_zmm_probe.c
TEST_GF64_POLY_MUL_KARATSUBA_SOURCES := test_gf64_poly_mul_karatsuba.c
TEST_GF64_FENGER_KERNEL_SOURCES := test_gf64_fenger_kernel.c
TEST_GF64_FENGER_PIPELINE_SOURCES := test_gf64_fenger_pipeline.c
TEST_GF64_FENGER_PADDED_SOURCES := test_gf64_fenger_padded.c
TEST_GF64_FENGER_BATCH_SOURCES := test_gf64_fenger_batch.c
TEST_GF64_INVMOD_MOD_PARITY_SOURCES := test_gf64_invmod_mod_parity.c
TEST_GF64_DIVMOD_PARITY_SOURCES := test_gf64_divmod_parity.c
TEST_GF64_INTERP_PARITY_SOURCES := test_gf64_interp_parity.c
TEST_GF64_ADDITIVE_FFT_HQC2026_SOURCES := test_gf64_additive_fft_hqc2026.c
TEST_GF64_MUL_BATCH_SHARED_SOURCES := test_gf64_mul_batch_shared.c
TEST_GF64_POLY_MUL_INTERNAL_DISPATCH_SOURCES := test_gf64_poly_mul_internal_dispatch.c
BENCH_GF64_FENGER_VS_CAUCHY_SOURCES := bench_gf64_fenger_vs_cauchy.c
BENCH_HQC_VS_KARATSUBA_SOURCES := bench_hqc_vs_karatsuba.c
BENCH_GF64_SUBQUADRATIC_SOURCES := bench_gf64_subquadratic.c

.PHONY: all test clean test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_cpu_detect test_gf64_zmm_probe test_gf64_poly_mul_karatsuba test_gf64_fenger_kernel test_gf64_fenger_pipeline test_gf64_fenger_padded test_gf64_invmod_mod_parity test_gf64_divmod_parity test_gf64_interp_parity test_gf64_additive_fft_hqc2026 test_gf64_mul_batch_shared test_gf64_poly_mul_internal_dispatch bench_gf64_fenger_vs_cauchy bench_hqc_vs_karatsuba bench_gf64_subquadratic

all: test test_all test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_cpu_detect test_gf64_zmm_probe test_gf64_poly_mul_karatsuba test_gf64_fenger_kernel test_gf64_fenger_pipeline test_gf64_fenger_padded test_gf64_invmod_mod_parity test_gf64_divmod_parity test_gf64_interp_parity test_gf64_additive_fft_hqc2026 test_gf64_mul_batch_shared test_gf64_poly_mul_internal_dispatch bench_gf64_fenger_vs_cauchy bench_hqc_vs_karatsuba bench_gf64_subquadratic

test: $(TEST_SOURCES) $(GF64_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_SOURCES) $(GF64_SOURCES)

test_all: $(TEST_ALL_SOURCES) $(GF64_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_ALL_SOURCES) $(GF64_SOURCES)

# Parity test for the AVX-512 vectorized gf64_mul_avx512 (par3-cauchy-fft-kernel T1).
# Compiles only gf64_mul_avx512.c + gf64_single.c (the public scalar reference
# used for cross-checking). Needs AVX-512F + VPCLMULQDQ at runtime.
test_gf64_mul_avx512: $(TEST_GF64_MUL_AVX512_SOURCES) ../gf64_mul_avx512.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_MUL_AVX512_SOURCES) ../gf64_mul_avx512.c ../gf64_single.c

# Parity test for the GF(2^64) Frobenius squaring (par3-cauchy-fft-kernel T2).
# Compiles only gf64_square.c + gf64_single.c (the public scalar SSE2 reference
# used for cross-checking). Needs AVX-512F at runtime.
test_gf64_square: $(TEST_GF64_SQUARE_SOURCES) ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_SQUARE_SOURCES) ../gf64_square.c ../gf64_single.c

# Parity test for the Itoh-Tsujii batched inversion (par3-cauchy-fft-kernel T5).
# Compiles only the four TUs the inverter transitively touches: the new
# gf64_invert_ita.c plus the primitives it depends on (gf64_square.c,
# gf64_mul_avx512.c) and the scalar reference gf64_single.c. Needs
# AVX-512F + VPCLMULQDQ at runtime.
test_gf64_invert_ita: $(TEST_GF64_INVERT_ITA_SOURCES) ../gf64_invert_ita.c ../gf64_square.c ../gf64_mul_avx512.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_INVERT_ITA_SOURCES) ../gf64_invert_ita.c ../gf64_square.c ../gf64_mul_avx512.c ../gf64_single.c

# Scalar + AVX-512 Gao-Mateer-style additive FFT tests (par3-cauchy-fft-kernel T3 + T4).
# The AVX-512 implementation lives in gf64_additive_fft.c (same TU as the
# scalar version) and depends on gf64_mul_avx512.c (T1) for the butterfly
# alpha * odd multiplications. gf64_square.c (T2) supplies the Frobenius
# squaring. gf64_poly_mul_karatsuba.c (T6 helper, PR #42) is required
# because gf64_poly_mul_internal now dispatches into it at n >= 128.
# Needs AVX-512F + VPCLMULQDQ at runtime.
test_gf64_additive_fft: $(TEST_GF64_ADDITIVE_FFT_SOURCES) $(GF64_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_ADDITIVE_FFT_SOURCES) $(GF64_SOURCES)

# HQC 2026 TCHES §2.3 Algorithm 2 (LCH14 addFFT) parity tests. Same scalar
# primitive dependencies as test_gf64_additive_fft above. The HQC TU also
# uses gf64_mul_reference (from gf64_single.c) for butterfly multiplications
# and gf64_inverse (from gf64_invert.c) for butterfly inverse operations.
# Cap is GF64_HQC_MAX_MATRIXFORM_N = 16384 for the matrix-form path and
# GF64_HQC_MAX_LM_N = 131072 for the matrix-free recursive path. Test
# sizes go up to 131072 for the recursive family.
test_gf64_additive_fft_hqc2026: $(TEST_GF64_ADDITIVE_FFT_HQC2026_SOURCES) $(GF64_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_ADDITIVE_FFT_HQC2026_SOURCES) $(GF64_SOURCES)

# Batch-shared mul parity test (issue #59 §4 A3): K products with one
# shared operand must be bit-exact to K sequential
# gf64_addfft64_poly_mul_recursive_scratch calls (scalar and AVX-512).
test_gf64_mul_batch_shared: $(TEST_GF64_MUL_BATCH_SHARED_SOURCES) $(GF64_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_MUL_BATCH_SHARED_SOURCES) $(GF64_SOURCES)

# FFT-based bottom-up subproduct tree test (par3-cauchy-fft-kernel T6).
# gf64_subproduct.c (T6) depends on gf64_additive_fft.c (T3) for gf64_poly_mul,
# which in turn depends on gf64_mul_avx512.c (T1) and gf64_square.c (T2).
# gf64_poly_mul_karatsuba.c (PR #42 helper) is required because
# gf64_poly_mul_internal now dispatches into it at n >= 128.
# gf64_mpe.c (PR #43, T8a) is required for gf64_poly_divmod / gf64_poly_invmod
# (the Newton-reciprocal divmod path, issue #59 A1). Since A1 the tree build
# no longer calls gf64_poly_invmod_mod (the O(N^2) inverse cache was removed).
# gf64_invert_ita.c (T5) is required because gf64_mpe.c calls
# gf64_invert_ita_one for polynomial modular inverse's leading-coeff invert.
# Needs AVX-512F + VPCLMULQDQ at runtime.
test_gf64_subproduct_tree: $(TEST_GF64_SUBPRODUCT_TREE_SOURCES) $(GF64_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_SUBPRODUCT_TREE_SOURCES) $(GF64_SOURCES)

# Barycentric weight extraction test (par3-cauchy-fft-kernel T7).
# gf64_barycentric.c (T7) depends on the subproduct tree (T6) and the
# Itoh-Tsujii batched inversion (T5). The ITA batch in turn depends on the
# AVX-512 scalar multiply (T1) and the Frobenius squaring (T2). The full
# additive FFT (T3) is needed because gf64_subproduct.c calls gf64_poly_mul,
# which now dispatches into gf64_poly_mul_karatsuba.c (PR #42 helper) at
# n >= 128. gf64_mpe.c (PR #43, T8a) is required for the fast divmod path.
# Needs AVX-512F + VPCLMULQDQ at runtime.
test_gf64_barycentric: $(TEST_GF64_BARYCENTRIC_SOURCES) $(GF64_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_BARYCENTRIC_SOURCES) $(GF64_SOURCES)

# Multi-point evaluation test (par3-cauchy-fft-kernel T8). gf64_mpe.c (T8)
# depends on the subproduct tree (T6) for its evaluation entry point, the
# Itoh-Tsujii batched inversion (T5) for the polynomial inverse seed, and
# the additive FFT's gf64_poly_mul (T3) for the Newton-iteration squaring.
# In turn, those primitives depend on gf64_mul_avx512.c (T1) and
# gf64_square.c (T2). gf64_poly_mul_karatsuba.c (PR #42 helper) is
# required because gf64_poly_mul_internal dispatches into it at n >= 128.
# Needs AVX-512F + VPCLMULQDQ at runtime.
test_gf64_mpe: $(TEST_GF64_MPE_SOURCES) $(GF64_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_MPE_SOURCES) $(GF64_SOURCES)

# Standalone unit test for the gf64_detect_method_internal() dispatch
# algorithm (avx512-wsl2-detect, Todo 3 of pr-26-recovery plan). The test
# exercises both the "ZMM probe returns 1" path (POSIX-positive=1) and
# the "ZMM probe returns 0" path (POSIX-negative=0) in a single binary.
# Compiled with -DGF64_CPU_DETECT_STANDALONE so test_cpu_detect.c skips
# gf64_global.h and brings its own minimal type shims; cpu_detect.c is
# compiled alongside to confirm both files build cleanly. Needs AVX-512F
# at runtime (not required — the test stubs both the CPUID and the ZMM
# probe, so the actual host hardware is irrelevant).
test_cpu_detect: $(TEST_CPU_DETECT_SOURCES) ../cpu_detect.c
	$(CC) $(CFLAGS) -DGF64_CPU_DETECT_STANDALONE $(LDFLAGS) -o $@ $(TEST_CPU_DETECT_SOURCES) ../cpu_detect.c

# Test for the cached one-shot ZMM probe (cubic review 4915648282 P1). The
# test exercises the real release/acquire publish pattern in gf64_zmm_probe()
# without mocking CPUID or the SIGILL probe body — we link the real
# cpu_detect.c (not the -DGF64_CPU_DETECT_STANDALONE compile mode used by
# test_cpu_detect above) and the test simply reads back what the probe
# populated. The probe is host-deterministic (SIGILL or not), so any
# disagreement across calls or threads indicates a real memory-ordering
# bug. The multithreaded test (Test 5) is gated on _POSIX_THREADS so the
# build remains portable to non-POSIX hosts.
test_gf64_zmm_probe: $(TEST_GF64_ZMM_PROBE_SOURCES) ../cpu_detect.c
	$(CC) $(CFLAGS) $(LDFLAGS) -pthread -o $@ $(TEST_GF64_ZMM_PROBE_SOURCES) ../cpu_detect.c

# Karatsuba dispatch test (par3-cauchy-fft-kernel Phase 2a — Todo 5). Tests
# both gf64_poly_mul_karatsuba (the new T6 helper) and the dispatch into it
# from gf64_poly_mul_internal inside gf64_additive_fft.c. The additive FFT
# TU brings in gf64_mul_avx512.c (T1) and gf64_square.c (T2); gf64_single.c
# provides gf64_mul_reference used by both scalar and Karatsuba paths.
test_gf64_poly_mul_karatsuba: $(TEST_GF64_POLY_MUL_KARATSUBA_SOURCES) $(GF64_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_POLY_MUL_KARATSUBA_SOURCES) $(GF64_SOURCES)

# Phase 0 of issue #28 (Fenger Toeplitz pipeline — Todo 8): sanity-check that
# the Fenger decomposition (Lagrange interpolation + subproduct polynomial
# evaluation) produces bit-exact output against the explicit Cauchy
# matrix-vector product at small (N, R, B). Minimal dependencies: scalar
# field multiply + gf64_invert for the per-row division.
test_gf64_fenger_kernel: $(TEST_GF64_FENGER_KERNEL_SOURCES) ../gf64_single.c ../gf64_invert.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_FENGER_KERNEL_SOURCES) ../gf64_single.c ../gf64_invert.c

# Phase 1 of issue #28 (Fenger Toeplitz pipeline — Todo 8): bit-exact parity
# of the Bostan-Schost top-down gf64_fenger_matvec against the explicit
# Cauchy reference. Dependencies mirror test_gf64_mpe (the pipeline's
# per-word interpolation/evaluation) plus gf64_fenger.c, gf64_barycentric.c,
# and gf64_invert.c.
test_gf64_fenger_pipeline: $(TEST_GF64_FENGER_PIPELINE_SOURCES) $(GF64_SOURCES) ../gf64_fenger.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_FENGER_PIPELINE_SOURCES) $(GF64_SOURCES) ../gf64_fenger.c

# K5 (issue #46): bit-exact parity of the PADDED Fenger path (non-power-of-2
# real counts padded up to the next power of 2 with synthetic zero-data
# inputs at a disjoint base) against the explicit Cauchy reference. Same
# source set as test_gf64_fenger_pipeline.
test_gf64_fenger_padded: $(TEST_GF64_FENGER_PADDED_SOURCES) $(GF64_SOURCES) ../gf64_fenger.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_FENGER_PADDED_SOURCES) $(GF64_SOURCES) ../gf64_fenger.c

# PR #49 step 1: contract test for gf64_poly_invmod_mod (half-extended GCD;
# its internal divmod steps route through the fast Newton-reciprocal
# gf64_poly_divmod since issue #59 A1). Verifies g * u == 1 (mod f)
# bit-exactly across degrees {4, 8, 16, 32, 64, 128, 256} plus unbalanced
# (deg_g, deg_f) pairs. Same source set as test_gf64_mpe (the function
# lives in gf64_mpe.c).
test_gf64_invmod_mod_parity: $(TEST_GF64_INVMOD_MOD_PARITY_SOURCES) $(GF64_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_INVMOD_MOD_PARITY_SOURCES) $(GF64_SOURCES)

# Issue #59 A1: parity of the fast Newton-reciprocal gf64_poly_divmod
# against the schoolbook reference gf64_poly_divmod_schoolbook across the
# dispatch boundary (quotient length m = 95/96/97), at FFT-mul sizes
# (up to 4096/2048), and at unbalanced / edge shapes (linear divisor,
# deg_f < deg_g, deg_f == deg_g, constant divisor). Same source set as
# test_gf64_mpe.
test_gf64_divmod_parity: $(TEST_GF64_DIVMOD_PARITY_SOURCES) $(GF64_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_DIVMOD_PARITY_SOURCES) $(GF64_SOURCES)

# Issue #59 A1: contract tests for the derivative-based interpolation
# (gf64_multi_point_interp_internal): round-trip against MPE at N in
# {2..1024}, direct Lagrange-basis construction at N=8, N==1 edge, and
# repeated-call determinism. Same source set as test_gf64_mpe.
test_gf64_interp_parity: $(TEST_GF64_INTERP_PARITY_SOURCES) $(GF64_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_INTERP_PARITY_SOURCES) $(GF64_SOURCES)

# Issue #51 Step 7(b): boundary regression for gf64_poly_mul_internal's
# three-tier dispatch (HQC FFT / Karatsuba / schoolbook). Loops
# (len_a, len_b, out_len) through {95, 96, 127, 128, 1024x64, 4096}
# and pins the dispatch's tier selection per case via the
# gf64_dispatch_counts counters. Also verifies correctness bit-exactly
# against the schoolbook reference. Uses the full GF64_SOURCES set
# (gf64_poly_mul_internal sits in gf64_additive_fft.c and depends on
# the full kernel surface).
test_gf64_poly_mul_internal_dispatch: $(TEST_GF64_POLY_MUL_INTERNAL_DISPATCH_SOURCES) $(GF64_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_POLY_MUL_INTERNAL_DISPATCH_SOURCES) $(GF64_SOURCES)

# Phase 3 microbench (issue #28 — Todo 8): wall-clock throughput of
# gf64_fenger_matvec vs explicit Cauchy reference. Same source set as
# test_gf64_fenger_pipeline. Built with -fopenmp to expose the multi-threaded
# execute path via the prepare/execute/release API.
bench_gf64_fenger_vs_cauchy: $(BENCH_GF64_FENGER_VS_CAUCHY_SOURCES) $(GF64_SOURCES) ../gf64_fenger.c
	$(CC) $(CFLAGS) -fopenmp $(LDFLAGS) -o $@ $(BENCH_GF64_FENGER_VS_CAUCHY_SOURCES) $(GF64_SOURCES) ../gf64_fenger.c

# Issue #59 A3 (word-batched tree passes): bit-exact parity of the
# K-word-batched execute against the single-word reference, driven via
# PAR3_FENGER_BATCH_WORDS. Same source set as test_gf64_fenger_pipeline.
test_gf64_fenger_batch: $(TEST_GF64_FENGER_BATCH_SOURCES) $(GF64_SOURCES) ../gf64_fenger.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_FENGER_BATCH_SOURCES) $(GF64_SOURCES) ../gf64_fenger.c

# HQC FFT vs Karatsuba polynomial multiplication benchmark.
# Establishes GF64_HQC_FFT_MIN crossover — informs Step 5 dispatch wiring.
# HQC FFT vs Karatsuba polynomial multiplication benchmark.
# Establishes GF64_HQC_FFT_MIN crossover — informs Step 5 dispatch wiring.
# The bench only calls the public HQC recursive_scratch family and
# gf64_poly_mul_karatsuba; their transitive deps are gf64_additive_fft_hqc2026.c
# (defines the HQC symbols), gf64_poly_mul_karatsuba.c (defines the
# Karatsuba helper), and the scalar/AVX-512 primitives they call
# (gf64_mul_reference, gf64_mul_avx512, gf64_inverse, gf64_square).
# cpu_detect.c is linked solely for the VPCLMULQDQ probe
# (gf64_has_vpclmulqdq_probe) used to gate the AVX-512 column.
bench_hqc_vs_karatsuba: $(BENCH_HQC_VS_KARATSUBA_SOURCES) ../gf64_additive_fft_hqc2026.c ../gf64_poly_mul_karatsuba.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_invert.c ../gf64_single.c ../cpu_detect.c
	$(CC) $(CFLAGS) $(GF64_INCLUDES) -O3 $(LDFLAGS) -o $@ $(BENCH_HQC_VS_KARATSUBA_SOURCES) ../gf64_additive_fft_hqc2026.c ../gf64_poly_mul_karatsuba.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_invert.c ../gf64_single.c ../cpu_detect.c

# Issue #59 A1 evidence bench: tree build / MPE / interp phase timings at
# N = 4096..2^20 plus the P1 gate shape (bit-exact Fenger vs explicit
# Cauchy at N=131072 / R=4096). Same source set as
# bench_gf64_fenger_vs_cauchy.
bench_gf64_subquadratic: $(BENCH_GF64_SUBQUADRATIC_SOURCES) $(GF64_SOURCES) ../gf64_fenger.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(BENCH_GF64_SUBQUADRATIC_SOURCES) $(GF64_SOURCES) ../gf64_fenger.c

clean:
	rm -f test test_all test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_cpu_detect test_gf64_zmm_probe test_gf64_poly_mul_karatsuba test_gf64_fenger_kernel test_gf64_fenger_pipeline test_gf64_fenger_padded test_gf64_invmod_mod_parity test_gf64_divmod_parity test_gf64_interp_parity test_gf64_additive_fft_hqc2026 test_gf64_mul_batch_shared test_gf64_poly_mul_internal_dispatch bench_gf64_fenger_vs_cauchy bench_hqc_vs_karatsuba bench_gf64_subquadratic
