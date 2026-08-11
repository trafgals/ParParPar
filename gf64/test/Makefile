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
TEST_GF64_POLY_MUL_KARATSUBA_SOURCES := test_gf64_poly_mul_karatsuba.c
TEST_GF64_FENGER_KERNEL_SOURCES := test_gf64_fenger_kernel.c
TEST_GF64_FENGER_PIPELINE_SOURCES := test_gf64_fenger_pipeline.c
TEST_GF64_FENGER_PADDED_SOURCES := test_gf64_fenger_padded.c
TEST_GF64_INVMOD_MOD_PARITY_SOURCES := test_gf64_invmod_mod_parity.c
TEST_GF64_ADDITIVE_FFT_HQC2026_SOURCES := test_gf64_additive_fft_hqc2026.c
BENCH_GF64_FENGER_VS_CAUCHY_SOURCES := bench_gf64_fenger_vs_cauchy.c
BENCH_HQC_VS_KARATSUBA_SOURCES := bench_hqc_vs_karatsuba.c

.PHONY: all test clean test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_cpu_detect test_gf64_poly_mul_karatsuba test_gf64_fenger_kernel test_gf64_fenger_pipeline test_gf64_fenger_padded test_gf64_invmod_mod_parity test_gf64_additive_fft_hqc2026 bench_gf64_fenger_vs_cauchy bench_hqc_vs_karatsuba

all: test test_all test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_cpu_detect test_gf64_poly_mul_karatsuba test_gf64_fenger_kernel test_gf64_fenger_pipeline test_gf64_fenger_padded test_gf64_invmod_mod_parity test_gf64_additive_fft_hqc2026 bench_gf64_fenger_vs_cauchy bench_hqc_vs_karatsuba

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

# FFT-based bottom-up subproduct tree test (par3-cauchy-fft-kernel T6).
# gf64_subproduct.c (T6) depends on gf64_additive_fft.c (T3) for gf64_poly_mul,
# which in turn depends on gf64_mul_avx512.c (T1) and gf64_square.c (T2).
# gf64_poly_mul_karatsuba.c (PR #42 helper) is required because
# gf64_poly_mul_internal now dispatches into it at n >= 128.
# gf64_mpe.c (PR #43, T8a) is required because gf64_subproduct_tree_build
# calls gf64_poly_invmod_mod (defined there) to populate inv_mod_data.
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
# n >= 128. gf64_mpe.c (PR #43, T8a) is required because
# gf64_subproduct_tree_build calls gf64_poly_invmod_mod (defined there).
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

# PR #49 step 1: contract test for gf64_poly_invmod_mod (now using
# Newton-reciprocal inner divmod + Karatsuba-dispatched s_curr update).
# Verifies g * u == 1 (mod f) bit-exactly across degrees {4, 8, 16, 32,
# 64, 128, 256} plus unbalanced (deg_g, deg_f) pairs. Same source set
# as test_gf64_mpe (the function lives in gf64_mpe.c).
test_gf64_invmod_mod_parity: $(TEST_GF64_INVMOD_MOD_PARITY_SOURCES) $(GF64_SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_INVMOD_MOD_PARITY_SOURCES) $(GF64_SOURCES)

# Phase 3 microbench (issue #28 — Todo 8): wall-clock throughput of
# gf64_fenger_matvec vs explicit Cauchy reference. Same source set as
# test_gf64_fenger_pipeline. Built with -fopenmp to expose the multi-threaded
# execute path via the prepare/execute/release API.
bench_gf64_fenger_vs_cauchy: $(BENCH_GF64_FENGER_VS_CAUCHY_SOURCES) $(GF64_SOURCES) ../gf64_fenger.c
	$(CC) $(CFLAGS) -fopenmp $(LDFLAGS) -o $@ $(BENCH_GF64_FENGER_VS_CAUCHY_SOURCES) $(GF64_SOURCES) ../gf64_fenger.c

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

clean:
	rm -f test test_all test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_cpu_detect test_gf64_poly_mul_karatsuba test_gf64_fenger_kernel test_gf64_fenger_pipeline test_gf64_fenger_padded test_gf64_invmod_mod_parity test_gf64_additive_fft_hqc2026 bench_gf64_fenger_vs_cauchy bench_hqc_vs_karatsuba
