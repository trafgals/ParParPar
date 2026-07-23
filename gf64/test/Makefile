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
	../gf64_subproduct.c \
	../gf64_barycentric.c \
	../gf64_mpe.c \
	../gf64_dispatch.c \
	../cpu_detect.c \
	../gf64_invert.c \
	../gf64_invert_ssse3.c \
	../gf64_invert_avx2.c \
	../gf64_invert_avx512.c \
	../gf64_invert_ita.c
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

.PHONY: all test clean test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_cpu_detect

all: test test_all test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_cpu_detect

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
# squaring. Needs AVX-512F + VPCLMULQDQ at runtime.
test_gf64_additive_fft: $(TEST_GF64_ADDITIVE_FFT_SOURCES) ../gf64_additive_fft.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_ADDITIVE_FFT_SOURCES) ../gf64_additive_fft.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c

# FFT-based bottom-up subproduct tree test (par3-cauchy-fft-kernel T6).
# gf64_subproduct.c (T6) depends on gf64_additive_fft.c (T3) for gf64_poly_mul,
# which in turn depends on gf64_mul_avx512.c (T1) and gf64_square.c (T2).
# Needs AVX-512F + VPCLMULQDQ at runtime.
test_gf64_subproduct_tree: $(TEST_GF64_SUBPRODUCT_TREE_SOURCES) ../gf64_subproduct.c ../gf64_additive_fft.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_SUBPRODUCT_TREE_SOURCES) ../gf64_subproduct.c ../gf64_additive_fft.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c

# Barycentric weight extraction test (par3-cauchy-fft-kernel T7).
# gf64_barycentric.c (T7) depends on the subproduct tree (T6) and the
# Itoh-Tsujii batched inversion (T5). The ITA batch in turn depends on the
# AVX-512 scalar multiply (T1) and the Frobenius squaring (T2). The full
# additive FFT (T3) is needed because gf64_subproduct.c calls gf64_poly_mul.
# Needs AVX-512F + VPCLMULQDQ at runtime.
test_gf64_barycentric: $(TEST_GF64_BARYCENTRIC_SOURCES) ../gf64_barycentric.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_BARYCENTRIC_SOURCES) ../gf64_barycentric.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c

# Multi-point evaluation test (par3-cauchy-fft-kernel T8). gf64_mpe.c (T8)
# depends on the subproduct tree (T6) for its evaluation entry point, the
# Itoh-Tsujii batched inversion (T5) for the polynomial inverse seed, and
# the additive FFT's gf64_poly_mul (T3) for the Newton-iteration squaring.
# In turn, those primitives depend on gf64_mul_avx512.c (T1) and
# gf64_square.c (T2). Needs AVX-512F + VPCLMULQDQ at runtime.
test_gf64_mpe: $(TEST_GF64_MPE_SOURCES) ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_MPE_SOURCES) ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c

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

clean:
	rm -f test test_all test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_cpu_detect
