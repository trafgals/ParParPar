# GF64 Test Makefile
# Builds and runs GF(2^64) arithmetic and SIMD tests

CC := gcc
CXX := g++
CFLAGS := -std=c99 -Wall -Wextra -O2 -march=native -D_POSIX_C_SOURCE=200112L -D_GNU_SOURCE
CXXFLAGS := -std=c++11 -Wall -Wextra -O2 -march=native -D_POSIX_C_SOURCE=200112L -D_GNU_SOURCE
LDFLAGS :=

# Source files
GF64_SOURCES := ../gf64_single.c ../gf64_region_scalar.c ../gf64_region_ssse3.c ../gf64_region_ssse3_arr.c ../gf64_region_avx2.c ../gf64_region_avx2_arr.c ../gf64_region_avx512.c ../gf64_region_avx512_arr.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_additive_fft.c ../gf64_dispatch.c ../cpu_detect.c ../gf64_invert.c ../gf64_invert_ssse3.c ../gf64_invert_avx2.c ../gf64_invert_avx512.c
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
TEST_GF64_POLY_INVMOD_FFT_SOURCES := test_gf64_poly_invmod_fft.c
TEST_GF64_POLY_MUL_KARATSUBA_SOURCES := test_gf64_poly_mul_karatsuba.c
TEST_GF64_FFT_POLY_MUL_SOURCES := test_gf64_fft_poly_mul.c
TEST_GF64_POLY_MUL_TOOM3_SOURCES := test_gf64_poly_mul_toom3.c

.PHONY: all test clean test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_gf64_poly_invmod_fft test_gf64_poly_mul_karatsuba test_gf64_fft_poly_mul test_gf64_poly_mul_toom3

all: test test_all test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_gf64_poly_invmod_fft test_gf64_poly_mul_karatsuba test_gf64_fft_poly_mul test_gf64_poly_mul_toom3

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

# Barycentric weight extraction test (par3-cauchy-fft-kernel T7). Since
# PR-3 the barycentric step 2 delegates to gf64_multi_point_eval (T8), so
# gf64_mpe.c is now in the dependency set as well. The same chain of TUs
# powers gf64_subproduct.c's call into gf64_poly_mul. Since PR-2a the
# Karatsuba dispatch in gf64_poly_mul_internal must also be linked.
# Needs AVX-512F + VPCLMULQDQ at runtime.
test_gf64_barycentric: $(TEST_GF64_BARYCENTRIC_SOURCES) ../gf64_barycentric.c ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_BARYCENTRIC_SOURCES) ../gf64_barycentric.c ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c

# Multi-point evaluation test (par3-cauchy-fft-kernel T8). gf64_mpe.c (T8)
# depends on the subproduct tree (T6) for its evaluation entry point, the
# Itoh-Tsujii batched inversion (T5) for the polynomial inverse seed, and
# the additive FFT's gf64_poly_mul (T3) for the Newton-iteration squaring.
# In turn, those primitives depend on gf64_mul_avx512.c (T1) and
# gf64_square.c (T2). Since PR-2a gf64_poly_mul_karatsuba.c must be linked
# too. Needs AVX-512F + VPCLMULQDQ at runtime.
test_gf64_mpe: $(TEST_GF64_MPE_SOURCES) ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_MPE_SOURCES) ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c

# PR-1 of the T8 follow-up: cross-check gf64_poly_mul (now FFT-based),
# gf64_poly_mul_padded (new), and gf64_poly_invmod (now FFT-delegating)
# across a wider grid than test_gf64_mpe.c exercises. Built against the same
# TUs as test_gf64_mpe. Since PR-2a, gf64_additive_fft.c references
# gf64_poly_mul_karatsuba (the Karatsuba dispatch in
# gf64_poly_mul_internal), so the Karatsuba TU must be linked here too.
test_gf64_poly_invmod_fft: $(TEST_GF64_POLY_INVMOD_FFT_SOURCES) ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_POLY_INVMOD_FFT_SOURCES) ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c

# Phase 2a of the "beat PAR2" plan: cross-check the Karatsuba
# polynomial-multiplication primitive (gf64_poly_mul_karatsuba.c) against
# the schoolbook reference, including the dispatch path through
# gf64_poly_mul_internal in gf64_additive_fft.c. Built against the same
# TUs as test_gf64_poly_invmod_fft.
test_gf64_poly_mul_karatsuba: $(TEST_GF64_POLY_MUL_KARATSUBA_SOURCES) ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_POLY_MUL_KARATSUBA_SOURCES) ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c

# Phase 2b probe: does the LCH14 additive FFT satisfy the convolution
# theorem (pointwise-multiply in evaluation basis = polynomial convolution)?
# If yes, poly_mul via FFT is a drop-in. The new LCH14 transform in
# gf64_additive_fft_lch14.c implements the Lin-Chung-Han (2014) novel
# polynomial basis with the correct butterfly coefficients.
test_gf64_fft_poly_mul: $(TEST_GF64_FFT_POLY_MUL_SOURCES) ../gf64_additive_fft.c ../gf64_additive_fft_lch14.c ../gf64_poly_mul_karatsuba.c ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_FFT_POLY_MUL_SOURCES) ../gf64_additive_fft.c ../gf64_additive_fft_lch14.c ../gf64_poly_mul_karatsuba.c ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c

# Phase 2a+ parity test: Toom-Cook 3 (Toom 2.3) poly_mul against schoolbook.
# Char-2 safe evaluation points {0, 1, 2, 3, inf} since -1 = 1 in char 2.
# Built against gf64_single.c (scalar reference) only — no AVX-512 deps.
# Karatsuba linked as a Toom-3 base-case fallback so sub-products below the
# Toom-3 threshold use the O(n^1.585) path instead of schoolbook.
test_gf64_poly_mul_toom3: $(TEST_GF64_POLY_MUL_TOOM3_SOURCES) ../gf64_poly_mul_toom3.c ../gf64_poly_mul_karatsuba.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_POLY_MUL_TOOM3_SOURCES) ../gf64_poly_mul_toom3.c ../gf64_poly_mul_karatsuba.c ../gf64_single.c

clean:
	rm -f test test_all test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_gf64_poly_invmod_fft test_gf64_poly_mul_karatsuba test_gf64_fft_poly_mul test_gf64_poly_mul_toom3 gen_toom3_constants bench_poly_mul_toom3_vs_karatsuba
