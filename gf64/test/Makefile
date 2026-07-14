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
TEST_GF64_MPE_INTERP_SOURCES := test_gf64_mpe_interp.c
TEST_GF64_INVMOD_MOD_SOURCES := test_gf64_invmod_mod.c
TEST_GF64_POLY_INVMOD_FFT_SOURCES := test_gf64_poly_invmod_fft.c
TEST_GF64_POLY_MUL_KARATSUBA_SOURCES := test_gf64_poly_mul_karatsuba.c
TEST_GF64_FFT_POLY_MUL_SOURCES := test_gf64_fft_poly_mul.c
TEST_GF64_POLY_MUL_TOOM3_SOURCES := test_gf64_poly_mul_toom3.c
TEST_GF64_POLY_MUL_VANDERMONDE_SOURCES := test_gf64_poly_mul_vandermonde.c
TEST_GF64_ADDITIVE_FFT_HQC2026_SOURCES := test_gf64_additive_fft_hqc2026.c

.PHONY: all test clean test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_additive_fft_hqc2026 test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_gf64_mpe_interp test_gf64_invmod_mod test_gf64_poly_invmod_fft test_gf64_poly_mul_karatsuba test_gf64_fft_poly_mul test_gf64_poly_mul_toom3 test_gf64_poly_mul_vandermonde

all: test test_all test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_additive_fft_hqc2026 test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_gf64_mpe_interp test_gf64_invmod_mod test_gf64_poly_invmod_fft test_gf64_poly_mul_karatsuba test_gf64_fft_poly_mul test_gf64_poly_mul_toom3 test_gf64_poly_mul_vandermonde

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
# which in turn depends on gf64_mul_avx512.c (T1), gf64_square.c (T2), and
# gf64_poly_mul_karatsuba.c (Phase 2a Karatsuba dispatch). Since PR-2b
# (issue #27), the subproduct build also calls gf64_poly_invmod_mod from
# gf64_mpe.c to cache polynomial-modular inverses for multi-point
# interpolation. gf64_mpe.c transitively calls gf64_invert_ita_one (the
# T5 Itoh-Tsujii single-value inverse) inside gf64_poly_divmod,
# gf64_poly_invmod, and the new gf64_poly_invmod_mod, so gf64_invert_ita.c
# joins the link set too.
test_gf64_subproduct_tree: $(TEST_GF64_SUBPRODUCT_TREE_SOURCES) ../gf64_subproduct.c ../gf64_mpe.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_SUBPRODUCT_TREE_SOURCES) ../gf64_subproduct.c ../gf64_mpe.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c

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

# Multi-point interpolation test (par3-cauchy-fft-kernel T8b, issue #27).
# Same dependency chain as test_gf64_mpe (T8a's eval direction); the new
# gf64_mpe.c now also contains gf64_multi_point_interp's Bostan-Schost
# top-down implementation, which uses the cached inverses populated at
# tree-build time in gf64_subproduct.c.
test_gf64_mpe_interp: $(TEST_GF64_MPE_INTERP_SOURCES) ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_MPE_INTERP_SOURCES) ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c

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

# Phase 2b probe: does the Vandermonde-matrix additive FFT satisfy the
# convolution theorem (pointwise-multiply in evaluation basis = polynomial
# convolution)? The Vandermonde FFT is correct by construction; the LCH14
# attempt in gf64_additive_fft_lch14.c is kept for reference but is no
# longer the focus.
test_gf64_fft_poly_mul: $(TEST_GF64_FFT_POLY_MUL_SOURCES) ../gf64_additive_fft.c ../gf64_additive_fft_vandermonde.c ../gf64_poly_mul_karatsuba.c ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_FFT_POLY_MUL_SOURCES) ../gf64_additive_fft.c ../gf64_additive_fft_vandermonde.c ../gf64_poly_mul_karatsuba.c ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c

# Phase 2a+ parity test: Toom-Cook 3 (Toom 2.3) poly_mul against schoolbook.
# Char-2 safe evaluation points {0, 1, 2, 3, inf} since -1 = 1 in char 2.
# Built against gf64_single.c (scalar reference) only — no AVX-512 deps.
# Karatsuba linked as a Toom-3 base-case fallback so sub-products below the
# Toom-3 threshold use the O(n^1.585) path instead of schoolbook.
test_gf64_poly_mul_toom3: $(TEST_GF64_POLY_MUL_TOOM3_SOURCES) ../gf64_poly_mul_toom3.c ../gf64_poly_mul_karatsuba.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_POLY_MUL_TOOM3_SOURCES) ../gf64_poly_mul_toom3.c ../gf64_poly_mul_karatsuba.c ../gf64_single.c

# Phase 2b Vandermonde-FFT poly_mul parity test. Correctness-only
# (bit-exact to schoolbook), tests 11 random cases across degrees up to
# 511. The Vandermonde FFT is correct by construction but O(N^2).
test_gf64_poly_mul_vandermonde: $(TEST_GF64_POLY_MUL_VANDERMONDE_SOURCES) ../gf64_additive_fft.c ../gf64_additive_fft_vandermonde.c ../gf64_poly_mul_karatsuba.c ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_POLY_MUL_VANDERMONDE_SOURCES) ../gf64_additive_fft.c ../gf64_additive_fft_vandermonde.c ../gf64_poly_mul_karatsuba.c ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c

# HQC 2026 TCHES §2.3 Algorithm 2 (LCH14 addFFT) over GF(2^64). Replaces
# the buggy multiplier in gf64_additive_fft_lch14.c with the correct
# affine-shift multiplier from the HQC paper. Convolution-theorem probe at
# n ∈ {2, 4, 8, 16, 32, 64}: forward-output, round-trip, and convolution.
# Uses gf64_mul_reference (scalar SSE2 PCLMULQDQ) + gf64_inverse (T5). The
# AVX-512 entrypoint can be wired in later by replacing the field-mul call
# site with gf64_mul_avx512 once the working n exceeds the SIMD setup
# overhead.
test_gf64_additive_fft_hqc2026: $(TEST_GF64_ADDITIVE_FFT_HQC2026_SOURCES) ../gf64_additive_fft_hqc2026.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c ../gf64_invert.c ../gf64_invert_avx512.c ../gf64_dispatch.c ../cpu_detect.c ../gf64_region_scalar.c ../gf64_region_ssse3.c ../gf64_region_ssse3_arr.c ../gf64_region_avx2.c ../gf64_region_avx2_arr.c ../gf64_region_avx512.c ../gf64_region_avx512_arr.c ../gf64_invert_ita.c ../gf64_invert_ssse3.c ../gf64_invert_avx2.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_ADDITIVE_FFT_HQC2026_SOURCES) ../gf64_additive_fft_hqc2026.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c ../gf64_invert.c ../gf64_invert_avx512.c ../gf64_dispatch.c ../cpu_detect.c ../gf64_region_scalar.c ../gf64_region_ssse3.c ../gf64_region_ssse3_arr.c ../gf64_region_avx2.c ../gf64_region_avx2_arr.c ../gf64_region_avx512.c ../gf64_region_avx512_arr.c ../gf64_invert_ita.c ../gf64_invert_ssse3.c ../gf64_invert_avx2.c

# Standalone polynomial modular-inverse test (Phase A3 of issue #27).
# Verifies that gf64_poly_invmod_mod computes inv such that g * inv ≡ 1
# (mod f) at degrees (1,1), (3,3), (5,5), (1,2), (2,1).
test_gf64_invmod_mod: $(TEST_GF64_INVMOD_MOD_SOURCES) ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_GF64_INVMOD_MOD_SOURCES) ../gf64_mpe.c ../gf64_subproduct.c ../gf64_invert_ita.c ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_single.c

clean:
	rm -f test test_all test_gf64_mul_avx512 test_gf64_square test_gf64_invert_ita test_gf64_additive_fft test_gf64_additive_fft_hqc2026 test_gf64_subproduct_tree test_gf64_barycentric test_gf64_mpe test_gf64_mpe_interp test_gf64_poly_invmod_fft test_gf64_poly_mul_karatsuba test_gf64_fft_poly_mul test_gf64_poly_mul_toom3 test_gf64_poly_mul_vandermonde gen_toom3_constants bench_poly_mul_toom3_vs_karatsuba bench_poly_mul_vandermonde_vs_karatsuba
