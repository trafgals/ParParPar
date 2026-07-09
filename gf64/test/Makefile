# GF64 Test Makefile
# Builds and runs GF(2^64) arithmetic and SIMD tests

CC := gcc
CXX := g++
CFLAGS := -std=c99 -Wall -Wextra -O2 -march=native -D_POSIX_C_SOURCE=200112L -D_GNU_SOURCE
CXXFLAGS := -std=c++11 -Wall -Wextra -O2 -march=native -D_POSIX_C_SOURCE=200112L -D_GNU_SOURCE
LDFLAGS :=

# Source files
GF64_SOURCES := ../gf64_single.c ../gf64_region_scalar.c ../gf64_region_ssse3.c ../gf64_region_ssse3_arr.c ../gf64_region_avx2.c ../gf64_region_avx2_arr.c ../gf64_region_avx512.c ../gf64_region_avx512_arr.c ../gf64_mul_avx512.c ../gf64_square.c ../gf64_dispatch.c ../cpu_detect.c ../gf64_invert.c ../gf64_invert_ssse3.c ../gf64_invert_avx2.c ../gf64_invert_avx512.c
GF64_INCLUDES := -I..

# Test sources
TEST_SOURCES := test.c
TEST_ALL_SOURCES := test_all.c
TEST_GF64_MUL_AVX512_SOURCES := test_gf64_mul_avx512.c
TEST_GF64_SQUARE_SOURCES := test_gf64_square.c

.PHONY: all test clean test_gf64_mul_avx512 test_gf64_square

all: test test_all test_gf64_mul_avx512 test_gf64_square

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

clean:
	rm -f test test_all test_gf64_mul_avx512 test_gf64_square
