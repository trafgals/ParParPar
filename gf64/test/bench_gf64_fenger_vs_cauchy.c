/*
 * bench_gf64_fenger_vs_cauchy.c — microbench: Fenger pipeline vs explicit
 *   Cauchy matvec at canonical PAR3 shapes (issue #28).
 *
 * Compares wall-clock throughput of:
 *
 *   cauchy_reference: out[r][w] = XOR_c in[c][w] / (y_r XOR x_c)
 *                     (explicit O(N) per output)
 *
 *   gf64_fenger_matvec: Bostan-Schost top-down Fenger pipeline (issue #28)
 *
 * Over a grid of (N, R, B). Throughput is reported in
 *   bytes/sec = (N * B + R * B) * 8 / time
 *
 * which approximates the "create" bandwidth for a slice-count of N input
 * slices, B gf64_t per slice, R recovery slices.
 *
 * Build & run:
 *   cd gf64/test
 *   gcc -O3 -march=native -I.. bench_gf64_fenger_vs_cauchy.c \
 *       ../gf64_fenger.c ../gf64_mpe.c ../gf64_subproduct.c \
 *       ../gf64_barycentric.c ../gf64_invert_ita.c ../gf64_invert.c \
 *       ../gf64_additive_fft.c ../gf64_poly_mul_karatsuba.c \
 *       ../gf64_poly_mul_toom3.c ../gf64_mul_avx512.c ../gf64_square.c \
 *       ../gf64_single.c -o bench_gf64_fenger_vs_cauchy
 *   ./bench_gf64_fenger_vs_cauchy
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../gf64_global.h"
#include "../gf64_fenger.h"

extern gf64_t gf64_mul_reference(gf64_t a, gf64_t b);
extern gf64_t gf64_inverse(gf64_t a);

static uint64_t g_rng = 0x1234567890ABCDEFULL;
static uint64_t splitmix64_next(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* ----- Explicit Cauchy matrix-vector product. */
static void cauchy_reference(
	const gf64_t *in, size_t N, size_t B,
	gf64_t *out, size_t R,
	uint64_t firstInput, uint64_t firstRecovery)
{
	for (size_t r = 0; r < R; r++) {
		gf64_t yr = (gf64_t)(firstRecovery + (uint64_t)r);
		gf64_t *out_row = out + r * B;
		for (size_t w = 0; w < B; w++) out_row[w] = 0;
		for (size_t c = 0; c < N; c++) {
			gf64_t xc = (gf64_t)(firstInput + (uint64_t)c);
			gf64_t denom = gf64_inverse(yr ^ xc);
			const gf64_t *in_row = in + c * B;
			for (size_t w = 0; w < B; w++) {
				out_row[w] ^= gf64_mul_reference(in_row[w], denom);
			}
		}
	}
}

static void bench_config(size_t N, size_t R, size_t B, int iters) {
	gf64_t *in         = (gf64_t *)malloc(N * B * sizeof(gf64_t));
	gf64_t *cauchy_out = (gf64_t *)calloc(R * B, sizeof(gf64_t));
	gf64_t *fenger_out = (gf64_t *)calloc(R * B, sizeof(gf64_t));
	if (!in || !cauchy_out || !fenger_out) abort();

	g_rng = 0xDEADBEEF;
	for (size_t i = 0; i < N * B; i++) in[i] = splitmix64_next();

	const uint64_t fi = 0x10000;
	const uint64_t fr = 0x1000000;

	/* Warmup. */
	cauchy_reference(in, N, B, cauchy_out, R, fi, fr);
	gf64_fenger_matvec(in, N, B, fenger_out, R, fi, fr);

	double t0 = now_sec();
	for (int i = 0; i < iters; i++) {
		cauchy_reference(in, N, B, cauchy_out, R, fi, fr);
	}
	double t_cauchy = now_sec() - t0;

	t0 = now_sec();
	for (int i = 0; i < iters; i++) {
		gf64_fenger_matvec(in, N, B, fenger_out, R, fi, fr);
	}
	double t_fenger = now_sec() - t0;

	const double bytes_total = (double)(N * B + R * B) * sizeof(gf64_t);
	const double bw_cauchy = bytes_total / t_cauchy / 1e6; /* MB/s */
	const double bw_fenger = bytes_total / t_fenger / 1e6;
	const double speedup  = t_cauchy / t_fenger;

	printf("  N=%-5zu R=%-5zu B=%-6zu  cauchy=%7.1f MB/s  fenger=%7.1f MB/s  speedup=%5.2fx\n",
	       N, R, B, bw_cauchy, bw_fenger, speedup);

	free(in); free(cauchy_out); free(fenger_out);
}

int main(void) {
	printf("Fenger Toeplitz pipeline — microbench (issue #28)\n");
	printf("Wall-clock throughput on small (N, R, B); extrapolate to 1GiB.\n\n");

	/* The canonical 1 GiB / 10K-slice / 1K-recovery workload has:
	 *   blockSize64 = 1 GiB / 10K slices / 8 = ~12.5K gf64_t
	 * which is too big to bench at this level (would need ~120 GB RAM).
	 * We bench a downscaled shape that preserves the (N, R) ratio:
	 *
	 *   N=64, R=64, B=12.5K — 64*12.5K*8 = 6.4 MB input, similar output.
	 *
	 * and progressively larger N to expose the asymptotic regime. */

	bench_config(8, 8, 4096, 20);
	bench_config(16, 16, 4096, 20);
	bench_config(32, 32, 4096, 20);
	bench_config(64, 64, 4096, 10);
	bench_config(128, 128, 4096, 5);
	bench_config(256, 64, 4096, 5);
	bench_config(512, 256, 2048, 3);
	bench_config(1024, 256, 1024, 3);

	/* Smaller-B bench at larger N to expose MPE cost shape. */
	bench_config(256, 128, 256, 10);
	bench_config(512, 128, 256, 10);

	printf("\nDone.\n");
	return 0;
}
