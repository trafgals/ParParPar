/* Parity: interleaved batch-shared mul (B1-1.4) vs the scalar batch-shared
 * mul, plus fwd/inv transform isolation against the scalar FFT.
 *
 * Shapes cover the interp's combine sizes (n = 1024/512/256/64) with
 * K = 8 (full lanes), K = 4 (half), and an odd K = 5 (pad lanes), plus
 * out_len > full_len padding.
 *
 * Build & run from gf64/test/:
 *   make test_gf64_interleaved_batch && ./test_gf64_interleaved_batch
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../gf64_global.h"
#include "../gf64_additive_fft.h"
#include "../gf64_mul.h"
#include "../gf64_cantor_basis.h"

#define MAXN 2048

static gf64_t rnd64(void) {
	gf64_t x = 0;
	for (int i = 0; i < 8; i++) x = (x << 8) ^ (gf64_t)(rand() & 0xff);
	return x ? x : 1;
}

static int check_case(const char *name, size_t K,
                      size_t len_shared, size_t len_f, size_t out_len) {
	size_t full_len = len_shared + len_f - 1;
	if (full_len < out_len) full_len = out_len;
	size_t n = 1;
	while (n < full_len) n <<= 1;
	assert(n <= MAXN);

	static gf64_t shared[MAXN];
	static gf64_t f[8][MAXN];
	static gf64_t ref[8][MAXN];
	static gf64_t got[8][MAXN];
	static gf64_t scratch[64 * MAXN];

	for (size_t i = 0; i < len_shared; i++) shared[i] = rnd64();
	for (size_t k = 0; k < K; k++)
		for (size_t i = 0; i < len_f; i++) f[k][i] = rnd64();

	gf64_t *outs_ref[8], *outs_got[8];
	for (size_t k = 0; k < K; k++) { outs_ref[k] = ref[k]; outs_got[k] = got[k]; }

	gf64_addfft64_poly_mul_batch_shared(outs_ref, K, shared, len_shared,
	                                    &f[0][0], len_f, out_len,
	                                    scratch, 4 * n);
	gf64_addfft64_poly_mul_batch_shared_interleaved_avx512(
	    outs_got, K, shared, len_shared, &f[0][0], len_f, out_len,
	    scratch, 64 * MAXN);

	for (size_t k = 0; k < K; k++) {
		for (size_t i = 0; i < out_len; i++) {
			if (ref[k][i] != got[k][i]) {
				fprintf(stderr, "MISMATCH %s: K=%zu ls=%zu lf=%zu out=%zu "
				        "k=%zu i=%zu ref=%016llx got=%016llx\n",
				        name, K, len_shared, len_f, out_len, k, i,
				        (unsigned long long)ref[k][i],
				        (unsigned long long)got[k][i]);
				return 1;
			}
		}
	}
	printf("ok %s: K=%zu len_shared=%zu len_f=%zu out_len=%zu (n=%zu)\n",
	       name, K, len_shared, len_f, out_len, n);
	return 0;
}

int main(void) {
	gf64_init_dispatch();
	if (!gf64_zmm_works || !gf64_has_vpclmulqdq) {
		printf("SKIP: no AVX-512F + VPCLMULQDQ on this host\n");
		return 0;
	}
	srand(42);
	int fails = 0;
	fails += check_case("n1024-k8",  8, 513, 512, 1024);
	fails += check_case("n1024-k4",  4, 513, 512, 1024);
	fails += check_case("n512-k8",   8, 257, 256, 512);
	fails += check_case("n512-k5",   5, 257, 256, 512);
	fails += check_case("n256-k8",   8, 129, 128, 256);
	fails += check_case("n64-k8",    8, 33,  32,  64);
	fails += check_case("n64-k3",    3, 33,  32,  64);
	fails += check_case("pad-k8",    8, 100, 100, 256);
	fails += check_case("k1",        1, 513, 512, 1024);

	/* Roundtrip: interleaved fwd then inv = identity per lane. */
	{
		static const size_t rns[] = {8, 16, 32, 64, 128, 256, 512, 1024};
		for (size_t idx = 0; idx < sizeof(rns) / sizeof(rns[0]); idx++) {
		size_t rn = rns[idx];
		static gf64_t buf[8][2048];
		static gf64_t scratch[32 * 2048];
		const size_t W = 8;
		for (size_t i = 0; i < rn; i++)
			for (size_t k = 0; k < W; k++) buf[k][i] = rnd64();
		static gf64_t sbuf[8][2048];
		static gf64_t ssc[8 * 2048];
		static gf64_t ori[8][2048];
		for (size_t k = 0; k < W; k++)
			for (size_t i = 0; i < rn; i++) ori[k][i] = buf[k][i];
		for (size_t k = 0; k < W; k++)
			for (size_t i = 0; i < rn; i++) sbuf[k][i] = buf[k][i];
		static gf64_t tbuf[8 * 2048];
		for (size_t i = 0; i < rn; i++)
			for (size_t k = 0; k < W; k++)
				tbuf[i * W + k] = buf[k][i];
		gf64_addfft64_fwd_batch_avx512(tbuf, rn, scratch, 32 * 2048);
		static gf64_t bfwd[8][2048];
		for (size_t k = 0; k < W; k++)
			for (size_t i = 0; i < rn; i++) bfwd[k][i] = tbuf[i * W + k];
		gf64_addfft64_inv_batch_avx512(tbuf, rn, scratch, 32 * 2048);
		for (size_t k = 0; k < W; k++) {
			gf64_addfft64_fwd_recursive_scratch(sbuf[k], rn, ssc, 8 * 2048);
		}
		static gf64_t sfwd[8][2048];
		for (size_t k = 0; k < W; k++)
			for (size_t i = 0; i < rn; i++) sfwd[k][i] = sbuf[k][i];
		for (size_t k = 0; k < W; k++) {
			gf64_addfft64_inv_recursive_scratch(sbuf[k], rn, ssc, 8 * 2048);
		}
		for (size_t k = 0; k < W; k++)
			for (size_t i = 0; i < rn; i++)
				if (bfwd[k][i] != sfwd[k][i]) {
					fprintf(stderr, "FWD-RT n=%zu MISMATCH k=%zu i=%zu batch=%016llx scalar=%016llx orig=%016llx\n",
					        rn, k, i,
					        (unsigned long long)bfwd[k][i],
					        (unsigned long long)sfwd[k][i],
					        (unsigned long long)ori[k][i]);
					if (k == 0) {
						fprintf(stderr, "  bfwd: ");
						for (size_t j = 0; j < rn; j++)
							fprintf(stderr, "%s%016llx", j ? " " : "",
							        (unsigned long long)bfwd[0][j]);
						fprintf(stderr, "\n  sfwd: ");
						for (size_t j = 0; j < rn; j++)
							fprintf(stderr, "%s%016llx", j ? " " : "",
							        (unsigned long long)sfwd[0][j]);
						fprintf(stderr, "\n  orig: ");
						for (size_t j = 0; j < rn; j++)
							fprintf(stderr, "%s%016llx", j ? " " : "",
							        (unsigned long long)ori[0][j]);
						fprintf(stderr, "\n");
					}
					fails++;
					goto rt_done;
				}
		for (size_t k = 0; k < W; k++)
			for (size_t i = 0; i < rn; i++)
				if (tbuf[i * W + k] != sbuf[k][i]) {
					fprintf(stderr, "RTCMP n=%zu MISMATCH k=%zu i=%zu batch=%016llx scalar=%016llx orig=%016llx\n",
					        rn, k, i,
					        (unsigned long long)tbuf[i * W + k],
					        (unsigned long long)sbuf[k][i],
					        (unsigned long long)ori[k][i]);
					fails++;
					goto rt_done;
				}
		printf("ok roundtrip n=%zu K=8\n", rn);
		}
	}
rt_done: ;

	/* Full-lane fwd isolation: batch fwd (ALL 8 lanes nonzero) vs the
	 * per-lane scalar fwd — pin fwd vs inv as the lane-crossing stage. */
	{
		static const size_t rns[] = {8, 16, 32, 64, 128, 256, 512, 1024};
		for (size_t idx = 0; idx < sizeof(rns) / sizeof(rns[0]); idx++) {
		size_t rn = rns[idx];
		static gf64_t buf[8][2048];
		static gf64_t lane[8][2048];
		static gf64_t scratch[32 * 2048];
		static gf64_t sc[8 * 2048];
		const size_t W = 8;
		for (size_t i = 0; i < rn; i++)
			for (size_t k = 0; k < W; k++) buf[0][i * W + k] = rnd64();
		for (size_t k = 0; k < W; k++)
			for (size_t i = 0; i < rn; i++) lane[k][i] = buf[0][i * W + k];
		gf64_addfft64_fwd_batch_avx512(buf[0], rn, scratch, 32 * 2048);
		for (size_t k = 0; k < W; k++)
			gf64_addfft64_fwd_recursive_scratch(lane[k], rn, sc, 8 * 2048);
		for (size_t k = 0; k < W; k++)
			for (size_t i = 0; i < rn; i++)
				if (buf[0][i * W + k] != lane[k][i]) {
					fprintf(stderr, "FULLFWD n=%zu MISMATCH k=%zu i=%zu scalar=%016llx batch=%016llx\n",
					        rn, k, i, (unsigned long long)lane[k][i],
					        (unsigned long long)buf[0][i * W + k]);
					fails++;
					goto fullfwd_done;
				}
		printf("ok full-lane fwd n=%zu\n", rn);
		}
	}
fullfwd_done: ;

	/* Full-lane inv isolation: batch inv vs the per-lane scalar inv —
	 * the roundtrip fails with the fwd proven correct, so the inv is
	 * the suspect stage. */
	{
		static const size_t rns[] = {8, 16, 32, 64, 128, 256};
		for (size_t idx = 0; idx < sizeof(rns) / sizeof(rns[0]); idx++) {
		size_t rn = rns[idx];
		static gf64_t buf[8][2048];
		static gf64_t lane[8][2048];
		static gf64_t scratch[32 * 2048];
		static gf64_t sc[8 * 2048];
		const size_t W = 8;
		for (size_t i = 0; i < rn; i++)
			for (size_t k = 0; k < W; k++) buf[0][i * W + k] = rnd64();
		for (size_t k = 0; k < W; k++)
			for (size_t i = 0; i < rn; i++) lane[k][i] = buf[0][i * W + k];
		gf64_addfft64_inv_batch_avx512(buf[0], rn, scratch, 32 * 2048);
		for (size_t k = 0; k < W; k++)
			gf64_addfft64_inv_recursive_scratch(lane[k], rn, sc, 8 * 2048);
		for (size_t k = 0; k < W; k++)
			for (size_t i = 0; i < rn; i++)
				if (buf[0][i * W + k] != lane[k][i]) {
					fprintf(stderr, "FULLINV n=%zu MISMATCH k=%zu i=%zu scalar=%016llx batch=%016llx\n",
					        rn, k, i, (unsigned long long)lane[k][i],
					        (unsigned long long)buf[0][i * W + k]);
					fails++;
					goto fullinv_done;
				}
		printf("ok full-lane inv n=%zu\n", rn);
		}
	}
fullinv_done: ;

	/* Isolate: batch fwd (single word, lane 0) vs the scalar fwd. */
	{
		static const size_t nns[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
		for (size_t idx = 0; idx < sizeof(nns) / sizeof(nns[0]); idx++) {
		size_t nn = nns[idx];
		static gf64_t a[2048], buf[8][2048];
		static gf64_t scratch[32 * 2048];
		static gf64_t sc[8 * 2048];
		for (size_t i = 0; i < nn; i++) a[i] = rnd64();
		for (size_t k = 0; k < 8; k++)
			for (size_t i = 0; i < nn; i++) buf[0][i * 8 + k] = 0;
		for (size_t i = 0; i < nn; i++) buf[0][i * 8] = a[i];
		gf64_addfft64_fwd_batch_avx512(buf[0], nn, scratch, 32 * 2048);
		gf64_addfft64_fwd_recursive_scratch(a, nn, sc, 8 * 2048);
		for (size_t i = 0; i < nn; i++)
			if (buf[0][i * 8] != a[i]) {
				fprintf(stderr, "FWD-ISO n=%zu MISMATCH i=%zu scalar=%016llx batch=%016llx\n",
				        nn, i, (unsigned long long)a[i],
				        (unsigned long long)buf[0][i * 8]);
				fails++;
				goto fwd_done;
			}
		printf("ok fwd-isolation n=%zu\n", nn);
		}
	}
fwd_done:
	if (fails) { fprintf(stderr, "INTERLEAVED_PARITY_FAIL %d\n", fails); return 1; }
	printf("INTERLEAVED_PARITY_PASS\n");
	return 0;
}
