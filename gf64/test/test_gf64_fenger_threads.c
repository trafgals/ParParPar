/*
 * ============================================================================
 * test_gf64_fenger_threads.c — multi-threaded Fenger parity test (issue #59 T7-verify)
 *
 * Verifies that the per-thread B-slab sharding in gf64_fenger_execute
 * (src/par3_engine_fenger.cc) produces bit-identical output to the
 * single-thread reference regardless of the number of threads. The
 * threads race disjoint B-word slabs over a shared read-only Fenger
 * context, partitioning the B-axis (blockSize64) into roughly equal
 * contiguous slices via POSIX threads.
 *
 * XOR-linearity of the Cauchy matvec guarantees per-thread partition
 * invariance: the result at word w is determined by the input words at
 * index w across all c inputs, and each input is read by exactly one
 * thread (its slab contains w). The output is therefore bit-exact to
 * the single-thread reference up to the order in which slab results
 * are concatenated (which is deterministic: pthread_create, pthread_join).
 *
 * Implemented in plain C with POSIX threads to avoid the
 * extern-"C" wrapping gymnastics that would be needed if compiled with
 * g++ (gf64_fenger.c's function definitions are not in an extern "C"
 * block, so g++ would C++-mangle them and break the symbol resolution).
 *
 * Build & run from gf64/test/:
 *   make test_gf64_fenger_threads && ./test_gf64_fenger_threads
 *
 * Exit 0 on full pass; non-zero on any mismatch. Records to stdout the
 * N, B, num_threads tuple for each shape.
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

extern void gf64_fenger_matvec(const void *in, size_t N, size_t B,
                              void *out, size_t R,
                              uint64_t firstInput, uint64_t firstRecovery);
extern void *gf64_fenger_prepare(uint64_t firstInput, uint64_t firstRecovery,
                                 size_t N, size_t R);
extern void gf64_fenger_execute(void *ctx, const void *in, size_t B,
                                void *out, size_t w_start, size_t w_end);
extern void gf64_fenger_release(void *ctx);

static int g_passed = 0;
static int g_failed = 0;

static void pass(const char *name) {
	printf("  PASS: %s\n", name);
	g_passed++;
}

static void fail(const char *name) {
	printf("  FAIL: %s\n", name);
	g_failed++;
}

/* Per-thread work: execute a disjoint B-slab of the Fenger ctx. */
struct thread_arg {
	void *ctx;
	const void *in;
	size_t B;
	void *out;
	size_t w_start;
	size_t w_end;
};

static void *worker_main(void *arg_) {
	struct thread_arg *a = (struct thread_arg *)arg_;
	gf64_fenger_execute(a->ctx, a->in, a->B, a->out, a->w_start, a->w_end);
	return NULL;
}

/* Single-thread reference: gf64_fenger_matvec. N is the number of
 * input blocks (NOT 0 — N=0 is a trivial-input no-op that writes
 * nothing, see ComputeRecoveryBlocksBarycentric:104 the trivial-input
 * short-circuit). */
static void run_single(void *ctx, const void *in, size_t N, size_t B,
                       void *out, size_t R,
                       uint64_t firstInput, uint64_t firstRecovery) {
	(void)ctx;
	gf64_fenger_matvec(in, N, B, out, R, firstInput, firstRecovery);
}

static void run_threads(void *ctx, const void *in, size_t B,
                        void *out, size_t R, int num_threads) {
	pthread_t *workers = (pthread_t *)calloc((size_t)num_threads,
	                                        sizeof(pthread_t));
	struct thread_arg *args = (struct thread_arg *)calloc((size_t)num_threads,
	                                                  sizeof(struct thread_arg));
	const size_t chunk = (B + (size_t)num_threads - 1) / (size_t)num_threads;
	size_t w_start = 0;
	int created = 0;
	int t;
	for (t = 0; t < num_threads && w_start < B; t++) {
		size_t w_end = w_start + chunk;
		if (w_end > B) w_end = B;
		args[t].ctx = ctx;
		args[t].in = in;
		args[t].B = B;
		args[t].out = out;
		args[t].w_start = w_start;
		args[t].w_end = w_end;
		pthread_create(&workers[t], NULL, worker_main, &args[t]);
		created++;
		w_start = w_end;
	}
	for (t = 0; t < created; t++) {
		pthread_join(workers[t], NULL);
	}
	free(workers);
	free(args);
}

/* Deterministic PRNG for reproducible test data. */
static uint64_t splitmix64(uint64_t *state) {
	*state += 0x9E3779B97F4A7C15ULL;
	uint64_t z = *state;
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static int run_case(size_t N, size_t R, size_t B, int num_threads,
                   uint64_t seed) {
	uint64_t prng = seed;
	uint64_t *in = (uint64_t *)malloc((size_t)N * B * sizeof(uint64_t));
	for (size_t i = 0; i < (size_t)N * B; i++) {
		in[i] = splitmix64(&prng);
	}

	void *ctx = gf64_fenger_prepare(/*firstInput=*/0,
	                               /*firstRecovery=*/(uint64_t)N,
	                               N, R);
	if (!ctx) {
		printf("  FAIL (N=%zu/R=%zu): ctx null\n", N, R);
		g_failed++;
		free(in);
		return 1;
	}

	uint64_t *out_single = (uint64_t *)calloc((size_t)R * B, sizeof(uint64_t));
	uint64_t *out_thread = (uint64_t *)calloc((size_t)R * B, sizeof(uint64_t));

	run_single(ctx, in, N, B, out_single, R, 0, (uint64_t)N);
	run_threads(ctx, in, B, out_thread, R, num_threads);

	int mismatch = 0;
	for (size_t i = 0; i < (size_t)R * B; i++) {
		if (out_single[i] != out_thread[i]) {
			if (mismatch < 4) {
				printf("  N=%zu/R=%zu T=%d: out[%zu] = 0x%016llx vs 0x%016llx\n",
					N, R, num_threads,
					(unsigned long long)i,
					(unsigned long long)out_single[i],
					(unsigned long long)out_thread[i]);
			}
			mismatch++;
		}
	}

	gf64_fenger_release(ctx);

	char tag[128];
	snprintf(tag, sizeof(tag),
	         "N=%zu/R=%zu/B=%zu threads=%d mismatch=%d",
	         N, R, B, num_threads, mismatch);
	if (mismatch == 0) {
		pass(tag);
	} else {
		fail(tag);
	}

	free(in);
	free(out_single);
	free(out_thread);
	return mismatch ? 1 : 0;
}

int main(void) {
	printf("Multi-threaded Fenger execute parity (issue #59 T7-verify)\n");
	printf("==========================================================\n\n");

	int fails = 0;

	/* Plan-specified canonical shape: N=4096, B=256, fan {2,4,8}. */
	fails += run_case(/*N=*/4096, /*R=*/4096, /*B=*/256, /*T=*/2,
	                  0xCAFEBABEDEADBEEFULL);
	fails += run_case(4096, 4096, 256, 4, 0x1234567890ABCDEFULL);
	fails += run_case(4096, 4096, 256, 8, 0xDEADBEEFCAFEBABEULL);

	/* Off-by-one in chunked-slab boundary handling at B % num_threads. */
	fails += run_case(/*N=*/1024, /*R=*/1024, /*B=*/300, /*T=*/3,
	                  0xFEEDFACEDEADBEEFULL);
	fails += run_case(1024, 1024, 300, 7, 0x55AA55AA55AA55AAULL);

	/* Smaller B to ensure single-block-or-smaller slabs behave. */
	fails += run_case(/*N=*/256, /*R=*/256, /*B=*/9, /*T=*/2,
	                  0xAABBCCDDEEFF0011ULL);
	fails += run_case(256, 256, 9, 4, 0x1122334455667788ULL);

	printf("\n=== Summary ===\n");
	printf("Passed: %d\n", g_passed);
	printf("Failed: %d\n", g_failed);

	return fails ? 1 : 0;
}
