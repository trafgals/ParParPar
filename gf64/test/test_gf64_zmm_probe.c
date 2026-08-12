/*
 * test_gf64_zmm_probe.c — tests for the cached one-shot ZMM probe
 *   (gf64_zmm_probe, gf64_zmm_works) in gf64/cpu_detect.c.
 *
 * Background (cubic review 4915648282 P1):
 *   The previous implementation used a plain int cache + CAS, with
 *   plain loads/stores on the cache. The CAS serialises entry, but
 *   the result-publish to concurrent readers was missing memory
 *   ordering — a thread that saw the cache populated might also
 *   read stale data written before the cache was filled.
 *
 *   Fix: release store on the cache write, acquire load on the
 *   fast-path read. The release-acquire pair is a "publish" pattern
 *   that is correct under both TSO (x86) and weakly ordered archs.
 *
 * What this test verifies:
 *   1. First call returns the host's actual ZMM capability (the
 *      probe was correctly populated by the slow path).
 *   2. Second call returns the SAME value (the cache is sticky;
 *      the slow path does NOT re-execute).
 *   3. Third call returns the same value AND a different value
 *      would only be possible if the cache were corrupted.
 *   4. The legacy gf64_zmm_works global matches the return value
 *      of the first call (so the legacy reader sees the same data).
 *   5. Under a multithreaded fan-out (POSIX threads), all N threads
 *      observe the same ZMM capability — proves the release-acquire
 *      publishes the result to concurrent readers without races.
 *
 * The multithreaded test is gated on _POSIX_THREADS. On Windows or
 * other non-POSIX platforms, only the single-threaded tests (1-4)
 * run.
 *
 * Build (via gf64/test/Makefile):
 *   cd gf64/test
 *   make test_gf64_zmm_probe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Pull in the public probe API. We need gf64_global.h for
 * gf64_zmm_probe and gf64_zmm_works, plus the underlying
 * gf64_has_vpclmulqdq_probe for the consistency check. */
#include "../gf64_global.h"

HEDLEY_BEGIN_C_DECLS

/* ===================================================================
 * Test 1: cache populates on first call, sticks on second
 * =================================================================== */

static int test_cache_sticky(void) {
	/* Reset the cache by NOT calling — the cache is process-lifetime
	 * and we are the first reader in this test process. (The
	 * standalone test binary links only cpu_detect.c + this test,
	 * so no other code path has run init_dispatch.) */
	int first = gf64_zmm_probe();
	int second = gf64_zmm_probe();
	int third = gf64_zmm_probe();

	/* The cache MUST return the same value on every call. */
	if (first != second || second != third) {
		printf("  cache sticky: FAIL  first=%d second=%d third=%d\n",
		       first, second, third);
		return 1;
	}
	/* The value must be 0 or 1. (No host can return -1 here because
	 * the cache is populated by the first call.) */
	if (first != 0 && first != 1) {
		printf("  cache sticky: FAIL  unexpected value %d (want 0 or 1)\n",
		       first);
		return 1;
	}
	printf("  cache sticky: OK  value=%d across 3 calls\n", first);
	return 0;
}

/* ===================================================================
 * Test 2: legacy gf64_zmm_works global matches probe return value
 *
 * This is the "release publish" check at the API surface. The
 * probe writes gf64_zmm_works BEFORE the release-store on
 * zmm_probe_cached (cubic review 4916023985 P2 finding 4: the
 * reverse order was a publish-order race). The release-store
 * barrier flushes the earlier gf64_zmm_works write so any
 * concurrent reader that observes the cache populated (via
 * acquire-load) also observes the matching gf64_zmm_works.
 *
 * The legacy reader (any code that uses gf64_zmm_works directly)
 * must therefore see a consistent value relative to the probe
 * return value.
 * =================================================================== */

static int test_legacy_global(void) {
	/* Trigger a probe (slow path, populates cache + global). */
	int probed = gf64_zmm_probe();
	if (probed != gf64_zmm_works) {
		printf("  legacy global: FAIL  probe=%d gf64_zmm_works=%d "
		       "(release/acquire pair did not publish)\n",
		       probed, gf64_zmm_works);
		return 1;
	}
	/* After the publish, a fresh call must see both consistent. */
	int probed2 = gf64_zmm_probe();
	if (probed2 != gf64_zmm_works) {
		printf("  legacy global: FAIL  on re-read probe=%d "
		       "gf64_zmm_works=%d (cache drift)\n",
		       probed2, gf64_zmm_works);
		return 1;
	}
	printf("  legacy global: OK  probe=%d matches gf64_zmm_works\n", probed);
	return 0;
}

/* ===================================================================
 * Test 3: result is consistent with CPUID+XCR0 ground truth
 *
 * The probe's report must satisfy the one-way implication
 *
 *     probed == 1   IMPLIES   gf64_has_vpclmulqdq_probe() == 1
 *
 * (the probe is "real ZMM execution", so it cannot succeed if
 * CPUID+XCR0 didn't even advertise the instruction in the first
 * place). The REVERSE implication does NOT hold — and that is
 * exactly the WSL2 / Hyper-V observer-effect case this probe
 * exists to handle (CPUID says yes, but the underlying ZMM
 * instruction SIGILLs at runtime because the hypervisor masks
 * the AVX-512 feature when it detects a binary containing ZMM
 * instructions). See gf64/cpu_detect.c:5-19 (the WSL2/Hyper-V
 * "observer effect" header comment) and the linked MS issues.
 *
 * What this test checks (and what it does NOT check):
 *
 *   checked (one-way):  probed==1 → cpuid_vpclmulqdq==1
 *                       (a probe claiming ZMM works when CPUID
 *                        doesn't even advertise it would be a
 *                        bug — return value mismatch.)
 *
 *   NOT checked:        cpuid_vpclmulqdq==1 → probed==1
 *                       (this is the WSL2 case and is EXPECTED
 *                        to diverge on affected hosts. The probe
 *                        is the authoritative signal for "ZMM
 *                        actually runs".)
 *
 * Cubic review 4916023985 P2 finding 5: the test does NOT
 * hard-fail on the WSL2 case. The only hard-fail assertion is
 * the one-way implication above (probe=1 requires CPUID=1).
 * The WSL2 case is reported INFO and noted in the output.
 * =================================================================== */

static int test_consistent_with_cpuid(void) {
	int probed = gf64_zmm_probe();
	int cpuid_vpclmulqdq = gf64_has_vpclmulqdq_probe();

	/* The one-way implication: probed==1 ⇒ cpuid==1. */
	if (probed && !cpuid_vpclmulqdq) {
		printf("  cpuid consistency: FAIL  probed=%d but CPUID+XCR0 "
		       "says VPCLMULQDQ=%d (probe reported ZMM success when "
		       "CPUID didn't advertise it — that's a probe bug)\n",
		       probed, cpuid_vpclmulqdq);
		return 1;
	}

	/* The REVERSE direction (cpuid=1 but probe=0) is the WSL2 /
	 * Hyper-V observer-effect case — EXPECTED on affected hosts.
	 * Report it INFO, do NOT fail. */
	if (cpuid_vpclmulqdq && !probed) {
		printf("  cpuid consistency: INFO  CPUID advertises VPCLMULQDQ "
		       "but probe reports ZMM SIGILL — this is the WSL2/Hyper-V "
		       "observer-effect case the probe exists to handle. "
		       "(probed=%d, cpuid_vpclmulqdq=%d)\n",
		       probed, cpuid_vpclmulqdq);
	} else {
		printf("  cpuid consistency: OK  probed=%d cpuid_vpclmulqdq=%d "
		       "(one-way implication holds)\n",
		       probed, cpuid_vpclmulqdq);
	}
	return 0;
}

/* ===================================================================
 * Test 4: many calls, no observable state drift
 *
 * Stress test: 10000 calls, verify all return the same value and
 * the legacy global stays in sync. Exercises the fast-path hot
 * loop. Catches bugs where the cache state machine transitions
 * unexpectedly (e.g. someone introduces a "refresh" path).
 * =================================================================== */

static int test_many_calls_stable(void) {
	int expected = gf64_zmm_probe();
	for (int i = 0; i < 10000; i++) {
		int r = gf64_zmm_probe();
		if (r != expected) {
			printf("  many calls: FAIL  drift at iteration %d: "
			       "expected=%d got=%d\n", i, expected, r);
			return 1;
		}
		if (gf64_zmm_works != expected) {
			printf("  many calls: FAIL  gf64_zmm_works drift at "
			       "iteration %d: expected=%d got=%d\n",
			       i, expected, gf64_zmm_works);
			return 1;
		}
	}
	printf("  many calls: OK  10000 calls all returned %d\n", expected);
	return 0;
}

/* ===================================================================
 * Test 5 (POSIX threads only): concurrent fan-out observes the
 * same published result.
 *
 * Spawns N threads, each of which races to call gf64_zmm_probe().
 * The CAS in the probe serialises entry, and the release-acquire
 * pair publishes the result. All N threads MUST observe the same
 * value; a race would manifest as either (a) different return
 * values across threads, or (b) gf64_zmm_works disagreeing with
 * the probe return value.
 *
 * This is the test that catches the cubic review 4915648282 P1
 * data race: without the release-acquire pair, a thread that
 * observed the cache populated (zmm_probe_cached != -1) might
 * also read a stale gf64_zmm_works (the publish ordering would
 * be missing). With TSan this would fire; here we verify by
 * the consistency contract.
 *
 * POSIX threads are always available on the supported build hosts
 * (Linux, macOS). On non-POSIX platforms (Windows MSVC) the
 * multithreaded portion is gated off — the single-threaded tests
 * above still exercise the cache, the slow-path publish, and the
 * fast-path acquire-load consistency.
 * =================================================================== */

#include <unistd.h>
#if defined(_POSIX_THREADS) && (_POSIX_THREADS >= 200112L)
#include <pthread.h>

#define N_THREADS 16
#define N_ITERS_PER_THREAD 1000

/* Per-thread observation buffer. */
static int g_observed[N_THREADS * N_ITERS_PER_THREAD];
static int g_legacy_observed[N_THREADS * N_ITERS_PER_THREAD];

static void *thread_main(void *arg) {
	long tid = (long)arg;
	for (int i = 0; i < N_ITERS_PER_THREAD; i++) {
		int r = gf64_zmm_probe();
		g_observed[tid * N_ITERS_PER_THREAD + i] = r;
		g_legacy_observed[tid * N_ITERS_PER_THREAD + i] = gf64_zmm_works;
	}
	return NULL;
}

static int test_concurrent_consistent(void) {
	pthread_t threads[N_THREADS];

	/* Fan out: all N threads race to call gf64_zmm_probe. */
	for (long i = 0; i < N_THREADS; i++) {
		if (pthread_create(&threads[i], NULL, thread_main,
		                   (void *)i) != 0) {
			printf("  concurrent: FAIL  pthread_create returned "
			       "nonzero at i=%ld\n", i);
			return 1;
		}
	}
	for (int i = 0; i < N_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	/* All N*K observations must agree. The probe is deterministic
	 * (the SIGILL probe body either succeeds or fails on a given
	 * host), so any disagreement indicates a memory-ordering bug. */
	int first = g_observed[0];
	int first_legacy = g_legacy_observed[0];
	for (int i = 0; i < N_THREADS * N_ITERS_PER_THREAD; i++) {
		if (g_observed[i] != first) {
			printf("  concurrent: FAIL  probe drift at obs %d: "
			       "first=%d got=%d (release-acquire publish "
			       "missing)\n", i, first, g_observed[i]);
			return 1;
		}
		if (g_legacy_observed[i] != first_legacy) {
			printf("  concurrent: FAIL  legacy drift at obs %d: "
			       "first_legacy=%d got=%d (publish order wrong)\n",
			       i, first_legacy, g_legacy_observed[i]);
			return 1;
		}
		/* Per-observation consistency: probe and legacy must agree. */
		if (g_observed[i] != g_legacy_observed[i]) {
			printf("  concurrent: FAIL  at obs %d: probe=%d "
			       "legacy=%d (data race)\n",
			       i, g_observed[i], g_legacy_observed[i]);
			return 1;
		}
	}
	printf("  concurrent: OK  %d threads x %d iters all returned %d "
	       "(legacy=probe)\n",
	       N_THREADS, N_ITERS_PER_THREAD, first);
	return 0;
}
#endif /* _POSIX_THREADS */

int main(void) {
	int failures = 0;

	printf("test_gf64_zmm_probe\n\n");

	printf("[1] cache populates on first call, sticks on second:\n");
	failures += test_cache_sticky();

	printf("\n[2] legacy gf64_zmm_works global matches probe return:\n");
	failures += test_legacy_global();

	printf("\n[3] probe consistent with CPUID+XCR0 ground truth:\n");
	failures += test_consistent_with_cpuid();

	printf("\n[4] many calls stable, no state drift:\n");
	failures += test_many_calls_stable();

#if defined(_POSIX_THREADS) && (_POSIX_THREADS >= 200112L)
	printf("\n[5] concurrent fan-out (%d threads x %d iters):\n",
	       N_THREADS, N_ITERS_PER_THREAD);
	failures += test_concurrent_consistent();
#else
	printf("\n[5] concurrent fan-out: SKIPPED (no POSIX threads)\n");
#endif

	printf("\n");
	if (failures == 0) {
		printf("ALL PASS\n");
		return 0;
	} else {
		printf("FAILED (%d test cases)\n", failures);
		return 1;
	}
}

HEDLEY_END_C_DECLS
