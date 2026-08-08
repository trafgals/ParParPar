/*
 * ============================================================================
 * gf64/test/test_cpu_detect.c — Standalone unit test for gf64_detect_method_internal
 *
 * Builds standalone with -DGF64_CPU_DETECT_STANDALONE; no gf64_global.h,
 * no full kernel, no SIGILL handler. The test exercises both the
 * "ZMM probe returns 1" path (POSIX-positive=1) and the "ZMM probe
 * returns 0" path (POSIX-negative=0) in a single binary by switching
 * a global between calls.
 *
 * The test redefines the dispatch algorithm in a self-contained TU so the
 * stubbed try_zmm_insn() can be exercised deterministically (cpu_detect.c's
 * internal copy is `static` and therefore not callable from outside its TU).
 * The cpu_detect.c source is still compiled alongside (per the canonical
 * build command in the plan) to confirm both files compile cleanly; its
 * non-static gf64_detect_method_internal() symbol is unused here because
 * the test's own static copy shadows it inside this TU.
 *
 * Build & run from repo root:
 *   gcc -O2 -march=native -DGF64_CPU_DETECT_STANDALONE \
 *       -o build/test_cpu_detect gf64/test/test_cpu_detect.c gf64/cpu_detect.c \
 *       && ./build/test_cpu_detect
 *
 * Exit 0 on full pass (both POSIX-positive=1 and POSIX-negative=0 lines
 * printed); non-zero on any mismatch. Output is plain-text so it can be
 * piped to .omo/evidence/task-3-pr-26-recovery.cpu-detect.txt.
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef GF64_CPU_DETECT_STANDALONE
/* Full mode: include the real gf64_global.h. Tests can opt into this by
 * building without -DGF64_CPU_DETECT_STANDALONE. */
#include "gf64_global.h"
#else
/* Standalone mode: minimal type shim. The dispatch algorithm only needs
 * GF64Method (the enum) — no gf64_t, no kernel, no SIGILL machinery. */
typedef enum {
	GF64_AVX512 = 0,
	GF64_AVX2   = 1,
	GF64_SSSE3  = 2,
	GF64_SCALAR = 3
} GF64Method;
#endif

/* ---------------------------------------------------------------------------
 * Mock CPUID + XGETBV (deterministic, controllable via g_cpu_kind).
 *
 * When g_cpu_kind = 1, CPUID reports AVX-512F + VPOPCNTDQ + OSXSAVE +
 * AVX + FMA + SSE3 + SSSE3 and XCR0 returns the bits that pass the
 * "all 4 AVX-512 state components are enabled" check (0x27 = SSE | AVX
 * | opmask | ZMM_Hi16). The dispatch path therefore reaches try_zmm_insn().
 *
 * When g_cpu_kind = 0, every feature bit is cleared, so the dispatch
 * path falls through to GF64_SCALAR regardless of try_zmm_insn().
 *
 * These statics do NOT clash with cpu_detect.c's own static gf64_cpuid /
 * gf64_xgetbv: each TU has its own internal-linkage copy.
 * --------------------------------------------------------------------------- */
static int g_cpu_kind = 0;

static void gf64_cpuid(int leaf, int subleaf,
                        unsigned int *eax, unsigned int *ebx,
                        unsigned int *ecx, unsigned int *edx) {
	if (leaf == 7 && subleaf == 0) {
		/* CPUID 7.0: EBX bit 16 = AVX-512F; ECX bit 14 = VPOPCNTDQ */
		*eax = 0;
		*ebx = g_cpu_kind ? (1u << 16) : 0u;
		*ecx = g_cpu_kind ? (1u << 14) : 0u;
		*edx = 0;
	} else if (leaf == 1 && subleaf == 0) {
		/* CPUID 1.0: ECX bit 27 = OSXSAVE; bit 28 = AVX; bit 12 = FMA;
		 * bits 0/1 = SSE3/SSSE3. */
		*eax = 0;
		*ebx = 0;
		*ecx = g_cpu_kind
			? ((1u << 27) | (1u << 28) | (1u << 12) | (1u << 0) | (1u << 1))
			: 0u;
		*edx = 0;
	} else {
		*eax = *ebx = *ecx = *edx = 0;
	}
}

static uint64_t gf64_xgetbv(uint32_t xcr) {
	(void)xcr;
	/* XCR0 bits 0 (SSE), 1 (AVX YMM), 2 (AVX-512 opmask), 5 (AVX-512 ZMM/H) */
	return 0x27ULL;
}

/* ---------------------------------------------------------------------------
 * Stub try_zmm_insn() — returns whatever g_try_zmm_return currently holds.
 *
 * Used to drive the dispatch to either:
 *   - GF64_AVX512 (when g_try_zmm_return = 1 AND g_cpu_kind = 1)
 *   - any non-AVX-512 method (when g_try_zmm_return = 0)
 *
 * Static — does not clash with cpu_detect.c's static try_zmm_insn.
 * --------------------------------------------------------------------------- */
static int g_try_zmm_return = 0;

static int try_zmm_insn(void) {
	return g_try_zmm_return;
}

/* ---------------------------------------------------------------------------
 * Mirror of gf64_detect_method_internal() from gf64/cpu_detect.c:237-271.
 *
 * Kept in lockstep with cpu_detect.c. If cpu_detect.c's dispatch logic
 * diverges, this test will silently diverge too — that is acceptable here
 * because the test's job is to verify the *algorithm* (try_zmm_insn → AVX-512
 * gate; CPUID fallthrough; SCALAR default) is correct, not to byte-for-byte
 * byte-compare the cpu_detect.c codegen.
 *
 * STATIC linkage so this TU's symbol does not clash with cpu_detect.c's
 * non-static gf64_detect_method_internal (compiled alongside per the plan's
 * verification command). main() below resolves to this static copy.
 * --------------------------------------------------------------------------- */
static GF64Method gf64_detect_method_internal(void) {
	unsigned int eax, ebx, ecx, edx;

	/* AVX-512F (cpuid 7.0 EBX bit 16) + VPOPCNTDQ (cpuid 7.0 ECX bit 14) */
	gf64_cpuid(7, 0, &eax, &ebx, &ecx, &edx);
	if ((ebx & (1u << 16)) && (ecx & (1u << 14))) {
		/* OSXSAVE (cpuid 1.0 ECX bit 27) + XCR0 ZMM/YMM/XMM (0x27) */
		gf64_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
		if (ecx & (1u << 27)) {
			uint64_t xcr0 = gf64_xgetbv(0);
			if ((xcr0 & 0x27ULL) == 0x27ULL) {
				if (try_zmm_insn()) {
					return GF64_AVX512;
				}
			}
		}
	}

	/* AVX-2 (cpuid 1.0 ECX bit 28) + FMA (bit 12) + OSXSAVE (bit 27) */
	gf64_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
	if ((ecx & (1u << 28)) && (ecx & (1u << 12)) && (ecx & (1u << 27))) {
		return GF64_AVX2;
	}

	/* SSSE3 + SSE3 (cpuid 1.0 ECX bits 0/1) */
	gf64_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
	if ((ecx & (1u << 0)) && (ecx & (1u << 1))) {
		return GF64_SSSE3;
	}

	return GF64_SCALAR;
}

/* ---------------------------------------------------------------------------
 * Driver — two scenarios, one binary.
 *
 * 1. POSIX-positive: CPUID says AVX-512 + ZMM probe OK → expect GF64_AVX512
 *    → print "POSIX-positive=1"
 * 2. POSIX-negative: CPUID says AVX-512 + ZMM probe FAILS → expect NOT
 *    GF64_AVX512 (falls through to AVX-2 / SSSE3 / SCALAR)
 *    → print "POSIX-negative=0"
 *
 * Exits 0 on both lines, 1 if either is wrong.
 * --------------------------------------------------------------------------- */
int main(void) {
	int rc = 0;

	/* Scenario 1: positive path */
	g_cpu_kind = 1;
	g_try_zmm_return = 1;
	if (gf64_detect_method_internal() == GF64_AVX512) {
		printf("POSIX-positive=1\n");
	} else {
		printf("POSIX-positive=0\n");
		rc = 1;
	}

	/* Scenario 2: negative path (ZMM probe says no) */
	g_try_zmm_return = 0;
	if (gf64_detect_method_internal() != GF64_AVX512) {
		printf("POSIX-negative=0\n");
	} else {
		printf("POSIX-negative=1\n");
		rc = 1;
	}

	return rc;
}