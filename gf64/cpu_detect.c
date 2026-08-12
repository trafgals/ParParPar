/*
 * ============================================================================
 * gf64/cpu_detect.c — CPU feature detection with SIGILL probe
 *
 * THE WSL2/HYPER-V "OBSERVER EFFECT"
 * -----------------------------------------------------------------------
 * WSL2 (and other Hyper-V guests) have a documented bug where the
 * hypervisor monitors whether the running binary contains any AVX-512
 * instructions; if it does, the hypervisor MASKS the AVX-512 feature
 * bits in CPUID. The effect is: binary presence of AVX-512 → guest
 * CPUID reports NO AVX-512, even on hardware that natively has it.
 *
 * This means the historical "CPUID says AVX-512 → dispatch AVX-512"
 * logic CANNOT work for any binary that contains AVX-512 instructions
 * elsewhere (e.g., gf64_region_avx512.c). The detection sees a binary
 * full of ZMM instructions → CPUID returns no AVX-512 → we dispatch
 * the slower AVX-2 path even though ZMM would actually run fine.
 *
 * See:  https://github.com/microsoft/WSL/issues/14436
 *       https://github.com/microsoft/WSL/issues/3827
 *
 * THREE-LAYERED DEFENSE
 * -----------------------------------------------------------------------
 * (0) ARCHITECTURAL ISOLATION (parpar_gf64 target + binding.gyp):
 *     All CPUID + dispatch logic lives in this TU. A per-target
 *     `-mno-avx512f` override on the `parpar_gf64` build target (binding.gyp) ensures
 *     NO AVX-512 instructions are emitted in this file's codegen
 *     EXCEPT inside the SIGILL probe (`try_zmm_insn`, marked with
 *     `__attribute__((target("avx512f")))`). The rest of the kernel
 *     (gf64_region_avx512.c etc.) keeps its existing per-function
 *     `target("avx512f")` attributes — those files DO emit ZMM and
 *     are unaffected by the override. This keeps the WSL2 detection
 *     contract ("the binary contains ZMM") intact for the actual
 *     compute kernels, while allowing the detection TU to read CPUID
 *     truthfully via its own internal architecture.
 *
 * (1) SIGILL PROBE (this file, T0):
 *     Even with CPUID isolation, defense-in-depth requires verifying
 *     that ZMM actually executes at runtime. The probe emits a single
 *     ZMM instruction under a SIGILL handler; if it returns normally,
 *     ZMM works; if SIGILL fires, we fall through to AVX-2. This guards
 *     against any future change to the WSL2/Hyper-V detection contract
 *     — e.g., a hypervisor that masks CPUID only partially, or that
 *     fakes XCR0 without honouring lazy XSAVE state loading.
 *
 * (2) ENV VAR OVERRIDE (T3):
 *     PAR3_GF64_USE_AVX512 = 0|1 forces the detection result, for
 *     operators who know the host can/can't run AVX-512. Stub-only in
 *     T1, parser in T3.
 * ============================================================================
 */

#include "gf64_global.h"
#include <string.h>

#ifndef __GNUC__
/* Stub out GCC __attribute__((target(...))) under MSVC.
 * Use variadic macro so the entire trailing ((...)) parens are eaten as
 * a single comma-separated argument list. */
#define __attribute__(...) /* __attribute__ not supported under MSVC */
#endif


/* POSIX signal/sigsetjmp machinery is GCC/POSIX-only. Windows MSVC lacks
 * sigjmp_buf, sigsetjmp, siglongjmp, sigaction etc. The SIGILL probe is
 * a defence-in-depth layer for WSL2/Hyper-V hosts (Linux GCC); Windows
 * builds skip it and trust CPUID+XCR0 (the dispatch still works correctly
 * on Windows because Windows is never the WSL2 observer-effect context). */
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#include <setjmp.h>
#include <signal.h>
#endif

HEDLEY_BEGIN_C_DECLS

/* ----- CPUID + XCR0 wrappers (copied verbatim from gf64_dispatch.c) ----- */

static void gf64_cpuid(int leaf, int subleaf, unsigned int *eax, unsigned int *ebx, unsigned int *ecx, unsigned int *edx) {
#if defined(_MSC_VER)
	int info[4];
	__cpuidex(info, leaf, subleaf);
	*eax = (unsigned int)info[0];
	*ebx = (unsigned int)info[1];
	*ecx = (unsigned int)info[2];
	*edx = (unsigned int)info[3];
#elif defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
	__asm__ __volatile__ (
		"mov %%ebx, %%esi\n	"
		"cpuid\n	"
		"mov %%esi, %%ebx"
		: "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
		: "a"(leaf), "c"(subleaf)
		: "esi", "memory"
	);
#else
	*eax = leaf;
	*ebx = 0;
	*ecx = subleaf;
	*edx = 0;
#endif
}

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
static inline uint64_t gf64_xgetbv(uint32_t xcr) {
	uint32_t lo, hi;
	__asm__ __volatile__ (
		"xgetbv"
		: "=a"(lo), "=d"(hi)
		: "c"(xcr)
	);
	return ((uint64_t)hi << 32) | lo;
}
#elif defined(_MSC_VER)
#include <intrin.h>
static inline uint64_t gf64_xgetbv(uint32_t xcr) {
	return (uint64_t)_xgetbv(xcr);
}
#else
static inline uint64_t gf64_xgetbv(uint32_t xcr) {
	(void)xcr;
	return 0;
}
#endif

/* ----- SIGILL probe machinery (Layer 1) -----
 *
 * The probe runs a single ZMM instruction under a SIGILL handler. On a
 * host that cannot execute ZMM (real AVX-512 hardware masked by
 * WSL2/Hyper-V, or a CPU that lacks AVX-512 entirely), the kernel
 * delivers SIGILL and our handler longjmps back to sigsetjmp's save
 * point; `try_zmm_insn` then returns 0 and the caller falls through
 * to AVX-2. On a real AVX-512 host, the instruction executes normally
 * and we return 1.
 *
 * sigsetjmp/siglongjmp (NOT setjmp/longjmp) preserve the signal mask —
 * SIGILL is delivered at any point inside the probe and we need the
 * signal-mask state to be intact when the handler restores control.
 */

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
static __thread sigjmp_buf zmm_probe_jmp;
static __thread volatile sig_atomic_t zmm_probe_active = 0;

static void gf64_sigill_handler(int sig) {
	(void)sig;
	if (zmm_probe_active) {
		/* Jump back into try_zmm_insn; the caller's sigsetjmp returns
		 * non-zero and the probe interprets that as "ZMM did not run". */
		siglongjmp(zmm_probe_jmp, 1);
	}
	/* SIGILL wasn't from our probe — restore the default disposition
	 * and re-raise so the process fails cleanly rather than spinning
	 * in our handler. (Defensive: should be unreachable in practice.) */
	signal(SIGILL, SIG_DFL);
	raise(SIGILL);
}

/* Probe a single ZMM instruction. The function is the ONLY place in
 * this TU where AVX-512 codegen may appear; the file-level
 * `-mno-avx512f` override (binding.gyp, `parpar_gf64` target) keeps all non-kernel
 * TUs clean of ZMM opcodes; the per-function `__attribute__((target("avx512f")))`
 * independently enables AVX-512 for the kernel functions that need it.
 *
 * Returns 1 if the ZMM instruction executed without SIGILL, 0 otherwise. */
__attribute__((target("avx512f")))
static int try_zmm_insn(void) {
	struct sigaction sa, old_sa;
	sa.sa_handler = gf64_sigill_handler;
	sigemptyset(&sa.sa_mask);
	/* SA_NODEFER NOT set → SIGILL is blocked while the handler runs, so
	 * an immediate SIGILL in the handler body (shouldn't happen, but
	 * defensive) won't recurse. */
	sa.sa_flags = 0;
	
	if (sigaction(SIGILL, &sa, &old_sa) == -1) {
		return 0;
	}
	
	int ok;
	zmm_probe_active = 1;
	if (sigsetjmp(zmm_probe_jmp, 1) == 0) {
		/* Minimal ZMM instruction: vpaddd zmm0, zmm0, zmm0 is a single
		 * EVEX-encoded integer add on ZMM register 0. The result is
		 * thrown away; we only care whether the CPU will execute it
		 * without SIGILL. The "memory" clobber forces the compiler not
		 * to elide the inline asm. */
		__asm__ __volatile__ (
			"vpaddd %%zmm0, %%zmm0, %%zmm0"
			::: "zmm0", "memory"
		);
		ok = 1;
	} else {
		/* siglongjmp returned to sigsetjmp's save point — ZMM failed. */
		ok = 0;
	}
	zmm_probe_active = 0;
	
	/* Always restore the caller's SIGILL handler, even on SIGILL. */
	(void)sigaction(SIGILL, &old_sa, NULL);
	return ok;
}
#else
/* Non-GCC fallback (Windows MSVC, Clang, Intel CC): use Structured
 * Exception Handling to probe a single ZMM instruction at runtime.
 * This is the MSVC equivalent of the POSIX SIGILL probe above: just
 * as `sigaction(SIGILL, ...)` + `sigsetjmp`/`siglongjmp` let us catch
 * an illegal-instruction trap on POSIX, MSVC's `__try` / `__except`
 * lets us install a frame-local filter that the OS unwinds into when
 * an `EXCEPTION_ILLEGAL_INSTRUCTION` is raised by a ZMM op. We use
 * `EXCEPTION_ILLEGAL_INSTRUCTION` as the filter — NOT a catch-all
 * `EXCEPTION_EXECUTE_HANDLER` — so that access violations, stack
 * overflows, divide-by-zero, and the like continue to be surfaced to
 * the OS's default handler (and ultimately the debugger). The filter
 * queries the exception code via `GetExceptionCode()`, which is the
 * only SEH-blessed way to discriminate exception types from a `__try`
 * body; `GetLastError()` / thread-local state would race with any
 * subsequent API call in the same frame. The trailing
 * `_mm256_zeroupper()` is purely a compiler-elision guard: it reads
 * (and clears) the upper YMM/ZMM state as a side-effect, so the ZMM
 * intrinsic in the `__try` body cannot be dead-coded even though its
 * result is thrown away via `(void)z`. On AVX-512 hardware this is a
 * benign no-op; on a hypervisor that masks CPUID but traps ZMM ops,
 * the EXCEPTION_ILLEGAL_INSTRUCTION unwinds out of the `__try` body
 * before `_mm256_zeroupper()` runs and we return 0. */
#if defined(_M_AMD64) || defined(_M_X64)
#include <excpt.h>      /* EXCEPTION_EXECUTE_HANDLER, EXCEPTION_CONTINUE_SEARCH,
                         * GetExceptionCode — MSVC CRT SEH declarations.
                         * Lighter than <windows.h> (no WINBASEAPI / min/max). */
#include <immintrin.h>  /* _mm512_add_epi32 — emits a single EVEX-encoded ZMM op */

/* EXCEPTION_ILLEGAL_INSTRUCTION lives in <winnt.h>/<windows.h>, not <excpt.h>.
 * Define locally to avoid pulling in <windows.h>. Value is the stable NTSTATUS
 * STATUS_ILLEGAL_INSTRUCTION; see
 * https://learn.microsoft.com/en-us/windows/win32/debug/exception-record */
#ifndef EXCEPTION_ILLEGAL_INSTRUCTION
#define EXCEPTION_ILLEGAL_INSTRUCTION 0xC000001DL
#endif

static int try_zmm_insn(void) {
	__try {
		/* Minimal ZMM instruction: a single EVEX-encoded integer add on
		 * ZMM register 0. The intrinsic compiles to exactly one ZMM op
		 * (vpaddd zmm0, zmm0, zmm0 equivalent). The result is thrown
		 * away; we only care whether the CPU will execute it without
		 * EXCEPTION_ILLEGAL_INSTRUCTION. */
		__m512i z = _mm512_add_epi32(_mm512_setzero_si512(), _mm512_setzero_si512());
		/* Force the compiler not to elide the ZMM op. */
		_mm256_zeroupper();  /* clears upper YMM/ZMM state — benign no-op on AVX-512 hardware */
		(void)z;
		return 1;
	} __except ((GetExceptionCode() == EXCEPTION_ILLEGAL_INSTRUCTION) ?
	            EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
		/* ZMM execution raised an illegal-instruction exception.
		 * This is the WSL2/Hyper-V observer-effect case where CPUID
		 * reports AVX-512 but the hypervisor will SIGILL on any ZMM op.
		 * Fall through to AVX-2 dispatch. */
		return 0;
	}
}
#else
/* Non-x64 MSVC (32-bit x86, ARM): no ZMM hardware; return 0 to fall through. */
static int try_zmm_insn(void) {
	return 0;
}
#endif
#endif

/* ----- Public ZMM probe (cubic review 4914681432 P1, 4915459866 P1) -----
 *
 * Cached one-shot wrapper around the static try_zmm_insn() above. The
 * probe is the runtime defence against WSL2/Hyper-V hosts which report
 * AVX-512F in CPUID but SIGILL on the actual ZMM instruction. The
 * cached result is exposed as gf64_zmm_works for code paths that
 * require a working ZMM (e.g. gf64_barycentric_weights which calls
 * gf64_invert_ita_batch — VPCLMULQDQ + ZMM unconditionally).
 *
 * The cache is process-lifetime. Multiple callers (gf64_init_dispatch,
 * gf64_apply_method) share the same probe result, paying the SIGILL
 * handler cost only once.
 *
 * Thread safety (cubic review 4915459866 P1): on POSIX, the SIGILL
 * handler is a process-wide resource — if thread A's probe is in
 * flight and thread B takes an unrelated SIGILL (e.g. an off-the-shelf
 * third-party library that overflows a buffer), thread A's handler
 * would longjmp to thread A's per-thread sigsetjmp save point. The
 * probe and its handler installation must therefore be one-time per
 * process.
 *
 * On Windows MSVC, try_zmm_insn is a no-op (returns 0 — no ZMM
 * hardware on 32-bit x86 or ARM), so the SIGILL-handler concern does
 * not apply; the cache write is benign either way. We still use the
 * same portable guard for consistency.
 *
 * Implementation: portable "init exactly once" using a flag protected
 * by the compiler's atomic CAS. On GCC/Clang this is __atomic_*; on
 * MSVC this is _InterlockedCompareExchange. Both are intrinsics and
 * require no external header. We deliberately do NOT use
 * pthread_once (not available on Windows) or C11 call_once (not
 * available in c99 mode that this TU compiles under). */
static int zmm_probe_cached = -1;  /* -1 = not yet probed, 0 = no, 1 = yes */
int gf64_zmm_works = 0;

#if defined(_MSC_VER)
#include <intrin.h>
#define gf64_atomic_cas(ptr, expected, desired) \
	(_InterlockedCompareExchange((volatile long*)(ptr), (long)(desired), (long)(*(expected))) == (long)(*(expected)))
#else
#define gf64_atomic_cas(ptr, expected, desired) \
	__atomic_compare_exchange_n((ptr), (expected), (desired), 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
#endif

/* One-shot guard for the probe body. The first thread to CAS the flag
 * from 0 to 1 runs the probe; concurrent CAS attempts fail and skip
 * the probe. The cache is then populated by exactly one thread. */
static volatile int zmm_probe_once = 0;

int gf64_zmm_probe(void) {
	/* Fast path: cache hit. */
	if (zmm_probe_cached != -1) {
		return zmm_probe_cached;
	}
	/* Slow path: try to be the one thread that runs the probe. */
	int expected = 0;
	if (gf64_atomic_cas(&zmm_probe_once, &expected, 1)) {
		int r = try_zmm_insn();
		zmm_probe_cached = r;
		gf64_zmm_works = r;
		return r;
	}
	/* Another thread is running the probe; spin-wait briefly for the
	 * cache to be populated. The probe is short (single vpaddd zmm
	 * instruction under a SIGILL handler) so the wait is bounded. */
	while (zmm_probe_cached == -1) {
		/* Hint that we're spinning. */
#if defined(_MSC_VER)
		_mm_pause();
#else
		__builtin_ia32_pause();
#endif
	}
	return zmm_probe_cached;
}

/* ----- Detection entry point (exported, called by gf64_dispatch.c post-T1) -----
 *
 * Mirrors the body of gf64_dispatch.c's static gf64_detect_method_internal,
 * with one addition: after the CPUID+XCR0 check confirms AVX-512F+VPOPCNTDQ
 * +OSXSAVE+ZMM/YMM/XMM state, we call try_zmm_insn() as Layer 1. If the
 * probe SIGILLs, we fall through to the AVX-2 branch instead of trusting
 * CPUID+XCR0 alone.
 *
 * NOTE: in T0, gf64_dispatch.c still defines its OWN static copy of
 * gf64_detect_method_internal; this exported copy is dormant until T1
 * removes the static one and rewires gf64_detect_method() to call here.
 */
GF64Method gf64_detect_method_internal(void) {
	unsigned int eax, ebx, ecx, edx;

	/* Check AVX-512F (cpuid 7.0 EBX bit 16) + VPOPCNTDQ (cpuid 7.0 ECX bit 14) */
	gf64_cpuid(7, 0, &eax, &ebx, &ecx, &edx);
	if ((ebx & (1 << 16)) && (ecx & (1 << 14))) {
		/* Confirm OS support: OSXSAVE (cpuid 1.0 ECX bit 27) + XCR0 ZMM/YMM/XMM (bits 5,2,1,0) */
		gf64_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
		if (ecx & (1 << 27)) {
			uint64_t xcr0 = gf64_xgetbv(0);
			/* XCR0 bits 0 (SSE), 1 (AVX YMM), 2 (AVX-512 opmask), 5 (AVX-512 ZMM/H) must all be set */
			if ((xcr0 & 0x27ULL) == 0x27ULL) {
				/* CPUID+XCR0 say AVX-512 is supported. Layer 1: actually
				 * probe ZMM execution to defeat any hypervisor that masks
				 * CPUID only partially or fakes XCR0 without honouring
				 * lazy XSAVE state loading. If SIGILL fires, fall through. */
				if (gf64_zmm_probe()) {
					/* (cubic review 4915459866 P2): route through the
					 * cached probe wrapper instead of try_zmm_insn()
					 * directly. Detection is called 5+ times via
					 * gf64_detect_method's poll aggregate; routing the
					 * SIGILL probe through the cache makes it run
					 * exactly once per process. The Layer 2 (VPCLMULQDQ
					 * gate) and other behaviour are unchanged. */
					/* Layer 2 (cubic review 4910826158, P1+x): the
					 * AVX-512 codepath is gated on VPCLMULQDQ support,
					 * even for routines that don't use PCLMULQDQ directly.
					 * Reason: gf64_region_muladd_arr[_avx512] is __attribute__
					 * __target__("avx512f,vpclmulqdq") and SIGILLs on
					 * AVX-512-only-no-VPCLMULQDQ hosts (e.g. AVX-512 VNNI
					 * without the carryless-multiply quadword extension).
					 * Without this downgrade, dispatching AVX-512 here
					 * selects the VPCLMULQDQ codepath for the region
					 * muladd which then SIGILLs at runtime in
					 * gf64_inverse_batch / gf64_invert_ita_batch / the
					 * polynomial-mul butterfly. Force a downgrade to AVX-2
					 * if VPCLMULQDQ is absent. */
					if (gf64_has_vpclmulqdq_probe()) {
						return GF64_AVX512;
					}
					/* Fall through to AVX-2 detection below. */
				}
			}
		}
	}

	gf64_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
	if ((ecx & (1 << 28)) && (ecx & (1 << 12)) && (ecx & (1 << 27))) {
		return GF64_AVX2;
	}

	gf64_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
	if ((ecx & (1 << 0)) && (ecx & (1 << 1))) {
		return GF64_SSSE3;
	}

	return GF64_SCALAR;
}

/* Probe VPCLMULQDQ (cpuid 7.0 ECX bit 10). This is a host-availability flag
 * distinct from the workload-chosen gf64_current_method: PD2 can downgrade
 * the latter to GF64_AVX2 mid-workload to avoid the Zen4 downclock, but the
 * ISA capability itself is fixed. Code paths that call
 * __attribute__((target("avx512f,vpclmulqdq"))) functions MUST gate on the
 * return value of this probe, NOT on gf64_current_method.
 *
 * Returns 1 if CPUID+XCR0 report VPCLMULQDQ available, 0 otherwise. Does NOT
 * perform the ZMM SIGILL probe — VPCLMULQDQ without working ZMM is moot,
 * but the call sites that need VPCLMULQDQ also need working ZMM, so the
 * cost is paid elsewhere (the gf64_current_method==GF64_AVX512 path or the
 * per-function ZMM probe).
 *
 * IMPORTANT (cubic review 4914681432 P1): callers that route to VPCLMULQDQ
 * code paths AND the gf64_current_method dispatch state independently
 * (e.g. gf64_barycentric_weights) must additionally require that
 * gf64_current_method == GF64_AVX512. CPUID+XCR0 alone is NOT sufficient
 * on WSL2/Hyper-V hosts which report VPCLMULQDQ via CPUID but SIGILL on
 * the actual ZMM instruction. See gf64_init_dispatch() and the
 * gf64_has_vpclmulqdq global for the runtime check pattern. */
int gf64_has_vpclmulqdq_probe(void) {
	unsigned int eax, ebx, ecx, edx;

	/* VPCLMULQDQ = cpuid leaf 7, sub-leaf 0, ECX bit 10.
	 *
	 * NOT bit 11 — that is AVX512_VNNI (VPDPBUSD etc.). The drift from
	 * 10 to 11 was a copy-paste bug present since the probe was first
	 * introduced; the consequence is that hosts with VPCLMULQDQ +
	 * AVX512_F but WITHOUT AVX512_VNNI (rare today but legal per Intel SDM)
	 * would take the fallback path and miss the PCLMULQDQ acceleration,
	 * while hosts with AVX512_VNNI but no actual VPCLMULQDQ
	 * implementation would falsely enter the VPCLMULQDQ codepath and
	 * SIGILL on the first AVX-512 PCLMULQDQ instruction.
	 *
	 * Cross-checked against:
	 *   - Linux kernel X86_FEATURE_VPCLMULQDQ = 16*32+10 (bit 10, leaf 7.0
	 *     ECX). See arch/x86/include/asm/cpufeatures.h.
	 *   - Intel SDM Vol. 2, CPUID instruction table for leaf 7 sub-leaf 0.
	 *   - Intel Intrinsics Guide entry for `_mm512_clmulepi64_epi128`,
	 *     which documents the same bit.
	 *
	 * Other ECX bits in this sub-leaf for disambiguation:
	 *   - bit 9  = SSBD (Speculative Store Bypass Disable)
	 *   - bit 10 = AVX512_VL? NO — bit 10 is VPCLMULQDQ (this probe).
	 *   - bit 11 = AVX512_VNNI
	 *   - bit 12 = AVX512_BITALG
	 *   - bit 14 = AVX512_VPOPCNTDQ
	 */
	gf64_cpuid(7, 0, &eax, &ebx, &ecx, &edx);
	if (!(ecx & (1 << 10))) return 0;

	/* XCR0 mask 0xE6 = bits 1, 2, 5, 6, 7:
	 *   bit 1 = SSE state (XMM)
	 *   bit 2 = AVX state (YMM)
	 *   bit 5 = AVX-512 opmask (k0..k7)
	 *   bit 6 = AVX-512 ZMM_HI256 (high 256 of each ZMM register)
	 *   bit 7 = AVX-512 Hi16_ZMM (ZMM16..ZMM31)
	 *
	 * Without bits 6 and 7 set, OS_XSAVE is reported but ZMM
	 * instructions SIGILL even when CPUID advertises VPCLMULQDQ.
	 * (The legacy mask 0x27 = bits 0,1,2,5 used elsewhere in this file
	 * is sufficient for AVX-512F detection but not enough to actually
	 * run ZMM — see XSAVE state-component requirements in the SDM.) */
	gf64_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
	if (!(ecx & (1 << 27))) return 0;  /* OSXSAVE */
	{
		uint64_t xcr0 = gf64_xgetbv(0);
		if ((xcr0 & 0xE6ULL) != 0xE6ULL) return 0;
	}
	return 1;
}

HEDLEY_END_C_DECLS