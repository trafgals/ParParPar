#ifndef __GF64_GLOBAL_H
#define __GF64_GLOBAL_H

#include "../src/hedley.h"
#include "../src/stdint.h"
#include "../src/platform.h"

#define GF64_POLYNOMIAL 0x100000000000001BULL
#define UNUSED(...) (void)(__VA_ARGS__)
#define MAX_STACK_BUF 256

HEDLEY_BEGIN_C_DECLS

typedef uint64_t gf64_t;

typedef enum {
	GF64_AVX512=0,
	GF64_AVX2=1,
	GF64_SSSE3=2,
	GF64_SCALAR=3
} GF64Method;

typedef void (*gf64_region_mul_fn)(gf64_t *HEDLEY_RESTRICT out, const gf64_t *HEDLEY_RESTRICT in, size_t len, gf64_t constant);
typedef void (*gf64_region_mul_arr_fn)(gf64_t *HEDLEY_RESTRICT out, const gf64_t *HEDLEY_RESTRICT in, const gf64_t *HEDLEY_RESTRICT coeff, size_t len, size_t n_coeff);
typedef void (*gf64_region_muladd_arr_fn)(gf64_t *HEDLEY_RESTRICT out, const gf64_t *HEDLEY_RESTRICT in, const gf64_t *HEDLEY_RESTRICT coeff, size_t len, size_t n_coeff);
typedef void (*gf64_region_coupled_muladd_arr_fn)(gf64_t *HEDLEY_RESTRICT out, const gf64_t *HEDLEY_RESTRICT *HEDLEY_RESTRICT in_blocks, const gf64_t *HEDLEY_RESTRICT coeff_blocks, size_t len, size_t G);
typedef void (*gf64_region_fused_output_muladd_arr_fn)(gf64_t *HEDLEY_RESTRICT *HEDLEY_RESTRICT outs, const gf64_t *HEDLEY_RESTRICT in, const gf64_t *HEDLEY_RESTRICT *HEDLEY_RESTRICT coeff_block_starts, size_t len, size_t K);
typedef void (*gf64_region_2d_muladd_arr_fn)(gf64_t *HEDLEY_RESTRICT *HEDLEY_RESTRICT outs, size_t K, const gf64_t *HEDLEY_RESTRICT *HEDLEY_RESTRICT in_blocks, size_t G, const gf64_t *HEDLEY_RESTRICT coeff_block_2d, size_t K_stride, size_t len);
typedef void (*gf64_inverse_batch_fn)(gf64_t *HEDLEY_RESTRICT out, const gf64_t *HEDLEY_RESTRICT in, size_t N);

extern gf64_region_mul_fn gf64_region_mul;
extern gf64_region_mul_arr_fn gf64_region_mul_arr;
extern gf64_region_muladd_arr_fn gf64_region_muladd_arr;
extern gf64_region_coupled_muladd_arr_fn gf64_region_coupled_muladd_arr;
extern gf64_region_fused_output_muladd_arr_fn gf64_region_fused_output_muladd_arr;
extern gf64_region_2d_muladd_arr_fn gf64_region_2d_muladd_arr;
extern gf64_inverse_batch_fn gf64_inverse_batch;
extern GF64Method gf64_current_method;
/* Host-availability flag for VPCLMULQDQ (carryless multiplication on
 * AVX-512F). Set once by gf64_init_dispatch from CPUID leaf 7 sub-leaf 0
 * ECX bit 10 + XCR0 mask 0xE6 + a successful ZMM SIGILL probe.
 * Distinct from gf64_current_method, which is workload-chosen
 * (PD2 downclock heuristic can downgrade it to GF64_AVX2 even on a host
 * that supports VPCLMULQDQ). Code paths that call into
 * __attribute__((target("avx512f,vpclmulqdq"))) functions (ITA batch
 * invert, HQC FFT butterfly AVX-512) MUST gate on this flag, NOT on
 * gf64_current_method.
 *
 * Note (cubic review 4914681432 P1): the flag is "fully functional
 * VPCLMULQDQ" — it requires the ZMM SIGILL probe to also have
 * succeeded, not just CPUID+XCR0. WSL2/Hyper-V hosts that report
 * VPCLMULQDQ via CPUID+XCR0 but SIGILL on the actual ZMM
 * instruction will have this flag = 0 even when the underlying CPU
 * genuinely has the ISA. */
extern int gf64_has_vpclmulqdq;
/* Probe function used by gf64_init_dispatch. Reads cpuid 7.0 ECX bit 10
 * + XCR0 — returns 1 if VPCLMULQDQ is available per CPUID, 0 otherwise.
 * Does NOT perform the ZMM SIGILL probe (that's gf64_zmm_probe()). */
int gf64_has_vpclmulqdq_probe(void);

/* One-shot ZMM SIGILL probe (cubic review 4914681432 P1). Runs a single
 * ZMM instruction (vpaddd zmm0, zmm0, zmm0) under a SIGILL handler and
 * caches the result. Returns 1 if the ZMM instruction executed without
 * SIGILL, 0 if SIGILL fired.
 *
 * The probe is cached: the first call actually runs the SIGILL probe;
 * subsequent calls return the cached result. This makes it cheap to
 * call from hot paths (e.g. dispatch rebinds) without paying the signal
 * handler cost repeatedly.
 *
 * On non-x64 / non-MSVC builds, the probe returns 0 (no ZMM hardware). */
int gf64_zmm_probe(void);

/* Set to the cached result of gf64_zmm_probe(). 1 if a real ZMM
 * instruction has executed successfully on this host, 0 otherwise. */
extern int gf64_zmm_works;
extern gf64_t gf64_inverse(gf64_t a);

GF64Method gf64_detect_method(void);
int gf64_init_dispatch(void);

/* PD2: pick the optimal GF(2^64) ISA for a specific workload. On Zen4,
 * AVX-512 triggers a 2x frequency downclock; if the working set exceeds
 * 16 MiB, AVX-2 wins despite lower per-instruction throughput because it
 * keeps the nominal frequency. Env override PAR3_AVX512_FORCE=1|0 bypasses
 * the heuristic. Returns the chosen method; does NOT bind the dispatch
 * function pointers — caller must call gf64_apply_method() for that. */
GF64Method gf64_method_for_workload(size_t num_in, size_t num_out, size_t block_size);

/* Bind the global GF(2^64) function pointers to the given method.
 * No env lookup, no detection; idempotent. */
void gf64_apply_method(GF64Method method);

HEDLEY_END_C_DECLS

#endif