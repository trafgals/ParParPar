#include "par3_engine.h"
#include "platform.h"

#include "gf64_invert.h"

#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <errno.h>

// POSIX mmap(2) / open(2) / fstat(2) / close(2) for the A1 zero-copy
// mmap-based ComputeRecoveryBlocksFromFile entry (PAR3_GF64_USE_MMAP=1).
// Linux/macOS only; Windows builds fall back to file-read via
// Win32 CreateFile/MapViewOfFile (or just a plain read) below.
#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <io.h>
#  include <fcntl.h>
#  include <sys/stat.h>
#  define open  _open
#  define close _close
#  define fstat _fstat
#  define lseek _lseek
#  define O_RDONLY _O_RDONLY
#  define PARPAR_USE_MMAP 0
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  define PARPAR_USE_MMAP 1
#endif

#if defined(_MSC_VER)
#include <malloc.h>
#endif

#ifdef __linux__
#include <sched.h>
#endif

#include <wmmintrin.h>
#include <nmmintrin.h>

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <future>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <list>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include "par3_topology.h"

// ============================================================================
// Dispatch initialisation (one-shot)
// ============================================================================
static bool s_dispatch_initialized = false;

// ============================================================================
// L3 cache size detection
// ----------------------------------------------------------------------------
// Uses cross-platform CpuTopology discovery module (par3_topology.h).
// ============================================================================
static size_t GetL3CacheSize() {
	return GetCpuTopology().l3PerCluster;
}

// ============================================================================
// LRU cache for coefficient matrices
// ----------------------------------------------------------------------------
// Keyed by (numInputs, numRecovery, firstInput, firstRecovery) so repeated
// calls with the same recovery exponents reuse the same matrix.
// ============================================================================
struct CoeffCacheKey {
	size_t numInputs;
	size_t numRecovery;
	uint64_t firstInput;
	uint64_t firstRecovery;

	bool operator==(const CoeffCacheKey& o) const {
		return numInputs == o.numInputs && numRecovery == o.numRecovery &&
		       firstInput == o.firstInput && firstRecovery == o.firstRecovery;
	}
};

struct CoeffCacheKeyHash {
	size_t operator()(const CoeffCacheKey& k) const {
		return std::hash<size_t>()(k.numInputs) ^
		       std::hash<size_t>()(k.numRecovery) ^
		       std::hash<uint64_t>()(k.firstInput) ^
		       std::hash<uint64_t>()(k.firstRecovery);
	}
};

static const size_t COEFF_CACHE_MAX = 8;

static struct {
	std::unordered_map<CoeffCacheKey, gf64_t*, CoeffCacheKeyHash> map;
	std::list<CoeffCacheKey> lru;
} s_coeffCache;

static inline void EnsureDispatch() {
	if (!s_dispatch_initialized) {
		gf64_init_dispatch();
		s_dispatch_initialized = true;
	}
}

// ============================================================================
// Cauchy matrix construction worker count
// ----------------------------------------------------------------------------
// v2: tried 64 — slower than 8 due to std::async spawn overhead on small
// matrix builds. Reverted to 8 which matches the thread count on Zen4
// desktop CPUs.
// ============================================================================
static constexpr size_t kCauchyMaxWorkers = 8;
static size_t s_cauchyWorkerCount = 0;
// ============================================================================
// Effective CPU count  (affinity- and topology-aware)
// ----------------------------------------------------------------------------
// Returns the number of CPUs the process is allowed to run on according to
// OS affinity masks (Linux sched_getaffinity, Windows Job Objects / CPU sets /
// GetProcessGroupAffinity / GetProcessAffinityMask), falling back to
// GetCpuTopology().logicalCores and std::thread::hardware_concurrency().
// Capped at 128 to keep per-worker overhead bounded on high-core-count
// architectures.
// ============================================================================
static std::atomic<size_t> s_effectiveCpuCountCached{0};

void ResetEffectiveCpuCountCache() {
	s_effectiveCpuCountCached.store(0, std::memory_order_release);
}

size_t GetEffectiveCpuCount() {
#ifdef _WIN32
	char mockEffEnv[32];
	DWORD envLen = GetEnvironmentVariableA("PAR3_MOCK_EFFECTIVE_CPUS", mockEffEnv, sizeof(mockEffEnv));
	if (envLen > 0 && envLen < sizeof(mockEffEnv)) {
		int v = std::atoi(mockEffEnv);
		if (v > 0) {
			if (v > 128) v = 128;
			return (size_t)v;
		}
	}
#else
	const char* mockEffEnv = std::getenv("PAR3_MOCK_EFFECTIVE_CPUS");
	if (mockEffEnv && *mockEffEnv) {
		int v = std::atoi(mockEffEnv);
		if (v > 0) {
			if (v > 128) v = 128;
			return (size_t)v;
		}
	}
#endif

	size_t cached = s_effectiveCpuCountCached.load(std::memory_order_acquire);
	if (cached != 0) return cached;

	size_t count = 0;
#ifdef __linux__
	cpu_set_t mask;
	CPU_ZERO(&mask);
	if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
		count = (size_t)CPU_COUNT(&mask);
	}
#elif defined(_WIN32)
	// Track the process's effective affinity bitmask per group to enable
	// exact intersection with CPU sets (and prevent overcounting).
	std::unordered_map<WORD, KAFFINITY> effectiveGroupMasks;

	USHORT groupCount = 0;
	std::vector<USHORT> processGroups;
	if (GetProcessGroupAffinity(GetCurrentProcess(), &groupCount, nullptr) == 0 &&
	    GetLastError() == ERROR_INSUFFICIENT_BUFFER && groupCount > 0) {
		processGroups.resize(groupCount);
		if (!GetProcessGroupAffinity(GetCurrentProcess(), &groupCount, processGroups.data())) {
			processGroups.clear();
		}
	}

	if (processGroups.size() > 1) {
		// Multi-group: check if restricted by a multi-group Job Object first.
		std::vector<GROUP_AFFINITY> jobAffinities;
		DWORD jobLen = 0;
		if (!QueryInformationJobObject(NULL, (JOBOBJECTINFOCLASS)14 /* JobObjectGroupInformationEx */, nullptr, 0, &jobLen) &&
		    (GetLastError() == ERROR_MORE_DATA || GetLastError() == ERROR_INSUFFICIENT_BUFFER) && jobLen > 0) {
			jobAffinities.resize(jobLen / sizeof(GROUP_AFFINITY));
			if (!QueryInformationJobObject(NULL, (JOBOBJECTINFOCLASS)14, jobAffinities.data(), jobLen, &jobLen)) {
				jobAffinities.clear();
			}
		}

		size_t total_aff = 0;
		if (!jobAffinities.empty()) {
			for (USHORT g : processGroups) {
				size_t group_cpus = 0;
				bool found_job_aff = false;
				for (const auto& ga : jobAffinities) {
					if (ga.Group == g) {
						found_job_aff = true;
						KAFFINITY mask = ga.Mask;
						if (mask != 0) {
							effectiveGroupMasks[g] = mask;
							for (KAFFINITY m = mask; m > 0; m &= (m - 1)) group_cpus++;
						} else {
							effectiveGroupMasks[g] = (KAFFINITY)-1;
							group_cpus = (size_t)GetActiveProcessorCount(g);
						}
						break;
					}
				}
				if (!found_job_aff) {
					effectiveGroupMasks[g] = (KAFFINITY)-1;
					group_cpus = (size_t)GetActiveProcessorCount(g);
				}
				total_aff += group_cpus;
			}
		} else {
			// Enumerate RelationGroup entries and count active processors in assigned groups
			DWORD len = 0;
			GetLogicalProcessorInformationEx(RelationGroup, nullptr, &len);
			if (len > 0) {
				std::vector<uint8_t> buffer(len);
				if (GetLogicalProcessorInformationEx(RelationGroup,
				    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &len)) {
					DWORD offset = 0;
					while (offset < len) {
						auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
						if (info->Relationship == RelationGroup) {
							WORD activeGroupCount = info->Group.ActiveGroupCount;
							for (WORD gi = 0; gi < activeGroupCount; gi++) {
								WORD g = gi;
								KAFFINITY mask = info->Group.GroupInfo[gi].ActiveProcessorMask;
								bool processUsesGroup = false;
								for (USHORT pg : processGroups) {
									if (pg == g) { processUsesGroup = true; break; }
								}
								if (!processUsesGroup) continue;
								effectiveGroupMasks[g] = mask;
								for (KAFFINITY m = mask; m != 0; m &= (m - 1)) total_aff++;
							}
						}
						offset += info->Size;
					}
				}
			}
		}
		if (total_aff > 0) count = total_aff;
	} else {
		// Single-group or unassigned process: query process affinity mask first
		WORD primaryGroup = processGroups.empty() ? 0 : processGroups[0];
		DWORD_PTR processMask = 0, systemMask = 0;
		if (GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask) && processMask != 0) {
			size_t aff_count = 0;
			for (DWORD_PTR m = processMask; m > 0; m &= (m - 1)) aff_count++;
			count = aff_count;
			effectiveGroupMasks[primaryGroup] = (KAFFINITY)processMask;
		}
		if (count == 0 && !processGroups.empty()) {
			for (USHORT g : processGroups) {
				size_t gc = (size_t)GetActiveProcessorCount(g);
				if (gc > 0) count += gc;
				effectiveGroupMasks[g] = (KAFFINITY)-1;
			}
		}
	}

	// Cubic review 5b1b83d5 P2:
	// When CPU sets and process/JobObject affinity masks are both specified,
	// enumerate CPU set IDs and intersect with group masks rather than comparing counts.
	// NULL-buffer query returns ERROR_INSUFFICIENT_BUFFER when default CPU sets are configured.
	ULONG cpuSetCount = 0;
	std::vector<ULONG> cpuSetIds;
	if (!GetProcessDefaultCpuSets(GetCurrentProcess(), nullptr, 0, &cpuSetCount) &&
	    GetLastError() == ERROR_INSUFFICIENT_BUFFER && cpuSetCount > 0) {
		cpuSetIds.resize(cpuSetCount);
		if (!GetProcessDefaultCpuSets(GetCurrentProcess(), cpuSetIds.data(), cpuSetCount, &cpuSetCount)) {
			cpuSetIds.clear();
		}
	}

	if (!cpuSetIds.empty()) {
		ULONG sysLen = 0;
		GetSystemCpuSetInformation(nullptr, 0, &sysLen, GetCurrentProcess(), 0);
		if (sysLen > 0) {
			std::vector<uint8_t> sysBuf(sysLen);
			if (GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(sysBuf.data()), sysLen, &sysLen, GetCurrentProcess(), 0)) {
				size_t intersectionCount = 0;
				BYTE* ptr = sysBuf.data();
				while (ptr < sysBuf.data() + sysLen) {
					auto* item = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(ptr);
					if (item->Type == CpuSetInformation) {
						bool inCpuSets = false;
						for (ULONG id : cpuSetIds) {
							if (id == item->CpuSet.Id) { inCpuSets = true; break; }
						}
						if (inCpuSets) {
							WORD grp = item->CpuSet.Group;
							BYTE procIdx = item->CpuSet.LogicalProcessorIndex;
							auto it = effectiveGroupMasks.find(grp);
							if (it != effectiveGroupMasks.end()) {
								KAFFINITY grpMask = it->second;
								if (procIdx < 64 && (grpMask & (KAFFINITY(1) << procIdx))) {
									intersectionCount++;
								}
							}
						}
					}
					ptr += item->Size;
				}
				count = intersectionCount;
			}
		}
	}
#endif



	if (count == 0) {
		count = GetCpuTopology().logicalCores;
	}
	if (count == 0) {
		count = std::thread::hardware_concurrency();
	}
	if (count > 128) count = 128;
	if (count == 0) count = 1;

	s_effectiveCpuCountCached = count;
	return count;
}

// ============================================================================
// GF64Controller::BuildCauchyMatrix
// ----------------------------------------------------------------------------
// For each row r (recovery) and column c (input):
//   M[r][c] = 1/(firstInput^c XOR firstRecovery^r)
//   (denom == 0 is impossible with disjoint ranges, but guard with 1)
//
// Matches the JS implementation at lib/par3gen.js:594-604.
//
// Parallelization: distributes rows across std::async workers via row-stealing
// (std::atomic<size_t> nextRow). Single-row, single-worker, or small-matrix
// (<= 32768 elements) workloads fall through to a serial SIMD-batched loop to
// eliminate thread spawn overhead.
// ============================================================================
void GF64Controller::BuildCauchyMatrix(
	gf64_t* coeffMatrix,
	size_t numInputs, size_t numRecovery,
	uint64_t firstInput, uint64_t firstRecovery
) {
	if (s_cauchyWorkerCount == 0) {
		size_t n = std::min(GetEffectiveCpuCount(),
		                     kCauchyMaxWorkers);
		s_cauchyWorkerCount = (n == 0) ? 1 : n;
	}

	// v2: tried 2D-tile parallel (split columns across workers) — SLOWER
	// than row-stealing because each worker ends up doing numRecovery row
	// passes over its slice, multiplying the dispatch overhead. Reverted
	// to the row-stealing default path below.
	(void)0;

	if (numRecovery <= 1 || s_cauchyWorkerCount <= 1 || (numRecovery * numInputs <= 32768)) {
		// v2 fix: build the row's denominators into a flat array, call the
		// SIMD-batched gf64_inverse_batch_* once, scatter the inverses back
		// into the row. The original code called the scalar gf64_inverse()
		// per element (~30s for 1 GiB / 10K = 262M inverts); the batched
		// AVX-2 / AVX-512 path is instruction-level-parallelism across 4-8
		// lanes, not vectorized, but avoids the per-call function dispatch
		// overhead and lets the compiler hoist the `denoms` reads.
		// Avoid std::vector heap alloc by reusing a thread_local buffer.
		static thread_local std::vector<uint64_t> denoms_tls;
		if (denoms_tls.size() < numInputs) {
			denoms_tls.resize(numInputs);
		}
		uint64_t* denoms = denoms_tls.data();
		for (size_t r = 0; r < numRecovery; r++) {
			uint64_t y = firstRecovery + r;
			for (size_t c = 0; c < numInputs; c++) {
				uint64_t x = firstInput + c;
				uint64_t denom = x ^ y;
				denoms[c] = (denom == 0) ? 1 : denom;
			}
			gf64_inverse_batch(&coeffMatrix[r * numInputs], denoms, numInputs);
		}
		return;
	}

	// Parallel path: row-stealing via std::async workers.  nextRow is
	// destroyed only after all futures complete (we block below), so
	// workers can safely reference it by reference. Each worker has its
	// own denoms buffer (reused per row, not reallocated).
	std::atomic<size_t> nextRow{0};
	std::vector<std::future<void>> futures;
	futures.reserve(s_cauchyWorkerCount);

	for (size_t w = 0; w < s_cauchyWorkerCount; w++) {
		futures.push_back(std::async(std::launch::async, [=, &nextRow]() {
			std::vector<uint64_t> denoms(numInputs);
			for (size_t r = nextRow.fetch_add(1, std::memory_order_relaxed);
			     r < numRecovery;
			     r = nextRow.fetch_add(1, std::memory_order_relaxed)) {
				uint64_t y = firstRecovery + r;
				for (size_t c = 0; c < numInputs; c++) {
					uint64_t x = firstInput + c;
					uint64_t denom = x ^ y;
					denoms[c] = (denom == 0) ? 1 : denom;
				}
				gf64_inverse_batch(&coeffMatrix[r * numInputs], denoms.data(), numInputs);
			}
		}));
	}

	for (auto& f : futures) {
		f.wait();
	}
}

// v2-4: standalone matrix build returning a fresh buffer (caller frees).
gf64_t* GF64Controller::BuildCauchyMatrixAlloc(
	size_t numInputs, size_t numRecovery,
	uint64_t firstInput, uint64_t firstRecovery
) {
	if (numInputs == 0 || numRecovery == 0) return nullptr;
	gf64_t* matrix = (gf64_t*)malloc(numRecovery * numInputs * sizeof(gf64_t));
	if (!matrix) return nullptr;
	BuildCauchyMatrix(matrix, numInputs, numRecovery, firstInput, firstRecovery);
	return matrix;
}

// ============================================================================
// GetOrBuildCoeffMatrix  (LRU-cached)
// ----------------------------------------------------------------------------
// Returns a coefficient matrix from the LRU cache if one with the same
// (numInputs, numRecovery, firstInput, firstRecovery) exists, otherwise
// allocates and builds a new one.  The cache owns the memory — callers must
// NOT free the returned pointer.
// ============================================================================
static gf64_t* GetOrBuildCoeffMatrix(
	size_t numInputs, size_t numRecovery,
	uint64_t firstInput, uint64_t firstRecovery
) {
	CoeffCacheKey key = { numInputs, numRecovery, firstInput, firstRecovery };

	auto it = s_coeffCache.map.find(key);
	if (it != s_coeffCache.map.end()) {
		s_coeffCache.lru.remove(key);
		s_coeffCache.lru.push_front(key);
		return it->second;
	}

	gf64_t* matrix = (gf64_t*)malloc(numRecovery * numInputs * sizeof(gf64_t));
	if (!matrix) return nullptr;

	GF64Controller::BuildCauchyMatrix(matrix, numInputs, numRecovery, firstInput, firstRecovery);

	if (s_coeffCache.map.size() >= COEFF_CACHE_MAX) {
		auto evictKey = s_coeffCache.lru.back();
		s_coeffCache.lru.pop_back();
		auto evictIt = s_coeffCache.map.find(evictKey);
		if (evictIt != s_coeffCache.map.end()) {
			free(evictIt->second);
			s_coeffCache.map.erase(evictIt);
		}
	}

	s_coeffCache.map[key] = matrix;
	s_coeffCache.lru.push_front(key);
	return matrix;
}

// ============================================================================
// muladd_single_output  (shared 1D muladd-accumulate inner loop)
// ----------------------------------------------------------------------------
// Resets out to zero and XOR-accumulates in[j] * coeffRow[j] for j in
// [0, numIn). Three callers used to inline this exact pattern:
//   * GF64Controller::MultiplyAccumulate        (single-threaded recovery)
//   * GF64Controller::ComputeRepairBlocks       (per-chunk back-sub worker)
//   * WorkerThread small-R shortcut             (PR #40, total_num_out <= 32)
// Triplication was flagged by cubic as a P3 maintainability issue (no
// behavioural change): any future tweak to the coefficient/input addressing
// or memset/clear semantics would have to be kept in sync across three
// call sites or the paths would silently diverge. The helper centralises
// the pattern; dispatch is left to the outer function so we still call
// EnsureDispatch() exactly once per entry (not per output block).
// ============================================================================
static inline void muladd_single_output(
	gf64_t* out,
	const gf64_t* in,
	const gf64_t* coeffRow,
	size_t numIn,
	size_t B
) {
	memset(out, 0, B * sizeof(gf64_t));
	for (size_t j = 0; j < numIn; j++) {
		gf64_region_muladd_arr(out, in + j * B, &coeffRow[j], B, 1);
	}
}

// ============================================================================
// GF64Controller::MultiplyAccumulate  (single-threaded kernel)
// ----------------------------------------------------------------------------
// For each output block k:
//   out[k] = XOR_{j=0}^{numIn-1}  in[j] * coeff[k*numIn + j]
//
// Each gf64_region_mul call takes a single coefficient (n_coeff=1),
// producing tmp[i] = in[j][i] * coeff[k][j], which is then XOR-accumulated
// into out[k].  All calls happen in native C — zero JS→N-API crossings.
// ============================================================================
void GF64Controller::MultiplyAccumulate(
	gf64_t* out, size_t numOut,
	const gf64_t* in, size_t numIn,
	const gf64_t* coeffMatrix,
	size_t blockSize64
) {
	EnsureDispatch();

	for (size_t k = 0; k < numOut; k++) {
		const gf64_t* row = coeffMatrix + k * numIn;
		muladd_single_output(out + k * blockSize64,
		                     in, row, numIn, blockSize64);
	}
}

// ============================================================================
// Tunable group size for the Wave 2 engine refactor (T3).
// ----------------------------------------------------------------------------
// PAR3_GF64_GROUP overrides the number of recovery blocks grouped per worker
// for cache-friendly tiling. Range: 1..256; out-of-range or invalid values
// silently fall back to kDefaultGroupSize.
// ============================================================================
static constexpr size_t kDefaultGroupSize = 12;

static int ParseGroupSizeEnv() {
	const char* env = std::getenv("PAR3_GF64_GROUP");
	if (env == nullptr || *env == '\0') return 0;
	int v = std::atoi(env);
	if (v < 1 || v > 256) return 0;
	return v;
}

static int GetGroupSize() {
		static int v = ParseGroupSizeEnv();
		if (v > 0) return v;
		EnsureDispatch();
		// Default: 16 for AVX-2 (wider tiles improve cache utilisation without
		// ZMM register pressure), kDefaultGroupSize (12) for all other methods
		// (AVX-512 benefits diminish past 12 due to register file contention).
		return gf64_current_method == GF64_AVX2 ? 16 : static_cast<int>(kDefaultGroupSize);
	}

// ============================================================================
// Tunable K-group size for the Wave 3 fused-output engine refactor (PB7).
// ----------------------------------------------------------------------------
// PAR3_GF64_K_GROUP overrides the number of output blocks grouped per fused-
// output kernel call (one input block is applied to K outputs in each call).
// Range: 1..256; out-of-range or invalid values silently fall back to
// kDefaultKGroupSize. Default 12 mirrors PA7's kDefaultGroupSize so a single
// env var controls batch sizing for both coupled-input and fused-output paths.
// ============================================================================
static constexpr size_t kDefaultKGroupSize = 12;

static int ParseKGroupSizeEnv() {
	const char* env = std::getenv("PAR3_GF64_K_GROUP");
	if (env == nullptr || *env == '\0') return 0;
	int v = std::atoi(env);
	if (v < 1 || v > 256) return 0;
	return v;
}

static int GetKGroupSize() {
		static int v = ParseKGroupSizeEnv();
		if (v > 0) return v;
		EnsureDispatch();
		// Default: 16 for AVX-2 (wider tiles improve cache utilisation without
		// ZMM register pressure), kDefaultKGroupSize (12) for all other methods
		// (AVX-512 benefits diminish past 12 due to register file contention).
		return gf64_current_method == GF64_AVX2 ? 16 : static_cast<int>(kDefaultKGroupSize);
	}

// ============================================================================
// T0: binary flags for the v3 max-perf plan (env-gated; default off)
// ----------------------------------------------------------------------------
// PAR3_GF64_FAST_CREATE / PAR3_GF64_BENCH_NATIVE gate Phase A / B / C paths:
//   fast_create  = 1 → enable native fast-create path (legacy path when 0)
//   bench_native = 1 → enable native-only bench paths   (JS path when 0)
// Both default to 0 when unset, empty, or non-"1". The flags are pure
// on/off switches (no range, no clamping) so the parser collapses the
// nullptr / empty / non-1 cases into the same 0 return — matching the
// ParseAutotuneEnv() caching style. The PUBLIC accessors (no `static`)
// expose the cached value to future TUs (T1, A1, A2, B*, C*) that need
// to branch on these flags; the parser helpers stay file-local.
// ============================================================================
static int ParseFastCreateEnv() {
	static int cached = -1;
	if (cached < 0) {
		const char* env = std::getenv("PAR3_GF64_FAST_CREATE");
		cached = (env != nullptr && *env != '\0' && std::atoi(env) == 1) ? 1 : 0;
	}
	return cached;
}

static int ParseBenchNativeEnv() {
	static int cached = -1;
	if (cached < 0) {
		const char* env = std::getenv("PAR3_GF64_BENCH_NATIVE");
		cached = (env != nullptr && *env != '\0' && std::atoi(env) == 1) ? 1 : 0;
	}
	return cached;
}

int GetFastCreate() {
	return ParseFastCreateEnv();
}

int GetBenchNative() {
	return ParseBenchNativeEnv();
}

// ============================================================================
// PD3: BLOCK_SIZE autotune  (env-gated; default off)
// ----------------------------------------------------------------------------
// At compute-recovery time, scan {1, 4, 16, 64, 256} MiB candidate block
// sizes against a 1 MiB synthetic pilot (256 blocks at 4 KiB) and pick the
// size that maximises bytes/us through the existing
// `gf64_region_muladd_*_arr` dispatch. Env var:
//   PAR3_GF64_BLOCK_SIZE_AUTOTUNE=1  → enable
//   PAR3_GF64_BLOCK_SIZE_AUTOTUNE=0  (or unset) → disabled, return 0
//
// Layout-constraint note: the JS-side input/output buffers are sized at
// exactly `numInputs * (JS-passed blockSize)` bytes by `lib/par3gen.js`
// before the C++ entry is reached. The chosen block size therefore cannot
// be applied mid-flight — overriding `blockSize64` would corrupt the
// stride math in `WorkerRange` / `WorkerThread` (offsets `(k * B)`,
// `(j * B)`). The chosen size is reported for telemetry / future
// JS-aware refactors; the actual recovery computation continues to use
// the JS-passed blockSize64 unchanged. This is the safest behaviour given
// the MUST NOT `lib/par3gen.js` constraint.
//
// When the env var is unset, the function returns 0 immediately so the
// caller proceeds with the existing block size unchanged.
// ============================================================================
static int ParseAutotuneEnv() {
	static int cached = -1;
	if (cached < 0) {
		const char* env = std::getenv("PAR3_GF64_BLOCK_SIZE_AUTOTUNE");
		cached = (env != nullptr && *env != '\0' && std::atoi(env) == 1) ? 1 : 0;
	}
	return cached;
}

static size_t AutotuneBlockSize() {
	if (!ParseAutotuneEnv()) return 0;
	EnsureDispatch();

	// 1 MiB synthetic pilot (256 blocks at 4 KiB).
	constexpr size_t SAMPLE_BYTES = 1ULL * 1024 * 1024;
	constexpr size_t SAMPLE_WORDS = SAMPLE_BYTES / sizeof(gf64_t); // 131072 gf64_t
	constexpr int   N_ITER        = 64;
	constexpr int   N_WARMUP      = 3;

	gf64_t* sample_in  = (gf64_t*)std::malloc(SAMPLE_BYTES);
	gf64_t* sample_out = (gf64_t*)std::malloc(SAMPLE_BYTES);
	if (sample_in == nullptr || sample_out == nullptr) {
		std::free(sample_in);
		std::free(sample_out);
		return 0;
	}

	// Deterministic synthetic data (avoids all-zero / all-one edges).
	for (size_t i = 0; i < SAMPLE_WORDS; i++) {
		sample_in[i]  = (gf64_t)((uint64_t)i * 0x9E3779B97F4A7C15ULL ^ 0x123456789ABCDEFULL);
		sample_out[i] = (gf64_t)((uint64_t)i * 0xC6BC279692B5C323ULL ^ 0xFEDCBA9876543210ULL);
	}
	gf64_t coeff = (gf64_t)0x0123456789ABCDEFULL;

	// Candidate block sizes in gf64_t units (1 / 4 / 16 / 64 / 256 MiB).
	// Note: the kernel doesn't observe the block size — each measurement
	// runs the same `len = SAMPLE_WORDS` payload through
	// `gf64_region_muladd_arr`. The candidate name labels the working-set
	// dimension being benchmarked; the bytes/us proxy captures host-
	// specific cache / TLB behaviour at that scale.
	static const size_t CANDIDATES_GF64[5] = {
		(1ULL   * 1024 * 1024) / sizeof(gf64_t),  //  131072
		(4ULL   * 1024 * 1024) / sizeof(gf64_t),  //  524288
		(16ULL  * 1024 * 1024) / sizeof(gf64_t),  // 2097152
		(64ULL  * 1024 * 1024) / sizeof(gf64_t),  // 8388608
		(256ULL * 1024 * 1024) / sizeof(gf64_t)   // 33554432
	};

	size_t best_size_gf64 = 0;
	double best_bpus      = 0.0;

	for (int c = 0; c < 5; c++) {
		const size_t B = CANDIDATES_GF64[c];
		(void)B;  // naming only — kernel `len` is SAMPLE_WORDS for every measurement

		for (int w = 0; w < N_WARMUP; w++) {
			gf64_region_muladd_arr(sample_out, sample_in, &coeff, SAMPLE_WORDS, 1);
		}

		std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
		for (int i = 0; i < N_ITER; i++) {
			gf64_region_muladd_arr(sample_out, sample_in, &coeff, SAMPLE_WORDS, 1);
		}
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		const double us     = std::chrono::duration<double, std::micro>(t1 - t0).count();
		const double bpus   = ((double)SAMPLE_BYTES * (double)N_ITER) / us;

		if (bpus > best_bpus) {
			best_bpus      = bpus;
			best_size_gf64 = CANDIDATES_GF64[c];
		}
	}

	std::free(sample_in);
	std::free(sample_out);

	return best_size_gf64;
}

// ============================================================================
// Thread worker  —  drives MultiplyAccumulate on a contiguous range of
// recovery blocks or input blocks.
// ============================================================================
struct WorkerRange {
	gf64_t*       out_start = nullptr;       // first recovery block of this worker (or slice start)
	size_t        num_out = 0;               // how many recovery blocks this worker handles
	size_t        total_num_out = 0;         // total recovery blocks across all workers
	const gf64_t* in = nullptr;              // first input block of this worker (or slice start)
	size_t        num_in = 0;
	const gf64_t* coeff_row_start = nullptr; // coeffMatrix + outStart * coeff_stride
	size_t        coeff_stride = 0;          // row stride in coefficient matrix
	size_t        block_size64 = 0;          // slice length in words
	size_t        block_stride64 = 0;        // stride between blocks in words (defaults to block_size64 if 0)
	size_t        tile_size = 0;             // L3-aware input tile size (in blocks)
	bool          accumulate = false;        // accumulate into out_start rather than zeroing
};

static inline void xor_buffer(gf64_t* dst, const gf64_t* src, size_t count) {
	size_t i = 0;
	for (; i + 3 < count; i += 4) {
		dst[i + 0] ^= src[i + 0];
		dst[i + 1] ^= src[i + 1];
		dst[i + 2] ^= src[i + 2];
		dst[i + 3] ^= src[i + 3];
	}
	for (; i < count; i++) {
		dst[i] ^= src[i];
	}
}

// ============================================================================
// WorkerThread  (Wave 3: 2D-blocked batching — K outputs × G inputs per call)
// ----------------------------------------------------------------------------
// Combines PA7's coupled-input outer-product (G inputs at a time) with PB7's
// fused-output batching (K outputs at a time) into a single 2D kernel call:
// for each output tile (k_start..k_start+Kk) × input tile (j..j+Gk):
//
//   for k_local in [0..Kk):
//     for g_local in [0..Gk):
//       out_{k_start + k_local}[w] ^= in[j + g_local][w] * coeff[k_start + k_local][j + g_local]
//
// k_start ranges over [0..num_out) in steps of K; Kk = min(K, num_out - k_start).
// j ranges over [0..num_in) in steps of G; Gk = min(G, num_in - j) (or tile_size-bounded).
//
// K = GetKGroupSize()  (default 12; user-tunable via PAR3_GF64_K_GROUP, capped
// at 256 by ParseKGroupSizeEnv).
// G = GetGroupSize()    (default 12; user-tunable via PAR3_GF64_GROUP, capped
// at 256 by ParseGroupSizeEnv).
// ============================================================================
static void WorkerThread(const WorkerRange& range) {
	EnsureDispatch();
	const size_t num_in = range.num_in;
	const size_t num_out = range.num_out;
	const size_t total_num_out = range.total_num_out;
	const size_t slice_len = range.block_size64;
	const size_t stride_len = range.block_stride64 ? range.block_stride64 : slice_len;
	const size_t stride = range.coeff_stride;

	if (total_num_out <= 32) {
		// Small-R single-pass input stream: outer loop over inputs j,
		// inner loop over outputs k.
		// For small-R (total_num_out <= 32), all num_out output blocks stay
		// warm in L2/L3 cache while each input block is streamed from DRAM
		// exactly once. This eliminates the multi-pass input read traffic.
		if (!range.accumulate) {
			for (size_t k = 0; k < num_out; k++) {
				memset(range.out_start + k * stride_len, 0, slice_len * sizeof(gf64_t));
			}
		}
		if (num_out == 1) {
			const gf64_t* row = range.coeff_row_start;
			for (size_t j = 0; j < num_in; j++) {
				gf64_t c = row[j];
				if (c != 0) {
					gf64_region_muladd_arr(range.out_start, range.in + j * stride_len, &c, slice_len, 1);
				}
			}
			return;
		}

		gf64_t* outs_ptrs[32];
		for (size_t k = 0; k < num_out; k++) {
			outs_ptrs[k] = range.out_start + k * stride_len;
		}
		const gf64_t* coeff_ptrs[32];
		for (size_t j = 0; j < num_in; j++) {
			const gf64_t* in_block = range.in + j * stride_len;
			for (size_t k = 0; k < num_out; k++) {
				coeff_ptrs[k] = &range.coeff_row_start[k * stride + j];
			}
			gf64_region_fused_output_muladd_arr(outs_ptrs, in_block, coeff_ptrs, slice_len, num_out);
		}
		return;
	}

	const int K = GetKGroupSize();
	const int G = GetGroupSize();
	const size_t MAX_STACK_K = 256;  // matches kDefaultKGroupSize cap
	const size_t MAX_STACK_G = 256;  // matches kDefaultGroupSize cap

	// Storage for the K × G inner-loop pointer arrays. Lives in
	// WorkerThread's stack frame so its addresses remain valid across each
	// gf64_region_2d_muladd_arr call below.
	gf64_t* outs_stack[MAX_STACK_K];
	const gf64_t* in_blocks_stack[MAX_STACK_G];
	std::vector<gf64_t*> outs_heap;
	std::vector<const gf64_t*> in_blocks_heap;

	auto process_out = [&](size_t k_start) {
		const size_t Kk = std::min((size_t)K, num_out - k_start);
		gf64_t** outs_ptr = outs_stack;
		if (Kk > MAX_STACK_K) {
			outs_heap.resize(Kk);
			outs_ptr = outs_heap.data();
		}
		for (size_t k_local = 0; k_local < Kk; k_local++) {
			outs_ptr[k_local] = range.out_start + (k_start + k_local) * stride_len;
			if (!range.accumulate) {
				memset(outs_ptr[k_local], 0, slice_len * sizeof(gf64_t));
			}
		}

		// Coeff row for k_start..k_start+Kk-1, starting at column j.
		const gf64_t* coeff_base = range.coeff_row_start + k_start * stride;

		// L3-aware input tile: tile_size caps the j range to keep the
		// (K outputs + G inputs) working set L3-resident.
		if (range.tile_size == 0 || range.tile_size >= num_in) {
			for (size_t j = 0; j < num_in; j += (size_t)G) {
				size_t Gk = std::min((size_t)G, num_in - j);
				const gf64_t** in_blocks_ptr = in_blocks_stack;
				if (Gk > MAX_STACK_G) {
					in_blocks_heap.resize(Gk);
					in_blocks_ptr = in_blocks_heap.data();
				}
				for (size_t g_local = 0; g_local < Gk; g_local++) {
					in_blocks_ptr[g_local] = range.in + (j + g_local) * stride_len;
				}
				gf64_region_2d_muladd_arr(
					(gf64_t *HEDLEY_RESTRICT *HEDLEY_RESTRICT)outs_ptr,
					Kk,
					(const gf64_t *HEDLEY_RESTRICT *HEDLEY_RESTRICT)in_blocks_ptr,
					Gk,
					coeff_base + j,
					stride,
					slice_len);
			}
		} else {
			for (size_t j_tile = 0; j_tile < num_in; j_tile += range.tile_size) {
				size_t j_end = std::min(j_tile + range.tile_size, num_in);
				for (size_t j = j_tile; j < j_end; j += (size_t)G) {
					size_t Gk = std::min((size_t)G, j_end - j);
					const gf64_t** in_blocks_ptr = in_blocks_stack;
					if (Gk > MAX_STACK_G) {
						in_blocks_heap.resize(Gk);
						in_blocks_ptr = in_blocks_heap.data();
					}
					for (size_t g_local = 0; g_local < Gk; g_local++) {
						in_blocks_ptr[g_local] = range.in + (j + g_local) * stride_len;
					}
					gf64_region_2d_muladd_arr(
						(gf64_t *HEDLEY_RESTRICT *HEDLEY_RESTRICT)outs_ptr,
						Kk,
						(const gf64_t *HEDLEY_RESTRICT *HEDLEY_RESTRICT)in_blocks_ptr,
						Gk,
						coeff_base + j,
						stride,
						slice_len);
				}
			}
		}
	};

	for (size_t k_start = 0; k_start < num_out; k_start += (size_t)K) {
		process_out(k_start);
	}
}

// ============================================================================
// GF64Controller::ComputeRepairBlocks
// ----------------------------------------------------------------------------
// Reconstructs missing input blocks from available blocks + solve coefficients.
// This is the "back-substitution" step: given an already-factored system,
// multiply the available blocks by the solve matrix.
//
// For each missing block k:
//   repaired[k] = XOR_{j=0}^{nAvail-1}  avail[j] * solveMatrix[k*nAvail + j]
//
// Uses gf64_region_muladd_arr (same SIMD dispatch as create path).
// ============================================================================

// ============================================================================
// gf64_mul_combi — GF(2^64) multiply for solve_region
// ----------------------------------------------------------------------------
// Uses the PCLMULQDQ-based reduction from gf64_solve.c (identical logic).
// ============================================================================
static inline gf64_t gf64_mul_combi(gf64_t a, gf64_t b) {
	__m128i a128 = _mm_set_epi64x(0, a);
	__m128i b128 = _mm_set_epi64x(0, b);
	__m128i p = _mm_clmulepi64_si128(a128, b128, 0x00);
	uint64_t lo = _mm_cvtsi128_si64(p);
	uint64_t hi = _mm_cvtsi128_si64(_mm_srli_si128(p, 8));

	/* Lower 64 bits of hi * 0x1B (truncated at 64 bits by uint64_t). */
	uint64_t t_lo = (hi << 4) ^ (hi << 3) ^ (hi << 1) ^ hi;

	/* Overflow bits (64-67) of hi * 0x1B:
	 * (hi<<4) overflow: hi[60:63] → full_product[64:67]
	 * (hi<<3) overflow: hi[61:63] → full_product[64:66]
	 * (hi<<1) overflow: hi[63]   → full_product[64]
	 * R_hi[0] = full_product bit 64 = hi[60] ^ hi[61] ^ hi[63]
	 * R_hi[1] = full_product bit 65 = hi[61] ^ hi[62]
	 * R_hi[2] = full_product bit 66 = hi[62] ^ hi[63]
	 * R_hi[3] = full_product bit 67 = hi[63]
	 */
	uint64_t R_hi =
		(((hi >> 60) ^ (hi >> 61) ^ (hi >> 63)) & 1) |
		((((hi >> 61) ^ (hi >> 62)) & 1) << 1) |
		((((hi >> 62) ^ (hi >> 63)) & 1) << 2) |
		(((hi >> 63) & 1) << 3);

	/* Reduce R_hi: x^64 ≡ 0x1B, so R_hi * x^64 ≡ R_hi * 0x1B.
	 * R_hi < 16, so R_hi * 0x1B fits safely in uint64_t. */
	uint64_t t2 = (R_hi << 4) ^ (R_hi << 3) ^ (R_hi << 1) ^ R_hi;

	return lo ^ t_lo ^ t2;
}

// ============================================================================
// GF64Controller::SolveAndReconstruct
// ----------------------------------------------------------------------------
// Full solve-and-reconstruct pipeline for PAR3 block repair.
//
// 1. Gaussian-eliminate the n×n Cauchy sub-matrix (operates on a copy)
// 2. Apply the same elimination ops to the n×blockSizeWords RHS,
//    where RHS row i is the full block data from recovery equation i.
//    This step uses gf64_region_muladd_arr for the word-parallel
//    row operations — the same SIMD kernels as the create path.
// 3. Output: n reconstructed blocks, each blockSize64 words.
//
// The RHS is laid out as n blocks of blockSize64 words each:
//   rhs[i * blockSize64 + w] = w-th word of recovery block i
//
// Returns 0 on success, -1 on singular matrix.
// ============================================================================
int GF64Controller::SolveAndReconstruct(
	gf64_t* A,
	gf64_t* rhsBlocks,
	size_t n,
	size_t blockSize64,
	int numThreads
) {
	if (n == 0) return 0;
	EnsureDispatch();

	// Gaussian elimination on A (in-place)
	for (size_t col = 0; col < n; col++) {
		// Find pivot
		size_t pivot = col;
		while (pivot < n && A[pivot * n + col] == 0) pivot++;
		if (pivot == n) return -1;  // singular

		// Swap rows in A and RHS
		if (pivot != col) {
			for (size_t j = 0; j < n; j++) {
				gf64_t tmp = A[col * n + j];
				A[col * n + j] = A[pivot * n + j];
				A[pivot * n + j] = tmp;
			}
			gf64_t* r_col = rhsBlocks + col * blockSize64;
			gf64_t* r_piv = rhsBlocks + pivot * blockSize64;
			for (size_t w = 0; w < blockSize64; w++) {
				gf64_t tmp = r_col[w];
				r_col[w] = r_piv[w];
				r_piv[w] = tmp;
			}
		}

		// Scale pivot row
		gf64_t pv = A[col * n + col];
		gf64_t pv_inv = gf64_inverse(pv);
		if (pv != 1) {
			for (size_t j = 0; j < n; j++) {
				A[col * n + j] = gf64_mul_combi(A[col * n + j], pv_inv);
			}
			// Scale RHS pivot row: SET out = in * pv_inv (not XOR-accumulate)
			gf64_t* tmp = (gf64_t*)malloc(blockSize64 * sizeof(gf64_t));
			if (!tmp) return -1;
			memcpy(tmp, rhsBlocks + col * blockSize64, blockSize64 * sizeof(gf64_t));
			gf64_region_mul_arr(rhsBlocks + col * blockSize64, tmp, &pv_inv, blockSize64, 1);
			free(tmp);
		}

		// Eliminate column from other rows
		for (size_t row = 0; row < n; row++) {
			if (row == col) continue;
			gf64_t factor = A[row * n + col];
			if (factor == 0) continue;

			// A[row] ^= A[col] * factor
			for (size_t j = 0; j < n; j++) {
				A[row * n + j] ^= gf64_mul_combi(factor, A[col * n + j]);
			}

			// RHS[row] ^= RHS[col] * factor   — this is the hot loop
			gf64_region_muladd_arr(
				rhsBlocks + row * blockSize64,
				rhsBlocks + col * blockSize64,
				&factor, blockSize64, 1);
		}
	}

	return 0;
}

// ============================================================================
// GF64Controller::ComputeRecoveryBlocks
// ----------------------------------------------------------------------------
// High-level entry point:
//   1. Build Cauchy coefficient matrix (numRecovery × numInput)
//   2. Distribute recovery blocks across numThreads worker threads
//   3. Each thread independently calls the multiply-accumulate kernel
//
// Embarrassingly parallel — recovery blocks are independent because each
// output region is written by exactly one thread (no atomics needed).
// ============================================================================
void GF64Controller::ComputeRecoveryBlocks(
	const gf64_t* inputs, size_t numInputs,
	gf64_t*       recovery, size_t numRecovery,
	size_t        blockSize64,
	uint64_t      firstInput, uint64_t firstRecovery,
	int           numThreads,
	bool          accumulate
) {
	if (numInputs == 0 || numRecovery == 0) return;

	// --- 0. PD3 BLOCK_SIZE autotune (env-gated) ---
	// Runs once per process; cheap when disabled. The chosen size is
	// reported for telemetry — see AutotuneBlockSize comment for the
	// layout-constraint reason the JS-passed blockSize64 is preserved.
	{
		static const size_t s_autotune_once = AutotuneBlockSize();
		if (s_autotune_once != 0 && s_autotune_once != blockSize64) {
			std::fprintf(stderr,
				"[par3] BLOCK_SIZE autotune: chose %zu gf64_t (~%.2f MiB); JS-passed %zu gf64_t (~%.2f MiB) — using JS-passed (layout-locked)\n",
				s_autotune_once, (double)s_autotune_once * sizeof(gf64_t) / (1024.0 * 1024.0),
				blockSize64,    (double)blockSize64    * sizeof(gf64_t) / (1024.0 * 1024.0));
		}
	}

	// Issue #59 T7-verify: PAR3_FENGER_THREADS env override. Mirrors the
	// same override in ComputeRecoveryBlocksFenger — for the legacy
	// matvec path. When set to a positive integer N, forces numThreads=N
	// (subject to no upper cap; legacy path doesn't shard by B-size).
	// Set to 0 or unset = keep the existing auto-detect path.
	{
		const char *fenger_env = std::getenv("PAR3_FENGER_THREADS");
		if (fenger_env != nullptr && *fenger_env != '\0') {
			int forced = std::atoi(fenger_env);
			if (forced > 0) {
				numThreads = forced;
			}
		}
	}

	if (numThreads <= 0) {
		numThreads = (int)GetEffectiveCpuCount();
		if (numThreads <= 0) numThreads = 1;
	}

	// --- 1. Build coefficient matrix (via LRU cache) ---
	gf64_t* coeff = GetOrBuildCoeffMatrix(numInputs, numRecovery, firstInput, firstRecovery);
	if (!coeff) return;

	ComputeRecoveryBlocksWithCoeff(inputs, numInputs, recovery, numRecovery,
	                                blockSize64, coeff, numThreads, accumulate);
}

// ============================================================================
// GF64Controller::ComputeRecoveryBlocksWithCoeff
// ----------------------------------------------------------------------------
// v2-4: pre-computed coefficient matrix variant. Skips the matrix build
// entirely (assumes the matrix was already built and validated), so the
// caller can overlap the matrix build with other work (e.g. file read).
// The caller is responsible for freeing the coeff buffer after the call.
// ============================================================================
static std::atomic<int> s_last_decomposition_path{0};

int GF64Controller::GetLastDecompositionPath() {
	return s_last_decomposition_path.load();
}

void GF64Controller::ResetLastDecompositionPath() {
	s_last_decomposition_path.store(0);
}

static size_t GetInputDecompScratchCap() {
	const char* val_str = nullptr;
#ifdef _WIN32
	char env_buf[64];
	DWORD len = GetEnvironmentVariableA("PAR3_INPUT_DECOMP_SCRATCH_BYTES", env_buf, sizeof(env_buf));
	if (len > 0 && len < sizeof(env_buf)) {
		val_str = env_buf;
	}
#else
	const char* env = std::getenv("PAR3_INPUT_DECOMP_SCRATCH_BYTES");
	if (env != nullptr && *env != '\0') {
		val_str = env;
	}
#endif
	if (val_str != nullptr) {
		while (*val_str == ' ' || *val_str == '\t') val_str++;
		// Cubic review P2: reject leading signs ('-', '+') and non-digits so negative or signed values are not accepted
		if (*val_str >= '0' && *val_str <= '9') {
			char* endptr = nullptr;
			errno = 0;
			unsigned long long val = std::strtoull(val_str, &endptr, 10);
			if (errno != ERANGE && endptr != val_str) {
				while (*endptr == ' ' || *endptr == '\t') endptr++;
				// Must parse the complete string with no trailing garbage, fit in size_t, and be positive
				if (*endptr == '\0' && val > 0 && val <= (unsigned long long)SIZE_MAX) {
					return (size_t)val;
				}
			}
		}
	}
	// Default: 128 MiB on 64-bit systems, 32 MiB on 32-bit systems
	return (sizeof(void*) >= 8) ? (128 * 1024 * 1024) : (32 * 1024 * 1024);
}

void GF64Controller::ComputeRecoveryBlocksWithCoeff(
	const gf64_t* inputs, size_t numInputs,
	gf64_t*       recovery, size_t numRecovery,
	size_t        blockSize64,
	const gf64_t* coeff,
	int           numThreads,
	bool          accumulate
) {
	if (numInputs == 0 || numRecovery == 0 || !coeff) {
		ResetLastDecompositionPath();
		return;
	}

	// --- 2. Per-workload dispatch (PD2 AVX-512 downclock heuristic) ---
	gf64_apply_method(gf64_method_for_workload(numInputs, numRecovery, blockSize64));

	if (numThreads <= 0) {
		numThreads = (int)GetEffectiveCpuCount();
		if (numThreads <= 0) numThreads = 1;
	}

	// Compute L3-aware tile size for input blocks.
	size_t l3Size = GetL3CacheSize();
	size_t bytesPerBlock = blockSize64 * sizeof(gf64_t);
	size_t tileSize = 256;
	if (bytesPerBlock > 0 && l3Size > 0) {
		tileSize = l3Size / bytesPerBlock;
		tileSize = std::min(tileSize, (size_t)256);
		if (tileSize == 0) tileSize = 1;
	}

	// Case 1: Single thread
	if (numThreads == 1) {
		s_last_decomposition_path.store(1);
		WorkerRange r;
		r.out_start = recovery;
		r.num_out = numRecovery;
		r.total_num_out = numRecovery;
		r.in = inputs;
		r.num_in = numInputs;
		r.coeff_row_start = coeff;
		r.coeff_stride = numInputs;
		r.block_size64 = blockSize64;
		r.block_stride64 = blockSize64;
		r.tile_size = tileSize;
		r.accumulate = accumulate;
		WorkerThread(r);
		return;
	}

	// Case 2: Input-domain decomposition: partition numInputs across workers.
	// Thread 0 computes directly into `recovery`.
	// Threads 1..n_workers-1 compute into small temporary thread-local buffers and are XOR-reduced into `recovery`.
	// Adaptive worker sizing (Issue #115): instead of an all-or-nothing check for numThreads,
	// adaptively size n_workers based on how many thread-local scratch buffers
	// fit within the scratch budget (GetInputDecompScratchCap()).
	// Each worker after thread 0 needs total_out_bytes = numRecovery * blockSize64 * sizeof(gf64_t).
	//
	// Cubic review P1: prevent integer overflow in numRecovery * blockSize64 and total_out_words * sizeof(gf64_t)
	size_t scratch_cap = GetInputDecompScratchCap();
	bool overflow = (numRecovery > 0 && blockSize64 > SIZE_MAX / numRecovery);
	size_t total_out_words = overflow ? 0 : (numRecovery * blockSize64);
	if (!overflow && total_out_words > SIZE_MAX / sizeof(gf64_t)) {
		overflow = true;
	}
	size_t total_out_bytes = overflow ? 0 : (total_out_words * sizeof(gf64_t));

	size_t max_extra_workers = (!overflow && total_out_bytes > 0) ? (scratch_cap / total_out_bytes) : 0;
	size_t max_scratch_workers = 1 + max_extra_workers;
	size_t n_input_workers = std::min((size_t)numThreads, max_scratch_workers);
	if (n_input_workers > numInputs) n_input_workers = numInputs;

	// Issue #117: Only enable input-domain decomposition when the input reads are NOT
	// already being amortized by chunk streaming (accumulate == true).
	// The chunked path is already input-streaming, so adding input-domain partitioning
	// with thread-local scratch buffers and XOR reduction on top is a net performance loss.
	if (!accumulate && !overflow && n_input_workers > numRecovery && n_input_workers >= 2) {
		std::vector<gf64_t*> temp_bufs(n_input_workers, nullptr);
		bool alloc_ok = true;
		for (size_t t = 1; t < n_input_workers; t++) {
			void* ptr = nullptr;
			ALIGN_ALLOC(ptr, total_out_bytes, 64);
			if (!ptr) {
				alloc_ok = false;
				break;
			}
			memset(ptr, 0, total_out_bytes);
			temp_bufs[t] = (gf64_t*)ptr;
		}

		if (alloc_ok) {
			s_last_decomposition_path.store(2);
			WorkerRange r0;
			bool r0_set = false;
			std::vector<std::thread> workers;
			workers.reserve(n_input_workers > 1 ? (n_input_workers - 1) : 0);

			for (size_t t = 0; t < n_input_workers; t++) {
				size_t in_start = (t * numInputs) / n_input_workers;
				size_t in_end = ((t + 1) * numInputs) / n_input_workers;
				size_t count_in = in_end - in_start;
				if (count_in == 0) continue;

				WorkerRange r;
				r.out_start = (t == 0) ? recovery : temp_bufs[t];
				r.num_out = numRecovery;
				r.total_num_out = numRecovery;
				r.in = inputs + in_start * blockSize64;
				r.num_in = count_in;
				r.coeff_row_start = coeff + in_start;
				r.coeff_stride = numInputs;
				r.block_size64 = blockSize64;
				r.block_stride64 = blockSize64;
				r.tile_size = tileSize;
				r.accumulate = (t == 0) ? accumulate : false;

				if (t == 0) {
					r0 = r;
					r0_set = true;
				} else {
					workers.emplace_back(WorkerThread, r);
				}
			}

			if (r0_set) {
				WorkerThread(r0);
			}

			for (auto& w : workers) {
				w.join();
			}

			// Reduction: XOR all thread buffers into recovery
			for (size_t t = 1; t < n_input_workers; t++) {
				if (temp_bufs[t]) {
					size_t in_start = (t * numInputs) / n_input_workers;
					size_t in_end = ((t + 1) * numInputs) / n_input_workers;
					if (in_end > in_start) {
						xor_buffer(recovery, temp_bufs[t], total_out_words);
					}
					ALIGN_FREE(temp_bufs[t]);
				}
			}
			return;
		}

		// Allocation fallback: free any allocated buffers and fall back to output-domain decomposition
		for (size_t t = 1; t < n_input_workers; t++) {
			if (temp_bufs[t]) ALIGN_FREE(temp_bufs[t]);
		}
	}

	// Case 3: Output-domain & 2D Block-Domain Decomposition (Issue #119)
	// Partitions output recovery blocks (SR) and block byte ranges (SB) across workers.
	// Zero scratch allocations, zero XOR reductions, and calling-thread reuse for worker 0.
	s_last_decomposition_path.store(3);

	size_t SR = std::min((size_t)numThreads, numRecovery);
	if (SR == 0) SR = 1;

	size_t SB = 1;
	size_t block_bytes = blockSize64 * sizeof(gf64_t);
	size_t min_slice_bytes = 64 * 1024; // 64 KiB min slice
	if ((size_t)numThreads > SR && block_bytes >= min_slice_bytes && (blockSize64 % 8 == 0)) {
		size_t max_sb_threads = (size_t)numThreads / SR;
		size_t max_sb_blocks = block_bytes / min_slice_bytes;
		SB = std::min(max_sb_threads, max_sb_blocks);
		if (SB == 0) SB = 1;
	}

	size_t n_workers = SR * SB;
	std::vector<WorkerRange> ranges;
	ranges.reserve(n_workers);

	for (size_t wr = 0; wr < SR; wr++) {
		size_t r_start = (wr * numRecovery) / SR;
		size_t r_end = ((wr + 1) * numRecovery) / SR;
		size_t r_count = r_end - r_start;
		if (r_count == 0) continue;

		for (size_t wb = 0; wb < SB; wb++) {
			size_t o_start = ((wb * blockSize64) / SB) & ~7ULL; // 64-byte (8 words) aligned
			size_t o_end = (wb + 1 == SB) ? blockSize64 : ((((wb + 1) * blockSize64) / SB) & ~7ULL);
			size_t o_words = (o_end > o_start) ? (o_end - o_start) : 0;
			if (o_words == 0) continue;

			WorkerRange r;
			r.out_start       = recovery + r_start * blockSize64 + o_start;
			r.num_out         = r_count;
			r.total_num_out   = numRecovery;
			r.in              = inputs + o_start;
			r.num_in          = numInputs;
			r.coeff_row_start = coeff + r_start * numInputs;
			r.coeff_stride    = numInputs;
			r.block_size64    = o_words;
			r.block_stride64  = blockSize64;
			r.tile_size       = tileSize;
			r.accumulate      = accumulate;

			ranges.push_back(r);
		}
	}

	size_t active = ranges.size();
	if (active == 1) {
		WorkerThread(ranges[0]);
	} else if (active > 1) {
		std::vector<std::thread> workers;
		workers.reserve(active - 1);
		for (size_t i = 1; i < active; i++) {
			workers.emplace_back(WorkerThread, ranges[i]);
		}
		WorkerThread(ranges[0]);
		for (auto& w : workers) {
			w.join();
		}
	}
}

// ============================================================================
// GF64Controller::ComputeRecoveryBlocksFull
// ----------------------------------------------------------------------------
// Single-call entry point for the full recovery range.
// ============================================================================
void GF64Controller::ComputeRecoveryBlocksFull(
	const gf64_t* inputs, size_t numInputs,
	gf64_t*       recovery, size_t numRecovery,
	size_t        blockSize64,
	uint64_t      firstInput, uint64_t firstRecovery,
	int           numThreads,
	bool          accumulate
) {
	ComputeRecoveryBlocks(inputs, numInputs, recovery, numRecovery,
	                      blockSize64, firstInput, firstRecovery, numThreads, accumulate);
}

// ============================================================================
// GF64Controller::ComputeRecoveryBlocksFromFile  (A1: mmap-based zero-copy)
// ----------------------------------------------------------------------------
// Zero-copy variant of the create entry: maps `sourcePath` into memory with
// mmap(2) and passes the mapped region directly to the kernel — no fs.read
// round-trip into a JS Buffer. This is the A1 entry; wiring it into NAPI
// is A2's job, and exposing it through lib/par3gen.js is A3's job.
//
// Env gate: PAR3_GF64_USE_MMAP must be set to "1" to enable. Default off
// for backward compatibility with existing bench harnesses — the legacy
// ComputeRecoveryBlocks / ComputeRecoveryBlocksFull path remains the active
// create entry unless the operator opts in.
//
// mmap flags: MAP_PRIVATE | MAP_POPULATE
//   - MAP_PRIVATE  : copy-on-write; kernel can drop pages without flushing
//                    to disk because we never mutate. This is the standard
//                    choice for read-only input regions.
//   - MAP_POPULATE : prefault all pages up-front. Without this, the kernel
//                    services first-touch page faults lazily during the
//                    kernel's input scan, adding 100s of µs of latency
//                    per page on cold caches. POPULATE makes the first
//                    kernel pass touch faulted-and-warm pages instead.
//   - MAP_HUGEPAGE / MAP_HUGEPAGE_FLAG2MB are intentionally NOT used here
//     — that's E2 (Phase E huge-pages work) which depends on /proc/meminfo
//     and hugeadm and is out of scope for A1.
//
// fd lifecycle: opened O_RDONLY with mmap; closed after munmap on success.
// On any error path (open / fstat / mmap / size mismatch / kernel),
// munmap+close are called before returning -1.
//
// File size contract: `sourcePath` must contain at least
// `numInputs * blockSize64 * sizeof(gf64_t)` bytes (the kernel reads
// exactly that many gf64_t elements as input). The file size MUST be a
// multiple of `blockSize64 * sizeof(gf64_t)` — otherwise the trailing
// partial block would be silently dropped, which is detectable only by
// comparing against the JS path's expected block count, so we reject it
// up front with a clear error.
//
// Caller contract: `recovery` is caller-allocated (matches the existing
// ComputeRecoveryBlocksFull signature). numInputs is derived from the
// file size, not passed in — the caller knows the file but not the
// internal split. numRecovery, blockSize64, firstInput, firstRecovery,
// and numThreads are passed through to ComputeRecoveryBlocksFull.
// ============================================================================
int GF64Controller::ComputeRecoveryBlocksFromFile(
	const char* sourcePath,
	gf64_t*       recovery, size_t numRecovery,
	size_t        blockSize64,
	uint64_t      firstInput, uint64_t firstRecovery,
	int           numThreads
) {
#if !PARPAR_USE_MMAP
	// Windows (or any non-POSIX build where mmap is unavailable):
	// return -1 so the JS layer falls back to ComputeRecoveryBlocksFull
	// over a JS-side file read.
	(void)sourcePath; (void)recovery; (void)numRecovery;
	(void)blockSize64; (void)firstInput; (void)firstRecovery; (void)numThreads;
	std::fprintf(stderr,
		"[par3] ComputeRecoveryBlocksFromFile: mmap-based path is not available "
		"on this platform; use ComputeRecoveryBlocksFull with a pre-allocated buffer.\n");
	return -1;
#else
	// --- Env gate. Default off; A2 / A3 will read this too. ---
	const char* env = std::getenv("PAR3_GF64_USE_MMAP");
	if (env == nullptr || *env == '\0' || std::atoi(env) != 1) {
		std::fprintf(stderr,
			"[par3] PAR3_GF64_USE_MMAP=0; use ComputeRecoveryBlocks with pre-allocated inputs\n");
		return -1;
	}
	if (sourcePath == nullptr || recovery == nullptr) {
		std::fprintf(stderr,
			"[par3] ComputeRecoveryBlocksFromFile: null sourcePath or recovery\n");
		return -1;
	}
	if (blockSize64 == 0) {
		std::fprintf(stderr,
			"[par3] ComputeRecoveryBlocksFromFile: blockSize64 == 0\n");
		return -1;
	}

	// --- 1. open(2) ---
	int fd = ::open(sourcePath, O_RDONLY);
	if (fd < 0) {
		std::fprintf(stderr,
			"[par3] ComputeRecoveryBlocksFromFile: open(%s) failed: %s\n",
			sourcePath, std::strerror(errno));
		return -1;
	}

	// --- 2. fstat(2): get file size ---
	struct stat st;
	if (::fstat(fd, &st) != 0) {
		std::fprintf(stderr,
			"[par3] ComputeRecoveryBlocksFromFile: fstat(%s) failed: %s\n",
			sourcePath, std::strerror(errno));
		::close(fd);
		return -1;
	}

	const size_t fileSize = static_cast<size_t>(st.st_size);
	const size_t blockBytes = blockSize64 * sizeof(gf64_t);

	// Reject files smaller than one block (would underflow in numInputs calc).
	if (fileSize < blockBytes || (fileSize % blockBytes) != 0) {
		std::fprintf(stderr,
			"[par3] ComputeRecoveryBlocksFromFile: file size %zu is not a positive "
			"multiple of blockSize64 * sizeof(gf64_t) = %zu\n",
			fileSize, blockBytes);
		::close(fd);
		return -1;
	}

	const size_t numInputs = fileSize / blockBytes;

	// --- 3. mmap(2): MAP_PRIVATE | MAP_POPULATE for prefault + copy-on-write ---
	void* mapped = ::mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
	if (mapped == MAP_FAILED) {
		std::fprintf(stderr,
			"[par3] ComputeRecoveryBlocksFromFile: mmap(%s, %zu) failed: %s\n",
			sourcePath, fileSize, std::strerror(errno));
		::close(fd);
		return -1;
	}

	const gf64_t* inputs = static_cast<const gf64_t*>(mapped);

	// --- 4. Kernel call (delegates to ComputeRecoveryBlocksFull → ComputeRecoveryBlocks) ---
	ComputeRecoveryBlocksFull(
		inputs, numInputs,
		recovery, numRecovery,
		blockSize64, firstInput, firstRecovery,
		numThreads);

	// --- 5. Cleanup: munmap + close even on success. ---
	::munmap(mapped, fileSize);
	::close(fd);
	return 0;
#endif
}
