#include "par3_topology.h"

#include <cstdlib>
#include <cstring>
#include <thread>
#include <mutex>
#include <vector>
#include <string>
#include <set>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#include <dirent.h>
#include <cstdio>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

static std::mutex s_topoMutex;
static CpuTopology s_cachedTopology = {0, 0, 0, 0, 0};
static bool s_isCached = false;

static size_t ParseSizeWithSuffix(const char* str) {
	if (!str) return 0;
	char* endptr = nullptr;
	double val = std::strtod(str, &endptr);
	size_t multiplier = 1;
	while (endptr && (*endptr == ' ' || *endptr == '\t')) endptr++;
	if (endptr) {
		switch (*endptr) {
			case 'K': case 'k': multiplier = 1024; break;
			case 'M': case 'm': multiplier = 1024 * 1024; break;
			case 'G': case 'g': multiplier = 1024 * 1024 * 1024; break;
		}
	}
	return static_cast<size_t>(val * multiplier);
}

static CpuTopology DetectRawTopology() {
	CpuTopology topo = {0, 0, 0, 0, 0};

#if defined(_WIN32)
	DWORD len = 0;
	GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
	if (len > 0) {
		std::vector<uint8_t> buffer(len);
		if (GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer.data(), &len)) {
			DWORD offset = 0;
			while (offset < len) {
				PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
					(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buffer.data() + offset);
				if (info->Relationship == RelationProcessorCore) {
					topo.physicalCores++;
					for (WORD g = 0; g < info->Processor.GroupCount; g++) {
						KAFFINITY mask = info->Processor.GroupMask[g].Mask;
						for (KAFFINITY m = mask; m > 0; m &= (m - 1)) {
							topo.logicalCores++;
						}
					}
				} else if (info->Relationship == RelationCache) {
					if (info->Cache.Level == 3) {
						topo.numClusters++;
						topo.totalL3 += info->Cache.CacheSize;
						if (info->Cache.CacheSize > topo.l3PerCluster) {
							topo.l3PerCluster = info->Cache.CacheSize;
						}
					}
				}
				offset += info->Size;
			}
		}
	}
#elif defined(__linux__)
	// 1. Enumerate CPUs under /sys/devices/system/cpu/
	std::set<std::pair<int, int>> physicalCorePairs;
	std::set<std::string> l3Clusters;
	DIR* dir = opendir("/sys/devices/system/cpu");
	if (dir) {
		struct dirent* entry;
		while ((entry = readdir(dir)) != nullptr) {
			if (entry->d_type == DT_DIR || entry->d_type == DT_LNK || entry->d_type == DT_UNKNOWN) {
				if (std::strncmp(entry->d_name, "cpu", 3) == 0 && entry->d_name[3] >= '0' && entry->d_name[3] <= '9') {
					topo.logicalCores++;
					std::string cpuPath = std::string("/sys/devices/system/cpu/") + entry->d_name;

					// Read physical package ID and core ID
					int pkgId = 0, coreId = -1;
					std::string pkgPath = cpuPath + "/topology/physical_package_id";
					std::string corePath = cpuPath + "/topology/core_id";
					FILE* fpPkg = std::fopen(pkgPath.c_str(), "r");
					if (fpPkg) {
						if (std::fscanf(fpPkg, "%d", &pkgId) != 1) pkgId = 0;
						std::fclose(fpPkg);
					}
					FILE* fpCore = std::fopen(corePath.c_str(), "r");
					if (fpCore) {
						if (std::fscanf(fpCore, "%d", &coreId) != 1) coreId = -1;
						std::fclose(fpCore);
					}
					if (coreId >= 0) {
						physicalCorePairs.insert({pkgId, coreId});
					}

					// Read cache index3 (L3)
					for (int idx = 0; idx <= 4; idx++) {
						std::string idxPath = cpuPath + "/cache/index" + std::to_string(idx);
						std::string levelPath = idxPath + "/level";
						FILE* fpLvl = std::fopen(levelPath.c_str(), "r");
						if (fpLvl) {
							int lvl = 0;
							if (std::fscanf(fpLvl, "%d", &lvl) == 1 && lvl == 3) {
								// Read shared_cpu_list
								std::string sharedPath = idxPath + "/shared_cpu_list";
								char sbuf[256] = {0};
								FILE* fpShared = std::fopen(sharedPath.c_str(), "r");
								if (fpShared) {
									if (std::fgets(sbuf, sizeof(sbuf), fpShared)) {
										l3Clusters.insert(std::string(sbuf));
									}
									std::fclose(fpShared);
								}
								// Read size
								if (topo.l3PerCluster == 0) {
									std::string sizePath = idxPath + "/size";
									char szBuf[64] = {0};
									FILE* fpSz = std::fopen(sizePath.c_str(), "r");
									if (fpSz) {
										if (std::fgets(szBuf, sizeof(szBuf), fpSz)) {
											topo.l3PerCluster = ParseSizeWithSuffix(szBuf);
										}
										std::fclose(fpSz);
									}
								}
							}
							std::fclose(fpLvl);
						}
					}
				}
			}
		}
		closedir(dir);
	}
	if (!physicalCorePairs.empty()) {
		topo.physicalCores = physicalCorePairs.size();
	}
	if (!l3Clusters.empty()) {
		topo.numClusters = l3Clusters.size();
		topo.totalL3 = topo.l3PerCluster * topo.numClusters;
	}
#elif defined(__APPLE__)
	int val = 0;
	size_t len = sizeof(val);
	if (sysctlbyname("hw.physicalcpu", &val, &len, nullptr, 0) == 0 && val > 0) {
		topo.physicalCores = static_cast<size_t>(val);
	}
	len = sizeof(val);
	if (sysctlbyname("hw.logicalcpu", &val, &len, nullptr, 0) == 0 && val > 0) {
		topo.logicalCores = static_cast<size_t>(val);
	}
	uint64_t l3Size = 0;
	len = sizeof(l3Size);
	if (sysctlbyname("hw.l3cachesize", &l3Size, &len, nullptr, 0) == 0 && l3Size > 0) {
		topo.l3PerCluster = static_cast<size_t>(l3Size);
	} else {
		len = sizeof(l3Size);
		if (sysctlbyname("hw.perflevel0.l3cachesize", &l3Size, &len, nullptr, 0) == 0 && l3Size > 0) {
			topo.l3PerCluster = static_cast<size_t>(l3Size);
		} else {
			// Fall back to performance cluster L2 if no L3
			len = sizeof(l3Size);
			if (sysctlbyname("hw.perflevel0.l2cachesize", &l3Size, &len, nullptr, 0) == 0 && l3Size > 0) {
				topo.l3PerCluster = static_cast<size_t>(l3Size);
			}
		}
	}
	topo.numClusters = 1;
	topo.totalL3 = topo.l3PerCluster;
#endif

	// Sanity fallbacks
	if (topo.logicalCores == 0) {
		topo.logicalCores = std::thread::hardware_concurrency();
		if (topo.logicalCores == 0) topo.logicalCores = 1;
	}
	if (topo.physicalCores == 0) {
		topo.physicalCores = topo.logicalCores;
	}
	if (topo.numClusters == 0) {
		topo.numClusters = 1;
	}
	if (topo.l3PerCluster == 0) {
		topo.l3PerCluster = 32ULL * 1024 * 1024; // 32 MiB fallback
		topo.totalL3 = topo.l3PerCluster * topo.numClusters;
	}
	if (topo.totalL3 == 0) {
		topo.totalL3 = topo.l3PerCluster * topo.numClusters;
	}

	return topo;
}

static std::string GetEnvVar(const char* name) {
#if defined(_WIN32)
	char buf[256];
	DWORD len = GetEnvironmentVariableA(name, buf, sizeof(buf));
	if (len > 0 && len < sizeof(buf)) {
		return std::string(buf, len);
	}
	return "";
#else
	const char* val = std::getenv(name);
	return val ? std::string(val) : "";
#endif
}

CpuTopology GetCpuTopology() {
	std::lock_guard<std::mutex> lock(s_topoMutex);
	if (s_isCached) {
		return s_cachedTopology;
	}

	CpuTopology topo = DetectRawTopology();

	// Check environment overrides for testing / benchmarking
	std::string mockL3 = GetEnvVar("PAR3_MOCK_L3_BYTES");
	if (!mockL3.empty()) {
		size_t parsed = std::strtoull(mockL3.c_str(), nullptr, 10);
		if (parsed > 0) {
			topo.l3PerCluster = parsed;
			topo.totalL3 = topo.l3PerCluster * topo.numClusters;
		}
	}

	std::string mockCores = GetEnvVar("PAR3_MOCK_CORES");
	if (!mockCores.empty()) {
		size_t parsed = std::strtoull(mockCores.c_str(), nullptr, 10);
		if (parsed > 0) {
			topo.physicalCores = parsed;
			topo.logicalCores = parsed;
		}
	}

	std::string mockClusters = GetEnvVar("PAR3_MOCK_CLUSTERS");
	if (!mockClusters.empty()) {
		size_t parsed = std::strtoull(mockClusters.c_str(), nullptr, 10);
		if (parsed > 0) {
			topo.numClusters = parsed;
			topo.totalL3 = topo.l3PerCluster * topo.numClusters;
		}
	}

	s_cachedTopology = topo;
	s_isCached = true;
	return s_cachedTopology;
}

void ResetCpuTopologyCache() {
	std::lock_guard<std::mutex> lock(s_topoMutex);
	s_isCached = false;
	s_cachedTopology = {0, 0, 0, 0, 0};
}
