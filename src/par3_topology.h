#ifndef __PAR3_TOPOLOGY_H
#define __PAR3_TOPOLOGY_H

#include <cstddef>
#include <cstdint>

struct CpuTopology {
	size_t physicalCores;
	size_t logicalCores;
	size_t l3PerCluster;  // L3 cache bytes per CCD / CCX sharing domain
	size_t numClusters;   // Number of L3 cache sharing clusters (CCDs)
	size_t totalL3;       // Total L3 cache in bytes across all clusters
};

CpuTopology GetCpuTopology();
void ResetCpuTopologyCache();

#endif // __PAR3_TOPOLOGY_H
