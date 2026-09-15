// test/par3-topology-detection.js — Issue #119: CPU topology detection contract tests
"use strict";

var assert = require("assert");
var par3gen = require("../lib/par3gen");

var passed = 0;
var failed = 0;

function pass(name) {
	console.log("  PASS: " + name);
	passed++;
}

function fail(name, err) {
	console.error("  FAIL: " + name, err || "");
	failed++;
}

console.log("PAR3 CPU Topology Detection Tests (Issue #119)");
console.log("==============================================\n");

var binding = null;
try {
	binding = par3gen.gf64Binding;
	if (!binding) {
		binding = require("../build/Release/parpar_gf64.node");
	}
} catch (e) {
	binding = null;
}

var isX86ish = (process.arch === "x64" || process.arch === "ia32");
if (!binding || typeof binding !== "object") {
	if (isX86ish) {
		fail("Native addon could not be loaded on x86 platform");
		console.log("\n=================================");
		console.log("Summary: 0 passed, 1 failed");
		console.log("=================================");
		process.exit(1);
	} else {
		console.log("SKIP: native addon not available on non-x86 platform");
		process.exit(0);
	}
}

// 1. Native binding exports
if (typeof binding.get_cpu_topology === "function") {
	pass("binding.get_cpu_topology is exported as a function");
} else {
	fail("binding.get_cpu_topology is NOT exported");
}

if (typeof binding.reset_cpu_topology_cache === "function") {
	pass("binding.reset_cpu_topology_cache is exported as a function");
} else {
	fail("binding.reset_cpu_topology_cache is NOT exported");
}

if (failed > 0) {
	console.log("\n=================================");
	console.log("Summary: " + passed + " passed, " + failed + " failed");
	console.log("=================================");
	process.exit(1);
}

// 2. Query host topology
var hostTopo = null;
try {
	hostTopo = binding.get_cpu_topology();
} catch (err) {
	fail("binding.get_cpu_topology() threw an error", err);
	process.exit(1);
}

console.log("Host Topology detected:");
console.log("  Physical Cores: " + hostTopo.physicalCores);
console.log("  Logical Cores:  " + hostTopo.logicalCores);
console.log("  L3 Per Cluster: " + (hostTopo.l3PerCluster / 1048576).toFixed(2) + " MiB (" + hostTopo.l3PerCluster + " bytes)");
console.log("  Num Clusters:   " + hostTopo.numClusters);
console.log("  Total L3:       " + (hostTopo.totalL3 / 1048576).toFixed(2) + " MiB (" + hostTopo.totalL3 + " bytes)");

assert(hostTopo.physicalCores > 0, "physicalCores must be > 0");
assert(hostTopo.logicalCores >= hostTopo.physicalCores, "logicalCores must be >= physicalCores");
assert(hostTopo.l3PerCluster > 0, "l3PerCluster must be > 0");
assert(hostTopo.numClusters >= 1, "numClusters must be >= 1");
assert(hostTopo.totalL3 >= hostTopo.l3PerCluster, "totalL3 must be >= l3PerCluster");
pass("Host topology returns valid physical/logical cores and L3 cache sizes");

// 3. JS lib/par3gen.js getCpuTopology wrapper
var jsTopo = par3gen.getCpuTopology();
assert.strictEqual(jsTopo.physicalCores, hostTopo.physicalCores, "js physicalCores matches native");
assert.strictEqual(jsTopo.logicalCores, hostTopo.logicalCores, "js logicalCores matches native");
assert.strictEqual(jsTopo.l3PerCluster, hostTopo.l3PerCluster, "js l3PerCluster matches native");
assert.strictEqual(jsTopo.numClusters, hostTopo.numClusters, "js numClusters matches native");
pass("par3gen.getCpuTopology() wrapper matches native binding");

// 4. Test PAR3_MOCK_* environment overrides and cache invalidation
try {
	process.env.PAR3_MOCK_L3_BYTES = String(48 * 1024 * 1024);
	process.env.PAR3_MOCK_CORES = "16";
	process.env.PAR3_MOCK_CLUSTERS = "2";

	// Before cache reset, cached topology is still returned
	var stillCached = binding.get_cpu_topology();
	assert.strictEqual(stillCached.physicalCores, hostTopo.physicalCores, "cached topology retained before reset");
	pass("Topology is safely cached across calls");

	// After cache reset, mock values take effect
	par3gen.resetCpuTopologyCache();
	var mocked = par3gen.getCpuTopology();
	assert.strictEqual(mocked.l3PerCluster, 48 * 1024 * 1024, "mocked l3PerCluster");
	assert.strictEqual(mocked.physicalCores, 16, "mocked physicalCores");
	assert.strictEqual(mocked.logicalCores, 16, "mocked logicalCores");
	assert.strictEqual(mocked.numClusters, 2, "mocked numClusters");
	assert.strictEqual(mocked.totalL3, 96 * 1024 * 1024, "mocked totalL3 = clusters * l3PerCluster");
	pass("PAR3_MOCK_* env overrides work as expected after resetCpuTopologyCache()");
} finally {
	delete process.env.PAR3_MOCK_L3_BYTES;
	delete process.env.PAR3_MOCK_CORES;
	delete process.env.PAR3_MOCK_CLUSTERS;
	par3gen.resetCpuTopologyCache();
}

// 5. Restore clean host topology
var restored = par3gen.getCpuTopology();
assert.strictEqual(restored.physicalCores, hostTopo.physicalCores, "restored physicalCores");
assert.strictEqual(restored.l3PerCluster, hostTopo.l3PerCluster, "restored l3PerCluster");
pass("Host topology cleanly restored after clearing mock env");

console.log("\n=================================");
console.log("Summary: " + passed + " passed, " + failed + " failed");
console.log("=================================");

if (failed > 0) process.exit(1);
