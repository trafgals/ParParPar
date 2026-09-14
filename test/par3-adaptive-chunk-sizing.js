// test/par3-adaptive-chunk-sizing.js — Issue #119: Topology-Adaptive Chunk Sizing Tests
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

console.log("PAR3 Topology-Adaptive Chunk Sizing Tests (Issue #119)");
console.log("======================================================\n");

// 1. Export verification
if (typeof par3gen.getOptimalChunkCap === "function") {
	pass("par3gen.getOptimalChunkCap is exported");
} else {
	fail("par3gen.getOptimalChunkCap is NOT exported");
}

if (typeof par3gen.decideChunkCapBytes === "function") {
	pass("par3gen.decideChunkCapBytes is exported");
} else {
	fail("par3gen.decideChunkCapBytes is NOT exported");
}

// 2. Hardware profile assertions on getOptimalChunkCap(topology, blockSize)
var topo7800X3D = {
	physicalCores: 8,
	logicalCores: 8,
	l3PerCluster: 96 * 1024 * 1024,
	numClusters: 1,
	totalL3: 96 * 1024 * 1024
};

var cap7800 = par3gen.getOptimalChunkCap(topo7800X3D, 64 * 1024);
assert.strictEqual(cap7800, 64 * 1024 * 1024, "7800X3D profile yields exact 64 MiB chunk cap");
pass("AMD Ryzen 7 7800X3D (96 MiB L3) yields exact 64 MiB chunk cap");

// Ryzen 7600 / Intel standard desktop (32 MiB L3)
var topo7600 = {
	physicalCores: 6,
	logicalCores: 12,
	l3PerCluster: 32 * 1024 * 1024,
	numClusters: 1,
	totalL3: 32 * 1024 * 1024
};

var cap7600_64k = par3gen.getOptimalChunkCap(topo7600, 64 * 1024);
var expected7600 = Math.floor(Math.floor(32 * 1024 * 1024 * 2 / 3) / (64 * 1024)) * (64 * 1024);
assert.strictEqual(cap7600_64k, expected7600, "7600 profile yields 2/3 of 32M aligned to 64K");
assert.strictEqual(cap7600_64k % (64 * 1024), 0, "7600 chunk cap is aligned to 64 KiB");
pass("AMD Ryzen 5 7600 (32 MiB L3) yields ~21.3 MiB aligned chunk cap (" + (cap7600_64k / 1048576).toFixed(2) + " MiB)");

// Mobile laptop (16 MiB L3)
var topoLaptop = {
	physicalCores: 4,
	logicalCores: 8,
	l3PerCluster: 16 * 1024 * 1024,
	numClusters: 1,
	totalL3: 16 * 1024 * 1024
};

var capLaptop = par3gen.getOptimalChunkCap(topoLaptop, 64 * 1024);
assert.strictEqual(capLaptop, 16 * 1024 * 1024, "Laptop 16 MiB L3 clamps to 16 MiB floor");
pass("Mobile / Laptop (16 MiB L3) clamps to 16 MiB minimum chunk cap");

// Ultra-low budget CPU (4 MiB L3) -> clamps to 16 MiB
var topoLow = {
	physicalCores: 2,
	logicalCores: 4,
	l3PerCluster: 4 * 1024 * 1024,
	numClusters: 1,
	totalL3: 4 * 1024 * 1024
};

var capLow = par3gen.getOptimalChunkCap(topoLow, 64 * 1024);
assert.strictEqual(capLow, 16 * 1024 * 1024, "Low L3 clamps to 16 MiB floor");
pass("Ultra-low L3 CPU (4 MiB L3) clamps to 16 MiB minimum chunk cap");

// Large block size B=32 MiB exceeds 16 MiB clamp
var capLargeBlock = par3gen.getOptimalChunkCap(topoLaptop, 32 * 1024 * 1024);
assert.strictEqual(capLargeBlock, 32 * 1024 * 1024, "Large block size B=32M sets chunk cap to at least B");
pass("Large block size B=32 MiB dynamically sets chunk cap to B");

// 3. decideChunkCapBytes integration with topology
var capDecide7800 = par3gen.decideChunkCapBytes(32 * 1024 * 1024 * 1024, 8, 64 * 1024, "matvec", null, topo7800X3D);
assert.strictEqual(capDecide7800, 64 * 1024 * 1024, "decideChunkCapBytes uses 7800X3D topology to produce 64 MiB");
pass("decideChunkCapBytes properly integrates 7800X3D topology");

var capDecide7600 = par3gen.decideChunkCapBytes(32 * 1024 * 1024 * 1024, 8, 64 * 1024, "matvec", null, topo7600);
assert.strictEqual(capDecide7600, expected7600, "decideChunkCapBytes uses 7600 topology to produce ~21.3 MiB");
pass("decideChunkCapBytes properly integrates 7600 topology");

// 4. Overrides and bounding contracts are preserved
// Environment override
process.env.PAR3_STREAM_CHUNK_BYTES = String(8 * 1024 * 1024);
var capEnv = par3gen.decideChunkCapBytes(32 * 1024 * 1024 * 1024, 8, 64 * 1024, "matvec", null, topo7800X3D);
delete process.env.PAR3_STREAM_CHUNK_BYTES;
assert.strictEqual(capEnv, 8 * 1024 * 1024, "PAR3_STREAM_CHUNK_BYTES overrides topology-adaptive sizing");
pass("PAR3_STREAM_CHUNK_BYTES overrides topology chunk cap");

// Memory limit clamping (maxMemoryBudget / 2)
var capMemLimit = par3gen.decideChunkCapBytes(32 * 1024 * 1024 * 1024, 8, 64 * 1024, "matvec", 32 * 1024 * 1024, topo7800X3D);
assert.strictEqual(capMemLimit, 16 * 1024 * 1024, "memoryLimit=32M clamps chunk cap to 16M (memoryLimit/2)");
pass("memoryLimit clamp (memoryLimit/2) is preserved");

// Reserve deduction (384M - 352M reserve = 32M)
var capReserve = par3gen.decideChunkCapBytes(32 * 1024 * 1024 * 1024, 2816, 128 * 1024, "matvec", null, topo7800X3D);
assert.strictEqual(capReserve, 32 * 1024 * 1024, "352M reserve clamps 64M cap to 32M available budget");
pass("Recovery reserve deduction properly clamps chunk cap");

// Insufficient memory -> returns 0
var capZero = par3gen.decideChunkCapBytes(32 * 1024 * 1024 * 1024, 4096, 128 * 1024, "matvec", null, topo7800X3D);
assert.strictEqual(capZero, 0, "exceeding budget returns 0");
pass("Exceeding memory budget returns 0 unconditionally");

console.log("\n=================================");
console.log("Summary: " + passed + " passed, " + failed + " failed");
console.log("=================================");

if (failed > 0) process.exit(1);
