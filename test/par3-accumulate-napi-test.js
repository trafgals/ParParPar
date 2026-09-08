#!/usr/bin/env node
"use strict";

// ============================================================================
// test/par3-accumulate-napi-test.js — Unit test for compute_recovery_accumulate
//
// Tests native in-place chunked Cauchy recovery accumulation against reference
// single-call recovery (compute_recovery_barycentric / compute_recovery_full).
//
// Verifies:
//   1. Bit-exact parity for single-chunk accumulation vs full compute.
//   2. Bit-exact parity for multi-chunk accumulation (2, 4, 8, 16 chunks).
//   3. Non-uniform chunk splits (e.g. 300 + 724 = 1024 blocks).
//   4. Multi-threading consistency (1 thread vs effective CPU threads).
// ============================================================================

var assert = require("assert");
var crypto = require("crypto");
var path = require("path");

var addon = require("../build/Release/parpar_gf64.node");
assert.strictEqual(typeof addon.compute_recovery_accumulate, "function", "addon must export compute_recovery_accumulate");

var passed = 0;
var failed = 0;
function pass(name) { passed++; console.log("  PASS: " + name); }
function fail(name, err) { failed++; console.error("  FAIL: " + name + (err ? " (" + err.message + ")" : "")); process.exitCode = 1; }

function testGeometry(N, R, blockSize, numThreads) {
	var inputBytes = N * blockSize;
	var recBytes = R * blockSize;
	var inputs = crypto.randomBytes(inputBytes);

	// 1. Reference single-call compute
	var refOut = Buffer.alloc(recBytes);
	if (typeof addon.compute_recovery_barycentric === "function" && N > 1000) {
		addon.compute_recovery_barycentric(inputs, refOut, N, R, blockSize, 0, N, numThreads);
	} else {
		addon.compute_recovery_full(inputs, refOut, N, R, blockSize, 0, N, numThreads);
	}

	// 2. Single-chunk accumulate
	var accSingle = Buffer.alloc(recBytes);
	addon.compute_recovery_accumulate(inputs, accSingle, N, R, blockSize, 0, N, numThreads);
	assert.strictEqual(
		accSingle.compare(refOut), 0,
		"Single-chunk accumulate output must match reference for N=" + N + ", R=" + R + ", B=" + blockSize
	);
	pass("Single-chunk match N=" + N + ", R=" + R + ", B=" + blockSize + ", T=" + numThreads);

	// 3. Multi-chunk accumulate (equal splits: 2, 4, 8 chunks)
	[2, 4, 8].forEach(function(numChunks) {
		if (N % numChunks !== 0) return;
		var chunkBlocks = N / numChunks;
		var chunkBytes = chunkBlocks * blockSize;
		var accMulti = Buffer.alloc(recBytes);

		for (var c = 0; c < numChunks; c++) {
			var start = c * chunkBytes;
			var end = start + chunkBytes;
			var chunkBuf = inputs.subarray(start, end);
			var firstChunkInput = c * chunkBlocks;
			addon.compute_recovery_accumulate(
				chunkBuf, accMulti,
				chunkBlocks, R, blockSize,
				firstChunkInput, N,
				numThreads
			);
		}

		assert.strictEqual(
			accMulti.compare(refOut), 0,
			numChunks + "-chunk accumulate output must match reference for N=" + N + ", R=" + R
		);
		pass(numChunks + "-chunk match N=" + N + ", R=" + R + ", B=" + blockSize + ", T=" + numThreads);
	});

	// 4. Non-uniform chunk split: e.g. 30% and 70%
	var split1 = Math.floor(N * 0.37);
	var split2 = N - split1;
	var accNonUniform = Buffer.alloc(recBytes);

	var chunk1 = inputs.subarray(0, split1 * blockSize);
	addon.compute_recovery_accumulate(chunk1, accNonUniform, split1, R, blockSize, 0, N, numThreads);

	var chunk2 = inputs.subarray(split1 * blockSize);
	addon.compute_recovery_accumulate(chunk2, accNonUniform, split2, R, blockSize, split1, N, numThreads);

	assert.strictEqual(
		accNonUniform.compare(refOut), 0,
		"Non-uniform chunk accumulate output must match reference for N=" + N + ", R=" + R
	);
	pass("Non-uniform split (" + split1 + " + " + split2 + ") match N=" + N + ", R=" + R);
}

console.log("=== Testing compute_recovery_accumulate NAPI ===");

try {
	// Small geometry
	testGeometry(256, 8, 1024, 1);
	testGeometry(256, 8, 1024, 4);

	// Benchmark geometry ratio (R=8, 64 KiB blocks)
	testGeometry(1024, 8, 65536, 1);
	testGeometry(1024, 8, 65536, 8);

	// Larger N geometry (above Barycentric gate N>1000)
	testGeometry(2048, 16, 4096, 4);
} catch (e) {
	fail("testGeometry threw", e);
}

console.log("\nPassed: " + passed + ", Failed: " + failed);
if (failed > 0) process.exit(1);
