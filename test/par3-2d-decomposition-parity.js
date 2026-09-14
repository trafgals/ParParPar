// test/par3-2d-decomposition-parity.js — Issue #119: 2D Block-Domain Decomposition Parity Tests
"use strict";

var assert = require("assert");
var crypto = require("crypto");
var par3gen = require("../lib/par3gen");
var binding = par3gen.gf64Binding || require("../build/Release/parpar_gf64.node");

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

console.log("PAR3 2D Block-Domain Decomposition Parity Tests (Issue #119)");
console.log("============================================================\n");

// Ensure native binding is present and supports compute_recovery_full
if (!binding || typeof binding.compute_recovery_full !== "function") {
	console.error("Native binding with compute_recovery_full not available; skipping");
	process.exit(0);
}

// Helper to fill buffer with deterministic pseudorandom data
function fillPattern(buf) {
	for (var i = 0; i < buf.length; i += 4) {
		var v = ((i * 1103515245 + 12345) >>> 0);
		buf.writeUInt32LE(v, i);
	}
}

// Test matrix of configurations:
// Shape: { N, R, blockSize, threadCounts }
var testCases = [
	// R=2, B=128 KiB: S_B will be up to 16 with T=32
	{ N: 8, R: 2, B: 128 * 1024, threads: [1, 2, 4, 8, 16, 32] },
	// R=4, B=256 KiB: S_B will be up to 8 with T=32
	{ N: 16, R: 4, B: 256 * 1024, threads: [1, 2, 4, 8, 16, 32, 64] },
	// R=8, B=512 KiB: S_B will be up to 8 with T=64
	{ N: 32, R: 8, B: 512 * 1024, threads: [1, 2, 8, 16, 32, 64] },
	// Non-power-of-two thread counts (e.g. 3, 5, 6, 7, 12)
	{ N: 16, R: 2, B: 256 * 1024, threads: [3, 5, 6, 7, 12] },
	// Large block size B=1 MiB with R=2 and high thread count T=16
	{ N: 4, R: 2, B: 1024 * 1024, threads: [1, 4, 8, 16] },
];

for (var tc = 0; tc < testCases.length; tc++) {
	var c = testCases[tc];
	var inSize = c.N * c.B;
	var outSize = c.R * c.B;

	var inBuf = Buffer.alloc(inSize);
	fillPattern(inBuf);

	// Compute serial reference with T=1
	var refOut = Buffer.alloc(outSize);
	// Force accumulate=false
	binding.compute_recovery_full(inBuf, refOut, c.N, c.R, c.B, 0, c.N, 1, false);

	// Verify refOut is non-zero
	var nonZero = false;
	for (var z = 0; z < Math.min(refOut.length, 1024); z++) {
		if (refOut[z] !== 0) { nonZero = true; break; }
	}
	assert(nonZero, "Reference output must be non-zero");

	// Test each thread count
	for (var ti = 0; ti < c.threads.length; ti++) {
		var T = c.threads[ti];
		var testOut = Buffer.alloc(outSize);

		// Issue #117: accumulate=true forces Case 3 (2D decomposition), or when T > c.R and scratch cap is exceeded
		// Let's test with accumulate=true into zeroed buffer
		binding.compute_recovery_full(inBuf, testOut, c.N, c.R, c.B, 0, c.N, T, true);
		var path = binding.get_last_decomposition_path();

		if (testOut.equals(refOut)) {
			pass("N=" + c.N + " R=" + c.R + " B=" + (c.B / 1024) + "K T=" + T + " (path " + path + ") matches serial reference bit-exact");
		} else {
			fail("N=" + c.N + " R=" + c.R + " B=" + (c.B / 1024) + "K T=" + T + " output MISMATCH against serial reference");
		}

		if (T > 1) {
			assert.strictEqual(path, 3, "accumulate=true with T=" + T + " must use Case 3 (2D block decomposition)");
		}
	}

	// Test multi-pass accumulation (accumulating 2 inputs, then next 2 inputs, etc.)
	var accOut = Buffer.alloc(outSize);
	var batchInputs = 2;
	for (var b = 0; b < c.N; b += batchInputs) {
		var curN = Math.min(batchInputs, c.N - b);
		var sliceIn = inBuf.subarray(b * c.B, (b + curN) * c.B);
		binding.compute_recovery_full(sliceIn, accOut, curN, c.R, c.B, b, c.N, 16, true);
	}
	if (accOut.equals(refOut)) {
		pass("N=" + c.N + " R=" + c.R + " B=" + (c.B / 1024) + "K multi-chunk accumulation with T=16 matches serial bit-exact");
	} else {
		fail("N=" + c.N + " R=" + c.R + " B=" + (c.B / 1024) + "K multi-chunk accumulation MISMATCH");
	}
}

console.log("\n=================================");
console.log("Summary: " + passed + " passed, " + failed + " failed");
console.log("=================================");

if (failed > 0) process.exit(1);
