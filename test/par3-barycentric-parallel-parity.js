#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 Barycentric Parallel Parity Test (issue #46 Phase A)
//
// Pins the parallelization contract of ComputeRecoveryBlocksBarycentric
// (src/par3_engine_barycentric.cc, 2026-08-13):
//
//   1. numThreads=8 output is BYTE-IDENTICAL to numThreads=1 output on the
//      same inputs (row-stealing workers must not change the math).
//   2. numThreads=8 output matches the pure-JS GF(2^64) Cauchy reference
//      (the same reference test/par3-barycentric-parity.js uses).
//   3. numThreads=-1/0 (auto) output matches numThreads=1 as well — the
//      NAPI default must engage the parallel path without corrupting it.
//
// Regression guard: the v1 engine ignored numThreads entirely (single-
// threaded rows), which regressed create throughput ~6x on native Windows
// (1 GiB / 10K: 28.3 -> 4.78 MB/s). This test ensures the parallel path
// stays bit-exact; a future change that breaks row independence (shared
// scratch, aliasing, races on the output rows) fails here.
// ============================================================================

var assert = require('node:assert');
var path = require('path');

var addon;
try {
	addon = require('../build/Release/parpar_gf64.node');
} catch (e) {
	console.log('SKIPPED: native module not available');
	process.exit(0);
}

if (typeof addon.compute_recovery_barycentric !== 'function') {
	console.log('SKIPPED: compute_recovery_barycentric NAPI export not present');
	process.exit(0);
}

console.log('PAR3 Barycentric Parallel Parity Test (issue #46 Phase A)');
console.log('==========================================================\n');

var passed = 0;
var failed = 0;
var total = 0;

function check(condition, msg) {
	total++;
	if (condition) {
		console.log('  PASS: ' + msg);
		passed++;
	} else {
		console.error('  FAIL: ' + msg);
		failed++;
		process.exitCode = 1;
	}
}

// --- Pure-JS GF(2^64) Cauchy reference (mirrors test/par3-barycentric-parity.js) ---

var GF64_POLY = 0x1000000000000001Bn;
var GF64_MASK = 0xFFFFFFFFFFFFFFFFn;

function gf64_mul(a, b) {
	var result = 0n;
	while (b !== 0n) {
		if ((b & 1n) !== 0n) result ^= a;
		a <<= 1n;
		if ((a & 0x10000000000000000n) !== 0n) a ^= (GF64_POLY & GF64_MASK);
		b >>= 1n;
	}
	return result & GF64_MASK;
}

function gf64_inv(a) {
	// Fermat: a^(2^64-2) = a^-1 via square-and-multiply.
	if (a === 0n) return 0n;
	var result = 1n;
	var base = a;
	var exp = 0xFFFFFFFFFFFFFFFEn; // 2^64 - 2
	while (exp > 0n) {
		if ((exp & 1n) !== 0n) result = gf64_mul(result, base);
		base = gf64_mul(base, base);
		exp >>= 1n;
	}
	return result;
}

// Deterministic pseudo-random 64-bit words (xorshift64) so the test is
// reproducible across runs and platforms.
var seed = 0x9E3779B97F4A7C15n;
function nextWord() {
	seed ^= seed << 13n;
	seed ^= seed >> 7n;
	seed ^= seed << 17n;
	return seed & GF64_MASK;
}

function jsCauchyOutputs(numInputs, numRecovery, blockSize64, firstInput, firstRecovery, inputWords) {
	var out = Buffer.alloc(numRecovery * blockSize64 * 8);
	for (var r = 0; r < numRecovery; r++) {
		var y = BigInt(firstRecovery + r);
		for (var c = 0; c < numInputs; c++) {
			var x = BigInt(firstInput + c);
			var denom = x ^ y;
			if (denom === 0n) denom = 1n;
			var coeff = gf64_inv(denom);
			for (var w = 0; w < blockSize64; w++) {
				var word = inputWords[c * blockSize64 + w];
				var prod = gf64_mul(word, coeff);
				var off = (r * blockSize64 + w) * 8;
				out.writeBigUInt64LE(out.readBigUInt64LE(off) ^ prod, off);
			}
		}
	}
	return out;
}

// --- Scenario runner --------------------------------------------------------

function runScenario(numInputs, numRecovery, blockSize64, firstInput, firstRecovery) {
	var label = 'N=' + numInputs + ' R=' + numRecovery + ' B64=' + blockSize64;
	var inputWords = [];
	for (var i = 0; i < numInputs * blockSize64; i++) {
		inputWords.push(nextWord());
	}
	var inputs = Buffer.alloc(numInputs * blockSize64 * 8);
	for (var w = 0; w < inputWords.length; w++) {
		inputs.writeBigUInt64LE(inputWords[w], w * 8);
	}

	function runWith(threads) {
		var outputs = Buffer.alloc(numRecovery * blockSize64 * 8);
		var t0 = Date.now();
		addon.compute_recovery_barycentric(inputs, outputs, numInputs, numRecovery, blockSize64 * 8, firstInput, firstRecovery, threads);
		var ms = Date.now() - t0;
		return { outputs: outputs, ms: ms };
	}

	var serial = runWith(1);
	var parallel = runWith(8);
	var auto = runWith(0);

	check(serial.outputs.equals(parallel.outputs), label + ': numThreads=8 byte-identical to numThreads=1 (serial ' + serial.ms + 'ms, parallel ' + parallel.ms + 'ms)');
	check(serial.outputs.equals(auto.outputs), label + ': numThreads=0 (auto) byte-identical to numThreads=1');

	// JS-reference comparison only at small scale (BigInt reference is
	// O(N*R*B64) BigInt ops — see comment at the scenario list).
	if (numInputs * numRecovery * blockSize64 <= 262144) {
		var ref = jsCauchyOutputs(numInputs, numRecovery, blockSize64, firstInput, firstRecovery, inputWords);
		check(serial.outputs.equals(ref), label + ': native matches pure-JS Cauchy reference');
	}
}

// Two scenarios:
//   1. Small JS-reference scenario routed through the REAL Barycentric
//      path: R=64 > the engine's numRecovery<=32 ComputeRecoveryBlocks
//      shortcut, so processRow (denoms -> batch invert -> muladd) runs;
//      N*R*B64 = 64*64*64 = 262144 stays within the pure-JS BigInt
//      reference budget (the reference is O(N*R*B64) BigInt ops — keep
//      it at the existing parity test's scale or it takes minutes).
//   2. Large: native-vs-native only (serial vs parallel vs auto) at a
//      scale that spreads rows across all 8 workers (256 rows). N/R kept
//      moderate: the per-row Itoh-Tsujii batch inversion is ~130ms per
//      262K-element row, so bench-scale shapes would take minutes in the
//      serial leg.
runScenario(64, 64, 64, 1000, 100000);
runScenario(32768, 256, 8, 0, 32768);

console.log('\n' + passed + ' passed, ' + failed + ' failed');
if (failed) process.exit(1);
process.exit(0);
