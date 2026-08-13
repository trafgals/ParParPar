#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 GF64 Region MulArr Parity Test (Standalone TDD Red)
// ----------------------------------------------------------------------------
// Tests gf64_region_mul_arr via the NAPI binding `mul_arr` against a pure
// JS BigInt reference implementation.  Each (n_coeff, len) cell gets ten
// independent randomized inputs.
//
// Parameters — n_coeff ∈ {2, 4, 8, 16}, len ∈ {8, 64, 256, 1024}
// Total cells: 4 × 4 = 16  →  16 × 10 = 160 individual runs per launch.
//
// Usage:
//   node test/par3-gf64-mularr-parity.js
//
// Expected: exits with code 1 (TDD red) until the _arr kernels are green.
// ============================================================================

var addon = require('../build/Release/parpar_gf64.node');

// ============================================================================
// Pure JS GF(2^64) reference
// ============================================================================

var GF64_POLY = 0x1000000000000001Bn;
var GF64_MASK = 0xFFFFFFFFFFFFFFFFn;

function gf64_mul(a, b) {
	var result = 0n;
	while (b !== 0n) {
		if ((b & 1n) !== 0n) {
			result ^= a;
		}
		a <<= 1n;
		if ((a & 0x10000000000000000n) !== 0n) {
			a ^= 0x1Bn;
		}
		b >>= 1n;
	}
	return result & GF64_MASK;
}

// ============================================================================
// Deterministic PRNG (mulberry32)
// ============================================================================

var SEED = 0xC0FFEE00;

function mulberry32(seed) {
	return function() {
		seed |= 0;
		seed = seed + 0x6D2B79F5 | 0;
		var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
		t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
		return ((t ^ t >>> 14) >>> 0) / 4294967296;
	};
}

// ============================================================================
// Helpers
// ============================================================================

function fillWord64LE(buf, rng) {
	var words = buf.length / 8;
	for (var w = 0; w < words; w++) {
		var hi = (rng() * 4294967296) >>> 0;
		var lo = (rng() * 4294967296) >>> 0;
		var val = (BigInt(hi) << 32n) | BigInt(lo);
		buf.writeBigUInt64LE(val, w * 8);
	}
}

function bufToBigIntArray(buf) {
	var arr = [];
	var words = buf.length / 8;
	for (var i = 0; i < words; i++) {
		arr[i] = buf.readBigUInt64LE(i * 8);
	}
	return arr;
}

// ============================================================================
// JS reference for the _arr operation
// ----------------------------------------------------------------------------
// out[i] = XOR_c( gf64_mul(in[i], coeff[c]) )   for c = 0 .. n_coeff-1
// ============================================================================

function jsMulArrRef(inBuf, coeffBuf, nCoeff, numWords) {
	var inpWords = bufToBigIntArray(inBuf);
	var coeffWords = bufToBigIntArray(coeffBuf);
	var out = [];
	for (var w = 0; w < numWords; w++) {
		var sum = 0n;
		for (var c = 0; c < nCoeff; c++) {
			sum ^= gf64_mul(inpWords[w], coeffWords[c]);
		}
		out[w] = sum;
	}
	return out;
}

// ============================================================================
// Test state
// ============================================================================

var passed = 0;
var failed = 0;
var failuresByCell = {};

// ============================================================================
// Main
// ============================================================================

var encoder;
try {
	encoder = new addon.Gf64Encoder(0);
} catch (e) {
	console.error('FATAL: Could not create Gf64Encoder: ' + e.message);
	process.exit(1);
}

var N_COEFFS = [2, 4, 8, 16];
var LENS     = [8, 64, 256, 1024];

var rng = mulberry32(SEED);

console.log('PAR3 GF64 Region MulArr Parity Test (TDD Red)');
console.log('================================================\n');

for (var ci = 0; ci < N_COEFFS.length; ci++) {
	var nCoeff = N_COEFFS[ci];
	for (var li = 0; li < LENS.length; li++) {
		var len = LENS[li];
		var numWords = len / 8;
		var cellKey = 'n_coeff=' + nCoeff + ', len=' + len;
		var cellFailed = false;

		for (var trial = 0; trial < 10; trial++) {
			var inBuf   = Buffer.alloc(len);
			var coeffBuf = Buffer.alloc(nCoeff * 8);

			fillWord64LE(inBuf, rng);
			fillWord64LE(coeffBuf, rng);

			var expected = jsMulArrRef(inBuf, coeffBuf, nCoeff, numWords);

			var outBuf = Buffer.alloc(len);
			outBuf.fill(0);
			encoder.mul_arr(outBuf, inBuf, coeffBuf, numWords, nCoeff);
			var got = bufToBigIntArray(outBuf);

			var eq = true;
			for (var w = 0; w < numWords; w++) {
				if (got[w] !== expected[w]) {
					eq = false;
					break;
				}
			}

			if (!eq) {
				cellFailed = true;
				failed++;
			} else {
				passed++;
			}
		}

		if (cellFailed) {
			failuresByCell[cellKey] = true;
		}
	}
}

// ============================================================================
// Validation guards (cubic review f70a81ef P1 / f44ead49 P0+P1 /
// 5a3b44c9 P2)
// ----------------------------------------------------------------------------
// n_coeff == 0 with len > 0 must be rejected (the kernel indexes coeff by
// w % n_coeff — a divide-by-zero on the scalar path); counts that overflow
// the byte-size comparison must be rejected by the overflow-safe checks,
// not wrap past them. The overflow cases use 2^61: in the naive
// (size_t)len * 8 form, 2^61 * 8 = 2^64 wraps to 0 and the check passes
// (then the kernel reads out of bounds) — 2^60 * 8 = 2^63 still fits
// size_t and is rejected by BOTH the naive and the fixed check, so it
// would not pin the fix (cubic review 5a3b44c9 P2). Both the standalone
// export and the class method are pinned. The len=0/n_coeff=0 no-op
// stays legal.
// ============================================================================

function rejects(fn, what) {
	var threw = false;
	try {
		fn();
	} catch (e) {
		threw = (e instanceof RangeError) || (e instanceof TypeError);
	}
	if (!threw) {
		failed++;
		/* Guard failures must set the exit code directly: they are not
		 * parity cells, so recording them in failuresByCell would
		 * inflate the cell count in the report (cubic review 50f46d24
		 * P3) and — worse — a guard failure with all cells passing
		 * would otherwise exit 0. */
		process.exitCode = 1;
		console.error('  FAIL (RED): ' + what + ' was not rejected');
	} else {
		passed++;
		console.log('  PASS: ' + what + ' rejected');
	}
}

var gOut = Buffer.alloc(64);
var gIn  = Buffer.alloc(64);
var gCo  = Buffer.alloc(64);

// Standalone export (binding.mul_arr).
rejects(function () { addon.mul_arr(gOut, gIn, gCo, 8, 0); },
        'standalone: len>0, n_coeff=0 (div-by-zero guard)');
rejects(function () { addon.mul_arr(gOut, gIn, gCo, Math.pow(2, 61), 1); },
        'standalone: len=2^60 vs 64-byte buffers (overflow-safe check)');
rejects(function () { addon.mul_arr(gOut, gIn, gCo, 1, Math.pow(2, 61)); },
        'standalone: n_coeff=2^60 vs 64-byte coeff (overflow-safe check)');
// Class method (encoder.mul_arr).
rejects(function () { encoder.mul_arr(gOut, gIn, gCo, 8, 0); },
        'class: len>0, n_coeff=0 (div-by-zero guard)');
rejects(function () { encoder.mul_arr(gOut, gIn, gCo, Math.pow(2, 61), 1); },
        'class: len=2^60 vs 64-byte buffers (overflow-safe check)');
rejects(function () { encoder.mul_arr(gOut, gIn, gCo, 1, Math.pow(2, 61)); },
        'class: n_coeff=2^60 vs 64-byte coeff (overflow-safe check)');
// Zero-length no-op preserved on both entries.
addon.mul_arr(gOut, gIn, gCo, 0, 0);
encoder.mul_arr(gOut, gIn, gCo, 0, 0);
passed++;
console.log('  PASS: len=0, n_coeff=0 no-op preserved (both entries)');

// ============================================================================
// Report
// ============================================================================

var totalCells = N_COEFFS.length * LENS.length;
var failedCells = Object.keys(failuresByCell).length;

for (var ci = 0; ci < N_COEFFS.length; ci++) {
	var nCoeff = N_COEFFS[ci];
	for (var li = 0; li < LENS.length; li++) {
		var len = LENS[li];
		var cellKey = 'n_coeff=' + nCoeff + ', len=' + len;
		if (failuresByCell[cellKey]) {
			console.error('  FAIL (RED): ' + cellKey);
		} else {
			console.log('  PASS: ' + cellKey);
		}
	}
}

console.log('\n---');
console.log('Total cells:  ' + totalCells + ' (10 trials each)');
console.log('Cell status:  ' + (totalCells - failedCells) + ' passed, ' + failedCells + ' failed (RED)');

if (failedCells > 0) {
	process.exitCode = 1;
}
