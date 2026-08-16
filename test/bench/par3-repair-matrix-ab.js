#!/usr/bin/env node
// A/B: the B2 repair-coefficient work, isolated.
// Before: per-element JS BigInt Euclid inversion — the repair path computed
//   n x N_total coefficients (A submatrix + RHS) with invert64() per element.
// After: one native build_coefficient_matrix(N_total, n, ...) call.
// This measures exactly the replaced work (no RHS mul/XOR loop).
"use strict";
var assert = require('assert');
var addon = require('../../build/Release/parpar_gf64.node');
// Reuse the shipped JS implementation (the repair fallback's own library) so
// the JS leg measures the real code instead of a drifting copy.
var gf64js = require('../../lib/gf64_js.js');
var invert64 = gf64js.invert64;

var N = 10000, n = 1000;              // 1G/10K repair geometry (10% missing)
var firstInput = 0x1000, firstRecovery = 0x100000;
var total = n * N;                     // 10M coefficients

// Warmup
for (var w = 0; w < 10000; w++) invert64(BigInt(w) ^ 0x123456789ABCDEFn);
addon.build_coefficient_matrix(64, 8, firstInput, firstRecovery);

// JS leg: the exact per-element work the repair path used to do
var t0 = Date.now();
var sink = 0n;
for (var eq = 0; eq < n; eq++) {
	var y = BigInt(firstRecovery + eq);
	for (var k = 0; k < N; k++) {
		var d = BigInt(firstInput + k) ^ y;
		sink ^= invert64(d === 0n ? 1n : d);
	}
}
var tJs = Date.now() - t0;

// Native leg: one build_coefficient_matrix call for the same (n x N) range
var t1 = Date.now();
var M = addon.build_coefficient_matrix(N, n, firstInput, firstRecovery);
var tNat = Date.now() - t1;
var sink2 = 0n;
for (var i = 0; i < total; i++) sink2 ^= M.readBigUInt64LE(i * 8);

console.log('JS  invert64 x ' + total + ': ' + tJs + ' ms  (' + (total / tJs / 1000).toFixed(1) + ' M/s)');
console.log('native matrix  (' + n + 'x' + N + '): ' + tNat + ' ms  (' + (total / tNat / 1000).toFixed(1) + ' M/s)');
console.log('speedup: ' + (tJs / Math.max(tNat, 1)).toFixed(1) + 'x');
// The bit-identical claim is the bench's whole point — assert it.
assert.strictEqual(sink, sink2, 'native matrix and JS invert64 diverged');
console.log('sink check (identical): true');
