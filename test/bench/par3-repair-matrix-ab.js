#!/usr/bin/env node
// A/B: the B2 repair-coefficient work, isolated.
// Before: per-element JS BigInt Euclid inversion — the repair path computed
//   n x N_total coefficients (A submatrix + RHS) with invert64() per element.
// After: one native build_coefficient_matrix(N_total, n, ...) call.
// This measures exactly the replaced work (no RHS mul/XOR loop).
"use strict";
var addon = require('../../build/Release/parpar_gf64.node');

var GF64_MASK = 0xFFFFFFFFFFFFFFFFn;
var GF64_POLY = 0x1000000000000001Bn;
function invert64(val) {
	val = val & GF64_MASK;
	if (val === 0n) return 0n;
	if (val === 1n) return 1n;
	var u = val, v = GF64_POLY, x1 = 1n, x2 = 0n;
	while (u !== 1n && u !== 0n) {
		while ((u & 1n) === 0n) {
			u >>= 1n;
			if ((x1 & 1n) !== 0n) x1 = ((x1 ^ GF64_POLY) >> 1n) & GF64_MASK;
			else x1 >>= 1n;
		}
		if (u === 1n) continue;
		while ((v & 1n) === 0n) v >>= 1n;
		if (u < v) { var t = u; u = v; v = t; t = x1; x1 = x2; x2 = t; }
		u ^= v;
		x1 ^= x2;
	}
	return x1 & GF64_MASK;
}

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
console.log('sink check (identical): ' + (sink === sink2));
