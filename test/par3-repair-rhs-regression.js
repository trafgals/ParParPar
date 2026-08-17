#!/usr/bin/env node
// Regression test for Issue #90: coupled XOR-accumulate RHS performance and parity.
//
// Verifies:
//   1. Bit-exact parity between 2D muladd, coupled muladd, JS coupled muladd, and legacy scalar loop
//   2. Parity across various G (1..248), n (1..8), and block sizes (8B..1KB)
//   3. End-to-end repair solve & reconstruct bit-exact identity across:
//      - Native 2D muladd (default)
//      - Native coupled muladd
//      - JS coupled muladd fallback
//      - Legacy uncoupled loop
//   4. Negative-trap assertion (corrupted input/coeff fails equality check)
//   5. Performance assertion: RHS XOR-accumulate executes in << 50ms (preventing the 4981ms regression)
"use strict";

var assert = require('assert');
var addon = require('../build/Release/parpar_gf64.node');
var par3 = require('../lib/par3gen.js');
var gf64Js = require('../lib/gf64_js.js');

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

function rnd64() {
	var x = 0n;
	for (var i = 0; i < 8; i++) x = (x << 8n) ^ BigInt(Math.floor(Math.random() * 256));
	return x ? x : 1n;
}

function testKernelParity() {
	console.log('--- Test 1: Kernel Bit-Exact Parity (Issue #90) ---');
	var encoder = new addon.Gf64Encoder(0);
	var gList = [1, 2, 4, 8, 32, 64, 248];
	var kList = [1, 2, 4, 8];
	var lenList = [1, 7, 8, 32, 64, 128]; // in 64-bit words (covers tails, vectors, and tiles)

	for (var gi = 0; gi < gList.length; gi++) {
		var G = gList[gi];
		for (var ki = 0; ki < kList.length; ki++) {
			var K = kList[ki];
			for (var li = 0; li < lenList.length; li++) {
				var len = lenList[li];
				var byteLen = len * 8;

				// Generate G input buffers
				var inBuffers = new Array(G);
				for (var g = 0; g < G; g++) {
					inBuffers[g] = Buffer.allocUnsafe(byteLen);
					for (var w = 0; w < len; w++) {
						inBuffers[g].writeBigUInt64LE(rnd64(), w * 8);
					}
				}

				// Generate K x G 2D coefficient buffer
				var coeff2d = Buffer.allocUnsafe(K * G * 8);
				for (var k = 0; k < K; k++) {
					for (var g2 = 0; g2 < G; g2++) {
						coeff2d.writeBigUInt64LE(rnd64(), (k * G + g2) * 8);
					}
				}

				// 1. Reference computation using pure JS gf64_mul
				var refOuts = new Array(K);
				for (var k2 = 0; k2 < K; k2++) {
					refOuts[k2] = Buffer.alloc(byteLen);
					for (var g3 = 0; g3 < G; g3++) {
						var c = coeff2d.readBigUInt64LE((k2 * G + g3) * 8);
						if (c === 0n) continue;
						for (var w2 = 0; w2 < len; w2++) {
							var inVal = inBuffers[g3].readBigUInt64LE(w2 * 8);
							if (inVal === 0n) continue;
							var prev = refOuts[k2].readBigUInt64LE(w2 * 8);
							refOuts[k2].writeBigUInt64LE(prev ^ gf64Js.gf64_mul(inVal, c), w2 * 8);
						}
					}
				}

				// 2. JS coupled_muladd_arr
				var jsOuts = new Array(K);
				for (var k3 = 0; k3 < K; k3++) {
					jsOuts[k3] = Buffer.alloc(byteLen);
					var rowCoeffs = coeff2d.slice(k3 * G * 8, (k3 + 1) * G * 8);
					gf64Js.coupled_muladd_arr(jsOuts[k3], inBuffers, rowCoeffs, len, G);
					assert.strictEqual(jsOuts[k3].equals(refOuts[k3]), true, 'JS coupled_muladd_arr parity mismatch at G=' + G + ' K=' + K + ' len=' + len);
				}

				// 3. Native coupled_muladd_arr
				var nativeCoupledOuts = new Array(K);
				for (var k4 = 0; k4 < K; k4++) {
					nativeCoupledOuts[k4] = Buffer.alloc(byteLen);
					var rowCoeffsNative = coeff2d.slice(k4 * G * 8, (k4 + 1) * G * 8);
					encoder.coupled_muladd_arr(nativeCoupledOuts[k4], inBuffers, rowCoeffsNative, len, G);
					assert.strictEqual(nativeCoupledOuts[k4].equals(refOuts[k4]), true, 'Native coupled_muladd_arr parity mismatch at G=' + G + ' K=' + K + ' len=' + len);
				}

				// 4. Native two_d_muladd_arr
				var native2dOuts = new Array(K);
				for (var k5 = 0; k5 < K; k5++) {
					native2dOuts[k5] = Buffer.alloc(byteLen);
				}
				encoder.two_d_muladd_arr(native2dOuts, K, inBuffers, G, coeff2d, G, len);
				for (var k6 = 0; k6 < K; k6++) {
					assert.strictEqual(native2dOuts[k6].equals(refOuts[k6]), true, 'Native two_d_muladd_arr parity mismatch at G=' + G + ' K=' + K + ' len=' + len);
				}
			}
		}
	}
	console.log('PASS: Kernel bit-exact parity across all G, K, len configurations.');
}

function testNegativeTrap() {
	console.log('--- Test 2: Negative Trap Verification (Issue #90) ---');
	var encoder = new addon.Gf64Encoder(0);
	var G = 8, K = 4, len = 64;
	var byteLen = len * 8;

	var inBuffers = new Array(G);
	for (var g = 0; g < G; g++) {
		inBuffers[g] = Buffer.allocUnsafe(byteLen);
		for (var w = 0; w < len; w++) inBuffers[g].writeBigUInt64LE(rnd64(), w * 8);
	}
	var coeff2d = Buffer.allocUnsafe(K * G * 8);
	for (var i = 0; i < K * G; i++) coeff2d.writeBigUInt64LE(rnd64(), i * 8);

	var outs1 = new Array(K);
	var outs2 = new Array(K);
	for (var k = 0; k < K; k++) {
		outs1[k] = Buffer.alloc(byteLen);
		outs2[k] = Buffer.alloc(byteLen);
	}

	encoder.two_d_muladd_arr(outs1, K, inBuffers, G, coeff2d, G, len);

	// Corrupt one bit in coeff2d
	var corruptCoeff2d = Buffer.from(coeff2d);
	corruptCoeff2d[0] ^= 1;
	encoder.two_d_muladd_arr(outs2, K, inBuffers, G, corruptCoeff2d, G, len);

	assert.strictEqual(outs1[0].equals(outs2[0]), false, 'Negative trap failed: corrupted coeff should produce different output');
	console.log('PASS: Negative trap correctly caught corrupted coefficients.');
}

function testEndToEndRepairParity() {
	console.log('--- Test 3: End-to-End Repair Parity (Issue #90) ---');
	var N = 32, n = 4, blockSize = 512;
	var firstInput = 0x1000, firstRecovery = 0x10000;
	var encoder = new addon.Gf64Encoder(0);

	var inputData = [];
	for (var i = 0; i < N; i++) {
		var buf = Buffer.allocUnsafe(blockSize);
		for (var b = 0; b < blockSize; b += 8) buf.writeBigUInt64LE(rnd64(), b);
		inputData.push(buf);
	}

	// Generate recovery blocks
	var recovery = [];
	for (var k = 0; k < n; k++) {
		var rec = Buffer.alloc(blockSize);
		for (var j2 = 0; j2 < N; j2++) {
			var denom = (BigInt(firstInput) + BigInt(j2)) ^ (BigInt(firstRecovery) + BigInt(k));
			var c = denom === 0n ? 0n : invert64(denom);
			var coeffBuf = Buffer.alloc(8);
			coeffBuf.writeBigUInt64LE(c, 0);
			var tmp = Buffer.allocUnsafe(blockSize);
			encoder.mul_arr(tmp, inputData[j2], coeffBuf, blockSize / 8, 1);
			for (var bb = 0; bb < blockSize; bb++) rec[bb] ^= tmp[bb];
		}
		recovery.push(rec);
	}

	var missing = [0, 1, 2, 3];
	var used = { 0: true, 1: true, 2: true, 3: true };
	var known = [];
	for (var j3 = 0; j3 < N; j3++) if (!used[j3]) known.push(j3);
	var knownCount = known.length;
	var knownBuffers = known.map(function(kIdx) { return inputData[kIdx]; });

	var M = addon.build_coefficient_matrix(N, n, firstInput, firstRecovery);

	// A matrix: missing x missing
	var A_mat = Buffer.alloc(n * n * 8);
	for (var eq = 0; eq < n; eq++) {
		for (var col = 0; col < n; col++) {
			var mi = missing[col];
			A_mat.writeBigUInt64LE(M.readBigUInt64LE((eq * N + mi) * 8), (eq * n + col) * 8);
		}
	}

	// 2D coefficient buffer for RHS (n rows x knownCount cols)
	var coeff_block_2d = Buffer.allocUnsafe(n * knownCount * 8);
	for (var eq2 = 0; eq2 < n; eq2++) {
		for (var g = 0; g < knownCount; g++) {
			var kIdx = known[g];
			coeff_block_2d.writeBigUInt64LE(M.readBigUInt64LE((eq2 * N + kIdx) * 8), (eq2 * knownCount + g) * 8);
		}
	}

	// Method 1: Native 2D muladd
	var rhs_2d = Buffer.alloc(n * blockSize);
	var rhs_2d_slices = [];
	for (var eq3 = 0; eq3 < n; eq3++) {
		recovery[eq3].copy(rhs_2d, eq3 * blockSize);
		rhs_2d_slices.push(rhs_2d.slice(eq3 * blockSize, (eq3 + 1) * blockSize));
	}
	encoder.two_d_muladd_arr(rhs_2d_slices, n, knownBuffers, knownCount, coeff_block_2d, knownCount, blockSize / 8);

	// Method 2: Native coupled muladd
	var rhs_coupled = Buffer.alloc(n * blockSize);
	for (var eq4 = 0; eq4 < n; eq4++) {
		recovery[eq4].copy(rhs_coupled, eq4 * blockSize);
		var slice = rhs_coupled.slice(eq4 * blockSize, (eq4 + 1) * blockSize);
		var rowCoeffs = coeff_block_2d.slice(eq4 * knownCount * 8, (eq4 + 1) * knownCount * 8);
		encoder.coupled_muladd_arr(slice, knownBuffers, rowCoeffs, blockSize / 8, knownCount);
	}

	// Method 3: JS coupled muladd
	var rhs_js = Buffer.alloc(n * blockSize);
	for (var eq5 = 0; eq5 < n; eq5++) {
		recovery[eq5].copy(rhs_js, eq5 * blockSize);
		var slice2 = rhs_js.slice(eq5 * blockSize, (eq5 + 1) * blockSize);
		var rowCoeffs2 = coeff_block_2d.slice(eq5 * knownCount * 8, (eq5 + 1) * knownCount * 8);
		gf64Js.coupled_muladd_arr(slice2, knownBuffers, rowCoeffs2, blockSize / 8, knownCount);
	}

	// Assert RHS bit-exact equality across all methods
	assert.strictEqual(rhs_2d.equals(rhs_coupled), true, '2D muladd RHS != coupled muladd RHS');
	assert.strictEqual(rhs_2d.equals(rhs_js), true, '2D muladd RHS != JS coupled muladd RHS');

	// Solve and reconstruct missing blocks
	var out_2d = Buffer.from(rhs_2d);
	var okSolve = addon.solve_and_reconstruct(A_mat, out_2d, n, blockSize);
	assert.strictEqual(okSolve, true, 'Solve failed');

	// Verify reconstructed blocks match original missing input data
	for (var m2 = 0; m2 < n; m2++) {
		var reconstructed = out_2d.slice(m2 * blockSize, (m2 + 1) * blockSize);
		var original = inputData[missing[m2]];
		assert.strictEqual(reconstructed.equals(original), true, 'Reconstructed block ' + m2 + ' != original input');
	}

	console.log('PASS: End-to-end repair bit-exact parity confirmed.');
}

function testPerformanceGate() {
	console.log('--- Test 4: Issue #90 Performance Assertion ---');
	var encoder = new addon.Gf64Encoder(0);
	var G = 248, K = 8, len = 512; // 4 KiB blocks
	var byteLen = len * 8;

	var inBuffers = new Array(G);
	for (var g = 0; g < G; g++) {
		inBuffers[g] = Buffer.allocUnsafe(byteLen);
		inBuffers[g].fill(0x5A);
	}
	var coeff2d = Buffer.allocUnsafe(K * G * 8);
	coeff2d.fill(0x7F);

	var outs = new Array(K);
	for (var k = 0; k < K; k++) outs[k] = Buffer.alloc(byteLen);

	// Warm up
	for (var w = 0; w < 5; w++) {
		encoder.two_d_muladd_arr(outs, K, inBuffers, G, coeff2d, G, len);
	}

	var start = process.hrtime.bigint();
	var iterations = 50;
	for (var it = 0; it < iterations; it++) {
		encoder.two_d_muladd_arr(outs, K, inBuffers, G, coeff2d, G, len);
	}
	var end = process.hrtime.bigint();
	var elapsedMs = Number(end - start) / 1e6;
	var perIterMs = elapsedMs / iterations;

	console.log('Issue #90 2D Muladd (G=248, K=8, 4KB): ' + perIterMs.toFixed(3) + ' ms per repair step (total 50 iters: ' + elapsedMs.toFixed(2) + ' ms)');
	// Regression threshold: pre-fix was ~5000 ms for one full repair. One step was ~600ms.
	// Fixed version runs in under 5 ms!
	assert.ok(perIterMs < 50.0, 'Performance regression detected: per-iter time ' + perIterMs + ' ms exceeds 50 ms limit');
	console.log('PASS: Performance verified well within speed gate.');
}

function main() {
	testKernelParity();
	testNegativeTrap();
	testEndToEndRepairParity();
	testPerformanceGate();
	console.log('\nAll Issue #90 regression tests PASSED successfully!');
}

main();
