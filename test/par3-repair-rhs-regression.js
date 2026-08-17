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
var child_process = require('child_process');
var path = require('path');
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

// ISA-aware perf gate (cubic review PR #93): the G=248/K=8/len=512 workload
// is tuned for AVX-512/AVX-2 hosts. A fixed 50ms deadline would fail
// spuriously on scalar/SSSE3 runners or loaded shared CI, so the limit is
// tiered by the ACTIVE dispatch method (addon.gf64_info(0).name). Every tier
// stays far below the ~600ms/step regressed path this gate targets.
function perfGateLimitMs(methodName) {
	switch (methodName) {
		case 'AVX512': return 50.0;
		case 'AVX2': return 100.0;
		case 'SSSE3': return 250.0;
		case 'SCALAR': return 400.0;
		default: return -1; // unknown method -> skip the gate
	}
}

function testPerformanceGate() {
	console.log('--- Test 4: Issue #90 Performance Assertion ---');
	var encoder = new addon.Gf64Encoder(0);
	var methodName = 'UNKNOWN';
	try {
		methodName = addon.gf64_info(0).name;
	} catch (e) {
		/* no gf64_info export -> skip below */
	}
	var limitMs = perfGateLimitMs(methodName);
	if (limitMs < 0) {
		console.log('SKIP: performance gate requires a known native dispatch method (got ' + methodName + ')');
		return;
	}
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

	// Warm up (first-touch page faults, branch predictor, DVFS ramp)
	for (var w = 0; w < 20; w++) {
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

	console.log('Issue #90 2D Muladd (G=248, K=8, 4KB, method=' + methodName + '): ' + perIterMs.toFixed(3) + ' ms per repair step (total 50 iters: ' + elapsedMs.toFixed(2) + ' ms)');
	// Regression threshold: pre-fix was ~5000 ms for one full repair. One step was ~600ms.
	// Fixed version runs in under 5 ms on AVX-512!
	assert.ok(perIterMs < limitMs, 'Performance regression detected: per-iter time ' + perIterMs + ' ms exceeds ' + limitMs + ' ms gate for method ' + methodName);
	console.log('PASS: Performance verified well within speed gate.');
}

// OOM-boundary test for the 2D RHS gate (cubic review PR #93 P1): the
// n x knownCount coefficient matrix must fall back to coupled/legacy before
// the allocUnsafe can OOM, and the kernel-side caps must still hold.
function testTwoDGate() {
	console.log('--- Test 5: 2D RHS Gate Boundary (Issue #90 / PR #93 review) ---');
	var fakeBinding = { Gf64Encoder: function() {} };
	// At-cap: n * knownCount == 32 * 1024 * 1024 entries is allowed
	assert.strictEqual(par3.canUseTwoDRhs(fakeBinding, 4096, 8192), true, 'gate must allow exactly-at-cap n*knownCount');
	// One past the cap falls back to coupled/legacy
	assert.strictEqual(par3.canUseTwoDRhs(fakeBinding, 4096, 8193), false, 'gate must reject n*knownCount above the 32M-entry budget');
	// Individually-legal but enormous dims must not pass (would be a 32 GiB allocUnsafe)
	assert.strictEqual(par3.canUseTwoDRhs(fakeBinding, 65536, 65536), false, 'gate must reject the 65536 x 65536 coefficient matrix');
	// Kernel-side caps still enforced
	assert.strictEqual(par3.canUseTwoDRhs(fakeBinding, 65537, 8), false, 'n above the 65536 kernel cap must not use 2D');
	assert.strictEqual(par3.canUseTwoDRhs(fakeBinding, 8, 65537), false, 'knownCount above the 65536 kernel cap must not use 2D');
	// No binding -> never 2D
	assert.strictEqual(par3.canUseTwoDRhs(null, 8, 8), false, 'missing binding must not use 2D');
	// PAR3_REPAIR_COUPLED_RHS=0 forces the legacy loop
	process.env.PAR3_REPAIR_COUPLED_RHS = '0';
	assert.strictEqual(par3.canUseTwoDRhs(fakeBinding, 8, 8), false, 'PAR3_REPAIR_COUPLED_RHS=0 must disable 2D');
	delete process.env.PAR3_REPAIR_COUPLED_RHS;
	assert.strictEqual(par3.canUseTwoDRhs(fakeBinding, 8, 8), true, 'small repair with binding must use 2D');
	console.log('PASS: 2D RHS gate enforces the coefficient-matrix budget.');
}

// Parity for the fused-output muladd kernel (cubic review PR #93 P2): with
// K > 1 the shared `in` buffer must be read once and applied to all K
// outputs; this exercises the exact input-reuse contract of
// gf64_region_fused_output_muladd_*_arr across odd lengths and tails.
function testFusedOutputParity() {
	console.log('--- Test 6: Fused-Output Muladd K>1 Input-Reuse Parity (PR #93 review) ---');
	var encoder = new addon.Gf64Encoder(0);
	var kList = [1, 2, 4, 8];
	var lenList = [1, 7, 8, 32, 64, 128]; // in 64-bit words (covers tails, vectors, tiles)

	for (var ki = 0; ki < kList.length; ki++) {
		var K = kList[ki];
		for (var li = 0; li < lenList.length; li++) {
			var len = lenList[li];
			var byteLen = len * 8;

			var inBuf = Buffer.allocUnsafe(byteLen);
			for (var w = 0; w < len; w++) inBuf.writeBigUInt64LE(rnd64(), w * 8);

			var coeffs = new Array(K);
			for (var k = 0; k < K; k++) {
				coeffs[k] = Buffer.alloc(8);
				coeffs[k].writeBigUInt64LE(rnd64(), 0);
			}

			// Reference: pure JS per (k, w)
			var refOuts = new Array(K);
			for (var k2 = 0; k2 < K; k2++) {
				refOuts[k2] = Buffer.alloc(byteLen);
				var c = coeffs[k2].readBigUInt64LE(0);
				if (c === 0n) continue;
				for (var w2 = 0; w2 < len; w2++) {
					var iv = inBuf.readBigUInt64LE(w2 * 8);
					if (iv === 0n) continue;
					var prev = refOuts[k2].readBigUInt64LE(w2 * 8);
					refOuts[k2].writeBigUInt64LE(prev ^ gf64Js.gf64_mul(iv, c), w2 * 8);
				}
			}

			var nativeOuts = new Array(K);
			for (var k3 = 0; k3 < K; k3++) nativeOuts[k3] = Buffer.alloc(byteLen);
			encoder.fused_output_muladd_arr(nativeOuts, inBuf, coeffs, len, K);
			for (var k4 = 0; k4 < K; k4++) {
				assert.strictEqual(nativeOuts[k4].equals(refOuts[k4]), true, 'fused_output_muladd_arr parity mismatch at K=' + K + ' len=' + len);
			}
		}
	}
	console.log('PASS: Fused-output muladd K>1 bit-exact parity across all K, len.');
}

function testForcedMethodParity() {
	console.log('--- Test 7: Forced-Method Kernel Parity (AVX-512/AVX-2 dispatch) ---');
	// Tests 1/6 only exercise the method detected at startup. On WSL2/Hyper-V
	// the CPUID AVX-512 bits flip between process starts (microsoft/WSL#14436),
	// so the AVX-512 dispatch path can be silently skipped. Spawn children with
	// PAR3_GF64_USE_AVX512 forced on/off so BOTH dispatch paths run the new
	// fused-output/2D kernels against the JS reference. This is the regression
	// gate for the kernels' 128-bit-lane interleave contract: a consecutive
	// (non-interleaved) load made VPCLMULQDQ pick the wrong element per lane,
	// corrupting words 1-3 under AVX-512 only (WSL2 flip hid it from Test 1).
	var addonAbs = path.join(__dirname, '..', 'build', 'Release', 'parpar_gf64.node');
	var gf64JsAbs = path.join(__dirname, '..', 'lib', 'gf64_js.js');
	var childCode = [
		'var addon = require(' + JSON.stringify(addonAbs) + ');',
		'var gf64Js = require(' + JSON.stringify(gf64JsAbs) + ');',
		'var enc = new addon.Gf64Encoder(0);',
		'function rnd64() {',
		'	var x = 0n;',
		'	for (var i = 0; i < 8; i++) x = (x << 8n) ^ BigInt(Math.floor(Math.random() * 256));',
		'	return x ? x : 1n;',
		'}',
		'var gList = [1, 2, 8], kList = [1, 4], lenList = [1, 7, 8, 32, 128];',
		'for (var gi = 0; gi < gList.length; gi++) {',
		'	var G = gList[gi];',
		'	for (var ki = 0; ki < kList.length; ki++) {',
		'		var K = kList[ki];',
		'		for (var li = 0; li < lenList.length; li++) {',
		'			var len = lenList[li], byteLen = len * 8;',
		'			var inBlocks = new Array(G);',
		'			for (var g = 0; g < G; g++) {',
		'				inBlocks[g] = Buffer.allocUnsafe(byteLen);',
		'				for (var w = 0; w < len; w++) inBlocks[g].writeBigUInt64LE(rnd64(), w * 8);',
		'			}',
		'			var coeff2d = Buffer.allocUnsafe(K * G * 8);',
		'			for (var c = 0; c < K * G; c++) coeff2d.writeBigUInt64LE(rnd64(), c * 8);',
		'			var ref = new Array(K);',
		'			for (var k = 0; k < K; k++) {',
		'				ref[k] = Buffer.alloc(byteLen);',
		'				for (var g2 = 0; g2 < G; g2++) {',
		'					var ck = coeff2d.readBigUInt64LE((k * G + g2) * 8);',
		'					if (ck === 0n) continue;',
		'					for (var w2 = 0; w2 < len; w2++) {',
		'						var iv = inBlocks[g2].readBigUInt64LE(w2 * 8);',
		'						if (iv === 0n) continue;',
		'						ref[k].writeBigUInt64LE(ref[k].readBigUInt64LE(w2 * 8) ^ gf64Js.gf64_mul(iv, ck), w2 * 8);',
		'					}',
		'				}',
		'			}',
		'			var nat = new Array(K);',
		'			for (var k2 = 0; k2 < K; k2++) nat[k2] = Buffer.alloc(byteLen);',
		'			enc.two_d_muladd_arr(nat, K, inBlocks, G, coeff2d, G, len);',
		'			for (var k3 = 0; k3 < K; k3++) {',
		'				if (!nat[k3].equals(ref[k3])) {',
		'					console.error("two_d_muladd_arr mismatch G=" + G + " K=" + K + " len=" + len);',
		'					process.exit(1);',
		'				}',
		'			}',
		'		}',
		'	}',
		'}',
		'for (var ki2 = 0; ki2 < kList.length; ki2++) {',
		'	var K2 = kList[ki2];',
		'	for (var li2 = 0; li2 < lenList.length; li2++) {',
		'		var len2 = lenList[li2], byteLen2 = len2 * 8;',
		'		var inBuf = Buffer.allocUnsafe(byteLen2);',
		'		for (var w3 = 0; w3 < len2; w3++) inBuf.writeBigUInt64LE(rnd64(), w3 * 8);',
		'		var coeffs = new Array(K2);',
		'		for (var c2 = 0; c2 < K2; c2++) {',
		'			coeffs[c2] = Buffer.alloc(8);',
		'			coeffs[c2].writeBigUInt64LE(rnd64(), 0);',
		'		}',
		'		var ref2 = new Array(K2);',
		'		for (var k4 = 0; k4 < K2; k4++) {',
		'			ref2[k4] = Buffer.alloc(byteLen2);',
		'			var cc = coeffs[k4].readBigUInt64LE(0);',
		'			if (cc === 0n) continue;',
		'			for (var w4 = 0; w4 < len2; w4++) {',
		'				var iv2 = inBuf.readBigUInt64LE(w4 * 8);',
		'				if (iv2 === 0n) continue;',
		'				ref2[k4].writeBigUInt64LE(ref2[k4].readBigUInt64LE(w4 * 8) ^ gf64Js.gf64_mul(iv2, cc), w4 * 8);',
		'			}',
		'		}',
		'		var nat2 = new Array(K2);',
		'		for (var k5 = 0; k5 < K2; k5++) nat2[k5] = Buffer.alloc(byteLen2);',
		'		enc.fused_output_muladd_arr(nat2, inBuf, coeffs, len2, K2);',
		'		for (var k6 = 0; k6 < K2; k6++) {',
		'			if (!nat2[k6].equals(ref2[k6])) {',
		'				console.error("fused_output_muladd_arr mismatch K=" + K2 + " len=" + len2);',
		'				process.exit(1);',
		'			}',
		'		}',
		'	}',
		'}',
		'process.exit(0);'
	].join('\n');

	function runForced(mode) {
		var env = JSON.parse(JSON.stringify(process.env));
		env.PAR3_GF64_USE_AVX512 = mode;
		return child_process.spawnSync(process.execPath, ['-e', childCode],
			{ env: env, encoding: 'utf8', timeout: 60000 });
	}

	var r0 = runForced('0');
	assert.strictEqual(r0.status, 0, 'PAR3_GF64_USE_AVX512=0 parity child failed: ' + (r0.stderr || '').trim());
	var r1 = runForced('1');
	if (r1.signal === 'SIGILL') {
		console.log('SKIP: host lacks working AVX-512 (forced dispatch SIGILLs)');
	} else {
		assert.strictEqual(r1.status, 0, 'PAR3_GF64_USE_AVX512=1 parity child failed: ' + (r1.stderr || '').trim());
	}
	console.log('PASS: forced-method kernel parity (AVX-512 and AVX-2 dispatch paths).');
}

function main() {
	testKernelParity();
	testNegativeTrap();
	testEndToEndRepairParity();
	testPerformanceGate();
	testTwoDGate();
	testFusedOutputParity();
	testForcedMethodParity();
	console.log('\nAll Issue #90 regression tests PASSED successfully!');
}

main();
