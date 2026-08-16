#!/usr/bin/env node
// Parity: the B2 native repair-coefficient path vs the JS BigInt fallback.
//
// The repair path (lib/par3gen.js) now builds its Cauchy coefficients with
// one native build_coefficient_matrix call (n missing rows x N_total cols)
// when the binding is available and the footprint fits, reading both the
// A submatrix (missing x missing) and the RHS coefficients (known x missing)
// from the same buffer. PAR3_REPAIR_NATIVE_MATRIX=0 forces the JS fallback.
//
// This test pins:
//   1. A-matrix coefficients: native subset == JS BigInt invert64, bit-exact
//   2. RHS coefficients (mul-accumulated): native == JS, bit-exact
//   3. End-to-end par3gen.repair outputs: default (native) == forced-JS
//      fallback, byte-identical repaired files
"use strict";
var assert = require('assert');
var path = require('path');
var fs = require('fs');
var os = require('os');

var addon = require('../build/Release/parpar_gf64.node');
var par3 = require('../lib/par3gen.js');

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

function mulAdd(encoder, acc, data, coeff, blockSize) {
	var tmp = Buffer.allocUnsafe(blockSize);
	var coeffBuf = Buffer.alloc(8);
	coeffBuf.writeBigUInt64LE(coeff, 0);
	encoder.mul_arr(tmp, data, coeffBuf, blockSize / 8, 1);
	for (var b = 0; b < blockSize; b++) acc[b] ^= tmp[b];
}

function rnd64() {
	var x = 0n;
	for (var i = 0; i < 8; i++) x = (x << 8n) ^ BigInt(Math.floor(Math.random() * 256));
	return x ? x : 1n;
}

// ---------------------------------------------------------------------------
// Leg 1: coefficient parity (A matrix + RHS) at the repair geometry
// ---------------------------------------------------------------------------
function leg1() {
	var N = 64, n = 8, blockSize = 512;
	var firstInput = 0x1000, firstRecovery = 0x10000; // disjoint ranges
	var encoder = new addon.Gf64Encoder(0);

	// missing blocks (distinct, sorted)
	var missing = [];
	var used = {};
	while (missing.length < n) {
		var idx = Math.floor(Math.random() * N);
		if (!used[idx]) { used[idx] = true; missing.push(idx); }
	}
	missing.sort(function(a, b) { return a - b; });

	// input data + recovery blocks (Cauchy encoding, JS coeffs — same as the
	// test-only reference in par3-repair-parity.js)
	var inputData = {};
	for (var j = 0; j < N; j++) {
		inputData[j] = Buffer.alloc(blockSize);
		for (var b = 0; b < blockSize; b++) inputData[j][b] = Math.floor(Math.random() * 256);
	}
	var recovery = [];
	for (var k = 0; k < n; k++) {
		var rec = Buffer.alloc(blockSize);
		for (var j2 = 0; j2 < N; j2++) {
			var denom = (BigInt(firstInput) + BigInt(j2)) ^ (BigInt(firstRecovery) + BigInt(k));
			mulAdd(encoder, rec, inputData[j2], denom === 0n ? 0n : invert64(denom), blockSize);
		}
		recovery.push(rec);
	}

	// Native matrix (the B2 repair path's source)
	var M = addon.build_coefficient_matrix(N, n, firstInput, firstRecovery);

	// A matrix: native subset vs JS invert64
	var A_nat = Buffer.alloc(n * n * 8);
	var A_js = Buffer.alloc(n * n * 8);
	for (var eq = 0; eq < n; eq++) {
		for (var col = 0; col < n; col++) {
			var mi = missing[col];
			A_nat.writeBigUInt64LE(M.readBigUInt64LE((eq * N + mi) * 8), eq * n * 8 + col * 8);
			var d2 = (BigInt(firstRecovery) + BigInt(eq)) ^ (BigInt(firstInput) + BigInt(mi));
			A_js.writeBigUInt64LE(invert64(d2 === 0n ? 1n : d2), eq * n * 8 + col * 8);
		}
	}
	assert(A_nat.equals(A_js), 'A matrix: native subset != JS invert64');

	// RHS: recovery XOR known-input contributions, native coeffs vs JS coeffs
	var known = [];
	for (var j3 = 0; j3 < N; j3++) if (!used[j3]) known.push(j3);
	var rhs_nat = Buffer.alloc(n * blockSize);
	var rhs_js = Buffer.alloc(n * blockSize);
	for (var eq2 = 0; eq2 < n; eq2++) {
		recovery[eq2].copy(rhs_nat, eq2 * blockSize);
		recovery[eq2].copy(rhs_js, eq2 * blockSize);
		for (var a = 0; a < known.length; a++) {
			var ki = known[a];
			var cNat = M.readBigUInt64LE((eq2 * N + ki) * 8);
			var d3 = (BigInt(firstInput) + BigInt(ki)) ^ (BigInt(firstRecovery) + BigInt(eq2));
			var cJs = d3 === 0n ? 0n : invert64(d3);
			assert.strictEqual(cNat, cJs, 'RHS coefficient mismatch at eq=' + eq2 + ' k=' + ki);
			mulAdd(encoder, rhs_nat.slice(eq2 * blockSize, (eq2 + 1) * blockSize), inputData[ki], cNat, blockSize);
			mulAdd(encoder, rhs_js.slice(eq2 * blockSize, (eq2 + 1) * blockSize), inputData[ki], cJs, blockSize);
		}
	}
	assert(rhs_nat.equals(rhs_js), 'RHS blocks: native coeffs != JS coeffs');

	// Solve both and compare the repaired (missing) blocks
	var out_nat = Buffer.from(rhs_nat);
	var out_js = Buffer.from(rhs_js);
	var okN = addon.solve_and_reconstruct(A_nat, out_nat, n, blockSize);
	var okJ = addon.solve_and_reconstruct(A_js, out_js, n, blockSize);
	assert(okN === true && okJ === true, 'solve failed: native=' + okN + ' js=' + okJ);
	assert(out_nat.equals(out_js), 'repaired blocks: native != JS');

	console.log('leg1 ok: A matrix, RHS coefficients, repaired blocks bit-identical (N=' + N + ' n=' + n + ')');
}

// ---------------------------------------------------------------------------
// Leg 2: end-to-end par3gen.repair — default (native matrix) vs forced JS
// ---------------------------------------------------------------------------
function leg2(cb) {
	var tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'par3-native-matrix-'));
	var inFile = path.join(tempDir, 'in.bin');
	var base = path.join(tempDir, 'out');
	var blockSize = 4096;
	var numBlocks = 8;
	var fileSize = blockSize * numBlocks;

	// Deterministic input
	var fd = fs.openSync(inFile, 'w');
	var chunk = Buffer.alloc(blockSize);
	var seed = 0x12345678;
	for (var blk = 0; blk < numBlocks; blk++) {
		for (var j = 0; j < blockSize; j += 4) {
			seed = (seed * 1664525 + 1013904223) >>> 0;
			chunk.writeUInt32LE(seed, j);
		}
		fs.writeSync(fd, chunk);
	}
	fs.closeSync(fd);

	par3.create([inFile], base, {
		outputBase: base,
		recoverySlices: 4,
		blockSize: blockSize
	}, function(err) {
		if (err) { cleanup(); return cb(new Error('create failed: ' + err.message)); }

		// Corrupt two blocks (zero them)
		var corrupted = path.join(tempDir, 'corrupt.bin');
		fs.copyFileSync(inFile, corrupted);
		var cfd = fs.openSync(corrupted, 'r+');
		var zero = Buffer.alloc(blockSize);
		fs.writeSync(cfd, zero, 0, blockSize, 2 * blockSize);   // block 2
		fs.writeSync(cfd, zero, 0, blockSize, 5 * blockSize);   // block 5
		fs.closeSync(cfd);

		// Expected full input (deterministic LCG, pre-corruption)
		var expected = Buffer.alloc(fileSize);
		var eseed = 0x12345678;
		for (var eblk = 0; eblk < numBlocks; eblk++) {
			for (var ej = 0; ej < blockSize; ej += 4) {
				eseed = (eseed * 1664525 + 1013904223) >>> 0;
				expected.writeUInt32LE(eseed, eblk * blockSize + ej);
			}
		}

		var run = function(tag, env, cb2) {
			Object.keys(env).forEach(function(k) { process.env[k] = env[k]; });
			var outDir = path.join(tempDir, 'rep_' + tag);
			fs.mkdirSync(outDir);
			// The repair's data/output dir must hold the (corrupted) input file
			fs.copyFileSync(corrupted, path.join(outDir, 'in.bin'));
			par3.repair(base + '.par3', outDir, { verbose: 0 }, function(err2, result) {
				Object.keys(env).forEach(function(k) { delete process.env[k]; });
				if (err2) return cb2(new Error('repair failed: ' + err2.message));
				cb2(null, result, outDir);
			});
		};

		run('native', {}, function(e1, r1, d1) {
			if (e1) { cleanup(); return cb(e1); }
			run('js', { PAR3_REPAIR_NATIVE_MATRIX: '0' }, function(e2, r2, d2) {
				if (e2) { cleanup(); return cb(e2); }
				try {
					assert.strictEqual(r1.repaired, true, 'native repair did not repair');
					assert.strictEqual(r2.repaired, true, 'js repair did not repair');
					var f1 = fs.readFileSync(path.join(d1, 'block_0.dat'));
					var f2 = fs.readFileSync(path.join(d2, 'block_0.dat'));
					assert(f1.equals(f2), 'block_0.dat differs between native-matrix and JS-matrix repair');
					assert(f1.equals(expected), 'block_0.dat does not match the original input (repair wrong)');
					console.log('leg2 ok: default (native matrix) and forced-JS repairs byte-identical and equal the original (' + f1.length + ' bytes, ' + r1.blocksRepaired + ' blocks repaired)');
				} catch (e3) {
					cleanup();
					return cb(e3);
				}
				cleanup();
				cb(null);
			});
		});

		function cleanup() {
			try { fs.rmSync(tempDir, { recursive: true, force: true }); } catch (e) {}
		}
	});
}

// ---------------------------------------------------------------------------
var fails = 0;
try {
	leg1();
} catch (e) {
	fails++;
	console.error('LEG1 FAIL: ' + e.message);
}
leg2(function(err) {
	if (err) {
		fails++;
		console.error('LEG2 FAIL: ' + err.message);
	}
	if (fails) {
		console.error('NATIVE_MATRIX_PARITY_FAIL ' + fails);
		process.exit(1);
	}
	console.log('NATIVE_MATRIX_PARITY_PASS');
	process.exit(0);
});
