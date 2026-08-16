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
var gf64js = require('../lib/gf64_js.js');

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
// Leg 0: gate unit — boundary at the cap, one past, overlap, degenerate
// ---------------------------------------------------------------------------
function leg0() {
	var cap = 32 * 1024 * 1024;
	// boundary: exactly at the cap is allowed (input [0x1000, 0x11000),
	// recovery [0x20000, ...) — disjoint)
	assert.strictEqual(par3.repairNativeMatrixAllowed(65536, 512, 0x1000, 0x20000, cap), true, 'at-cap should be allowed');
	// one past the cap is denied (n x total = 65536*513 > cap)
	assert.strictEqual(par3.repairNativeMatrixAllowed(65536, 513, 0x1000, 0x20000, cap), false, 'past-cap should be denied');
	// overlapping ranges are denied (JS fallback keeps historical semantics)
	assert.strictEqual(par3.repairNativeMatrixAllowed(64, 8, 0x1000, 0x1000, cap), false, 'overlap should be denied');
	// disjoint ranges allowed
	assert.strictEqual(par3.repairNativeMatrixAllowed(64, 8, 0x1000, 0x10000, cap), true, 'disjoint should be allowed');
	// degenerate sizes denied
	assert.strictEqual(par3.repairNativeMatrixAllowed(0, 8, 0x1000, 0x10000, cap), false, 'zero inputs denied');
	assert.strictEqual(par3.repairNativeMatrixAllowed(64, 0, 0x1000, 0x10000, cap), false, 'zero missing denied');
	console.log('leg0 ok: gate boundary/overlap/degenerate cases');
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
// Leg 3: coupled_muladd parity — native, JS fallback, manual reference
// ---------------------------------------------------------------------------
function leg3() {
	var blockSize = 512, numWords = blockSize / 8;
	var G = 5;
	var encoder = new addon.Gf64Encoder(0);
	var in_blocks = [];
	for (var g = 0; g < G; g++) {
		var b = Buffer.alloc(blockSize);
		for (var i = 0; i < blockSize; i++) b[i] = Math.floor(Math.random() * 256);
		in_blocks.push(b);
	}
	var coeffs = [];
	var coeffBuf = Buffer.alloc(G * 8);
	for (var g2 = 0; g2 < G; g2++) {
		var c = rnd64();
		coeffs.push(c);
		coeffBuf.writeBigUInt64LE(c, g2 * 8);
	}
	var outNat = Buffer.alloc(blockSize);
	var outJs = Buffer.alloc(blockSize);
	var outRef = Buffer.alloc(blockSize);
	for (var i2 = 0; i2 < blockSize; i2++) {
		var v = Math.floor(Math.random() * 256);
		outNat[i2] = outJs[i2] = outRef[i2] = v;
	}
	encoder.coupled_muladd_arr(outNat, in_blocks, coeffBuf, numWords, G);
	gf64js.coupled_muladd_arr(outJs, in_blocks, coeffBuf, numWords, G);
	for (var w = 0; w < numWords; w++) {
		var acc = 0n;
		for (var g3 = 0; g3 < G; g3++) {
			acc ^= gf64js.gf64_mul(in_blocks[g3].readBigUInt64LE(w * 8), coeffs[g3]);
		}
		outRef.writeBigUInt64LE(outRef.readBigUInt64LE(w * 8) ^ acc, w * 8);
	}
	assert(outNat.equals(outJs), 'coupled: native != JS');
	assert(outNat.equals(outRef), 'coupled: native != manual reference');
	console.log('leg3 ok: coupled_muladd native == JS == manual (G=' + G + ')');
}

// ---------------------------------------------------------------------------
// Leg 4: coupled backend selector (cubic review 208da585 P2) — native
// Gf64Encoder caps G at 65536; above that the uncapped JS fallback must be
// used (the legacy loop had no cap either); PAR3_REPAIR_COUPLED_RHS=0 must
// yield null (legacy loop).
// ---------------------------------------------------------------------------
function leg4() {
	var sel = par3.selectCoupledMuladd;
	assert(sel && typeof sel === 'function', 'selectCoupledMuladd not exported');

	var fSmall = sel(addon, 100, gf64js);
	assert(fSmall !== null && typeof fSmall === 'function', 'G<=65536 should pick a coupled backend');
	assert(fSmall !== gf64js.coupled_muladd_arr, 'G<=65536 should pick the native backend (not the JS fallback)');

	var fCap = sel(addon, 65536, gf64js);
	assert(fCap !== null && fCap !== gf64js.coupled_muladd_arr, 'G==65536 boundary should still pick native');

	var fHuge = sel(addon, 65537, gf64js);
	assert(fHuge === gf64js.coupled_muladd_arr, 'G>65536 must fall back to the uncapped JS backend');

	// env=0 forces the legacy loop
	var prev = process.env.PAR3_REPAIR_COUPLED_RHS;
	process.env.PAR3_REPAIR_COUPLED_RHS = '0';
	var fOff = sel(addon, 100, gf64js);
	if (prev === undefined) delete process.env.PAR3_REPAIR_COUPLED_RHS;
	else process.env.PAR3_REPAIR_COUPLED_RHS = prev;
	assert(fOff === null, 'PAR3_REPAIR_COUPLED_RHS=0 must force the legacy loop');

	console.log('leg4 ok: coupled backend selector (native <= 65536 incl. boundary, JS above, env=0 legacy)');
}

var PAR3_MAGIC = Buffer.from('PAR3\0PKT');
var PAR3_PKT_HDR_SIZE = 48;

// Zero whole DATA packets (header + body) so the magic signature is
// destroyed and the streaming parser skips them — the repair then treats
// those blocks as genuinely missing and exercises its repair path
// (cf. e2e-par3-repair.js:95-105; zeroing input-file bytes is NOT enough).
function damageArchiveDataPackets(par3File, wantBlocks) {
	var fd = fs.openSync(par3File, 'r+');
	var stat = fs.fstatSync(fd);
	var offset = 0;
	var damaged = [];
	while (offset < stat.size && damaged.length < wantBlocks.length) {
		var header = Buffer.alloc(PAR3_PKT_HDR_SIZE);
		fs.readSync(fd, header, 0, PAR3_PKT_HDR_SIZE, offset);
		if (!header.slice(0, 8).equals(PAR3_MAGIC)) { offset += 8; continue; }
		var totalLen = Number(header.readBigUInt64LE(24));
		if (totalLen < PAR3_PKT_HDR_SIZE || offset + totalLen > stat.size + 8) break;
		var typeStr = header.slice(40, 48).toString('ascii');
		if (typeStr === 'PAR DAT\0') {
			var bodyOffset = offset + PAR3_PKT_HDR_SIZE;
			var idxBuf = Buffer.alloc(8);
			fs.readSync(fd, idxBuf, 0, 8, bodyOffset);
			var blockIndex = Number(idxBuf.readBigUInt64LE(0));
			if (wantBlocks.indexOf(blockIndex) !== -1) {
				fs.writeSync(fd, Buffer.alloc(totalLen), 0, totalLen, offset);
				damaged.push(blockIndex);
			}
		}
		offset += totalLen;
	}
	fs.fsyncSync(fd);
	fs.closeSync(fd);
	return damaged;
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

		// Damage the archive: destroy the DATA packets for blocks 2 and 5 so
		// the repair sees them as genuinely missing (zeroing input bytes is
		// not enough — the repair only treats absent packets as missing).
		var damaged = damageArchiveDataPackets(base + '.par3', [2, 5]);
		if (damaged.length !== 2) { cleanup(); return cb(new Error('damageArchive: expected 2 DATA packets damaged, got ' + damaged.length)); }

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
			// The repair's data/output dir holds the PRISTINE input file — the
			// missing blocks come from the damaged archive packets.
			fs.copyFileSync(inFile, path.join(outDir, 'in.bin'));
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
					// The default leg must have engaged the native matrix path and
					// the forced leg must have used the JS fallback — otherwise the
					// byte-identical assertion is JS-vs-JS and proves nothing.
					assert.strictEqual(r1.usedNativeMatrix, true, 'default repair did not use the native matrix');
					assert.strictEqual(r2.usedNativeMatrix, false, 'forced-JS repair used the native matrix');
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
	leg0();
	leg1();
	leg3();
	leg4();
} catch (e) {
	fails++;
	console.error('LEG0/1/3/4 FAIL: ' + e.message);
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
