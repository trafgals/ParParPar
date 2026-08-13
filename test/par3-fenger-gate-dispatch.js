#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 Fenger Gate Dispatch Test (issue #46 K3 root cause)
//
// Pins two contracts discovered while chasing "N is always odd" on the
// create bench (native-Windows 2026-08-13 session):
//
//   1. dispatchRecovery routes to compute_recovery_fenger ONLY when
//      numInputs AND numRecovery are 0/1/power-of-2, blockSize % 8 == 0,
//      and (PAR3_GF64_USE_FENGER forces, or the host is non-Windows).
//   2. The create bench pads the source up to a whole number of slices
//      (actualSize = ceil(size/slices) * slices), which makes totalBlocks
//      odd for every canonical bench shape — so the Fenger power-of-2
//      gate can never fire on those shapes. Verified end-to-end through
//      bench-helpers.computeCreateShape + PAR3Gen.totalBlocks +
//      dispatchRecovery with a mock binding.
//
// This is a contract-pinning suite: it documents current master behaviour
// so a future K5 padding / dispatch change cannot silently regress the
// gate semantics (AGENTS.md test discipline — issue #46 follow-up).
// ============================================================================

var path = require('path');
var assert = require('node:assert');

console.log('PAR3 Fenger Gate Dispatch Test (issue #46 N-always-odd root cause)');
console.log('==================================================================\n');

var passed = 0;
var failed = 0;
var skipped = 0;
var total = 0;
var failures = [];

function check(condition, msg) {
	total++;
	if (condition) {
		console.log('  PASS: ' + msg);
		passed++;
	} else {
		console.error('  FAIL: ' + msg);
		failed++;
		failures.push(msg);
		process.exitCode = 1;
	}
}

function isPow2OrTrivial(n) {
	return n === 0 || n === 1 || (n > 1 && (n & (n - 1)) === 0);
}

var par3genPath = path.resolve(__dirname, '..', 'lib', 'par3gen.js');
var par3gen;
try {
	par3gen = require(par3genPath);
} catch (e) {
	console.error('FATAL: could not require lib/par3gen.js: ' + (e && e.message));
	process.exit(1);
}

var dispatchRecovery = par3gen.dispatchRecovery;
if (typeof dispatchRecovery !== 'function') {
	console.error('FATAL: dispatchRecovery not exported from lib/par3gen.js');
	process.exit(1);
}

var helpers;
try {
	helpers = require('./bench/bench-helpers.js');
} catch (e) {
	console.error('FATAL: could not require test/bench/bench-helpers.js: ' + (e && e.message));
	process.exit(1);
}

// --- Mock binding -----------------------------------------------------------

function makeBinding(withFenger) {
	var binding = { calls: [] };
	binding.compute_recovery_fenger = function (a, b, c, d, e, f, g, h) {
		binding.calls.push({ kernel: 'fenger', args: { numInputs: c, numRecovery: d, blockSize: e, firstInput: f, firstRecovery: g, numThreads: h } });
		return 'fenger-result';
	};
	binding.compute_recovery_barycentric = function (a, b, c, d, e, f, g, h) {
		binding.calls.push({ kernel: 'barycentric', args: { numInputs: c, numRecovery: d, blockSize: e, firstInput: f, firstRecovery: g, numThreads: h } });
		return 'barycentric-result';
	};
	binding.compute_recovery_full = function (a, b, c, d, e, f, g, h) {
		binding.calls.push({ kernel: 'full', args: { numInputs: c, numRecovery: d, blockSize: e, firstInput: f, firstRecovery: g, numThreads: h } });
		return 'full-result';
	};
	binding.compute_recovery = function (a, b, c, d, e, f, g, h) {
		binding.calls.push({ kernel: 'per-batch', args: { numInputs: c, numRecovery: d, blockSize: e, firstInput: f, firstRecovery: g, numThreads: h } });
		return 'per-batch-result';
	};
	if (withFenger === false) {
		delete binding.compute_recovery_fenger;
	}
	return binding;
}

function withFengerEnv(value, fn) {
	var prev = process.env.PAR3_GF64_USE_FENGER;
	if (value === undefined || value === null) {
		delete process.env.PAR3_GF64_USE_FENGER;
	} else {
		process.env.PAR3_GF64_USE_FENGER = String(value);
	}
	try {
		return fn();
	} finally {
		if (prev === undefined) delete process.env.PAR3_GF64_USE_FENGER;
		else process.env.PAR3_GF64_USE_FENGER = prev;
	}
}

// ============================================================================
// Section 1 — dispatch gate contract (mock binding, no native addon needed)
// ============================================================================

console.log('--- Section 1: Fenger dispatch gate (mock binding) ---');

// Case table. expected is the kernel dispatchRecovery must call.
// For the platform row the expectation depends on the host (win32 vs not).
var cases = [
	{ name: 'N=2^18, R=2^11, B=8192, forced → fenger', N: 262144, R: 2048, B: 8192, env: '1', expect: 'fenger' },
	{ name: 'N=2^18, R=2^11, B=4096, forced → fenger', N: 262144, R: 2048, B: 4096, env: '1', expect: 'fenger' },
	{ name: 'N=1, R=1, forced → fenger (trivial)', N: 1, R: 1, B: 4096, env: '1', expect: 'fenger' },
	{ name: 'N=0, R=1, forced → fenger (trivial zero)', N: 0, R: 1, B: 4096, env: '1', expect: 'fenger' },
	{ name: 'N=262145 (1G/1000 bench shape), R=100, forced → NOT fenger (N odd) → barycentric', N: 262145, R: 100, B: 4096, env: '1', expect: 'barycentric' },
	{ name: 'N=262146 (1G/10K bench shape), R=1000, forced → NOT fenger (N odd) → barycentric', N: 262146, R: 1000, B: 4096, env: '1', expect: 'barycentric' },
	{ name: 'N=2^18, R=1000 (R not pow2), forced → NOT fenger → barycentric', N: 262144, R: 1000, B: 4096, env: '1', expect: 'barycentric' },
	{ name: 'N=2^18, R=2^11, B=4100 (B%8!=0), forced → NOT fenger → full (B%8!=0 also blocks barycentric)', N: 262144, R: 2048, B: 4100, env: '1', expect: 'full' },
	{ name: 'N=1000, R=512 (N small + odd), forced → NOT fenger → full fallback', N: 1000, R: 512, B: 4096, env: '1', expect: 'full' },
	{ name: 'N=2^18, R=2^11, B=8192, kill switch (env=0) → NOT fenger → barycentric', N: 262144, R: 2048, B: 8192, env: '0', expect: 'barycentric' },
	{ name: 'N=2^18, R=2^11, B=8192, no env, non-Windows → fenger (platform gate)', N: 262144, R: 2048, B: 8192, env: undefined, expect: (process.platform === 'win32') ? 'barycentric' : 'fenger' }
];

cases.forEach(function (c) {
	withFengerEnv(c.env, function () {
		var b = makeBinding();
		var result = dispatchRecovery(b, null, null, c.N, c.R, c.B, 0, 0n, 0);
		var called = b.calls.length ? b.calls[b.calls.length - 1].kernel : '(none)';
		check(result === c.expect + '-result' && called === c.expect, c.name + ' (called: ' + called + ')');
	});
});

// Missing compute_recovery_fenger on the binding + forced env → hard error.
withFengerEnv('1', function () {
	var b = makeBinding(false);
	var threw = false;
	try {
		dispatchRecovery(b, null, null, 262144, 2048, 8192, 0, 0n, 0);
	} catch (e) {
		threw = /lacks compute_recovery_fenger/.test(e.message);
	}
	check(threw, 'PAR3_GF64_USE_FENGER=1 + binding without compute_recovery_fenger → throws (got: ' + (threw ? 'throw' : 'no throw') + ')');
});

// Missing compute_recovery_fenger, no env → silent fallback (no throw).
withFengerEnv(undefined, function () {
	var b = makeBinding(false);
	var result = dispatchRecovery(b, null, null, 262144, 2048, 8192, 0, 0n, 0);
	var called = b.calls.length ? b.calls[b.calls.length - 1].kernel : '(none)';
	check(result === 'barycentric-result' && called === 'barycentric', 'no fenger on binding, no env → silent barycentric fallback (called: ' + called + ')');
});

// ============================================================================
// Section 2 — bench shape math (computeCreateShape)
// ============================================================================

console.log('\n--- Section 2: bench shape math (computeCreateShape) ---');

function shapeCheck(size, slices, blockSize, expSlice, expActual, expRec, expBlocks, label) {
	var s = helpers.computeCreateShape(size, slices, blockSize);
	check(
		s.sliceSize === expSlice && s.actualSize === expActual &&
		s.recoverySlices === expRec && s.totalBlocks === expBlocks,
		label + ' → sliceSize=' + s.sliceSize + ' actualSize=' + s.actualSize +
		' recoverySlices=' + s.recoverySlices + ' totalBlocks=' + s.totalBlocks +
		' (expected ' + expSlice + '/' + expActual + '/' + expRec + '/' + expBlocks + ')'
	);
}

var G = 1024 * 1024 * 1024;
shapeCheck(1 * G, 1000, 4096, 1073742, 1073742000, 100, 262145, '1G/1000 slice pad (observed on bench: N=262145)');
shapeCheck(1 * G, 10000, 4096, 107375, 1073750000, 1000, 262146, '1G/10K slice pad (observed on bench: N=262146)');
shapeCheck(10 * G, 100000, 4096, 107375, 10737500000, 10000, 2621460, '10G/100k slice pad (observed on bench: N=2621460)');
shapeCheck(1 * G, 65536, 4096, 16384, 1 * G, 6553, 262144, '1G/65536 exact division → N IS power-of-2 (but R=6553 is not)');
shapeCheck(1 * G, 20480, 8192, 52429, 1073745920, 2048, 131073, '1G/20480 → R IS power-of-2 (2048) but N=131073 is not');

// The "N always odd" property for every canonical bench scenario: no
// canonical 10%-recovery shape has BOTH totalBlocks and recoverySlices
// power-of-2 — that is why the Fenger gate never fires on create benches.
console.log('\n--- Section 2b: canonical bench scenarios never satisfy the Fenger gate ---');
var scenarios = [
	{ size: 1 * G, slices: 1000, blockSize: 4096, label: '1G/1K' },
	{ size: 1 * G, slices: 10000, blockSize: 4096, label: '1G/10K' },
	{ size: 10 * G, slices: 100000, blockSize: 4096, label: '10G/100k (B1 acceptance)' },
	{ size: 10 * G, slices: 1000000, blockSize: 4096, label: '10G/1M (large mode)' },
	{ size: 1 * G, slices: 100000, blockSize: 4096, label: '1G/100K' },
	{ size: 1 * G, slices: 1000000, blockSize: 4096, label: '1G/1M' }
];
scenarios.forEach(function (sc) {
	var s = helpers.computeCreateShape(sc.size, sc.slices, sc.blockSize);
	var nPow2 = isPow2OrTrivial(s.totalBlocks);
	var rPow2 = isPow2OrTrivial(s.recoverySlices);
	check(!(nPow2 && rPow2), sc.label + ': N=' + s.totalBlocks + ' pow2=' + nPow2 + ', R=' + s.recoverySlices + ' pow2=' + rPow2 + ' → gate blocked (never both)');
});

// ============================================================================
// Section 3 — engine block accounting (PAR3Gen.totalBlocks)
// ============================================================================

console.log('\n--- Section 3: PAR3Gen.totalBlocks accounting ---');

var bindingAvailable = true;
try {
	require('../build/Release/parpar_gf64.node');
} catch (e) {
	bindingAvailable = false;
}

function totalBlocksFor(size, blockSize) {
	var p = new par3gen.PAR3Gen([{ path: 'bench-source.bin', size: size, name: 'bench-source.bin' }], blockSize, { gfMethod: 'avx2' });
	return p.totalBlocks;
}

if (bindingAvailable) {
	check(totalBlocksFor(1 * G, 4096) === 262144, '2^30 bytes @ 4096 → 262144 (power-of-2; Fenger-reachable N)');
	check(totalBlocksFor(1 * G, 8192) === 131072, '2^30 bytes @ 8192 → 131072 (power-of-2)');
	check(totalBlocksFor(1 * G + 1, 4096) === 262145, '2^30+1 bytes @ 4096 → 262145 (odd)');
	check(totalBlocksFor(1073742000, 4096) === 262145, '1G/1K bench actualSize → 262145 (odd — root cause)');
	check(totalBlocksFor(1073750000, 4096) === 262146, '1G/10K bench actualSize → 262146 (odd — root cause)');
	check(totalBlocksFor(10737500000, 4096) === 2621460, '10G/100k bench actualSize → 2621460 (not pow2 — B1 acceptance)');
} else {
	skipped++;
	console.log('  SKIP: native addon not built — PAR3Gen accounting section skipped');
}

// ============================================================================
// Section 4 — end-to-end: the canonical bench shape never dispatches Fenger
// ============================================================================

console.log('\n--- Section 4: end-to-end on canonical bench shapes (mock binding) ---');
scenarios.forEach(function (sc) {
	var s = helpers.computeCreateShape(sc.size, sc.slices, sc.blockSize);
	withFengerEnv('1', function () {
		var b = makeBinding();
		dispatchRecovery(b, null, null, s.totalBlocks, s.recoverySlices, sc.blockSize, 0, 0n, 0);
		var called = b.calls.length ? b.calls[b.calls.length - 1].kernel : '(none)';
		check(called !== 'fenger', sc.label + ' (N=' + s.totalBlocks + ', R=' + s.recoverySlices + ') with FORCE env → falls back to ' + called + ', never fenger');
	});
});

// ============================================================================

console.log('\n' + passed + ' passed, ' + failed + ' failed, ' + skipped + ' skipped');
if (failed) {
	console.error('Failures:');
	failures.forEach(function (f) { console.error('  - ' + f); });
	process.exit(1);
}
process.exit(0);
