#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 Fenger Gate Dispatch Test (issue #46 K3 root cause + issue #59 A2)
//
// Pins the dispatch contract for dispatchRecovery (issue #59 §3 always-pad
// policy, A2 + A4):
//
//   1. Fenger routes when: the binding has compute_recovery_fenger, N is
//      padding-reasonable (next_pow2(N) <= 2N — the engine pads
//      non-power-of-2 N internally with synthetic zero-data inputs),
//      R is 0/1/power-of-2, blockSize % 8 == 0, and
//      (PAR3_GF64_USE_FENGER forces, or the host is non-Windows, or —
//      A4/#62 — Windows with N <= PAR3_FENGER_WINDOWS_MAX_INPUTS, an
//      OPTIONAL defensive cap that defaults to NO cap now that the
//      default-on blocker (Node-20 windows-2025 SIGILL from whole-TU
//      /arch:AVX512 auto-vectorized EVEX) is fixed by the gf64_pipeline
//      TU split). PAR3_FENGER_WINDOWS_ENABLE is dead and ignored.
//   2. The create bench pads the source up to a whole number of slices
//      (actualSize = ceil(size/slices) * slices). With the N gate removed
//      (A2), every canonical bench shape is Fenger-ELIGIBLE on N but still
//      blocked by R: 10% recovery slices are never power-of-2. Verified
//      end-to-end through bench-helpers.computeCreateShape +
//      PAR3Gen.totalBlocks + dispatchRecovery with a mock binding.
//
// This is a contract-pinning suite (AGENTS.md test discipline).
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
// For the platform rows the expectation depends on the host (win32 vs not).
var cases = [
	{ name: 'N=2^18, R=2^11, B=8192, forced → fenger', N: 262144, R: 2048, B: 8192, env: '1', expect: 'fenger' },
	{ name: 'N=2^18, R=2^11, B=4096, forced → fenger', N: 262144, R: 2048, B: 4096, env: '1', expect: 'fenger' },
	{ name: 'N=1, R=1, forced → fenger (trivial)', N: 1, R: 1, B: 4096, env: '1', expect: 'fenger' },
	{ name: 'N=0, R=1, forced → fenger (trivial zero)', N: 0, R: 1, B: 4096, env: '1', expect: 'fenger' },
	// A2: N no longer needs to be power-of-2 (engine pads internally).
	{ name: 'N=262145 (1G/1000 bench shape), R=2048, forced → fenger (padded N, pow2 R)', N: 262145, R: 2048, B: 4096, env: '1', expect: 'fenger' },
	{ name: 'N=1000 (odd, small), R=512, forced → fenger (padded N, pow2 R)', N: 1000, R: 512, B: 4096, env: '1', expect: 'fenger' },
	{ name: 'N=262145 (1G/1000 bench shape), R=100, forced → NOT fenger (R not pow2) → barycentric', N: 262145, R: 100, B: 4096, env: '1', expect: 'barycentric' },
	{ name: 'N=262146 (1G/10K bench shape), R=1000, forced → NOT fenger (R not pow2) → barycentric', N: 262146, R: 1000, B: 4096, env: '1', expect: 'barycentric' },
	{ name: 'N=2^18, R=1000 (R not pow2), forced → NOT fenger → barycentric', N: 262144, R: 1000, B: 4096, env: '1', expect: 'barycentric' },
	{ name: 'N=2^18, R=2^11, B=4100 (B%8!=0), forced → NOT fenger → full (B%8!=0 also blocks barycentric)', N: 262144, R: 2048, B: 4100, env: '1', expect: 'full' },
	// A2: next_pow2(N) <= 2N always holds, so the N gate is fully
	// removed. The huge-N no-hang pins live in Section 1b as CHILD
	// PROCESS runs: if _fengerPaddingReasonable regressed to the
	// 32-bit-truncating 'p <<= 1', an in-parent row here would hang
	// this suite before the pin could report a FAIL (cubic review
	// 50f46d24 P2).
	{ name: 'N=2^18, R=2^11, B=8192, kill switch (env=0) → NOT fenger → barycentric', N: 262144, R: 2048, B: 8192, env: '0', expect: 'barycentric' },
	{ name: 'N=2^18, R=2048, B=8192, no env → barycentric (R=2048 < FENGER_MIN_R=8192)', N: 262144, R: 2048, B: 8192, env: undefined, expect: 'barycentric' },
	// A4/#62: the Windows opt-in gate is REMOVED (the default-on blocker
	// — Node-20 windows-2025 SIGILL — was whole-TU /arch:AVX512 EVEX
	// auto-vectorization, fixed by the gf64_pipeline TU split). Windows
	// dispatches Fenger by the same rules as other hosts; the old
	// PAR3_FENGER_WINDOWS_ENABLE is dead and must be ignored. An
	// OPTIONAL defensive cap (PAR3_FENGER_WINDOWS_MAX_INPUTS) remains,
	// defaulting to NO cap.
	{ name: 'N=1000, R=512, no env → full (R=512 < FENGER_MIN_R=8192, N=1000 ≤ BARY_MIN)', N: 1000, R: 512, B: 4096, env: undefined, expect: 'full' },
	{ name: 'N=1000, R=512, PAR3_FENGER_WINDOWS_ENABLE=0 (dead env ignored) → full', N: 1000, R: 512, B: 4096, env: undefined, winEnable: '0', expect: 'full' },
	{ name: 'N=2^18, R=512, no env → barycentric (R=512 < FENGER_MIN_R=8192)', N: 262144, R: 512, B: 4096, env: undefined, expect: 'barycentric' },
	{ name: 'N=2^18, R=512, PAR3_FENGER_WINDOWS_ENABLE=1 (dead env ignored) → barycentric', N: 262144, R: 512, B: 4096, env: undefined, winEnable: '1', expect: 'barycentric' },
	{ name: 'N=2000, R=512, no env, PAR3_FENGER_WINDOWS_MAX_INPUTS=100 → full (R=512 < FENGER_MIN_R=8192, N=2000 ≤ BARY_MIN)', N: 2000, R: 512, B: 4096, env: undefined, winMax: '100', expect: 'full' },
	{ name: 'N=2000, R=512, PAR3_FENGER_WINDOWS_MAX_INPUTS=100 (cap wins over dead enable) → full', N: 2000, R: 512, B: 4096, env: undefined, winEnable: '1', winMax: '100', expect: 'full' },
	// FENGER_MIN_R boundary (cost-model gate, issue #96 P0):
	// Fenger is selected when R >= FENGER_MIN_R (default 8192).
	{ name: 'N=2^18, R=8192 (FENGER_MIN_R boundary), no env → fenger (R >= FENGER_MIN_R)', N: 262144, R: 8192, B: 4096, env: undefined, expect: 'fenger' },
	{ name: 'N=2^18, R=8192, kill switch (env=0) → barycentric', N: 262144, R: 8192, B: 4096, env: '0', expect: 'barycentric' },
	{ name: 'N=2^18, R=4096 (just below FENGER_MIN_R), no env → barycentric', N: 262144, R: 4096, B: 4096, env: undefined, expect: 'barycentric' },
	{ name: 'N=2^18, R=2048, PAR3_FENGER_MIN_R=1024 → fenger (env override lowers threshold)', N: 262144, R: 2048, B: 4096, env: undefined, fengerMinR: '1024', expect: 'fenger' }
];

cases.forEach(function (c) {
	withFengerEnv(c.env, function () {
		var prevMax = process.env.PAR3_FENGER_WINDOWS_MAX_INPUTS;
		var prevEnable = process.env.PAR3_FENGER_WINDOWS_ENABLE;
		var prevMinR = process.env.PAR3_FENGER_MIN_R;
		if (c.winMax === undefined) delete process.env.PAR3_FENGER_WINDOWS_MAX_INPUTS;
		else process.env.PAR3_FENGER_WINDOWS_MAX_INPUTS = c.winMax;
		if (c.winEnable === undefined) delete process.env.PAR3_FENGER_WINDOWS_ENABLE;
		else process.env.PAR3_FENGER_WINDOWS_ENABLE = c.winEnable;
		if (c.fengerMinR === undefined) delete process.env.PAR3_FENGER_MIN_R;
		else process.env.PAR3_FENGER_MIN_R = c.fengerMinR;
		try {
			var b = makeBinding();
			var result = dispatchRecovery(b, null, null, c.N, c.R, c.B, 0, 0n, 0);
			var called = b.calls.length ? b.calls[b.calls.length - 1].kernel : '(none)';
			check(result === c.expect + '-result' && called === c.expect, c.name + ' (called: ' + called + ')');
		} finally {
			if (prevMax === undefined) delete process.env.PAR3_FENGER_WINDOWS_MAX_INPUTS;
			else process.env.PAR3_FENGER_WINDOWS_MAX_INPUTS = prevMax;
			if (prevEnable === undefined) delete process.env.PAR3_FENGER_WINDOWS_ENABLE;
			else process.env.PAR3_FENGER_WINDOWS_ENABLE = prevEnable;
			if (prevMinR === undefined) delete process.env.PAR3_FENGER_MIN_R;
			else process.env.PAR3_FENGER_MIN_R = prevMinR;
		}
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
// Section 1b — no-hang pin for huge N (cubic review c509dd2b P3)
// ----------------------------------------------------------------------------
// The two huge-N rows above verify routing, but if _fengerPaddingReasonable
// regressed to the 32-bit-truncating `p <<= 1` (wraps to 0 for N >= 2^30+1
// and loops forever) they would HANG the whole suite and only surface as a
// CI timeout. Run the same dispatch in a child process with a hard timeout
// so the regression produces a FAIL instead of an infinite loop.
// ============================================================================

console.log('\n--- Section 1b: no-hang pin for huge N (child process, hard timeout) ---');

(function () {
	var cp = require('node:child_process');
	var childSrc = [
		'"use strict";',
		'var par3gen = require(' + JSON.stringify(par3genPath) + ');',
		'process.env.PAR3_GF64_USE_FENGER = "1";',
		'var binding = {',
		'  compute_recovery_fenger: function () { return "fenger-result"; },',
		'  compute_recovery_barycentric: function () { return "barycentric-result"; },',
		'  compute_recovery_full: function () { return "full-result"; },',
		'  compute_recovery: function () { return "per-batch-result"; }',
		'};',
		'var result = null;',
		'var ns = [1610612737, 2147483649]; /* 3*2^29+1, 2^31+1 */',
		'for (var i = 0; i < ns.length; i++) {',
		'  result = par3gen.dispatchRecovery(binding, null, null, ns[i], 2048, 4096, 0, 0n, 0);',
		'  if (result !== "fenger-result") { console.error("N=" + ns[i] + " routed to: " + result); process.exit(2); }',
		'}',
		'console.log("child-ok");'
	].join('\n');
	var child = cp.spawnSync(process.execPath, ['-e', childSrc], { timeout: 20000, encoding: 'utf8' });
	var ok;
	var why;
	if (child.error && child.error.code === 'ETIMEDOUT') {
		ok = false;
		why = 'child timed out (would hang the suite on regression)';
	} else if (child.status !== 0) {
		ok = false;
		why = 'child exited ' + child.status + ': ' + ((child.stdout || '') + (child.stderr || '')).trim();
	} else {
		ok = /child-ok/.test(child.stdout || '');
		why = ok ? 'terminates and routes to fenger' : 'child did not report ok: ' + child.stdout;
	}
	check(ok, 'N=2^31+1 eligibility terminates and routes to fenger (child, 20s cap): ' + why);
})();

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

// With the N gate removed (issue #59 A2), the blocker for every canonical
// 10%-recovery bench shape is R: recoverySlices is never power-of-2 —
// that is why the Fenger gate still does not fire on create benches.
console.log('\n--- Section 2b: canonical bench scenarios blocked by R (N gate removed by A2) ---');
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
	var rPow2 = isPow2OrTrivial(s.recoverySlices);
	check(!rPow2, sc.label + ': N=' + s.totalBlocks + ' (A2: N-gate removed), R=' + s.recoverySlices + ' pow2=' + rPow2 + ' → gate blocked by R (never pow2)');
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
// Section 4 — end-to-end: canonical bench shapes still never dispatch Fenger
// ============================================================================

console.log('\n--- Section 4: end-to-end on canonical bench shapes (mock binding) ---');
scenarios.forEach(function (sc) {
	var s = helpers.computeCreateShape(sc.size, sc.slices, sc.blockSize);
	withFengerEnv('1', function () {
		var b = makeBinding();
		dispatchRecovery(b, null, null, s.totalBlocks, s.recoverySlices, sc.blockSize, 0, 0n, 0);
		var called = b.calls.length ? b.calls[b.calls.length - 1].kernel : '(none)';
		check(called !== 'fenger', sc.label + ' (N=' + s.totalBlocks + ', R=' + s.recoverySlices + ') with FORCE env → falls back to ' + called + ', never fenger (R gate)');
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
