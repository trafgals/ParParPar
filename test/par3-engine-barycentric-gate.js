#!/usr/bin/env node
"use strict";

// ============================================================================
// par3-engine-barycentric-gate.js — gate contract test for
//   src/par3_engine_barycentric.cc:185.
//
// Background — cubic review 4910826158 P1 (two P1 findings):
//
//   1. On a hypervisor/OS where CPUID/XCR0 report VPCLMULQDQ but the
//      ZMM probe rejects AVX-512, the original gate
//      `if (gf64_has_vpclmulqdq)` still called
//      gf64_invert_ita_batch and could SIGILL at recovery.
//
//   2. A host reporting AVX-512F/VPOPCNTDQ but lacking VPCLMULQDQ
//      would pass the new false branch, then SIGILL in the
//      subsequent gf64_region_muladd_arr because dispatch selected
//      the VPCLMULQDQ implementation.
//
//   The fix in src/par3_engine_barycentric.cc:185-188 (gated by
//   PR #53 commit 302b59b) re-tightens the gate so that
//   gf64_has_vpclmulqdq = (CPUID+XCR0) & (ZMM probe) — both must
//   succeed. The non-VPCLMULQDQ fallback dispatches through
//   gf64_inverse_batch (scalar / SSSE3 / AVX-2 / AVX-512 inverted
//   batch), which is the GF64Method-table-bound function pointer.
//
// What this test verifies:
//
//   Test 1: with PAR3_GF64_USE_AVX512=1 (operator force-on on a
//           VPCLMULQDQ-capable host), compute_recovery_barycentric
//           completes without throwing / SIGILL and produces the
//           same output as the reference. Validates the
//           gf64_has_vpclmulqdq=1 path.
//
//   Test 2: with PAR3_GF64_USE_AVX512=0 (operator force-off or
//           WSL2-style ZMM-fails), compute_recovery_barycentric
//           STILL completes without throwing / SIGILL. This is
//           the regression gate for the cubic 4910826158 finding:
//           before the fix, on hosts where CPUID reports
//           VPCLMULQDQ but ZMM SIGILLs, this path would crash.
//           After the fix, the gate correctly takes the
//           gf64_inverse_batch fallback path. We verify by
//           comparing the output to the JS reference — if the
//           fallback path silently corrupted the output (or hit
//           UB), the parity would fail.
//
//   Test 3: when dispatched via gf64_inverse_batch (PAR3_GF64_USE_AVX512=0),
//           the recovery result is bit-identical to Test 1 (which
//           uses gf64_invert_ita_batch via gf64_has_vpclmulqdq=1).
//           This pins down that the two paths produce the same
//           result, which is the core correctness contract under
//           the gate.
//
// All tests run compute_recovery_barycentric (the kernel name from
// the binding surface). The test is auto-SKIPped if the addon
// doesn't export compute_recovery_barycentric (the binding is
// shipped in T9 and may be absent on minimal builds).
// ============================================================================

var path = require('path');
var child_process = require('child_process');

var ADDON_PATH = path.join(__dirname, '..', 'build', 'Release', 'parpar_gf64.node');

// JS reference for the Cauchy matrix-vector product — mirrors
// test/par3-barycentric-parity.js:121-127 (uses cauchyCoeff + gf64_mul,
// NOT plain BigInt multiplication).
function jsRecovery(inputs, numInputs, numRecovery, blockSize, firstInput, firstRecovery) {
	var out = Buffer.alloc(numRecovery * blockSize);
	out.fill(0);
	var numWords = blockSize / 8;
	for (var r = 0; r < numRecovery; r++) {
		for (var c = 0; c < numInputs; c++) {
			var xc = (BigInt(firstInput) + BigInt(c)) & 0xFFFFFFFFFFFFFFFFn;
			var yr = (BigInt(firstRecovery) + BigInt(r)) & 0xFFFFFFFFFFFFFFFFn;
			var denom = xc ^ yr;
			if (denom === 0n) denom = 1n; // mirrors the engine's 0->1 mapping
			var inv = gf64Inverse(denom);
			for (var w = 0; w < numWords; w++) {
				var inOff = c * blockSize + w * 8;
				var outOff = r * blockSize + w * 8;
				var inWord = inputs.readBigUInt64LE(inOff);
				var prod = gf64Mul(inWord, inv);
				out.writeBigUInt64LE(
					out.readBigUInt64LE(outOff) ^ prod,
					outOff
				);
			}
		}
	}
	return out;
}

// Modular inverse over GF(2^64) with polynomial 0x100000000000001B.
// Matches par3-barycentric-parity.js: invert64 via PGCD-like algorithm.
function gf64Inverse(a) {
	var GF64_POLY = 0x1000000000000001Bn;
	var GF64_MASK = 0xFFFFFFFFFFFFFFFFn;
	var val = a & GF64_MASK;
	if (val === 0n) return 0n;
	if (val === 1n) return 1n;

	var u = val;
	var v = GF64_POLY;
	var x1 = 1n;
	var x2 = 0n;

	while (u !== 1n && u !== 0n) {
		while ((u & 1n) === 0n) {
			u >>= 1n;
			if ((x1 & 1n) !== 0n) {
				x1 = ((x1 ^ GF64_POLY) >> 1n) & GF64_MASK;
			} else {
				x1 >>= 1n;
			}
		}
		if (u === 1n) break;
		while ((v & 1n) === 0n) {
			v >>= 1n;
		}
		if (u < v) {
			var t = u; u = v; v = t;
			t = x1; x1 = x2; x2 = t;
		}
		u ^= v;
		x1 ^= x2;
	}
	return x1 & GF64_MASK;
}

function gf64Mul(a, b) {
	var GF64_MASK = 0xFFFFFFFFFFFFFFFFn;
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

function genInputs(seed, numInputs, blockSize) {
	var buf = Buffer.alloc(numInputs * blockSize);
	// splitmix64
	var s = BigInt(seed >>> 0);
	var next = function () {
		s = (s + 0x9E3779B97F4A7C15n) & 0xFFFFFFFFFFFFFFFFn;
		var z = s;
		z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & 0xFFFFFFFFFFFFFFFFn;
		z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & 0xFFFFFFFFFFFFFFFFn;
		return (z ^ (z >> 31n)) & 0xFFFFFFFFFFFFFFFFn;
	};
	for (var i = 0; i < numInputs * blockSize / 8; i++) {
		buf.writeBigUInt64LE(next(), i * 8);
	}
	return buf;
}

// Spawn a child node process with the given env, calling
// compute_recovery_barycentric with the given parameters. Returns
// { ok, output, error, exitCode }.
function runRecovery(env, numInputs, numRecovery, blockSize, firstInput, firstRecovery) {
	var inputs = genInputs(0xC0FFEE01, numInputs, blockSize);
	var outputs = Buffer.alloc(numRecovery * blockSize);
	outputs.fill(0);

	var ADDON_ABS = ADDON_PATH;
	var PROBE_CODE = [
		'var addon;',
		'try { addon = require(' + JSON.stringify(ADDON_ABS) + '); }',
		'catch (e) { process.stdout.write(JSON.stringify({error: "require: " + e.message}) + "\\n"); process.exit(0); }',
		'if (typeof addon.compute_recovery_barycentric !== "function") {',
		'  process.stdout.write(JSON.stringify({error: "no_compute_recovery_barycentric"}) + "\\n");',
		'  process.exit(0);',
		'}',
		// Sentinel to make sure we got past the export check before
		// the actual call (so the SIGILL / signal-exit diagnostic
		// is distinguishable from "binding missing").
		'process.stdout.write(JSON.stringify({phase: "ready"}) + "\\n");',
		'var inputs = Buffer.from(' + JSON.stringify(inputs.toString('base64')) + ', "base64");',
		'var outputs = Buffer.alloc(' + (numRecovery * blockSize) + '); outputs.fill(0);',
		'try {',
		'  addon.compute_recovery_barycentric(',
		'    inputs, outputs,',
		'    ' + numInputs + ', ' + numRecovery + ', ' + blockSize + ',',
		'    ' + firstInput + ', ' + firstRecovery + ', 0',
		'  );',
		'  process.stdout.write(JSON.stringify({phase: "done", output: outputs.toString("base64")}) + "\\n");',
		'} catch (e) {',
		'  process.stdout.write(JSON.stringify({phase: "threw", error: e && e.message ? e.message : String(e)}) + "\\n");',
		'}'
	].join('');

	var child = child_process.spawnSync(process.execPath, ['-e', PROBE_CODE], {
		env: env,
		encoding: 'utf8',
		timeout: 60000
	});

	if (child.error) {
		return { ok: false, error: 'spawn: ' + child.error.message };
	}
	if (child.signal) {
		return { ok: false, error: 'signal: ' + child.signal + ' stderr=' + (child.stderr || '').trim() };
	}
	var stdout = (child.stdout || '').trim();
	var last = stdout.split('\n').filter(Boolean).pop();
	try {
		return JSON.parse(last);
	} catch (e) {
		return { ok: false, error: 'parse: ' + e.message + ' raw=' + last };
	}
}

function recordPass(name) {
	console.log('  PASS: ' + name);
	process.exitCode = 0;
}
function recordFail(name, detail) {
	console.error('  FAIL: ' + name + (detail ? ' — ' + detail : ''));
	process.exitCode = 1;
}

// Main
console.log('PAR3 Engine Barycentric Gate Test (cubic review 4910826158)');
console.log('==========================================================\n');

// Step 1: check the binding exists at all.
var probe = runRecovery({}, 4, 2, 8, 0, 100);
if (probe.error === 'no_compute_recovery_barycentric') {
	console.log('SKIPPED: addon does not export compute_recovery_barycentric');
	console.log('(binding shipped in T9; minimal builds may omit it)');
	process.exit(0);
}
if (probe.error && probe.error.indexOf('require:') === 0) {
	console.log('SKIPPED: native addon not built (' + probe.error + ')');
	process.exit(0);
}
if (probe.phase !== 'ready' && probe.phase !== 'done') {
	console.error('FATAL: probe failed before reaching the recovery call: ' +
		JSON.stringify(probe));
	process.exit(1);
}

// Test parameters: numRecovery must be > 32 to clear the small-R
// threshold at src/par3_engine_barycentric.cc:103 — below 32 the
// engine short-circuits to ComputeRecoveryBlocks (legacy 1D-muladd)
// and the VPCLMULQDQ/ITA batch gate at line 185 NEVER executes.
// (Cubic review 4916023985 P2 finding 2: the previous numRecovery=8
// passed on unfixed code because the gate was bypassed.)
//
// numInputs above the legacy small-R threshold, blockSize small to
// keep wall-clock low. The firstInput and firstRecovery are spaced
// so no Cauchy denominator is zero.
var numInputs = 64;
var numRecovery = 64;  // > 32: forces barycentric path with VPCLMULQDQ gate
var blockSize = 64;
var firstInput = 0;
var firstRecovery = numInputs + 1;

// Generate the reference output ONCE so Tests 1, 2, 3 compare
// against the same ground truth.
var inputs = genInputs(0xC0FFEE01, numInputs, blockSize);
var refOutputs = jsRecovery(inputs, numInputs, numRecovery, blockSize, firstInput, firstRecovery);

// --- Test 1: PAR3_GF64_USE_AVX512=1 (force AVX-512 path; only safe
// on a host that actually has working VPCLMULQDQ+ZMM). Verifies the
// gate's "positive" path does not regress.
console.log('Test 1: PAR3_GF64_USE_AVX512=1 → compute_recovery_barycentric succeeds...');
{
	var env1 = JSON.parse(JSON.stringify(process.env));
	env1.PAR3_GF64_USE_AVX512 = '1';
	var r = runRecovery(env1, numInputs, numRecovery, blockSize, firstInput, firstRecovery);
	if (r.error) {
		recordFail('Test 1: spawn error', r.error);
	} else if (r.phase !== 'done') {
		recordFail('Test 1: phase=' + r.phase, r.error || 'no output');
	} else {
		var got = Buffer.from(r.output, 'base64');
		if (got.equals(refOutputs)) {
			recordPass('Test 1: PAR3_GF64_USE_AVX512=1 produces JS-reference output');
		} else {
			recordFail('Test 1: output differs from JS reference', '');
		}
	}
}

// --- Test 2: PAR3_GF64_USE_AVX512=0 (force-off or WSL2-style
// ZMM-fails). This is the regression gate for cubic 4910826158 P1
// finding 1: before the fix, this would SIGILL when CPUID reports
// VPCLMULQDQ but ZMM probe rejects AVX-512. After the fix, the
// gate correctly falls through to gf64_inverse_batch (the
// dispatch-table-bound fallback), which completes without
// SIGILL and produces the correct output.
console.log('\nTest 2: PAR3_GF64_USE_AVX512=0 → compute_recovery_barycentric '
	+ 'succeeds via fallback (cubic 4910826158 P1 finding 1 regression gate)...');
{
	var env2 = JSON.parse(JSON.stringify(process.env));
	env2.PAR3_GF64_USE_AVX512 = '0';
	var r = runRecovery(env2, numInputs, numRecovery, blockSize, firstInput, firstRecovery);
	if (r.error) {
		recordFail('Test 2: spawn error', r.error);
	} else if (r.phase !== 'done') {
		// SIGILL would show up as signal:SIGILL — already caught
		// by r.error. A throw means the binding itself returned
		// an error, which is also a regression.
		recordFail('Test 2: phase=' + r.phase, r.error || 'no output');
	} else {
		var got = Buffer.from(r.output, 'base64');
		if (got.equals(refOutputs)) {
			recordPass('Test 2: PAR3_GF64_USE_AVX512=0 produces JS-reference output '
				+ '(gate correctly takes gf64_inverse_batch fallback)');
		} else {
			recordFail('Test 2: output differs from JS reference (fallback path corrupted)', '');
		}
	}
}

// --- Test 3: both code paths produce IDENTICAL output. This is the
// core correctness contract — the gate's correctness relies on the
// fallback path being equivalent to the primary path. A divergence
// here would mean the fallback uses a different inversion algorithm
// or has a subtle bug.
console.log('\nTest 3: outputs from force-on (Test 1) and force-off (Test 2) '
	+ 'are bit-identical (cubic 4910826158 P1 finding 2 regression gate)...');
{
	var env1 = JSON.parse(JSON.stringify(process.env));
	env1.PAR3_GF64_USE_AVX512 = '1';
	var env2 = JSON.parse(JSON.stringify(process.env));
	env2.PAR3_GF64_USE_AVX512 = '0';
	var r1 = runRecovery(env1, numInputs, numRecovery, blockSize, firstInput, firstRecovery);
	var r2 = runRecovery(env2, numInputs, numRecovery, blockSize, firstInput, firstRecovery);
	if (r1.error || r2.error) {
		recordFail('Test 3: spawn error', 'r1=' + (r1.error || 'ok') + ' r2=' + (r2.error || 'ok'));
	} else if (r1.phase !== 'done' || r2.phase !== 'done') {
		recordFail('Test 3: r1.phase=' + r1.phase + ' r2.phase=' + r2.phase, '');
	} else {
		var got1 = Buffer.from(r1.output, 'base64');
		var got2 = Buffer.from(r2.output, 'base64');
		if (got1.equals(got2)) {
			recordPass('Test 3: gf64_invert_ita_batch (force-on) and gf64_inverse_batch '
				+ '(force-off) produce identical output — fallback equivalence holds');
		} else {
			recordFail('Test 3: force-on and force-off outputs differ — fallback path '
				+ 'is NOT equivalent to primary', '');
		}
	}
}

console.log('');
if (process.exitCode === 1) {
	console.log('FAILED');
} else {
	console.log('ALL PASS');
}
