#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 Routing-Gate Contract Test (issue #59 T1)
//
// Pins the deterministic kernel-routing decision table for the
// decideRecoveryKernel cost model inserted into lib/par3gen.js BEFORE the
// coeff worker spawn. The gate replaces the dense-matrix-OOM accident that
// previously engaged the Fenger fast-path for large pow2 slice counts.
//
// Cost model (lower = cheaper):
//   matvec   = R*N*blockSize + R*N*8              (byte-traffic + dense matrix footprint)
//   fenger   = (N+R)*log2(N+R)^2*FENGER_FIELD_OP_COST + R*blockSize
//   bary     = N*R*blockSize                       (direct Cauchy matvec)
//
// Decision table under test (default env; binding has all three kernels):
//   {N=1024,    R=128,    B=4096}  -> matvec       (small; matrix=1 MiB fits)
//   {N=32768,   R=4096,   B=4096}  -> fenger|matvec (determinism; matrix=1 GiB infeasible -> fenger)
//   {N=65536,   R=8192,   B=4096}  -> fenger       (R >= FENGER_MIN_R=8192)
//   {N=262144,  R=32768,  B=4096}  -> fenger       (R >> FENGER_MIN_R)
//   non-pow2 R                       -> barycentric (fenger requires pow2 R; matvec infeasible at large N)
//
// AGENTS.md test discipline: every code change ships its regression test in
// the SAME commit. This file lives next to the gate change.
// ============================================================================

var path = require('path');
var assert = require('node:assert');

console.log('PAR3 Routing-Gate Contract Test (issue #59 T1)');
console.log('==============================================\n');

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

var par3genPath = path.resolve(__dirname, '..', 'lib', 'par3gen.js');
var par3gen;
try {
	par3gen = require(par3genPath);
} catch (e) {
	console.error('FATAL: could not require lib/par3gen.js: ' + (e && e.message));
	process.exit(1);
}

var decideRecoveryKernel = par3gen.decideRecoveryKernel;
if (typeof decideRecoveryKernel !== 'function') {
	console.error('FATAL: decideRecoveryKernel not exported from lib/par3gen.js');
	process.exit(1);
}

var dispatchRecovery = par3gen.dispatchRecovery;
if (typeof dispatchRecovery !== 'function') {
	console.error('FATAL: dispatchRecovery not exported from lib/par3gen.js');
	process.exit(1);
}

var BARY_MIN_INPUTS_DEFAULT = par3gen.BARY_MIN_INPUTS_DEFAULT;
if (typeof BARY_MIN_INPUTS_DEFAULT !== 'number') {
	console.error('FATAL: BARY_MIN_INPUTS_DEFAULT not exported');
	process.exit(1);
}

// --- Mock binding -----------------------------------------------------------

function makeBinding() {
	var b = { calls: [] };
	b.compute_recovery_fenger = function () {
		b.calls.push('fenger');
		return 'fenger-result';
	};
	b.compute_recovery_barycentric = function () {
		b.calls.push('barycentric');
		return 'barycentric-result';
	};
	b.compute_recovery_with_coeff = function () {
		b.calls.push('matvec');
		return 'matvec-result';
	};
	b.compute_recovery_full = function () {
		b.calls.push('full');
		return 'full-result';
	};
	b.build_coefficient_matrix = function () {
		b.calls.push('build_coefficient_matrix');
		return null;
	};
	return b;
}

// Run a child process with PAR3_ROUTING_DEBUG=1 and capture stdout; verifies
// the gate emits the verbose log line on every call.
function runWithRoutingDebug(envOverrides, source) {
	var cp = require('node:child_process');
	var childSrc = source;
	var env = Object.assign({}, process.env, envOverrides);
	var r = cp.spawnSync(process.execPath, ['-e', childSrc], { env: env, encoding: 'utf8' });
	return { status: r.status, stdout: r.stdout || '', stderr: r.stderr || '' };
}

// ============================================================================
// Section 1 — decision table (default env, all kernels available)
// ============================================================================

console.log('--- Section 1: decision table (default env, all kernels available) ---');

function isPow2OrTrivial(n) {
	return n === 0 || n === 1 || (n > 1 && (n & (n - 1)) === 0);
}

// Clear any inherited env that could perturb the routing decision.
var prevUseFenger = process.env.PAR3_GF64_USE_FENGER;
var prevMinR = process.env.PAR3_FENGER_MIN_R;
delete process.env.PAR3_GF64_USE_FENGER;
delete process.env.PAR3_FENGER_MIN_R;

try {
	var binding = makeBinding();

	// Case 1: small shape — matrix=1 MiB, R=128 < FENGER_MIN_R=8192, matvec wins.
	var d1 = decideRecoveryKernel(1024, 128, 4096, binding);
	check(d1.kernel === 'matvec',
		'N=1024,R=128,B=4096 -> matvec (got: ' + d1.kernel + ', reason: ' + d1.reason + ')');
	check(d1.matvecFeasible === true,
		'N=1024,R=128,B=4096: matvec feasible (matrix=1 MiB <= 128 MiB cap)');

	// Case 2: matrix infeasible (1 GiB > 128 MiB), R=4096 pow2 — cost-model
	// picks fenger via the matvec-infeasible override. Assert determinism
	// (same answer across runs) and that the answer is one of the two
	// acceptable values (fenger or matvec).
	var d2a = decideRecoveryKernel(32768, 4096, 4096, binding);
	var d2b = decideRecoveryKernel(32768, 4096, 4096, binding);
	var d2c = decideRecoveryKernel(32768, 4096, 4096, binding);
	check(d2a.kernel === d2b.kernel && d2b.kernel === d2c.kernel,
		'N=32768,R=4096,B=4096 deterministic across 3 runs: ' + d2a.kernel + ' (a=' + d2a.kernel + ' b=' + d2b.kernel + ' c=' + d2c.kernel + ')');
	check(d2a.kernel === 'fenger' || d2a.kernel === 'matvec',
		'N=32768,R=4096,B=4096 -> fenger|matvec (got: ' + d2a.kernel + ', reason: ' + d2a.reason + ')');
	check(d2a.matvecFeasible === false,
		'N=32768,R=4096,B=4096: matvec infeasible (matrix=1 GiB > 128 MiB cap; got feasible=' + d2a.matvecFeasible + ')');

	// Case 3: large-pow2 bench geometry — R=8192 >= FENGER_MIN_R=8192.
	var d3 = decideRecoveryKernel(65536, 8192, 4096, binding);
	check(d3.kernel === 'fenger',
		'N=65536,R=8192,B=4096 -> fenger (R >= FENGER_MIN_R; got: ' + d3.kernel + ', reason: ' + d3.reason + ')');

	// Case 4: extreme large-pow2 — R=32768 well above FENGER_MIN_R.
	var d4 = decideRecoveryKernel(262144, 32768, 4096, binding);
	check(d4.kernel === 'fenger',
		'N=262144,R=32768,B=4096 -> fenger (got: ' + d4.kernel + ', reason: ' + d4.reason + ')');

	// Case 5: non-pow2 R at large N — fenger requires pow2 R, matvec
	// infeasible (matrix=209 MiB > 128 MiB), so barycentric wins.
	var d5 = decideRecoveryKernel(262145, 100, 4096, binding);
	check(d5.kernel === 'barycentric',
		'N=262145,R=100 (non-pow2),B=4096 -> barycentric (got: ' + d5.kernel + ', reason: ' + d5.reason + ')');
	check(d5.fengerEligible === false,
		'N=262145,R=100: fenger ineligible (R not pow2)');

	// Case 6: non-pow2 R at small N — matvec feasible (matrix fits), so
	// cost-model still picks matvec (lower priority than fenger when
	// fenger is ineligible, but cheaper than bary by tie-break).
	// N=1024, R=100: matrix=800 KiB; bary ineligible (N <= BARY_MIN).
	var d6 = decideRecoveryKernel(1024, 100, 4096, binding);
	check(d6.kernel === 'matvec',
		'N=1024,R=100,B=4096 -> matvec (matrix fits, bary ineligible; got: ' + d6.kernel + ')');

	// Case 7: deterministic ordering — many shapes, multiple runs.
	console.log('\n--- Section 1b: 10-run determinism (all bench-shape shapes) ---');
	var determinismShapes = [
		{ N: 1024, R: 128, B: 4096 },
		{ N: 1024, R: 100, B: 4096 },
		{ N: 32768, R: 4096, B: 4096 },
		{ N: 65536, R: 8192, B: 4096 },
		{ N: 131072, R: 16384, B: 8192 },
		{ N: 262144, R: 32768, B: 4096 },
		{ N: 262144, R: 100, B: 4096 },
		{ N: 262145, R: 100, B: 4096 }
	];
	var stable = true;
	for (var si = 0; si < determinismShapes.length; si++) {
		var s = determinismShapes[si];
		var first = decideRecoveryKernel(s.N, s.R, s.B, binding).kernel;
		for (var run = 1; run < 10; run++) {
			var next = decideRecoveryKernel(s.N, s.R, s.B, binding).kernel;
			if (next !== first) {
				stable = false;
				check(false, 'N=' + s.N + ',R=' + s.R + ',B=' + s.B + ' deterministic (run ' + run + ': ' + next + ' vs first: ' + first + ')');
				break;
			}
		}
		if (stable) {
			check(true, 'N=' + s.N + ',R=' + s.R + ',B=' + s.B + ' deterministic across 10 runs: ' + first);
		}
	}
} finally {
	if (prevUseFenger === undefined) delete process.env.PAR3_GF64_USE_FENGER;
	else process.env.PAR3_GF64_USE_FENGER = prevUseFenger;
	if (prevMinR === undefined) delete process.env.PAR3_FENGER_MIN_R;
	else process.env.PAR3_FENGER_MIN_R = prevMinR;
}

// ============================================================================
// Section 2 — env-override matrix (PAR3_FENGER_MIN_R, PAR3_GF64_USE_FENGER)
// ============================================================================

console.log('\n--- Section 2: env-override matrix ---');

function withEnv(overrides, fn) {
	var prev = {};
	Object.keys(overrides).forEach(function (k) {
		prev[k] = process.env[k];
		if (overrides[k] === undefined || overrides[k] === null) delete process.env[k];
		else process.env[k] = String(overrides[k]);
	});
	try {
		return fn();
	} finally {
		Object.keys(prev).forEach(function (k) {
			if (prev[k] === undefined) delete process.env[k];
			else process.env[k] = prev[k];
		});
	}
}

// PAR3_FENGER_MIN_R lower threshold: R=2048 with FENGER_MIN_R=1024 picks fenger.
withEnv({ PAR3_FENGER_MIN_R: '1024', PAR3_GF64_USE_FENGER: undefined }, function () {
	var d = decideRecoveryKernel(262144, 2048, 4096, makeBinding());
	check(d.kernel === 'fenger',
		'PAR3_FENGER_MIN_R=1024 lowers the Fenger floor (N=262144,R=2048 -> ' + d.kernel + ')');
});

// PAR3_FENGER_MIN_R raise threshold: R=8192 with FENGER_MIN_R=16384 does NOT
// pick fenger by cost; matrix is also infeasible (R*N*8 = 16 GiB > 128 MiB).
// With the matrix-infeasible override, fenger becomes eligible at R=8192.
withEnv({ PAR3_FENGER_MIN_R: '16384', PAR3_GF64_USE_FENGER: undefined }, function () {
	var d = decideRecoveryKernel(65536, 8192, 4096, makeBinding());
	check(d.kernel === 'fenger',
		'matrix-infeasible override engages fenger even when FENGER_MIN_R=16384 (N=65536,R=8192 -> ' + d.kernel + ')');
});

// PAR3_GF64_USE_FENGER=1 forces fenger.
withEnv({ PAR3_GF64_USE_FENGER: '1', PAR3_FENGER_MIN_R: undefined }, function () {
	var d = decideRecoveryKernel(1024, 128, 4096, makeBinding());
	check(d.kernel === 'fenger',
		'PAR3_GF64_USE_FENGER=1 forces fenger (N=1024,R=128 normally -> matvec; got: ' + d.kernel + ')');
});

// PAR3_GF64_USE_FENGER=0 disables fenger — falls through to next-best.
withEnv({ PAR3_GF64_USE_FENGER: '0', PAR3_FENGER_MIN_R: undefined }, function () {
	var d = decideRecoveryKernel(65536, 8192, 4096, makeBinding());
	check(d.kernel === 'barycentric',
		'PAR3_GF64_USE_FENGER=0 disables fenger (N=65536,R=8192 -> ' + d.kernel + ' instead of fenger)');
});

// ============================================================================
// Section 3 — cost-model invariants (monotonicity, fenger override)
// ============================================================================

console.log('\n--- Section 3: cost-model invariants ---');

// Invariant 1: for a fixed (R, blockSize), increasing N monotonically
// increases the matvec cost by at least blockSize+8 per row.
withEnv({ PAR3_FENGER_MIN_R: undefined, PAR3_GF64_USE_FENGER: undefined }, function () {
	var prevCost = -1;
	var monotonic = true;
	var prevN = -1;
	for (var N = 1; N <= 65536; N *= 2) {
		var d = decideRecoveryKernel(N, 128, 4096, makeBinding());
		if (d.costs.matvec < prevCost) {
			monotonic = false;
			check(false, 'matvec cost monotonic increasing in N (N=' + N + ' got ' + d.costs.matvec + ' < prev ' + prevCost + ')');
			break;
		}
		prevCost = d.costs.matvec;
		prevN = N;
	}
	if (monotonic) {
		check(true, 'matvec cost monotonic increasing in N for R=128 (8 points, N=1..65536)');
	}
});

// Invariant 2: matrix infeasibility always enables the fenger override
// (regardless of FENGER_MIN_R), so long as R is pow2 and binding has fenger.
withEnv({ PAR3_FENGER_MIN_R: '1000000', PAR3_GF64_USE_FENGER: undefined }, function () {
	var d = decideRecoveryKernel(262144, 32768, 4096, makeBinding());
	check(d.fengerEligible === true && d.kernel === 'fenger',
		'matrix-infeasible override enables fenger even with FENGER_MIN_R=1000000 (got kernel=' + d.kernel + ', eligible=' + d.fengerEligible + ')');
});

// Invariant 3: matrix footprint exactly matches the documented formula.
withEnv({ PAR3_FENGER_MIN_R: undefined, PAR3_GF64_USE_FENGER: undefined }, function () {
	var d = decideRecoveryKernel(1024, 128, 4096, makeBinding());
	check(d.matrixBytes === 1024 * 128 * 8,
		'matrixBytes = R*N*8 (got ' + d.matrixBytes + ' for N=1024,R=128)');
});

// Invariant 4: returns a deterministic result object shape.
withEnv({ PAR3_FENGER_MIN_R: undefined, PAR3_GF64_USE_FENGER: undefined }, function () {
	var d = decideRecoveryKernel(1024, 128, 4096, makeBinding());
	var keys = Object.keys(d).sort();
	var expected = ['baryEligible', 'costs', 'fengerEligible', 'kernel', 'matvecFeasible', 'matrixBytes', 'reason'].sort();
	check(JSON.stringify(keys) === JSON.stringify(expected),
		'result object has the documented schema (got: ' + keys.join(',') + ')');
});

// ============================================================================
// Section 4 — PAR3_ROUTING_DEBUG verbose log
// ============================================================================

console.log('\n--- Section 4: PAR3_ROUTING_DEBUG verbose log ---');

(function () {
	var src = [
		'"use strict";',
		'var par3gen = require(' + JSON.stringify(par3genPath) + ');',
		'var binding = {',
		'  compute_recovery_fenger: function () { return "f"; },',
		'  compute_recovery_barycentric: function () { return "b"; },',
		'  compute_recovery_with_coeff: function () { return "m"; },',
		'  compute_recovery_full: function () { return "x"; },',
		'  build_coefficient_matrix: function () { return null; }',
		'};',
		'par3gen.decideRecoveryKernel(1024, 128, 4096, binding);',
		'par3gen.decideRecoveryKernel(65536, 8192, 4096, binding);',
		'console.log("CHILD_DONE");'
	].join('\n');
	var r = runWithRoutingDebug({ PAR3_ROUTING_DEBUG: '1' }, src);
	var matvecLine = /\[par3gen\] routing shape N=1024,R=128,B=4096 -> matvec \(reason: cost model\)/.test(r.stdout);
	var fengerLine = /\[par3gen\] routing shape N=65536,R=8192,B=4096 -> fenger \(reason: cost model\)/.test(r.stdout);
	check(r.status === 0 && /CHILD_DONE/.test(r.stdout),
		'PAR3_ROUTING_DEBUG=1 child process exits 0 (status=' + r.status + ')');
	check(matvecLine,
		'PAR3_ROUTING_DEBUG=1 emits matvec log line for {N=1024,R=128,B=4096}');
	check(fengerLine,
		'PAR3_ROUTING_DEBUG=1 emits fenger log line for {N=65536,R=8192,B=4096}');
})();

// Without PAR3_ROUTING_DEBUG: no verbose log line emitted (silent).
(function () {
	var src = [
		'"use strict";',
		'var par3gen = require(' + JSON.stringify(par3genPath) + ');',
		'var binding = {',
		'  compute_recovery_fenger: function () { return "f"; },',
		'  compute_recovery_barycentric: function () { return "b"; },',
		'  compute_recovery_with_coeff: function () { return "m"; },',
		'  compute_recovery_full: function () { return "x"; },',
		'  build_coefficient_matrix: function () { return null; }',
		'};',
		'par3gen.decideRecoveryKernel(1024, 128, 4096, binding);',
		'par3gen.decideRecoveryKernel(65536, 8192, 4096, binding);',
		'console.log("CHILD_DONE");'
	].join('\n');
	var r = runWithRoutingDebug({ PAR3_ROUTING_DEBUG: undefined }, src);
	var hasRoutingLine = /\[par3gen\] routing shape/.test(r.stdout);
	check(r.status === 0 && /CHILD_DONE/.test(r.stdout),
		'PAR3_ROUTING_DEBUG unset -> child process exits 0 (status=' + r.status + ')');
	check(!hasRoutingLine,
		'PAR3_ROUTING_DEBUG unset -> no [par3gen] routing shape lines emitted');
})();

// ============================================================================
// Section 5 — non-regression: dispatchRecovery contract unchanged
// ============================================================================

console.log('\n--- Section 5: dispatchRecovery contract unchanged ---');

// The gate is a separate code path. dispatchRecovery's own behavior must
// remain identical to pre-T1 (the existing par3-fenger-gate-dispatch.js
// suite pins it). Here we just spot-check a few invariants via the
// exported dispatchRecovery.
withEnv({ PAR3_GF64_USE_FENGER: undefined, PAR3_FENGER_MIN_R: undefined }, function () {
	var binding = makeBinding();
	// N=2^18, R=8192 -> dispatchRecovery picks fenger (R >= FENGER_MIN_R).
	var result = dispatchRecovery(binding, null, null, 262144, 8192, 4096, 0, 0n, 0);
	check(binding.calls[binding.calls.length - 1] === 'fenger' && result === 'fenger-result',
		'dispatchRecovery N=2^18,R=8192 still routes to fenger (R >= FENGER_MIN_R; got: ' + binding.calls[binding.calls.length - 1] + ')');
});

// ============================================================================

console.log('\n' + passed + ' passed, ' + failed + ' failed, ' + skipped + ' skipped');
if (failed) {
	console.error('Failures:');
	failures.forEach(function (f) { console.error('  - ' + f); });
	process.exit(1);
}
process.exit(0);
