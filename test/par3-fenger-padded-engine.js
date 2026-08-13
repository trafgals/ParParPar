#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 Fenger Padded-Engine Routing Test (issue #59 A2)
//
// The engine (GF64Controller::ComputeRecoveryBlocksFenger) now routes ANY
// workload through the Fenger pipeline: non-power-of-2 counts are padded
// with synthetic zero-data inputs at a base above both ranges. The padded
// route must be BIT-EXACT to the legacy Cauchy path (compute_recovery_full)
// at every shape — including odd N (padded inputs) and odd R (padded
// recovery rows, only the real ones written).
//
// Requires the native addon (build/Release/parpar_gf64.node) built from
// the A2 sources. Skips (exit 0) when the addon is absent.
// ============================================================================

var assert = require('node:assert');

console.log('PAR3 Fenger Padded-Engine Routing Test (issue #59 A2)');
console.log('======================================================\n');

var binding;
try {
	binding = require('../build/Release/parpar_gf64.node');
} catch (e) {
	console.log('  SKIP: native addon not built — padded engine routing test skipped');
	process.exit(0);
}

if (typeof binding.compute_recovery_fenger !== 'function' ||
    typeof binding.compute_recovery_full !== 'function') {
	console.log('  SKIP: binding lacks compute_recovery_fenger / compute_recovery_full');
	process.exit(0);
}

var passed = 0;
var failed = 0;
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

// Deterministic PRNG (splitmix64) — same family as the C tests.
var state = 0x123456789ABCDEF0n;
function next64() {
	state = (state + 0x9E3779B97F4A7C15n) & 0xFFFFFFFFFFFFFFFFn;
	var z = state;
	z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & 0xFFFFFFFFFFFFFFFFn;
	z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & 0xFFFFFFFFFFFFFFFFn;
	return z ^ (z >> 31n);
}

// One parity case: fenger (padded route where needed) vs a legacy
// Cauchy reference. blockSize must be a multiple of 8 (the binding
// enforces it); B is the word count per block (blockSize/8). refFn
// selects the reference binding — default compute_recovery_full, or
// compute_recovery when the engine's fallback target is the one under
// test (cubic review 5a3b44c9 P3: dedupes the fill/compare/report
// blocks the collision case used to duplicate).
function runCase(N, R, blockSize, firstInput, firstRecovery, label, refFn) {
	var B = blockSize / 8;
	var inputs = Buffer.allocUnsafe(N * blockSize);
	for (var i = 0; i < N * B; i++) {
		inputs.writeBigUInt64LE(next64(), i * 8);
	}
	var outFenger = Buffer.alloc(R * blockSize);
	var outFull = Buffer.alloc(R * blockSize);

	binding.compute_recovery_fenger(inputs, outFenger, N, R, blockSize, firstInput, firstRecovery, 0);
	(refFn || binding.compute_recovery_full)(inputs, outFull, N, R, blockSize, firstInput, firstRecovery, 0);

	var ok = outFenger.equals(outFull);
	if (!ok) {
		// Report the first differing word.
		for (var w = 0; w < R * B; w++) {
			if (outFenger.readBigUInt64LE(w * 8) !== outFull.readBigUInt64LE(w * 8)) {
				console.error('    first mismatch at word ' + w + ': fenger=0x' +
					outFenger.readBigUInt64LE(w * 8).toString(16) + ' ref=0x' +
					outFull.readBigUInt64LE(w * 8).toString(16));
				break;
			}
		}
	}
	check(ok, label + ' (N=' + N + ', R=' + R + ', B=' + B + '): fenger == legacy Cauchy bit-exact');
}

// Odd N (padded inputs), power-of-2 R.
runCase(1000, 512, 128, 0x10000, 0x1000000, 'odd N padded inputs, pow2 R');
// Odd N AND odd R (both padded; only real R rows written).
runCase(1000, 600, 128, 0x10000, 0x1000000, 'odd N and odd R (both padded)');
// Power-of-2 N with odd R (only R padded).
runCase(1024, 600, 128, 0x10000, 0x1000000, 'pow2 N, odd R (recovery padded)');
// Trivial edge: N=1.
runCase(1, 512, 128, 0x10000, 0x1000000, 'N=1 trivial');
// Collision fallback (cubic review f70a81ef P2 / f44ead49 P1): the
// recovery range [66300, 66812) overlaps the real input range
// [65536, 66536) while numRecovery=512 is a POWER OF 2 — no recovery
// padding, so the pre-fix guard (gated on padRecovery) would NOT have
// fired and Fenger would compute V(y_r) == 0 on the colliding points,
// emitting zero rows instead of the legacy rows. The fixed guard must
// fall back to the legacy path (compute_recovery), which is bit-exact
// with compute_recovery_full here. (cubic review c509dd2b P3: R=600 was
// itself padded and did not exercise the bypass; 5a3b44c9 P3: routed
// through runCase with the reference-binding parameter.)
runCase(1000, 512, 128, 0x10000, 66300,
	'overlapping input/recovery ranges fall back to legacy',
	binding.compute_recovery);
// blockSize above the 32-bit size_t range (cubic review 50f46d24 P1):
// on 32-bit hosts the (size_t) cast truncates to 0 and the size checks
// would divide by zero (native crash). The guard must reject cleanly
// with a RangeError on every host.
(function () {
	var b = Buffer.alloc(8);
	var threw = false;
	try {
		binding.compute_recovery_fenger(b, b, 1, 1, Math.pow(2, 32) + 8, 0x10000, 0x1000000, 0);
	} catch (e) {
		threw = (e instanceof RangeError) || (e instanceof TypeError);
	}
	check(threw, 'blockSize=2^32+8 rejected cleanly (no div-by-zero on 32-bit)');
})();
// Shift-overflow boundary (cubic review f70a81ef P2): a count above the
// largest representable power of two (2^63) must be rejected cleanly by
// the binding — never wrap the pad loop to zero and hang.
(function () {
	var bigIn = Buffer.alloc(8);
	var bigOut = Buffer.alloc(8);
	var threw = false;
	try {
		binding.compute_recovery_fenger(bigIn, bigOut, Math.pow(2, 63), 2, 8, 0x10000, 0x1000000, 0);
	} catch (e) {
		threw = (e instanceof RangeError) || (e instanceof TypeError);
	}
	check(threw, 'numInputs=2^63 rejected cleanly (no pad-loop wrap / hang)');
})();
// NOTE: the P1 gate shape (N=131072/R=4096 bit-exact vs Cauchy) is a
// bench item (gf64/test/bench_gf64_subquadratic gate_shape), not a unit
// test — the padded route at N=2^18+1 pads to 2^19 points and takes
// minutes on the WSL2 scalar-HQC path.

console.log('\n' + passed + ' passed, ' + failed + ' failed, ' + total + ' total');
if (failed) {
	console.error('Failures:');
	failures.forEach(function (f) { console.error('  - ' + f); });
	process.exit(1);
}
process.exit(0);
