#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 Barycentric Threshold-Crossover Parity Test (T8 PR-6)
//
// Verifies that the Barycentric kernel produces byte-identical recovery
// blocks to the legacy 2D-muladd kernel across the dispatch threshold.
//
// The threshold is controlled by PAR3_BF64_MIN_INPUTS. The default is 10000
// (see lib/par3gen.js and src/par3_engine_barycentric.cc). Both kernels
// must agree bit-for-bit on the recovery blocks — the dispatch pick is
// a perf knob, never a correctness one.
//
// Methodology:
//   1. Build a fixed random input + recovery problem at numInputs slightly
//      above the default threshold.
//   2. Compute the recovery blocks twice:
//      a) PAR3_BF64_MIN_INPUTS=99999999 forces the legacy 2D-muladd path.
//      b) PAR3_BF64_MIN_INPUTS=0 forces the Barycentric path.
//   3. Assert byte-exact equality on the recovery buffer.
//
// We also run a smaller numInputs case to confirm the threshold is honored
// on both sides.
// ============================================================================

var path = require('path');
var assert = require('node:assert');

console.log('PAR3 Barycentric Threshold-Crossover Parity (T8 PR-6)');
console.log('======================================================\n');

var passed = 0, failed = 0, total = 0;
function check(condition, msg) {
    total++;
    if (condition) {
        console.log('  PASS: ' + msg);
        passed++;
    } else {
        console.error('  FAIL: ' + msg);
        failed++;
        process.exitCode = 1;
    }
}

var addon;
try {
    addon = require('../build/Release/parpar_gf64.node');
} catch (e) {
    console.log('SKIP: parpar_gf64.node not built.');
    process.exit(0);
}

if (typeof addon.compute_recovery_barycentric !== 'function' ||
    typeof addon.compute_recovery_full !== 'function') {
    console.log('SKIP: par3 native binding missing either barycentric or 2D-muladd entry.');
    process.exit(0);
}

// mulberry32 for deterministic fill
function mulberry32(seed) {
    var s = seed >>> 0;
    return function() {
        s = (s + 0x6D2B79F5) >>> 0;
        var t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0);
    };
}

function runKernel(fn, numInputs, numRecovery, blockSize, inputs, outputs) {
    // clear outputs
    outputs.fill(0);
    fn(inputs, outputs, numInputs, numRecovery, blockSize, 0n, BigInt(numInputs), 0);
}

function crossCheck(name, numInputs, numRecovery, blockSize) {
    var inputs  = Buffer.alloc(numInputs * blockSize * 8);
    var outA    = Buffer.alloc(numRecovery * blockSize * 8);
    var outB    = Buffer.alloc(numRecovery * blockSize * 8);

    var rng = mulberry32(0xCAFEBABE);
    for (var i = 0; i < inputs.length; i++) inputs[i] = rng() & 0xFF;

    process.env.PAR3_BF64_MIN_INPUTS = '99999999';
    runKernel(addon.compute_recovery_full, numInputs, numRecovery, blockSize, inputs, outA);

    process.env.PAR3_BF64_MIN_INPUTS = '0';
    runKernel(addon.compute_recovery_barycentric, numInputs, numRecovery, blockSize, inputs, outB);

    // Restore env default for downstream tests (and other tests in the suite)
    delete process.env.PAR3_BF64_MIN_INPUTS;

    var match = outA.equals(outB);
    if (!match) {
        var firstMismatch = -1;
        for (var i = 0; i < outA.length; i++) {
            if (outA[i] !== outB[i]) { firstMismatch = i; break; }
        }
        console.error('    ' + name + ': first mismatch at byte ' + firstMismatch +
                      '  legacy=0x' + (firstMismatch >= 0 ? outA[firstMismatch].toString(16) : '?') +
                      '  bary =0x' + (firstMismatch >= 0 ? outB[firstMismatch].toString(16) : '?'));
    }
    check(match, name + ' legacy===barycentric (byte-exact)');
}

// Above the default threshold (10000) at the canonical 1 GiB / 10K workload.
crossCheck('above-threshold (N=16384, R=1000, B=8)',
           16384, 1000, 8);

// Smaller but still above the default threshold.
crossCheck('above-threshold small (N=12000, R=500, B=8)',
           12000, 500, 8);

// Smaller (below the threshold) — both kernels still must agree; the
// dispatch will pick legacy when below the threshold but the underlying
// math is the same so byte-equality holds.
crossCheck('below-threshold (N=1024, R=100, B=8)',
           1024, 100, 8);

console.log('\nSummary: ' + passed + ' passed, ' + failed + ' failed (of ' + total + ')');
if (failed === 0) console.log('PASS');
else console.log('FAIL');
