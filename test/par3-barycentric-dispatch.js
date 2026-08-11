#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 Barycentric Dispatch Test (T12 - par3-cauchy-fft-kernel)
//
// Verifies the dispatch logic in isolation. Injects a mock binding so the
// test does not require a compiled native addon. Tests the four gating
// conditions:
//
//   1. N=64 (small)              → compute_recovery_full
//   2. N=20000 (above threshold) → compute_recovery_barycentric
//   3. PAR3_BF64_MIN_INPUTS=50000, N=20000 → compute_recovery_full
//   4. blockSize % 8 != 0        → compute_recovery_full
//
// Plus env-threshold edge cases (PAR3_BF64_MIN_INPUTS unset, 0, negative,
// non-numeric) and the explicit fallback path for non-single-call sites.
// ============================================================================

var path = require('path');
var assert = require('node:assert');

console.log('PAR3 Barycentric Dispatch Test (T12)');
console.log('=====================================\n');

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

// Fresh mock binding per scenario. Track which kernel was called.
// Methods close over `binding.calls` (not `this`) so the dispatcher can call
// them as plain functions — strict-mode would otherwise lose the receiver.
function makeBinding() {
    var binding = { calls: [] };
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
    return binding;
}

// Snapshot and restore PAR3_BF64_MIN_INPUTS around each scenario so the
// module-level env read inside dispatchRecovery sees the expected value.
function withEnv(value, fn) {
    var prev = process.env.PAR3_BF64_MIN_INPUTS;
    if (value === undefined || value === null) {
        delete process.env.PAR3_BF64_MIN_INPUTS;
    } else {
        process.env.PAR3_BF64_MIN_INPUTS = String(value);
    }
    try {
        return fn();
    } finally {
        if (prev === undefined) delete process.env.PAR3_BF64_MIN_INPUTS;
        else process.env.PAR3_BF64_MIN_INPUTS = prev;
    }
}

console.log('--- Test 1: N=64 (small) → existing 2D-muladd path ---');
withEnv(undefined, function () {
    var b = makeBinding();
    var result = dispatchRecovery(b, null, null, 64, 1, 4096, 0, 0n, 0);
    check(result === 'full-result', 'returned compute_recovery_full result (got: ' + JSON.stringify(result) + ')');
    check(b.calls.length === 1, 'exactly one kernel call (got: ' + b.calls.length + ')');
    check(b.calls[0] && b.calls[0].kernel === 'full', 'kernel was compute_recovery_full (got: ' + (b.calls[0] && b.calls[0].kernel) + ')');
});

console.log('\n--- Test 2: N=20000 (above threshold) → Barycentric path ---');
withEnv(undefined, function () {
    var b = makeBinding();
    var result = dispatchRecovery(b, null, null, 20000, 100, 4096, 0, 20000n, 4);
    check(result === 'barycentric-result', 'returned Barycentric result (got: ' + JSON.stringify(result) + ')');
    check(b.calls.length === 1, 'exactly one kernel call (got: ' + b.calls.length + ')');
    check(b.calls[0] && b.calls[0].kernel === 'barycentric', 'kernel was compute_recovery_barycentric (got: ' + (b.calls[0] && b.calls[0].kernel) + ')');
    check(b.calls[0] && b.calls[0].args.numInputs === 20000, 'forwarded numInputs=20000');
    check(b.calls[0] && b.calls[0].args.numRecovery === 100, 'forwarded numRecovery=100');
    check(b.calls[0] && b.calls[0].args.blockSize === 4096, 'forwarded blockSize=4096');
});

// Helper to stringify call records safely (BigInt args can't go through JSON.stringify).
function callsToString(calls) {
    return calls.map(function (c) {
        return c.kernel + '(' + Object.keys(c.args).map(function (k) {
            var v = c.args[k];
            return k + '=' + (typeof v === 'bigint' ? v.toString() + 'n' : v);
        }).join(', ') + ')';
    }).join('; ');
}

console.log('\n--- Test 3: PAR3_BF64_MIN_INPUTS=50000, N=20000 → 2D-muladd fallback ---');
withEnv(50000, function () {
    var b = makeBinding();
    var result = dispatchRecovery(b, null, null, 20000, 1, 4096, 0, 0n, 0);
    check(result === 'full-result', 'returned compute_recovery_full result (got: ' + JSON.stringify(result) + ')');
    check(b.calls.length === 1 && b.calls[0].kernel === 'full', 'kernel was compute_recovery_full (got: ' + callsToString(b.calls) + ')');
});

console.log('\n--- Test 4: blockSize % 8 != 0 → Barycentric rejected, 2D-muladd selected ---');
withEnv(undefined, function () {
    var b = makeBinding();
    var result = dispatchRecovery(b, null, null, 20000, 1, 4097, 0, 0n, 0);
    check(result === 'full-result', 'returned compute_recovery_full result (got: ' + JSON.stringify(result) + ')');
    check(b.calls.length === 1 && b.calls[0].kernel === 'full', 'kernel was compute_recovery_full (got: ' + callsToString(b.calls) + ')');
});

console.log('\n--- Test 5: numRecovery=0 → Barycentric rejected, 2D-muladd selected ---');
withEnv(undefined, function () {
    var b = makeBinding();
    var result = dispatchRecovery(b, null, null, 20000, 0, 4096, 0, 0n, 0);
    check(result === 'full-result', 'returned compute_recovery_full result (got: ' + JSON.stringify(result) + ')');
    check(b.calls.length === 1 && b.calls[0].kernel === 'full', 'kernel was compute_recovery_full');
});

console.log('\n--- Test 6: Barycentric absent in binding → fallback to compute_recovery_full ---');
withEnv(undefined, function () {
    var b = makeBinding();
    delete b.compute_recovery_barycentric;
    var result = dispatchRecovery(b, null, null, 20000, 1, 4096, 0, 0n, 0);
    check(result === 'full-result', 'returned compute_recovery_full result');
    check(b.calls.length === 1 && b.calls[0].kernel === 'full', 'kernel was compute_recovery_full');
});

console.log('\n--- Test 7: Threshold at N=10000 boundary (N=10001 → Barycentric, N=10000 → full) ---');
withEnv(undefined, function () {
    var b1 = makeBinding();
    dispatchRecovery(b1, null, null, 10001, 1, 4096, 0, 0n, 0);
    check(b1.calls[0].kernel === 'barycentric', 'N=10001 dispatches to Barycentric (strict > threshold)');

    var b2 = makeBinding();
    dispatchRecovery(b2, null, null, 10000, 1, 4096, 0, 0n, 0);
    check(b2.calls[0].kernel === 'full', 'N=10000 dispatches to full (NOT strictly greater)');
});

console.log('\n--- Test 8: Explicit fallbackFn (per-batch path pattern) ---');
withEnv(undefined, function () {
    var b = makeBinding();
    var fallback = b.compute_recovery.bind(b);
    var result = dispatchRecovery(b, null, null, 64, 1, 4096, 0, 0n, 0, fallback);
    check(result === 'per-batch-result', 'returned per-batch fallback result (got: ' + JSON.stringify(result) + ')');
    check(b.calls.length === 1 && b.calls[0].kernel === 'per-batch', 'kernel was per-batch fallback');
});

console.log('\n--- Test 9: Explicit fallbackFn wins over compute_recovery_full when Barycentric rejects ---');
withEnv(undefined, function () {
    var b = makeBinding();
    var fallback = b.compute_recovery.bind(b);
    var result = dispatchRecovery(b, null, null, 20000, 1, 4097, 0, 0n, 0, fallback);
    check(result === 'per-batch-result', 'returned per-batch fallback result on blockSize%8!=0 path');
    check(b.calls.length === 1 && b.calls[0].kernel === 'per-batch', 'kernel was per-batch fallback (NOT compute_recovery_full)');
});

console.log('\n--- Test 10: Env threshold edge cases (0, negative, non-numeric → default 10000) ---');
withEnv('0', function () {
    var b = makeBinding();
    dispatchRecovery(b, null, null, 10001, 1, 4096, 0, 0n, 0);
    check(b.calls[0].kernel === 'barycentric', 'PAR3_BF64_MIN_INPUTS=0 falls back to default 10000 → N=10001 dispatches to Barycentric');
});
withEnv('-5', function () {
    var b = makeBinding();
    dispatchRecovery(b, null, null, 10001, 1, 4096, 0, 0n, 0);
    check(b.calls[0].kernel === 'barycentric', 'PAR3_BF64_MIN_INPUTS=-5 falls back to default 10000 → N=10001 dispatches to Barycentric');
});
withEnv('garbage', function () {
    var b = makeBinding();
    dispatchRecovery(b, null, null, 10001, 1, 4096, 0, 0n, 0);
    check(b.calls[0].kernel === 'barycentric', 'PAR3_BF64_MIN_INPUTS=garbage falls back to default 10000 → N=10001 dispatches to Barycentric');
});

console.log('\n--- Test 11: No kernels available + no fallbackFn → throws ---');
withEnv(undefined, function () {
    var threw = null;
    try {
        dispatchRecovery({}, null, null, 20000, 1, 4096, 0, 0n, 0);
    } catch (e) {
        threw = e;
    }
    check(threw !== null, 'throws when no fallback kernel is available');
    check(threw && /dispatchRecovery/.test(threw.message), 'error message identifies the dispatcher');
});

console.log('\n--- Test 12: BARY_MIN_INPUTS_DEFAULT constant is exposed and is 10000 ---');
check(par3gen.BARY_MIN_INPUTS_DEFAULT === 10000, 'BARY_MIN_INPUTS_DEFAULT === 10000 (got: ' + par3gen.BARY_MIN_INPUTS_DEFAULT + ')');

// ============================================================================
// Fenger dispatch tests (issue #46 K3): the Fenger pipeline is the DEFAULT
// for power-of-2 workloads. PAR3_GF64_USE_FENGER=0 kills it; =1 forces it
// (and errors if the binding lacks the kernel).
// ============================================================================

function makeFengerBinding() {
    var b = makeBinding();
    b.compute_recovery_fenger = function (a, b2, c, d, e, f, g, h) {
        b.calls.push({ kernel: 'fenger', args: { numInputs: c, numRecovery: d, blockSize: e, firstInput: f, firstRecovery: g, numThreads: h } });
        return 'fenger-result';
    };
    return b;
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

console.log('\n--- Test 13: power-of-2 workload, env unset → Fenger by default ---');
withFengerEnv(undefined, function () {
    var b = makeFengerBinding();
    var result = dispatchRecovery(b, null, null, 64, 4, 4096, 0, 0n, 0);
    check(result === 'fenger-result', 'returned Fenger result by default (got: ' + JSON.stringify(result) + ')');
    check(b.calls.length === 1 && b.calls[0].kernel === 'fenger', 'kernel was compute_recovery_fenger (got: ' + callsToString(b.calls) + ')');
});

console.log('\n--- Test 14: power-of-2 workload, PAR3_GF64_USE_FENGER=0 → kill switch, skip Fenger ---');
withFengerEnv('0', function () {
    var b = makeFengerBinding();
    var result = dispatchRecovery(b, null, null, 64, 4, 4096, 0, 0n, 0);
    check(result === 'full-result', 'returned compute_recovery_full (got: ' + JSON.stringify(result) + ')');
    check(b.calls.length === 1 && b.calls[0].kernel === 'full', 'kernel was compute_recovery_full, NOT fenger (got: ' + callsToString(b.calls) + ')');
});
withFengerEnv('false', function () {
    var b = makeFengerBinding();
    dispatchRecovery(b, null, null, 64, 4, 4096, 0, 0n, 0);
    check(b.calls.length === 1 && b.calls[0].kernel === 'full', 'PAR3_GF64_USE_FENGER=false also kills Fenger (got: ' + callsToString(b.calls) + ')');
});

console.log('\n--- Test 15: power-of-2 workload, PAR3_GF64_USE_FENGER=1 → forced Fenger ---');
withFengerEnv('1', function () {
    var b = makeFengerBinding();
    var result = dispatchRecovery(b, null, null, 64, 4, 4096, 0, 0n, 0);
    check(result === 'fenger-result', 'returned Fenger result (got: ' + JSON.stringify(result) + ')');
    check(b.calls.length === 1 && b.calls[0].kernel === 'fenger', 'kernel was compute_recovery_fenger');
});

console.log('\n--- Test 16: PAR3_GF64_USE_FENGER=1 but binding lacks Fenger → throws ---');
withFengerEnv('1', function () {
    var threw = null;
    try {
        dispatchRecovery(makeBinding(), null, null, 64, 4, 4096, 0, 0n, 0);
    } catch (e) {
        threw = e;
    }
    check(threw !== null, 'throws when Fenger is forced but binding lacks it');
    check(threw && /compute_recovery_fenger/.test(threw.message), 'error names the missing binding entry');
});

// Tests 17/18 originally expected Fenger via K5 padding to handle
// non-power-of-2 workloads (20000 → 32768, 3 → 4). The dispatch gate in
// lib/par3gen.js gates Fenger on the literal `(numInputs | numRecovery) == 0
// || is_power_of_2(...)` predicate (line 168); K5 padding happens INSIDE
// compute_recovery_fenger, not inside the dispatch selection. The
// pre-conditions at this layer mean 20000/3 fall through:
//   - Test 17: N=20000 > BARY_MIN_INPUTS_DEFAULT (10000) → Barycentric wins.
//   - Test 18: R=3 is below both the small-R shortcut (32) AND the Fenger
//     gate's power-of-2 check (3 is non-zero, non-trivial, non-power-of-2)
//     → falls all the way through to compute_recovery_full.
// The previous test expectations were aspirational and pre-emptively encoded
// the not-yet-implemented K5 *dispatch-level* gate. Cubic review 4904130419
// (item 1) flagged the misleading expectation; the suite now asserts the
// actual behaviour.
console.log('\n--- Test 17: non-power-of-2 N=20000, env unset → Barycentric (N > BARY_MIN) ---');
withFengerEnv(undefined, function () {
    var b = makeFengerBinding();
    dispatchRecovery(b, null, null, 20000, 100, 4096, 0, 20000n, 4);
    check(b.calls.length === 1 && b.calls[0].kernel === 'barycentric',
        'N=20000 (non-power-of-2, >BARY_MIN) dispatches to Barycentric (got: ' + callsToString(b.calls) + ')');
});

console.log('\n--- Test 18: non-power-of-2 R=3, env unset → full (small-R shortcut, Fenger gate rejects non-power-of-2 R) ---');
withFengerEnv(undefined, function () {
    var b = makeFengerBinding();
    dispatchRecovery(b, null, null, 64, 3, 4096, 0, 0n, 0);
    check(b.calls.length === 1 && b.calls[0].kernel === 'full',
        'numRecovery=3 (non-power-of-2, small) dispatches to full (got: ' + callsToString(b.calls) + ')');
});

console.log('\n--- Test 19: blockSize % 8 != 0, env unset → Fenger rejected even for power-of-2 ---');
withFengerEnv(undefined, function () {
    var b = makeFengerBinding();
    dispatchRecovery(b, null, null, 64, 4, 4097, 0, 0n, 0);
    check(b.calls.length === 1 && b.calls[0].kernel === 'full', 'blockSize=4097 (%8!=0) dispatches to full, NOT Fenger (got: ' + callsToString(b.calls) + ')');
});

console.log('\n--- Test 20: numInputs=0 (trivial), env unset → Fenger (0 is power-of-2-or-trivial) ---');
withFengerEnv(undefined, function () {
    var b = makeFengerBinding();
    dispatchRecovery(b, null, null, 0, 4, 4096, 0, 0n, 0);
    check(b.calls.length === 1 && b.calls[0].kernel === 'fenger', 'numInputs=0 (trivial) dispatches to Fenger (got: ' + callsToString(b.calls) + ')');
});

// ============================================================================
// Summary
// ============================================================================
console.log('\n---');
if (failed > 0) {
    console.log('FAILED (' + failed + ' failure(s), ' + passed + ' passed, ' + total + ' total)');
    failures.forEach(function (f) { console.error('  - ' + f); });
    process.exit(1);
} else {
    console.log('PASS (' + passed + ' passed, ' + total + ' total)');
    console.log('Barycentric dispatch logic verified across all gating conditions.');
}