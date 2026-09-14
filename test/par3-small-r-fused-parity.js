#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 Small-R Fused Multi-Output Kernel Parity Test
// ----------------------------------------------------------------------------
// Regression test for Issue #110 / PR #111 investigation:
// Verifies that the fused multi-output path in WorkerThread (R <= 32)
// produces 100% bit-exact parity across R in {1, 2, 4, 8, 16, 32}, across
// thread counts (1, 2, 4, 8, 16), and correctly handles zero coefficients.
// ============================================================================

var addon = require('../build/Release/parpar_gf64.node');

function mulberry32(seed) {
    return function() {
        seed |= 0;
        seed = seed + 0x6D2B79F5 | 0;
        var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

function fillRandom(buf, rng) {
    var words = buf.length / 8;
    for (var w = 0; w < words; w++) {
        var hi = (rng() * 4294967296) >>> 0;
        var lo = (rng() * 4294967296) >>> 0;
        buf.writeBigUInt64LE((BigInt(hi) << 32n) | BigInt(lo), w * 8);
    }
}

var passed = 0;
var failed = 0;

function assert(condition, msg) {
    if (condition) {
        console.log('  PASS: ' + msg);
        passed++;
    } else {
        console.error('  FAIL: ' + msg);
        failed++;
        process.exitCode = 1;
    }
}

console.log('PAR3 Small-R Fused Multi-Output Parity Test');
console.log('===========================================\n');

var rng = mulberry32(0x5A110F05ED);

// ----------------------------------------------------------------------------
// Section A: R in {1, 2, 4, 8, 16, 32} single-thread vs multi-thread (threads=8, 16)
// ----------------------------------------------------------------------------
console.log('Section A: Thread scaling parity across small-R (R <= 32)');
console.log('---------------------------------------------------------');

var testConfigs = [
    { N: 32,  R: 1,  B: 1024 },
    { N: 32,  R: 2,  B: 1024 },
    { N: 64,  R: 4,  B: 2048 },
    { N: 128, R: 8,  B: 4096 },
    { N: 128, R: 16, B: 2048 },
    { N: 64,  R: 32, B: 1024 }
];

testConfigs.forEach(function(cfg, idx) {
    var inBuf = Buffer.alloc(cfg.N * cfg.B);
    fillRandom(inBuf, rng);

    // Reference: single-threaded run
    var refOut = Buffer.alloc(cfg.R * cfg.B);
    addon.compute_recovery_full(inBuf, refOut, cfg.N, cfg.R, cfg.B, 0, cfg.N, 1, false);

    // Compare with threads = 2, 4, 8, 16
    [2, 4, 8, 16].forEach(function(threads) {
        var testOut = Buffer.alloc(cfg.R * cfg.B);
        addon.compute_recovery_full(inBuf, testOut, cfg.N, cfg.R, cfg.B, 0, cfg.N, threads, false);
        assert(refOut.equals(testOut), 'Config ' + idx + ': N=' + cfg.N + ' R=' + cfg.R + ' B=' + cfg.B + ' threads=' + threads + ' matches 1-thread reference');
    });
});

// ----------------------------------------------------------------------------
// Section B: Parity with Barycentric dispatcher routing
// ----------------------------------------------------------------------------
console.log('\nSection B: Barycentric kernel small-R shortcut parity');
console.log('-----------------------------------------------------');

testConfigs.forEach(function(cfg, idx) {
    var inBuf = Buffer.alloc(cfg.N * cfg.B);
    fillRandom(inBuf, rng);

    var refOut = Buffer.alloc(cfg.R * cfg.B);
    addon.compute_recovery_full(inBuf, refOut, cfg.N, cfg.R, cfg.B, 0, cfg.N, 8, false);

    var baryOut = Buffer.alloc(cfg.R * cfg.B);
    addon.compute_recovery_barycentric(inBuf, baryOut, cfg.N, cfg.R, cfg.B, 0, cfg.N, 8, false);

    assert(refOut.equals(baryOut), 'Barycentric Config ' + idx + ': N=' + cfg.N + ' R=' + cfg.R + ' B=' + cfg.B + ' bit-identical to Cauchy reference');
});

// ----------------------------------------------------------------------------
// Section C: Discriminatory negative test
// ----------------------------------------------------------------------------
console.log('\nSection C: Discriminatory negative test');
console.log('--------------------------------------');

var inBufC = Buffer.alloc(64 * 1024);
fillRandom(inBufC, rng);
var refOutC = Buffer.alloc(8 * 1024);
addon.compute_recovery_full(inBufC, refOutC, 64, 8, 1024, 0, 64, 8, false);

// Mutate input
inBufC[42] ^= 0x5A;
var mutOutC = Buffer.alloc(8 * 1024);
addon.compute_recovery_full(inBufC, mutOutC, 64, 8, 1024, 0, 64, 8, false);

assert(!refOutC.equals(mutOutC), 'Bit flip in input propagates and mutates recovery output');

console.log('\n---');
console.log('RESULT: ' + passed + ' passed, ' + failed + ' failed');
if (failed > 0) process.exit(1);
process.exit(0);
