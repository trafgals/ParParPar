#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 Input-Domain Parallel Parity Test
// ----------------------------------------------------------------------------
// Verifies that when numThreads > numRecovery, input-domain partitioning across
// all threads (threads 1..T-1 computing into thread-local buffers and merged
// via XOR reduction) produces bit-identical output to single-threaded execution.
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

console.log('PAR3 Input-Domain Parallel Parity Test');
console.log('======================================\n');

var rng = mulberry32(0x7800E3D);

// Test matrix: numRecovery < numThreads (e.g. R=2, R=4, R=8 with T=4, 8, 16)
var cases = [
    { N: 64, R: 2, B: 512, threads: [1, 2, 4, 8, 16] },
    { N: 256, R: 4, B: 1024, threads: [1, 2, 8, 16] },
    { N: 512, R: 8, B: 4096, threads: [1, 8, 12, 16] },
    { N: 1024, R: 8, B: 8192, threads: [1, 8, 16] } // mirrors 64GB benchmark shape with R=8 on 16-thread Zen4
];

cases.forEach(function(testCase, idx) {
    var N = testCase.N;
    var R = testCase.R;
    var B = testCase.B;

    var inputBuf = Buffer.alloc(N * B);
    fillRandom(inputBuf, rng);

    // Reference output computed with 1 thread
    var refOut = Buffer.alloc(R * B);
    addon.compute_recovery_full(inputBuf, refOut, N, R, B, 0, N, 1, false);

    testCase.threads.forEach(function(t) {
        var testOut = Buffer.alloc(R * B);
        testOut.fill(0x5A); // dirty buffer to check clean write

        addon.compute_recovery_full(inputBuf, testOut, N, R, B, 0, N, t, false);

        var eq = refOut.equals(testOut);
        assert(eq, 'Case ' + idx + ': N=' + N + ' R=' + R + ' B=' + B + ' threads=' + t + ' bit-identical to serial');
    });
});

console.log('\n---');
console.log('RESULT: ' + passed + ' passed, ' + failed + ' failed');
if (failed > 0) process.exit(1);
