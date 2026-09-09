#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 accumulate binding parity test
// ----------------------------------------------------------------------------
// Verifies that native compute_recovery_full and compute_recovery_barycentric
// with the optional 9th argument accumulate=true produce bit-identical results
// when chunking inputs across multiple passes vs. computing all inputs in one pass.
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

console.log('PAR3 Native Accumulate Binding Parity Test');
console.log('===========================================\n');

var rng = mulberry32(0x110BADC0DE);

var testConfigs = [
    { numInputs: 16, numRecovery: 2, blockSize: 64, numChunks: 2 },
    { numInputs: 32, numRecovery: 4, blockSize: 512, numChunks: 4 },
    { numInputs: 64, numRecovery: 8, blockSize: 4096, numChunks: 4 },
    { numInputs: 128, numRecovery: 8, blockSize: 16384, numChunks: 8 },
    { numInputs: 256, numRecovery: 16, blockSize: 4096, numChunks: 4 },
    { numInputs: 64, numRecovery: 32, blockSize: 1024, numChunks: 2 },
    { numInputs: 48, numRecovery: 34, blockSize: 512, numChunks: 3 } // across R=32 threshold
];

console.log('Section A: compute_recovery_full chunked accumulation');
console.log('-----------------------------------------------------\n');

testConfigs.forEach(function(cfg, idx) {
    var ni = cfg.numInputs;
    var nr = cfg.numRecovery;
    var bs = cfg.blockSize;
    var chunks = cfg.numChunks;
    var chunkSize = Math.floor(ni / chunks);

    var inputBuf = Buffer.alloc(ni * bs);
    fillRandom(inputBuf, rng);

    // 1. Single pass (reference)
    var singleOut = Buffer.alloc(nr * bs);
    addon.compute_recovery_full(inputBuf, singleOut, ni, nr, bs, 0, ni, 0, false);

    // 2. Chunked accumulation pass
    var chunkedOut = Buffer.alloc(nr * bs);
    chunkedOut.fill(0xAA); // dirty buffer to test first chunk zeroing when accumulate=false

    var inBlockIdx = 0;
    for (var c = 0; c < chunks; c++) {
        var startIdx = c * chunkSize;
        var endIdx = (c === chunks - 1) ? ni : (c + 1) * chunkSize;
        var count = endIdx - startIdx;
        var sub = inputBuf.subarray(startIdx * bs, endIdx * bs);
        var isAccum = (c > 0);

        addon.compute_recovery_full(sub, chunkedOut, count, nr, bs, startIdx, ni, 0, isAccum);
    }

    var eq = singleOut.equals(chunkedOut);
    assert(eq, 'Config ' + idx + ': N=' + ni + ' R=' + nr + ' B=' + bs + ' chunks=' + chunks + ' bit-identical');
});

console.log('\nSection B: compute_recovery_barycentric chunked accumulation');
console.log('------------------------------------------------------------\n');

testConfigs.forEach(function(cfg, idx) {
    var ni = cfg.numInputs;
    var nr = cfg.numRecovery;
    var bs = cfg.blockSize;
    var chunks = cfg.numChunks;
    var chunkSize = Math.floor(ni / chunks);

    var inputBuf = Buffer.alloc(ni * bs);
    fillRandom(inputBuf, rng);

    // 1. Single pass
    var singleOut = Buffer.alloc(nr * bs);
    addon.compute_recovery_barycentric(inputBuf, singleOut, ni, nr, bs, 0, ni, 0, false);

    // 2. Chunked accumulation
    var chunkedOut = Buffer.alloc(nr * bs);
    chunkedOut.fill(0x55);

    for (var c = 0; c < chunks; c++) {
        var startIdx = c * chunkSize;
        var endIdx = (c === chunks - 1) ? ni : (c + 1) * chunkSize;
        var count = endIdx - startIdx;
        var sub = inputBuf.subarray(startIdx * bs, endIdx * bs);
        var isAccum = (c > 0);

        addon.compute_recovery_barycentric(sub, chunkedOut, count, nr, bs, startIdx, ni, 0, isAccum);
    }

    var eq = singleOut.equals(chunkedOut);
    assert(eq, 'Barycentric Config ' + idx + ': N=' + ni + ' R=' + nr + ' B=' + bs + ' chunks=' + chunks + ' bit-identical');
});

console.log('\nSection C: compute_recovery_accumulate alias check');
console.log('---------------------------------------------------\n');

assert(typeof addon.compute_recovery_accumulate === 'function', 'compute_recovery_accumulate is exported as function');

var testIn = Buffer.alloc(8 * 64);
fillRandom(testIn, rng);
var outA = Buffer.alloc(2 * 64);
var outB = Buffer.alloc(2 * 64);
addon.compute_recovery_full(testIn, outA, 8, 2, 64, 0, 8, 0, false);
addon.compute_recovery_accumulate(testIn, outB, 8, 2, 64, 0, 8, 0, false);
assert(outA.equals(outB), 'compute_recovery_accumulate produces identical output to compute_recovery_full');

console.log('\nSection D: Negative trap (bit flip in chunk must fail)');
console.log('------------------------------------------------------\n');

var trapIn = Buffer.alloc(16 * 64);
fillRandom(trapIn, rng);
var refOut = Buffer.alloc(4 * 64);
addon.compute_recovery_full(trapIn, refOut, 16, 4, 64, 0, 16, 0, false);

// Chunk 1 gets a bit flip
var corruptedIn = Buffer.from(trapIn);
corruptedIn[9 * 64 + 3] ^= 0x01;

var trapOut = Buffer.alloc(4 * 64);
addon.compute_recovery_full(corruptedIn.subarray(0, 8 * 64), trapOut, 8, 4, 64, 0, 16, 0, false);
addon.compute_recovery_full(corruptedIn.subarray(8 * 64, 16 * 64), trapOut, 8, 4, 64, 8, 16, 0, true);

assert(!refOut.equals(trapOut), 'Negative trap: flipped bit in chunk 1 correctly fails equality check');

console.log('\n---');
console.log('RESULT: ' + passed + ' passed, ' + failed + ' failed');
if (failed > 0) process.exit(1);
