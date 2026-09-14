#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 Small-R Fused Multi-Output Kernel Parity Test
// ----------------------------------------------------------------------------
// Regression test for Issue #110 / PR #111 & PR #112 investigation:
// Verifies that the fused multi-output path in WorkerThread (R <= 32)
// produces 100% bit-exact parity across R in {1, 2, 4, 8, 16, 32}, across
// thread counts (1, 2, 4, 8, 16), and directly validates fused_output_muladd_arr
// zero-coefficient filtering and K > 256 chunking.
// ============================================================================

var addon = require('../build/Release/parpar_gf64.node');

var GF64_MASK = 0xFFFFFFFFFFFFFFFFn;
function gf64_mul(a, b) {
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

// ----------------------------------------------------------------------------
// Section D: fused_output_muladd_arr zero-coefficient handling (cubic review P3)
// ----------------------------------------------------------------------------
console.log('\nSection D: fused_output_muladd_arr zero-coefficient handling');
console.log('------------------------------------------------------------');

var enc = new addon.Gf64Encoder(0);
var numWordsD = 64;
var inBufD = Buffer.alloc(numWordsD * 8);
fillRandom(inBufD, rng);

var zeroCoeffs = [0n, 0x123456789ABCDEF0n, 0n, 0xFEDCBA9876543210n, 0n, 0n, 0x5555AAAA5555AAAAn, 0n];
var KD = zeroCoeffs.length;
var outBufsD = [];
var refBufsD = [];
var coeffBufsD = [];
for (var k = 0; k < KD; k++) {
    outBufsD.push(Buffer.alloc(numWordsD * 8));
    refBufsD.push(Buffer.alloc(numWordsD * 8));
    var cb = Buffer.alloc(8);
    cb.writeBigUInt64LE(zeroCoeffs[k], 0);
    coeffBufsD.push(cb);
    for (var w = 0; w < numWordsD; w++) {
        var inW = inBufD.readBigUInt64LE(w * 8);
        refBufsD[k].writeBigUInt64LE(gf64_mul(inW, zeroCoeffs[k]), w * 8);
    }
}
enc.fused_output_muladd_arr(outBufsD, inBufD, coeffBufsD, numWordsD, KD);
var okD = true;
for (var k = 0; k < KD; k++) {
    if (!outBufsD[k].equals(refBufsD[k])) okD = false;
}
assert(okD, 'fused_output_muladd_arr correctly skips zero coefficients and matches reference');

// Also test all-zero coefficients
var allZeroOut = [Buffer.alloc(numWordsD * 8), Buffer.alloc(numWordsD * 8), Buffer.alloc(numWordsD * 8), Buffer.alloc(numWordsD * 8)];
var allZeroBufs = [0n, 0n, 0n, 0n].map(function() { return Buffer.alloc(8); });
enc.fused_output_muladd_arr(allZeroOut, inBufD, allZeroBufs, numWordsD, 4);
var okAllZero = allZeroOut.every(function(b) { return b.equals(Buffer.alloc(numWordsD * 8)); });
assert(okAllZero, 'fused_output_muladd_arr with all-zero coefficients leaves buffers zeroed');

// ----------------------------------------------------------------------------
// Section E: fused_output_muladd_arr with K > 256 (cubic review P1)
// ----------------------------------------------------------------------------
console.log('\nSection E: fused_output_muladd_arr with K > 256 (chunking verification)');
console.log('----------------------------------------------------------------------');

var KE = 300; // strictly exercises K > 256 chunk boundary
var numWordsE = 32;
var inBufE = Buffer.alloc(numWordsE * 8);
fillRandom(inBufE, rng);

var outBufsE = [];
var refBufsE = [];
var coeffBufsE = [];
for (var k = 0; k < KE; k++) {
    outBufsE.push(Buffer.alloc(numWordsE * 8));
    refBufsE.push(Buffer.alloc(numWordsE * 8));
    var cVal = (BigInt((rng() * 4294967296) >>> 0) << 32n) | BigInt((rng() * 4294967296) >>> 0);
    var cbE = Buffer.alloc(8);
    cbE.writeBigUInt64LE(cVal, 0);
    coeffBufsE.push(cbE);
    for (var w = 0; w < numWordsE; w++) {
        var inWE = inBufE.readBigUInt64LE(w * 8);
        refBufsE[k].writeBigUInt64LE(gf64_mul(inWE, cVal), w * 8);
    }
}
enc.fused_output_muladd_arr(outBufsE, inBufE, coeffBufsE, numWordsE, KE);
var okE = true;
var firstBad = -1;
for (var k = 0; k < KE; k++) {
    if (!outBufsE[k].equals(refBufsE[k])) {
        okE = false;
        if (firstBad === -1) firstBad = k;
    }
}
assert(okE, 'fused_output_muladd_arr K=' + KE + ' (> 256) all outputs match reference (firstBad=' + firstBad + ')');

// ----------------------------------------------------------------------------
// Section F: fileInfo progressive updates and canonical ordering (cubic review P2)
// ----------------------------------------------------------------------------
console.log('\nSection F: fileInfo progressive updates and canonical ordering');
console.log('--------------------------------------------------------------');

var fs = require('fs');
var path = require('path');
var os = require('os');
var par3gen = require('../lib/par3gen.js');

var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'par3-fileinfo-test-'));
var testFiles = [];
for (var f = 0; f < 6; f++) {
    var fPath = path.join(tmpDir, 'file_' + f + '.bin');
    fs.writeFileSync(fPath, Buffer.alloc(1024 * (f + 1), f + 1));
    testFiles.push(fPath);
}

var returnedArray = par3gen.fileInfo(testFiles, function(err, orderedResults) {
    assert(!err, 'fileInfo completed without error');
    assert(Array.isArray(orderedResults), 'fileInfo returned array');
    assert(orderedResults.length === testFiles.length, 'fileInfo returned all ' + testFiles.length + ' files');

    // Verify canonical input order preservation
    var orderOk = true;
    for (var i = 0; i < testFiles.length; i++) {
        if (orderedResults[i].name !== testFiles[i]) orderOk = false;
    }
    assert(orderOk, 'fileInfo callback results preserve exact canonical input order');

    // Clean up
    testFiles.forEach(function(f) { try { fs.unlinkSync(f); } catch(e) {} });
    try { fs.rmdirSync(tmpDir); } catch(e) {}

    runSectionG();
});
assert(Array.isArray(returnedArray), 'fileInfo returns results array synchronously for progressive progress tracking');

// ----------------------------------------------------------------------------
// Section G: Batched read short-read resilience test (cubic review P1)
// ----------------------------------------------------------------------------
function runSectionG() {
    console.log('\nSection G: Batched read short-read resilience test');
    console.log('--------------------------------------------------');

    var gTmp = fs.mkdtempSync(path.join(os.tmpdir(), 'par3-shortread-test-'));
    var gSrc = path.join(gTmp, 'source.bin');
    var gSrcBuf = Buffer.alloc(128 * 1024); // 128 KiB input
    fillRandom(gSrcBuf, rng);
    fs.writeFileSync(gSrc, gSrcBuf);

    var refBase = path.join(gTmp, 'ref');
    var shortBase = path.join(gTmp, 'short');
    var refArc = path.join(gTmp, 'ref.par3');
    var shortArc = path.join(gTmp, 'short.par3');

    // Force chunked recovery path (128 KiB input with 32 KiB cap = 4 chunks)
    // Fixes cubic review 5193091436 P2: PAR3_SIMULATED_BUFFER_CAP is the correct env var in par3gen.js
    process.env.PAR3_SIMULATED_BUFFER_CAP = "32768";

    // Create reference archive with normal fs.readSync
    par3gen.create([gSrc], refBase, { blockSize: 4096, recoverySlices: 4 }, function(err) {
        assert(!err, 'Reference create completed');

        // Monkey-patch fs.readSync to return at most 512 bytes per read
        var realReadSync = fs.readSync;
        var shortReadCalls = 0;
        fs.readSync = function(fd, buffer, offset, length, position) {
            shortReadCalls++;
            var cappedLen = Math.min(length, 512);
            return realReadSync.call(fs, fd, buffer, offset, cappedLen, position);
        };

        par3gen.create([gSrc], shortBase, { blockSize: 4096, recoverySlices: 4 }, function(err2) {
            // Restore fs.readSync and buffer cap
            fs.readSync = realReadSync;
            delete process.env.PAR3_SIMULATED_BUFFER_CAP;

            assert(!err2, 'Short-read create completed without error');
            assert(shortReadCalls >= 256, 'Short-read monkey-patch intercepted at least 256 chunked read calls (actual: ' + shortReadCalls + ')');

            var refData = fs.readFileSync(refArc);
            var shortData = fs.readFileSync(shortArc);

            function extractBodies(buf, targetType) {
                var out = [];
                var off = 0;
                while (off + 48 <= buf.length) {
                    if (buf.slice(off, off + 8).toString("latin1") !== "PAR3\u0000PKT") break;
                    var len = Number(buf.readBigUInt64LE(off + 24));
                    var type = buf.slice(off + 40, off + 48).toString("latin1");
                    if (type === targetType) out.push(buf.slice(off + 48, off + len));
                    off += len;
                }
                return Buffer.concat(out);
            }

            var refDataPkts = extractBodies(refData, "PAR DAT\0");
            var shortDataPkts = extractBodies(shortData, "PAR DAT\0");
            assert(refDataPkts.length > 0 && refDataPkts.equals(shortDataPkts),
                'DATA packets with fragmented short reads are bit-identical to reference');

            var refRecPkts = extractBodies(refData, "PAR REC\u0000");
            var shortRecPkts = extractBodies(shortData, "PAR REC\u0000");
            assert(refRecPkts.length > 0 && refRecPkts.equals(shortRecPkts),
                'Recovery packets with fragmented short reads are bit-identical to reference');

            // Cleanup
            try {
                fs.unlinkSync(gSrc);
                fs.unlinkSync(refArc);
                fs.unlinkSync(shortArc);
                fs.rmdirSync(gTmp);
            } catch(e) {}

            console.log('\n---');
            console.log('RESULT: ' + passed + ' passed, ' + failed + ' failed');
            if (failed > 0) process.exit(1);
            process.exit(0);
        });
    });
}
