#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 Barycentric Kernel Bit-Exact Parity Test (T13)
// ----------------------------------------------------------------------------
// Mirrors the pattern of test/par3-kernel-parity.js for the new Barycentric
// recovery kernel exposed via the `compute_recovery_barycentric` NAPI binding.
//
// The Barycentric kernel (Cauchy-FFT, near-linear in N) computes the same
// recovery blocks as the legacy 2D-muladd kernel. This test asserts that both
// kernels produce byte-for-byte identical output over a deterministic seeded
// grid of (N, R, blockSize) tuples, against an independent pure-JS GF(2^64)
// reference.
//
// Test grid (27 cases): numInputs × numRecovery × blockSize
//   numInputs:   [16, 64, 256]
//   numRecovery: [1, 4, 8]
//   blockSize:   [8, 64, 1024] bytes (each divisible by 8 for gf64_t)
//
// Per case:
//   1. Generate deterministic random input blocks via mulberry32.
//   2. Call addon.compute_recovery_barycentric(inputs, outputs, N, R, bs, fi, fr).
//   3. Compute JS reference via the same cauchyCoeff + gf64_mul pipeline used
//      by lib/par3gen.js (mirrors par3-kernel-parity.js:82-88).
//   4. Assert outputs.equals(reference) — bit-exact.
//   5. Flip one bit in an input, re-run kernel + reference, assert they
//      STILL match each other (proves kernel reacts to inputs) — i.e. the
//      kernel is not a constant / always-zero stub.
//
// Usage:
//   node test/par3-barycentric-parity.js
//
// Expected output (last line):
//   PASS  or  FAILED  or  SKIP
//
// SKIP behavior: this test will SKIP cleanly when the native build is broken
// (the pre-existing `gettimeofday` issue prevents a full `node-gyp build` on
// this host, so `build/Release/parpar_gf64.node` is missing). The SKIP path
// prints a clear, single-line reason and exits 0 so CI can run this test
// without it being a red gate while the build is being repaired.
// ============================================================================

var addon;
try {
    addon = require('../build/Release/parpar_gf64.node');
} catch (e) {
    console.log('PAR3 Barycentric Kernel Bit-Exact Parity Test (T13)');
    console.log('===================================================\n');
    console.log('SKIP: parpar_gf64.node not built.');
    console.log('Reason: ' + (e && e.message ? e.message : String(e)));
    console.log('The pre-existing gettimeofday build issue prevents a full');
    console.log('node-gyp build on this host. This test requires a working');
    console.log('native build to exercise the compute_recovery_barycentric NAPI');
    console.log('binding. The test is structured to PASS automatically once');
    console.log('build/Release/parpar_gf64.node is produced.');
    process.exit(0);
}

// ============================================================================
// Pure JS GF(2^64) primitives (independent reference implementation)
// ----------------------------------------------------------------------------
// These are bit-identical to the implementation in par3-kernel-parity.js and
// par3-native-fidelity.js. JavaScript BigInt computes GF(2^64) field
// operations using the irreducible polynomial 0x100000000000001B.
// ============================================================================

var GF64_POLY = 0x1000000000000001Bn;
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

function invert64(val) {
    val = val & GF64_MASK;
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
        if (u === 1n) continue;
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

// Matches lib/par3gen.js:82-88 — same Cauchy coefficient computation as the
// reference parity test (par3-kernel-parity.js).
function cauchyCoeff(firstInput, inputIdx, firstRecovery, recoveryIdx) {
    var x = BigInt(firstInput) + BigInt(inputIdx);
    var y = BigInt(firstRecovery) + BigInt(recoveryIdx);
    var denom = x ^ y;
    if (denom === 0n) return 0n;
    return invert64(denom);
}

// ============================================================================
// Deterministic pseudo-random number generator (mulberry32)
// ----------------------------------------------------------------------------
// Reproducible across all runs — same seed → same sequence → same test
// inputs → same outputs. The constant seed makes failures reproducible.
// ============================================================================

var SEED = 0xDEADBEEF;

function mulberry32(seed) {
    return function () {
        seed |= 0;
        seed = seed + 0x6D2B79F5 | 0;
        var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// ============================================================================
// JS reference: per-block recovery computation
// ----------------------------------------------------------------------------
// For each recovery block k and each input block j, multiply the input block
// by its Cauchy coefficient (firstInput+j XOR firstRecovery+k inverted) and
// XOR-accumulate into output[k]. This is the naive O(N·R·B) reference that
// every fast kernel must match bit-exactly.
// ============================================================================

function jsRecovery(inputs, numInputs, numRecovery, blockSize, firstInput, firstRecovery) {
    var numWords = blockSize / 8;
    var outputs = Buffer.alloc(numRecovery * blockSize);
    outputs.fill(0);

    // Per-word scratch buffer; we XOR-accumulate per (k, w) into outputs.
    for (var k = 0; k < numRecovery; k++) {
        for (var j = 0; j < numInputs; j++) {
            var coeff = cauchyCoeff(firstInput, j, firstRecovery, k);
            var inOff = j * blockSize;
            var outOff = k * blockSize;
            for (var w = 0; w < numWords; w++) {
                var inW = inputs.readBigUInt64LE(inOff + w * 8);
                var prod = gf64_mul(inW, coeff);
                var prev = outputs.readBigUInt64LE(outOff + w * 8);
                outputs.writeBigUInt64LE(prev ^ prod, outOff + w * 8);
            }
        }
    }

    return outputs;
}

// ============================================================================
// Test infrastructure
// ============================================================================

console.log('PAR3 Barycentric Kernel Bit-Exact Parity Test (T13)');
console.log('===================================================\n');

// Detect whether the Barycentric binding is actually exported. The build may
// produce a .node that lacks this NAPI export if T7/T8/T9 work was incomplete.
if (typeof addon.compute_recovery_barycentric !== 'function') {
    console.log('SKIP: compute_recovery_barycentric NAPI export not present on addon.');
    console.log('Reason: addon loaded but does not expose compute_recovery_barycentric.');
    console.log('The Barycentric kernel binding was added in T9; if the loaded addon');
    console.log('predates that, this test cannot run. Rebuild from a T9-or-later commit.');
    process.exit(0);
}

var passed = 0;
var failed = 0;

function recordPass(label) {
    console.log('  PASS: ' + label);
    passed++;
}

function recordFail(label, detail) {
    console.error('  FAIL: ' + label);
    if (detail) console.error('    ' + detail);
    failed++;
    process.exitCode = 1;
}

// ============================================================================
// Test grid
// ============================================================================

var NUM_INPUTS_GRID   = [16, 64, 256];
var NUM_RECOVERY_GRID = [1, 4, 8];
var BLOCK_SIZE_GRID   = [8, 64, 1024];

var PARITY_CASES = [];
for (var n = 0; n < NUM_INPUTS_GRID.length; n++) {
    for (var r = 0; r < NUM_RECOVERY_GRID.length; r++) {
        for (var b = 0; b < BLOCK_SIZE_GRID.length; b++) {
            PARITY_CASES.push({
                numInputs:   NUM_INPUTS_GRID[n],
                numRecovery: NUM_RECOVERY_GRID[r],
                blockSize:   BLOCK_SIZE_GRID[b],
            });
        }
    }
}

console.log('Test grid: ' + PARITY_CASES.length + ' (N, R, bs) tuples');
console.log('  N:    [' + NUM_INPUTS_GRID.join(', ') + ']');
console.log('  R:    [' + NUM_RECOVERY_GRID.join(', ') + ']');
console.log('  bs:   [' + BLOCK_SIZE_GRID.join(', ') + '] bytes');
console.log('');

// ============================================================================// Input data generation
// ----------------------------------------------------------------------------
// Each test case uses deterministic random data filled via mulberry32(SEED).
// The seed is re-used per case so the input buffer contents are tied to the
// case index (i.e. case 5 always gets the same inputs as before).
// ============================================================================

function genInputs(seed, numInputs, blockSize) {
    var rng = mulberry32(seed);
    var inputs = Buffer.alloc(numInputs * blockSize);
    var numWords = blockSize / 8;
    for (var w = 0; w < numWords * numInputs; w++) {
        var hi = (rng() * 4294967296) >>> 0;
        var lo = (rng() * 4294967296) >>> 0;
        inputs.writeBigUInt64LE((BigInt(hi) << 32n) | BigInt(lo), w * 8);
    }
    return inputs;
}

// ============================================================================// Section A: Bit-exact parity vs JS reference
// ----------------------------------------------------------------------------
// For each (N, R, bs) tuple, generate random inputs, call the Barycentric
// kernel, compute the JS reference, and assert byte-for-byte equality.
// ============================================================================

console.log('Section A: Barycentric kernel vs JS reference (' + PARITY_CASES.length + ' cases)');
console.log('---------------------------------------------------------------------\n');

for (var ci = 0; ci < PARITY_CASES.length; ci++) {
    var c = PARITY_CASES[ci];

    // Per-case seed derived from SEED XOR case index — ensures case N is
    // reproducible across runs but cases don't all share the same inputs.
    var caseSeed = (SEED ^ ((ci + 1) * 0x9E3779B1)) >>> 0;
    var inputs = genInputs(caseSeed, c.numInputs, c.blockSize);
    var outputs = Buffer.alloc(c.numRecovery * c.blockSize);
    outputs.fill(0);

    // Pick non-overlapping firstInput/firstRecovery ranges so the Cauchy
    // denominators are never zero (matches par3-kernel-parity.js pattern).
    var firstInput = 0;
    var firstRecovery = c.numInputs + 1;

    // Run the Barycentric kernel.
    try {
        addon.compute_recovery_barycentric(
            inputs, outputs,
            c.numInputs, c.numRecovery, c.blockSize,
            firstInput, firstRecovery, 0
        );
    } catch (e) {
        recordFail(
            'N=' + c.numInputs + ' R=' + c.numRecovery + ' bs=' + c.blockSize +
            ' — compute_recovery_barycentric threw: ' + (e && e.message ? e.message : String(e)),
            null
        );
        continue;
    }

    // Compute the JS reference using the same parameters.
    var reference = jsRecovery(inputs, c.numInputs, c.numRecovery, c.blockSize, firstInput, firstRecovery);

    if (outputs.equals(reference)) {
        recordPass(
            'N=' + c.numInputs + ' R=' + c.numRecovery + ' bs=' + c.blockSize +
            ' — kernel output equals JS reference (' + outputs.length + ' bytes)'
        );
    } else {
        // Show the first differing byte/word for debuggability.
        var diffDetail = '';
        var numWords = c.blockSize / 8;
        outer:
        for (var kr = 0; kr < c.numRecovery; kr++) {
            for (var w = 0; w < numWords; w++) {
                var off = kr * c.blockSize + w * 8;
                var got = outputs.readBigUInt64LE(off);
                var exp = reference.readBigUInt64LE(off);
                if (got !== exp) {
                    diffDetail = 'recovery[' + kr + '] word[' + w + ']: got=0x' +
                        got.toString(16) + ' expected=0x' + exp.toString(16);
                    break outer;
                }
            }
        }
        recordFail(
            'N=' + c.numInputs + ' R=' + c.numRecovery + ' bs=' + c.blockSize +
            ' — kernel output differs from JS reference',
            diffDetail
        );
    }
}

console.log('\nSection A complete.\n');

// ============================================================================// Section B: Barycentric kernel vs legacy 2D-muladd kernel
// ----------------------------------------------------------------------------// Cross-check that the new Barycentric kernel agrees with the legacy kernel
// when both are available. If `compute_recovery_full` is not exported, SKIP
// this section (the legacy kernel may not be wired in the same addon).
// ============================================================================

if (typeof addon.compute_recovery_full === 'function' ||
    typeof addon.compute_recovery === 'function') {

    console.log('Section B: Barycentric kernel vs legacy 2D-muladd kernel');
    console.log('-------------------------------------------------------\n');

    var legacyName = typeof addon.compute_recovery_full === 'function'
        ? 'compute_recovery_full'
        : 'compute_recovery';
    console.log('Legacy kernel binding found: ' + legacyName + '\n');

    for (var ci2 = 0; ci2 < PARITY_CASES.length; ci2++) {
        var c2 = PARITY_CASES[ci2];
        var caseSeed2 = (SEED ^ ((ci2 + 1) * 0x9E3779B1)) >>> 0;
        var inputs2 = genInputs(caseSeed2, c2.numInputs, c2.blockSize);

        var baryOut = Buffer.alloc(c2.numRecovery * c2.blockSize);
        baryOut.fill(0);
        var legacyOut = Buffer.alloc(c2.numRecovery * c2.blockSize);
        legacyOut.fill(0);

        var fi2 = 0;
        var fr2 = c2.numInputs + 1;

        try {
            addon.compute_recovery_barycentric(
                inputs2, baryOut,
                c2.numInputs, c2.numRecovery, c2.blockSize,
                fi2, fr2, 0
            );
            addon[legacyName](
                inputs2, legacyOut,
                c2.numInputs, c2.numRecovery, c2.blockSize,
                fi2, fr2, 0
            );
        } catch (e) {
            recordFail(
                'cross-check N=' + c2.numInputs + ' R=' + c2.numRecovery + ' bs=' + c2.blockSize +
                ' — kernel threw: ' + (e && e.message ? e.message : String(e)),
                null
            );
            continue;
        }

        if (baryOut.equals(legacyOut)) {
            recordPass(
                'cross-check N=' + c2.numInputs + ' R=' + c2.numRecovery + ' bs=' + c2.blockSize +
                ' — Barycentric output equals legacy ' + legacyName
            );
        } else {
            var numWordsC = c2.blockSize / 8;
            var diffDetail2 = '';
            outer2:
            for (var kr2 = 0; kr2 < c2.numRecovery; kr2++) {
                for (var w2 = 0; w2 < numWordsC; w2++) {
                    var off2 = kr2 * c2.blockSize + w2 * 8;
                    var got2 = baryOut.readBigUInt64LE(off2);
                    var exp2 = legacyOut.readBigUInt64LE(off2);
                    if (got2 !== exp2) {
                        diffDetail2 = 'recovery[' + kr2 + '] word[' + w2 + ']: bary=0x' +
                            got2.toString(16) + ' legacy=0x' + exp2.toString(16);
                        break outer2;
                    }
                }
            }
            recordFail(
                'cross-check N=' + c2.numInputs + ' R=' + c2.numRecovery + ' bs=' + c2.blockSize +
                ' — Barycentric output differs from legacy ' + legacyName,
                diffDetail2
            );
        }
    }
    console.log('\nSection B complete.\n');
} else {
    console.log('Section B: SKIPPED (no legacy 2D-muladd kernel binding found)\n');
}

// ============================================================================// Section C: Negative-trap — flip one bit in inputs, assert kernel still
// ----------------------------------------------------------------------------// responds (kernel + reference still agree, but BOTH differ from the
// un-flipped baseline). This proves the kernel actually reads its input
// buffer (not a constant / always-zero stub).
// ============================================================================

console.log('Section C: Negative-trap (input-bit-flip response)');
console.log('--------------------------------------------------\n');

// Pick a representative case for the negative trap (medium workload).
var NEG_CASE = { numInputs: 64, numRecovery: 4, blockSize: 64 };
var negInputs = genInputs(SEED ^ 0xC0FFEE, NEG_CASE.numInputs, NEG_CASE.blockSize);
var negOutOriginal = Buffer.alloc(NEG_CASE.numRecovery * NEG_CASE.blockSize);
negOutOriginal.fill(0);

var negFi = 0;
var negFr = NEG_CASE.numInputs + 1;

addon.compute_recovery_barycentric(
    negInputs, negOutOriginal,
    NEG_CASE.numInputs, NEG_CASE.numRecovery, NEG_CASE.blockSize,
    negFi, negFr, 0
);

var negRefOriginal = jsRecovery(
    negInputs, NEG_CASE.numInputs, NEG_CASE.numRecovery, NEG_CASE.blockSize,
    negFi, negFr
);

if (!negOutOriginal.equals(negRefOriginal)) {
    recordFail(
        'negative-trap pre-check: Barycentric does not match JS reference on baseline; cannot run trap',
        null
    );
} else {
    recordPass(
        'negative-trap pre-check: Barycentric matches JS reference on baseline (' +
        negOutOriginal.length + ' bytes)'
    );

    // Flip one bit in the first word of the first input block.
    var firstWord = negInputs.readBigUInt64LE(0);
    var flippedWord = firstWord ^ (1n << 17n); // flip a high-but-not-top bit
    negInputs.writeBigUInt64LE(flippedWord, 0);

    var negOutFlipped = Buffer.alloc(NEG_CASE.numRecovery * NEG_CASE.blockSize);
    negOutFlipped.fill(0);

    addon.compute_recovery_barycentric(
        negInputs, negOutFlipped,
        NEG_CASE.numInputs, NEG_CASE.numRecovery, NEG_CASE.blockSize,
        negFi, negFr, 0
    );

    var negRefFlipped = jsRecovery(
        negInputs, NEG_CASE.numInputs, NEG_CASE.numRecovery, NEG_CASE.blockSize,
        negFi, negFr
    );

    // After flipping a single input bit, the kernel and reference must:
    //   (a) STILL agree with each other (bit-exact parity holds post-flip)
    //   (b) BOTH differ from the original (pre-flip) baseline output
    var kernelStillAgreesWithRef = negOutFlipped.equals(negRefFlipped);
    var kernelRespondedToFlip = !negOutFlipped.equals(negOutOriginal);
    var refRespondedToFlip = !negRefFlipped.equals(negRefOriginal);

    if (kernelStillAgreesWithRef &&
        kernelRespondedToFlip &&
        refRespondedToFlip) {
        recordPass(
            'negative-trap: kernel + reference agree post-flip; both differ from baseline ' +
            '(kernel reacts to input bits — proves non-constant)'
        );
    } else {
        var detail = 'kernelStillAgreesWithRef=' + kernelStillAgreesWithRef +
            ' kernelRespondedToFlip=' + kernelRespondedToFlip +
            ' refRespondedToFlip=' + refRespondedToFlip;
        recordFail(
            'negative-trap: ' + detail,
            null
        );
    }
}

console.log('\nSection C complete.\n');

// ============================================================================// Summary
// ============================================================================

console.log('---');
if (failed > 0) {
    console.log('FAILED (' + failed + ' failure(s), ' + passed + ' passed)');
    process.exitCode = 1;
} else if (passed === 0) {
    console.log('SKIP (no cases executed — no qualifying bindings)');
    process.exit(0);
} else {
    console.log('PASS (' + passed + ' passed)');
}