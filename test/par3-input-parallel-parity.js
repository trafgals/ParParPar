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

// Verify NAPI export and initial sentinel
assert(typeof addon.get_last_decomposition_path === 'function', 'get_last_decomposition_path exported as function');
assert(addon.get_last_decomposition_path() === 0, 'Initial decomposition path is 0 (sentinel before calls)');

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

        // Cubic review P2 & P3: assert decomposition path matching all three preconditions:
        // t > R, N >= t, and scratch <= 64 MiB
        var decompPath = addon.get_last_decomposition_path();
        var fitsScratchCap = (t > 1) && ((R * B * (t - 1)) <= 64 * 1024 * 1024);
        if (t === 1) {
            assert(decompPath === 1, 'Case ' + idx + ' threads=' + t + ' used single-threaded path (1)');
        } else if (t > R && N >= t && fitsScratchCap) {
            assert(decompPath === 2, 'Case ' + idx + ' threads=' + t + ' actively exercised input-domain decomposition (2)');
        } else {
            assert(decompPath === 3, 'Case ' + idx + ' threads=' + t + ' used output-domain decomposition (3)');
        }
    });
});

// Cubic review P1: Test that oversized scratch (>64 MiB) safely falls back to output-domain decomposition (path 3)
console.log('\nTesting bounded scratch fallback (>64 MiB):');
// Sanity check: small scratch uses input-domain decomposition (path 2)
var smallR = 2;
var smallB = 64;
var smallN = 16;
var smallIn = Buffer.alloc(smallN * smallB);
addon.compute_recovery_full(smallIn, Buffer.alloc(smallR * smallB), smallN, smallR, smallB, 0, smallN, 16, false);
assert(addon.get_last_decomposition_path() === 2, 'Small scratch uses input-domain decomposition (2)');

// Cubic review P3: Pin PAR3_INPUT_DECOMP_SCRATCH_BYTES explicitly so adaptR test is hermetic
process.env.PAR3_INPUT_DECOMP_SCRATCH_BYTES = (128 * 1024 * 1024).toString();
var adaptR = 8;
var adaptB = 1024 * 1024; // 1 MiB
var adaptN = 16;
var adaptIn = Buffer.alloc(adaptN * adaptB);
var adaptOut = Buffer.alloc(adaptR * adaptB);
addon.compute_recovery_full(adaptIn, adaptOut, adaptN, adaptR, adaptB, 0, adaptN, 16, false);
assert(addon.get_last_decomposition_path() === 2, 'Adaptive scratch (120 MiB <= 128 MiB cap) uses input-domain decomposition (2)');
delete process.env.PAR3_INPUT_DECOMP_SCRATCH_BYTES;

// Oversized scratch fallback: 16 recovery x 1 MiB block x 16 threads (would need 240 MiB scratch > 128 MiB cap;
// max workers that fit scratch is 9 <= R=16) safely falls back to output-domain decomposition (path 3).
var capR = 16;
var capB = 1024 * 1024; // 1 MiB
var capN = 16;
var capIn = Buffer.alloc(capN * capB);
var capOut = Buffer.alloc(capR * capB);
addon.compute_recovery_full(capIn, capOut, capN, capR, capB, 0, capN, 16, false);
assert(addon.get_last_decomposition_path() === 3, 'Oversized scratch (240 MiB > 128 MiB cap) safely falls back to output-domain decomposition (3)');

// Cubic review P2: PAR3_INPUT_DECOMP_SCRATCH_BYTES parser validation tests (rejects signs, non-digits, ERANGE)
// Tests are discriminating: a small valid cap (1 MiB) forces path 3 on adaptR (needs 120 MiB),
// whereas rejected invalid values fall back to default 128 MiB cap yielding path 2.
// Similarly, wrap-around values like negative or ERANGE would yield ULLONG_MAX (path 2 on capR),
// whereas rejected invalid values fall back to default 128 MiB cap yielding path 3 on capR.
console.log('\nTesting PAR3_INPUT_DECOMP_SCRATCH_BYTES parser validation (cubic review P2):');

// Control baseline: valid small cap (1 MiB) correctly forces path 3 on adaptR
process.env.PAR3_INPUT_DECOMP_SCRATCH_BYTES = '1048576';
addon.compute_recovery_full(adaptIn, adaptOut, adaptN, adaptR, adaptB, 0, adaptN, 16, false);
assert(addon.get_last_decomposition_path() === 3, 'Control: valid 1 MiB cap forces output-domain decomposition (path 3)');

// Leading plus on 1 MiB: if accepted yields path 3, but rejected falls back to default 128 MiB cap -> path 2
process.env.PAR3_INPUT_DECOMP_SCRATCH_BYTES = '+1048576';
addon.compute_recovery_full(adaptIn, adaptOut, adaptN, adaptR, adaptB, 0, adaptN, 16, false);
assert(addon.get_last_decomposition_path() === 2, 'Leading plus (+1048576) rejected, falls back to default cap (path 2)');

// Trailing non-digits on 1 MiB: if accepted yields path 3, but rejected falls back to default 128 MiB cap -> path 2
process.env.PAR3_INPUT_DECOMP_SCRATCH_BYTES = '1048576bytes';
addon.compute_recovery_full(adaptIn, adaptOut, adaptN, adaptR, adaptB, 0, adaptN, 16, false);
assert(addon.get_last_decomposition_path() === 2, 'Trailing characters (1048576bytes) rejected, falls back to default cap (path 2)');

// Negative value: if strtoull parsed it, wraps to ULLONG_MAX (> 240 MiB) yielding path 2 on capR;
// rejected falls back to default 128 MiB cap yielding path 3 on capR
process.env.PAR3_INPUT_DECOMP_SCRATCH_BYTES = '-1';
addon.compute_recovery_full(capIn, capOut, capN, capR, capB, 0, capN, 16, false);
assert(addon.get_last_decomposition_path() === 3, 'Negative scratch cap (-1) rejected, falls back to default cap (path 3)');

// ERANGE overflow: if unchecked, strtoull returns ULLONG_MAX (> 240 MiB) yielding path 2 on capR;
// rejected falls back to default 128 MiB cap yielding path 3 on capR
process.env.PAR3_INPUT_DECOMP_SCRATCH_BYTES = '9999999999999999999999999999999999999999';
addon.compute_recovery_full(capIn, capOut, capN, capR, capB, 0, capN, 16, false);
assert(addon.get_last_decomposition_path() === 3, 'ERANGE overflow rejected, falls back to default cap (path 3)');

// Clean up env
delete process.env.PAR3_INPUT_DECOMP_SCRATCH_BYTES;

// Issue #117: Test that accumulate=true safely bypasses input-domain decomposition (path 2)
// and routes directly to output-domain decomposition (path 3), preserving bit-exact parity
// while eliminating 15 scratch allocations, 15 thread joins, and 15-way XOR reductions.
console.log('\nTesting accumulate=true path selection (Issue #117):');
var accOut = Buffer.alloc(smallR * smallB);
// When accumulate=false, small workload uses path 2 (input-domain decomp)
addon.compute_recovery_full(smallIn, accOut, smallN, smallR, smallB, 0, smallN, 16, false);
assert(addon.get_last_decomposition_path() === 2, 'Issue #117: accumulate=false uses input-domain decomposition (path 2)');

// When accumulate=true, the engine must skip input-domain decomp and use path 3 (output-domain)
var accOut2 = Buffer.alloc(smallR * smallB);
addon.compute_recovery_full(smallIn, accOut2, smallN, smallR, smallB, 0, smallN, 16, true);
assert(addon.get_last_decomposition_path() === 3, 'Issue #117: accumulate=true forces output-domain decomposition (path 3)');
assert(accOut.equals(accOut2), 'Issue #117: accumulate=true into zeroed buffer produces bit-exact parity with accumulate=false');

// Set path back to 3 for sentinel test
addon.compute_recovery_full(capIn, capOut, capN, capR, capB, 0, capN, 16, false);

// Cubic review P3: Test that Barycentric and Fenger reset decomposition path to 0 (sentinel)
console.log('\nTesting non-routing kernel path reset (0 sentinel):');
assert(addon.get_last_decomposition_path() === 3, 'Pre-condition: path is currently 3');
addon.compute_recovery_fenger(Buffer.alloc(16 * 64), Buffer.alloc(2 * 64), 16, 2, 64, 0x1000, 0x2000, 0);
assert(addon.get_last_decomposition_path() === 0, 'Fenger call resets decomposition path to 0 (sentinel)');

// Test native xor_buffers
console.log('\nTesting native xor_buffers:');
assert(typeof addon.xor_buffers === 'function', 'xor_buffers exported as function');
var xorA = Buffer.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
var xorB = Buffer.from([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
var expectedXor = Buffer.alloc(16);
for (var i = 0; i < 16; i++) expectedXor[i] = xorA[i] ^ xorB[i];
addon.xor_buffers(xorA, xorB);
assert(xorA.equals(expectedXor), 'xor_buffers produces bit-exact XOR result');

// Cubic review P1 (round 4): xor_buffers length-mismatch contract
//   P1 risk: silently truncating when dst.length !== src.length discards the tail
//   of every chunk — chunked Fenger recovery becomes a partial-XOR reduction and
//   the produced recovery packets are wrong. The contract must throw instead.
console.log('\nTesting xor_buffers length-mismatch contract:');
var xorShort = Buffer.from([1, 2, 3, 4]);      // 4 bytes
var xorLong  = Buffer.from([5, 6, 7, 8, 9, 10]); // 6 bytes (dst longer than src)
var threwLengthMismatch = false;
try { addon.xor_buffers(xorShort, xorLong); }
catch (e) {
    threwLengthMismatch = true;
    assert(/length/i.test(String(e && e.message || e)), 'xor_buffers length-mismatch error message mentions length');
}
assert(threwLengthMismatch, 'xor_buffers throws on dst.length !== src.length instead of silently truncating');

// Also assert the symmetric case (src longer than dst) throws:
var xorLongSrc = Buffer.from([5, 6, 7, 8, 9, 10]);
var xorShortDst = Buffer.from([1, 2, 3, 4]);
var threwLengthMismatch2 = false;
try { addon.xor_buffers(xorLongSrc, xorShortDst); }
catch (e) { threwLengthMismatch2 = true; }
assert(threwLengthMismatch2, 'xor_buffers throws when src is longer than dst');

// Equal lengths still works (round-trip the existing 16-byte case):
var xorC = Buffer.alloc(8, 0xAA);
var xorD = Buffer.alloc(8, 0x55);
addon.xor_buffers(xorC, xorD);
assert(xorC.equals(Buffer.alloc(8, 0xFF)), 'xor_buffers equal-length XOR produces 0xFF');

console.log('\n---');
console.log('RESULT: ' + passed + ' passed, ' + failed + ' failed');
if (failed > 0) process.exit(1);
