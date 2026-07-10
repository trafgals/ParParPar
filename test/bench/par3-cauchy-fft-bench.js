#!/usr/bin/env node
"use strict";

/*
 * PAR3 Cauchy FFT (Barycentric) Throughput Benchmark
 *
 * LOCAL ONLY — never run in CI.
 *
 * Compares the legacy 2D-muladd path against the new Barycentric kernel
 * for the two canonical PAR3 workloads:
 *
 *   - Canonical (1 GiB / 10K recovery) — bench gate
 *   - Regression (100 MiB / 1K recovery) — fast check
 *
 * Reports throughput in MB/s for each kernel, forcing one or the other via
 * the PAR3_BF64_MIN_INPUTS env knob (set absurdly high to force 2D-muladd;
 * delete to allow the dispatch to pick Barycentric).
 *
 * SKIPs cleanly if the native binding is unavailable.
 */

var addon;
try {
    addon = require('../../build/Release/parpar_gf64.node');
} catch(e) {
    console.log('SKIP: parpar_gf64.node not built.');
    process.exit(0);
}

var PAR3 = require('../../lib/par3gen');

// mulberry32 PRNG for deterministic fill
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

function benchKernelOnly(numInputs, numRecovery, blockSize, useBarycentric) {
    var inputs = Buffer.alloc(numInputs * blockSize * 8);
    var outputs = Buffer.alloc(numRecovery * blockSize * 8);
    // Fill with random data
    var rng = mulberry32(0xDEADBEEF);
    for (var i = 0; i < inputs.length; i++) inputs[i] = rng() & 0xFF;

    var fn = useBarycentric ? addon.compute_recovery_barycentric : addon.compute_recovery_full;
    if (typeof fn !== 'function') {
        return null;
    }

    var start = process.hrtime.bigint();
    fn(inputs, outputs, numInputs, numRecovery, blockSize, 0n, BigInt(numInputs), 0);
    var elapsedNs = Number(process.hrtime.bigint() - start);
    var elapsedSec = elapsedNs / 1e9;

    var totalBytes = numInputs * blockSize * 8 + numRecovery * blockSize * 8;
    var mbps = (totalBytes / elapsedSec) / 1e6;
    return mbps;
}

function runWorkload(name, numInputs, numRecovery, blockSize) {
    console.log('\n=== ' + name + ' (' + numInputs + ' inputs, ' + numRecovery + ' recovery, ' + blockSize + ' words) ===');

    // Force 2D-muladd via env
    process.env.PAR3_BF64_MIN_INPUTS = '99999999';
    var legacy = benchKernelOnly(numInputs, numRecovery, blockSize, false);

    // Reset env to allow Barycentric
    delete process.env.PAR3_BF64_MIN_INPUTS;
    var bary = benchKernelOnly(numInputs, numRecovery, blockSize, true);

    console.log('  legacy 2D-muladd:  ' + (legacy !== null ? legacy.toFixed(2) + ' MB/s' : 'SKIP'));
    console.log('  Barycentric:      ' + (bary !== null ? bary.toFixed(2) + ' MB/s' : 'SKIP'));
}

// Canonical: 1 GiB / 10K
// 1 GiB = 1024*1024*1024 bytes = 268435456 gf64_t; numInputs = 262144, blockSize = 1024
runWorkload('Canonical (1 GiB / 10K recovery)', 262144, 10000, 1024);

// Regression: 100 MiB / 1K
// 100 MiB = 104857600 bytes = 26214400 gf64_t
runWorkload('Regression (100 MiB / 1K recovery)', 26214, 1000, 1024);
