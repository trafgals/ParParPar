#!/usr/bin/env node
"use strict";

/*
 * PAR3 Barycentric vs 2D-muladd kernel microbench (Phase 2c microbench)
 *
 * Compares compute_recovery_barycentric (Barycentric kernel) against
 * compute_recovery_full (legacy 2D-muladd kernel) at the actual PAR3
 * creator workload sizes. Forces the Barycentric kernel to be exercised
 * (PAR3_BF64_MIN_INPUTS=0) so the dispatch always takes the Barycentric
 * path.
 *
 * Run:
 *   wsl -e bash -c 'cd /mnt/c/code/trafgals/ParParPar && node test/bench_par3_barycentric.js'
 *   # native Windows: from build/Release with the .node binary
 */

var addon;
try {
    addon = require("../build/Release/parpar_gf64.node");
} catch(e) {
    console.log("SKIP: parpar_gf64.node not built.");
    process.exit(0);
}

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

function bench(numInputs, numRecovery, blockSize, useBarycentric) {
    var inputs = Buffer.alloc(numInputs * blockSize * 8);
    var outputs = Buffer.alloc(numRecovery * blockSize * 8);
    var rng = mulberry32(0xDEADBEEF);
    for (var i = 0; i < inputs.length; i++) inputs[i] = rng() & 0xFF;
    var fn = useBarycentric ? addon.compute_recovery_barycentric : addon.compute_recovery_full;
    if (typeof fn !== "function") return null;
    fn(inputs, outputs, numInputs, numRecovery, blockSize, 0n, BigInt(numInputs), 0);
    var start = process.hrtime.bigint();
    fn(inputs, outputs, numInputs, numRecovery, blockSize, 0n, BigInt(numInputs), 0);
    var elapsedSec = Number(process.hrtime.bigint() - start) / 1e9;
    var totalBytes = numInputs * blockSize * 8 + numRecovery * blockSize * 8;
    return { mbps: (totalBytes / elapsedSec) / 1e6, sec: elapsedSec };
}

function run(name, numInputs, numRecovery, blockSize) {
    console.log("\n=== " + name + " (N=" + numInputs + " R=" + numRecovery + " B=" + blockSize + ") ===");
    process.env.PAR3_BF64_MIN_INPUTS = "0";
    var bary = bench(numInputs, numRecovery, blockSize, true);
    delete process.env.PAR3_BF64_MIN_INPUTS;
    var legacy = bench(numInputs, numRecovery, blockSize, false);
    if (legacy) console.log("  legacy 2D-muladd:  " + legacy.mbps.toFixed(2) + " MB/s  (" + legacy.sec.toFixed(3) + "s)");
    if (bary)   console.log("  Barycentric:      " + bary.mbps.toFixed(2) + " MB/s  (" + bary.sec.toFixed(3) + "s)");
    if (legacy && bary) {
        var pct = ((bary.mbps / legacy.mbps - 1) * 100);
        console.log("  ratio:            " + (pct >= 0 ? "+" : "") + pct.toFixed(1) + "% (positive = Barycentric faster)");
    }
}

console.log("PAR3 Barycentric vs 2D-muladd kernel microbench (Phase 2c)");
console.log("============================================================");
run("256 inputs / 64 recovery (small)",      256,    64,   64);
run("512 inputs / 64 recovery",               512,    64,   64);
run("1024 inputs / 128 recovery",             1024,   128,  64);
run("2048 inputs / 256 recovery",             2048,   256,  64);
run("4096 inputs / 512 recovery",             4096,   512,  64);
run("8192 inputs / 512 recovery",             8192,   512,  64);
run("16000 inputs / 1000 recovery",           16000,  1000, 1024);