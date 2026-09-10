#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 HQC additive-FFT cache data-race regression test (FAST variant)
// ----------------------------------------------------------------------------
// Same fix verification as test/par3-hqc-cache-race-regression.js, but with
// a smaller fixture (4 MiB instead of 32 MiB) and higher concurrency so the
// loop completes in <30s on slow CI hosts while still reproducing the race.
//
// Pre-fix: ~50-100% per-run crash on Linux (gated by glibc's malloc
// detection timing). Post-fix: 0% per-run crash.
// ============================================================================

var fs = require("fs");
var path = require("path");
var os = require("os");
var crypto = require("crypto");

var par3gen = require("../lib/par3gen.js");

var passed = 0;
var failed = 0;
function assertMsg(cond, msg) {
    if (cond) { console.log("  PASS: " + msg); passed++; return; }
    console.error("  FAIL: " + msg); failed++; process.exitCode = 1;
}

console.log("PAR3 HQC cache data-race regression test (fast)");
console.log("=================================================\n");

// Shape:
//   N=512 R=32 B=8192 -> matvec=128 MB infeasible (128 MiB cap), fenger not pow2
//   -> Barycentric selected (which routes through HQC addFFT for the
//      small-shape Cauchy fall-through via ComputeRecoveryBlocks).
// 4 MiB input = 512 blocks * 8192 bytes.
var BLOCK_SIZE = 8192;
var NUM_BLOCKS = 512;
var RECOVERY = 32;
var REPS = 8;          // 8 concurrent creates
var OUTER_LOOPS = 5;   // repeat 5x to push repro rate to ~100% pre-fix

var tmp = fs.mkdtempSync(path.join(os.tmpdir(), "par3-hqc-race-"));
var inFile = path.join(tmp, "input.bin");
var fd = fs.openSync(inFile, "w");
var buf = crypto.randomBytes(BLOCK_SIZE);
for (var i = 0; i < NUM_BLOCKS; i++) fs.writeSync(fd, buf, 0, BLOCK_SIZE);
fs.closeSync(fd);

console.log("Running " + OUTER_LOOPS + " outer x " + REPS + " concurrent = " + (OUTER_LOOPS * REPS) + " total creates...");
var loopIdx = 0;
var doneLoops = 0;
var errored = 0;
function runOneOuter() {
    var pending = REPS;
    for (var r = 0; r < REPS; r++) {
        par3gen.create([inFile], path.join(tmp, "out-" + loopIdx + "-" + r), {
            blockSize: BLOCK_SIZE,
            recoverySlices: RECOVERY
        }, function(err) {
            if (err) {
                console.error("  outer " + loopIdx + " create error: " + err.message);
                errored++;
            }
            pending--;
            if (pending === 0) {
                doneLoops++;
                loopIdx++;
                if (doneLoops < OUTER_LOOPS) {
                    runOneOuter();
                } else {
                    try {
                        fs.readdirSync(tmp).forEach(function(f) { fs.unlinkSync(path.join(tmp, f)); });
                        fs.rmdirSync(tmp);
                    } catch (e) { /* best effort */ }
                    console.log("\n---");
                    console.log("RESULT: " + passed + " passed, " + failed + " failed");
                    // The test passes iff no create errored AND the
                    // process didn't SIGABRT from malloc detecting the
                    // race-induced corruption (the latter manifests
                    // as a non-zero process exit, which Node reports
                    // to our caller via process.exitCode).
                    assertMsg(errored === 0, "no create errors across " + (OUTER_LOOPS * REPS) + " concurrent creates (errored=" + errored + ")");
                    if (failed > 0) process.exit(1);
                }
            }
        });
    }
}
runOneOuter();
