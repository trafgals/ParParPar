#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 HQC additive-FFT cache data-race regression test
// ----------------------------------------------------------------------------
// Pins the fix for a data race in the global `hqc_cache` + `hqc_vindex`
// registries inside gf64/gf64_additive_fft_hqc2026.c.
//
// The race: every worker thread spawned by `ComputeRecoveryBlocks`
// (std::async row-stealing workers in master PR #108, std::thread
// workers in PR #111's input-domain decomposition, and any future
// parallel path that touches HQC addFFT) races on the global
// `hqc_cache[s].initialized` flag and on `hqc_vindex_count` inside
// `hqc_vindex_build/drop`. Concurrent reads+writes produce torn state.
//
// Symptom: glibc's malloc detects the corruption on the FIRST subsequent
// free() of an invalid chunk pointer. Most commonly this manifests as
// `double free or corruption (out)` from the per-thread
// `hqc_vtable_cache` destructor on pthread_exit, when the destructor
// walks the corrupted `hqc_vindex` registry to drop entries.
//
// Reproducer: ~20% per-run crash on `par3-chunked-inputs.js` (master
// and PR #111 alike — pre-existing race, NOT introduced by PR #111),
// traced via gdb on a WSL Ubuntu / Node 22.22.1 core dump:
//
//   #0  __GI_abort ()
//   #7  malloc_printerr ("double free or corruption (out)")
//   #8  _int_free_merge_chunk (size=10481008)
//   #10 hqc_vtable_cache_destroy () from parpar_gf64.node
//   #11 __GI___nptl_deallocate_tsd ()
//
// Fix: protect both registries with a mutex (POSIX pthread_mutex_t;
// Win32 CRITICAL_SECTION initialized via InitOnceExecuteOnce) — added
// at the top of gf64/gf64_additive_fft_hqc2026.c.
//
// This regression test exercises the race path with high concurrency:
// 5 outer x 8 concurrent = 40 total creates, each one forces the
// Barycentric Cauchy matvec kernel (matvec-infeasible on this shape,
// Barycentric selected) which routes through HQC addFFT and populates
// the hqc_cache + hqc_vindex globals across multiple worker threads.
//
// Pre-fix: ~50-100% per-run SIGABRT crash (gated by glibc's malloc
// detection timing). Post-fix: 0% per-run crash, no leaks.
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

console.log("PAR3 HQC cache data-race regression test");
console.log("==========================================\n");

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
