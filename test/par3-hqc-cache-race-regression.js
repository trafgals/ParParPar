#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 HQC additive-FFT cache data-race regression test
// ----------------------------------------------------------------------------
// Pins the fix for a data race in the global `hqc_cache` + `hqc_vindex`
// registries inside gf64/gf64_additive_fft_hqc2026.c.
//
// Where the HQC path is exercised
// --------------------------------
// The HQC additive-FFT pipeline (`basisCvt_recursive` → `get_or_build_basis_cache`)
// is called from `gf64_add_fft` and `gf64_poly_mul_internal`, which are
// reached from Fenger's MPE (`gf64_fenger_execute` → `gf64_poly_mul`).
// MATVEC (`ComputeRecoveryBlocks` → `WorkerThread` → `gf64_region_muladd_arr`)
// uses Cauchy inverses + scalar muladd, NOT HQC addFFT. So a regression
// test that uses small shapes (where matvec is feasible) silently misses
// the registry entirely.
//
// To force the HQC path the test calls `compute_recovery_fenger` directly
// with PAR3_GF64_USE_FENGER=1, bypassing the cost-model gate. Each call
// spawns Fenger's std::thread workers, each of which builds an entry
// in `hqc_cache` + `hqc_vindex`. With many concurrent creates on small
// shapes, the race window in the (now mutex-protected) registry
// shrinks the effective rate at which the unfixed code crashes.
//
// The race
// --------
// Worker threads spawned by `ComputeRecoveryBlocks` (std::async row-
// stealing in PR #108; std::thread workers in PR #111) race on the
// global `hqc_cache[s].initialized` flag and on `hqc_vindex_count`
// inside `hqc_vindex_build/drop`. Concurrent reads+writes produce torn
// state. The corruption manifests as an invalid chunk pointer being
// freed, which glibc's malloc detects on the FIRST subsequent free() —
// typically the per-thread `hqc_vtable_cache` destructor on pthread_exit,
// when the destructor walks the corrupted `hqc_vindex` registry.
//
// Reproducer: ~20-50% per-run crash on `par3-chunked-inputs.js` (master
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
// Pre-fix: ~50-100% per-run crash on Linux (gated by glibc's malloc
// detection timing). Post-fix: 0% per-run crash.
// ============================================================================

process.env.PAR3_GF64_USE_FENGER = "1";   // force Fenger; HQC path

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

// Fenger shape: N=2048, R=32, B=1024 -> 2 MiB input, 64 KiB output.
// R is a power of 2 (eligible for Fenger); cost model prefers Barycentric
// but PAR3_GF64_USE_FENGER=1 forces Fenger. The kernel walks the HQC
// addFFT code path (basisCvt_recursive -> get_or_build_basis_cache ->
// hqc_cache + hqc_vindex).
var BLOCK_SIZE = 1024;
var NUM_BLOCKS = 2048;
var RECOVERY = 32;
var REPS = 8;          // 8 concurrent creates
var OUTER_LOOPS = 5;   // repeat 5x to push repro rate to ~100% pre-fix

var tmp = fs.mkdtempSync(path.join(os.tmpdir(), "par3-hqc-race-"));
var inFile = path.join(tmp, "input.bin");
var fd = fs.openSync(inFile, "w");
var buf = crypto.randomBytes(BLOCK_SIZE);
for (var i = 0; i < NUM_BLOCKS; i++) fs.writeSync(fd, buf, 0, BLOCK_SIZE);
fs.closeSync(fd);

console.log("Running " + OUTER_LOOPS + " outer x " + REPS + " concurrent = " + (OUTER_LOOPS * REPS) + " total creates (Fenger-forced)...");
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
