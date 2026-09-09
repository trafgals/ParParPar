#!/usr/bin/env node
"use strict";

// ============================================================================
// test/par3-chunked-inputs.js — T11 (issue #91) chunked-input path
//
// CI regression test.
//
// Forces the chunked-input code path via PAR3_SIMULATED_BUFFER_CAP and
// verifies the chunked output is bit-equal to the single-call output
// (XOR-linearity: f(A xor B) = f(A) xor f(B) over GF(2^64)).
//
// Uses REC packet-body comparison (not full-archive comparison) to
// avoid the legitimate per-create variation in STA timestamp / UNX mtime
// that also breaks test/par3-create-streaming-parity leg2 on master.
//
// Two legs cover the contract:
//   1. 4 MiB leg (NUM_BLOCKS=1024, BLOCK_SIZE=4096, RECOVERY=32) — exercises
//      six buffer caps including a non-block-aligned cap (50000) to verify
//      the chunkSizeBytes = floor(cap/blockSize)*blockSize alignment.
//   2. 14 MiB-class leg (NUM_BLOCKS=14336, BLOCK_SIZE=1024, RECOVERY=2048)
//      — exercises a multi-chunk geometry at cap=1048576
//      (synthetic numChunks = ceil(14680064/1048576) = 14, well above
//      the ≥ 4 minimum). Plan-specified BLOCK_SIZE=64 was invalid
//      (par3gen.js:1267 enforces blockSize >= 1024), so BLOCK_SIZE is
//      bumped to the 1024 minimum and NUM_BLOCKS scaled to keep the
//      ~14 MiB fixture while preserving the ~14% recovery ratio
//      (32768/229376 ≈ 14.3%, RECOVERY/NUM_BLOCKS = 2048/14336 ≈ 14.3%).
// ============================================================================

var fs = require("fs");
var path = require("path");
var os = require("os");
var crypto = require("crypto");
var assert = require("node:assert");

var par3gen = require("../lib/par3gen.js");

var passed = 0;
var failed = 0;
function pass(name) { passed++; console.log("  PASS: " + name); }
function fail(name) { failed++; console.error("  FAIL: " + name); process.exitCode = 1; }

function extractRecBodies(buf) {
	var out = [];
	var off = 0;
	while (off + 48 <= buf.length) {
		if (buf.slice(off, off + 8).toString("latin1") !== "PAR3\u0000PKT") break;
		var len = Number(buf.readBigUInt64LE(off + 24));
		var type = buf.slice(off + 40, off + 48).toString("latin1");
		if (type === "PAR REC\u0000") out.push(buf.slice(off + 48, off + len));
		off += len;
	}
	return Buffer.concat(out);
}

function createWith(cap, inFilePath, outBase, BLOCK_SIZE, RECOVERY, cb) {
	var prev = process.env.PAR3_SIMULATED_BUFFER_CAP;
	if (cap === null) delete process.env.PAR3_SIMULATED_BUFFER_CAP;
	else process.env.PAR3_SIMULATED_BUFFER_CAP = String(cap);
	par3gen.create([inFilePath], outBase, {
		blockSize: BLOCK_SIZE,
		recoverySlices: RECOVERY
	}, function(err) {
		if (prev === undefined) delete process.env.PAR3_SIMULATED_BUFFER_CAP;
		else process.env.PAR3_SIMULATED_BUFFER_CAP = prev;
		cb(err);
	});
}

function runLeg(NUM_BLOCKS, BLOCK_SIZE, RECOVERY, capSweep, options, cb) {
	options = options || {};
	var legLabel = options.label || "leg";
	var inputBytes = NUM_BLOCKS * BLOCK_SIZE;

	var tmp = fs.mkdtempSync(path.join(os.tmpdir(), "chunked-" + legLabel + "-"));
	var inFile = path.join(tmp, "input.bin");
	var fd = fs.openSync(inFile, "w");
	var buf = crypto.randomBytes(BLOCK_SIZE);
	for (var i = 0; i < NUM_BLOCKS; i++) fs.writeSync(fd, buf, 0, BLOCK_SIZE);
	fs.closeSync(fd);

	console.log("\n=== leg: " + legLabel + " (NUM_BLOCKS=" + NUM_BLOCKS + ", BLOCK_SIZE=" + BLOCK_SIZE + ", RECOVERY=" + RECOVERY + ", inputBytes=" + inputBytes + ") ===");

	function cleanup() {
		try {
			fs.readdirSync(tmp).forEach(function(f) {
				fs.unlinkSync(path.join(tmp, f));
			});
			fs.rmdirSync(tmp);
		} catch (e) { /* best effort */ }
	}

	createWith(null, inFile, path.join(tmp, "base"), BLOCK_SIZE, RECOVERY, function(err1) {
		if (err1) { fail(legLabel + " base (unchunked) create: " + err1.message); cleanup(); return cb(); }
		var baseRecs = extractRecBodies(fs.readFileSync(path.join(tmp, "base.par3")));
		if (baseRecs.length === 0) { fail(legLabel + " no REC bodies in base archive"); cleanup(); return cb(); }

		var pending = capSweep.length;
		var allOk = true;
		capSweep.forEach(function(cap) {
			createWith(cap, inFile, path.join(tmp, "c" + cap), BLOCK_SIZE, RECOVERY, function(err2) {
				var out = path.join(tmp, "c" + cap + ".par3");
				if (err2) {
					fail(legLabel + " chunked cap=" + cap + ": " + err2.message);
					allOk = false;
				} else {
					var chunkedRecs = extractRecBodies(fs.readFileSync(out));
					if (!baseRecs.equals(chunkedRecs)) {
						fail(legLabel + " REC bodies differ at cap=" + cap + " (base=" + baseRecs.length + ", chunked=" + chunkedRecs.length + ")");
						allOk = false;
					}
				}
				if (--pending === 0) {
					if (allOk) pass(legLabel + " chunked REC bodies == unchunked for caps " + JSON.stringify(capSweep));
					cleanup();
					return cb();
				}
			});
		});
	});
}

// Leg 1: existing 4 MiB leg — UNCHANGED cap sweep, UNCHANGED expectations
runLeg(1024, 4096, 32, [32768, 65536, 131072, 262144, 1048576, 50000 /* not-mult-of-blockSize */], {
	label: "4MiB"
}, function() {
	// Leg 2: 14 MiB-class leg — exercises multi-chunk geometry at
	// cap=1048576, 524288, 262144. RECOVERY=2048 keeps the
	// ~14% recovery ratio of the original 32768/229376 geometry.
	runLeg(14336, 1024, 2048, [1048576, 524288, 262144], {
		label: "14MiB"
	}, function() {
		console.log("\n" + passed + " passed, " + failed + " failed");
		// par3gen's coeffWorker / N-API binding keep the event loop alive
		// after the last callback, so without this the script would hang
		// past CI's step timeout. Matches test/par3-recovery-perf.js.
		process.exit(failed === 0 ? 0 : 1);
	});
});