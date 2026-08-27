#!/usr/bin/env node
"use strict";

// ============================================================================
// test/par3-chunked-inputs.js — T11 (issue #91) chunked-input path

// CI regression test.
//
// Forces the chunked-input code path via PAR3_SIMULATED_BUFFER_CAP and
// verifies the chunked output is bit-equal to the single-call output
// (XOR-linearity: f(A xor B) = f(A) xor f(B) over GF(2^64)).
//
// Uses REC packet-body comparison (not full-archive comparison) to
// avoid the legitimate per-create variation in STA timestamp / UNX mtime
// that also breaks test/par3-create-streaming-parity leg2 on master.
// ============================================================================

var fs = require("fs");
var path = require("path");
var os = require("os");
var crypto = require("crypto");
var assert = require("node:assert");

var par3gen = require("../lib/par3gen.js");

var NUM_BLOCKS = 1024;
var BLOCK_SIZE = 4096;
var RECOVERY = 32;

var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "chunked-"));
var inFile = path.join(tmpDir, "input.bin");

var passed = 0;
var failed = 0;
function pass(name) { passed++; console.log("  PASS: " + name); }
function fail(name) { failed++; console.error("  FAIL: " + name); process.exitCode = 1; }

process.on("exit", function() {
	try {
		fs.readdirSync(tmpDir).forEach(function(f) {
			fs.unlinkSync(path.join(tmpDir, f));
		});
		fs.rmdirSync(tmpDir);
	} catch (e) { /* best effort */ }
});

function makeInput() {
	var fd = fs.openSync(inFile, "w");
	var buf = crypto.randomBytes(BLOCK_SIZE);
	for (var i = 0; i < NUM_BLOCKS; i++) fs.writeSync(fd, buf, 0, BLOCK_SIZE);
	fs.closeSync(fd);
}

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

function createWith(cap, outBase, cb) {
	var prev = process.env.PAR3_SIMULATED_BUFFER_CAP;
	if (cap === null) delete process.env.PAR3_SIMULATED_BUFFER_CAP;
	else process.env.PAR3_SIMULATED_BUFFER_CAP = String(cap);
	par3gen.create([inFile], outBase, {
		blockSize: BLOCK_SIZE,
		recoverySlices: RECOVERY
	}, function(err) {
		if (prev === undefined) delete process.env.PAR3_SIMULATED_BUFFER_CAP;
		else process.env.PAR3_SIMULATED_BUFFER_CAP = prev;
		cb(err);
	});
}

makeInput();

createWith(null, path.join(tmpDir, "base"), function(err1) {
	if (err1) { fail("base (unchunked) create: " + err1.message); return done(); }
	var baseRecs = extractRecBodies(fs.readFileSync(path.join(tmpDir, "base.par3")));
	if (baseRecs.length === 0) { fail("no REC bodies in base archive"); return done(); }

	var caps = [32768, 65536, 131072, 262144, 1048576, 50000 /* not-mult-of-blockSize */];
	var pending = caps.length;
	var allOk = true;
	caps.forEach(function(cap) {
		createWith(cap, path.join(tmpDir, "c" + cap), function(err2) {
			var out = path.join(tmpDir, "c" + cap + ".par3");
			if (err2) {
				fail("chunked cap=" + cap + ": " + err2.message);
				allOk = false;
			} else {
				var chunkedRecs = extractRecBodies(fs.readFileSync(out));
				if (!baseRecs.equals(chunkedRecs)) {
					fail("REC bodies differ at cap=" + cap + " (base=" + baseRecs.length + ", chunked=" + chunkedRecs.length + ")");
					allOk = false;
				}
			}
			if (--pending === 0) {
				if (allOk) pass("chunked REC bodies == unchunked for caps " + JSON.stringify(caps) + " (incl. non-aligned 50000)");
				return done();
			}
		});
	});
});

function done() {
	console.log("\n" + passed + " passed, " + failed + " failed");
}
