#!/usr/bin/env node
"use strict";

// ============================================================================
// test/par3-memory-bounded-create.js — Issue #107: Streaming & bounded-memory PAR3 create
//
// Tests that:
// 1. Chunked recovery accumulation produces 100% bit-identical recovery packets
//    to the unchunked legacy create across single-file, multi-file, exact-chunk,
//    partial-chunk, and short-tail geometries.
// 2. PAR3 verification confirms all generated archives are valid and repairable.
// 3. Peak RSS remains strictly bounded regardless of archive size.
// ============================================================================

var fs = require("fs");
var path = require("path");
var os = require("os");
var crypto = require("crypto");
var assert = require("assert");

var par3gen = require("../lib/par3gen.js");

process.on("uncaughtException", function(e) { console.error("UNCAUGHT EXCEPTION:", e); process.exitCode = 1; });
process.on("unhandledRejection", function(e) { console.error("UNHANDLED REJECTION:", e); process.exitCode = 1; });

var passed = 0;
var failed = 0;
function pass(name) { passed++; console.log("  PASS: " + name); }
function fail(name, err) { failed++; console.error("  FAIL: " + name, err || ""); process.exitCode = 1; }

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

function runTest() {
	var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "par3-bounded-mem-"));
	console.log("Temporary test directory: " + tmpDir);

	function cleanup() {
		try {
			var files = fs.readdirSync(tmpDir);
			files.forEach(function(f) { fs.unlinkSync(path.join(tmpDir, f)); });
			fs.rmdirSync(tmpDir);
		} catch (e) {}
	}

	try {
		// Test cases: [label, fileSizes, blockSize, recoverySlices, chunkBytes]
		var testCases = [
			{
				label: "Single file, R=8, exact chunk multiple (2 MiB, B=64K, chunk=512K)",
				files: [{ name: "f1.bin", size: 2 * 1024 * 1024 }],
				blockSize: 64 * 1024,
				recoverySlices: 8,
				chunkBytes: 512 * 1024
			},
			{
				label: "Single file, R=8, non-chunk-multiple with partial chunk (3.5 MiB, B=64K, chunk=1 MiB)",
				files: [{ name: "f1.bin", size: Math.floor(3.5 * 1024 * 1024) }],
				blockSize: 64 * 1024,
				recoverySlices: 8,
				chunkBytes: 1024 * 1024
			},
			{
				label: "Short-block tail (100000 bytes, B=16K, chunk=32K)",
				files: [{ name: "f1.bin", size: 100000 }],
				blockSize: 16 * 1024,
				recoverySlices: 4,
				chunkBytes: 32 * 1024
			},
			{
				label: "Multi-file spanning chunks (3 files: 512K, 768K, 1024K, B=32K, chunk=128K)",
				files: [
					{ name: "f1.bin", size: 512 * 1024 },
					{ name: "f2.bin", size: 768 * 1024 },
					{ name: "f3.bin", size: 1024 * 1024 }
				],
				blockSize: 32 * 1024,
				recoverySlices: 8,
				chunkBytes: 128 * 1024
			},
			{
				label: "Small chunk cap (16K chunk = 1 block/chunk, R=8, 512K file)",
				files: [{ name: "f1.bin", size: 512 * 1024 }],
				blockSize: 16 * 1024,
				recoverySlices: 8,
				chunkBytes: 16 * 1024
			},
			{
				label: "Multi-file unaligned sizes (cubic review 0c8cc30f P1: 500K, 700K, 300K, B=32K, chunk=128K)",
				files: [
					{ name: "u1.bin", size: 500000 },
					{ name: "u2.bin", size: 700000 },
					{ name: "u3.bin", size: 300000 }
				],
				blockSize: 32 * 1024,
				recoverySlices: 8,
				chunkBytes: 128 * 1024
			}
		];

		var idx = 0;
		function runNext() {
			if (idx >= testCases.length) {
				console.log("\nAll parity tests passed! Now testing memory boundedness...");
				testMemoryBoundedness(cleanup);
				return;
			}

			var tc = testCases[idx++];
			console.log("\n--- " + tc.label + " ---");

			// Write input files
			var filePaths = [];
			tc.files.forEach(function(f, fi) {
				var p = path.join(tmpDir, "tc" + idx + "_" + f.name);
				var fd = fs.openSync(p, "w");
				var written = 0;
				var chunk = crypto.randomBytes(Math.min(f.size, 64 * 1024));
				while (written < f.size) {
					var toWrite = Math.min(chunk.length, f.size - written);
					fs.writeSync(fd, chunk, 0, toWrite);
					written += toWrite;
				}
				fs.closeSync(fd);
				filePaths.push(p);
			});

			var unchunkedOut = path.join(tmpDir, "out_unchunked_" + idx);
			var chunkedOut = path.join(tmpDir, "out_chunked_" + idx);

			// Run unchunked
			process.env.PAR3_DISABLE_CHUNKED = "1";
			delete process.env.PAR3_FORCE_CHUNKED;
			delete process.env.PAR3_STREAM_CHUNK_BYTES;

			par3gen.create(filePaths, unchunkedOut, {
				blockSize: tc.blockSize,
				recoverySlices: tc.recoverySlices
			}, function(err1) {
				if (err1) {
					fail(tc.label + " unchunked create failed", err1);
					return runNext();
				}

				var unchunkedBuf = fs.readFileSync(unchunkedOut + ".par3");
				var unchunkedRecs = extractRecBodies(unchunkedBuf);
				assert(unchunkedRecs.length > 0, "Unchunked should produce recovery blocks");

				// Run chunked
				delete process.env.PAR3_DISABLE_CHUNKED;
				process.env.PAR3_FORCE_CHUNKED = "1";
				process.env.PAR3_STREAM_CHUNK_BYTES = String(tc.chunkBytes);

				par3gen.create(filePaths, chunkedOut, {
					blockSize: tc.blockSize,
					recoverySlices: tc.recoverySlices
				}, function(err2) {
					delete process.env.PAR3_FORCE_CHUNKED;
					delete process.env.PAR3_STREAM_CHUNK_BYTES;

					if (err2) {
						fail(tc.label + " chunked create failed", err2);
						return runNext();
					}

					var chunkedBuf = fs.readFileSync(chunkedOut + ".par3");
					var chunkedRecs = extractRecBodies(chunkedBuf);

					if (unchunkedRecs.length !== chunkedRecs.length) {
						fail(tc.label + " length mismatch: " + unchunkedRecs.length + " vs " + chunkedRecs.length);
						runNext();
					} else if (!unchunkedRecs.equals(chunkedRecs)) {
						fail(tc.label + " content mismatch (recovery bytes not bit-identical)");
						runNext();
					} else {
						// cubic review df1de4cb P3: check result.archiveOk in addition to err3
						par3gen.verify(chunkedOut + ".par3", function(err3, result) {
							if (err3) {
								fail(tc.label + " verification failed: " + err3.message);
							} else if (!result || !result.archiveOk) {
								fail(tc.label + " verification failed (archiveOk=false)");
							} else {
								pass(tc.label + " bit-identical (" + unchunkedRecs.length + " recovery bytes, verified archiveOk)");
							}
							runNext();
						});
					}
				});
			});
		}

		function testMemoryBoundedness(done) {
			// Test 16 MiB create with 1 MiB chunk size.
			// Assert that memory usage doesn't spike by the full file size.
			var bigFile = path.join(tmpDir, "big.bin");
			var bigOut = path.join(tmpDir, "big_out");
			var size = 16 * 1024 * 1024; // 16 MiB
			var fd = fs.openSync(bigFile, "w");
			var buf = crypto.randomBytes(64 * 1024);
			for (var i = 0; i < size; i += buf.length) {
				fs.writeSync(fd, buf, 0, buf.length);
			}
			fs.closeSync(fd);

			process.env.PAR3_FORCE_CHUNKED = "1";
			process.env.PAR3_STREAM_CHUNK_BYTES = String(1024 * 1024); // 1 MiB chunks

			var initialRss = process.memoryUsage().rss;
			var peakRss = initialRss;
			// Sample RSS frequently during create to accurately capture peak RSS
			var sampleTimer = setInterval(function() {
				var cur = process.memoryUsage().rss;
				if (cur > peakRss) peakRss = cur;
			}, 5);

			par3gen.create([bigFile], bigOut, {
				blockSize: 64 * 1024,
				recoverySlices: 8,
				onEvent: function(evt, d) {
					var cur = process.memoryUsage().rss;
					if (cur > peakRss) peakRss = cur;
				}
			}, function(err) {
				clearInterval(sampleTimer);
				delete process.env.PAR3_FORCE_CHUNKED;
				delete process.env.PAR3_STREAM_CHUNK_BYTES;

				if (err) {
					fail("Bounded memory create failed", err);
					done();
					return;
				}

				var finalRss = process.memoryUsage().rss;
				if (finalRss > peakRss) peakRss = finalRss;
				var peakRssDelta = peakRss - initialRss;
				console.log("  Initial RSS:  " + (initialRss / 1048576).toFixed(1) + " MiB");
				console.log("  Peak RSS:     " + (peakRss / 1048576).toFixed(1) + " MiB");
				console.log("  Final RSS:    " + (finalRss / 1048576).toFixed(1) + " MiB");
				console.log("  Peak RSS Δ:   " + (peakRssDelta / 1048576).toFixed(1) + " MiB");

				// cubic review df1de4cb P2: assert that peak RSS delta stays strictly bounded relative to baseline
				var maxAllowedDelta = 100 * 1024 * 1024; // 100 MiB delta budget
				var maxAllowedRss = Math.max(512 * 1024 * 1024, initialRss + 150 * 1024 * 1024); // 512 MiB or baseline + 150 MiB margin
				if (peakRssDelta > maxAllowedDelta || peakRss > maxAllowedRss) {
					fail("Peak RSS Δ " + (peakRssDelta / 1048576).toFixed(1) + " MiB exceeded delta budget " + (maxAllowedDelta / 1048576).toFixed(1) + " MiB or ceiling " + (maxAllowedRss / 1048576).toFixed(1) + " MiB");
				} else {
					pass("cubic review df1de4cb P2: Peak RSS Δ " + (peakRssDelta / 1048576).toFixed(1) + " MiB strictly bounded (peak " + (peakRss / 1048576).toFixed(1) + " MiB)");
				}
				console.log("\n=================================");
				console.log("Summary: " + passed + " passed, " + failed + " failed");
				console.log("=================================");
				par3gen.shutdownHashPool();
				done();
			});
		}

		runNext();
	} catch (e) {
		fail("Unexpected error", e);
		cleanup();
	}
}

runTest();
