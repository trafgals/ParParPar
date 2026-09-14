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
var skipped = 0;
function pass(name) { passed++; console.log("  PASS: " + name); }
function fail(name, err) { failed++; console.error("  FAIL: " + name, err || ""); process.exitCode = 1; }
function skip(name) { skipped++; console.log("  SKIP: " + name); }

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
				// cubic review 59dd8dd8 P2: verify fileInfo with concurrency > 1 preserves strict input ordering
				testFileInfoConcurrency(tmpDir, function() {
					console.log("\n=================================");
					console.log("Summary: " + passed + " passed, " + failed + " failed" + (skipped > 0 ? ", " + skipped + " skipped" : ""));
					console.log("=================================");
					par3gen.shutdownHashPool();
					done();
				});
			});
		}

		function testFileInfoConcurrency(tmpDir, next) {
			console.log("\n--- fileInfo concurrency & deterministic ordering (cubic review 59dd8dd8 P2) ---");
			var count = 8;
			var fpaths = [];
			for (var i = 0; i < count; i++) {
				var p = path.join(tmpDir, "order_test_" + i + ".bin");
				fs.writeFileSync(p, crypto.randomBytes((i + 1) * 8192));
				fpaths.push(p);
			}

			par3gen.fileInfo(fpaths, false, false, 4, function(err, info) {
				if (err) {
					fail("fileInfo with concurrency=4 failed", err);
				} else if (!info || info.length !== count) {
					fail("fileInfo returned " + (info ? info.length : 0) + " items, expected " + count);
				} else {
					var orderMatches = true;
					for (var i = 0; i < count; i++) {
						if (info[i].name !== fpaths[i]) {
							orderMatches = false;
							break;
						}
					}
					if (orderMatches) {
						pass("cubic review 59dd8dd8 P2: fileInfo concurrency=4 preserves strict input order across " + count + " files");
					} else {
						fail("cubic review 59dd8dd8 P2: fileInfo returned out-of-order results");
					}
				}
				testFileInfoNonFileError(tmpDir, function() {
					testIssue113DefaultChunkBounds(tmpDir, next);
				});
			});
		}

		function testIssue113DefaultChunkBounds(tmpDir, next) {
			console.log("\n--- Issue #113 regression: default 64 MiB chunking bounds & 8-byte alignment ---");

			// Ensure all forced chunking env vars are clean to test DEFAULT behavior
			// cubic review P3: Ensure all chunking and memory-budget env vars are clean to test DEFAULT behavior
			delete process.env.PAR3_FORCE_CHUNKED;
			delete process.env.PAR3_DISABLE_CHUNKED;
			delete process.env.PAR3_STREAM_CHUNK_BYTES;
			delete process.env.PAR3_SIMULATED_BUFFER_CAP;
			delete process.env.PAR3_INPUT_CHUNK_CAP;
			delete process.env.PAR3_MEMORY_LIMIT;
			delete process.env.PAR3_BATCH_SIZE;
			delete process.env.PAR3_GF64_LEGACY_FULL;

			// 1. cubic review (PR #114 delivery 24411f3a P2): Distinguish MODULE_NOT_FOUND from broken build
			try {
				var nativeBinding = null;
				var loadError = null;
				try {
					nativeBinding = require("../build/Release/parpar_gf64.node");
				} catch(e) {
					if (e && e.code !== "MODULE_NOT_FOUND") {
						loadError = e;
					} else {
						try {
							nativeBinding = require("../build/Debug/parpar_gf64.node");
						} catch(e2) {
							if (e2 && e2.code !== "MODULE_NOT_FOUND") {
								loadError = e2;
							}
						}
					}
				}

				if (loadError) {
					fail("native addon load failed with unexpected error (broken build / dlopen failure)", loadError);
				} else if (nativeBinding && typeof nativeBinding.compute_recovery_full === "function") {
					assert.strictEqual(typeof nativeBinding.isAlignedBuffer, "function", "native binding must export isAlignedBuffer when compute_recovery_full is implemented");

					// Allocate a buffer and find an offset that is 8-byte aligned but strictly NOT 64-byte aligned
					var rawIn = Buffer.alloc(256);
					var inOffset = 8;
					while (inOffset < 128 && nativeBinding.isAlignedBuffer(rawIn.subarray(inOffset), 64)) {
						inOffset += 8;
					}
					var misalignedIn = rawIn.subarray(inOffset, inOffset + 128);
					assert(nativeBinding.isAlignedBuffer(misalignedIn, 8), "input buffer must be 8-byte aligned");
					assert(!nativeBinding.isAlignedBuffer(misalignedIn, 64), "input buffer must NOT be 64-byte aligned");
					for (var i = 0; i < misalignedIn.length; i++) misalignedIn[i] = (i * 17 + 3) & 0xff;

					var rawOut = Buffer.alloc(256);
					var outOffset = 8;
					while (outOffset < 128 && nativeBinding.isAlignedBuffer(rawOut.subarray(outOffset), 64)) {
						outOffset += 8;
					}
					var misalignedOut = rawOut.subarray(outOffset, outOffset + 128);
					assert(nativeBinding.isAlignedBuffer(misalignedOut, 8), "output buffer must be 8-byte aligned");
					assert(!nativeBinding.isAlignedBuffer(misalignedOut, 64), "output buffer must NOT be 64-byte aligned");

					var refOut = Buffer.alloc(128);
					var refIn = Buffer.alloc(128);
					misalignedIn.copy(refIn);

					// Compute via native binding on non-64-byte aligned buffers (N=2, R=2, B=64)
					nativeBinding.compute_recovery_full(misalignedIn, misalignedOut, 2, 2, 64, 0, 2, 1, false);
					nativeBinding.compute_recovery_full(refIn, refOut, 2, 2, 64, 0, 2, 1, false);

					assert(misalignedOut.equals(refOut), "non-64-byte aligned buffer output must match reference output bit-exact");
					pass("cubic review 24411f3a P2: 8-byte aligned (strictly NOT 64-byte aligned) buffers execute bit-identically in native addon without requiring 64-byte alignment");
				} else {
					skip("native x86 addon not available for 8-byte alignment verification (MODULE_NOT_FOUND or non-x86 stub); proceeding with bounded create test");
				}
			} catch(errAlign) {
				fail("Issue #113 alignment verification error", errAlign);
			}

			// 2. Test default chunked create with 70 MiB file (> 64 MiB default chunkCapBytes)
			var file70M = path.join(tmpDir, "issue113_70M.bin");
			var out70M = path.join(tmpDir, "issue113_70M_out");
			var size70M = 70 * 1024 * 1024; // 70 MiB
			var fd = fs.openSync(file70M, "w");
			var chunk = crypto.randomBytes(64 * 1024);
			for (var w = 0; w < size70M; w += chunk.length) {
				fs.writeSync(fd, chunk, 0, Math.min(chunk.length, size70M - w));
			}
			fs.closeSync(fd);

			var initialRss = process.memoryUsage().rss;
			var peakRss = initialRss;
			var sampleTimer = setInterval(function() {
				var cur = process.memoryUsage().rss;
				if (cur > peakRss) peakRss = cur;
			}, 5);

			par3gen.create([file70M], out70M, {
				blockSize: 64 * 1024,
				recoverySlices: 8,
				onEvent: function(evt, d) {
					var cur = process.memoryUsage().rss;
					if (cur > peakRss) peakRss = cur;
				}
			}, function(err) {
				clearInterval(sampleTimer);
				if (err) {
					fail("Issue #113: 70 MiB default create failed", err);
					next();
					return;
				}

				var peakRssDelta = peakRss - initialRss;
				console.log("  Default 70M create Initial RSS: " + (initialRss / 1048576).toFixed(1) + " MiB");
				console.log("  Default 70M create Peak RSS:    " + (peakRss / 1048576).toFixed(1) + " MiB");
				console.log("  Default 70M create Peak RSS Δ:  " + (peakRssDelta / 1048576).toFixed(1) + " MiB");

				// Sized to 64 MiB chunk: peak RSS delta must not spike by the full 70 MiB + full recovery
				var maxAllowedDelta = 120 * 1024 * 1024; // 120 MiB ceiling
				if (peakRssDelta > maxAllowedDelta) {
					fail("Issue #113: Peak RSS Δ " + (peakRssDelta / 1048576).toFixed(1) + " MiB exceeded budget " + (maxAllowedDelta / 1048576).toFixed(1) + " MiB");
				} else {
					pass("Issue #113: 70 MiB default create bounded to 64 MiB chunks with Peak RSS Δ " + (peakRssDelta / 1048576).toFixed(1) + " MiB");
				}

				par3gen.verify(out70M + ".par3", function(errV, resV) {
					if (errV || !resV || !resV.archiveOk) {
						fail("Issue #113: 70 MiB archive verification failed", errV);
					} else {
						pass("Issue #113: 70 MiB archive verified successfully (archiveOk=true)");
					}
					testIssue115AdaptiveChunkingAndInputDecomp(tmpDir, next);
				});
			});
		}

		function testIssue115AdaptiveChunkingAndInputDecomp(tmpDir, next) {
			console.log("\n--- Issue #115 regression: adaptive chunk sizing & input-domain decomposition ---");

			// 1. Verify native input-domain adaptive worker sizing
			var nativeBinding = null;
			try {
				nativeBinding = require("../build/Release/parpar_gf64.node");
			} catch (e) {
				try {
					nativeBinding = require("../build/Debug/parpar_gf64.node");
				} catch (e2) {}
			}

			if (nativeBinding && typeof nativeBinding.compute_recovery_full === "function" && typeof nativeBinding.get_last_decomposition_path === "function") {
				var N = 16, R = 8, B = 1024 * 1024;
				var inBuf = Buffer.alloc(N * B);
				var outBuf = Buffer.alloc(R * B);

				// Under default 128 MiB scratch cap (pinned explicitly for 32-bit compatibility per Cubic review P2),
				// 8 * 1 MiB * 15 = 120 MiB <= 128 MiB fits in input decomp
				process.env.PAR3_INPUT_DECOMP_SCRATCH_BYTES = (128 * 1024 * 1024).toString();
				nativeBinding.compute_recovery_full(inBuf, outBuf, N, R, B, 0, N, 16, false);
				var pathDefault = nativeBinding.get_last_decomposition_path();
				delete process.env.PAR3_INPUT_DECOMP_SCRATCH_BYTES;
				if (pathDefault === 2) {
					pass("Issue #115: 120 MiB scratch under 128 MiB cap engages input-domain decomposition (path 2)");
				} else {
					fail("Issue #115: expected path 2 under 128 MiB cap, got path " + pathDefault);
				}

				// With 32 MiB scratch cap, 120 MiB exceeds budget and max workers (5) <= R (8) -> falls back to path 3
				process.env.PAR3_INPUT_DECOMP_SCRATCH_BYTES = "33554432";
				nativeBinding.compute_recovery_full(inBuf, outBuf, N, R, B, 0, N, 16, false);
				var pathCapped = nativeBinding.get_last_decomposition_path();
				delete process.env.PAR3_INPUT_DECOMP_SCRATCH_BYTES;
				if (pathCapped === 3) {
					pass("Issue #115: oversized scratch under 32 MiB cap safely falls back to output-domain decomposition (path 3)");
				} else {
					fail("Issue #115: expected path 3 under 32 MiB cap, got path " + pathCapped);
				}
			} else {
				console.log("  SKIP: native addon not available for decomposition path check");
			}

			// Issue #117 / #119: Direct unit assertion on chunkCapBytes contract
			console.log("\n  Testing chunkCapBytes contract (Issue #117 / #119):");
			if (typeof par3gen.decideChunkCapBytes === "function") {
				var expectedCap = typeof par3gen.getOptimalChunkCap === "function"
					? par3gen.getOptimalChunkCap(null, 64 * 1024)
					: (64 * 1024 * 1024);

				// < 4 GiB
				var cap1G = par3gen.decideChunkCapBytes(1 * 1024 * 1024 * 1024, 8, 64 * 1024, "matvec");
				if (cap1G === expectedCap) {
					pass("Issue #117 / #119: < 4 GiB yields expected chunk cap (" + (expectedCap / 1048576).toFixed(1) + " MiB)");
				} else {
					fail("Issue #117 / #119: expected " + expectedCap + ", got " + cap1G);
				}

				// 4 GiB .. 16 GiB
				var cap8G = par3gen.decideChunkCapBytes(8 * 1024 * 1024 * 1024, 8, 64 * 1024, "matvec");
				if (cap8G === expectedCap) {
					pass("Issue #117 / #119: 4 GiB .. 16 GiB yields expected chunk cap (" + (expectedCap / 1048576).toFixed(1) + " MiB)");
				} else {
					fail("Issue #117 / #119: expected " + expectedCap + ", got " + cap8G);
				}

				// >= 16 GiB
				var cap32G = par3gen.decideChunkCapBytes(32 * 1024 * 1024 * 1024, 8, 64 * 1024, "matvec");
				if (cap32G === expectedCap) {
					pass("Issue #117 / #119: >= 16 GiB yields expected chunk cap (" + (expectedCap / 1048576).toFixed(1) + " MiB)");
				} else {
					fail("Issue #117 / #119: expected " + expectedCap + ", got " + cap32G);
				}

				// Explicit 96 MiB Zen4 topology yields exact 64 MiB regardless of host
				var topo96M = { l3PerCluster: 96 * 1024 * 1024 };
				var capExplicit96M = par3gen.decideChunkCapBytes(32 * 1024 * 1024 * 1024, 8, 64 * 1024, "matvec", null, topo96M);
				if (capExplicit96M === 64 * 1024 * 1024) {
					pass("Issue #119: explicit 96 MiB L3 topology yields exact 64 MiB chunk cap");
				} else {
					fail("Issue #119: expected 64 MiB for 96M topology, got " + capExplicit96M);
				}

				// Recovery reserve deduction (matvec):
				// 384 MiB default buffer budget - (1 * 2816 * 128 KiB = 352 MiB reserve) = 32 MiB available (< 64 MiB)
				var capWithReserve = par3gen.decideChunkCapBytes(32 * 1024 * 1024 * 1024, 2816, 128 * 1024, "matvec");
				if (capWithReserve === 32 * 1024 * 1024) {
					pass("Issue #117: recovery reserve properly clamps chunk cap on 384 MiB budget");
				} else {
					fail("Issue #117: expected 32 MiB clamped cap, got " + capWithReserve);
				}

				// Recovery reserve deduction (fenger):
				// 384 MiB default buffer budget - (2 * 1408 * 128 KiB = 352 MiB reserve) = 32 MiB available (< 64 MiB)
				var capWithFengerReserve = par3gen.decideChunkCapBytes(32 * 1024 * 1024 * 1024, 1408, 128 * 1024, "fenger");
				if (capWithFengerReserve === 32 * 1024 * 1024) {
					pass("Issue #117: Fenger 2x recovery reserve properly clamps chunk cap on 384 MiB budget");
				} else {
					fail("Issue #117: expected 32 MiB clamped cap for Fenger, got " + capWithFengerReserve);
				}

				// Explicit env override
				process.env.PAR3_STREAM_CHUNK_BYTES = (1024 * 1024).toString();
				var capExplicit = par3gen.decideChunkCapBytes(32 * 1024 * 1024 * 1024, 8, 64 * 1024, "matvec");
				delete process.env.PAR3_STREAM_CHUNK_BYTES;
				if (capExplicit === 1024 * 1024) {
					pass("Cubic review P3: explicit PAR3_STREAM_CHUNK_BYTES overrides adaptive scaling");
				} else {
					fail("Cubic review P3: expected 1024 * 1024 explicit cap, got " + capExplicit);
				}

				// Cubic review P2 (round 2): Recovery reserve exceeds budget -> returns 0 (cannot fit)
				// E.g. 4096 recovery x 128 KiB = 512 MiB reserve > 384 MiB default budget
				var capExceeded = par3gen.decideChunkCapBytes(32 * 1024 * 1024 * 1024, 4096, 128 * 1024, "matvec");
				if (capExceeded === 0) {
					pass("Cubic review P2: returns 0 when recovery reserve exceeds default 384 MiB budget");
				} else {
					fail("Cubic review P2: expected 0 when reserve exceeds budget, got " + capExceeded);
				}

				// Under explicit memory limit: 64 MiB limit with 64 MiB recovery reserve -> returns 0 (cannot fit 1 block)
				var capLimitExceeded = par3gen.decideChunkCapBytes(1 * 1024 * 1024 * 1024, 1024, 64 * 1024, "matvec", 64 * 1024 * 1024);
				if (capLimitExceeded === 0) {
					pass("Cubic review P2: returns 0 when recovery reserve exceeds explicit memory limit");
				} else {
					fail("Cubic review P2: expected 0 when reserve exceeds explicit limit, got " + capLimitExceeded);
				}

				// Cubic review P1: Explicit chunk cap combined with memory limit where reserve exceeds budget
				// must return 0 unconditionally (cannot fit 1 block) instead of permitting an allocation that exceeds budget
				process.env.PAR3_STREAM_CHUNK_BYTES = (1024 * 1024).toString();
				var capExplicitWithLimitExceeded = par3gen.decideChunkCapBytes(1 * 1024 * 1024 * 1024, 1024, 64 * 1024, "matvec", 64 * 1024 * 1024);
				delete process.env.PAR3_STREAM_CHUNK_BYTES;
				if (capExplicitWithLimitExceeded === 0) {
					pass("Cubic review P1: explicit chunk cap with reserve exceeding memory limit unconditionally returns 0");
				} else {
					fail("Cubic review P1: expected 0, got " + capExplicitWithLimitExceeded);
				}
				// Cubic review P1: Explicit chunk cap larger than memoryLimit is clamped to memoryLimit / 2
				process.env.PAR3_STREAM_CHUNK_BYTES = (10 * 1024 * 1024).toString();
				var capClamped = par3gen.decideChunkCapBytes(4 * 1024 * 1024, 8, 64 * 1024, "matvec", 4 * 1024 * 1024);
				delete process.env.PAR3_STREAM_CHUNK_BYTES;
				if (capClamped === 2 * 1024 * 1024) {
					pass("Cubic review P1: explicit chunk cap larger than memoryLimit clamped to memoryLimit / 2");
				} else {
					fail("Cubic review P1: expected 2 MiB clamped cap, got " + capClamped);
				}
			} else {
				fail("Cubic review P3: par3gen.decideChunkCapBytes is not exported");
			}

			// 2. End-to-end create with 1 MiB forced chunking on 4 MiB file (assert exactly 4 chunk flushes)
			var fileLarge = path.join(tmpDir, "adaptive_chunk_test.bin");
			var outLarge = path.join(tmpDir, "adaptive_chunk_test_out");
			fs.writeFileSync(fileLarge, crypto.randomBytes(4 * 1024 * 1024));

			process.env.PAR3_FORCE_CHUNKED = "1";
			process.env.PAR3_STREAM_CHUNK_BYTES = (1024 * 1024).toString();

			var actualFlushes = 0;
			par3gen.create([fileLarge], outLarge, {
				blockSize: 64 * 1024,
				recoverySlices: 8,
				onEvent: function(evt, d) {
					if (evt === "chunk_flush") actualFlushes++;
				}
			}, function(err) {
				delete process.env.PAR3_FORCE_CHUNKED;
				delete process.env.PAR3_STREAM_CHUNK_BYTES;

				if (err) {
					fail("Issue #115: chunked create failed", err);
					next();
					return;
				}
				pass("Issue #115: chunked create completed without error");

				if (actualFlushes === 4) {
					pass("Cubic review P3: 4 MiB input with 1 MiB chunk size performed exactly 4 chunk flushes");
				} else {
					fail("Cubic review P3: expected 4 chunk flushes, got " + actualFlushes);
				}

				par3gen.verify(outLarge + ".par3", function(errV, resV) {
					if (errV || !resV || !resV.archiveOk) {
						fail("Issue #115: chunked archive verification failed", errV);
						next();
						return;
					}
					pass("Issue #115: chunked archive verified successfully (archiveOk=true)");

					// 3. Cubic review P2: Workload where recovery reserve exceeds memory limit routes cleanly to per-batch path
					// (Note: routing to per-batch is decided solely by recoveryReserve + blockSize > memoryLimit)
					var fileBatch = path.join(tmpDir, "per_batch_route_test.bin");
					var outBatch = path.join(tmpDir, "per_batch_route_test_out");
					fs.writeFileSync(fileBatch, crypto.randomBytes(512 * 1024));

					// 16 recovery x 16 KiB = 256 KiB recovery reserve.
					// memoryLimit: 200 KiB (< 256 KiB reserve + 16 KiB block)
					par3gen.create([fileBatch], outBatch, {
						blockSize: 16 * 1024,
						recoverySlices: 16,
						memoryLimit: 200 * 1024
					}, function(errB) {
						if (errB) {
							fail("Cubic review P2: create with recovery reserve > memoryLimit failed", errB);
							next();
							return;
						}
						pass("Cubic review P2: workload with recovery reserve > memoryLimit routed cleanly to per-batch path");

						par3gen.verify(outBatch + ".par3", function(errVB, resVB) {
							if (errVB || !resVB || !resVB.archiveOk) {
								fail("Cubic review P2: per-batch archive verification failed", errVB);
								next();
								return;
							}
							pass("Cubic review P2: per-batch archive verified successfully (archiveOk=true)");

							// 4. Cubic review P1 / P3: Explicit chunk cap combined with memoryLimit actively clamps chunk size in create
							// (recoveryReserve=512 KiB + 64 KiB block <= 4 MiB memoryLimit; requested 10 MiB chunk cap is clamped to memoryLimit/2=2 MiB)
							var fileClamped = path.join(tmpDir, "expl_chunk_cap_clamped.bin");
							var outClamped = path.join(tmpDir, "expl_chunk_cap_clamped_out");
							fs.writeFileSync(fileClamped, crypto.randomBytes(4 * 1024 * 1024));

							process.env.PAR3_FORCE_CHUNKED = "1";
							process.env.PAR3_STREAM_CHUNK_BYTES = (10 * 1024 * 1024).toString();
							var clampedFlushes = 0;
							par3gen.create([fileClamped], outClamped, {
								blockSize: 64 * 1024,
								recoverySlices: 8,
								memoryLimit: 4 * 1024 * 1024,
								onEvent: function(evt, d) {
									if (evt === "chunk_flush") clampedFlushes++;
								}
							}, function(errC) {
								delete process.env.PAR3_FORCE_CHUNKED;
								delete process.env.PAR3_STREAM_CHUNK_BYTES;
								if (errC) {
									fail("Cubic review P1: create with clamped explicit chunk cap failed", errC);
									next();
									return;
								}
								pass("Cubic review P1: explicit chunk cap clamped to memoryLimit/2 created successfully");

								// 4 MiB file with 2 MiB clamped chunk size = exactly 2 flushes (not 1 flush)
								if (clampedFlushes === 2) {
									pass("Cubic review P1: 4 MiB file with 10 MiB requested cap clamped to 2 MiB performed exactly 2 flushes");
								} else {
									fail("Cubic review P1: expected 2 flushes, got " + clampedFlushes);
								}

								par3gen.verify(outClamped + ".par3", function(errVC, resVC) {
									if (errVC || !resVC || !resVC.archiveOk) {
										fail("Cubic review P1: clamped archive verification failed", errVC);
									} else {
										pass("Cubic review P1: clamped archive verified successfully (archiveOk=true)");
									}
									next();
								});
							});
						});
					});
				});
			});
		}

		function testFileInfoNonFileError(tmpDir, next) {
			console.log("\n--- fileInfo non-file error handling (cubic review 41b8069e P1) ---");
			var origStat = fs.stat;
			var testFile = path.join(tmpDir, "non_file_test.bin");
			fs.writeFileSync(testFile, "dummy");

			fs.stat = function(p, cb) {
				origStat.call(fs, p, function(err, stat) {
					if (err) return cb(err);
					if (p === testFile) {
						var fakeStat = Object.create(stat);
						fakeStat.isFile = function() { return false; };
						fakeStat.isDirectory = function() { return false; };
						fakeStat.isSymbolicLink = function() { return false; };
						return cb(null, fakeStat);
					}
					cb(null, stat);
				});
			};

			// cubic review 710c0747 P3: guard completion with finished flag so next() runs only once
			var finished = false;
			var timeout = setTimeout(function() {
				if (finished) return;
				finished = true;
				fs.stat = origStat;
				fail("cubic review 710c0747 P3: fileInfo hung on non-file (BufferPool corrupted by double-put)");
				next();
			}, 3000);

			par3gen.fileInfo([testFile], false, false, 2, function(err, info) {
				if (finished) return;
				finished = true;
				clearTimeout(timeout);
				fs.stat = origStat;
				if (err && err.message && err.message.indexOf("is not a valid file") !== -1) {
					pass("cubic review 41b8069e P1: fileInfo cleanly reports non-file error without hanging or double-put");
				} else {
					fail("cubic review 41b8069e P1: expected 'is not a valid file' error, got: " + err);
				}
				next();
			});
		}

		runNext();
	} catch (e) {
		fail("Unexpected error", e);
		cleanup();
	}
}

runTest();
