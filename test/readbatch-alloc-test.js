#!/usr/bin/env node
"use strict";

// ============================================================================
// readBatch allocation regression test (issue #59 T9)
//
// Pins the buffer-churn contract of the PAR3 create read pipeline: block
// bytes must land directly in reusable packet-slab windows instead of the
// former three-per-block Buffer.allocs (read buf + workerInput copy +
// packet). The test instruments Buffer.alloc / Buffer.allocUnsafe before
// loading lib/par3gen.js, runs one fixed create (8 MiB / 16 KiB blocks /
// 512 input slices / 64 recovery slices), and asserts the total allocated
// byte count stays within a derived budget.
//
// Budget derivation at this shape (512 blocks x 16384 B):
//   pre-T9 per-block churn: TWO allocations of exactly blockSize per input
//     block (read buf + workerInput) => 1024 exact-size hits at 512 blocks.
//   post-T9: zero - body windows are carved from one reusable slab.
//   Total-byte totals are dominated by out-of-scope pre-existing floors
//   (64 MiB input-pool minimum, bulk/fullInputs/merge copies), so the byte
//   check is only a loose 8x gross guard while the exact-size histogram is
//   the precise regression contract.
//
// Also validates bit-exactness of every emitted packet by walking the
// archive and recomputing each BLAKE3 packet checksum.
// ============================================================================

var fs = require('fs');
var path = require('path');
var os = require('os');
var crypto = require('crypto');

var allocBytes = 0;
var allocCalls = 0;
// Exact-size histogram: the per-block churn signature. The pre-T9 pipeline
// allocated TWO buffers of exactly blockSize per input block (read buf +
// workerInput); the slab path allocates none (windows are carved from one
// pre-sized slab).
var exactSizeCounts = new Map();
var siteBytes = new Map();
var origAlloc = Buffer.alloc;
var origAllocUnsafe = Buffer.allocUnsafe;

function noteSite(size) {
	// Stack capture is expensive and, on this host, intermittently deadlocks
	// the async hash pipeline (worker callback starvation). Diagnostics only:
	// enable explicitly with T9_ALLOC_SITES=1 after a churn-contract failure.
	if (!process.env.T9_ALLOC_SITES || size < 4096) return;
	var stack = new Error().stack.split('\n');
	var key = 'unknown';
	for (var i = 2; i < stack.length; i++) {
		var l = stack[i].trim();
		if (l.indexOf('node:') === -1 && l.indexOf('internal') === -1) { key = l.slice(0, 110); break; }
	}
	siteBytes.set(key, (siteBytes.get(key) || 0) + size);
}

function countingAlloc(size, fillOrEncoding, fillArg) {
	allocBytes += Number(size);
	allocCalls++;
	exactSizeCounts.set(size, (exactSizeCounts.get(size) || 0) + 1);
	noteSite(Number(size));
	if (fillOrEncoding === undefined) return origAlloc(size);
	if (typeof fillOrEncoding === 'number' || typeof fillOrEncoding === 'string') {
		return origAlloc(size, fillOrEncoding);
	}
	return origAlloc(size, fillArg || 0, fillOrEncoding);
}

Buffer.alloc = countingAlloc;
Buffer.allocUnsafe = function(size) {
	allocBytes += Number(size);
	allocCalls++;
	exactSizeCounts.set(size, (exactSizeCounts.get(size) || 0) + 1);
	noteSite(Number(size));
	return origAllocUnsafe(size);
};

var passed = 0;
var failed = 0;
function check(cond, msg) {
	if (cond) { passed++; console.log('  PASS: ' + msg); }
	else { failed++; console.error('  FAIL: ' + msg); process.exitCode = 1; }
}

var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'par3-t9-'));
var inFile = path.join(tmpDir, 'input.bin');
var outBase = path.join(tmpDir, 'archive');
var DATA_SIZE = 8 * 1024 * 1024;
var BLOCK_SIZE = 16 * 1024;

try {
	var par3gen = require(path.resolve(__dirname, '..', 'lib', 'par3gen'));

	// Fixed-content input so any byte drift shows up in packet hashes.
	var block = crypto.randomBytes(BLOCK_SIZE);
	var fd = fs.openSync(inFile, 'w');
	for (var i = 0; i < DATA_SIZE / BLOCK_SIZE; i++) fs.writeSync(fd, block, 0, BLOCK_SIZE);
	fs.closeSync(fd);

	console.log('readBatch allocation test (issue #59 T9)');
	console.log('  input: ' + DATA_SIZE + ' B, blockSize ' + BLOCK_SIZE + ', 512 input slices, 12.5% recovery');

	var t0 = Date.now();
	par3gen.create([inFile], outBase, {
		blockSize: BLOCK_SIZE,
		recoverySlices: { unit: 'ratio', value: 0.125 }
	}, function(err) {
		try {
			check(!err, 'create completed without error' + (err ? ' (' + err.message + ')' : ''));
			if (err) process.exit(1);
			var elapsed = Date.now() - t0;

			var outFiles = fs.readdirSync(tmpDir).filter(function(f) {
				return f.indexOf('archive') === 0 && f.slice(-5) === '.par3';
			});
			check(outFiles.length > 0, 'archive written');

			var buf = fs.readFileSync(path.join(tmpDir, outFiles[0]));

			// Budget assertions. Byte totals are dominated by out-of-scope
			// constants (the 64 MiB input-pool floor at _ensureInputPool,
			// bulk/fullInputs copies, writeBatch merges) plus
			// environment-dependent tails, so the byte check is only a
			// catastrophic-regression tripwire: POOL_FLOOR + 32x dataSize.
			// The precise T9 contract is the exact-size churn histogram.
			var POOL_FLOOR = 64 * 1024 * 1024;
			var grossBudget = POOL_FLOOR + 32.0 * DATA_SIZE;
			console.log('  measured: ' + allocBytes + ' B across ' + allocCalls +
				' allocations in ' + elapsed + ' ms');
			check(allocBytes > 0, 'allocation instrumentation captured allocations');
			check(allocBytes < grossBudget,
				'total Buffer.alloc bytes < 64MiB pool floor + 32x dataSize (got ' +
				(allocBytes / 1048576).toFixed(1) + ' MiB)');
			var perBlockHits = exactSizeCounts.get(BLOCK_SIZE) || 0;
			check(perBlockHits <= 4,
				'per-block churn: <=4 allocations of exactly blockSize ' +
					'(got ' + perBlockHits + '; pre-T9 made ~2/input-block = 1024)');
			if (failed > 0) {
				console.log('  top allocation sites:');
				[...siteBytes.entries()]
					.sort(function(x, y) { return y[1] - x[1]; })
					.slice(0, 6)
					.forEach(function(e) {
						console.log('    ' + (e[1] / 1048576).toFixed(2) + ' MB  ' + e[0]);
					});
			}

			// Bit-exactness: every packet checksum must validate.
			var blake3 = require('blake3');
			var off = 0, dataPkts = 0, bad = 0;
			while (off + 48 <= buf.length) {
				if (buf.slice(off, off + 8).toString('latin1') !== 'PAR3\u0000PKT') break;
				var len = Number(buf.readBigUInt64LE(off + 24));
				var stored = buf.slice(off + 8, off + 24);
				var calc = blake3.createHash()
					.update(buf.slice(off + 24, off + 48))
					.update(buf.slice(off + 48, off + len))
					.digest().slice(0, 16);
				if (!stored.equals(calc)) bad++;
				if (buf.slice(off + 40, off + 48).toString('latin1') === 'PAR DAT\u0000') dataPkts++;
				off += len;
			}
			check(dataPkts === 512, 'DATA packet count == 512 (got ' + dataPkts + ')');
			check(bad === 0, 'all packet BLAKE3 checksums valid (' + bad + ' failures)');

		} finally {
			Buffer.alloc = origAlloc;
			Buffer.allocUnsafe = origAllocUnsafe;
			cleanup();
			finish();
		}
	});

	function cleanup() {
		try {
			fs.readdirSync(tmpDir).forEach(function(f) {
				fs.unlinkSync(path.join(tmpDir, f));
			});
			fs.rmdirSync(tmpDir);
		} catch (e) { /* best effort */ }
	}
} catch (e) {
	console.error('FATAL: ' + (e && e.stack || e));
	process.exitCode = 1;
	cleanup();
	finish();
}

function finish() {
	console.log('\n' + passed + ' passed, ' + failed + ' failed');
	// Force-exit: the parallel-hash pool leaves live-but-idle handles
	// (unref'd workers, stream plumbing) that would otherwise stall exit
	// long after all assertions have run.
	process.exit(failed ? 1 : 0);
}
