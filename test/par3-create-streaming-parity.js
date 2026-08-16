#!/usr/bin/env node
// A2-rev parity: the opt-in streaming create path (PAR3_GF64_FAST_CREATE=1)
// is now authoritative for single-file creates within the in-memory cap —
// the C++ par3_create_streaming call hands its recovery buffer to JS and the
// legacy compute is skipped.
//
// Leg 1 (binding contract): par3_create_streaming returns a recoveryBuffer
// whose content is byte-identical to the legacy compute_recovery_full output
// for the same input + matrix.
// Leg 2 (e2e parity): a FAST_CREATE=1 create produces a byte-identical .par3
// to the default create, both verify clean, and the 'complete' event reports
// usedStreamingRecovery=true (the streaming buffer actually engaged).
"use strict";
var assert = require('assert');
var path = require('path');
var fs = require('fs');
var os = require('os');
var addon = require('../build/Release/parpar_gf64.node');
var par3 = require('../lib/par3gen.js');

var BLOCK = 64 * 1024;
var NUM_BLOCKS = 64;          // 4 MiB
var RECOVERY = 8;

function makeInput(inFile) {
	var fd = fs.openSync(inFile, 'w');
	var buf = Buffer.alloc(BLOCK);
	var seed = 0x9E3779B9;
	for (var blk = 0; blk < NUM_BLOCKS; blk++) {
		for (var j = 0; j < BLOCK; j += 4) {
			seed = (seed * 1664525 + 1013904223) >>> 0;
			buf.writeUInt32LE(seed, j);
		}
		fs.writeSync(fd, buf);
	}
	fs.closeSync(fd);
}

function verifyLeg(par3File, cb) {
	par3.verify(par3File, function(err, result) {
		cb(err, result);
	});
}

var fails = 0;

// ---------------------------------------------------------------------------
// Leg 1: binding contract — streaming recovery buffer == legacy compute
// ---------------------------------------------------------------------------
function leg1(tempDir) {
	var inFile = path.join(tempDir, 'leg1.bin');
	makeInput(inFile);

	// Legacy reference: compute_recovery_full with the same points
	// (firstInput=0, firstRecovery=N) the streaming call uses.
	var totalInputBlocks = NUM_BLOCKS;
	var firstInput = 0, firstRecovery = totalInputBlocks;
	var inputs = Buffer.alloc(totalInputBlocks * BLOCK);
	var fd = fs.openSync(inFile, 'r');
	fs.readSync(fd, inputs, 0, inputs.length, 0);
	fs.closeSync(fd);
	var legacy = Buffer.alloc(RECOVERY * BLOCK);
	addon.compute_recovery_full(inputs, legacy, totalInputBlocks, RECOVERY, BLOCK, firstInput, firstRecovery, 0);

	// Streaming call (mmap disabled so the read path is uniform; save and
	// restore any pre-existing value so the env is not leaked)
	var prevMmapEnv = process.env.PAR3_GF64_USE_MMAP;
	delete process.env.PAR3_GF64_USE_MMAP;
	var streamResult = null;
	var streamErr = null;
	try {
		addon.par3_create_streaming(inFile, {
			recoverySlices: RECOVERY,
			blockSize: BLOCK,
			firstInput: firstInput,
			firstRecovery: firstRecovery,
			numThreads: 0
		}, function(err, res) {
			streamErr = err;
			streamResult = res;
		});
	} finally {
		// Restore the mmap env even if the call throws (the call is synchronous)
		if (prevMmapEnv === undefined) delete process.env.PAR3_GF64_USE_MMAP;
		else process.env.PAR3_GF64_USE_MMAP = prevMmapEnv;
	}
	assert(streamErr === null, 'streaming call failed: ' + (streamErr && streamErr.message));
	assert(streamResult && streamResult.recoveryBuffer, 'streaming result lacks recoveryBuffer');
	assert.strictEqual(streamResult.recoveryBuffer.length, RECOVERY * BLOCK, 'recoveryBuffer size mismatch');
	assert(streamResult.recoveryBuffer.equals(legacy), 'streaming recovery != legacy compute (byte-identical contract)');
	console.log('leg1 ok: streaming recoveryBuffer == legacy compute_recovery_full (' + (RECOVERY * BLOCK) + ' bytes)');
}

// ---------------------------------------------------------------------------
// Leg 2: e2e parity — FAST_CREATE=1 archive == default archive, engaged
// ---------------------------------------------------------------------------
function leg2(tempDir, cb) {
	var inFile = path.join(tempDir, 'leg2.bin');
	makeInput(inFile);

	var runCreate = function(tag, extraEnv, cb2) {
		Object.keys(extraEnv).forEach(function(k) {
			if (extraEnv[k] === null) delete process.env[k];
			else process.env[k] = extraEnv[k];
		});
		var outBase = path.join(tempDir, 'out_' + tag);
		var usedStreaming = null;
		var gen = new par3.PAR3Gen([{ name: inFile, size: fs.statSync(inFile).size }], BLOCK, {
			outputBase: outBase,
			recoverySlices: RECOVERY,
			blockSize: BLOCK
		});
		gen.run(function(ev, data) {
			if (ev === 'complete') usedStreaming = data.usedStreamingRecovery;
		}, function(err) {
			try { gen.close(); } catch (e) {}
			Object.keys(extraEnv).forEach(function(k) {
				if (extraEnv[k] === null) { /* already deleted */ }
				else delete process.env[k];
			});
			cb2(err, outBase, usedStreaming);
		});
	};

	runCreate('default', {}, function(err1, base1, used1) {
		if (err1) return cb(new Error('default create failed: ' + err1.message));
		assert.strictEqual(used1, false, 'default create must NOT report usedStreamingRecovery');

		runCreate('stream', { PAR3_GF64_FAST_CREATE: '1' }, function(err2, base2, used2) {
			if (err2) return cb(new Error('streaming create failed: ' + err2.message));
			assert.strictEqual(used2, true, 'FAST_CREATE=1 create must report usedStreamingRecovery=true');

			var a = fs.readFileSync(base1 + '.par3');
			var b = fs.readFileSync(base2 + '.par3');
			// Archives legitimately differ in the random per-create fileId
			// (crypto.randomBytes(16) in _createFilePackets) and in every
			// derived checksum (packet checksums, the ROOT packet's filelist
			// digest). Mask the checksum fields + the fileId, then require
			// byte-identity — the recovery data and input packets must match
			// exactly.
			var maskArchive = function(buf) {
				var out = Buffer.from(buf);
				var off = 0;
				while (off + 48 <= out.length) {
					if (out.slice(off, off + 8).toString('latin1') !== 'PAR3\u0000PKT') break;
					var len = Number(out.readBigUInt64LE(off + 24));
					// packet checksum: bytes 8..24 of the header
					out.fill(0, off + 8, off + 24);
					if (out.slice(off + 40, off + 48).toString('latin1') === 'PAR FIL\u0000') {
						// fileId: 16 bytes at 52 + nameLen (nameLen uint16 at 48)
						var nameLen = out.readUInt16LE(off + 48);
						out.fill(0, off + 52 + nameLen, off + 52 + nameLen + 16);
					}
					if (out.slice(off + 40, off + 48).toString('latin1') === 'PAR ROO\u0000') {
						// the ROOT packet embeds the FIL packet checksum as the
						// last 16 body bytes — derived from the random fileId
						out.fill(0, off + len - 16, off + len);
					}
					off += len;
				}
				return out;
			};
			assert(maskArchive(a).equals(maskArchive(b)), 'streaming archive != default archive after masking the random fileId');
			// The recovery DATA itself must match even unmasked — walk the
			// packets and require every non-FIL/ROO packet byte-identical.
			var off2 = 0;
			while (off2 + 48 <= a.length) {
				var len2 = Number(a.readBigUInt64LE(off2 + 24));
				var type2 = a.slice(off2 + 40, off2 + 48).toString('latin1');
				if (type2 !== 'PAR FIL\u0000' && type2 !== 'PAR ROO\u0000') {
					assert(a.slice(off2, off2 + len2).equals(b.slice(off2, off2 + len2)),
						'packet ' + JSON.stringify(type2) + ' differs unmasked');
				}
				off2 += len2;
			}

			verifyLeg(base1 + '.par3', function(e1, r1) {
				verifyLeg(base2 + '.par3', function(e2, r2) {
					try {
						assert(!e1 && r1 && r1.verified, 'default archive verify failed: ' + (e1 && e1.message));
						assert(!e2 && r2 && r2.verified, 'streaming archive verify failed: ' + (e2 && e2.message));
						console.log('leg2 ok: FAST_CREATE=1 archive byte-identical to default, both verify clean, streaming engaged');
						cb();
					} catch (e) {
						cb(e);
					}
				});
			});
		});
	});
}

// ---------------------------------------------------------------------------
// Leg 3: TOCTOU invalidation — a source whose stat changes between the
// streaming read and the JS data read must NOT be used (cubic review
// 5443592 P2 on #80). Simulated deterministically: the post-read
// fs.statSync returns a shifted mtime, forcing the guard's mismatch.
// ---------------------------------------------------------------------------
function leg3(tempDir, cb) {
	var inFile = path.join(tempDir, 'leg3.bin');
	makeInput(inFile);
	var outBase = path.join(tempDir, 'out_leg3');
	var gen = new par3.PAR3Gen([{ name: inFile, size: fs.statSync(inFile).size }], BLOCK, {
		outputBase: outBase,
		recoverySlices: RECOVERY,
		blockSize: BLOCK
	});
	process.env.PAR3_GF64_FAST_CREATE = '1';
	var origStat = fs.statSync;
	var statCalls = 0;
	// The create flow stats the file: (1) pre-streaming streamStat,
	// (2) post-read postStat. Fake the 2nd so the guard sees a change.
	fs.statSync = function(p) {
		var s = origStat(p);
		if (p === inFile && ++statCalls === 2) {
			s = Object.assign({}, s, { mtimeMs: s.mtimeMs + 5000 });
		}
		return s;
	};
	var usedStreaming = null;
	gen.run(function(ev, data) {
		if (ev === 'complete') usedStreaming = data.usedStreamingRecovery;
	}, function(err) {
		fs.statSync = origStat;
		delete process.env.PAR3_GF64_FAST_CREATE;
		try { gen.close(); } catch (e) {}
		if (err) return cb(new Error('leg3 create failed: ' + err.message));
		par3.verify(outBase + '.par3', function(verr, vres) {
			if (verr) return cb(new Error('leg3 verify failed: ' + verr.message));
			try {
				assert.strictEqual(usedStreaming, false, 'changed source must invalidate the streaming recovery');
				assert(vres.verified, 'leg3 archive must verify clean (legacy compute took over)');
				console.log('leg3 ok: mid-flight source change invalidates the streaming recovery, archive verifies clean');
				cb();
			} catch (e) {
				cb(e);
			}
		});
	});
}

// ---------------------------------------------------------------------------
// Leg 4: bulk-read failure — the per-block fallback path must invalidate
// the streaming recovery outright (cubic review 5443592 P2 on #80).
// Forced deterministically: the bulk readSync throws once, sending the
// create down the fallback path.
// ---------------------------------------------------------------------------
function leg4(tempDir, cb) {
	var inFile = path.join(tempDir, 'leg4.bin');
	makeInput(inFile);
	var outBase = path.join(tempDir, 'out_leg4');
	var gen = new par3.PAR3Gen([{ name: inFile, size: fs.statSync(inFile).size }], BLOCK, {
		outputBase: outBase,
		recoverySlices: RECOVERY,
		blockSize: BLOCK
	});
	process.env.PAR3_GF64_FAST_CREATE = '1';
	var origRead = fs.readSync;
	var threw = false;
	fs.readSync = function(fd, buf, off, len, pos) {
		if (!threw && len > 1024 * 1024) {
			threw = true;
			throw new Error('simulated bulk-read failure');
		}
		return origRead.apply(this, arguments);
	};
	var usedStreaming = null;
	gen.run(function(ev, data) {
		if (ev === 'complete') usedStreaming = data.usedStreamingRecovery;
	}, function(err) {
		fs.readSync = origRead;
		delete process.env.PAR3_GF64_FAST_CREATE;
		try { gen.close(); } catch (e) {}
		if (err) return cb(new Error('leg4 create failed: ' + err.message));
		par3.verify(outBase + '.par3', function(verr, vres) {
			if (verr) return cb(new Error('leg4 verify failed: ' + verr.message));
			try {
				assert(threw, 'the simulated bulk-read failure must have engaged');
				assert.strictEqual(usedStreaming, false, 'bulk-read failure must invalidate the streaming recovery');
				assert(vres.verified, 'leg4 archive must verify clean (fallback path)');
				console.log('leg4 ok: bulk-read failure falls back and invalidates the streaming recovery, archive verifies clean');
				cb();
			} catch (e) {
				cb(e);
			}
		});
	});
}

var tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'par3-stream-parity-'));
try {
	leg1(tempDir);
} catch (e) {
	fails++;
	console.error('LEG1 FAIL: ' + e.message);
}
leg2(tempDir, function(err) {
	if (err) {
		fails++;
		console.error('LEG2 FAIL: ' + err.message);
	}
	leg3(tempDir, function(err3) {
		if (err3) {
			fails++;
			console.error('LEG3 FAIL: ' + err3.message);
		}
		leg4(tempDir, function(err4) {
			if (err4) {
				fails++;
				console.error('LEG4 FAIL: ' + err4.message);
			}
			try { fs.rmSync(tempDir, { recursive: true, force: true }); } catch (e) {}
			if (fails > 0) {
				console.error('STREAMING_PARITY_FAIL ' + fails);
				process.exit(1);
			}
			console.log('STREAMING_PARITY_PASS');
			process.exit(0);
		});
	});
});
