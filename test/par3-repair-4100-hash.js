"use strict";

var path = require('path');
var fs = require('fs');
var par3 = require('../lib/par3gen.js');
var helpers = require('./e2e/helpers');

var BLOCK_SIZE = 1024 * 1024;
var TARGET_BLOCK = 4099; // last block — hits 4 GiB boundary
// Fixture must be >= (TARGET_BLOCK + 1) * BLOCK_SIZE = 4100 MiB so the
// target block index exists. A smaller file cannot substitute because the
// bug being guarded only triggers past the 4 GiB boundary.
var FIXTURE_SIZE = 4100 * 1024 * 1024;
var FIXTURE = path.join(__dirname, 'test4100m.bin');

// Deterministic corruption of a specific block in the PAR3 archive.
// Scans for the DATA packet with block_index === target, then writes
// 16 bytes of 0xFF at bodyOffset + 8 to break the BLAKE3 checksum.
// Adapted from helpers.corruptArchiveDataPackets but deterministic (no Math.random).
function corruptBlock(par3File, targetBlock) {
	var PAR3_MAGIC = Buffer.from('PAR3\0PKT');
	var HDR_SIZE = 48;
	var fd = fs.openSync(par3File, 'r+');
	var fileSize = fs.fstatSync(fd).size;
	var offset = 0;
	while (offset + HDR_SIZE <= fileSize) {
		var hdr = Buffer.alloc(HDR_SIZE);
		fs.readSync(fd, hdr, 0, HDR_SIZE, offset);
		if (!hdr.slice(0, 8).equals(PAR3_MAGIC)) break;
		var totalLen = Number(hdr.readBigUInt64LE(24));
		if (totalLen < HDR_SIZE) break;
		var type = hdr.slice(40, 48).toString('ascii').replace(/\0/g, '').trim();
		if (type === 'PAR DAT') {
			var blockIdxBuf = Buffer.alloc(8);
			fs.readSync(fd, blockIdxBuf, 0, 8, offset + HDR_SIZE);
			var blockIdx = Number(blockIdxBuf.readBigUInt64LE(0));
			if (blockIdx === targetBlock) {
				var corruptBuf = Buffer.alloc(16, 0xFF);
				fs.writeSync(fd, corruptBuf, 0, 16, offset + HDR_SIZE + 8);
				fs.fsyncSync(fd);
				fs.closeSync(fd);
				return true;
			}
		}
		offset += totalLen;
	}
	fs.closeSync(fd);
	throw new Error('corruptBlock: block ' + targetBlock + ' not found in ' + par3File);
}

function main() {
	if (!fs.existsSync(FIXTURE)) {
		// Bug only triggers at block index >= 4096 (4 GiB boundary), so a
		// smaller fixture cannot substitute. Generate on the fly instead of
		// silently exiting 0 (which masks regressions on CI runs that don't
		// pre-provision the 4.3 GiB fixture).
		console.error('Fixture missing: ' + FIXTURE);
		console.error('Generating ' + (FIXTURE_SIZE / (1024 * 1024)) + ' MiB fixture on the fly...');
		try {
			helpers.createTestFile(FIXTURE_SIZE, FIXTURE);
			console.error('  Generated: ' + FIXTURE + ' (' + fs.statSync(FIXTURE).size + ' bytes)\n');
		} catch (e) {
			// Cubic cycle 2 #98 P2: unlink partial fixture so the next run regenerates
			// instead of treating a truncated FIXTURE as valid via fs.existsSync.
			try { fs.unlinkSync(FIXTURE); } catch (e2) { /* may not exist */ }
			console.error('ERROR: failed to generate fixture: ' + e.message);
			console.error('A ' + (FIXTURE_SIZE / 1073741824).toFixed(2) + ' GiB fixture is required for this test.');
			process.exit(1);
		}
	}

	var tempDir = helpers.getTempDir();
	var fixtureCopy = path.join(tempDir, 'test.bin');
	var outputBase = path.join(tempDir, 'out');
	var par3File = outputBase + '.par3';

	try {
		console.log('PAR3 Repair 4100-Block Hash Mismatch Test (T1 RED)');
		console.log('====================================================\n');
		console.log('Fixture: ' + path.basename(FIXTURE));
		console.log('Target block: ' + TARGET_BLOCK + ' (4 GiB boundary)\n');

		// 1. Copy fixture
		console.log('Copying fixture...');
		fs.copyFileSync(FIXTURE, fixtureCopy);
		var fileSize = fs.statSync(fixtureCopy).size;
		var blockCount = Math.floor(fileSize / BLOCK_SIZE);
		console.log('  File: ' + (fileSize / 1073741824).toFixed(2) + ' GiB, ' + blockCount + ' blocks\n');

		// 2. Hash original FULL file
		console.log('Hashing original file (full)...');
		var originalHash = helpers.hashFile(fixtureCopy);
		console.log('  SHA256: ' + originalHash + '\n');

		// 3. Create PAR3 archive
		console.log('Creating PAR3 archive (ratio 0.1 → 512 recovery slices)...');
		par3.create([fixtureCopy], outputBase, {
			outputBase: outputBase,
			recoverySlices: { unit: 'ratio', value: 0.1 }
		}, function(err) {
			if (err) {
				console.error('  Create failed: ' + err.message);
				finish(false, 'create failed');
				return;
			}
			console.log('  Archive created: ' + par3File + '\n');

			// 4. Verify archive is intact
			console.log('Verifying archive...');
			par3.verify(par3File, function(err2, vres) {
				if (err2) {
					console.error('  Verify error: ' + err2.message);
					finish(false, 'verify error');
					return;
				}
				console.log('  verified=' + vres.verified + ' archiveOk=' + vres.archiveOk +
					' inputBlocks=' + vres.inputBlocks + ' recoveryBlocks=' + vres.recoveryBlocks + '\n');

				// 5. Corrupt block 4099 (deterministic)
				console.log('Corrupting block ' + TARGET_BLOCK + ' (deterministic)...');
				corruptBlock(par3File, TARGET_BLOCK);
				console.log('  Block ' + TARGET_BLOCK + ' corrupted (16 bytes 0xFF at BLAKE3 position)\n');

				// 6. Repair
				console.log('Running repair...');
				par3.repair(par3File, tempDir, {}, function(err3, result) {
					if (err3) {
						console.error('  Repair failed: ' + err3.message);
						finish(false, 'repair failed');
						return;
					}
					console.log('  repaired=' + result.repaired + ' blocksRepaired=' + result.blocksRepaired + '\n');

				// 7. Hash repaired FULL file
				var repairedFile = path.join(tempDir, 'block_0.dat');
				if (!fs.existsSync(repairedFile)) {
					console.error('  ERROR: Repaired file not found: ' + repairedFile);
					finish(false, 'no repaired file');
					return;
				}

				console.log('Hashing repaired file (full)...');
				var repairedHash = helpers.hashFile(repairedFile);
				console.log('  SHA256: ' + repairedHash + '\n');

					// 8. Compare
					if (repairedHash === originalHash) {
						console.log('PASS: Repaired hash === Original hash');
						finish(true);
					} else {
						console.error('FAIL: Repaired hash != Original hash');
						console.error('  Expected: ' + originalHash);
						console.error('  Got:      ' + repairedHash);
						finish(false, 'hash mismatch');
					}
				});
			});
		});

	} catch (e) {
		console.error('Exception: ' + e.message);
		finish(false, e.message);
	}

	function finish(passed, errMsg) {
		try { helpers.cleanup(tempDir); } catch (e) { /* ignore */ }
		console.log('\n' + (passed ? 'TEST PASSED' : 'TEST FAILED: ' + (errMsg || '')));
		process.exit(passed ? 0 : 1);
	}
}

main();
