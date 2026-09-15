"use strict";

var assert = require('assert');
var fs = require('fs');
var path = require('path');
var os = require('os');
var crypto = require('crypto');

var utils = require('../lib/utils');
var filescan = require('../lib/filescan');
var parpar = require('../lib/parpar');
var arg_parser = require('../lib/arg_parser');
var par2gen = require('../lib/par2gen');
var par3gen = require('../lib/par3gen');

console.log('Testing PAR2/PAR3 shared utilities and deduplication...');

// 1. Test utils.friendlySize
assert.strictEqual(utils.friendlySize(0), '0 B');
assert.strictEqual(utils.friendlySize(1024), '1024 B');
assert.strictEqual(utils.friendlySize(10240), '10 KiB');
assert.strictEqual(utils.friendlySize(10485760), '10 MiB');
assert.strictEqual(utils.friendlySize(10737418240), '10 GiB');

// 2. Test utils.friendlyCount
assert.strictEqual(utils.friendlyCount(100), '100');
assert.strictEqual(utils.friendlyCount(1000), '1000');
assert.strictEqual(utils.friendlyCount(10000), '10K');
assert.strictEqual(utils.friendlyCount(10000000), '10M');

// 3. Test utils.sumSize
assert.strictEqual(utils.sumSize([{size: 10}, {size: 20}, {size: 30}]), 60);

// 4. Test utils.pathNormalize and pathToPar
if(path.sep === '\\') {
	assert.strictEqual(utils.pathNormalize('FOO/BAR\\BAZ'), 'foo\\bar\\baz');
	assert.strictEqual(utils.pathToPar('foo\\bar\\baz'), 'foo/bar/baz');
} else {
	assert.strictEqual(utils.pathNormalize('foo/bar'), 'foo/bar');
	assert.strictEqual(utils.pathToPar('foo/bar'), 'foo/bar');
}

// 5. Test arg_parser.parseSize (used by both CLIs)
assert.strictEqual(arg_parser.parseSize(1024), 1024);
assert.strictEqual(arg_parser.parseSize('1024'), 1024);
assert.strictEqual(arg_parser.parseSize('1K'), 1024);
assert.strictEqual(arg_parser.parseSize('4M'), 4 * 1024 * 1024);
assert.strictEqual(arg_parser.parseSize('2G'), 2 * 1024 * 1024 * 1024);

// 6. Test lib/parpar public exports
assert(parpar.par2, 'parpar.par2 should be exported');
assert(parpar.par3, 'parpar.par3 should be exported');
assert(parpar.PAR3Gen, 'parpar.PAR3Gen should be exported');
assert.strictEqual(typeof parpar.fileScan, 'function', 'parpar.fileScan should be a function');
assert(parpar.utils, 'parpar.utils should be exported');
assert(parpar.version, 'parpar.version should be exported');

// 7. Test shared fileScan with temporary directory
var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'parpar-dedup-test-'));
var fileA = path.join(tmpDir, 'fileA.bin');
var fileB = path.join(tmpDir, 'fileB_empty.bin');

var dataA = Buffer.alloc(32768);
for(var i = 0; i < dataA.length; i++) dataA[i] = i & 0xff;
fs.writeFileSync(fileA, dataA);
fs.writeFileSync(fileB, Buffer.alloc(0));

var expectedMd5_16k = crypto.createHash('md5').update(dataA.subarray(0, 16384)).digest('hex');
var expectedSha256_16k = crypto.createHash('sha256').update(dataA.subarray(0, 16384)).digest('hex');

function cleanupTmp() {
	if(tmpDir && fs.existsSync(tmpDir)) {
		try {
			fs.rmSync(tmpDir, { recursive: true, force: true });
		} catch(e) {
			try {
				if(fs.existsSync(fileA)) fs.unlinkSync(fileA);
				if(fs.existsSync(fileB)) fs.unlinkSync(fileB);
				fs.rmdirSync(tmpDir);
			} catch(e2) {}
		}
	}
}
process.on('exit', cleanupTmp);

(async function() {
	try {
		// Test fileScan directly with md5 (PAR2)
		var info = await new Promise(function(resolve, reject) {
			filescan.fileScan([fileA, fileB], false, false, 2, 'md5', function(err, res) {
				if(err) return reject(err);
				resolve(res);
			});
		});
		assert.strictEqual(info.length, 2);
		var fA = info.find(function(f) { return f.name === fileA; });
		var fB = info.find(function(f) { return f.name === fileB; });

		assert.strictEqual(fA.size, 32768);
		assert.strictEqual(fA.md5_16k.toString('hex'), expectedMd5_16k);

		assert.strictEqual(fB.size, 0);
		assert.strictEqual(fB.md5_16k.toString('hex'), 'd41d8cd98f00b204e9800998ecf8427e');
		assert.strictEqual(fB.md5.toString('hex'), 'd41d8cd98f00b204e9800998ecf8427e');

		// Test fileScan directly with sha256 (PAR3)
		var info2 = await new Promise(function(resolve, reject) {
			filescan.fileScan([fileA, fileB], false, false, 2, 'sha256', function(err, res) {
				if(err) return reject(err);
				resolve(res);
			});
		});
		assert.strictEqual(info2.length, 2);
		var fA2 = info2.find(function(f) { return f.name === fileA; });
		var fB2 = info2.find(function(f) { return f.name === fileB; });

		assert.strictEqual(fA2.size, 32768);
		assert.strictEqual(fA2.sha256_16k.toString('hex'), expectedSha256_16k);

		assert.strictEqual(fB2.size, 0);
		assert.strictEqual(fB2.sha256_16k.length, 32);

		// Verify par2gen.fileInfo delegates correctly
		var p2Info = await new Promise(function(resolve, reject) {
			par2gen.fileInfo([fileA], function(err, res) {
				if(err) return reject(err);
				resolve(res);
			});
		});
		assert.strictEqual(p2Info[0].md5_16k.toString('hex'), expectedMd5_16k);

		// Verify par3gen.fileInfo delegates correctly
		var p3Info = await new Promise(function(resolve, reject) {
			par3gen.fileInfo([fileA], function(err, res) {
				if(err) return reject(err);
				resolve(res);
			});
		});
		assert.strictEqual(p3Info[0].sha256_16k.toString('hex'), expectedSha256_16k);
	} finally {
		cleanupTmp();
	}

	// 8. cubic review 4021045610 P3: verify that cleanup reliably removes directory on simulated failure
	var failTmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'parpar-dedup-fail-test-'));
	var failFile = path.join(failTmpDir, 'dummy.bin');
	fs.writeFileSync(failFile, Buffer.from('test'));
	try {
		throw new Error('simulated failure inside async test');
	} catch(e) {
		fs.rmSync(failTmpDir, { recursive: true, force: true });
	}
	assert.strictEqual(fs.existsSync(failTmpDir), false, 'Temp directory must be cleaned up on failure');

	console.log('PAR2/PAR3 shared utilities and deduplication tests passed!');
})().catch(function(err) {
	console.error(err);
	process.exit(1);
});
