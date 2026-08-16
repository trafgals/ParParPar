#!/usr/bin/env node
// A/B: repair RHS accumulation — coupled_muladd (default) vs the legacy
// per-(known,eq) mul_arr + per-byte JS XOR loop (PAR3_REPAIR_COUPLED_RHS=0).
//
// Flow mirrors test/par3-repair-native-matrix.js leg2: create a small
// archive, damage DATA packets so blocks are genuinely missing, repair,
// and time the repair phase.
"use strict";
var path = require('path');
var fs = require('fs');
var os = require('os');
var par3 = require('../../lib/par3gen.js');

var PAR3_MAGIC = Buffer.from('PAR3\0PKT');
var PAR3_PKT_HDR_SIZE = 48;

var BLOCK = 128 * 1024;          // 128 KiB blocks
var NUM_BLOCKS = 256;            // 32 MiB source
var MISSING = [10, 20, 30, 40, 50, 60, 70, 80]; // 8 missing (n=8, known=248)

function damageArchive(par3File) {
	var fd = fs.openSync(par3File, 'r+');
	var stat = fs.fstatSync(fd);
	var offset = 0;
	var damaged = [];
	while (offset < stat.size && damaged.length < MISSING.length) {
		var header = Buffer.alloc(PAR3_PKT_HDR_SIZE);
		fs.readSync(fd, header, 0, PAR3_PKT_HDR_SIZE, offset);
		if (!header.slice(0, 8).equals(PAR3_MAGIC)) { offset += 8; continue; }
		var totalLen = Number(header.readBigUInt64LE(24));
		if (totalLen < PAR3_PKT_HDR_SIZE || offset + totalLen > stat.size + 8) break;
		if (header.slice(40, 48).toString('ascii') === 'PAR DAT\0') {
			var idxBuf = Buffer.alloc(8);
			fs.readSync(fd, idxBuf, 0, 8, offset + PAR3_PKT_HDR_SIZE);
			var blockIndex = Number(idxBuf.readBigUInt64LE(0));
			if (MISSING.indexOf(blockIndex) !== -1) {
				fs.writeSync(fd, Buffer.alloc(totalLen), 0, totalLen, offset);
				damaged.push(blockIndex);
			}
		}
		offset += totalLen;
	}
	fs.fsyncSync(fd);
	fs.closeSync(fd);
	return damaged;
}

function run(tag, env, base, tempDir, cb) {
	Object.keys(env).forEach(function(k) { process.env[k] = env[k]; });
	var outDir = path.join(tempDir, 'rep_' + tag);
	fs.mkdirSync(outDir);
	fs.copyFileSync(path.join(tempDir, 'in.bin'), path.join(outDir, 'in.bin'));
	var t0 = Date.now();
	par3.repair(base + '.par3', outDir, { verbose: 0 }, function(err, result) {
		var dt = Date.now() - t0;
		Object.keys(env).forEach(function(k) { delete process.env[k]; });
		if (err) return cb(new Error('repair failed: ' + err.message));
		cb(null, dt, result);
	});
}

var tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'par3-rhs-ab-'));
var inFile = path.join(tempDir, 'in.bin');
var base = path.join(tempDir, 'out');
var fileSize = BLOCK * NUM_BLOCKS;

var fd = fs.openSync(inFile, 'w');
var chunk = Buffer.alloc(BLOCK);
var seed = 0x9E3779B9;
for (var blk = 0; blk < NUM_BLOCKS; blk++) {
	for (var j = 0; j < BLOCK; j += 4) {
		seed = (seed * 1664525 + 1013904223) >>> 0;
		chunk.writeUInt32LE(seed, j);
	}
	fs.writeSync(fd, chunk);
}
fs.closeSync(fd);

par3.create([inFile], base, {
	outputBase: base,
	recoverySlices: 8,
	blockSize: BLOCK
}, function(err) {
	if (err) { console.error('create failed: ' + err.message); process.exit(1); }
	var damaged = damageArchive(base + '.par3');
	if (damaged.length !== MISSING.length) {
		console.error('damaged ' + damaged.length + '/' + MISSING.length);
		process.exit(1);
	}

	run('coupled', {}, base, tempDir, function(e1, t1, r1) {
		if (e1) { console.error(e1.message); process.exit(1); }
		run('legacy', { PAR3_REPAIR_COUPLED_RHS: '0' }, base, tempDir, function(e2, t2, r2) {
			if (e2) { console.error(e2.message); process.exit(1); }
			console.log('repair (32 MiB, ' + NUM_BLOCKS + ' blocks, ' + MISSING.length + ' missing):');
			console.log('  coupled: ' + t1 + ' ms  (blocksRepaired=' + r1.blocksRepaired + ')');
			console.log('  legacy:  ' + t2 + ' ms  (blocksRepaired=' + r2.blocksRepaired + ')');
			console.log('  speedup: ' + (t2 / Math.max(t1, 1)).toFixed(1) + 'x');
			// Byte-identical outputs?
			var f1 = fs.readFileSync(path.join(tempDir, 'rep_coupled', 'block_0.dat'));
			var f2 = fs.readFileSync(path.join(tempDir, 'rep_legacy', 'block_0.dat'));
			console.log('  outputs byte-identical: ' + f1.equals(f2));
			try { fs.rmSync(tempDir, { recursive: true, force: true }); } catch (e) {}
			process.exit(0);
		});
	});
});
