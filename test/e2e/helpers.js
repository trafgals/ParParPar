"use strict";

var path = require('path');
var fs = require('fs');
var os = require('os');
var crypto = require('crypto');
var { spawn } = require('child_process');

var BIN_PATH = path.join(__dirname, '..', '..', 'bin', 'par3.js');
var PARPAR_BIN_PATH = path.join(__dirname, '..', '..', 'bin', 'parpar.js');

function getTempDir() {
	return fs.mkdtempSync(path.join(os.tmpdir(), 'parpar-test-'));
}

function cleanup(dir) {
	if (!dir || dir === '/') return; // Safety: never delete root
	try {
		var files = fs.readdirSync(dir);
		files.forEach(function(f) {
			var filePath = path.join(dir, f);
			var stat = fs.statSync(filePath);
			if (stat.isDirectory()) {
				cleanup(filePath);
			} else {
				fs.unlinkSync(filePath);
			}
		});
		fs.rmdirSync(dir);
	} catch (e) {
		// Ignore cleanup errors
	}
}

function createTestFile(size, filePath) {
	var dir = path.dirname(filePath);
	if (!fs.existsSync(dir)) {
		fs.mkdirSync(dir, { recursive: true });
	}
	
	var chunkSize = 64 * 1024 * 1024;
	var fd = fs.openSync(filePath, 'w');
	for (var offset = 0; offset < size; offset += chunkSize) {
		var toWrite = Math.min(chunkSize, size - offset);
		var chunk = crypto.randomBytes(toWrite);
		fs.writeSync(fd, chunk, 0, toWrite, offset);
	}
	fs.fsyncSync(fd);
	fs.closeSync(fd);
	return filePath;
}

function hashFile(filePath) {
	var hash = crypto.createHash('sha256');
	var fd = fs.openSync(filePath, 'r');
	var stat = fs.fstatSync(fd);
	var chunkSize = 64 * 1024;
	var offset = 0;
	
	while (offset < stat.size) {
		var remaining = stat.size - offset;
		var toRead = Math.min(chunkSize, remaining);
		var buf = Buffer.alloc(toRead);
		fs.readSync(fd, buf, 0, toRead, offset);
		hash.update(buf);
		offset += toRead;
	}
	fs.closeSync(fd);
	return hash.digest('hex');
}

function runPar3(args) {
	return new Promise(function(resolve, reject) {
		var proc = spawn(process.execPath, [BIN_PATH].concat(args), {
			cwd: path.dirname(BIN_PATH),
			stdio: ['ignore', 'pipe', 'pipe']
		});
		
		var stdout = '';
		var stderr = '';
		
		proc.stdout.on('data', function(chunk) { stdout += chunk; });
		proc.stderr.on('data', function(chunk) { stderr += chunk; });
		
		proc.on('close', function(code) {
			resolve({ code: code, stdout: stdout, stderr: stderr });
		});
		
		proc.on('error', function(err) {
			reject(err);
		});
	}).catch(function(err) {
		return { code: -1, stdout: '', stderr: err.message };
	});
}

function runPar3Sync(args) {
	var proc;
	try {
		proc = require('child_process').spawnSync(process.execPath, [BIN_PATH].concat(args), {
			cwd: path.dirname(BIN_PATH),
			stdio: ['ignore', 'pipe', 'pipe']
		});
		return { code: proc.status, stdout: proc.stdout ? proc.stdout.toString() : '', stderr: proc.stderr ? proc.stderr.toString() : '' };
	} catch (e) {
		return { code: -1, stdout: '', stderr: e.message };
	}
}

function runParPar(args) {
	var proc;
	try {
		proc = require('child_process').spawnSync(process.execPath, [PARPAR_BIN_PATH].concat(args), {
			cwd: path.dirname(PARPAR_BIN_PATH),
			stdio: ['ignore', 'pipe', 'pipe']
		});
		return { code: proc.status, stdout: proc.stdout ? proc.stdout.toString() : '', stderr: proc.stderr ? proc.stderr.toString() : '' };
	} catch (e) {
		return { code: -1, stdout: '', stderr: e.message };
	}
}

function runParParSync(args) {
	var proc;
	try {
		proc = require('child_process').spawnSync(process.execPath, [PARPAR_BIN_PATH].concat(args), {
			cwd: path.dirname(PARPAR_BIN_PATH),
			stdio: ['ignore', 'pipe', 'pipe']
		});
		return { code: proc.status, stdout: proc.stdout ? proc.stdout.toString() : '', stderr: proc.stderr ? proc.stderr.toString() : '' };
	} catch (e) {
		return { code: -1, stdout: '', stderr: e.message };
	}
}

function deleteRandomSlices(filePath, sliceSize, count) {
	var stat = fs.statSync(filePath);
	var fileSize = stat.size;
	var totalSlices = Math.floor(fileSize / sliceSize);
	
	if (count > totalSlices) {
		count = totalSlices;
	}
	
	// Generate random slice indices (no duplicates)
	var indices = [];
	while (indices.length < count) {
		var idx = Math.floor(Math.random() * totalSlices);
		if (indices.indexOf(idx) === -1) {
			indices.push(idx);
		}
	}
	
	var fd = fs.openSync(filePath, 'r+');
	var zeroBuf = Buffer.alloc(sliceSize);
	
	for (var i = 0; i < indices.length; i++) {
		var offset = indices[i] * sliceSize;
		fs.writeSync(fd, zeroBuf, 0, sliceSize, offset);
	}
	
	fs.fsyncSync(fd);
	fs.closeSync(fd);

	if (indices.length > 0) {
		var verifyFd = fs.openSync(filePath, 'r');
		var verifyBuf = Buffer.alloc(16);
		var verifyOffset = indices[0] * sliceSize;
		fs.readSync(verifyFd, verifyBuf, 0, 16, verifyOffset);
		fs.closeSync(verifyFd);
		var allZero = true;
		for (var v = 0; v < verifyBuf.length; v++) {
			if (verifyBuf[v] !== 0) {
				allZero = false;
				break;
			}
		}
		if (!allZero) {
			throw new Error('deleteRandomSlices: verification failed - slice not zeroed');
		}
	}

	return indices;
}

// Corrupt N random DATA packets inside a PAR3 archive by flipping bytes in
// their body. This breaks the BLAKE3 packet checksum so that par3_repair's
// validatePacketChecksum() fails and treats those blocks as missing, forcing
// reconstruction from recovery data.
//
// PAR3 archive layout (sequential packet stream):
//   [48-byte header][body]
//   Header: [0..7] magic 'PAR3\0PKT', [8..23] BLAKE3 checksum,
//           [24..31] totalLen (48+bodySize), [32..39] inputSetID,
//           [40..47] type (8-byte ASCII)
//   DATA body: [0..7] block_index (uint64 LE), [8..] block data
//
// Returns the list of corrupted block indices.
function corruptArchiveDataPackets(par3File, count) {
	var PAR3_MAGIC = Buffer.from('PAR3\0PKT');
	var HDR_SIZE = 48;

	var fd = fs.openSync(par3File, 'r+');
	var fileSize = fs.fstatSync(fd).size;

	// Pass 1: scan for DATA packets, record (bodyOffset, blockIndex)
	var dataPackets = [];
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
			dataPackets.push({
				bodyOffset: offset + HDR_SIZE,
				blockIndex: Number(blockIdxBuf.readBigUInt64LE(0))
			});
		}
		offset += totalLen;
	}

	if (dataPackets.length === 0) {
		fs.closeSync(fd);
		throw new Error('corruptArchiveDataPackets: no DATA packets found in ' + par3File);
	}

	// Clamp count to available DATA packets
	if (count > dataPackets.length) count = dataPackets.length;

	// Select `count` random DATA packets (Fisher-Yates partial shuffle)
	var corrupted = [];
	var corruptBuf = Buffer.alloc(16, 0xFF);
	for (var i = 0; i < count; i++) {
		var j = i + Math.floor(Math.random() * (dataPackets.length - i));
		var tmp = dataPackets[i];
		dataPackets[i] = dataPackets[j];
		dataPackets[j] = tmp;
		var pkt = dataPackets[i];
		// Flip 16 bytes in the body (after the 8-byte block_index) to break BLAKE3
		fs.writeSync(fd, corruptBuf, 0, 16, pkt.bodyOffset + 8);
		corrupted.push(pkt.blockIndex);
	}

	fs.fsyncSync(fd);
	fs.closeSync(fd);
	return corrupted;
}

module.exports = {
	getTempDir: getTempDir,
	cleanup: cleanup,
	createTestFile: createTestFile,
	hashFile: hashFile,
	runPar3: runPar3,
	runPar3Sync: runPar3Sync,
	runParPar: runParPar,
	runParParSync: runParParSync,
	deleteRandomSlices: deleteRandomSlices,
	corruptArchiveDataPackets: corruptArchiveDataPackets
};