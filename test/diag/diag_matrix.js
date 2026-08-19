"use strict";
// Diagnostic: create a PAR3 archive crossing the 2^32 byte boundary, then
// dump GROUND TRUTH from the archive bytes (matrix body, DATA packet count,
// block indices, and the block just before/after 2^32). This tells us whether
// the matrix declares more inputs than DATA packets exist (create bug) or
// whether a DATA packet's content is corrupt (write bug).
//
// Usage:
//   node diag_matrix.js                 -> create fresh, then dump
//   node diag_matrix.js <path.par3>     -> skip create, dump existing archive
var fs = require('fs');
var path = require('path');
var os = require('os');
var crypto = require('crypto');
var par3 = require('./lib/par3gen.js');

var BLOCK_SIZE = 1024 * 1024; // 1 MiB

var existingPath = process.argv[2];
if (existingPath) {
  dumpArchive(existingPath, BLOCK_SIZE, null);
  process.exit(0);
}

// 4100 blocks -> block 4095 starts at byte 4095*2^20 = 2^32 exactly.
var SIZE = 4100 * BLOCK_SIZE;

var dir = fs.mkdtempSync(path.join(os.tmpdir(), 'diag-matrix-'));
var srcPath = path.join(dir, 'src.bin');
var outBase = path.join(dir, 'out');

console.log('dir =', dir);
console.log('SIZE =', SIZE, 'bytes =', SIZE / 1048576, 'MiB');
console.log('block 4095 data starts at byte', 4095 * BLOCK_SIZE, '(= 2^32?', 4095 * BLOCK_SIZE === 0x100000000, ')');

// Deterministic pseudo-random source so hashes are reproducible.
function fillSource(p, size) {
  var fd = fs.openSync(p, 'w');
  var chunk = 1 << 20;
  var seed = 0x12345678;
  function next() { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return seed >>> 0; }
  var off = 0;
  while (off < size) {
    var n = Math.min(chunk, size - off);
    var b = Buffer.alloc(n);
    for (var i = 0; i < n; i++) { b[i] = (next() >> ((i & 3) * 8)) & 0xff; }
    fs.writeSync(fd, b, 0, n, off);
    off += n;
  }
  fs.fsyncSync(fd); fs.closeSync(fd);
}
console.log('Filling source...');
fillSource(srcPath, SIZE);
var srcSha = (function(p){var h=crypto.createHash('sha256');var fd=fs.openSync(p,'r');var s=fs.fstatSync(fd);var o=0;while(o<s.size){var t=Math.min(1<<20,s.size-o);var b=Buffer.alloc(t);fs.readSync(fd,b,0,t,o);h.update(b);o+=t;}fs.closeSync(fd);return h.digest('hex');})(srcPath);
console.log('source sha256 =', srcSha);

console.log('\n=== CREATE ===');
var t0 = Date.now();
par3.create([srcPath], outBase, { recoverySlices: { unit: 'ratio', value: 0.1 } }, function(err) {
  if (err) { console.error('CREATE ERR', err && err.message, err && err.stack); process.exit(1); }
  console.log('create done in', Date.now() - t0, 'ms');
  dumpArchive(path.join(dir, 'out.par3'), BLOCK_SIZE, srcSha);
});

function dumpArchive(par3File, blockSize, srcSha) {
  console.log('\n=== ARCHIVE BYTES (ground truth) ===');
  var fd = fs.openSync(par3File, 'r');
  var fileSize = fs.fstatSync(fd).size;
  console.log('archive size =', fileSize, 'bytes');

  function readU64(b, off) { return b.readUInt32LE(off) + b.readUInt32LE(off + 4) * 0x100000000; }

  // PAR3 packet framing (48-byte header):
  //   [0..7]    magic 'PAR3\0PKT'
  //   [8..23]   checksum (16B)
  //   [24..31]  total length (header+body), uint64 LE
  //   [32..39]  inputSetId (8B)
  //   [40..47]  ASCII type (8B) e.g. 'PAR DAT\0'
  // DATA body:   [0..7] blockIndex uint64 LE, [8..] data
  var HDR = 48;
  var MAGIC = Buffer.from('PAR3\0PKT');
  var T = {
    START:'PAR STA\0', CREATOR:'PAR CRE\0', FILE:'PAR FIL\0', DIR:'PAR DIR\0',
    DATA:'PAR DAT\0', EXT_DATA:'PAR EFD\0', MATRIX:'PAR MAT\0', CAUCHY:'PAR CAU\0',
    SPARSE:'PAR SPR\0', EXPLICIT:'PAR EXP\0', RECOVERY:'PAR REC\0', EXT_REC:'PAR ERD\0',
    UNIX_PERM:'PAR UNX\0', FAT_PERM:'PAR FAT\0', ROOT:'PAR ROO\0'
  };
  var dataIdx = [];
  var matrixBody = null, matrixBodyLen = 0;
  var recoveryPkts = [];
  var filePkts = 0, startBody = null, cauchyBody = null;
  var typeCounts = {};
  var pos = 0;
  var guard = 0;
  while (pos + HDR <= fileSize && guard < 20000000) {
    guard++;
    var hdr = Buffer.alloc(HDR);
    var r = fs.readSync(fd, hdr, 0, HDR, pos);
    if (r < HDR) break;
    if (!hdr.slice(0,8).equals(MAGIC)) { console.log('bad magic at pos', pos, JSON.stringify(hdr.slice(0,8))); break; }
    var typeStr = hdr.slice(40,48).toString('ascii').replace(/\0$/,'');
    var totalLen = readU64(hdr, 24);
    var bodyLen = totalLen - HDR;
    if (bodyLen < 0 || pos + totalLen > fileSize) { console.log('bad totalLen', totalLen, 'at pos', pos); break; }
    typeCounts[typeStr] = (typeCounts[typeStr]||0)+1;
    var body = Buffer.alloc(bodyLen);
    var rb = fs.readSync(fd, body, 0, bodyLen, pos+HDR);
    if (rb < bodyLen) { console.log('short body at pos', pos); break; }
    pos += totalLen;

    if (typeStr === 'PAR DAT') {
      dataIdx.push(readU64(body, 0));
    } else if (typeStr === 'PAR CAU' || typeStr === 'PAR MAT' || typeStr === 'PAR EXP') {
      matrixBody = body; matrixBodyLen = bodyLen;
    } else if (typeStr === 'PAR REC') {
      recoveryPkts.push({ first: readU64(body, 0), count: readU64(body, 8), bodyLen: bodyLen });
    } else if (typeStr === 'PAR FIL') {
      filePkts++;
    } else if (typeStr === 'PAR STA') {
      startBody = body;
    }
  }
  fs.closeSync(fd);

  console.log('walked to pos', pos, 'of', fileSize, '  typeCounts =', JSON.stringify(typeCounts));

  console.log('\nDATA packets found:', dataIdx.length);
  if (dataIdx.length) {
    console.log('  first 5 indices:', dataIdx.slice(0,5).join(', '));
    console.log('  last 5 indices :', dataIdx.slice(-5).join(', '));
    // check for gaps / dupes
    var s = {}; dataIdx.forEach(function(i){s[i]=(s[i]||0)+1;});
    var dupes = Object.keys(s).filter(function(k){return s[k]>1;});
    var minI = Math.min.apply(null, dataIdx), maxI = Math.max.apply(null, dataIdx);
    console.log('  min index', minI, 'max index', maxI, 'expected count if dense =', maxI-minI+1);
    if (dupes.length) console.log('  DUPLICATES at:', dupes.join(','));
    var missingInRange = [];
    for (var i = minI; i <= maxI; i++) if (!s[i]) missingInRange.push(i);
    if (missingInRange.length <= 20) console.log('  missing within [min,max]:', missingInRange.join(','));
    else console.log('  missing within [min,max] count:', missingInRange.length, 'first few:', missingInRange.slice(0,10).join(','));
  }

  console.log('\nRECOVERY packets:', recoveryPkts.length);
  recoveryPkts.slice(0,3).forEach(function(rp){ console.log('  first='+rp.first+' count='+rp.count); });
  var recTotal = recoveryPkts.reduce(function(a,r){return a+r.count;},0);
  console.log('  total recovery blocks =', recTotal);

  if (matrixBody) {
    console.log('\nMATRIX body (len='+matrixBodyLen+'):');
    var first_input = readU64(matrixBody, 0);
    var last_input  = readU64(matrixBody, 8);
    var first_rec   = readU64(matrixBody, 16);
    var rec_count   = matrixBodyLen >= 32 ? readU64(matrixBody, 24) : 'n/a';
    console.log('  first_input  =', first_input);
    console.log('  last_input   =', last_input, '  => inputCount =', last_input - first_input + 1);
    console.log('  first_recovery =', first_rec);
    console.log('  recovery_count =', rec_count);
    console.log('  raw body hex =', matrixBody.toString('hex'));
  }

  console.log('\nFILE packets:', filePkts);
  if (startBody) {
    console.log('START body: gf_size='+startBody[0], 'block_size='+readU64(startBody,16), 'block_pow='+startBody.readUInt32LE(24));
  }

  // Cross-check: does dataIdx cover [0, last_input] fully?
  if (matrixBody && dataIdx.length) {
    var last_input = readU64(matrixBody, 8);
    var first_input = readU64(matrixBody, 0);
    var have = {}; dataIdx.forEach(function(i){have[i]=1;});
    var undeclared = [];
    for (var i = first_input; i <= last_input; i++) if (!have[i]) undeclared.push(i);
    console.log('\n>>> Matrix declares inputs', first_input, '..', last_input, '('+(last_input-first_input+1)+'), but DATA packets present =', dataIdx.length);
    if (undeclared.length) console.log('>>> MISSING DECLARED INPUTS:', undeclared.join(','));
    else console.log('>>> all declared inputs have DATA packets');
    var extra = dataIdx.filter(function(i){ return i < first_input || i > last_input; });
    if (extra.length) console.log('>>> DATA packets OUTSIDE declared range:', extra.join(','));
  }
  console.log('\nDONE');
  process.exit(0);
}
