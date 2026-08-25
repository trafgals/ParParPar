"use strict";
// Extract every DATA block from the archive (in index order), concatenate,
// and compare to the source file. Pinpoint which block(s) are corrupt.
var fs = require('fs');
var crypto = require('crypto');
var ARCHIVE = process.argv[2];
var SOURCE  = process.argv[3];
if (!ARCHIVE || !SOURCE) { console.error('usage: node diag_extract.js <archive.par3> <source.bin>'); process.exit(2); }

function readU64(b, off) { return b.readUInt32LE(off) + b.readUInt32LE(off + 4) * 0x100000000; }
var MAGIC = Buffer.from('PAR3\0PKT');
var fd = fs.openSync(ARCHIVE, 'r');
var fileSize = fs.fstatSync(fd).size;

// First pass: collect DATA packet (pos, index, bodyLen) without holding all bodies.
var dataPkts = [];
var pos = 0, guard = 0;
while (pos + 48 <= fileSize && guard < 20000000) {
  guard++;
  var hdr = Buffer.alloc(48);
  var r = fs.readSync(fd, hdr, 0, 48, pos);
  if (r < 48) break;
  if (!hdr.slice(0,8).equals(MAGIC)) break;
  var typeStr = hdr.slice(40,48).toString('ascii').replace(/\0$/,'');
  var totalLen = readU64(hdr, 24);
  var bodyLen = totalLen - 48;
  if (bodyLen < 0 || pos + totalLen > fileSize) break;
  if (typeStr === 'PAR DAT') {
    var body = Buffer.alloc(bodyLen);
    fs.readSync(fd, body, 0, bodyLen, pos + 48);
    var idx = readU64(body, 0);
    dataPkts.push({ pos: pos, idx: idx, bodyLen: bodyLen, dataLen: bodyLen - 8 });
  }
  pos += totalLen;
}
fs.closeSync(fd);
dataPkts.sort(function(a,b){ return a.idx - b.idx; });
if (dataPkts.length === 0) {
  console.log('DATA packets = 0  (no DATA packets in archive — corrupt or non-PAR3 input)');
  console.log('Cannot extract any blocks; aborting.');
  process.exit(1);
}
console.log('DATA packets =', dataPkts.length, '  indices', dataPkts[0].idx, '..', dataPkts[dataPkts.length-1].idx);

// Reopen and stream-extract in index order, hashing + comparing to source.
var fd2 = fs.openSync(ARCHIVE, 'r');
var srcFd = fs.openSync(SOURCE, 'r');
var srcSize = fs.fstatSync(srcFd).size;
console.log('source size =', srcSize, '  total data bytes in archive =', dataPkts.reduce(function(a,p){return a+p.dataLen;},0));

var combined = crypto.createHash('sha256');
var srcHash = crypto.createHash('sha256');
var badBlocks = [];
var srcOff = 0;
var CHUNK = 1 << 20;
for (var pi = 0; pi < dataPkts.length; pi++) {
  var p = dataPkts[pi];
  var off = 0;
  var blockHash = crypto.createHash('sha256');
  while (off < p.dataLen) {
    var n = Math.min(CHUNK, p.dataLen - off);
    var b = Buffer.alloc(n);
    fs.readSync(fd2, b, 0, n, p.pos + 48 + 8 + off); // skip 48 hdr + 8 idx
    combined.update(b);
    blockHash.update(b);
    // source slice for this block
    var sb = Buffer.alloc(n);
    fs.readSync(srcFd, sb, 0, n, srcOff + off);
    srcHash.update(sb);
    if (!b.equals(sb)) { badBlocks.push(p.idx); break; }
    off += n;
  }
  srcOff += p.dataLen;
}
fs.closeSync(fd2); fs.closeSync(srcFd);

console.log('\ncombined(archive data) SHA256 =', combined.digest('hex'));
console.log('source                SHA256 =', srcHash.digest('hex'));
if (badBlocks.length) {
  console.log('\n*** CORRUPT BLOCK(S):', badBlocks.join(', '));
  badBlocks.forEach(function(idx){
    var start = idx * (dataPkts[0].dataLen);
    console.log('  block', idx, 'starts at source byte', start, '(= 2^32?', start === 0x100000000, ')');
  });
} else {
  console.log('\n*** ALL BLOCKS MATCH SOURCE (no content corruption)');
}
process.exit(0);
