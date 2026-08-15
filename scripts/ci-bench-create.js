#!/usr/bin/env node
/* Single-leg create throughput measurement for the CI benchmark badge
 * workflow. Mirrors test/b1-baseline.js's default leg: 1 GiB file,
 * 1 MiB blocks, R=8, default env (auto threads). Prints:
 *   create-1g-1m: <MB/s> (<seconds> s)
 */
"use strict";
var path = require('path');
var fs = require('fs');
var os = require('os');

var SIZE = 1024 * 1024 * 1024;
var BLOCK = 1024 * 1024;
var tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ci-bench-'));
var par3genPath = require.resolve('../lib/par3gen.js');

function makeFile() {
  var f = path.join(tempDir, 'in.bin');
  var fd = fs.openSync(f, 'w');
  var chunk = Buffer.alloc(BLOCK);
  var seed = 0x13579BDF;
  for (var off = 0; off < SIZE; off += BLOCK) {
    for (var j = 0; j < BLOCK; j += 4) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      chunk.writeUInt32LE(seed, j);
    }
    fs.writeSync(fd, chunk);
  }
  fs.closeSync(fd);
  return f;
}

delete require.cache[par3genPath];
delete global.__par3Profile;
var par3 = require(par3genPath);
var f = makeFile();
var base = path.join(tempDir, 'out');
var t0 = Date.now();
par3.create([f], base, { outputBase: base, recoverySlices: 8, blockSize: BLOCK }, function(err) {
  var dt = (Date.now() - t0) / 1000;
  var mb = SIZE / (1024 * 1024);
  if (err) {
    console.log('create-1g-1m: ERROR ' + err.message);
    process.exit(1);
  }
  console.log('create-1g-1m: ' + (mb / dt).toFixed(1) + ' MB/s (' + dt.toFixed(1) + ' s)');
  try { fs.rmSync(tempDir, { recursive: true, force: true }); } catch (e) {}
  process.exit(0);
});
