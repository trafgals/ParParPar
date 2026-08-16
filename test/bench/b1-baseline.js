#!/usr/bin/env node
// B1 Phase 1.1: baseline pinning at 1G/1M — A/B the threads + hash pool,
// and the phase split (profile-global reset fix: the module's global
// profile init happens once per process, so a fresh global per run).
"use strict";
var path = require('path');
var fs = require('fs');
var os = require('os');

var SIZE = 1024 * 1024 * 1024;
var BLOCK = 1024 * 1024;
var tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'parpar-b1base-'));
var par3genPath = require.resolve('../../lib/par3gen.js');

function makeFile() {
  var f = path.join(tempDir, 'in.bin');
  if (fs.existsSync(f)) return f;
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

function freshRequire() {
  delete require.cache[par3genPath];
  // The profile global persists across requires — wipe it so the env is
  // re-read (the module's init guard skips re-init otherwise).
  delete global.__par3Profile;
  return require('../lib/par3gen.js');
}

function runOnce(tag, opts, cb) {
  Object.keys(opts).forEach(function(k) {
    if (opts[k] === null) delete process.env[k];
    else process.env[k] = String(opts[k]);
  });
  var par3 = freshRequire();
  var f = makeFile();
  var base = path.join(tempDir, 'out_' + tag.replace(/[^a-z0-9]/gi, '_'));
  var t0 = Date.now();
  par3.create([f], base, { recoverySlices: 8, blockSize: BLOCK, numThreads: opts.PAR3_NUMTHREADS ? parseInt(opts.PAR3_NUMTHREADS, 10) : undefined }, function(err) {
    var dt = (Date.now() - t0) / 1000;
    var mb = SIZE / (1024 * 1024);
    console.log(tag + ': ' + dt.toFixed(1) + ' s (' + (mb / dt).toFixed(1) + ' MB/s)' + (err ? ' ERR ' + err.message : ''));
    if (opts.PAR3_PROFILE === '1' && !err && global.__par3Profile && global.__par3Profile.phases) {
      var p = global.__par3Profile.phases;
      Object.keys(p).forEach(function(k) { console.log('    ' + k + ': ' + (p[k] / 1000).toFixed(2) + ' s'); });
    }
    Object.keys(opts).forEach(function(k) { delete process.env[k]; });
    cb(null, dt);
  });
}

// 1) default (everything on, auto threads)
// 2) hash pool off
// 3) single-threaded kernel (numThreads=1)
// 4) profile run (fresh process semantics via the global wipe)
runOnce('default', {}, function(e1, t1) {
  runOnce('hash-off', { PAR3_GF64_PARALLEL_HASH: '0' }, function(e2, t2) {
    runOnce('threads1', { PAR3_NUMTHREADS: '1' }, function(e3, t3) {
      runOnce('profile', { PAR3_PROFILE: '1' }, function(e4) {
        try { fs.rmSync(tempDir, { recursive: true, force: true }); } catch (e) {}
        process.exit(0);
      });
    });
  });
});
