#!/usr/bin/env node
// Crossover calibration: find smallest R where Fenger overtakes Barycentric.
// Fixed N=16384, blockSize=64KiB, R = 512 / 1024 / 2048 (each capped at 240s).
// Fenger per-word = O(N log^3 N) (R-independent); Barycentric = O(N*R).
// We want the smallest R where fenger_mbs > bary_mbs.
//
// Uses execFileSync against par3-create-bench.js (the canonical
// `par3.create([sourceFile], outputBase, opts, cb)` path), and toggles
// PAR3_GF64_USE_FENGER in the child env to actually dispatch
// Fenger vs Barycentric (default is Fenger=on, so Bary must override to '0').
'use strict';
const { execFileSync } = require('child_process');
const path = require('path');

const bench = path.join(__dirname, 'par3-create-bench.js');
const BASE_ARGS = ['--size=1G', '--slices=16384', '--block-size=65536'];

function timed(label, env, R, cap) {
  const childEnv = Object.assign({}, process.env, env);
  const args = BASE_ARGS.concat(['--recovery=' + R]);
  try {
    const out = execFileSync(process.execPath, [bench].concat(args), {
      env: childEnv,
      encoding: 'utf8',
      timeout: cap * 1000,
    });
    const m = out.match(/Throughput:\s*([\d.]+)\s*MB\/s/);
    const mbps = m ? parseFloat(m[1]) : null;
    console.log(`R=${R} ${label}: ${mbps !== null ? mbps.toFixed(2) + ' MB/s' : 'no-output'}`);
    return mbps;
  } catch (e) {
    console.log(`R=${R} ${label}: TIMEOUT/CAP (>${cap}s)`);
    return null;
  }
}

for (const R of [512, 1024, 2048]) {
  console.log('--- R=' + R + ' ---');
  timed('Fenger', { PAR3_GF64_USE_FENGER: '1' }, R, 240);
  timed('Bary',   { PAR3_GF64_USE_FENGER: '0' }, R, 240);
}
