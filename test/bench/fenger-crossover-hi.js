#!/usr/bin/env node
// Crossover calibration: find the R where Fenger overtakes Barycentric.
// Fixed N=16384, blockSize=64KiB, R = 512 / 1024 / 2048 (each capped at 240s).
// Fenger per-word = O(N log^3 N) (R-independent); Barycentric = O(N*R).
// We want the smallest R where fenger_mbs > bary_mbs.
'use strict';
const { par3_create } = require('../lib/par3gen.js');
const { randomFillSync } = require('crypto');

const N = 16384, BS = 65536;
const data = Buffer.alloc(N * BS);
randomFillSync(data);
const inBufs = [];
for (let c = 0; c < N; c++) inBufs.push(data.subarray(c * BS, (c + 1) * BS));

function run(label, env) {
  const t0 = Date.now();
  try {
    const res = par3_create({
      inputs: inBufs,
      blockSize: BS,
      recoverySlices: env,
      numThreads: 8,
    });
    const sec = (Date.now() - t0) / 1000;
    const mbs = (data.length / sec) / 1e6;
    console.log(`R=${env} ${label}: ${mbs.toFixed(2)} MB/s (${sec.toFixed(1)}s)`, 'ERR=' + 0);
    return mbs;
  } catch (e) {
    console.log(`R=${env} ${label}: ERROR ${e.message.slice(0, 80)}`);
    return null;
  }
}

function timed(label, env, cap) {
  // run in a child to hard-cap each measurement
  const { execFileSync } = require('child_process');
  const probe = `
    (async () => {
      const { par3_create } = require('${process.cwd()}/lib/par3gen.js');
      const { randomFillSync } = require('crypto');
      const N=${N},BS=${BS};
      const data=Buffer.alloc(N*BS); randomFillSync(data);
      const inB=[]; for(let c=0;c<N;c++) inB.push(data.subarray(c*BS,(c+1)*BS));
      const t0=Date.now();
      par3_create({inputs:inB, blockSize:BS, recoverySlices:${env}, numThreads:8});
      const sec=(Date.now()-t0)/1000;
      console.log('MBPS', (data.length/sec)/1e6);
    })().catch(e=>{console.log('ERR',e.message.slice(0,60)); process.exit(1);});
  `;
  try {
    const out = execFileSync(process.execPath, ['-e', probe], { timeout: cap * 1000, encoding: 'utf8' });
    const m = out.match(/MBPS\s+([\d.]+)/);
    const mbs = m ? parseFloat(m[1]) : null;
    console.log(`R=${env} ${label}: ${mbs !== null ? mbs.toFixed(2) + ' MB/s' : 'no-output'}`);
    return mbs;
  } catch (e) {
    console.log(`R=${env} ${label}: TIMEOUT/CAP (>${cap}s)`);
    return null;
  }
}

for (const R of [512, 1024, 2048]) {
  console.log('--- R=' + R + ' ---');
  timed('Fenger', R, 240);
  timed('Bary', R, 240);
}
