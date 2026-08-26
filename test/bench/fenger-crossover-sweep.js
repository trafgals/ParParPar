#!/usr/bin/env node
// Crossover sweep: Fenger vs Cauchy (compute_recovery_full) at fixed N=16384
// across power-of-2 R values, to find where Fenger overtakes Cauchy.
// Magic shape is R=8; we expect Cauchy to win there.
const { execFileSync } = require('child_process');
const path = require('path');
const bench = path.join(__dirname, 'par3-create-bench.js');

function run(env, args) {
  const e = Object.assign({}, process.env, env);
  try {
    const out = execFileSync(process.execPath, [bench, ...args], {
      env: e, encoding: 'utf8', timeout: 180000,
    });
    const m = out.match(/Throughput:\s*([\d.]+)\s*MB\/s/);
    const kernel = out.match(/using (\w+ \w+)/);
    return { mbps: m ? parseFloat(m[1]) : null, kernel: kernel ? kernel[1] : '?' };
  } catch (err) {
    const out = (err.stdout || '') + (err.stderr || '');
    const m = out.match(/Throughput:\s*([\d.]+)\s*MB\/s/);
    const kernel = out.match(/using (\w+ \w+)/);
    return { mbps: m ? parseFloat(m[1]) : null, kernel: kernel ? kernel[1] : 'ERR', err: String(err.message).slice(0,80) };
  }
}

const base = ['--size=1G', '--slices=16384', '--block-size=65536', '--recovery=__R__'];
const Rvals = [8, 64, 256, 1024, 4096];
const results = [];
for (const R of Rvals) {
  const args = base.map(a => a.replace('__R__', String(R)));
  const f = run({ PAR3_GF64_USE_FENGER: '1', PAR3_PROFILE: '1' }, args);
  const c = run({ PAR3_GF64_USE_FENGER: '0', PAR3_PROFILE: '1' }, args);
  results.push({ R, fenger: f.mbps, cauchy: c.mbps, fK: f.kernel, cK: c.kernel });
  console.error('R=' + R + ' fenger=' + f.mbps + ' cauchy=' + c.mbps + ' (kernels ' + f.kernel + '/' + c.kernel + ')');
}
console.log(JSON.stringify(results, null, 2));
