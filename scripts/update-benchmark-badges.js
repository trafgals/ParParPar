#!/usr/bin/env node
/* Regenerate the shields.io endpoint badge JSONs from sources.json.
 *
 * Usage (from the repo root, on the feat/ci-benchmark-badge branch):
 *   node scripts/update-benchmark-badges.js
 *
 * Reads benchmarks/badges/sources.json (the single source of truth for
 * benchmark numbers), writes one endpoint JSON per badge into
 * benchmarks/badges/<id>.json, and prints the README markdown snippet.
 * Commit the generated files + sources.json together; the README badge
 * URLs on master point at this branch via raw.githubusercontent.com.
 */
'use strict';
const fs = require('fs');
const path = require('path');

const SRC = path.join(__dirname, '..', 'benchmarks', 'badges', 'sources.json');
const OUT_DIR = path.join(__dirname, '..', 'benchmarks', 'badges');

const src = JSON.parse(fs.readFileSync(SRC, 'utf8'));
if (!Array.isArray(src.badges) || src.badges.length === 0) {
  console.error('sources.json: no badges array');
  process.exit(1);
}

let md = '';
for (const b of src.badges) {
  if (!b.id || !b.label || !b.message) {
    console.error('sources.json: badge missing id/label/message: ' + JSON.stringify(b));
    process.exit(1);
  }
  const ep = {
    schemaVersion: 1,
    label: b.label,
    message: b.message,
    color: b.color || 'brightgreen',
  };
  if (b.detail) ep.labelColor = '555';
  const file = path.join(OUT_DIR, b.id + '.json');
  fs.writeFileSync(file, JSON.stringify(ep, null, 2) + '\n');
  const url = 'https://raw.githubusercontent.com/' + src.repo + '/' +
              src.branch + '/benchmarks/badges/' + b.id + '.json';
  const shields = 'https://img.shields.io/endpoint?url=' + encodeURIComponent(url) +
                  '&style=flat-square';
  md += '[![' + b.label + '](' + shields + ')](https://github.com/' +
        src.repo + '#throughput)\n';
  console.log('wrote ' + file);
}
console.log('\nREADME snippet:\n' + md);
