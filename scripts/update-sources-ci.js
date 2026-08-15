#!/usr/bin/env node
/* Rewrite the CI badge messages/colors in sources-ci.json from env vars,
 * preserving everything else. Runs in the ci-benchmark-badge workflow
 * after the benches complete.
 *
 * Env: CI_CREATE_MBS, CI_TREE_S, CI_MPE_S, CI_INTERP_S (floats as strings).
 * Colors are thresholded so a regressing CI runner shows up at a glance.
 */
'use strict';
const fs = require('fs');
const path = require('path');

const FILE = path.join(__dirname, '..', 'benchmarks', 'badges', 'sources-ci.json');
const src = JSON.parse(fs.readFileSync(FILE, 'utf8'));

function val(name, unit) {
  const v = process.env[name];
  if (v === undefined || v === '') return null;
  const f = parseFloat(v);
  return (isFinite(f) ? f.toFixed(1) : v) + unit;
}

const create = val('CI_CREATE_MBS', ' MB/s');
const tree = val('CI_TREE_S', ' s');
const mpe = val('CI_MPE_S', ' s');
const interp = val('CI_INTERP_S', ' s');

const colors = {
  create: create === null ? 'lightgrey'
    : (parseFloat(create) >= 25 ? 'brightgreen'
      : (parseFloat(create) >= 15 ? 'green' : 'yellow')),
  tree: tree === null ? 'lightgrey'
    : (parseFloat(tree) <= 40 ? 'brightgreen'
      : (parseFloat(tree) <= 80 ? 'green' : 'yellow')),
  mpe: mpe === null ? 'lightgrey'
    : (parseFloat(mpe) <= 80 ? 'brightgreen'
      : (parseFloat(mpe) <= 160 ? 'green' : 'yellow')),
  interp: interp === null ? 'lightgrey'
    : (parseFloat(interp) <= 100 ? 'brightgreen'
      : (parseFloat(interp) <= 200 ? 'green' : 'yellow')),
};

for (const b of src.badges) {
  switch (b.id) {
    case 'ci-create-1g-1m': if (create) { b.message = create; b.color = colors.create; } break;
    case 'ci-tree-131072': if (tree) { b.message = tree; b.color = colors.tree; } break;
    case 'ci-mpe-131072': if (mpe) { b.message = mpe; b.color = colors.mpe; } break;
    case 'ci-interp-131072': if (interp) { b.message = interp; b.color = colors.interp; } break;
    default: break;
  }
}

fs.writeFileSync(FILE, JSON.stringify(src, null, 2) + '\n');
console.log('sources-ci.json updated: create=' + create + ' tree=' + tree +
            ' mpe=' + mpe + ' interp=' + interp);
