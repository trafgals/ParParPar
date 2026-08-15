#!/usr/bin/env node
/* Rewrite the CI badge messages/colors in sources-ci.json from env vars,
 * preserving everything else. Runs in the ci-benchmark-badge workflow
 * after the benches complete. ES5 only (repo runtime is Node >=0.10).
 *
 * Env: CI_CREATE_MBS, CI_TREE_S, CI_MPE_S, CI_INTERP_S (floats as strings).
 * Values are validated strictly (finite, non-negative) — a malformed
 * measurement must never be published as a badge.
 * Colors are thresholded so a regressing CI runner shows up at a glance.
 */
'use strict';
var fs = require('fs');
var path = require('path');

var FILE = path.join(__dirname, '..', 'benchmarks', 'badges', 'sources-ci.json');
var src = JSON.parse(fs.readFileSync(FILE, 'utf8'));

/* Strict: optional sign-free decimal, finite, >= 0. Returns the number or
 * null when the env var is unset/empty; exits nonzero on malformed input. */
function val(name, unit) {
  var v = process.env[name];
  if (v === undefined || v === '') return null;
  if (!/^\d+(\.\d+)?$/.test(v)) {
    console.error('malformed ' + name + '="' + v + '" (expected a non-negative number)');
    process.exit(1);
  }
  var f = parseFloat(v);
  if (!isFinite(f) || f < 0) {
    console.error('out-of-range ' + name + '="' + v + '"');
    process.exit(1);
  }
  return f.toFixed(1) + unit;
}

var create = val('CI_CREATE_MBS', ' MB/s');
var tree = val('CI_TREE_S', ' s');
var mpe = val('CI_MPE_S', ' s');
var interp = val('CI_INTERP_S', ' s');

var colors = {
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
      : (parseFloat(interp) <= 200 ? 'green' : 'yellow'))
};

src.badges.forEach(function (b) {
  switch (b.id) {
    case 'ci-create-1g-1m': if (create) { b.message = create; b.color = colors.create; } break;
    case 'ci-tree-131072': if (tree) { b.message = tree; b.color = colors.tree; } break;
    case 'ci-mpe-131072': if (mpe) { b.message = mpe; b.color = colors.mpe; } break;
    case 'ci-interp-131072': if (interp) { b.message = interp; b.color = colors.interp; } break;
    default: break;
  }
});

fs.writeFileSync(FILE, JSON.stringify(src, null, 2) + '\n');
console.log('sources-ci.json updated: create=' + create + ' tree=' + tree +
            ' mpe=' + mpe + ' interp=' + interp);
