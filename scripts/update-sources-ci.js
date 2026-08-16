#!/usr/bin/env node
/* Rewrite the CI badge messages/colors in sources-ci.json from env vars,
 * preserving everything else. Runs in the ci-benchmark-badge workflow
 * after the monthly (30th 20:00 UTC) bench legs complete. ES5 only
 * (repo runtime is Node >=0.10).
 *
 * Env: CI_PAR2_1G_32K_MBS, CI_PAR3_1G_32K_MBS, CI_PAR3_1G_131072_MBS,
 *      CI_PAR3_1G_262144_MBS (floats as strings). The two 10 GiB rows are
 *      kernel-bound beyond the 6 h job cap and stay "N/A >6h" — the
 *      workflow never touches them.
 * Values are validated strictly (finite, non-negative) — a malformed
 * measurement must never be published as a badge. Colors are thresholded
 * so a regressing CI runner shows up at a glance.
 */
'use strict';
var fs = require('fs');
var path = require('path');

var FILE = path.join(__dirname, '..', 'benchmarks', 'badges', 'sources-ci.json');
var src = JSON.parse(fs.readFileSync(FILE, 'utf8'));

/* Strict: optional sign-free decimal, finite, >= 0. Returns the number or
 * null when the env var is unset/empty; exits nonzero on malformed input. */
function val(name) {
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
  return f.toFixed(1) + ' MB/s';
}

/* higher is better */
function colorForHigher(value, bestFloor, okFloor) {
  if (value === null) return 'lightgrey';
  var v = parseFloat(value);
  if (v >= bestFloor) return 'brightgreen';
  if (v >= okFloor) return 'green';
  return 'yellow';
}

/* Thresholds: CI lands at roughly 1/2.4 of the Zen4 reference on these
 * shapes (measured 1G/1M: 54.4 -> 13.7 MB/s was the scratch build; the
 * master ratio is ~2.4x). bestFloor ~= 2x the observed CI rate, okFloor
 * ~= the observed CI rate — a halved runner shows yellow. */
var thresholds = {
  'par2-1g-32k-ci':       { best: 400, ok: 200 },   /* Zen4: 613 MB/s */
  'par3-1g-32k-ci':       { best: 1.0, ok: 0.5 },   /* Zen4: ~1.5 MB/s */
  'par3-1g-131072-ci':    { best: 0.35, ok: 0.15 }, /* Zen4: ~0.5 MB/s */
  'par3-1g-262144-ci':    { best: 0.15, ok: 0.08 }  /* Zen4: ~0.2 MB/s */
};

var values = {
  'par2-1g-32k-ci':    val('CI_PAR2_1G_32K_MBS'),
  'par3-1g-32k-ci':    val('CI_PAR3_1G_32K_MBS'),
  'par3-1g-131072-ci': val('CI_PAR3_1G_131072_MBS'),
  'par3-1g-262144-ci': val('CI_PAR3_1G_262144_MBS')
};

/* Stamp the measurement date so the badge data is self-dating. */
src.date = new Date().toISOString().slice(0, 10);

src.badges.forEach(function (b) {
  if (values.hasOwnProperty(b.id) && values[b.id] !== null) {
    b.message = values[b.id];
    b.color = colorForHigher(values[b.id], thresholds[b.id].best, thresholds[b.id].ok);
  }
  /* Rows without a measured value (10 GiB legs, or a missing leg) keep
   * their committed message/color. */
});

fs.writeFileSync(FILE, JSON.stringify(src, null, 2) + '\n');
console.log('sources-ci.json updated: ' + Object.keys(values)
  .map(function (k) { return k + '=' + values[k]; }).join(' '));
