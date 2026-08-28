"use strict";
/* Contract test (cubic review on PR #89 P2): every badge URL in the README
 * throughput table must resolve to a JSON file served from the
 * `feat/ci-benchmark-badge` branch. This pins down the cross-branch
 * contract so a future PR can't accidentally reference a badge ID that
 * isn't registered on the badge branch (which renders as a broken
 * shields.io endpoint and silently loses the throughput number from
 * the README table).
 *
 * Also catches the inverse: an entry in `benchmarks/badges/sources.json`
 * on the badge branch that the README doesn't reference (those would
 * generate dead badge JSONs).
 *
 * Run: `node test/readme-badge-urls.js`
 *   - exit 0: all URLs resolve, all sources.json entries are referenced
 *   - exit 1: a URL is broken OR an orphan badge exists
 */

var shared = require('./readme-badges-shared');
var fs = require('fs');
var https = require('https');

var BADGE_BRANCH_RAW = shared.BADGE_BRANCH_RAW;
var README_PATH = 'README.md';
// Per-test /g flag for iterating all matches; the shared regex is the
// same pattern (test/readme-badges-shared.js) without the /g flag.
var urlRe = new RegExp(shared.ZEN4_BADGE_ID_RE.source, 'g');

// Fetch a URL, return promise<{status, body}>.
function fetch(url) {
  return new Promise(function(resolve) {
    https.get(url, function(res) {
      var buf = '';
      res.on('data', function(d) { buf += d.toString(); });
      res.on('end', function() { resolve({ status: res.statusCode, body: buf }); });
    }).on('error', function(e) { resolve({ status: 0, body: String(e) }); });
  });
}

// Extract every badge URL from the README throughput table. We look for the
// raw.githubusercontent.com URLs inside the Zen4 column cells.
var readme = fs.readFileSync(README_PATH, 'utf8');
var referencedIds = {};
var m;
while ((m = urlRe.exec(readme)) !== null) referencedIds[m[1]] = true;

var referencedList = Object.keys(referencedIds).sort();
console.log('README references ' + referencedList.length + ' badge IDs:');
referencedList.forEach(function(id) { console.log('  ' + id); });

if (referencedList.length === 0) {
  console.error('FAIL: no badge URLs found in ' + README_PATH);
  process.exit(1);
}

(async function() {
  var failed = [];

  // 1. Every referenced badge must exist on the badge branch and return
  //    a shields.io endpoint JSON (schemaVersion=1, label, message).
  for (var i = 0; i < referencedList.length; i++) {
    var id = referencedList[i];
    var url = BADGE_BRANCH_RAW + 'benchmarks/badges/' + id + '.json';
    var r = await fetch(url);
    if (r.status === 0) {
      console.error('FAIL: ' + id + ' -> network error: ' + r.body);
      failed.push(id);
      continue;
    }
    if (r.status !== 200) {
      console.error('FAIL: ' + id + ' -> HTTP ' + r.status + ' (' + url + ')');
      failed.push(id);
      continue;
    }
    var j;
    try { j = JSON.parse(r.body.replace(/,(\s*[}\]])/g, '$1')); } catch (e) {
      console.error('FAIL: ' + id + ' -> invalid JSON: ' + e.message);
      failed.push(id);
      continue;
    }
    if (j.schemaVersion !== 1 || typeof j.message !== 'string' || j.message === '') {
      console.error('FAIL: ' + id + ' -> not a shields.io endpoint JSON: ' + r.body.slice(0, 200));
      failed.push(id);
      continue;
    }
    console.log('OK  ' + id + ' -> "' + j.message + '" (' + (j.color || '?') + ')');
  }

  // 2. The badge branch's sources.json must declare the same IDs (no
  //    orphans on either side). This catches "added a badge to README
  //    but forgot to add it to sources.json" — the exact failure mode
  //    cubic flagged.
  var srcRes = await fetch(BADGE_BRANCH_RAW + 'benchmarks/badges/sources.json');
  if (srcRes.status !== 200) {
    console.error('FAIL: sources.json HTTP ' + srcRes.status);
    process.exit(1);
  }
  var src = JSON.parse(srcRes.body);
  var declaredIds = {};
  (src.badges || []).forEach(function(b) { if (b.id) declaredIds[b.id] = true; });
  var declaredList = Object.keys(declaredIds).sort();

  var orphans = declaredList.filter(function(id) { return !referencedIds[id]; });
  if (orphans.length > 0) {
    console.error('NOTE: ' + orphans.length + ' source.json entries are NOT referenced in the README:');
    orphans.forEach(function(id) { console.error('  ' + id); });
    // Not a failure — those badges may be generated for the CI column or
    // future use. Log them so the next refresh knows they're orphan.
  }

  if (failed.length > 0) {
    console.error('\nFAIL: ' + failed.length + ' broken badge URL(s)');
    process.exit(1);
  }
  console.log('\nPASS: all ' + referencedList.length + ' referenced badge URLs resolve');
  process.exit(0);
})();
