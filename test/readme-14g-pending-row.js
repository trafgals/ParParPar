"use strict";
/* Contract test for the 14 GiB / 229376 / 64 KiB / R=32768 pending row
 * in the README throughput table.
 *
 * Pin:
 *   1. The 14G row exists, references `par3-14g-229376-zen4`, and has
 *      `(pending)` in the workload cell.
 *   2. The 14G `Notes` cell contains Fenger / padded / <sup>[3]</sup>.
 *      Footnote [3] independently contains the full technical contract:
 *      Fenger, padded, next_pow2, Node 22, test/par3-fenger-padded-engine.js,
 *      and test/par3-chunked-inputs.js. Missing any one would let a
 *      future edit silently drop a contract clause.
 *   3. The footnote attributes 14G pending to Node 22, not to pow2 or
 *      V8 Buffer cap (those are different rows' causes).
 *   4. The live Zen4 badge JSON (feat/ci-benchmark-badge) and the local
 *      sources.json both declare the id with `message: "pending"`.
 *
 * Run: `node test/readme-14g-pending-row.js`
 */

var fs = require('fs');
var path = require('path');
var https = require('https');
var shared = require('./readme-badges-shared');

var README_PATH = path.join(__dirname, '..', 'README.md');
var SOURCES_PATH = path.join(__dirname, '..', 'benchmarks', 'badges', 'sources.json');
var BADGE_BRANCH_RAW = shared.BADGE_BRANCH_RAW;
var BADGE_ID = 'par3-14g-229376-zen4';

function fail(msg, detail) {
  console.error('FAIL: ' + msg + (detail ? ' — ' + detail : ''));
  if (typeof failures === 'number') failures++;
  process.exitCode = 1;
}

function fetchJson(url) {
  return new Promise(function(resolve) {
    https.get(url, function(res) {
      var buf = '';
      res.on('data', function(d) { buf += d.toString(); });
      res.on('end', function() {
        if (res.statusCode !== 200) return resolve({ status: res.statusCode, body: null, error: 'HTTP ' + res.statusCode });
        // Tolerate trailing commas (live badge JSONs have `"color": "...",\n}`).
        var cleaned = buf.replace(/,(\s*[}\]])/g, '$1');
        try { resolve({ status: 200, body: JSON.parse(cleaned) }); }
        catch (e) { resolve({ status: 200, body: null, error: 'parse: ' + e.message }); }
      });
    }).on('error', function(e) { resolve({ status: 0, body: null, error: String(e) }); });
  });
}

var readme = fs.readFileSync(README_PATH, 'utf8');
var sources = JSON.parse(fs.readFileSync(SOURCES_PATH, 'utf8'));

var failures = 0;
function check(label, ok, detail) {
  if (ok) {
    console.log('OK  ' + label);
  } else {
    console.error('FAIL: ' + label + (detail ? ' — ' + detail : ''));
    failures++;
  }
}

// Find the throughput table section (between the header row and the `---`
// closing line) to scope matches to the table area only.
var tableStart = readme.indexOf('| Project / Format | Workload');
if (tableStart < 0) fail('could not locate throughput table header in README');
var tableEndMatch = readme.substring(tableStart).match(/\r?\n---\r?\n/);
var tableEnd = tableStart + (tableEndMatch ? tableEndMatch.index : readme.length);
var tableSection = readme.substring(tableStart, tableEnd);

// Pull every data row from the table (starts with `| **`, ignores the
// `|:---|` separator row).
var dataRows = tableSection.split(/\r?\n/).filter(function(line) {
  return line.startsWith('| **') && line.indexOf('| :--- |') < 0;
});

// Find the 14G row by its badge id inside the Zen4 cell.
var zen4IdRe = new RegExp(shared.ZEN4_BADGE_ID_RE.source, 'i');
var fourteenGRow = null;
for (var i = 0; i < dataRows.length; i++) {
  var cells = dataRows[i].replace(/^\| /, '').replace(/ \|$/, '').split(' | ');
  if (cells.length < 7) continue;
  var m = zen4IdRe.exec(cells[4]);
  if (m && m[1] === BADGE_ID) { fourteenGRow = { cells: cells, raw: dataRows[i] }; break; }
}
check('14G pending row exists in README throughput table', !!fourteenGRow);

if (!fourteenGRow) {
  console.error('\nFAIL: 14G row not found in the README throughput table; cannot evaluate further clauses');
  process.exit(1);
}

// Clause 1: workload cell says "(pending)" — no measured number for this row yet.
var workload = fourteenGRow.cells[1];
check('14G workload cell flags (pending)',
  /pending/i.test(workload),
  'workload="' + workload + '"');

// Extract footnote [3] specifically (cubic review on PR #106 violation 1):
var fn3Match = tableSection.match(/- \*\*\[3\][^\n]*:?([\s\S]*?)(?=\n- \*\*\[|\n####|\n---|$)/);
var footnote3 = fn3Match ? fn3Match[0].trim() : '';
check('Footnote [3] exists for 14G shape in tableSection', !!footnote3, 'could not extract footnote [3]');

// Clause 2: Check notes cell and footnote [3] independently.
var notes = fourteenGRow.cells[6];
// The Notes cell must contain the key row descriptors:
check('14G notes cell contains "Fenger"', /\bFenger\b/i.test(notes), 'notes="' + notes + '"');
check('14G notes cell contains "padded"', /\bpadded?\b/i.test(notes), 'notes="' + notes + '"');
check('14G notes cell references footnote [3]', /<sup>\[3\]<\/sup>/.test(notes), 'notes="' + notes + '"');

// Footnote [3] must contain all technical contract clauses:
var footnoteClauses = [
  { name: 'Fenger (kernel identity at R >= FENGER_MIN_R)', re: /\bFenger\b/i },
  { name: 'padded (engine always-pad policy at non-pow2 N)', re: /\bpadded\b/i },
  { name: 'next_pow2 = 262144 (Fenger rounding target)', re: /(next[ _\\]*pow2|262[, ]?144)/i },
  { name: 'Node 22+ required (platform bump lifts V8 Buffer cap)', re: /\bNode[ ]?(22|[ ]?22\+)/i },
  { name: 'test/par3-fenger-padded-engine.js (kernel parity contract)', re: /test\/par3-fenger-padded-engine\.js/ },
  { name: 'test/par3-chunked-inputs.js (multi-chunk pipeline regression)', re: /test\/par3-chunked-inputs\.js/ }
];
footnoteClauses.forEach(function(c) {
  check('14G footnote [3] contains "' + c.name + '"', c.re.test(footnote3));
});

// Clause 3: footnote [3] attributes 14G pending to Node 22, NOT to pow2
// or V8 Buffer cap (those are different rows' causes).
if (!footnote3) {
  check('14G footnote [3] presence', false, 'could not locate footnote [3]');
} else {
  check('14G footnote [3] mentions the 14G/229376 row', /(14 GiB|14GB|229,?376)/i.test(footnote3));
  // Positive attribution: pending to Node 22, NOT to V8 Buffer cap
  // (which is the 16G row's cause). Don't accept "V8 Buffer cap" as
  // a passing alternative — that's the 16G cause and would falsely
  // pass an edit attributing 14G pending to it.
  check('14G footnote [3] attributes pending to Node 22 platform bump',
    /\b14 GiB[\s\S]{0,300}(Node[ ]?(22|[ ]?22\+)|engines\.node)/i.test(footnote3) ||
    /\b14 GiB\/229376[\s\S]{0,300}(Node[ ]?(22|[ ]?22\+)|engines\.node)/i.test(footnote3));
  // Anti-attribute checks: forbid "14G is pending because of <cause>".
  // pow2 / #87 is the 10G row's cause; V8 Buffer cap is the 16G row's
  // cause. The bridge pattern (pending ... <cause> OR <cause> ...
  // pending) catches the causal phrasing without false-positiving on
  // parenthetical context like "Node 22+ required (V8 Buffer cap
  // lifted in Node 22)" — which explains the prerequisite, not the
  // cause for 14G pending.
  check('14G footnote [3] does NOT attribute pending to pow2 alone (the 10G cause)',
    !/14 GiB[\s\S]{0,300}pending[\s\S]{0,200}(pow2|power[- ]of[- ]2|#[ ]?87)/i.test(footnote3) &&
    !/14 GiB[\s\S]{0,200}(pow2|power[- ]of[- ]2|#[ ]?87)[\s\S]{0,80}pending/i.test(footnote3));
  check('14G footnote [3] does NOT attribute pending to V8 Buffer cap alone (the 16G cause)',
    !/14 GiB[\s\S]{0,300}pending[\s\S]{0,200}(V8 Buffer cap|#[ ]?91)/i.test(footnote3) &&
    !/14 GiB[\s\S]{0,200}(V8 Buffer cap|#[ ]?91)[\s\S]{0,80}pending/i.test(footnote3));
}

// Clause 4: live badge + sources.json both declare pending.
var sourcesEntry = (sources.badges || []).find(function(b) { return b.id === BADGE_ID; });
check('sources.json declares ' + BADGE_ID, !!sourcesEntry);
if (sourcesEntry) {
  check('sources.json entry for ' + BADGE_ID + ' has message: "pending"',
    sourcesEntry.message === 'pending',
    'actual="' + sourcesEntry.message + '"');
}

(async function() {
  var r = await fetchJson(BADGE_BRANCH_RAW + 'benchmarks/badges/' + BADGE_ID + '.json');
  if (r.error) {
    check('live badge JSON fetch', false, r.error);
  } else {
    check('live badge JSON fetch', true);
    check('live badge JSON has message: "pending"',
      r.body && r.body.message === 'pending',
      'actual="' + (r.body && r.body.message) + '"');
    check('live badge JSON has schemaVersion=1',
      r.body && r.body.schemaVersion === 1);
  }

  if (failures > 0) {
    console.error('\nFAIL: ' + failures + ' 14G pending-row contract violation(s)');
    process.exit(1);
  }
  console.log('\nPASS: 14G pending-row contract holds (row present, all clauses pinned, badge live)');
  process.exit(0);
})().catch(function(e) { console.error('FAIL: uncaught: ' + e.stack); process.exit(1); });
