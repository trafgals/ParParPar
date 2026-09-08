"use strict";
/* Contract test for the 14 GiB / 229376 / 64 KiB / R=32768 pending row
 * added to the README throughput table by PR #103 (closes issue #91 on
 * the chunked-input pipeline CI leg + Node 22 buffer-cap lift).
 *
 * Pin:
 *   1. The 14G row exists in the README throughput table, references
 *      the `par3-14g-229376-zen4` Zen4 badge id, and has `(pending)`
 *      flagged in its workload cell (no measured MB/s figure).
 *   2. The 14G row's `Notes` cell names every contract clause the
 *      plan established:
 *        - "Fenger" (kernel identity at R >= FENGER_MIN_R)
 *        - "padded" (engine's always-pad policy at non-pow2 N=229376)
 *        - "next_pow2" or `262144` (the rounding target)
 *        - "Node 22" (Node 22 platform bump required to lift the
 *          V8 Buffer cap that previously blocked this geometry)
 *        - "test/par3-fenger-padded-engine.js" (the test that pins
 *          kernel parity at the always-pad boundary)
 *        - "test/par3-chunked-inputs.js" (the test that exercises the
 *          multi-chunk pipeline path on CI)
 *      This is the multi-faceted contract per `.omo/plans/variable-
 *      archive-fenger.md`; missing any one of these would let a future
 *      maintainer drop the test references or revert to a non-pow2
 *      treatment without the test flagging it.
 *   3. The table-level footnote mentions the 14G pending row and
 *      attributes the pending state to the Node 22 platform change
 *      (NOT to V8 Buffer cap or pow2 — those are the 16G and 10G
 *      historical pending states).
 *   4. The Zen4 badge id resolves to a live shields.io endpoint JSON
 *      served from `feat/ci-benchmark-badge` with `message: "pending"`,
 *      and `sources.json` declares the same id with a matching
 *      `message: "pending"`.
 *
 * Run: `node test/readme-14g-pending-row.js`
 *   - exit 0: all four contract clauses pass
 *   - exit 1: any clause fails, OR the badge fetch/parse errors
 *
 * Verify the test FAILS on the unfixed README (where the 14G row is
 * absent, the footnote doesn't mention 14G, and the badge fetch would
 * 404). See the verify-test-fails recipe in
 * references/zen4-rerun-and-cubic-p3-content-consistency.md.
 */

var fs = require('fs');
var path = require('path');
var https = require('https');
var shared = require('./readme-badges-shared');

var README_PATH = path.join(__dirname, '..', 'README.md');
var SOURCES_PATH = path.join(__dirname, '..', 'benchmarks', 'badges', 'sources.json');
var BADGE_BRANCH_RAW = shared.BADGE_BRANCH_RAW;
var BADGE_ID = 'par3-14g-229376-zen4';

function fail(msg) {
  console.error('FAIL: ' + msg);
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

// Clause 2: notes cell names all six contract clauses.
var notes = fourteenGRow.cells[6];
var contractClauses = [
  { name: 'Fenger (kernel identity at R >= FENGER_MIN_R)', re: /\bFenger\b/i },
  { name: 'padded (engine always-pad policy at non-pow2 N)', re: /\bpadded?\b/i },
  { name: 'next_pow2 = 262144 (Fenger rounding target)', re: /(next[ _\\]*pow2|262[, ]?144)/i },
  { name: 'Node 22+ required (platform bump lifts V8 Buffer cap)', re: /\bNode[ ]?(22|[ ]?22\+)/i },
  { name: 'test/par3-fenger-padded-engine.js (kernel parity contract)', re: /test\/par3-fenger-padded-engine\.js/ },
  { name: 'test/par3-chunked-inputs.js (multi-chunk pipeline regression)', re: /test\/par3-chunked-inputs\.js/ }
];
contractClauses.forEach(function(c) {
  check('14G notes cell contains "' + c.name + '"', c.re.test(notes));
});

// Clause 3: table-level footnote attributes the 14G pending row to the
// Node 22 platform bump (NOT to V8 Buffer cap or pow2 — those are the
// 16G and 10G historical pending states).
var footnoteMatch = tableSection.match(/\*All throughput[\s\S]*?branch\.\*/);
if (!footnoteMatch) {
  check('14G footnote presence', false, 'could not locate table footnote');
} else {
  var footnote = footnoteMatch[0];
  check('14G footnote mentions the 14G/229376 row', /(14 GiB|14GB|229,?376)/i.test(footnote));
  check('14G footnote attributes pending to Node 22 platform bump',
    /\b14 GiB[\s\S]{0,300}(Node[ ]?(22|[ ]?22\+)|V8 Buffer cap lifted|engines\.node)/i.test(footnote) ||
    /\b14 GiB\/229376[\s\S]{0,300}(Node[ ]?(22|[ ]?22\+)|V8 Buffer cap lifted|engines\.node)/i.test(footnote));
  // Anti-attribute: the 14G footnote must NOT lump pending under V8 Buffer cap only
  // (that's the 16G row's cause) or under pow2 (that's the 10G row's cause).
  check('14G footnote does NOT attribute pending to pow2 alone (the 10G cause)',
    !/14 GiB[\s\S]{0,200}(pow2|power[- ]of[- ]2|#[ ]?87)/i.test(footnote));
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
