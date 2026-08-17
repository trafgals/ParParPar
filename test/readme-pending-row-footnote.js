"use strict";
/* Contract test (cubic review 4949074978 on PR #92, P3): every row in the
 * README throughput table whose Zen4 badge resolves to a "pending" message
 * must have its own per-row "Notes" cell explaining the cause, and the
 * table-level footnote must NOT mis-attribute that row's pending to a
 * cause different from the per-row note.
 *
 * The P3 finding: PR #92's footnote said "The 16 GiB and 10 GiB/262k rows
 * remain `pending` (V8 4 GiB Buffer cap blocks the larger shapes — see #91)"
 * but the 10 GiB/262144/40 KiB row's own per-row note said
 * "10 GiB / 262144 / 40 KiB is non-pow2 (rejected by #87); pending" — i.e.
 * the 10 GiB row is blocked earlier by #87's pow2 contract, not by V8's
 * Buffer cap. The two pending rows have different causes and the footnote
 * must distinguish them.
 *
 * Run: `node test/readme-pending-row-footnote.js`
 *   - exit 0: pending rows are self-consistent
 *   - exit 1: a pending row's footnote mis-attributes its cause, or a
 *     pending row has no per-row note at all
 */

var fs = require('fs');
var path = require('path');
var https = require('https');

var README_PATH = path.join(__dirname, '..', 'README.md');
var BADGE_BRANCH_RAW = 'https://raw.githubusercontent.com/trafgals/ParParPar/feat/ci-benchmark-badge/';

var readme = fs.readFileSync(README_PATH, 'utf8');

// Locate the throughput table.
var tableStart = readme.indexOf('| Project / Format | Workload');
if (tableStart < 0) {
  console.error('FAIL: could not locate throughput table header in README');
  process.exit(1);
}
var tableEndMatch = readme.substring(tableStart).match(/\r?\n---\r?\n/);
if (!tableEndMatch) {
  console.error('FAIL: could not locate end of throughput table in README');
  process.exit(1);
}
var tableEnd = tableStart + tableEndMatch.index;
// The footnote is the italicized paragraph BETWEEN the last data row and the `---`.
var tableSection = readme.substring(tableStart, tableEnd);
var dataRows = [];
var lines = tableSection.split(/\r?\n/);
for (var i = 0; i < lines.length; i++) {
  var line = lines[i];
  if (!line.startsWith('| **')) continue;
  if (line.indexOf('| :--- |') >= 0) continue;
  dataRows.push(line);
}

// Extract the Zen4 badge ID from a row's Zen4 cell. The cell looks like
// `[![label](shields-url)](anchor)` where the shields URL contains the
// raw.githubusercontent.com path. We match the URL-encoded form to get
// the JSON filename stem.
var zen4IdRe = /raw\.githubusercontent\.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2F([a-z0-9-]+)\.json/i;

function splitRow(line) {
  var trimmed = line.replace(/^\| /, '').replace(/ \|$/, '');
  return trimmed.split(' | ');
}

function fetchJson(url) {
  return new Promise(function(resolve) {
    https.get(url, function(res) {
      var buf = '';
      res.on('data', function(d) { buf += d.toString(); });
      res.on('end', function() {
        if (res.statusCode !== 200) return resolve({ status: res.statusCode, body: null });
        try { resolve({ status: 200, body: JSON.parse(buf) }); } catch (e) { resolve({ status: 200, body: null, parseError: e.message }); }
      });
    }).on('error', function(e) { resolve({ status: 0, body: null, error: String(e) }); });
  });
}

(async function() {
  // First pass: identify all pending Zen4 rows.
  var pendingRows = [];
  for (var i = 0; i < dataRows.length; i++) {
    var cells = splitRow(dataRows[i]);
    if (cells.length < 7) continue;
    var projectFormat = cells[0];
    var workload = cells[1];
    var zen4Cell = cells[4];
    var ciCell = cells[5];
    var notes = cells[6];
    var m = zen4IdRe.exec(zen4Cell);
    if (!m) continue;
    var badgeId = m[1];
    var url = BADGE_BRANCH_RAW + 'benchmarks/badges/' + badgeId + '.json';
    var r = await fetchJson(url);
    var isPending = r.body && typeof r.body.message === 'string' && /pending/i.test(r.body.message);
    if (isPending) {
      pendingRows.push({
        projectFormat: projectFormat,
        workload: workload,
        badgeId: badgeId,
        message: r.body.message,
        ciCell: ciCell,
        notes: notes
      });
    }
  }

  console.log('Found ' + pendingRows.length + ' pending row(s) (Zen4 message = "pending"):');
  for (var i = 0; i < pendingRows.length; i++) {
    var p = pendingRows[i];
    console.log('  ' + p.projectFormat + ' / ' + p.workload + '  (badge: ' + p.badgeId + ')');
    console.log('    Notes: ' + p.notes);
  }
  if (pendingRows.length === 0) {
    console.error('FAIL: no pending rows detected — test precondition is no longer relevant; remove the test');
    process.exit(1);
  }

  var failed = 0;

  // Rule 1: every pending row's per-row Notes cell must contain a concrete
  // cause explanation (a non-trivial cause word or issue reference).
  var causeWords = /\b(V8|Buffer cap|cap|limit|pow2|power[- ]of[- ]2|non-pow2|rejected|block[- ]size|recovery contract|#[ ]?\d+|issue|architectural|16 GiB|10 GiB|40 KiB|64 KiB|rejected by)/i;
  for (var i = 0; i < pendingRows.length; i++) {
    var p = pendingRows[i];
    if (!causeWords.test(p.notes)) {
      console.error('FAIL: pending row "' + p.projectFormat + ' / ' + p.workload + '" has no cause explanation in its Notes cell: "' + p.notes + '"');
      failed++;
    }
  }

  // Rule 2: the table-level footnote must NOT lump together pending rows
  // with different causes under a single shared cause.
  //
  // The P3 finding was: the 16 GiB row is blocked by the V8 Buffer cap
  // (#91), and the 10 GiB/262k row is blocked by the pow2 contract (#87).
  // A correct footnote must distinguish them.
  var footnoteMatch = tableSection.match(/\*All throughput[\s\S]*?branch\.\*/);
  if (!footnoteMatch) {
    console.error('FAIL: could not locate table footnote (expected *All throughput ...* in the table section)');
    failed++;
    // Skip rule-2 evaluations; just report failures.
    console.error('\nFAIL: ' + failed + ' pending-row footnote contract violation(s)');
    process.exit(1);
  }
  var footnote = footnoteMatch[0];

  // Look at the specifically-named pending rows we expect to find.
  var has16GiB = pendingRows.some(function(p) { return /16 GiB/i.test(p.workload); });
  var has10GiB_262k = pendingRows.some(function(p) { return /10 GiB.*262k/i.test(p.workload) || /10 GiB.*262144/i.test(p.workload); });

  // The footnote's 16 GiB mention must point at V8 Buffer cap / #91.
  // We look 200 chars after the first "16 GiB" to give the parenthetical
  // explanation room.
  if (has16GiB) {
    var s16 = footnote.match(/16 GiB[\s\S]{0,200}/);
    if (!s16) {
      console.error('FAIL: footnote does not mention 16 GiB');
      failed++;
    } else if (!/(V8|Buffer cap|#[ ]?91|architectural)/i.test(s16[0])) {
      console.error('FAIL: footnote mentions 16 GiB but does not attribute it to V8 Buffer cap / #91: "' + s16[0] + '"');
      failed++;
    }
  }
  // The footnote's 10 GiB/262k mention must point at pow2 / #87.
  if (has10GiB_262k) {
    var s10 = footnote.match(/10 GiB[\s\S]{0,200}/);
    if (!s10) {
      console.error('FAIL: footnote does not mention 10 GiB/262k');
      failed++;
    } else if (!/(pow2|power[- ]of[- ]2|#[ ]?87)/i.test(s10[0])) {
      console.error('FAIL: footnote mentions 10 GiB/262k but does not attribute it to pow2 / #87: "' + s10[0] + '"');
      failed++;
    }
  }

  // Rule 3: a single sentence in the footnote must NOT attribute two
  // differently-caused pending rows to the same cause. We use a simple
  // test: the footnote should not contain "X and Y rows remain `pending`
  // (causeA — see #NNN)" when the actual per-row causes for X and Y
  // are different. Concretely, we look for the original P3 form:
  //   "16 GiB and 10 GiB/262k rows remain `pending` (V8 4 GiB Buffer cap
  //    blocks the larger shapes — see #91)"
  // which lumps both under the V8 Buffer cap. The post-#92 footnote uses
  // two separate sentences, so the lumping pattern should not match.
  var lumpsBoth = /16 GiB[\s\S]{0,80}10 GiB[\s\S]{0,200}(V8|Buffer cap|#[ ]?91)/i.test(footnote);
  if (lumpsBoth) {
    console.error('FAIL: footnote lumps 16 GiB and 10 GiB/262k under the SAME cause (the original P3 finding):');
    var m = footnote.match(/16 GiB[\s\S]{0,300}10 GiB[\s\S]{0,300}/);
    if (m) console.error('    ' + m[0]);
    failed++;
  }

  if (failed > 0) {
    console.error('\nFAIL: ' + failed + ' pending-row footnote contract violation(s)');
    process.exit(1);
  }
  console.log('\nPASS: all ' + pendingRows.length + ' pending row(s) have per-row causes; footnote splits the 16 GiB and 10 GiB/262k causes correctly');
  process.exit(0);
})().catch(function(e) { console.error('FAIL: uncaught: ' + e.stack); process.exit(1); });
