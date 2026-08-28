"use strict";
/* Contract test (cubic review 4949074978 on PR #92, P3; updated 2026-08-28
 * for the post-16G-measured state):
 *
 *   - For every row in the README throughput table whose Zen4 badge
 *     resolves to a "pending" message, the per-row "Notes" cell MUST
 *     contain a concrete cause explanation, and the table-level footnote
 *     MUST NOT mis-attribute that row's pending to a cause different
 *     from the per-row note.
 *   - The P3 finding (PR #92): footnote said "The 16 GiB and 10 GiB/262k
 *     rows remain `pending` (V8 4 GiB Buffer cap blocks the larger shapes
 *     — see #91)" but the 10 GiB/262144/40 KiB row's own per-row note said
 *     "10 GiB / 262144 / 40 KiB is non-pow2 (rejected by #87); pending" —
 *     i.e. the 10 GiB row is blocked earlier by #87's pow2 contract, not
 *     by V8's Buffer cap. The two pending rows have different causes and
 *     the footnote must distinguish them.
 *   - Update (2026-08-28, PR for Zen4 2026-08-28 re-measure): the 16 GiB
 *     row is now measured (no longer pending) on Node 22.22.3. The 10 GiB
 *     row was removed from the README in #99 (40 KiB blocks fail
 *     lib/par3gen.js:1270's `Block size must be a power of 2` check). The
 *     test handles the 0-pending-rows case as a PASS (the contract is
 *     satisfied vacuously), while still applying the per-row + lumping
 *     checks if any future PR adds a new pending row.
 *   - cubic P2 (PR #101): if any badge fetch returns a network/HTTP/parse
 *     error, the test fails immediately. Previously a fetch failure
 *     left `body === null`, the row was treated as non-pending, and the
 *     vacuous-truth path could silently skip the per-row + lumping
 *     checks — a false PASS on broken CI.
 *
 * Run: `node test/readme-pending-row-footnote.js`
 *   - exit 0: no pending rows are mis-attributed
 *   - exit 1: a badge fetch/parse failed, a pending row has no per-row
 *     note, OR the footnote lumps two differently-caused pending rows
 *     under the same cause
 */

var fs = require('fs');
var path = require('path');
var https = require('https');
var shared = require('./readme-badges-shared');

var README_PATH = path.join(__dirname, '..', 'README.md');
var BADGE_BRANCH_RAW = shared.BADGE_BRANCH_RAW;

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
// the JSON filename stem. (The pattern is shared with test/readme-badge-urls.js
// via test/readme-badges-shared.js.)
var zen4IdRe = new RegExp(shared.ZEN4_BADGE_ID_RE.source, 'i');

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
        if (res.statusCode !== 200) return resolve({ status: res.statusCode, body: null, fetchError: 'HTTP ' + res.statusCode });
        // Tolerate trailing commas (the live badge JSONs have a trailing `,\n}`)
        var cleaned = buf.replace(/,(\s*[}\]])/g, '$1');
        try { resolve({ status: 200, body: JSON.parse(cleaned) }); } catch (e) { resolve({ status: 200, body: null, parseError: e.message }); }
      });
    }).on('error', function(e) { resolve({ status: 0, body: null, error: String(e) }); });
  });
}

// Track fetch/parse failures so the test cannot false-PASS when badge
// resolution silently breaks. Without this, a row with a broken badge
// would be treated as non-pending (the vacuous-truth path) and the
// per-row + lumping checks would silently not run.
var fetchFailures = [];
function trackFetch(label, r) {
  if (r.fetchError) {
    console.error('FAIL: ' + label + ' badge fetch error: ' + r.fetchError);
    fetchFailures.push(label);
  } else if (r.parseError) {
    console.error('FAIL: ' + label + ' badge JSON parse error: ' + r.parseError);
    fetchFailures.push(label);
  } else if (r.error) {
    console.error('FAIL: ' + label + ' badge network error: ' + r.error);
    fetchFailures.push(label);
  }
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
    trackFetch(badgeId, r);
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

  // Fail fast if any badge fetch/parse failed — otherwise a broken badge
  // could leave us with zero pending rows and the vacuous-truth path
  // would silently not run the per-row + lumping checks.
  if (fetchFailures.length > 0) {
    console.error('FAIL: ' + fetchFailures.length + ' badge fetch/parse failure(s) — cannot determine pending state');
    process.exit(1);
  }

  console.log('Found ' + pendingRows.length + ' pending row(s) (Zen4 message = "pending"):');
  for (var i = 0; i < pendingRows.length; i++) {
    var p = pendingRows[i];
    console.log('  ' + p.projectFormat + ' / ' + p.workload + '  (badge: ' + p.badgeId + ')');
    console.log('    Notes: ' + p.notes);
  }
  // Vacuous-truth: zero pending rows means the contract is satisfied.
  // (Previously this exited with failure; that was the wrong contract for a
  // test — a test should pass when its assertion holds, including when
  // there is nothing to assert. We still want to catch the lumping bug
  // below in case a future PR adds a pending row, so we fall through.)
  if (pendingRows.length === 0) {
    console.log('PASS: 0 pending rows in the README — contract is vacuously satisfied');
    // We still evaluate the lumping check below (a future PR with two
    // pending rows would re-trigger it); but since there's nothing to
    // attribute, the lumping pattern can't fire. Fall through.
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
  // The Workload cell is the natural-language description (e.g.
  // "10 GiB Create ($R=8$)") and does NOT contain the slice count;
  // match on the badge id (e.g. par3-10g-262144-zen4) so the rule
  // actually fires — the old code matched on p.workload, which made
  // the 10 GiB/262k branch always false (workload doesn't contain
  // "262k"/"262144"; the slice count lives in a separate cell), so
  // Rule 2's 10 GiB → pow2/#87 attribution check silently never
  // ran and the test would not have caught a future footnote
  // dropping that attribution (cubic review 95db4ba2 P2 on PR #95).
  // The 16 GiB branch's old /16 GiB/i workload regex DID match the
  // "16 GiB Create ($R=8$)" cell, so the 16 GiB attribution check
  // fired under the old code; we switch to badgeId for both so the
  // future maintainer only has to remember one rule.
  var has16GiB = pendingRows.some(function(p) { return /16g-262144/.test(p.badgeId); });
  var has10GiB_262k = pendingRows.some(function(p) { return /10g-262144/.test(p.badgeId); });

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
  if (pendingRows.length === 0) {
    console.log('\nPASS: 0 pending rows in the README — no cause-attribution checks needed');
  } else {
    console.log('\nPASS: all ' + pendingRows.length + ' pending row(s) have per-row causes; footnote splits the causes correctly');
  }
  process.exit(0);
})().catch(function(e) { console.error('FAIL: uncaught: ' + e.stack); process.exit(1); });
