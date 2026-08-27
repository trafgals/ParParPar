"use strict";
/* Contract test for the 32G/64G throughput rows (PR #102, 2026-08-28):
 *
 *   - For each row in the README throughput table whose Zen4 badge ID matches
 *     par3-32g-524288-zen4 or par3-64g-1048576-zen4:
 *       1. The kernel label must identify the kernel used (Barycentric or
 *          other cost-model-dispatched kernel) and must NOT say "Fenger
 *          fast-path" (which is the wrong kernel at R=8 — Fenger's
 *          FENGER_MIN_R=8192 gate is well above R=8).
 *       2. The per-row Notes cell must mention peak RSS (memory pressure
 *          is the headline caveat for these rows).
 *       3. The per-row Notes cell must mention the swap / pagefile
 *          behavior (Windows pagefile auto-grow is what made these
 *          feasible on a 63 GiB-RAM host).
 *       4. The 64 GiB row MUST be flagged as swap-bound — the alloc
 *          pool is 64 GiB + 64 KiB, larger than the 63 GiB physical RAM,
 *          and the throughput number reflects swap-fault overhead, not
 *          pure kernel cost.
 *
 *   - The table-level footnote must distinguish the 32G row (alloc pool
 *     fits in RAM) from the 64G row (alloc pool exceeds RAM, swap-bound).
 *
 *   - cubic P2 (PR #101, applied to this test too): if any badge fetch
 *     returns a network/HTTP/parse error, the test fails immediately.
 *     Otherwise a fetch failure could leave us with zero pending rows
 *     and the vacuous-truth path would silently skip the per-row
 *     checks (a false PASS on broken CI).
 *
 * Run: `node test/readme-throughput-shape-feasibility.js`
 *   - exit 0: every 32G/64G row's contract assertions pass
 *   - exit 1: a badge fetch failed, a kernel label is wrong, a per-row
 *     note is missing the required caveats, or the footnote fails to
 *     distinguish the two shapes
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
var tableSection = readme.substring(tableStart, tableEnd);

// Parse data rows.
var lines = tableSection.split(/\r?\n/);
var dataRows = [];
for (var i = 0; i < lines.length; i++) {
	var line = lines[i];
	if (!line.startsWith('| **')) continue;
	if (line.indexOf('| :--- |') >= 0) continue;
	dataRows.push(line);
}

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
				// Tolerate trailing commas in the live badge JSONs.
				var cleaned = buf.replace(/,(\s*[}\]])/g, '$1');
				try { resolve({ status: 200, body: JSON.parse(cleaned) }); } catch (e) { resolve({ status: 200, body: null, parseError: e.message }); }
			});
		}).on('error', function(e) { resolve({ status: 0, body: null, error: String(e) }); });
	});
}

// Track fetch/parse failures so the test cannot false-PASS when badge
// resolution silently breaks (cubic P2 from PR #101, applied here too).
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
	// 1. Locate the 32G and 64G rows in the throughput table.
	var rows32g = [];
	var rows64g = [];
	for (var i = 0; i < dataRows.length; i++) {
		var cells = splitRow(dataRows[i]);
		if (cells.length < 7) continue;
		var m = zen4IdRe.exec(cells[4]);
		if (!m) continue;
		var badgeId = m[1];
		var url = BADGE_BRANCH_RAW + 'benchmarks/badges/' + badgeId + '.json';
		var r = await fetchJson(url);
		trackFetch(badgeId, r);
		// We don't actually need the badge body — we just want the
		// per-row Notes cell for the row whose badgeId matches.
		if (/par3-32g-524288/.test(badgeId)) rows32g.push({ cells: cells, badgeId: badgeId, badge: r.body });
		if (/par3-64g-1048576/.test(badgeId)) rows64g.push({ cells: cells, badgeId: badgeId, badge: r.body });
	}

	if (fetchFailures.length > 0) {
		console.error('FAIL: ' + fetchFailures.length + ' badge fetch/parse failure(s)');
		process.exit(1);
	}

	console.log('Found ' + rows32g.length + ' 32G row(s) and ' + rows64g.length + ' 64G row(s)');

	var failed = 0;

	// 2. Per-row assertions.
	function assertRowContains(row, label, re, hint) {
		if (!re.test(row.cells[6])) {
			console.error('FAIL: ' + label + ' per-row Notes missing ' + hint + ': "' + row.cells[6] + '"');
			failed++;
		}
	}
	function assertRowDoesNotContain(row, label, re, hint) {
		if (re.test(row.cells[6])) {
			console.error('FAIL: ' + label + ' per-row Notes incorrectly contains ' + hint + ': "' + row.cells[6] + '"');
			failed++;
		}
	}

	// 32G assertions:
	//   - Kernel label identifies Barycentric (cost-model dispatch at R=8 < FENGER_MIN_R=8192)
	//   - Mentions peak RSS
	//   - Mentions pagefile (Windows auto-grow was the headline enabler)
	//   - Does NOT contain "Fenger fast-path" (Fenger is the WRONG kernel at R=8)
	//   - Does NOT contain "swap-bound" (alloc pool fits in RAM)
	var reBary32 = /\b(Barycentric|barycentric|cost[- ]model|FENGER|kernel)\b/i;
	var reRss32 = /\b(RSS|peak|GiB|32\.25)\b/;
	var rePagefile32 = /\b(pagefile|swap|memory wall|alloc pool|pagefile)\b/i;
	var reNoFenger32 = /\bFenger fast-path\b/;
	var reNoSwapBound32 = /\bswap-bound\b/;

	rows32g.forEach(function(row) {
		assertRowContains(row, '32G', reBary32, 'a kernel label (Barycentric / cost-model dispatch)');
		assertRowContains(row, '32G', reRss32, 'peak RSS');
		assertRowContains(row, '32G', rePagefile32, 'pagefile / memory wall mention');
		assertRowDoesNotContain(row, '32G', reNoFenger32, '"Fenger fast-path" (Fenger is the wrong kernel at R=8 < FENGER_MIN_R=8192)');
		assertRowDoesNotContain(row, '32G', reNoSwapBound32, '"swap-bound" (alloc pool fits in RAM)');
	});

	// 64G assertions:
	//   - Kernel label identifies Barycentric
	//   - Mentions peak RSS
	//   - Mentions pagefile (Windows auto-grow to 173 GiB was critical)
	//   - MUST mention "swap-bound" (the alloc pool exceeds 64 GiB physical RAM)
	//   - MUST mention "memory wall" or equivalent (the binding constraint)
	//   - Does NOT contain "Fenger fast-path"
	var reBary64 = /\b(Barycentric|barycentric|cost[- ]model|FENGER|kernel)\b/i;
	var reRss64 = /\b(RSS|peak|GiB|48\.34)\b/;
	var rePagefile64 = /\b(pagefile|swap|swap-bound|memory wall|alloc pool)\b/i;
	var reSwapBound64 = /\bswap-bound\b/i;
	var reMemoryWall64 = /\b(memory wall|exceeds)\b/i;
	var reNoFenger64 = /\bFenger fast-path\b/;

	rows64g.forEach(function(row) {
		assertRowContains(row, '64G', reBary64, 'a kernel label');
		assertRowContains(row, '64G', reRss64, 'peak RSS');
		assertRowContains(row, '64G', rePagefile64, 'pagefile / memory wall / swap-bound mention');
		assertRowContains(row, '64G', reSwapBound64, '"swap-bound" (the alloc pool is 64 GiB which exceeds 63 GiB physical RAM)');
		assertRowContains(row, '64G', reMemoryWall64, '"memory wall" or equivalent');
		assertRowDoesNotContain(row, '64G', reNoFenger64, '"Fenger fast-path"');
	});

	// 3. Footnote assertions.
	// The table-level footnote must:
	//   - Mention both 32G/524288 and 64G/1048576 (per-row attribution)
	//   - Distinguish 32G (alloc pool fits in RAM) from 64G (alloc pool
	//     exceeds RAM, swap-bound)
	//   - NOT lump them together under the same cause (the cubic P3 finding
	//     from PR #92, generalized to this row pair)
	//
	// The lumping check fires when a single sentence attributes BOTH
	// rows to the SAME cause. We split the footnote into sentences
	// (by `.`) and flag any sentence that contains BOTH a 32G reference
	// AND a 64G reference AND a shared cause phrase. Cross-sentence
	// attributions are fine (the natural way to write about two rows
	// with different causes) — only SAME-sentence lumping is wrong.
	var footnoteMatch = tableSection.match(/\*All throughput[\s\S]*?branch\.\*/);
	if (!footnoteMatch) {
		console.error('FAIL: could not locate table footnote');
		failed++;
	} else {
		var footnote = footnoteMatch[0];
		if (!/32 GiB|32G/.test(footnote)) {
			console.error('FAIL: footnote does not mention the 32G row');
			failed++;
		}
		if (!/64 GiB|64G/.test(footnote)) {
			console.error('FAIL: footnote does not mention the 64G row');
			failed++;
		}
		// Per-sentence lumping check: split on `.` and flag any sentence
		// that names BOTH row sizes AND a CAUSE-attribution phrase.
		// The cause words we look for are specifically the ones that
		// describe WHY a row is pending/limited (not protocol metadata
		// like "1 warmup + 1 measured" or "memory-wall probe", which
		// legitimately apply to multiple rows by design).
		var causeWords = /\b(exceeds|swap-bound|alloc pool is|input pool is|alloc pool fits|alloc pool =|alloc pool \+|alloc pool \(|input pool \(|pagefile auto-grew)\b/i;
		var sentences = footnote.split(/\.\s+/);
		var lumps = false;
		for (var si = 0; si < sentences.length; si++) {
			var sent = sentences[si];
			var has32 = /32 GiB|32G\/524288/.test(sent);
			var has64 = /64 GiB|64G\/1048576/.test(sent);
			var hasCause = causeWords.test(sent);
			if (has32 && has64 && hasCause) {
				console.error('FAIL: single sentence lumps 32G and 64G under a shared cause:');
				console.error('    "' + sent.trim() + '"');
				lumps = true;
				failed++;
				break;
			}
		}
	}

	if (failed > 0) {
		console.error('\nFAIL: ' + failed + ' throughput-shape feasibility contract violation(s)');
		process.exit(1);
	}
	console.log('\nPASS: all 32G/64G throughput row assertions satisfied; kernel labels correct, pagefile/memory-wall caveats present, footnote distinguishes the two shapes');
	process.exit(0);
})().catch(function(e) { console.error('FAIL: uncaught: ' + e.stack); process.exit(1); });
