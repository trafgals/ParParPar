"use strict";
/* Contract test for the 32G/64G throughput rows.
 *
 * PR #108 (streaming & bounded-memory PAR3 create) broke the 64 GiB memory
 * wall — the chunked create path caps the input pool at 64 MiB and resets
 * between batches, so peak RSS stays bounded (≈280 MiB on this host)
 * regardless of source size.
 *
 * PR #109 dropped the Notes column from the throughput table (replaced by
 * the Peak RSS badge column). The kernel / chunked-path / peak-RSS contracts
 * previously pinned in the row's Notes cell now live in each badge's
 * `detail` string on `feat/ci-benchmark-badge` (and the README's Peak RSS
 * badge cell resolves to that JSON).
 *
 * Per-row contract for the 32G and 64G rows (asserted on the badge detail):
 *   1. Detail names the kernel used (Barycentric — R=8 < FENGER_MIN_R so the
 *      Fenger fast path does not engage).
 *   2. Detail mentions "peak RSS" with the measured value.
 *   3. Detail mentions the chunked create path (the PR #108 enabler).
 *   4. Detail does NOT claim "swap-bound" (chunked create bounds memory
 *      at ~280 MiB; the pre-#108 swap-bound framing is obsolete).
 *
 * cubic P2 (PR #101): if any badge fetch returns a network/HTTP/parse
 * error, the test fails immediately. Otherwise a fetch failure could
 * leave us with zero rows and the vacuous-truth path would silently skip
 * the per-row checks — a false PASS on broken CI.
 *
 * Run: `node test/readme-throughput-shape-feasibility.js`
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
	var line = lines[i].trim();
	if (!line.startsWith('|')) continue;
	if (/^\|\s*Project \/ Format/i.test(line)) continue;
	if (/^\|\s*:?---/.test(line)) continue;
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

// Fetch sources.json from the badge branch. The endpoint JSONs only carry
// schemaVersion/label/message/color; the `detail` text we want to assert
// against lives in sources.json (the source of truth for badge content).
var sourcesPromise = fetchJson(BADGE_BRANCH_RAW + 'benchmarks/badges/sources.json');
sourcesPromise.then(function(s) { trackFetch('sources.json', s); });

(async function() {
	// 1. Locate the 32G and 64G Zen4 throughput rows in the throughput table.
	var rows32g = [];
	var rows64g = [];
	for (var i = 0; i < dataRows.length; i++) {
		var cells = splitRow(dataRows[i]);
		if (cells.length < 7) continue;
		var m = zen4IdRe.exec(cells[4]);
		if (!m) continue;
		var badgeId = m[1];
		if (/par3-32g-524288/.test(badgeId)) rows32g.push({ badgeId: badgeId });
		if (/par3-64g-1048576/.test(badgeId)) rows64g.push({ badgeId: badgeId });
	}

	// 2. Fetch sources.json so we can assert on each badge's `detail`.
	var srcR = await sourcesPromise;
	if (fetchFailures.length > 0) {
		console.error('FAIL: ' + fetchFailures.length + ' badge fetch/parse failure(s)');
		process.exit(1);
	}
	var detailById = {};
	if (srcR.body && Array.isArray(srcR.body.badges)) {
		srcR.body.badges.forEach(function(b) {
			if (b.id) detailById[b.id] = b.detail || '';
		});
	}

	console.log('Found ' + rows32g.length + ' 32G row(s) and ' + rows64g.length + ' 64G row(s)');

	// Contract precondition: the test must FAIL if either row is missing
	// from the README. Without this guard, the per-row assertions below
	// would silently run zero times and the test would exit 0 even if
	// both rows were removed from the table.
	if (rows32g.length < 1) {
		console.error('FAIL: no 32G/524288 row found in the README throughput table (the contract has no rows to check)');
		process.exit(1);
	}
	if (rows64g.length < 1) {
		console.error('FAIL: no 64G/1048576 row found in the README throughput table (the contract has no rows to check)');
		process.exit(1);
	}

	var failed = 0;

	// 3. Assert on the badge detail string.
	function checkDetail(badgeId, label) {
		var detail = detailById[badgeId];
		if (typeof detail !== 'string') {
			console.error('FAIL: ' + label + ' badge ' + badgeId + ' has no detail string in sources.json');
			failed++;
			return;
		}
		var reBary = /\b(Barycentric|barycentric|cost[- ]model)\b/i;
		var reRss = /\bpeak\s+RSS\b/i;
		var reChunked = /\bchunked\b/i;
		// cubic P3 on PR #109 round 2: add /i for case-insensitive
		// matching so "fenger fast-path" (lowercase) doesn't slip past.
		var reNoFenger = /\bFenger fast-path\b/i;
		var reNoSwapBound = /\bswap-bound\b/i;

		if (!reBary.test(detail)) {
			console.error('FAIL: ' + label + ' detail missing kernel label (Barycentric / cost-model dispatch): "' + detail + '"');
			failed++;
		}
		if (!reRss.test(detail)) {
			console.error('FAIL: ' + label + ' detail missing peak RSS: "' + detail + '"');
			failed++;
		}
		if (!reChunked.test(detail)) {
			console.error('FAIL: ' + label + ' detail missing chunked create path: "' + detail + '"');
			failed++;
		}
		if (reNoFenger.test(detail)) {
			console.error('FAIL: ' + label + ' detail incorrectly contains "Fenger fast-path" (Fenger is the wrong kernel at R=8 < FENGER_MIN_R=8192): "' + detail + '"');
			failed++;
		}
		if (reNoSwapBound.test(detail)) {
			console.error('FAIL: ' + label + ' detail incorrectly contains "swap-bound" (chunked create path bounds memory at ~280 MiB): "' + detail + '"');
			failed++;
		}
	}

	rows32g.forEach(function(row) { checkDetail(row.badgeId, '32G'); });
	rows64g.forEach(function(row) { checkDetail(row.badgeId, '64G'); });

	// 4. Footnote assertions: post-#109 the table has no Footnotes section
	// (the Notes column was dropped), so this rule is a no-op. Kept for
	// documentation — if a future PR adds footnotes back, the lumping
	// check should re-engage.
	if (failed > 0) {
		console.error('\nFAIL: ' + failed + ' throughput-shape feasibility contract violation(s)');
		process.exit(1);
	}
	console.log('\nPASS: all 32G/64G throughput row assertions satisfied; kernel labels correct, peak RSS + chunked path mentions present, no swap-bound / Fenger fast-path claims');
	process.exit(0);
})().catch(function(e) { console.error('FAIL: uncaught: ' + e.stack); process.exit(1); });
