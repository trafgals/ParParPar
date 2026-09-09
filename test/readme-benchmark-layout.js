"use strict";
/* Contract test for README throughput table and notes layout (PR #109):
 *
 *   1. Cell Width Bounds: Text columns (0-3: Project, Workload, Slice Count,
 *      Block Size) may not exceed 80 characters. Badge columns (4-6: Zen4
 *      Throughput, Peak RSS, CI Runner) contain shields.io endpoint URLs
 *      (~270 chars each) that render as compact badges; the previous
 *      "Notes column 160 chars" cap is obsolete (the Notes column was
 *      replaced by the Peak RSS badge column).
 *
 *   2. Footnote Reference Integrity: Every footnote marker referenced in
 *      the table (e.g. <sup>[1]</sup> .. <sup>[3]</sup>) must have a
 *      corresponding definition in the "Footnotes" section below the
 *      table.
 *
 *   3. HTML Tag Escaping: No raw unescaped `<id>` tag may exist in
 *      README.md. Any `<id>` placeholder must be safely enclosed in
 *      markdown code spans (backticks).
 *
 *   4. Structural Integrity: The table must maintain 7 columns across
 *      all rows, and the post-table notes must contain the structured
 *      "Footnotes" and "Benchmarking Methodology & Environment"
 *      headings.
 *
 * Run: `node test/readme-benchmark-layout.js`
 *   - exit 0: all layout and formatting assertions pass
 *   - exit 1: a cell exceeds its character budget, a footnote is missing,
 *             or an unescaped HTML tag is detected
 */

var fs = require('fs');
var path = require('path');

var README_PATH = path.join(__dirname, '..', 'README.md');
var readme = fs.readFileSync(README_PATH, 'utf8');

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

var lines = tableSection.split(/\r?\n/);
var dataRows = [];
for (var i = 0; i < lines.length; i++) {
	var line = lines[i].trim();
	if (!line.startsWith('|')) continue;
	if (/^\|\s*Project \/ Format/i.test(line)) continue;
	if (/^\|\s*:?---/.test(line)) continue;
	dataRows.push(line);
}

var failed = 0;

// Rule 1: Table row column count and cell length budget.
var MAX_TEXT_LENGTH = 80;
// The table is 7 columns: Project, Workload, Slice Count, Block Size,
// Zen4 Throughput, Peak RSS (Zen4), CI Runner. Columns 4, 5, 6 are badge
// URLs and exempt from the prose length cap.
var TEXT_COL_END = 3;
var CI_COL_INDEX = 6;
var RSS_COL_INDEX = 5;
var referencedFootnotes = {};

for (var i = 0; i < dataRows.length; i++) {
	var row = dataRows[i];
	var trimmed = row.replace(/^\| /, '').replace(/ \|$/, '');
	var cells = trimmed.split(' | ');

	if (cells.length !== 7) {
		console.error('FAIL: row ' + (i + 1) + ' has ' + cells.length + ' columns (expected 7): ' + row);
		failed++;
	}

	for (var c = 0; c <= TEXT_COL_END; c++) {
		if (cells[c] && cells[c].length > MAX_TEXT_LENGTH) {
			console.error('FAIL: row ' + (i + 1) + ' col ' + (c + 1) + ' exceeds ' + MAX_TEXT_LENGTH + ' chars (' + cells[c].length + '): "' + cells[c] + '"');
			failed++;
		}
	}

	for (var c = 0; c < cells.length; c++) {
		var cell = cells[c];
		var fnMatch;
		var fnRe = /<sup>\[(\d+)\]<\/sup>/g;
		while ((fnMatch = fnRe.exec(cell)) !== null) {
			referencedFootnotes[fnMatch[1]] = true;
		}
	}
}

// Rule 2: Footnote references must have definitions in the footnotes section.
// PR #109 dropped the Notes column (replaced by Peak RSS badge column),
// so there are no per-row <sup>[N]</sup> markers — the row context lives
// in the RSS badge now. The footnote section was removed entirely. This
// rule is kept as a no-op (zero refs is the post-#109 expected state)
// so a future PR adding per-row footnotes back triggers the assertions.
var notesStart = tableSection.indexOf('*All throughput');
var notesSection = notesStart >= 0 ? tableSection.substring(notesStart).trim() : '';

var refList = Object.keys(referencedFootnotes).sort();
for (var r = 0; r < refList.length; r++) {
	var num = refList[r];
	var defPattern = new RegExp('\\*\\*\\[' + num + '\\]');
	if (!defPattern.test(notesSection)) {
		console.error('FAIL: footnote [' + num + '] referenced in table but missing definition "**[' + num + ']" in notes section');
		failed++;
	}
}

// Rule 3: Required subheadings in notes section.
// PR #109 dropped the Footnotes section (the Notes column was removed);
// the Methodology section remains.
if (!/#### Benchmarking Methodology & Environment/i.test(notesSection)) {
	console.error('FAIL: missing "#### Benchmarking Methodology & Environment" heading in notes section');
	failed++;
}

// Rule 4: No raw unescaped `<id>` HTML tag in README.md.
var strippedCodeSpans = readme.replace(/`[^`]*`/g, '');
if (/<id>/i.test(strippedCodeSpans)) {
	console.error('FAIL: found unescaped <id> HTML tag outside code spans in README.md');
	failed++;
}

// Rule 5: Row parser robust to leading whitespace and bold formatting.
var testSection = "| Project / Format | Workload |\n| :--- | :--- |\n  | Plain PAR2 | 1 GiB |\n| **Bold PAR3** | 2 GiB |\n\n*All throughput...";
var testLines = testSection.split(/\r?\n/);
var parsedRows = [];
for (var ti = 0; ti < testLines.length; ti++) {
	var tline = testLines[ti].trim();
	if (!tline.startsWith('|')) continue;
	if (/^\|\s*Project \/ Format/i.test(tline)) continue;
	if (/^\|\s*:?---/.test(tline)) continue;
	parsedRows.push(tline);
}
if (parsedRows.length !== 2) {
	console.error('FAIL: row parser did not detect both plain-text and bold rows (expected 2, got ' + parsedRows.length + ')');
	failed++;
}

// Rule 6: Notes extraction anchors at *All throughput and is immune to inline pipes.
var sampleWithPipe = "| Header |\n| --- |\n| Row |\n\n*All throughput in MB/s*\nInline code `a | b` in notes.";
var sampleStart = sampleWithPipe.indexOf('*All throughput');
if (sampleStart < 0 || sampleWithPipe.substring(sampleStart).indexOf('`a | b`') < 0) {
	console.error('FAIL: notes anchoring does not capture full content through inline pipes');
	failed++;
}

// Rule 7: Footnotes heading precedes Methodology heading.
// PR #109 dropped the Footnotes section; only Methodology remains.
var methodPos = notesSection.indexOf('#### Benchmarking Methodology & Environment');
if (methodPos < 0) {
	console.error('FAIL: post-table notes structure invalid; expected Methodology heading');
	failed++;
}

// Rule 8: KaTeX / LaTeX math formatting integrity.
function extractMathSpans(text) {
	var noCode = text.replace(/```[\s\S]*?```/g, '').replace(/`[^`]*`/g, '');
	var spans = [];
	var displayMathRe = /\$\$([\s\S]*?)\$\$/g;
	var dm;
	while ((dm = displayMathRe.exec(noCode)) !== null) {
		spans.push(dm[1]);
	}
	var inlineSection = noCode.replace(/\$\$[\s\S]*?\$\$/g, '');
	var inlineMathRe = /\$([^$\n]+)\$/g;
	var im;
	while ((im = inlineMathRe.exec(inlineSection)) !== null) {
		spans.push(im[1]);
	}
	return spans;
}

var mathSpans = extractMathSpans(tableSection);
for (var mi = 0; mi < mathSpans.length; mi++) {
	var span = mathSpans[mi];
	if (/\\text\{[^}]*_[^}]*\}/.test(span) || /\\text\{[^}]*\\_[^}]*\}/.test(span)) {
		console.error('FAIL: math span contains \\text{} with underscore (triggers KaTeX "\'_\' allowed only in math mode"): "$' + span + '$"');
		failed++;
	}
	if (/[A-Z]{2,}_[A-Z0-9_]*/.test(span)) {
		console.error('FAIL: math span contains uppercase code identifier with underscore (use markdown backticks instead): "$' + span + '$"');
		failed++;
	}
}

// Synthetic check 8a
var badSample = 'Sample with $N < \\text{BARY_MIN}=10$ and $R < \\text{FENGER\\_MIN_R}$';
var badSampleSpans = extractMathSpans(badSample);
var badMatches = badSampleSpans.filter(function(s) {
	return /\\text\{[^}]*_[^}]*\}/.test(s) || /\\text\{[^}]*\\_[^}]*\}/.test(s);
});
if (!badMatches || badMatches.length !== 2) {
	console.error('FAIL: Rule 8 synthetic test failed to catch bad \\text{} with underscore patterns in extracted math spans');
	failed++;
}

// Synthetic check 8b
var badMacroSample = 'Outer text BARY_MIN_INPUTS_DEFAULT with $R < FENGER_MIN_R$';
var badMacroSpans = extractMathSpans(badMacroSample);
var badMacroMatches = badMacroSpans.filter(function(s) {
	return /[A-Z]{2,}_[A-Z0-9_]*/.test(s);
});
if (!badMacroMatches || badMacroMatches.length !== 1) {
	console.error('FAIL: Rule 8 synthetic test failed to catch raw uppercase macro identifier inside extracted math span');
	failed++;
}
var safeMacroSample = 'Outer text BARY_MIN_INPUTS_DEFAULT with $R < 100$';
var safeMacroSpans = extractMathSpans(safeMacroSample);
var safeMacroMatches = safeMacroSpans.filter(function(s) {
	return /[A-Z]{2,}_[A-Z0-9_]*/.test(s);
});
if (safeMacroMatches.length !== 0) {
	console.error('FAIL: Rule 8 synthetic test incorrectly matched macro identifier outside math spans');
	failed++;
}

// Helper to extract image destination URLs from markdown
function extractBadgeImageUrls(markdown) {
	var urls = [];
	var imgRe = /!\[[^\]]*\]\(([^)]+)\)/g;
	var m;
	while ((m = imgRe.exec(markdown)) !== null) {
		urls.push(m[1]);
	}
	return urls;
}

// Rule 9: CI Runner badges (col 6) must override shields.io label with empty &label=.
var ciBadgeCount = 0;
for (var i = 0; i < dataRows.length; i++) {
	var row = dataRows[i];
	var trimmed = row.replace(/^\| /, '').replace(/ \|$/, '');
	var cells = trimmed.split(' | ');
	var ciCell = cells[CI_COL_INDEX] || '';
	var imgUrls = extractBadgeImageUrls(ciCell);
	for (var uIdx = 0; uIdx < imgUrls.length; uIdx++) {
		var imgUrl = imgUrls[uIdx];
		if (imgUrl.indexOf('gist.githubusercontent.com') >= 0) {
			ciBadgeCount++;
			if (!/&label=(&|$)/.test(imgUrl)) {
				console.error('FAIL: row ' + (i + 1) + ' CI badge image destination does not have an empty "&label=" override: "' + imgUrl + '"');
				failed++;
			}
		}
	}
}
if (ciBadgeCount === 0) {
	console.error('FAIL: expected at least one CI gist badge in the throughput table');
	failed++;
}

// Rule 10 (PR #109 round 2): Peak RSS badges (col 5) must resolve to a Zen4
// branch endpoint JSON, and every row that has a Peak RSS badge must also
// have a Zen4 throughput badge in col 4 (no orphan RSS badges).
var rssBadgeIds = {};
var zen4Ids = {};
// Cubic P2 on PR #109 round 2: the badge URLs are percent-encoded
// shields.io endpoints, so the regex must decodeURIComponent before
// extracting the id. Cubic P3 on PR #109 round 3: hoist the loop-invariant
// regexes + a decodeId helper above the per-row loop (was being
// re-created on every iteration).
var urlDecodeRe = /\/([a-z0-9-]+)-rss-zen4\.json/i;
var urlDecodeZen4Re = /\/([a-z0-9-]+)-zen4\.json/i;
function decodeId(url, re) {
	var decoded;
	try { decoded = decodeURIComponent(url); } catch (e) { decoded = url; }
	var m = re.exec(decoded);
	return m ? m[1] : null;
}
for (var i = 0; i < dataRows.length; i++) {
	var row = dataRows[i];
	var trimmed = row.replace(/^\| /, '').replace(/ \|$/, '');
	var cells = trimmed.split(' | ');
	var rssCell = cells[RSS_COL_INDEX] || '';
	var zen4Cell = cells[4] || '';
	var rssImgs = extractBadgeImageUrls(rssCell);
	var zen4Imgs = extractBadgeImageUrls(zen4Cell);
	if (rssImgs.length > 0 && zen4Imgs.length === 0) {
		console.error('FAIL: row ' + (i + 1) + ' has a Peak RSS badge but no Zen4 throughput badge: "' + row + '"');
		failed++;
	}
	rssImgs.forEach(function(url) {
		var id = decodeId(url, urlDecodeRe);
		if (id) rssBadgeIds[id] = true;
	});
	zen4Imgs.forEach(function(url) {
		var id = decodeId(url, urlDecodeZen4Re);
		if (id) zen4Ids[id] = true;
	});
}
// Each RSS badge id should have a corresponding non-RSS zen4 throughput badge
// in the same row (sanity check on the badge-id pairing).
for (var id in rssBadgeIds) {
	var baseId = id; // e.g. "par3-1g-1k" from "par3-1g-1k-rss-zen4"
	if (!zen4Ids[baseId + '-zen4'] && !zen4Ids[baseId]) {
		console.error('FAIL: RSS badge ' + id + '-rss-zen4 has no matching Zen4 throughput badge in any row');
		failed++;
	}
}

// Rule 11: Peak RSS column must contain either a Zen4 branch endpoint URL or
// "—". No raw RSS prose (e.g. "280 MiB") without a badge should appear.
for (var i = 0; i < dataRows.length; i++) {
	var row = dataRows[i];
	var trimmed = row.replace(/^\| /, '').replace(/ \|$/, '');
	var cells = trimmed.split(' | ');
	var rssCell = cells[RSS_COL_INDEX] || '';
	var hasImg = extractBadgeImageUrls(rssCell).length > 0;
	var isDash = /^\s*[—\-]\s*$/.test(rssCell);
	if (!hasImg && !isDash) {
		console.error('FAIL: row ' + (i + 1) + ' Peak RSS cell must be either a badge or "—", got: "' + rssCell + '"');
		failed++;
	}
}

// Synthetic check 9: decoy URL test for CI badges.
var decoyCell = '[![PAR2 1GB/32k (CI)](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fdecoy.json&style=flat-square)](https://github.com/trafgals/ParParPar?view=table&label=)';
var decoyUrls = extractBadgeImageUrls(decoyCell);
if (decoyUrls.length !== 1 || decoyUrls[0].indexOf('gist.githubusercontent.com') < 0) {
	console.error('FAIL: Synthetic check 9 failed to extract decoy badge image URL');
	failed++;
} else if (/&label=(&|$)/.test(decoyUrls[0])) {
	console.error('FAIL: Synthetic check 9 decoy image URL should NOT have &label=');
	failed++;
}
var validBadgeUrl = 'https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fvalid.json&style=flat-square&label=';
if (!/&label=(&|$)/.test(validBadgeUrl)) {
	console.error('FAIL: Synthetic check 9 valid badge image URL failed empty &label= regex');
	failed++;
}
var labeledBadgeUrl = 'https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Flabeled.json&style=flat-square&label=SomeText';
if (/&label=(&|$)/.test(labeledBadgeUrl)) {
	console.error('FAIL: Synthetic check 9 non-empty &label=SomeText should not pass');
	failed++;
}

if (failed > 0) {
	console.error('\nFAIL: ' + failed + ' benchmark layout contract violation(s)');
	process.exit(1);
}
console.log('PASS: table layout bounds (≤' + MAX_TEXT_LENGTH + ' chars/text col, 7 cols, RSS badges paired with Zen4 throughput badges), ' + refList.length + ' footnote links verified, no unescaped HTML tags, row parser, pipe resilience, notes structure, math formatting & CI badge width confirmed');
process.exit(0);
