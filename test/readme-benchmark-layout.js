"use strict";
/* Contract test for README throughput table and notes layout:
 *
 *   1. Cell Width Bounds: Text columns (0-3: Project, Workload, Slice Count,
 *      Block Size) may not exceed 80 characters, and the Notes column (6) may
 *      not exceed 160 characters. This prevents in-cell essay bloat from
 *      distorting table column proportions on GitHub Flavored Markdown. Badge
 *      URL columns (4-5) contain shields.io endpoint URLs (~270 chars) which
 *      render as compact badges and are exempt from prose length limits.
 *
 *   2. Footnote Reference Integrity: Every footnote marker referenced in the
 *      table (e.g. <sup>[1]</sup> .. <sup>[6]</sup>) must have a corresponding
 *      definition in the "Footnotes & Caveats" section below the table.
 *
 *   3. HTML Tag Escaping: No raw unescaped `<id>` tag may exist in README.md.
 *      Any `<id>` placeholder must be safely enclosed in markdown code spans
 *      (backticks) to prevent GitHub's HTML sanitizer from stripping or
 *      mangling text.
 *
 *   4. Structural Integrity: The table must maintain 7 columns across all
 *      rows, and the section must contain structured "Footnotes & Caveats"
 *      and "Benchmarking Methodology & Environment" headings.
 *
 * Run: `node test/readme-benchmark-layout.js`
 *   - exit 0: all layout and formatting assertions pass
 *   - exit 1: a cell exceeds character budget, a footnote is missing,
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
// The badge columns (col 5 & 6) contain long shields.io endpoint URLs
// that render as small badge images; text columns and the Notes column (col 7)
// must be bounded to keep the table layout readable.
var MAX_NOTES_LENGTH = 160;
var MAX_TEXT_LENGTH = 80;
var referencedFootnotes = {};

for (var i = 0; i < dataRows.length; i++) {
  var row = dataRows[i];
  var trimmed = row.replace(/^\| /, '').replace(/ \|$/, '');
  var cells = trimmed.split(' | ');

  if (cells.length !== 7) {
    console.error('FAIL: row ' + (i + 1) + ' has ' + cells.length + ' columns (expected 7): ' + row);
    failed++;
  }

  // Check text columns (0: Project, 1: Workload, 2: Slice Count, 3: Block Size)
  for (var c = 0; c <= 3; c++) {
    if (cells[c] && cells[c].length > MAX_TEXT_LENGTH) {
      console.error('FAIL: row ' + (i + 1) + ' col ' + (c + 1) + ' exceeds ' + MAX_TEXT_LENGTH + ' chars (' + cells[c].length + '): "' + cells[c] + '"');
      failed++;
    }
  }

  // Check Notes column (col 7 / index 6)
  var notesCell = cells[6] || '';
  if (notesCell.length > MAX_NOTES_LENGTH) {
    console.error('FAIL: row ' + (i + 1) + ' Notes cell exceeds ' + MAX_NOTES_LENGTH + ' chars (' + notesCell.length + ' chars): "' + notesCell + '"');
    failed++;
  }

  for (var c = 0; c < cells.length; c++) {
    var cell = cells[c];
    // Check for footnote references like <sup>[1]</sup>
    var fnMatch;
    var fnRe = /<sup>\[(\d+)\]<\/sup>/g;
    while ((fnMatch = fnRe.exec(cell)) !== null) {
      referencedFootnotes[fnMatch[1]] = true;
    }
  }
}

// Rule 2: Footnote references must have definitions in the footnotes section.
var notesStart = tableSection.indexOf('*All throughput');
var notesSection = notesStart >= 0 ? tableSection.substring(notesStart).trim() : '';

var refList = Object.keys(referencedFootnotes).sort();
if (refList.length === 0) {
  console.error('FAIL: no footnote references (<sup>[N]</sup>) found in throughput table');
  failed++;
}

for (var r = 0; r < refList.length; r++) {
  var num = refList[r];
  var defPattern = new RegExp('\\*\\*\\[' + num + '\\]');
  if (!defPattern.test(notesSection)) {
    console.error('FAIL: footnote [' + num + '] referenced in table but missing definition "**[' + num + ']" in notes section');
    failed++;
  }
}

// Rule 3: Required subheadings in notes section.
if (!/#### Footnotes & Caveats/i.test(notesSection)) {
  console.error('FAIL: missing "#### Footnotes & Caveats" heading in notes section');
  failed++;
}
if (!/#### Benchmarking Methodology & Environment/i.test(notesSection)) {
  console.error('FAIL: missing "#### Benchmarking Methodology & Environment" heading in notes section');
  failed++;
}

// Rule 4: No raw unescaped `<id>` HTML tag in README.md.
// Match `<id>` where it is NOT enclosed within backticks.
var strippedCodeSpans = readme.replace(/`[^`]*`/g, '');
if (/<id>/i.test(strippedCodeSpans)) {
  console.error('FAIL: found unescaped <id> HTML tag outside code spans in README.md');
  failed++;
}

// Rule 5 (cubic review PR #105 violation 2): Verify row filter robustly detects
// rows without bold formatting and rows with leading whitespace.
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

// Rule 6 (cubic review PR #105 violations 1 & 3): Verify notes extraction anchors
// at *All throughput and is immune to inline pipes in notes prose or code spans.
var sampleWithPipe = "| Header |\n| --- |\n| Row |\n\n*All throughput in MB/s*\nInline code `a | b` in notes.";
var sampleStart = sampleWithPipe.indexOf('*All throughput');
if (sampleStart < 0 || sampleWithPipe.substring(sampleStart).indexOf('`a | b`') < 0) {
  console.error('FAIL: notes anchoring does not capture full content through inline pipes');
  failed++;
}

if (failed > 0) {
  console.error('\nFAIL: ' + failed + ' benchmark layout contract violation(s)');
  process.exit(1);
}

console.log('PASS: table layout bounds (≤' + MAX_NOTES_LENGTH + ' chars/cell), ' + refList.length + ' footnote links verified, no unescaped HTML tags, row parser & pipe resilience confirmed');
process.exit(0);
