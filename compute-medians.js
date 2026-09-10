#!/usr/bin/env node
// Compute median, mean, stdev of the 3 runs per shape, plus RSS
// median (in MiB). Output as JSON for easy machine-parseable updates
// to sources.json.

var fs = require('fs');
var path = require('path');

var dir = require('os').tmpdir() + '/readme-bench-results';
var files = fs.readdirSync(dir).filter(f => f.endsWith('.txt'));

function median(xs) {
  var s = xs.slice().sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
}
function mean(xs) { return xs.reduce((a, b) => a + b, 0) / xs.length; }
function stdev(xs) {
  var m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
}
function rssBytes(s) {
  // parse "1.23 GiB" or "262.78 MiB"
  var m = s.match(/([\d.]+)\s*(MiB|GiB|KiB|B)/);
  if (!m) return 0;
  var n = parseFloat(m[1]), unit = m[2];
  if (unit === 'GiB') return Math.round(n * 1024 * 1024 * 1024);
  if (unit === 'MiB') return Math.round(n * 1024 * 1024);
  if (unit === 'KiB') return Math.round(n * 1024);
  return Math.round(n);
}

var rows = {};
for (var f of files) {
  var name = f.replace(/\.txt$/, '');
  var lines = fs.readFileSync(path.join(dir, f), 'utf8').trim().split('\n');
  var mbps = [], rss = [];
  for (var l of lines) {
    // "1 664.5 2.37 GiB" -> split on whitespace and reassemble the unit
    var parts = l.trim().split(/\s+/);
    var idx = parseInt(parts[0]);
    var m = parseFloat(parts[1]);
    var unitStr = parts.slice(2).join(' ');
    rss.push(rssBytes(unitStr));
    mbps.push(m);
  }
  rows[name] = {
    mbps_runs: mbps,
    rss_runs: rss,
    mbps_median: Math.round(median(mbps) * 100) / 100,
    mbps_mean: Math.round(mean(mbps) * 100) / 100,
    mbps_stdev: Math.round(stdev(mbps) * 100) / 100,
    rss_median_bytes: median(rss),
    rss_median_mib: Math.round(median(rss) / (1024 * 1024) * 100) / 100
  };
}

console.log(JSON.stringify(rows, null, 2));
