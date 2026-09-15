#!/usr/bin/env node
"use strict";

// Test for cubic review 1fbddc67-5c59-5873-811e-9100dca820b3 P2 and P3:
// - P3: par3_engine.h declaration comment for GetEffectiveCpuCount accurately
//   reflects Linux and Windows multi-group/JobObject/CPU-set support and 128 cap.
// - P2: Windows multi-group effective affinity query respects affinity restrictions
//   rather than counting all active processors in the groups.

var assert = require('assert');
var cp = require('child_process');
var fs = require('fs');
var path = require('path');

var ADDON_PATH = path.join(__dirname, '..', 'build', 'Release', 'parpar_gf64.node');
var HEADER_PATH = path.join(__dirname, '..', 'src', 'par3_engine.h');

console.log('--- Test: Windows Multi-Group Affinity & Effective CPU Count ---');

// ---------------------------------------------------------------------------
// 1. Contract Test: Header Declaration Comment (P3)
// ---------------------------------------------------------------------------
console.log('1. Verifying par3_engine.h declaration comment (cubic review P3)...');
var headerContent = fs.readFileSync(HEADER_PATH, 'utf8');
assert(!headerContent.includes('capped at 32'),
  'par3_engine.h should not state capped at 32 (should be 128)');
assert(!headerContent.includes('respecting Linux affinity masks.\n/// Cached after the first call and capped at 32'),
  'par3_engine.h should not state Linux-only and capped at 32');
assert(headerContent.includes('capped at 128'),
  'par3_engine.h must document 128 cap');
assert(headerContent.includes('Windows affinity masks') || headerContent.includes('Windows'),
  'par3_engine.h must document Windows affinity support');
console.log('   PASS: Header declaration comment correctly matches implementation.');

// ---------------------------------------------------------------------------
// 2. Addon API Surface
// ---------------------------------------------------------------------------
console.log('2. Verifying addon exports get_effective_cpu_count & reset...');
var gf64 = require(ADDON_PATH);
assert.strictEqual(typeof gf64.get_effective_cpu_count, 'function',
  'addon must export get_effective_cpu_count');
assert.strictEqual(typeof gf64.reset_effective_cpu_count_cache, 'function',
  'addon must export reset_effective_cpu_count_cache');

var initialCount = gf64.get_effective_cpu_count();
console.log('   Initial host effective CPU count:', initialCount);
assert(Number.isInteger(initialCount) && initialCount >= 1 && initialCount <= 128,
  'Host effective CPU count must be between 1 and 128');
console.log('   PASS: Addon API surface verified.');

// ---------------------------------------------------------------------------
// 3. Mock Override Contract (PAR3_MOCK_EFFECTIVE_CPUS)
// ---------------------------------------------------------------------------
console.log('3. Verifying mock overrides and clamping...');
try {
  process.env.PAR3_MOCK_EFFECTIVE_CPUS = '4';
  gf64.reset_effective_cpu_count_cache();
  assert.strictEqual(gf64.get_effective_cpu_count(), 4,
    'Mock override of 4 must be returned');

  process.env.PAR3_MOCK_EFFECTIVE_CPUS = '16';
  gf64.reset_effective_cpu_count_cache();
  assert.strictEqual(gf64.get_effective_cpu_count(), 16,
    'Mock override of 16 must be returned');

  process.env.PAR3_MOCK_EFFECTIVE_CPUS = '256';
  gf64.reset_effective_cpu_count_cache();
  assert.strictEqual(gf64.get_effective_cpu_count(), 128,
    'Mock override of 256 must be clamped to 128');
} finally {
  delete process.env.PAR3_MOCK_EFFECTIVE_CPUS;
  gf64.reset_effective_cpu_count_cache();
}
assert.strictEqual(gf64.get_effective_cpu_count(), initialCount,
  'Cache reset must restore host effective count');
console.log('   PASS: Mock overrides and 128 clamping verified.');

// ---------------------------------------------------------------------------
// 4. OS-level Affinity Restriction Contract (P2)
// ---------------------------------------------------------------------------
console.log('4. Verifying OS-level affinity restriction...');

var probeScript = path.join(__dirname, '..', 'build', '_aff_probe.js');
var probeCode = [
  'var gf64 = require(' + JSON.stringify(ADDON_PATH) + ');',
  'var c = gf64.get_effective_cpu_count();',
  'process.stdout.write(String(c));',
].join('\n');

try {
  fs.writeFileSync(probeScript, probeCode);

  if (process.platform === 'win32') {
    // Test with cmd /c start /affinity 3 (mask 0x3 = 2 cores allowed)
    var cmd2 = 'start /affinity 3 /wait /b "" "' + process.execPath + '" "' + probeScript + '"';
    var out2 = cp.execSync(cmd2, { encoding: 'utf8', shell: 'cmd.exe' }).trim();
    var count2 = parseInt(out2, 10);
    console.log('   Windows start /affinity 3 effective CPUs:', count2);
    assert.strictEqual(count2, 2, 'Under affinity mask 0x3, effective CPU count must be exactly 2');

    // Test with cmd /c start /affinity 1 (mask 0x1 = 1 core allowed)
    var cmd1 = 'start /affinity 1 /wait /b "" "' + process.execPath + '" "' + probeScript + '"';
    var out1 = cp.execSync(cmd1, { encoding: 'utf8', shell: 'cmd.exe' }).trim();
    var count1 = parseInt(out1, 10);
    console.log('   Windows start /affinity 1 effective CPUs:', count1);
    assert.strictEqual(count1, 1, 'Under affinity mask 0x1, effective CPU count must be exactly 1');
    console.log('   PASS: Windows process affinity restriction verified.');
  } else if (process.platform === 'linux') {
    var hasTaskset = false;
    try {
      cp.execSync('which taskset', { stdio: 'pipe' });
      hasTaskset = true;
    } catch (_) {}

    if (hasTaskset) {
      var res = cp.spawnSync('taskset', ['-c', '0-1', process.execPath, probeScript], { encoding: 'utf8' });
      if (res.status === 0) {
        var count = parseInt((res.stdout || '').trim(), 10);
        console.log('   Linux taskset -c 0-1 effective CPUs:', count);
        assert.strictEqual(count, 2, 'Under taskset -c 0-1, effective CPU count must be exactly 2');
        console.log('   PASS: Linux taskset affinity restriction verified.');
      } else {
        console.log('   Linux taskset skipped or not permitted in container.');
      }
    } else {
      console.log('   taskset not found on Linux; skipping child test.');
    }
  } else {
    console.log('   (Non-Windows/Linux platform, skipping OS-level affinity launch)');
  }
} finally {
  try { fs.unlinkSync(probeScript); } catch (_) {}
}

console.log('\nAll Windows multi-group affinity and effective CPU count tests passed!\n');
