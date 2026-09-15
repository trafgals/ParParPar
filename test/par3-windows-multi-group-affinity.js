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
var { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

if (!isMainThread) {
  if (workerData.role === 'staller') {
    // Intentionally stall without posting 'done' to exercise watchdog timeout contract
    setTimeout(function() {}, 60000);
    return;
  }
  // Worker thread executing concurrent cache reads or resets for section 5
  var gf64Worker = require(workerData.addonPath);
  var iters = workerData.iterations || 5000;
  if (workerData.role === 'reader') {
    for (var wi = 0; wi < iters; wi++) {
      var countRead = gf64Worker.get_effective_cpu_count();
      if (countRead < 1 || countRead > 128) {
        throw new Error('Worker observed invalid effective CPU count: ' + countRead);
      }
    }
  } else if (workerData.role === 'resetter') {
    for (var wj = 0; wj < iters; wj++) {
      gf64Worker.reset_effective_cpu_count_cache();
    }
  }
  parentPort.postMessage('done');
  return;
}

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
assert(headerContent.includes('multi-group') && headerContent.includes('Windows'),
  'par3_engine.h must document Windows multi-group support');
assert(headerContent.includes('CPU sets'),
  'par3_engine.h must document CPU sets support');
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
      assert.strictEqual(res.status, 0, 'taskset child process must exit with code 0: ' + (res.stderr || res.stdout));
      var count = parseInt((res.stdout || '').trim(), 10);
      console.log('   Linux taskset -c 0-1 effective CPUs:', count);
      assert.strictEqual(count, 2, 'Under taskset -c 0-1, effective CPU count must be exactly 2');
      console.log('   PASS: Linux taskset affinity restriction verified.');
    } else {
      console.log('   taskset not found on Linux; skipping child test.');
    }
  } else {
    console.log('   (Non-Windows/Linux platform, skipping OS-level affinity launch)');
  }
} finally {
  try { fs.unlinkSync(probeScript); } catch (_) {}
}

// ---------------------------------------------------------------------------
// 5. Concurrency / Thread-Safety Test with Watchdog (cubic review 5b1b83d5 P2, 0f33d3d0 P2, 2aa8af0f P3, 12274c8e P2)
// ---------------------------------------------------------------------------

function runConcurrentThreadSafetyTest(opts) {
  opts = opts || {};
  var timeoutMs = opts.timeoutMs || 15000;
  var numReaders = opts.testWatchdog ? 0 : 4;
  var numResetters = opts.testWatchdog ? 0 : 2;
  var numStallers = opts.testWatchdog ? 1 : 0;
  var iterations = opts.iterations || 5000;
  var totalWorkers = numReaders + numResetters + numStallers;
  var completedWorkers = 0;
  var workers = [];
  var settled = false;

  return new Promise(function(resolve, reject) {
    function terminateAllWorkers() {
      return Promise.all(workers.map(function(w) {
        try {
          return w.terminate().catch(function() {});
        } catch (_) {
          return Promise.resolve();
        }
      }));
    }

    function cleanupAndReject(err) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      terminateAllWorkers().then(function() {
        reject(err);
      });
    }

    function cleanupAndResolve() {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      terminateAllWorkers().then(function() {
        resolve();
      });
    }

    var timer = setTimeout(function() {
      cleanupAndReject(new Error('Watchdog timeout: worker threads did not complete within ' + timeoutMs + 'ms (cubic review 2aa8af0f P3 & 12274c8e P2)'));
    }, timeoutMs);

    function onWorkerMessage() {
      completedWorkers++;
      if (completedWorkers === totalWorkers) {
        cleanupAndResolve();
      }
    }

    function onWorkerError(err) {
      cleanupAndReject(err);
    }

    function onWorkerExit(code) {
      if (code !== 0 && !settled) {
        cleanupAndReject(new Error('Worker exited unexpectedly with exit code ' + code));
      }
    }

    for (var r = 0; r < numReaders; r++) {
      var wReader = new Worker(__filename, {
        workerData: { addonPath: ADDON_PATH, role: 'reader', iterations: iterations }
      });
      wReader.on('message', onWorkerMessage);
      wReader.on('error', onWorkerError);
      wReader.on('exit', onWorkerExit);
      workers.push(wReader);
    }

    for (var s = 0; s < numResetters; s++) {
      var wResetter = new Worker(__filename, {
        workerData: { addonPath: ADDON_PATH, role: 'resetter', iterations: iterations }
      });
      wResetter.on('message', onWorkerMessage);
      wResetter.on('error', onWorkerError);
      wResetter.on('exit', onWorkerExit);
      workers.push(wResetter);
    }

    for (var t = 0; t < numStallers; t++) {
      var wStaller = new Worker(__filename, {
        workerData: { addonPath: ADDON_PATH, role: 'staller' }
      });
      wStaller.on('error', onWorkerError);
      wStaller.on('exit', onWorkerExit);
      workers.push(wStaller);
    }
  });
}

console.log('5a. Verifying watchdog timeout and worker termination on stall (cubic review 2aa8af0f P3 & 12274c8e P2)...');
runConcurrentThreadSafetyTest({ testWatchdog: true, timeoutMs: 150 }).then(function() {
  assert.fail('Expected watchdog timeout on stalled worker');
}).catch(function(err) {
  assert(err.message.includes('Watchdog timeout'), 'Error must be watchdog timeout: ' + err.message);
  console.log('   PASS: Watchdog cleanly caught stall, awaited worker termination, and returned diagnostic.');

  console.log('5b. Verifying cache read / reset thread safety across concurrent worker threads...');
  return runConcurrentThreadSafetyTest({ timeoutMs: 15000 });
}).then(function() {
  console.log('   PASS: Cache read / reset thread safety verified across 6 concurrent worker threads (20,000 reads, 10,000 resets).');
  console.log('\nAll Windows multi-group affinity and effective CPU count tests passed!\n');
}).catch(function(err) {
  console.error('Thread safety test failed:', err);
  process.exit(1);
});

