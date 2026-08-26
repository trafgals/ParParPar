"use strict";

var path = require('path');
var fs = require('fs');
var crypto = require('crypto');
var par3 = require('../lib/par3gen.js');
var helpers = require('./e2e/helpers');

var BLOCK_SIZE = 1024 * 1024;
var DELETE_RATIO = 0.1;
var RECOVERY_RATIO = 0.1;

var FIXTURE_LOCAL = path.join(__dirname, 'test4100m.bin');
var FIXTURE_CI = path.join(__dirname, 'test2200m.bin');

function formatDuration(ms) {
	if (ms < 1000) return ms + 'ms';
	if (ms < 60000) return (ms / 1000).toFixed(1) + 's';
	return (ms / 60000).toFixed(1) + 'm';
}

function formatBytes(bytes) {
	var units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
	for (var i = 0; i < units.length; i++) {
		if (bytes < 10000) break;
		bytes /= 1024;
	}
	return Math.round(bytes * 100) / 100 + ' ' + units[i];
}

function parseArgs() {
	var args = process.argv.slice(2);
	var mode = 'auto';
	var jsonMetrics = false;

	for (var i = 0; i < args.length; i++) {
		if (args[i] === '--ci') {
			mode = 'ci';
		} else if (args[i] === '--local') {
			mode = 'local';
		} else if (args[i] === '--json-metrics') {
			jsonMetrics = true;
		}
	}

	return { mode: mode, jsonMetrics: jsonMetrics };
}

function run() {
	var opts = parseArgs();
	var fixturePath;

	// Determine which fixture to use
	if (opts.mode === 'ci') {
		fixturePath = FIXTURE_CI;
	} else if (opts.mode === 'local') {
		fixturePath = FIXTURE_LOCAL;
	} else {
		// Auto-detect: try local (4.3G) first, then CI (2.3G)
		if (fs.existsSync(FIXTURE_LOCAL)) {
			fixturePath = FIXTURE_LOCAL;
		} else if (fs.existsSync(FIXTURE_CI)) {
			fixturePath = FIXTURE_CI;
		} else {
			console.log('SKIP: no large test fixture (need test4100m.bin or test2200m.bin)');
			process.exit(0);
		}
	}

	// Validate chosen fixture exists
	if (!fs.existsSync(fixturePath)) {
		console.log('SKIP: large test fixture not found: ' + path.basename(fixturePath));
		process.exit(0);
	}

	var stat = fs.statSync(fixturePath);
	var fileSize = stat.size;
	var sliceSize = BLOCK_SIZE;
	var actualDataSlices = Math.floor(fileSize / BLOCK_SIZE);
	var slicesToDelete = Math.floor(actualDataSlices * DELETE_RATIO);

	var tempDir = helpers.getTempDir();
	var fixtureCopy = path.join(tempDir, 'test.bin');
	var outputBase = path.join(tempDir, 'out');
	var par3File = outputBase + '.par3';

	var metrics = {
		fixture: path.basename(fixturePath),
		fileSize: fileSize,
		fileSizeHuman: formatBytes(fileSize),
		sliceCount: actualDataSlices,
		sliceSize: sliceSize,
		blocksCorrupted: slicesToDelete,
		durations: {},
		durationsHuman: {},
		throughput: {},
		memoryUsage: {},
		result: 'FAIL'
	};

	var peakRSS = 0;
	var memoryCheck = setInterval(function() {
		var mem = process.memoryUsage();
		if (mem.rss > peakRSS) peakRSS = mem.rss;
	}, 100);

	console.log('PAR3 Large File E2E Test');
	console.log('=========================\n');
	console.log('Fixture: ' + path.basename(fixturePath));
	console.log('File size: ' + formatBytes(fileSize));
	console.log('Data slices: ' + actualDataSlices + ' (' + formatBytes(sliceSize) + ' each)');
	console.log('Recovery ratio: ' + (RECOVERY_RATIO * 100) + '%');
	console.log('Blocks to corrupt: ' + slicesToDelete + ' (' + (DELETE_RATIO * 100) + '%)\n');

	var startTime = Date.now();
	var copyFixtureStart, hashOriginalStart, createPar3Start, verifyStart, corruptArchiveStart, repairStart, hashRepairedStart;

	try {
		copyFixtureStart = Date.now();
		console.log('Copying fixture to temp dir...');
		fs.copyFileSync(fixturePath, fixtureCopy);
		console.log('  Copied: ' + fixtureCopy + ' (' + fs.statSync(fixtureCopy).size + ' bytes)\n');

		hashOriginalStart = Date.now();
		console.log('Hashing original file...');
		var originalHash = helpers.hashFile(fixtureCopy);
		console.log('  SHA256: ' + originalHash + '\n');

		createPar3Start = Date.now();
		console.log('Creating PAR3 archive with ' + (RECOVERY_RATIO * 100) + '% recovery...');

		par3.create([fixtureCopy], outputBase, {
			outputBase: outputBase,
			recoverySlices: { unit: 'ratio', value: RECOVERY_RATIO }
		}, function(err) {
			if (err) {
				console.error('  Create failed: ' + err.message);
				finish(null, err);
				return;
			}
			console.log('  Create succeeded\n');

			verifyStart = Date.now();
			console.log('Verifying PAR3 archive...');

			par3.verify(par3File, function(err2, verifyResult) {
				if (err2) {
					if (err2.message && err2.message.indexOf('greater than 2 GiB') !== -1) {
						console.log('  Verify threw expected >2GiB error (T10-T12 not yet applied): ' + err2.message + '\n');
					} else {
						console.error('  Verify failed: ' + err2.message);
						finish(null, err2);
						return;
					}
				} else {
					console.log('  Verify result:');
					console.log('    verified: ' + verifyResult.verified);
					console.log('    archiveOk: ' + verifyResult.archiveOk);
					console.log('    canRepair: ' + verifyResult.canRepair);
					console.log('    inputBlocks: ' + verifyResult.inputBlocks);
					console.log('    recoveryBlocks: ' + verifyResult.recoveryBlocks);
					console.log('    missingBlocks: ' + verifyResult.missingBlocks + '\n');

					// #97 regression guard: a freshly-created archive must be
					// fully intact. The verify FILE case must NOT inflate
					// inputCount (phantom block per file). Ground truth for the
					// input-block count is the create-side formula
					// (par3gen.js:1002): Math.ceil(totalSize / blockSize).
					var expectedInputBlocks = Math.ceil(fileSize / BLOCK_SIZE);
					if (verifyResult.inputBlocks !== expectedInputBlocks) {
						console.error('  ERROR: inputBlocks=' + verifyResult.inputBlocks + ' expected ' + expectedInputBlocks + ' (phantom-block bug #97)');
						finish(null, new Error('inputBlocks mismatch: got ' + verifyResult.inputBlocks + ', expected ' + expectedInputBlocks + ' (#97)'));
						return;
					}
					if (verifyResult.missingBlocks !== 0) {
						console.error('  ERROR: missingBlocks=' + verifyResult.missingBlocks + ' expected 0 (fresh archive)');
						finish(null, new Error('fresh archive reports ' + verifyResult.missingBlocks + ' missing blocks (#97)'));
						return;
					}
					if (verifyResult.archiveOk !== true) {
						console.error('  ERROR: archiveOk=' + verifyResult.archiveOk + ' expected true (fresh archive)');
						finish(null, new Error('fresh archive reports archiveOk=false (#97)'));
						return;
					}
					console.log('  Verify assertions passed: inputBlocks=' + verifyResult.inputBlocks + ', missingBlocks=0, archiveOk=true\n');
				}

				corruptArchiveStart = Date.now();
				console.log('Corrupting ' + slicesToDelete + ' random DATA packets in archive...');
				var corruptedBlocks = helpers.corruptArchiveDataPackets(par3File, slicesToDelete);
				console.log('  Corrupted block indices: [' + corruptedBlocks.join(', ') + ']\n');

				repairStart = Date.now();
				console.log('Running repair...');

				par3.repair(par3File, tempDir, { verbose: 1 }, function(err3, result) {
					if (err3) {
						console.error('  Repair failed: ' + err3.message);
						finish(null, err3);
						return;
					}

					console.log('  Repair result:');
					console.log('    repaired: ' + result.repaired);
					console.log('    blocksRepaired: ' + result.blocksRepaired);
					console.log('    missingBlocks: ' + result.missingBlocks + '\n');

					if (result.missingBlocks !== corruptedBlocks.length) {
						console.error('  ERROR: missingBlocks=' + result.missingBlocks + ' expected ' + corruptedBlocks.length);
						finish(null, new Error('missingBlocks mismatch: got ' + result.missingBlocks + ', expected ' + corruptedBlocks.length));
						return;
					}
					if (result.blocksRepaired !== corruptedBlocks.length) {
						console.error('  ERROR: blocksRepaired=' + result.blocksRepaired + ' expected ' + corruptedBlocks.length);
						finish(null, new Error('blocksRepaired mismatch: got ' + result.blocksRepaired + ', expected ' + corruptedBlocks.length));
						return;
					}

					hashRepairedStart = Date.now();
					console.log('Hashing repaired file...');

					var repairedFile = path.join(tempDir, 'block_0.dat');
					var exists = fs.existsSync(repairedFile);

					if (!exists) {
						console.error('  ERROR: Repaired file not found in output directory');
						finish(null, new Error('No repaired file produced'));
						return;
					}

					var repairedHash = helpers.hashFile(repairedFile);
					console.log('  SHA256: ' + repairedHash + '\n');

					if (repairedHash !== originalHash) {
						console.error('  ERROR: Hash mismatch!');
						console.error('  Expected: ' + originalHash);
						console.error('  Got:      ' + repairedHash);
						finish(null, new Error('Hash mismatch - repair failed'));
						return;
					}

					console.log('  Hash match: ORIGINAL === REPAIRED\n');
					finish({ success: true, result: result }, null);
				});
			});
		});
	} catch (err) {
		finish(null, err);
	}

	function finish(context, err) {
		clearInterval(memoryCheck);

		var endTime = Date.now();
		// P3 cubic-review fix: when an early error path (verify/repair
		// assertion, repaired-file-not-found) returns before later phases
		// start, their xxxStart variables are still undefined and the
		// duration arithmetic produces NaN. Fall back to "now" so the
		// duration for an unreached phase equals time spent failing.
		var nowFallback = endTime;
		copyFixtureStart = copyFixtureStart !== undefined ? copyFixtureStart : nowFallback;
		hashOriginalStart = hashOriginalStart !== undefined ? hashOriginalStart : nowFallback;
		createPar3Start = createPar3Start !== undefined ? createPar3Start : nowFallback;
		verifyStart = verifyStart !== undefined ? verifyStart : nowFallback;
		corruptArchiveStart = corruptArchiveStart !== undefined ? corruptArchiveStart : nowFallback;
		repairStart = repairStart !== undefined ? repairStart : nowFallback;
		hashRepairedStart = hashRepairedStart !== undefined ? hashRepairedStart : nowFallback;

		metrics.durations.copyFixture = copyFixtureStart - startTime;
		metrics.durations.hashOriginal = hashOriginalStart - copyFixtureStart;
		metrics.durations.createPar3 = createPar3Start - hashOriginalStart;
		metrics.durations.verify = verifyStart - createPar3Start;
		metrics.durations.corruptArchive = corruptArchiveStart - verifyStart;
		metrics.durations.repair = repairStart - corruptArchiveStart;
		metrics.durations.hashRepaired = hashRepairedStart - repairStart;
		metrics.durations.total = endTime - startTime;

		metrics.durationsHuman = {
			copyFixture: formatDuration(metrics.durations.copyFixture),
			hashOriginal: formatDuration(metrics.durations.hashOriginal),
			createPar3: formatDuration(metrics.durations.createPar3),
			verify: formatDuration(metrics.durations.verify),
			corruptArchive: formatDuration(metrics.durations.corruptArchive),
			repair: formatDuration(metrics.durations.repair),
			hashRepaired: formatDuration(metrics.durations.hashRepaired),
			total: formatDuration(metrics.durations.total)
		};

		metrics.memoryUsage.peakRSS = peakRSS;
		metrics.memoryUsage.peakRSSHuman = formatBytes(peakRSS);

		if (metrics.durations.createPar3 > 0) {
			metrics.throughput.createPar3MBps = (fileSize / 1048576) / (metrics.durations.createPar3 / 1000);
		}
		if (metrics.durations.repair > 0) {
			metrics.throughput.repairMBps = (fileSize / 1048576) / (metrics.durations.repair / 1000);
		}

		if (context && context.success) {
			metrics.result = 'PASS';
			console.log('====================');
			console.log('TEST PASSED');
			console.log('====================');
		} else {
			metrics.result = 'FAIL';
			metrics.error = err ? err.message : 'Unknown error';
			console.error('\n====================');
			console.error('TEST FAILED: ' + (err ? err.message : 'Unknown error'));
			console.error('====================');
			process.exitCode = 1;
		}

		console.log('\nPerformance Metrics:');
		console.log('  Fixture: ' + metrics.fixture);
		console.log('  File size: ' + metrics.fileSizeHuman);
		console.log('  Slice count: ' + metrics.sliceCount + ' (' + formatBytes(metrics.sliceSize) + ' each)');
		console.log('  Blocks corrupted: ' + metrics.blocksCorrupted);
		console.log('\nDurations:');
		console.log('  copyFixture: ' + metrics.durationsHuman.copyFixture);
		console.log('  hashOriginal: ' + metrics.durationsHuman.hashOriginal);
		console.log('  createPar3: ' + metrics.durationsHuman.createPar3);
		console.log('  verify: ' + metrics.durationsHuman.verify);
		console.log('  corruptArchive: ' + metrics.durationsHuman.corruptArchive);
		console.log('  repair: ' + metrics.durationsHuman.repair);
		console.log('  hashRepaired: ' + metrics.durationsHuman.hashRepaired);
		console.log('  TOTAL: ' + metrics.durationsHuman.total);
		console.log('\nThroughput:');
		// Cubic cycle 2 #98 P2: guard against undefined throughput on early-failure paths (otherwise .toFixed(2) crashes finish()).
		console.log('  createPar3: ' + (metrics.throughput.createPar3MBps || 0).toFixed(2) + ' MB/s');
		console.log('  repair: ' + (metrics.throughput.repairMBps || 0).toFixed(2) + ' MB/s');
		console.log('\nMemory:');
		console.log('  peakRSS: ' + metrics.memoryUsage.peakRSSHuman);

		if (opts.jsonMetrics) {
			console.error('\n---METRICS JSON START---');
			console.error(JSON.stringify(metrics, null, 2));
			console.error('---METRICS JSON END---');
		}

		console.log('\nCleaning up...');
		helpers.cleanup(tempDir);
		console.log('Done.');
	}
}

run();
