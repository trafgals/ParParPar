#!/usr/bin/env node
// Pow2 recovery contract (issue #59 C2): the Fenger/Bostan-Schost fast
// path requires a power-of-2 recovery count. Explicit numeric counts
// must be pow2 (arbitrary counts are a PAR2 feature); derived specs
// (ratio/bytes) round UP to the next power of 2.
"use strict";
var assert = require('assert');
var par3 = require('../lib/par3gen.js');
var fs = require('fs'), path = require('path'), os = require('os');

var tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'par3-pow2-'));
var testFile = path.join(tempDir, 'test.bin');
fs.writeFileSync(testFile, Buffer.alloc(64 * 1024, 7)); // 64 KiB
var base = path.join(tempDir, 'out');
var opts = { outputBase: base, blockSize: 4096 };

var fails = 0;
function check(name, fn) {
	try {
		fn();
		console.log('ok: ' + name);
	} catch (e) {
		fails++;
		console.error('FAIL: ' + name + ' — ' + e.message);
	}
}

/* ---- Constructor (sync) checks ---- */
check('explicit non-pow2 count (1000) throws in the constructor', function() {
	assert.throws(function() {
		new par3.PAR3Gen([{ name: testFile, size: 64 * 1024 }], 4096,
			{ recoverySlices: 1000, blockSize: 4096 });
	}, /power of 2/, 'non-pow2 explicit count must throw the pow2 error');
});

check('negative count throws (cubic review on #87 P2.1)', function() {
	assert.throws(function() {
		new par3.PAR3Gen([{ name: testFile, size: 64 * 1024 }], 4096,
			{ recoverySlices: -1, blockSize: 4096 });
	}, /finite non-negative integer/, 'negative count must throw the finiteness error');
});

check('fractional count throws (cubic review on #87 P2.1)', function() {
	assert.throws(function() {
		new par3.PAR3Gen([{ name: testFile, size: 64 * 1024 }], 4096,
			{ recoverySlices: 4.5, blockSize: 4096 });
	}, /finite non-negative integer/, 'fractional count must throw the finiteness error');
});

check('NaN count throws (cubic review on #87 P2.1)', function() {
	assert.throws(function() {
		new par3.PAR3Gen([{ name: testFile, size: 64 * 1024 }], 4096,
			{ recoverySlices: NaN, blockSize: 4096 });
	}, /finite non-negative integer/, 'NaN count must throw the finiteness error');
});

check('explicit pow2 count (8) is accepted', function() {
	var gen = new par3.PAR3Gen([{ name: testFile, size: 64 * 1024 }], 4096,
		{ recoverySlices: 8, blockSize: 4096 });
	assert.strictEqual(gen.opts.recoverySlices, 8, 'pow2 count passes through');
});

check('ratio spec rounds UP to pow2 (cubic review on #87 P2.5)', function() {
	// 1024 blocks x 10% = 102.4 -> Math.ceil -> 103 -> next pow2 = 128
	var gen = new par3.PAR3Gen([{ name: testFile, size: 1024 * 4096 }], 4096,
		{ recoverySlices: { unit: 'ratio', value: 0.1 }, blockSize: 4096 });
	assert.strictEqual(gen.opts.recoverySlices, 128, '10% of 1024 blocks -> 128');
});

check('ratio spec just above pow2 boundary still rounds up (cubic P2.5)', function() {
	// 1024 blocks x 12.5% = 128 -> already pow2, stays 128
	var gen = new par3.PAR3Gen([{ name: testFile, size: 1024 * 4096 }], 4096,
		{ recoverySlices: { unit: 'ratio', value: 0.125 }, blockSize: 4096 });
	assert.strictEqual(gen.opts.recoverySlices, 128, '12.5% of 1024 blocks -> 128');
});

check('ratio spec already-pow2 is unchanged', function() {
	// 4096 blocks x 12.5% = 512 (pow2)
	var gen = new par3.PAR3Gen([{ name: testFile, size: 4096 * 4096 }], 4096,
		{ recoverySlices: { unit: 'ratio', value: 0.125 }, blockSize: 4096 });
	assert.strictEqual(gen.opts.recoverySlices, 512, '12.5% of 4096 blocks -> 512');
});

check('default (10% ratio) derives pow2', function() {
	var gen = new par3.PAR3Gen([{ name: testFile, size: 1024 * 4096 }], 4096,
		{ blockSize: 4096 });
	var r = gen.opts.recoverySlices;
	assert(r > 1 && (r & (r - 1)) === 0, 'default derivation must be pow2 (got ' + r + ')');
});

check('minRecoverySlices clamp stays pow2 (cubic review on #87 P2.2)', function() {
	var gen = new par3.PAR3Gen([{ name: testFile, size: 1024 * 4096 }], 4096,
		{ recoverySlices: { unit: 'ratio', value: 0.01 }, minRecoverySlices: 32, blockSize: 4096 });
	assert.strictEqual(gen.opts.recoverySlices, 32, 'min clamp (pow2) wins');
});

check('maxRecoverySlices clamp numeric value rounds up (cubic review on #87 P2.2)', function() {
	// 1024 blocks @ 50% = 512; maxRecoverySlices=400 (non-pow2) clamps to 400,
	// then rounds up to 512 to honor the pow2 contract (without the round
	// the cap would be rejected by the final check).
	var gen = new par3.PAR3Gen([{ name: testFile, size: 1024 * 4096 }], 4096,
		{ recoverySlices: { unit: 'ratio', value: 0.5 }, maxRecoverySlices: 400, blockSize: 4096 });
	assert.strictEqual(gen.opts.recoverySlices, 512, 'numeric max cap rounds up to pow2');
});

/* ---- Public create() API checks (cubic review on #87 P2.3): the contract
 * test must cover the spec mapping in par3_create, not just the
 * PAR3Gen constructor. */
par3.create([testFile], base, { recoverySlices: 1000, blockSize: 4096 }, function(err) {
	check('create() callback error includes the pow2 message', function() {
		assert(err, 'the create callback must receive the pow2 error');
		assert(/power of 2/.test(err.message), 'error must mention power of 2 (got: ' + err.message + ')');
	});
	par3.create([testFile], base, { recoverySlices: 8, blockSize: 4096 }, function(err2) {
		check('create() with explicit pow2 count succeeds', function() {
			assert.strictEqual(err2, null, 'create() with pow2 count should not error (got: ' + (err2 && err2.message) + ')');
		});
		par3.create([testFile], base, { recoverySlices: { unit: 'ratio', value: 0.1 }, blockSize: 4096 }, function(err3) {
			check('create() with ratio spec succeeds (proportional)', function() {
				assert.strictEqual(err3, null, 'create() with ratio spec should not error (got: ' + (err3 && err3.message) + ')');
			});
			try { fs.rmSync(tempDir, { recursive: true, force: true }); } catch (e) {}
			if (fails > 0) {
				console.error('POW2_CONTRACT_FAIL ' + fails);
				process.exit(1);
			}
			console.log('POW2_CONTRACT_PASS');
			process.exit(0);
		});
	});
});