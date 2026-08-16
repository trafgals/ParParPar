"use strict";

/*
 * Test CLI help and option parser contract for bin/par3.js
 * Reference: cubic review 7cd436ab-4b5d-4b56-b650-20c4cfe44258 (PR #84)
 *
 * Asserts:
 * 1. `bin/par3.js --help` outputs `--block-size, -b <size>` with (default: 1MB).
 * 2. `bin/par3.js --help` outputs `--gf-method, -m <method>`.
 * 3. `bin/par3.js --help` outputs `--memory-limit <size>` (and does not alias -m to memory).
 * 4. `package.json` declares bin mapping for both `par3` and `parpar`.
 */

var assert = require('assert');
var spawnSync = require('child_process').spawnSync;
var path = require('path');
var fs = require('fs');

var repoRoot = path.join(__dirname, '..');
var par3Bin = path.join(repoRoot, 'bin', 'par3.js');
var pkgJsonPath = path.join(repoRoot, 'package.json');

console.log('Testing bin/par3.js CLI options and package.json bin contract...');

// 1. Check package.json bin declaration
var pkg = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8'));
assert(typeof pkg.bin === 'object' && pkg.bin !== null, 'package.json must declare bin object mapping');
assert.strictEqual(pkg.bin.par3, 'bin/par3.js', 'package.json must expose bin.par3');
assert.strictEqual(pkg.bin.parpar, 'bin/parpar.js', 'package.json must expose bin.parpar');
console.log('  PASS: package.json declares bin.par3 and bin.parpar');

// 2. Check par3.js --help output
var res = spawnSync(process.execPath, [par3Bin, '--help'], { encoding: 'utf8' });
assert.strictEqual(res.status, 0, 'par3 --help should exit with status 0');
var output = res.stderr || res.stdout; // bin/par3.js writes help to stderr

assert(output.indexOf('--block-size, -b <size>') !== -1, 'help must document --block-size, -b');
assert(output.indexOf('default: 1MB') !== -1, 'help must document default block size as 1MB');
assert(output.indexOf('--gf-method, -m <method>') !== -1, 'help must document -m as --gf-method');
assert(output.indexOf('--memory-limit <size>') !== -1, 'help must document --memory-limit');
assert(output.indexOf('--slice-size') === -1, 'help must not contain non-existent --slice-size');

console.log('  PASS: bin/par3.js --help contract matches documented options (default: 1MB, -m gf-method, --memory-limit)');

// 3. Check JS API contract: require('./lib/par3gen') exports create and repair
var par3gen = require('../lib/par3gen.js');
assert.strictEqual(typeof par3gen.create, 'function', 'par3gen must export create');
assert.strictEqual(typeof par3gen.repair, 'function', 'par3gen must export repair');
console.log('  PASS: require("./lib/par3gen") exports create and repair functions');

console.log('ALL TESTS PASSED (cubic review 7cd436ab-4b5d-4b56-b650-20c4cfe44258 / be1cb78c-140c-4f0b-b4e5-24abc4cd3dd6)');
