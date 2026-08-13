#!/usr/bin/env node
"use strict";
// TEMPORARY issue #62 repro — DELETED before the PR. Exercises the Fenger
// engine NAPI entry directly at the crash shape (N=1/R=2, 1 MiB blocks,
// per the windows-2025 + Node 20 CI failure) plus control shapes, under
// whatever dispatch env the caller sets (PAR3_GF64_USE_AVX512 /
// PAR3_AVX512_FORCE). A native crash (SIGSEGV) kills the process without
// a JS catch — the caller's exit code is the signal.
var binding = require('../build/Release/parpar_gf64.node');

function info() {
	var out = {};
	try { out.gf64_info = binding.gf64_info(); } catch (e) { out.gf64_info = 'n/a: ' + e.message; }
	out.node = process.version;
	out.platform = process.platform + '/' + process.arch;
	return out;
}
console.log('INFO ' + JSON.stringify(info()));

function tryCase(label, N, R, blockSize, fi, fr, threads) {
	console.log('CASE ' + label + ' START');
	var inputs = Buffer.alloc(N * blockSize);
	var out = Buffer.alloc(R * blockSize);
	var t0 = Date.now();
	try {
		binding.compute_recovery_fenger(inputs, out, N, R, blockSize, fi, fr, threads);
		console.log('OK   ' + label + ' (' + (Date.now() - t0) + ' ms)');
		return 0;
	} catch (e) {
		console.log('THROW ' + label + ': ' + (e && e.message));
		return 2;
	}
}

function tryFullControl(label, N, R, blockSize, fi, fr) {
	console.log('CASE ' + label + ' START');
	var inputs = Buffer.alloc(N * blockSize);
	var out = Buffer.alloc(R * blockSize);
	try {
		binding.compute_recovery_full(inputs, out, N, R, blockSize, fi, fr, 0);
		console.log('OK   ' + label);
		return 0;
	} catch (e) {
		console.log('THROW ' + label + ': ' + (e && e.message));
		return 2;
	}
}

var rc = 0;
rc |= tryCase('N=1/R=2/1MiB (crash shape)', 1, 2, 1048576, 0, 2, 0);
rc |= tryCase('N=1/R=2/1MiB threads=1', 1, 2, 1048576, 0, 2, 1);
rc |= tryCase('N=1/R=2/128B (small B)', 1, 2, 128, 0, 2, 0);
rc |= tryCase('N=1/R=2/8B (B=1)', 1, 2, 8, 0, 2, 0);
rc |= tryCase('N=2/R=2/128B (small unpadded)', 2, 2, 128, 0, 2, 0);
rc |= tryCase('N=1/R=1/1MiB', 1, 1, 1048576, 0, 1, 0);
rc |= tryCase('N=1000/R=512/128B (padded path)', 1000, 512, 128, 65536, 1048576, 0);
rc |= tryCase('N=8192/R=512/128B (unpadded pow2)', 8192, 512, 128, 0, 1048576, 0);
rc |= tryFullControl('control N=1/R=2/1MiB via compute_recovery_full', 1, 2, 1048576, 0, 2);
process.exit(rc);
