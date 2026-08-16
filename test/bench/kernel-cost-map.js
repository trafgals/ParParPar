#!/usr/bin/env node
// Cost-map probe: kernel time vs (N, R) at fixed word count.
// R-scaling  -> walk-depth (per-level overhead, small muls at the leaves)
// N-scaling  -> mul-size cost (the transforms)
"use strict";
var addon = require('../../build/Release/parpar_gf64.node');

function bench(N, R, BLOCK) {
	var inputs = Buffer.alloc(N * BLOCK);
	var seed = 0x9E3779B9;
	for (var w = 0; w < inputs.length / 8; w++) {
		seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
		inputs.writeUInt32LE(seed, w * 8);
		seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
		inputs.writeUInt32LE(seed, w * 8 + 4);
	}
	var out = Buffer.alloc(R * BLOCK);
	var best = Infinity;
	for (var r = 0; r < 3; r++) {
		var t0 = process.hrtime.bigint();
		addon.compute_recovery_fenger(inputs, out, N, R, BLOCK, 0, N, 8);
		var t1 = process.hrtime.bigint();
		var ms = Number(t1 - t0) / 1e6;
		if (ms < best) best = ms;
	}
	return best;
}

// Fixed words per block (128 KiB = 16384 words), vary N and R
var WORDS = 16384;
console.log('N\\R     R=2      R=8      R=32     (ms, 128 KiB blocks)');
[1024, 4096, 16384].forEach(function(N) {
	var row = 'N=' + String(N).padStart(5) + '  ';
	[2, 8, 32].forEach(function(R) {
		row += String(bench(N, R, WORDS * 8).toFixed(0)).padStart(8) + ' ';
	});
	console.log(row);
});
// Word-count scaling at fixed N/R (the per-word amortization)
console.log('\nwords scaling at N=4096, R=8:');
[2048, 16384, 131072].forEach(function(W) {
	console.log('  words=' + String(W).padStart(6) + ': ' + bench(4096, 8, W * 8).toFixed(1) + ' ms');
});
