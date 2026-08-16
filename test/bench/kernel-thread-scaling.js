#!/usr/bin/env node
// Kernel thread-scaling probe: compute_recovery_fenger at a 1G/1M-like
// shape (256 MiB / 256 x 1 MiB blocks / R=8) across thread counts.
// The signal: if throughput scales ~linearly, the kernel is bandwidth-
// bound (fix = fewer passes / wider batches); if flat, it's serialized
// or latency-bound (fix = pass-level parallelism / cache-blocked tree).
"use strict";
var addon = require('../../build/Release/parpar_gf64.node');

var N = 256, R = 8, BLOCK = 1024 * 1024;
var inputs = Buffer.alloc(N * BLOCK);
// deterministic pseudo-random-ish fill (no crypto needed)
var seed = 0x9E3779B9;
for (var w = 0; w < inputs.length / 8; w++) {
	seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
	inputs.writeUInt32LE(seed, w * 8);
	seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
	inputs.writeUInt32LE(seed, w * 8 + 4);
}
var out = Buffer.alloc(R * BLOCK);

[1, 2, 4, 8, 0].forEach(function(t) {
	// warmup (first-call setup: dispatch init, tree prepare) + best-of-3
	out.fill(0);
	addon.compute_recovery_fenger(inputs, out, N, R, BLOCK, 0, N, t);
	var best = Infinity;
	for (var r = 0; r < 3; r++) {
		out.fill(0);
		var t0 = process.hrtime.bigint();
		addon.compute_recovery_fenger(inputs, out, N, R, BLOCK, 0, N, t);
		var t1 = process.hrtime.bigint();
		var ms = Number(t1 - t0) / 1e6;
		if (ms < best) best = ms;
	}
	var gbs = (N * BLOCK) / (best / 1000) / (1024 * 1024 * 1024);
	console.log('threads=' + t + ': ' + best.toFixed(1) + ' ms  (' + gbs.toFixed(2) + ' GiB/s input streamed)');
});
