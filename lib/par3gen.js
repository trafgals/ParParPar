"use strict";

var emitter = require('events').EventEmitter;
var async = require('async');
var fs = require('fs');
var path = require('path');
var crypto = require('crypto');
var blake3 = require('blake3');
var workerThreads = require('worker_threads');
var os = require('os');

var gf64Binding = null;
var gf64BindingLoadAttempted = false;

// Minimum number of exported keys required for a usable addon build.
// Matches the .github/workflows/par3-tests.yml native-addon load gate.
var GF64_MIN_EXPORT_KEYS = 5;

// x64/ia32 are the architectures where the full native GF(2^64) addon is built.
// On non-x86 (e.g. macOS arm64) the build falls through to src/gf64_stub.cc and
// exports an essentially empty object — callers must fall back to the JS kernel
// there instead of treating the stub as a build failure.
var isX86ish = (process.arch === 'x64' || process.arch === 'ia32');

// T3 (par3gen-addon-load-contract): native GF(2^64) addon loader.
// Behavior table (per Todo 3 of .omo/plans/ci-flow-fix.md):
//   1. no candidate file exists (any arch)     -> stderr warning + return null
//   2. require() throws on x64/ia32            -> throw diagnostic after loop
//   3. require() throws on non-x86             -> stderr warning + return null
//   4. loads with >= 5 keys (any arch)         -> use it (return; later candidates not tried)
//   5. loads with < 5 keys on x64/ia32         -> throw diagnostic after loop
//   6. loads with < 5 keys on non-x86          -> stderr warning + return null
// Callers (PAR3Gen constructor, _processRecoveryBatch, run, close, repair) all
// handle `null` by falling back to the JS kernel, so the change is safe.
function getGf64Binding() {
	if(gf64BindingLoadAttempted) return gf64Binding;
	gf64BindingLoadAttempted = true;
	var loadErrors = [];
	var triedPaths = [];
	var anyExisted = false;
	var buildDir = null;
	var candidates = null;
	try {
		buildDir = path.join(path.dirname(path.dirname(__filename)), 'build', 'Release');
		candidates = ['parpar_gf64.node', 'parpar_gf64_native.node', 'gf64_addon.node'];
	} catch(e) {
		loadErrors.push('path resolution: ' + (e && e.message ? e.message : String(e)));
	}
	if(candidates) {
		for(var i = 0; i < candidates.length; i++) {
			var p = path.join(buildDir, candidates[i]);
			triedPaths.push(p);
			if(!fs.existsSync(p)) continue;
			anyExisted = true;
			try {
				var mod = require(p);
				if(!mod || typeof mod !== 'object') {
					loadErrors.push(p + ': require() returned non-object');
					continue;
				}
				var exportKeys = Object.keys(mod).length;
				if(exportKeys < GF64_MIN_EXPORT_KEYS) {
					loadErrors.push(p + ': only ' + exportKeys + ' export keys (need >= ' + GF64_MIN_EXPORT_KEYS + ')');
					continue;
				}
				// T8 (par3-create-throughput-400mbps): flag the single-call entry
				// so PAR3Gen.run() can pick it instead of looping compute_recovery.
				if(typeof mod.compute_recovery_full === 'function') {
					mod.hasComputeRecoveryFull = true;
				} else {
					mod.hasComputeRecoveryFull = false;
				}
				gf64Binding = mod;
				return gf64Binding;
			} catch(e) {
				loadErrors.push(p + ': ' + (e && e.message ? e.message : String(e)));
			}
		}
	}
	// After the loop: no candidate succeeded. Branch on what actually happened.
	if(!anyExisted) {
		// Case 1 (any arch): no candidate file exists. Warn and return null so
		// callers fall back to the JS kernel. This also covers the degenerate
		// "path resolution failed" case (triedPaths is empty then).
		process.stderr.write(
			'Warning: no native GF(2^64) addon found' +
			(buildDir ? ' at ' + buildDir : '') +
			' (tried: ' + (triedPaths.length > 0 ? triedPaths.join(', ') : '(path resolution failed)') +
			'); using pure-JS fallback\n'
		);
		gf64Binding = null;
		return null;
	}
	if(isX86ish) {
		// Cases 2 + 5 (x64/ia32): candidates existed but require() threw or
		// the build is under-built (< 5 keys). This is a real build failure
		// on x86 — surface a clear diagnostic and throw.
		throw new Error(
			'par3gen: failed to load a usable native GF(2^64) addon ' +
			'(need >= ' + GF64_MIN_EXPORT_KEYS + ' exported keys). Tried: ' +
			loadErrors.join(' | ') + '. ' +
			'Run `npm install` (or `npm run build`) and verify `build/Release/parpar_gf64.node` exists.'
		);
	}
	// Cases 3 + 6 (non-x86): candidates existed but require() threw or the
	// stub build is under-built (macOS arm64 is the canonical case). Warn
	// and fall back so the library still works on those platforms.
	process.stderr.write(
		'Warning: native addon missing or under-built on non-x86 ' +
		'(need >= ' + GF64_MIN_EXPORT_KEYS + ' keys for x86); using pure-JS fallback. ' +
		'Tried: ' + loadErrors.join(' | ') + '\n'
	);
	gf64Binding = null;
	return null;
}

// T12 (par3-cauchy-fft-kernel): Barycentric kernel dispatch.
// Picks compute_recovery_barycentric (Cauchy-FFT, near-linear in N) for large
// workloads; falls back to the legacy kernel otherwise.
// Threshold: PAR3_BF64_MIN_INPUTS env (default 10000). Requires blockSize%8==0.
// Binding is the first arg so the test layer can inject a mock. The optional
// fallbackFn lets non-single-call sites (per-batch path) point at a kernel
// whose output buffer matches their XOR-into-accumulator semantics.
var BARY_MIN_INPUTS_DEFAULT = 10000;
var _dispatchRecoveryWarned = false;
var _fengerFallbackWarned = false;

function _isPowerOfTwoOrTrivial(n) {
	return n === 0 || n === 1 || (n > 1 && (n & (n - 1)) === 0);
}

// Padding-cost bound (issue #59 §3): the always-pad policy pads N up to
// next_pow2(N). NOTE: next_pow2(N) <= 2N holds for every N >= 1, so this
// bound is mathematically vacuous — the N gate is fully removed and any
// input count is Fenger-eligible on N (the engine pads internally). The
// helper is kept as the documented policy surface (and pins the
// no-32-bit-truncation property: `p *= 2` NOT `p <<= 1`, which overflows
// to a negative/zero Number for N >= 2^30+1 and would loop forever).
function _fengerPaddingReasonable(n) {
	if(n <= 1) return true;
	var p = 1;
	while(p < n) p *= 2;
	return p <= n * 2;
}

// A4 (issue #59 §8, #62): the Windows Fenger opt-in gate is REMOVED.
// The default-on blocker — a Node-20-only native SIGILL on the
// windows-2025 CI runner (silent exit 132 at trivial shapes N=1/R=2)
// — was root-caused to MSVC's whole-TU /arch:AVX512 flag letting the
// auto-vectorizer emit EVEX/ZMM into the scalar pipeline (SIGILL on
// non-AVX-512 runners), and fixed by confining /arch:AVX512 to the
// intrinsic TUs (gf64_pipeline at /arch:AVX2). Windows now dispatches
// Fenger by the same rules as every other platform. An OPTIONAL
// defensive size cap remains (PAR3_FENGER_WINDOWS_MAX_INPUTS,
// default: no cap); PAR3_FENGER_WINDOWS_ENABLE is dead and ignored.
function _fengerWindowsMaxInputs() {
	var v = parseInt(process.env.PAR3_FENGER_WINDOWS_MAX_INPUTS, 10);
	if(!isFinite(v) || v <= 0) v = Infinity;
	return v;
}

function dispatchRecovery(binding, inputs, outputs, numInputs, numRecovery, blockSize, firstInput, firstRecovery, numThreads, fallbackFn) {
	// Fenger dispatch (issue #46 K3): default for power-of-2 workloads on
	// every host. The Windows-only exclusion is gone: the native crash
	// that forced it (issue #62 — SIGILL from whole-TU /arch:AVX512
	// auto-vectorized EVEX in the scalar pipeline) is fixed by the
	// gf64_pipeline TU split, so Windows uses the same default.
	// Env contract:
	//   PAR3_GF64_USE_FENGER=1 / true / yes / on  -> force Fenger (error if
	//                                               the binding lacks it)
	//   PAR3_GF64_USE_FENGER=0 / false / no / off -> kill switch: skip Fenger
	//   unset                                    -> Fenger by default when
	//                                               power-of-2 eligible
	//   PAR3_FENGER_WINDOWS_MAX_INPUTS           -> optional defensive cap
	//                                               (Windows only; default:
	//                                               no cap)
	var useFengerEnv = process.env.PAR3_GF64_USE_FENGER;
	var fengerForced = useFengerEnv === '1' || useFengerEnv === 'true' || useFengerEnv === 'yes' || useFengerEnv === 'on';
	var fengerDisabled = useFengerEnv === '0' || useFengerEnv === 'false' || useFengerEnv === 'no' || useFengerEnv === 'off';
	var hasFenger = binding && typeof binding.compute_recovery_fenger === 'function';
	// Always-pad eligibility (issue #59 §3 / A2): the engine pads
	// non-power-of-2 numInputs up to next_pow2(numInputs) with synthetic
	// zero-data inputs (bit-exact; test_gf64_fenger_padded + the engine
	// routing contract), so the N power-of-2 gate is replaced by the
	// padding-cost bound. R keeps the power-of-2 gate: the create policy
	// derives recoverySlices as a 1/2^k fraction of the PADDED N, so R
	// is power-of-2 by construction on every create; legacy archives
	// with odd R keep falling through to the legacy path (no regression).
	// A4/#62: the Windows opt-in gate is REMOVED — the SIGILL was the
	// whole-TU /arch:AVX512 auto-vectorization (fixed by the TU split),
	// so Windows dispatches Fenger by the same rules as other hosts.
	// An optional defensive cap remains: PAR3_FENGER_WINDOWS_MAX_INPUTS
	// (default: no cap). PAR3_FENGER_WINDOWS_ENABLE is dead/ignored.
	var fengerEligible = hasFenger
		&& _fengerPaddingReasonable(numInputs)
		&& _isPowerOfTwoOrTrivial(numRecovery)
		&& blockSize % 8 === 0
		&& (fengerForced || process.platform !== 'win32'
			|| numInputs <= _fengerWindowsMaxInputs());

	if(fengerForced) {
		if(!hasFenger) {
			throw new Error('PAR3_GF64_USE_FENGER=1 but binding lacks compute_recovery_fenger; rebuild the native addon');
		}
		if(!fengerEligible) {
			if(!_fengerFallbackWarned) {
				_fengerFallbackWarned = true;
				console.warn('[par3gen] fenger-fallback-scalar: PAR3_GF64_USE_FENGER=1 but numRecovery=' + numRecovery + ' is not 0/1/power-of-2 (or N=' + numInputs + ' exceeds the platform gate); falling back to scalar kernel');
			}
		}
	}

	if(fengerEligible && !fengerDisabled) {
		if(!_dispatchRecoveryWarned) {
			_dispatchRecoveryWarned = true;
			console.log('[par3gen] using Fenger kernel for numInputs=' + numInputs + ', numRecovery=' + numRecovery);
		}
		return binding.compute_recovery_fenger(
			inputs, outputs, numInputs, numRecovery, blockSize, firstInput, firstRecovery, numThreads
		);
	}

	var baryMinInputs = parseInt(process.env.PAR3_BF64_MIN_INPUTS, 10);
	if(!isFinite(baryMinInputs) || baryMinInputs <= 0) baryMinInputs = BARY_MIN_INPUTS_DEFAULT;

	var hasBarycentric = binding && typeof binding.compute_recovery_barycentric === 'function';
	var useBarycentric = hasBarycentric
		&& numInputs > baryMinInputs
		&& numRecovery > 0
		&& blockSize % 8 === 0;

	if(useBarycentric) {
		if(!_dispatchRecoveryWarned) {
			_dispatchRecoveryWarned = true;
			console.log('[par3gen] using Barycentric kernel for numInputs=' + numInputs);
		}
		return binding.compute_recovery_barycentric(
			inputs, outputs, numInputs, numRecovery, blockSize, firstInput, firstRecovery, numThreads
		);
	}

	var fallback = (typeof fallbackFn === 'function') ? fallbackFn : (binding && binding.compute_recovery_full);
	if(typeof fallback !== 'function') {
		throw new Error('dispatchRecovery: no usable fallback kernel (binding lacks compute_recovery_barycentric and compute_recovery_full, and no fallbackFn supplied)');
	}
	return fallback(
		inputs, outputs, numInputs, numRecovery, blockSize, firstInput, firstRecovery, numThreads
	);
}

var gf64Js = require('./gf64_js');

// P31 drive-by (issue #59): no-op defaults for the profiler hooks so
// direct _processRecoveryBatch callers (test/par3-golden.js invokes it
// without run()) don't crash with "global.__par3ProfStart is not a
// function" before run() installs the real hooks at line ~1547.
if (typeof global.__par3ProfStart !== 'function') global.__par3ProfStart = function() {};
if (typeof global.__par3ProfEnd !== 'function') global.__par3ProfEnd = function() {};

var allocBuffer = (Buffer.allocUnsafe || Buffer);
var bufferSlice = Buffer.prototype.readBigInt64BE ? Buffer.prototype.subarray : Buffer.prototype.slice;

var MAX_BUFFER_SIZE = (require('buffer').kMaxLength || (1024*1024*1024-1)) - 1024-68;

// Simple buffer pool for batch concatenation (reduces GC pressure)
var concatPool = [];
var CONCAT_POOL_MAX = 4;
function getConcatBuffer(minSize) {
	for(var i = 0; i < concatPool.length; i++) {
		if(concatPool[i] && concatPool[i].length >= minSize) {
			var buf = concatPool[i];
			concatPool[i] = null;
			return buf;
		}
	}
	// No suitable buffer found, create new (but don't exceed max pool entries)
	if(concatPool.length < CONCAT_POOL_MAX) {
		var newBuf = Buffer.alloc(minSize);
		return newBuf;
	}
	// Pool full, find the smallest suitable slot (or any empty slot)
	var smallestIdx = 0;
	var smallestSize = Infinity;
	for(var i = 0; i < concatPool.length; i++) {
		if(concatPool[i] === null) {
			concatPool[i] = Buffer.alloc(minSize);
			return concatPool[i];
		}
		if(concatPool[i].length < smallestSize) {
			smallestSize = concatPool[i].length;
			smallestIdx = i;
		}
	}
	// Replace the smallest buffer
	concatPool[smallestIdx] = Buffer.alloc(minSize);
	return concatPool[smallestIdx];
}
function releaseConcatBuffer(buf) {
	for(var i = 0; i < concatPool.length; i++) {
		if(concatPool[i] === null) {
			concatPool[i] = buf;
			return;
		}
	}
	// Pool full, try to grow if under max
	if(concatPool.length < CONCAT_POOL_MAX) {
		concatPool.push(buf);
	}
	// otherwise discard
}

// A3 / T1: module-level input buffer pool for the create path. The pool is
// auto-sized to the workload at create time (totalInputBlocks * blockSize,
// 64 MiB floor), replacing the old pre-allocated 64 MiB pool that would
// exhaust on archives whose total read exceeded that. The slice shares memory
// with the pool, eliminating the GC pressure from per-block Buffer.alloc()
// at high block counts (e.g. 1 GiB at 4 KiB blocks = 262144 allocs).
//
// The pool is created lazily by _ensureInputPool(totalSize) on the first
// create call. PAR3_GF64_INPUT_POOL_SIZE env var overrides the computed size.
// acquireInputBuffer() throws a clear error if the override is too small.
var INPUT_BUFFER_POOL_SIZE = null;
var inputBufferPool = null;
var inputBufferPoolOffset = 0;

function acquireInputBuffer(size) {
	// Fallback: sizes larger than the pool (rare — only for a single very
	// large block) get a one-off Buffer.alloc. Avoids the throw on genuinely
	// oversize reads while keeping the common-case path allocation-free.
	if(size > INPUT_BUFFER_POOL_SIZE) return Buffer.alloc(size);
	if(inputBufferPoolOffset + size > INPUT_BUFFER_POOL_SIZE) {
		throw new Error(
			'Input buffer pool exhausted (' + inputBufferPoolOffset + ' + ' + size +
			' > ' + INPUT_BUFFER_POOL_SIZE + ' bytes). Reduce input size, raise ' +
			'INPUT_BUFFER_POOL_SIZE, or wait for B2 (LRU eviction).'
		);
	}
	// .slice() shares memory with the underlying pool — this is the GC win.
	// The returned slice aliases inputBufferPool; callers must .copy() the
	// data out before the offset wraps (B2) or before the next create
	// resets the pool.
	var slice = inputBufferPool.slice(inputBufferPoolOffset, inputBufferPoolOffset + size);
	inputBufferPoolOffset += size;
	return slice;
}

// Intentionally a no-op for A3. Within a single create, the per-block buf
// is consumed by .copy() into fullInputs and into the packet synchronously
// in the same iteration, so the next acquire() can safely overwrite the
// same region (the data is already in two independent buffers by then).
// Full LRU eviction / wrap-around is deferred to B2.
function releaseInputBuffer(slice) { /* intentional no-op for A3 */ }

function _resetInputBufferPool() {
	inputBufferPoolOffset = 0;
}

// T1: auto-size the input buffer pool to the workload. Called once at the
// create entry point with totalInputBlocks * blockSize. PAR3_GF64_INPUT_POOL_SIZE
// env var overrides the computed size. If the pool is already initialized (from a
// prior create in the same process), this is a no-op — use _resetInputBufferPool
// to reset the offset between creates.
function _ensureInputPool(totalSize) {
	if(inputBufferPool !== null) return;
	var envSize = parseInt(process.env.PAR3_GF64_INPUT_POOL_SIZE, 10);
	var poolSize;
	if(!isNaN(envSize) && envSize > 0) {
		poolSize = envSize;
	} else {
		// 64 MiB floor + 64 KiB headroom for small archives
		poolSize = Math.max(64 * 1024 * 1024, totalSize + 64 * 1024);
	}
	poolSize = Math.min(poolSize, Number.MAX_SAFE_INTEGER);
	INPUT_BUFFER_POOL_SIZE = poolSize;
	inputBufferPool = Buffer.alloc(poolSize);
	inputBufferPoolOffset = 0;
}

// B2 (par3-1200mbps): module-level LRU buffer pool for recovery blocks.
// Replaces per-create Buffer.alloc(totalRecoverySize) in the create path with
// a pool of zero-initialized buffers capped at 16 MiB total. When the same
// Node process runs multiple creates (CLI retry, batch script, worker_threads
// pool, repeated PAR3Gen.run() calls in one process), the second+ create
// reuses the buffer from the previous create instead of allocating fresh —
// eliminates GC pressure from the recovery accumulator across calls.
//
// Size choice: 16 MiB > the typical recovery accumulator (a few MiB for 10%
// recovery at 4 KiB blocks), but small enough to keep pool pressure bounded.
// Oversize accumulators (> 16 MiB) bypass the pool entirely (Buffer.alloc
// fallback) so a single huge create doesn't blow the budget.
//
// Eviction policy: LRU on acquire/release. acquireRecoveryBuffer updates the
// entry's lastUsed timestamp; releaseRecoveryBuffer marks the buffer as
// available and evicts the oldest unused entries until the pool total is
// back under 16 MiB. Single-threaded (Node event loop) — no locking needed.
//
// Zero-init invariant: all pool entries are Buffer.alloc(size), and any
// pool-reused buffer is .fill(0)'d on acquire. This is critical because the
// per-batch _processRecoveryBatch path XORs partial results into the
// accumulator (initial value matters), and compute_recovery_full's Cauchy
// RHS leaves unwritten bytes that would corrupt recovery output if non-zero
// (a prior bug — keep zero-init). DO NOT switch to allocUnsafe and DO NOT
// pool buffers that came from a larger buffer (e.g., via .slice()).
//
// _resetRecoveryBufferPool() exists as an explicit escape hatch for fully-
// unrelated create operations. It is NOT called automatically at the start
// of run() — the pool persists across creates so the multi-call benefit
// actually applies. A future B3 may add a small warm-cache strategy.
var RECOVERY_BUFFER_POOL_MAX_BYTES = 16 * 1024 * 1024;
var recoveryBufferPool = [];
var _recoveryBufferLastUsedSeq = 0;

function acquireRecoveryBuffer(size) {
	_recoveryBufferLastUsedSeq++;
	var now = _recoveryBufferLastUsedSeq;
	// Oversize fallback: bypass the pool. The buffer is allocated fresh and
	// NOT added to the pool — releaseRecoveryBuffer will be a no-op for it
	// (the buffer isn't in the pool, so the search misses).
	if(size > RECOVERY_BUFFER_POOL_MAX_BYTES) return Buffer.alloc(size);
	// Look for an available buffer of exact size.
	for(var i = 0; i < recoveryBufferPool.length; i++) {
		var entry = recoveryBufferPool[i];
		if(entry && entry.size === size && entry.inUse === false) {
			entry.inUse = true;
			entry.lastUsed = now;
			// Zero-init on pool reuse — Cauchy XOR accumulation requires
			// zero-initialized accumulator (see header comment).
			entry.buf.fill(0);
			return entry.buf;
		}
	}
	// No match. Allocate fresh — Buffer.alloc (NOT allocUnsafe) per the
	// zero-init invariant. Reuse a null slot if eviction left one, else push.
	var buf = Buffer.alloc(size);
	var newEntry = { buf: buf, size: size, lastUsed: now, inUse: true };
	for(var k = 0; k < recoveryBufferPool.length; k++) {
		if(recoveryBufferPool[k] === null) {
			recoveryBufferPool[k] = newEntry;
			_evictRecoveryBufferPoolIfOverBudget();
			return buf;
		}
	}
	recoveryBufferPool.push(newEntry);
	_evictRecoveryBufferPoolIfOverBudget();
	return buf;
}

function releaseRecoveryBuffer(buf) {
	for(var i = 0; i < recoveryBufferPool.length; i++) {
		var entry = recoveryBufferPool[i];
		if(entry && entry.buf === buf) {
			entry.inUse = false;
			entry.lastUsed = ++_recoveryBufferLastUsedSeq;
			break;
		}
	}
	// Eviction is a no-op for in-use-only oversize workloads; safe to call.
	_evictRecoveryBufferPoolIfOverBudget();
}

// Evicts the oldest unused entries until total pool size <= budget. Leaves
// nulls in the array (acquire reuses null slots); doesn't compact so the
// active count stays bounded by RECOVERY_BUFFER_POOL_MAX_BYTES / min_entry.
function _evictRecoveryBufferPoolIfOverBudget() {
	var total = 0;
	for(var i = 0; i < recoveryBufferPool.length; i++) {
		var e = recoveryBufferPool[i];
		if(e) total += e.size;
	}
	while(total > RECOVERY_BUFFER_POOL_MAX_BYTES) {
		var oldestIdx = -1;
		var oldestTime = Infinity;
		for(var j = 0; j < recoveryBufferPool.length; j++) {
			var e2 = recoveryBufferPool[j];
			if(e2 && e2.inUse === false && e2.lastUsed < oldestTime) {
				oldestTime = e2.lastUsed;
				oldestIdx = j;
			}
		}
		if(oldestIdx === -1) break; // no evictable entry (all in use)
		total -= recoveryBufferPool[oldestIdx].size;
		recoveryBufferPool[oldestIdx] = null;
	}
}

function _resetRecoveryBufferPool() {
	recoveryBufferPool = [];
}

// B3 (par3-1200mbps): parallel per-block packet-checksum via worker_threads.
// Pool size: Math.min(4, os.cpus().length). 4 matches the bench protocol
// (taskset -c 0-3); over-subscribing past 4 yields no win on this workload.
// Hash function: blake3 — same algorithm and input layout as the inline
// finalizePacketHeader path, so output is bit-exact identical.
// Ordering: hashes are dispatched in block-index order but may complete
// out of order; the read loop batches them by POOL_SIZE and writes
// packets in block-index order, preserving wire order.
var HASHER_WORKER_PATH = path.join(__dirname, 'par3_hasher_worker.js');
var HASH_POOL_SIZE = Math.min(4, (os.cpus() || []).length || 1);
var HASH_POOL_ENABLED = process.env.PAR3_GF64_PARALLEL_HASH !== '0';
var HASH_BATCH_MULT = 16;

var hashPool = null;
var hashPoolInitAttempted = false;
var hashPoolInitFailed = false;

function _ensureHashPool() {
	if(hashPool) return hashPool;
	if(hashPoolInitAttempted) {
		// Allow one retry after init failure
		if(hashPoolInitFailed) return null;
		hashPoolInitAttempted = false;
	}
	if(hashPoolInitAttempted) return null;
	hashPoolInitAttempted = true;
	try {
		var workers = [];
		for(var i = 0; i < HASH_POOL_SIZE; i++) {
			var w = new workerThreads.Worker(HASHER_WORKER_PATH);
			w.unref(); // Do not keep the event loop alive — see _shutdownHashPool
			w.busy = false;
			w.cb = null;
			(function(worker) {
				worker.on('message', function(msg) {
					var cb = worker.cb;
					worker.cb = null;
					worker.busy = false;
					if(msg && msg.error) {
						_destroyHashPool();
						if(cb) cb(new Error('Hasher worker error: ' + msg.error));
						_failHashPoolQueue(new Error('Hasher worker error: ' + msg.error));
						return;
					}
					if(cb) cb(null, msg);
					_dispatchFromQueueTo(worker);
				});
				worker.on('error', function(err) {
					_destroyHashPool();
					if(worker.cb) {
						var cbErr = worker.cb;
						worker.cb = null;
						cbErr(err);
					}
					_failHashPoolQueue(err);
				});
			})(w);
			workers.push(w);
		}
		hashPool = { workers: workers, next: 0, queue: [] };
		return hashPool;
	} catch(e) {
		hashPoolInitFailed = true;
		return null;
	}
}

function _destroyHashPool() {
	if(!hashPool) return;
	var oldWorkers = hashPool.workers;
	hashPool = null;
	oldWorkers.forEach(function(w) {
		try { w.terminate(); } catch(e) { /* best effort */ }
	});
}

function _shutdownHashPool() {
	if(!hashPool) return;
	var oldWorkers = hashPool.workers;
	hashPool = null;
	var terminations = oldWorkers.map(function(w) {
		return w.terminate().catch(function() { /* best effort */ });
	});
	return Promise.all(terminations);
}

process.on('beforeExit', function() {
	_shutdownHashPool();
});

// Unref stdio so the process can exit when stdin/stdout/stderr are pipes
// (CI runners, npm wrappers). Without this, the parent pipes keep the
// event loop alive even after workers are unref'd.
function _unrefStdio() {
	if(process.stdin && typeof process.stdin.unref === 'function' && process.stdin.readable) {
		process.stdin.unref();
	}
	if(process.stdout && typeof process.stdout.unref === 'function') {
		process.stdout.unref();
	}
	if(process.stderr && typeof process.stderr.unref === 'function') {
		process.stderr.unref();
	}
}
_unrefStdio();

function _failHashPoolQueue(err) {
	if(!hashPool) return;
	var queued = hashPool.queue;
	hashPool = null;
	while(queued.length > 0) {
		var item = queued.shift();
		if(item.cb) item.cb(err);
	}
}

function _dispatchFromQueueTo(worker) {
	if(!hashPool) return;
	while(hashPool.queue.length > 0 && !worker.busy) {
		var item = hashPool.queue.shift();
		worker.busy = true;
		worker.cb = item.cb;
		try {
			worker.postMessage({
				blockIndex: item.blockIndex,
				blockData: item.blockData,
				afterChecksum: item.afterChecksum
			}, [item.blockData]);
		} catch(e) {
			worker.busy = false;
			worker.cb = null;
			if(item.cb) item.cb(e);
		}
		return;
	}
}

function _inlineHashBlock(blockIndex, blockData, afterChecksum, cb) {
	try {
		var blockIndexBuf = Buffer.alloc(8);
		blockIndexBuf.writeBigUInt64LE(BigInt(blockIndex), 0);
		var fbHash = blake3.createHash()
			.update(afterChecksum)
			.update(blockIndexBuf)
			.update(blockData)
			.digest()
			.slice(0, 16);
		process.nextTick(function() {
			cb(null, { blockIndex: blockIndex, hash: fbHash });
		});
	} catch(e) {
		process.nextTick(function() { cb(e); });
	}
}

function _hashBlockParallel(blockIndex, blockData, afterChecksum, cb) {
	if(!HASH_POOL_ENABLED) { _inlineHashBlock(blockIndex, blockData, afterChecksum, cb); return; }
	if(hashPoolInitFailed) { _inlineHashBlock(blockIndex, blockData, afterChecksum, cb); return; }
	if(!_ensureHashPool()) { _inlineHashBlock(blockIndex, blockData, afterChecksum, cb); return; }
	var pool = hashPool;
	var found = null;
	for(var i = 0; i < pool.workers.length; i++) {
		var w = pool.workers[(pool.next + i) % pool.workers.length];
		if(!w.busy) { found = w; pool.next = (pool.next + i + 1) % pool.workers.length; break; }
	}
	if(found) {
		found.busy = true;
		found.cb = cb;
		try {
			found.postMessage({
				blockIndex: blockIndex,
				blockData: blockData,
				afterChecksum: afterChecksum
			}, [blockData]);
		} catch(e) {
			found.busy = false;
			found.cb = null;
			cb(e);
		}
		return;
	}
	pool.queue.push({
		blockIndex: blockIndex,
		blockData: blockData,
		afterChecksum: afterChecksum,
		cb: cb
	});
}

var friendlySize = function(s) {
	var units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB'];
	for(var i=0; i<units.length; i++) {
		if(s < 10000) break;
		s /= 1024;
	}
	return (Math.round(s *100)/100) + ' ' + units[i];
};

// GF64 method enum (matches gf64_global.h)
var GF64_METHODS = {
	AUTO: '',
	SCALAR: 'scalar',
	SSSE3: 'ssse3',
	AVX2: 'avx2',
	AVX512: 'avx512'
};

// Normalize path for comparison
var pathNormalize, pathToPar3;
if(path.sep == '\\') {
	pathNormalize = function(p) {
		return p.replace(/\//g, '\\').toLowerCase();
	};
	pathToPar3 = function(p) {
		return p.replace(/\\/g, '/');
	};
} else {
	pathToPar3 = pathNormalize = function(p) {
		return p;
	};
}

var sumSize = function(ar) {
	return ar.reduce(function(sum, e) {
		return sum + e.size;
	}, 0);
};

// PAR3 packet types
var PAR3_PKT_TYPE = {
	START: 'PAR STA\0',
	CREATOR: 'PAR CRE\0',
	FILE: 'PAR FIL\0',
	DIRECTORY: 'PAR DIR\0',
	DATA: 'PAR DAT\0',
	EXT_DATA: 'PAR EFD\0',
	MATRIX: 'PAR MAT\0',
	CAUCHY: 'PAR CAU\0',
	SPARSE: 'PAR SPR\0',
	EXPLICIT: 'PAR EXP\0',
	RECOVERY: 'PAR REC\0',
	EXT_REC: 'PAR ERD\0',
	UNIX_PERM: 'PAR UNX\0',
	FAT_PERM: 'PAR FAT\0',
	ROOT: 'PAR ROO\0'
};

// PAR3 constants
var PAR3_MAGIC = Buffer.from('PAR3\0PKT');
var PAR3_BLOCK_SIZE_DEFAULT = 1024 * 1024; // 1MB default block size
var PAR3_GF_SIZE = 64; // GF(2^64)

// Calculate friendly block count
var friendlyCount = function(n) {
	var units = ['', 'K', 'M', 'G', 'T'];
	for(var i=0; i<units.length; i++) {
		if(n < 10000) break;
		n /= 1000;
	}
	return (Math.round(n *100)/100) + units[i];
};

// ============================================================================
// PAR3 Packet Creation Helpers
// ============================================================================

function writeUInt64LE(buf, val, offset) {
	var lo = val >>> 0;
	var hi = Math.floor(val / 0x100000000);
	buf[offset] = lo & 0xff;
	buf[offset+1] = (lo >>> 8) & 0xff;
	buf[offset+2] = (lo >>> 16) & 0xff;
	buf[offset+3] = (lo >>> 24) & 0xff;
	buf[offset+4] = hi & 0xff;
	buf[offset+5] = (hi >>> 8) & 0xff;
	buf[offset+6] = (hi >>> 16) & 0xff;
	buf[offset+7] = (hi >>> 24) & 0xff;
}

function readUInt64LE(buf, offset) {
	var lo = buf.readUInt32LE(offset);
	var hi = buf.readUInt32LE(offset + 4);
	return lo + hi * 0x100000000;
}

// GF(2^64) inverse using extended Euclidean algorithm
// Polynomial: 0x100000000000001B
function invert64(val) {
	// GF(2^64) inverse using binary extended Euclidean algorithm
	// Polynomial: 0x100000000000001B
	val = val & 0xFFFFFFFFFFFFFFFFn;
	if(val === 0n) return 0n;
	if(val === 1n) return 1n;

	var u = val;
	var POLY = 0x1000000000000001Bn;  // x^64 + x^4 + x^3 + x + 1
	var v = POLY;
	var x1 = 1n;
	var x2 = 0n;

	while(u !== 1n && u !== 0n) {
		while((u & 1n) === 0n) {
			u >>= 1n;
			if((x1 & 1n) !== 0n) {
				x1 = ((x1 ^ POLY) >> 1n) & 0xFFFFFFFFFFFFFFFFn;
			} else {
				x1 >>= 1n;
			}
		}
		if(u === 1n) continue;  // Early exit: u is already 1
		while((v & 1n) === 0n) {
			v >>= 1n;
		}
		if(u < v) {
			var t = u; u = v; v = t;
			t = x1; x1 = x2; x2 = t;
		}
		u ^= v;
		x1 ^= x2;
	}
	return x1 & 0xFFFFFFFFFFFFFFFFn;
}

function gf64_mul(a, b) {
	// GF(2^64) multiplication using Russian Peasant algorithm
	// Polynomial: 0x100000000000001B
	var result = 0n;
	while(b !== 0n) {
		if((b & 1n) !== 0n) {
			result ^= a;
		}
		a <<= 1n;
		if((a & 0x10000000000000000n) !== 0n) {
			a ^= 0x1Bn;  // Low 64 bits of irreducible polynomial x^64 + x^4 + x^3 + x + 1
		}
		b >>= 1n;
	}
	return result & 0xFFFFFFFFFFFFFFFFn;
}

// PAR3 packet header size
var PAR3_PKT_HDR_SIZE = 48;

// Create PAR3 packet header
function createPacketHeader(type, bodySize, inputSetId) {
	var header = allocBuffer(PAR3_PKT_HDR_SIZE + bodySize);

	// Magic
	PAR3_MAGIC.copy(header, 0);

	// Checksum (placeholder - Blake3 would go here)
	header.fill(0, 8, 24);

	// Length
	writeUInt64LE(header, PAR3_PKT_HDR_SIZE + bodySize, 24);

	// Input Set ID
	if(typeof inputSetId === 'string') {
		// Take first 8 bytes of InputSetID hash
		inputSetId.copy(header, 32);
	} else if(Buffer.isBuffer(inputSetId)) {
		inputSetId.copy(header, 32);
	} else {
		header.fill(0, 32, 40);
	}

	// Type
	header.write(type, 40, 8, 'ascii');

	return header;
}

// Finalize PAR3 packet header with BLAKE3 checksum
// PAR3 spec: checksum = BLAKE3(totalLen + inputSetID + type + body)
// header is the full packet buffer (48-byte header + body), so take only
// the 24 post-checksum header bytes; the body is chained via .update(body).
function finalizePacketHeader(header, body) {
	var afterChecksum = header.slice(24, 48);
	var hash = blake3.createHash().update(afterChecksum).update(body).digest();
	hash.copy(header, 8, 0, 16);
	return header;
}

// Validate BLAKE3 checksum in a parsed packet header
// Returns true if checksum is zero (backward compat) or matches BLAKE3(totalLen+inputSetID+type+body)
function validatePacketChecksum(header, body) {
	var storedChecksum = header.slice(8, 24);
	var isZero = true;
	for(var i = 0; i < 16; i++) {
		if(storedChecksum[i] !== 0) { isZero = false; break; }
	}
	if(isZero) return true;
	var afterChecksum = header.slice(24);
	var computed = blake3.createHash().update(afterChecksum).update(body).digest().slice(0, 16);
	if(!storedChecksum.equals(computed)) {
		console.warn('WARNING: Packet checksum mismatch');
		return false;
	}
	return true;
}

// ============================================================================
// PAR3 Generator Class
// ============================================================================

function PAR3Gen(fileInfo, blockSize, opts) {
	if(!(this instanceof PAR3Gen))
		return new PAR3Gen(fileInfo, blockSize, opts);

	var o = this.opts = {
		outputBase: '',
		blockSize: blockSize || PAR3_BLOCK_SIZE_DEFAULT,
		recoverySlices: { unit: 'ratio', value: 0.1 }, // 10% default
		minRecoverySlices: null,
		maxRecoverySlices: null,
		recoveryOffset: 0,
		memoryLimit: null,
		minChunkSize: 1024 * 1024, // 1MB minimum
		processBatchSize: null,
		hashBatchSize: 8,
		comments: [],
		creator: 'ParPar/PAR3 v' + require('../package').version + ' [https://animetosho.org/app/parpar]',
		outputOverwrite: false,
		outputSync: false,
		outputIndex: true,
		outputFileCount: 0,
		seqReadSize: 4 * 1024 * 1024, // 4MB
		chunkReadThreads: 2,
		chunkReadThrottle: null,
		readBuffers: 8,
		readHashQueue: 5,
		numThreads: null,
		gfMethod: null, // auto-detect
		openclDevices: [],
		cpuMinChunkSize: 65536,
		matrixType: 'cauchy', // 'cauchy' or 'sparse'
	};

	if(opts) {
		for(var k in opts) {
			if(k in o) o[k] = opts[k];
		}
	}

	// Validate inputs
	if(!fileInfo || (typeof fileInfo != 'object'))
		throw new Error('No input files supplied');
	if(!fileInfo.length) fileInfo = [];

	var totalSize = 0, dataFiles = 0;
	fileInfo.forEach(function(file) {
		if(file.size == 0) return;
		totalSize += file.size;
		dataFiles++;
	});
	this.totalSize = totalSize;
	this.dataFiles = dataFiles;

	if(dataFiles > 0x7FFFFFFF)
		throw new Error('Too many input files');

	// Block size must be power of 2 and >= 1024
	if(o.blockSize < 1024)
		throw new Error('Block size must be at least 1024 bytes');
	if((o.blockSize & (o.blockSize - 1)) !== 0)
		throw new Error('Block size must be a power of 2');

	// Calculate number of blocks
	this.totalBlocks = Math.ceil(totalSize / o.blockSize);
	if(this.totalBlocks > 0x7FFFFFFF)
		throw new Error('Too many input blocks');

	// Calculate recovery slices
	o.recoverySlices = this._calcRecoverySlices(o.recoverySlices);
	if(o.minRecoverySlices !== null) {
		var minRec = this._calcRecoverySlices(o.minRecoverySlices);
		o.recoverySlices = Math.max(o.recoverySlices, minRec);
	}
	if(o.maxRecoverySlices !== null) {
		var maxRec = this._calcRecoverySlices(o.maxRecoverySlices);
		o.recoverySlices = Math.min(o.recoverySlices, maxRec);
	}

// GF64 method detection
	var gfInfo = this._detectGfMethod(o.gfMethod);
	this.gfMethod = gfInfo.id;
	this.gfMethodName = gfInfo.name;

	// Initialize encoder
	this.encoder = null;
	this.writer = null;

	var binding = getGf64Binding();
	if(binding && typeof binding.Gf64Encoder_create === 'function') {
		this.encoder = binding.Gf64Encoder_create(this.gfMethod, o.numThreads || 0);
	} else if(binding) {
		// Native addon exists but lacks Gf64Encoder_create (e.g. macOS arm64 stub
		// build where parpar_gf64 falls through to src/gf64_stub.cc and exports
		// an empty object). Mirror the T2 fallback pattern: warn and let the
		// rest of the pipeline use the pure-JS kernel (lib/gf64_js.js).
		process.stderr.write('Warning: native addon lacks Gf64Encoder_create; using pure-JS fallback\n');
		this.encoder = null;
	}

	// Generate display filenames
	fileInfo.forEach(function(file) {
		if(!('displayName' in file) && ('name' in file)) {
			file.displayName = pathToPar3(path.basename(file.name));
		}
	});
	this.files = fileInfo;

	// Create InputSetID
	this.inputSetId = this._computeInputSetId();

	// Output files
	this.recoveryFiles = [];
	this._initRecoveryFiles();
}

PAR3Gen.prototype = {
	encoder: null,
	writer: null,
	files: null,
	totalSize: 0,
	totalBlocks: 0,
	dataFiles: 0,
	gfMethod: 0,
	gfMethodName: 'auto',
	_outputStream: null,
	_outputPath: null,

	_calcRecoverySlices: function(spec) {
		if(typeof spec === 'number') return spec;
		if(!spec) return 0;

		var scale = spec.scale || 1;
		if(spec.unit === 'ratio') {
			return Math.round(scale * this.totalBlocks * spec.value);
		}
		if(spec.unit === 'slices' || spec.unit === 'count') {
			return Math.round(scale * spec.value);
		}
		if(spec.unit === 'bytes') {
			return Math.round(scale * (spec.value / this.opts.blockSize));
		}
		return Math.round(scale * spec.value);
	},

	_detectGfMethod: function(method) {
		var methods = ['scalar', 'ssse3', 'avx2', 'avx512'];
		var names = ['scalar', 'SSSE3', 'AVX2', 'AVX512'];

		if(!method || method === 'auto' || method === '') {
			// Use startup microbenchmark to pick the fastest SIMD method.
			// On Zen4, CPUID picks AVX-512 but AVX2 is often faster due
			// to the double-pump penalty. The microbench measures actual
			// throughput and switches only when >= 5% difference.
			var bench = require('./gf_method_bench');
			var result = bench.pickBestMethod();
			if(result.method >= 0) {
				// Map bench method name (C++ enum: 0=avx512,1=avx2,2=ssse3,3=scalar)
				// to the par3gen internal index via the methods array.
				var idx = methods.indexOf(result.name);
				if(idx >= 0) {
					return { id: idx, name: names[idx] };
				}
			}
			// Fallback: CPU features
			if(process.arch === 'x64') {
				return { id: 2, name: 'AVX2' }; // Default to AVX2 on x64
			}
			return { id: 0, name: 'scalar' };
		}

		var idx = methods.indexOf(method);
		if(idx < 0) idx = 0;
		return { id: idx, name: names[idx] };
	},

	_computeInputSetId: function() {
		// PAR3 InputSetID is first 8 bytes of BLAKE3(Start packet body)
		// Compute from file info using BLAKE3
		var hash = blake3.createHash();
		this.files.forEach(function(file) {
			hash.update(Buffer.from(file.name, 'utf8'));
			var sizeBuf = Buffer.alloc(8);
			sizeBuf.writeBigUInt64LE(BigInt(file.size), 0);
			hash.update(sizeBuf);
		});
		return hash.digest().slice(0, 8);
	},

	_initRecoveryFiles: function() {
		var o = this.opts;
		var totalRecSlices = o.recoverySlices + o.recoveryOffset;

		if(totalRecSlices <= 0) return;

		if(o.outputFileCount > totalRecSlices)
			throw new Error('Cannot allocate ' + totalRecSlices + ' recovery slices to ' + o.outputFileCount + ' volumes');

		// Simple scheme: divide recovery slices evenly
		var numFiles = o.outputFileCount || Math.ceil(totalRecSlices / 65535);
		var slicesPerFile = Math.ceil(totalRecSlices / numFiles);

		var sliceOffset = 0;
		for(var i = 0; i < numFiles && sliceOffset < totalRecSlices; i++) {
			var slices = Math.min(slicesPerFile, totalRecSlices - sliceOffset);
			var filename = o.outputBase + '.par3';
			if(i > 0 || o.outputFileCount > 1) {
				var digits = String(numFiles).length;
				var numStr = String(i).padStart(digits, '0');
				filename = o.outputBase + '.vol' + numStr + '-*.par3';
			}

			this.recoveryFiles.push({
				name: filename.replace('*', String(sliceOffset).padStart(digits, '0')),
				sliceOffset: sliceOffset,
				numSlices: slices,
				data: null
			});
			sliceOffset += slices;
		}
	},

	// Encode data using GF(2^64)
	_encodeBlocks: function(inputData, coefficients, cb) {
		if(!this.encoder) {
			process.nextTick(function() {
				cb(new Error('GF64 encoder not available'));
			});
			return;
		}

		try {
			var output = allocBuffer(inputData.length);
			var len = Math.floor(inputData.length / 8);
			var nCoeff = Math.floor(coefficients.length / 8);
			this.encoder.mul_arr(output, inputData, coefficients, len, nCoeff);
			cb(null, output);
		} catch(e) {
			cb(e);
		}
	},

	// Create PAR3 packets
	_createStartPacket: function() {
		// PAR3 START body layout (matches the repair read offsets in this file):
		//   body[0]      : gf_size (uint8) — read by repair as `body[offset]`
		//   body[1-15]   : reserved (15 bytes, zero-filled — the "unique" field)
		//   body[16-23]  : block_size (uint64 LE) — repair reads this with readUInt64LE(body, offset+16)
		//   body[24-27]  : block_pow (uint32 LE) — repair reads this with readUInt32LE(offset+24)
		//   body[28-29]  : gf_size (uint16 LE)
		//   body[30-33]  : generator (4 bytes — default 0x1D)
		var bodySize = 34;
		var header = createPacketHeader(PAR3_PKT_TYPE.START, bodySize, this.inputSetId);

		// The previous code wrote blockSize at body offset 24 as uint32, but the repair
		// code reads blockSize at body offset 16 as uint64 (line ~1607). With blockSize=0,
		// every DATA/RECOVERY packet was read as a 0-byte buffer and the "no missing blocks"
		// branch always ran. Write blockSize as uint64 at the body offset the repair code reads.
		header[48] = 8;                                                    // body[0]  = gf_size (uint8)
		writeUInt64LE(header, this.opts.blockSize, 48 + 16);               // body[16] = block_size (uint64 LE) <- FIX
		header.writeUInt32LE(Math.log2(this.opts.blockSize) | 0, 48 + 24); // body[24] = block_pow (uint32)
		header.writeUInt16LE(8, 48 + 28);                                  // body[28] = gf_size (uint16)
		header.writeUInt32LE(0x1D, 48 + 30);                               // body[30] = generator (4 bytes)

		finalizePacketHeader(header, header.slice(48));
		return header;
	},

	_createFilePackets: function() {
		var packets = [];
		var self = this;

		this.files.forEach(function(file, idx) {
			var nameBuf = Buffer.from(file.displayName, 'utf8');
			var bodySize = 4 + nameBuf.length + 16 + 8 + 8 + 4;

			var header = createPacketHeader(PAR3_PKT_TYPE.FILE, bodySize, self.inputSetId);

			header.writeUInt16LE(nameBuf.length, 48);
			header.writeUInt16LE(0, 50);
			nameBuf.copy(header, 52);

			var fileId = crypto.randomBytes(16);
			file.fileId = fileId;
			fileId.copy(header, 52 + nameBuf.length);

			writeUInt64LE(header, file.size, 52 + nameBuf.length + 16);
			writeUInt64LE(header, 0, 52 + nameBuf.length + 24);
			header.writeUInt32LE(file.mode || 0x81A4, 52 + nameBuf.length + 32);

			finalizePacketHeader(header, header.slice(48, 48 + bodySize));

			packets.push({ type: 'file', data: header });

			if((file.mode || 0x81A4) !== 0x81A4) {
				packets.push({ type: 'perm', data: self._createPermissionsPacket(file) });
			}
		});

		return packets;
	},

	_createMatrixPacket: function() {
		var bodySize = 40;
		var header = createPacketHeader(PAR3_PKT_TYPE.CAUCHY, bodySize, this.inputSetId);

		writeUInt64LE(header, 0, 48);
		writeUInt64LE(header, this.totalBlocks > 0 ? this.totalBlocks - 1 : 0, 56);
		writeUInt64LE(header, this.totalBlocks, 64);
		writeUInt64LE(header, this.opts.recoverySlices, 72);
		header.writeUInt32LE(this.opts.recoverySlices, 80);
		header.writeUInt32LE(this.totalBlocks, 84);

		finalizePacketHeader(header, header.slice(48, 88));
		return { type: 'matrix', data: header };
	},

	_createCreatorPacket: function() {
		var creatorStr = this.opts.creator;
		if (typeof creatorStr !== 'string') {
			creatorStr = 'ParPar/PAR3 v' + require('../package').version + ' [https://animetosho.org/app/parpar]';
		}
		var nameBuf = Buffer.from(creatorStr, 'utf8');
		var bodySize = nameBuf.length;
		var header = createPacketHeader(PAR3_PKT_TYPE.CREATOR, bodySize, this.inputSetId);
		nameBuf.copy(header, 48);
		finalizePacketHeader(header, header.slice(48));
		return header;
	},

	_createRootPacket: function(filePackets) {
		var bodySize = 8 + 1 + 4 + (filePackets ? filePackets.length * 16 : 0);
		var header = createPacketHeader(PAR3_PKT_TYPE.ROOT, bodySize, this.inputSetId);

		// Lowest unused input block index (= totalBlocks)
		writeUInt64LE(header, this.totalBlocks, 48);

		// Attributes: 0 = relative path
		header[56] = 0;

		// Number of options (links) = 0
		header.writeUInt32LE(0, 57);

		// Checksums of File/Directory packets
		if (filePackets) {
			for (var i = 0; i < filePackets.length; i++) {
				// Copy 16-byte fingerprint hash from packet header bytes [8..23]
				filePackets[i].data.copy(header, 61 + i * 16, 8, 24);
			}
		}

		finalizePacketHeader(header, header.slice(48));
		return header;
	},

	_createPermissionsPacket: function(file) {
		// UNIX Permissions packet: contains file mode and ownership (Amendment 16)
		// body layout: File ID (16 bytes) + permission bits (4 bytes)
		var bodySize = 20;
		var header = createPacketHeader(PAR3_PKT_TYPE.UNIX_PERM, bodySize, this.inputSetId);

		// File ID (16 bytes) at body[0..15]
		if(file.fileId) {
			file.fileId.copy(header, 48);
		}

		// Permission bits (4 bytes) at body[16..19]
		header.writeUInt32LE(file.mode || 0x81A4, 64);

		finalizePacketHeader(header, header.slice(48, 68));
		return header;
	},

	_createRecoveryPackets: function(recoveryData, cb) {
		var packets = [];
		var self = this;

		async.eachSeries(recoveryData, function(rec, cb) {
			var data = rec.data;
			var blockIndex = rec.blockIndex;
			var bodySize = 16 + data.length; // first_recovery_block + block_count + data
			var header = createPacketHeader(PAR3_PKT_TYPE.RECOVERY, bodySize, self.inputSetId);

			writeUInt64LE(header, blockIndex, 48);
			writeUInt64LE(header, 1, 56); // block_count = 1

			var fullPacket = header;
			data.copy(fullPacket, PAR3_PKT_HDR_SIZE + 16);
			finalizePacketHeader(header, header.slice(48));
			packets.push({ type: 'recovery', data: fullPacket });
			cb();
		}, function(err) {
			cb(err, packets);
		});
	},

	// Process a batch of input blocks: compute partial recovery for the batch
	// and XOR the results into `accumulator`. This is the core of the
	// batched recovery flow -- instead of holding every input block in
	// memory, we accumulate them in BATCH_SIZE groups, computing partial
	// recovery and XOR-combining into a single accumulator buffer of size
	// numRecovery * blockSize. By linearity, XOR-combining partial results
	// is mathematically identical to a single pass over all inputs.
	//
	// Parameters:
	//   batch           - Array<Buffer> of input blocks (each blockSize bytes)
	//   firstInputIdx   - BigInt: the global index of the first block in this batch
	//   firstRecovery   - BigInt: global index of the first recovery block
	//   numRecovery     - Number of recovery blocks to compute
	//   accumulator     - Buffer(numRecovery * blockSize) -- partial results are XORed in
	_processRecoveryBatch: function(batch, firstInputIdx, firstRecovery, numRecovery, accumulator) {
		var self = this;
		var o = this.opts;
		var numBatch = batch.length;
		var blockSize = o.blockSize;

		if(numBatch === 0 || numRecovery === 0) return;

		var useJsKernel = process.env.PAR3_USE_JS_KERNEL === '1';
		var totalRecoverySize = numRecovery * blockSize;
		var accumSize = accumulator.length;

		var hasNativeEncoder = this.encoder && typeof this.encoder.mul_arr === 'function';
		if(!useJsKernel) {
			var binding = getGf64Binding();
			// Stub build (macOS arm64, Windows) loads an empty binding object —
			// fall back to the JS kernel for any missing compute_recovery function,
			// mirroring the Gf64Encoder_create + gf64_info typeof guards.
			if(!binding || typeof binding.compute_recovery !== 'function') useJsKernel = true;
		}
		if(useJsKernel) {
			var mulArrFn;
			if(hasNativeEncoder) {
				mulArrFn = this.encoder.mul_arr.bind(this.encoder);
			} else {
				mulArrFn = gf64Js.mul_arr;
			}
			global.__par3ProfStart('matrix');
			var coeffMatrix = new Array(numRecovery);
			for(var recIdx = 0; recIdx < numRecovery; recIdx++) {
				// JS-kernel fallback: convert the Number indices to BigInt here
				// (the native path passes them straight through to NAPI, where
				// get_uint64_from_value accepts plain Numbers).
				var yi = BigInt(firstRecovery) + BigInt(recIdx);
				var row = new Array(numBatch);
				for(var batchIdx = 0; batchIdx < numBatch; batchIdx++) {
					var xj = BigInt(firstInputIdx) + BigInt(batchIdx);
					var denom = xj ^ yi;
					if(denom === 0n) denom = 1n;
					row[batchIdx] = invert64(denom);
				}
				coeffMatrix[recIdx] = row;
			}
			global.__par3ProfEnd('matrix');
			global.__par3ProfStart('kernel');
			var bufferCount = Math.min(numRecovery, 4);
			var tmpBuffers = new Array(bufferCount);
			for(var b = 0; b < bufferCount; b++) {
				tmpBuffers[b] = allocBuffer(blockSize);
			}
			var coeff1 = Buffer.alloc(8);
			var len = Math.floor(blockSize / 8);

			for(var batchIdx = 0; batchIdx < numBatch; batchIdx++) {
				var inputBlock = batch[batchIdx];
				var accOff = 0;
				for(var k = 0; k < numRecovery; k++) {
					if(accOff + blockSize > accumSize) break;
					var tmp = tmpBuffers[k % bufferCount];
					coeff1.writeBigUInt64LE(coeffMatrix[k][batchIdx], 0);
					mulArrFn(tmp, inputBlock, coeff1, len, 1);
					for(var i = 0; i < blockSize; i++) {
						accumulator[accOff + i] ^= tmp[i];
					}
					accOff += blockSize;
				}
			}
			global.__par3ProfEnd('kernel');
		} else {
		global.__par3ProfStart('kernel');
			var binding = getGf64Binding();
			var concatBatch = getConcatBuffer(numBatch * blockSize);
			var offset = 0;
			for(var b = 0; b < batch.length; b++) {
				var blk = batch[b];
				blk.copy(concatBatch, offset);
				offset += blk.length;
			}
			// P1.1 fix: when totalRecoverySize exceeds the in-memory cap
			// (MAX_IN_MEMORY = 512 MiB), the per-batch path cannot complete in
			// a single native compute_recovery call -- the binding expects an
			// output buffer of numRecovery * blockSize bytes, but our partial
			// buffer is clipped to at most accumulator.length. To keep the
			// binding's contract honest, shrink numRecovery for this single call
			// so that partialSize = numRecoveryThisCall * blockSize exactly,
			// and stream the full numRecovery in successive chunks. Each
			// chunk accumulates into the in-memory accumulator at the same
			// starting firstRecovery + chunkIndex offset so the final recovery
			// packet set is identical to a single-shot call.
			var maxChunkBytes = Math.min(totalRecoverySize, accumulator.length);
			var numRecoveryThisCall = Math.floor(maxChunkBytes / blockSize);
			if(numRecoveryThisCall <= 0) numRecoveryThisCall = 0;
			if(numRecoveryThisCall > numRecovery) numRecoveryThisCall = numRecovery;
			var partialSize = numRecoveryThisCall * blockSize;
			var partial = allocBuffer(partialSize);
			var firstRecoveryThisChunk = firstRecovery;

			// T12: per-batch path passes compute_recovery as the explicit fallback so
			// the dispatcher's Barycentric gate is applied here too. Without the
			// explicit fallback the dispatcher would fall back to
			// compute_recovery_full, which writes a different output buffer size.
			dispatchRecovery(
				binding,
				concatBatch,
				partial,
				numBatch,
				numRecoveryThisCall,
				blockSize,
				firstInputIdx,
				firstRecoveryThisChunk,
				this.opts.numThreads || 0,
				binding.compute_recovery ? binding.compute_recovery.bind(binding) : undefined
			);

			var partWords = partialSize >> 3;
			if(partWords > 0) {
				var accView = new BigInt64Array(accumulator.buffer, accumulator.byteOffset, partWords);
				var partView = new BigInt64Array(partial.buffer, partial.byteOffset, partWords);
				for(var i = 0; i < partWords; i++) {
					accView[i] ^= partView[i];
				}
			}
			releaseConcatBuffer(concatBatch);
			global.__par3ProfEnd('kernel');

			// If we couldn't fit all numRecovery in one chunk, recurse for the
			// remainder. This preserves correctness when totalRecoverySize
			// exceeds MAX_IN_MEMORY on the per-batch path. The Cauchy
			// coefficients depend on firstRecovery, so advancing it by
			// numRecoveryThisCall for the next chunk is correct.
			if(numRecoveryThisCall < numRecovery) {
				var remaining = numRecovery - numRecoveryThisCall;
				var nextFirstRecovery = firstRecoveryThisChunk + numRecoveryThisCall;
				// Self-call via the method table: _processRecoveryBatch is
				// declared on PAR3Gen.prototype and not as a local in this
				// scope, so we route through self's method-table lookup.
				self._processRecoveryBatch(batch, firstInputIdx, nextFirstRecovery, remaining, accumulator);
			}
		}
	},

	// Build recovery packets from a finalised accumulator buffer.
	// `accumulator` is a Buffer of size numRecovery * blockSize, or a file path
	// string if the total recovery data exceeded the 4 GiB Buffer limit.
	_finalizeRecoveryBlocks: function(accumulator, numInput, eventCb, cb) {
		var self = this;
		var o = this.opts;
		var numRecovery = o.recoverySlices;
		var blockSize = o.blockSize;

		if(numRecovery === 0 || numInput === 0 || !accumulator) {
			return process.nextTick(function() { cb(null); });
		}

		var recoveryBlocks = [];
		if(typeof accumulator === 'string') {
			var fd = fs.openSync(accumulator, 'r');
			var tmpBuf = allocBuffer(blockSize);
			for(var r = 0; r < numRecovery; r++) {
				fs.readSync(fd, tmpBuf, 0, blockSize, r * blockSize);
				recoveryBlocks.push({
					blockIndex: numInput + r,
					data: Buffer.from(tmpBuf)
				});
			}
			fs.closeSync(fd);
			self._createRecoveryPackets(recoveryBlocks, function(err, packets) {
				fs.unlinkSync(accumulator);
				if(err) return cb(err);
				writePackets(packets);
			});
		} else {
			for(var r = 0; r < numRecovery; r++) {
				recoveryBlocks.push({
					blockIndex: numInput + r,
					data: bufferSlice.call(accumulator, r * blockSize, (r + 1) * blockSize)
				});
			}
			self._createRecoveryPackets(recoveryBlocks, function(err, packets) {
				if(err) return cb(err);
				writePackets(packets);
			});
		}

		function writePackets(packets) {
			var pktIdx = 0;
			function writeNext() {
				if(pktIdx >= packets.length) {
					eventCb('recovery_complete', { recoveryBlocks: numRecovery });
					return cb(null);
				}
				var ok = self._outputStream.write(packets[pktIdx].data);
				pktIdx++;
				if(!ok) {
					self._outputStream.once('drain', writeNext);
				} else {
					setImmediate(writeNext);
				}
			}
			writeNext();
		}
	},

	// Get GF method info
	gf_info: function() {
		return {
			method: this.gfMethod,
			methodName: this.gfMethodName,
			available: !!this.encoder
		};
	},

	// Run PAR3 generation
run: function(eventCb, completeCb) {
	var self = this;
	var o = this.opts;

	// T8/v2 profiling: opt-in via PAR3_PROFILE=1; per-phase wall-clock
	// accumulators in milliseconds. Print at run end so the operator can
	// see which phase dominates. Globals so the per-batch path's
	// _processRecoveryBatch (called BEFORE this closure captures locals)
	// can also write into the same buckets.
	if (typeof global.__par3Profile === 'undefined') {
		global.__par3Profile = {
			enabled: process.env.PAR3_PROFILE === '1',
			start: 0,
			phases: null,
			_t: 0n,
			reset: function() {
				this.start = Date.now();
				this.phases = this.enabled ? {
					setup: 0, read: 0, hash: 0, matrix: 0, kernel: 0, write: 0, finalize: 0
				} : null;
			}
		};
		global.__par3Profile.reset();
	}
	// Hoist profStart/profEnd to module scope via the global, so all
	// internal functions (incl. _processRecoveryBatch on the per-batch
	// path) can call them. We attach the helpers as global functions.
	function _profStart(name) {
		if (!global.__par3Profile.enabled) return;
		global.__par3Profile._t = process.hrtime.bigint();
	}
	function _profEnd(name) {
		if (!global.__par3Profile.enabled) return;
		var dt = Number(process.hrtime.bigint() - global.__par3Profile._t) / 1e6;
		global.__par3Profile.phases[name] += dt;
	}
	// Expose for _processRecoveryBatch etc.
	process.env.PAR3_PROFILE_HELPER_INSTALLED = '1';
	// Use the globals as aliases so the existing in-run() call sites work.
	var PROFILE = global.__par3Profile.enabled;
	var profilePhases = global.__par3Profile.phases;
	function profStart(name) { return _profStart(name); }
	function profEnd(name) { return _profEnd(name); }
	// Also patch the in-file references to call _profStart/_profEnd
	// directly — but that requires touching every call site. The simpler
	// workaround: have the per-batch path's _processRecoveryBatch call
	// the globals by string name.
	global.__par3ProfStart = _profStart;
	global.__par3ProfEnd = _profEnd;

	eventCb('begin', this);

	global.__par3ProfStart('setup');
		var outputPath = o.outputBase + '.par3';
		this._outputPath = outputPath;
		this._outputStream = fs.createWriteStream(outputPath);
		this._outputStream.on('error', function(err) {
			eventCb('error', err);
		});
		this._outputStream.on('error', function(err) {
			self._streamError = err;
		});

		if(o.creator) {
			this._outputStream.write(self._createCreatorPacket());
		}
		this._outputStream.write(self._createStartPacket());
		var matrixPkt = self._createMatrixPacket();
		this._outputStream.write(matrixPkt.data);
		var filePackets = self._createFilePackets();
		filePackets.forEach(function(pkt) {
			self._outputStream.write(pkt.data);
		});
		this._outputStream.write(self._createRootPacket(filePackets));

		// Amendment 18: empty input set — C++ binding rejects zero input blocks.
		if(this.totalBlocks === 0) {
			eventCb('start', { totalBlocks: 0 });
			self._outputStream.end(function() {
				self._outputStream = null;
				eventCb('complete', { processedBlocks: 0 });
				completeCb(null);
			});
			return;
		}

		// T1: auto-size the input buffer pool to the workload (lazy-init on
		// first create call). Uses totalInputBlocks * blockSize with a 64 MiB
		// floor; PAR3_GF64_INPUT_POOL_SIZE env var overrides the computed value.
		_ensureInputPool(this.totalBlocks * o.blockSize);

		// A3: reset the input buffer pool offset for this create call. The
		// pool is append-only within a create; resetting here ensures the
		// previous run's leftover state doesn't bleed into this one.
		_resetInputBufferPool();

		var processedBlocks = 0;
		var totalBlocks = this.totalBlocks;
		var blockSize = o.blockSize;
		var numRecovery = o.recoverySlices;
		var totalInputBlocks = this.totalBlocks;
		// Number-safe: totalInputBlocks <= 100k for the target workloads,
		// far below 2^53. Numbers avoid BigInt marshalling in the NAPI
		// bindings (get_uint64_from_value accepts both).
		var firstRecovery = totalInputBlocks;

		var MAX_IN_MEMORY = 512 * 1024 * 1024;
		var totalRecoverySize = numRecovery * blockSize;
		var tmpFilePath = null;
		if(totalRecoverySize > 4294967296) {
			tmpFilePath = path.join(path.dirname(this._outputPath), '.recovery.tmp');
		}

		// T8 (par3-create-throughput-400mbps): pick single-call vs. per-batch
		// path. Per-batch is used when PAR3_BATCH_SIZE is set (env-override),
		// PAR3_USE_JS_KERNEL=1 (the new path is C++-only), the binding lacks
		// compute_recovery_full (pre-T7 build), or totalRecoverySize exceeds
		// the 512 MiB accumulator cap (single C++ call writes numRecovery *
		// blockSize bytes; existing temp-file spill stays on the per-batch
		// path for >=4 GiB cases).
		var useComputeRecoveryFull = false;
		var computeRecoveryFullBinding = null;
		if(!process.env.PAR3_BATCH_SIZE &&
		   process.env.PAR3_USE_JS_KERNEL !== '1' &&
		   totalRecoverySize <= MAX_IN_MEMORY) {
			computeRecoveryFullBinding = getGf64Binding();
			if(computeRecoveryFullBinding && typeof computeRecoveryFullBinding.compute_recovery_full === 'function') {
				useComputeRecoveryFull = true;
			}
		}

		// B1 (par3-1200mbps): streaming opt-in via PAR3_GF64_FAST_CREATE=1,
		// escape hatch via PAR3_GF64_LEGACY_CREATE=1. The streaming NAPI
		// (A2's par3_create_streaming) is called per source file alongside
		// the legacy path below. NOTE: A2's C++ binding computes recovery in
		// C++ but frees the buffer before invoking the callback (only the
		// byte count + throughput are returned for telemetry), so the legacy
		// compute_recovery_full path remains the source of truth for the
		// actual archive recovery data — the streaming call is exercised as
		// a wire-up + throughput-measurement step, not a replacement. Once
		// A2-rev exposes the recovery buffer, the legacy call can be removed
		// and the streaming result fed straight into _finalizeRecoveryBlocks.
		var useStreamingCreate = false;
		if(useComputeRecoveryFull &&
		   process.env.PAR3_GF64_FAST_CREATE === '1' &&
		   process.env.PAR3_GF64_LEGACY_CREATE !== '1' &&
		   computeRecoveryFullBinding &&
		   typeof computeRecoveryFullBinding.par3_create_streaming === 'function') {
			useStreamingCreate = true;
		}

		var accumulator = null;
		var recoveryFlushedBytes = 0;
		function ensureAccumulator() {
			if(accumulator !== null) return;
			// B2: pool-backed acquire (LRU); acquireRecoveryBuffer zero-fills
			// on pool reuse per the zero-init invariant.
			accumulator = acquireRecoveryBuffer(Math.min(totalRecoverySize, MAX_IN_MEMORY));
		}

		function flushCompletedBlocks() {
			if(!tmpFilePath || !accumulator) return;
			var accBytes = accumulator.length;
			if(accBytes <= recoveryFlushedBytes) return;
			var newBytes = accBytes - recoveryFlushedBytes;
			var flushStream = fs.createWriteStream(tmpFilePath, { flags: recoveryFlushedBytes === 0 ? 'w' : 'a' });
			flushStream.write(accumulator.slice(recoveryFlushedBytes, accBytes));
			flushStream.end();
			recoveryFlushedBytes = accBytes;
		}

		if(useComputeRecoveryFull) {
			// Single-call path: full input buffer + one compute_recovery_full
			// call, then reuse _finalizeRecoveryBlocks / packet streaming.
			var fullInputs = allocBuffer(totalInputBlocks * blockSize);
			var nextBlockIdx = 0;
			var bindingRef = computeRecoveryFullBinding;

			// v2-4: kick off the Cauchy matrix build in a worker thread BEFORE
			// the file read begins, so the matrix-build cost overlaps with the
			// read phase. The worker returns a Buffer (the matrix); we await
			// it inside finalizeRecovery().
			var precomputedCoeff = null;
			var precomputedCoeffErr = null;
			var precomputedCoeffDone = false;
			var coeffWorker = null;
			if (bindingRef && typeof bindingRef.build_coefficient_matrix === 'function'
			                  && typeof bindingRef.compute_recovery_with_coeff === 'function') {
				try {
					var Worker = require('worker_threads').Worker;
					// Inline-worker source: a single block of JS that calls the
					// native binding and posts the Buffer back. Using a Worker
					// (rather than a Promise) gives us true OS-thread parallelism
					// for the matrix build, which is CPU-bound and ~30s on the
					// 1 GiB / 10K workload.
					var workerSrc = "var w={exports:{}};var __b=require('" + path.join(__dirname, '..', 'build', 'Release', 'parpar_gf64.node').replace(/\\\\/g, '/') + "');"
						+ "process.parentPort.on('message', function(m){"
						+ "  try {"
						+ "    var c = __b.build_coefficient_matrix(m.numInputs, m.numRecovery, m.firstInput, m.firstRecovery);"
						+ "    process.parentPort.postMessage({ok: true, matrix: c});"
						+ "  } catch(e) {"
						+ "    process.parentPort.postMessage({ok: false, err: e.message});"
						+ "  }"
						+ "});";
					coeffWorker = new Worker(workerSrc, { eval: true });
					coeffWorker.on('message', function(msg) {
						if (msg.ok) precomputedCoeff = msg.matrix;
						else precomputedCoeffErr = msg.err;
						precomputedCoeffDone = true;
					});
					coeffWorker.on('error', function(err) {
						precomputedCoeffErr = err.message;
						precomputedCoeffDone = true;
					});
					coeffWorker.postMessage({
						numInputs: totalInputBlocks,
						numRecovery: numRecovery,
						firstInput: 0,
						firstRecovery: totalInputBlocks
					});
				} catch (e) {
					precomputedCoeffErr = e.message;
					precomputedCoeffDone = true;
				}
			} else {
				precomputedCoeffDone = true;
			}

			var finalizeRecovery = function() {
				ensureAccumulator();
				global.__par3ProfEnd('read');
				// v2-4 (Todo 2 ci-flow-fix): do NOT block the main thread here.
			// The previous implementation called Atomics.wait(i32, 0, 0) on a
			// SAB that was never posted to the worker, so the wake-up handler
			// (attached on the same blocked thread) could never fire — a
			// classic self-deadlock that hung every tiny create. Instead: trust
			// the racing worker. If it finished, use the precomputed matrix;
			// otherwise fall back to the inline build path. Either way, the
			// worker is no longer needed once compute is done, so terminate it
			// exactly once — single-point cleanup, .catch(()=>{}) on the
			// promise (never await on the hot path), set the local to null
			// immediately to make the call idempotent.
			function cleanupCoeffWorker() {
				var w = coeffWorker;
				if (!w) return;
				coeffWorker = null;
				try {
					var p = w.terminate();
					if (p && typeof p.catch === 'function') p.catch(function() {});
				} catch (e) { /* best effort */ }
			}
			global.__par3ProfStart('kernel');
			try {
				// If the matrix-build worker finished, use the pre-computed
				// matrix; otherwise fall back to the inline build path.
				if (precomputedCoeff && typeof bindingRef.compute_recovery_with_coeff === 'function') {
					bindingRef.compute_recovery_with_coeff(
						precomputedCoeff,
						fullInputs,
						accumulator,
						totalInputBlocks,
						numRecovery,
						blockSize,
						0,
						totalInputBlocks,
						self.opts.numThreads || 0
					);
				} else {
					// T12: dispatch through Barycentric kernel for large workloads.
					dispatchRecovery(
						bindingRef,
						fullInputs,
						accumulator,
						totalInputBlocks,
						numRecovery,
						blockSize,
						0,
						totalInputBlocks,
						self.opts.numThreads || 0
					);
				}
			} catch (e) {
				cleanupCoeffWorker();
				throw e;
			}
			cleanupCoeffWorker();
			global.__par3ProfEnd('kernel');
				flushCompletedBlocks();
				eventCb('generating_recovery', { inputBlocks: totalInputBlocks, recoveryBlocks: numRecovery });
				var finalAcc = tmpFilePath || accumulator;
				self._finalizeRecoveryBlocks(finalAcc, totalInputBlocks, eventCb, function(err) {
					// B2: release accumulator back to the pool (data already .copy()'d into packets).
					releaseRecoveryBuffer(accumulator);
					self._outputStream.end(function() {
						self._outputStream = null;
						eventCb('complete', { processedBlocks: processedBlocks });
						// v2: profile dump (PAR3_PROFILE=1) -- single-call path
						if (PROFILE) {
							var totalMs = Date.now() - (global.__par3Profile.start || Date.now());
							var keys = Object.keys(profilePhases).filter(function(k){return k!=='_t';});
							var lines = ['[PAR3_PROFILE] create took ' + totalMs + 'ms total'];
							keys.forEach(function(k){
								var pct = totalMs > 0 ? (profilePhases[k]/totalMs*100).toFixed(1) : '0.0';
								lines.push('  ' + k.padEnd(8) + ' ' + profilePhases[k].toFixed(1).padStart(8) + 'ms  (' + pct + '%)');
							});
							try {
								var fs = require('fs');
								var dumpPath = path.join(process.cwd(), '.par3-profile-dump.txt');
								fs.writeFileSync(dumpPath, lines.join('\n') + '\n');
							} catch (e) { /* ignore */ }
						}
						completeCb(err);
					});
				});
				};

			var processFileFull = function(idx) {
				if(idx >= self.files.length) {
					finalizeRecovery();
					return;
				}

				var file = self.files[idx];
				if(!file.size) {
					// A3: pool reset between files keeps the offset bounded
					// for multi-file creates with small per-file read totals.
					_resetInputBufferPool();
					processFileFull(idx + 1);
					return;
				}

				eventCb('processing_file', file, idx);

				// A3: reset before reading the next file so the pool offset
				// doesn't accumulate across files in a multi-file create.
				_resetInputBufferPool();

				// B1: when streaming is opted in, fire per-file streaming call
				// for throughput telemetry (fire-and-forget — legacy path below
				// is the source of truth). Sets PAR3_GF64_USE_MMAP=1 (A1's gate)
				// for files <= 2 GiB; cleared in the callback to avoid leaking
				// the mmap choice into subsequent calls.
				if(useStreamingCreate && file.name) {
					var streamingUseMmap = file.size <= (2 * 1024 * 1024 * 1024);
					if(streamingUseMmap) process.env.PAR3_GF64_USE_MMAP = '1';
					try {
						bindingRef.par3_create_streaming(file.name, {
							recoverySlices: numRecovery,
							blockSize: blockSize,
							firstInput: 0,
							firstRecovery: totalInputBlocks,
							numThreads: self.opts.numThreads || 0
						}, function(streamErr, streamResult) {
							if(streamingUseMmap) delete process.env.PAR3_GF64_USE_MMAP;
							// Telemetry only — legacy path produces the actual recovery data.
						});
					} catch(streamEx) {
						if(streamingUseMmap) delete process.env.PAR3_GF64_USE_MMAP;
						// Telemetry only — non-fatal.
					}
				}

				var fd;
				var remaining = file.size;
				var bulkFileBuf;

				// W2-T1: single bulk read replaces per-block readSync loop
				try {
					fd = fs.openSync(file.name, 'r');
					var fileStat = fs.fstatSync(fd);
					bulkFileBuf = Buffer.alloc(fileStat.size);
					fs.readSync(fd, bulkFileBuf, 0, fileStat.size, 0);
					fs.closeSync(fd);
					fd = null;
					bulkFileBuf.copy(fullInputs, 0, 0, Math.min(bulkFileBuf.length, fullInputs.length));
					remaining = bulkFileBuf.length;
				} catch(e) {
					if(fd) { try{fs.closeSync(fd);}catch(e2){} }
					bulkFileBuf = null;
					fd = fs.openSync(file.name, 'r');
					remaining = file.size;
				}
				readBlockFull();

				function readBlockFull() {
					// B3: batch reads + dispatch hashes in parallel (batchSize=poolSize)
					// when enabled; otherwise batchSize=1 preserves one-block-at-a-time
					// behavior.
			var batchSize = HASH_POOL_ENABLED ? HASH_POOL_SIZE * HASH_BATCH_MULT : 1;

			function readBatch() {
						if(self._streamError) {
							if(fd) { try{fs.closeSync(fd);}catch(e){} }
							completeCb(self._streamError);
							return;
						}
						if(remaining <= 0) {
							if(fd) fs.closeSync(fd);
							eventCb('file_complete', file, processedBlocks);
							// A3: pool reset after each file completes (per spec).
							_resetInputBufferPool();
							processFileFull(idx + 1);
							return;
						}

						var items = [];
						while(items.length < batchSize && remaining > 0) {
							var toRead = Math.min(remaining, blockSize);
							var buf;
							if(bulkFileBuf) {
								// W2-T1: serve from single bulk read instead of per-block syscall
								var offset = bulkFileBuf.length - remaining;
								if(HASH_POOL_ENABLED) {
									buf = Buffer.alloc(toRead);
									bulkFileBuf.copy(buf, 0, offset, offset + toRead);
								} else {
									buf = acquireInputBuffer(toRead);
									bulkFileBuf.copy(buf, 0, offset, offset + toRead);
								}
							} else if(HASH_POOL_ENABLED) {
								// B3: per-block allocation — A3 pool slices alias
								// the pool and would be overwritten before the
								// async worker callbacks fire for previous blocks.
								buf = Buffer.alloc(toRead);
							} else {
								// A3: pool-backed buf (slice shares memory with inputBufferPool).
								buf = acquireInputBuffer(toRead);
							}
							if(bulkFileBuf) {
								remaining -= toRead;
							} else {
								var bytesRead = fs.readSync(fd, buf, 0, toRead, file.size - remaining);
								if(bytesRead === 0) { remaining = 0; break; }
								remaining -= bytesRead;
							}

							processedBlocks++;
							var blockIndex = processedBlocks - 1;

							if(!bulkFileBuf) {
								var writeOffset = nextBlockIdx * blockSize;
								nextBlockIdx++;
								if(buf.length < blockSize) {
									var padded = Buffer.alloc(blockSize);
									buf.copy(padded);
									padded.copy(fullInputs, writeOffset);
								} else {
									buf.copy(fullInputs, writeOffset);
								}
							}

							items.push({ buf: buf, blockIndex: blockIndex });

							if(processedBlocks % 100 === 0) {
								eventCb('progress', processedBlocks, totalBlocks);
							}
						}

						if(items.length === 0) {
							if(fd) fs.closeSync(fd);
							eventCb('complete', { processedBlocks: processedBlocks });
							completeCb(null);
							return;
						}

						if(!HASH_POOL_ENABLED || batchSize === 1) {
							items.forEach(function(item) {
								item.packet = self._createDataPacket(item.buf, item.blockIndex);
							});
							writeBatch(items, self._outputStream, readBatch);
							return;
						}

						// Parallel: dispatch all hashes in batch, await all,
						// then writeBatch writes packets in block-index order.
						var pending = items.length;
						items.forEach(function(item) {
							var bodySize = 8 + item.buf.length;
							var afterChecksum = self._getDataAfterChecksumFor(bodySize);
							var workerInput = Buffer.alloc(item.buf.length);
							item.buf.copy(workerInput);
							_hashBlockParallel(item.blockIndex, workerInput, afterChecksum, function(err, result) {
								var hash;
								if(err) {
									var bib = Buffer.alloc(8);
									bib.writeBigUInt64LE(BigInt(item.blockIndex), 0);
									hash = blake3.createHash()
										.update(afterChecksum)
										.update(bib)
										.update(item.buf)
										.digest()
										.slice(0, 16);
								} else {
									hash = result.hash;
								}
								var packet = createPacketHeader(PAR3_PKT_TYPE.DATA, bodySize, self.inputSetId);
								writeUInt64LE(packet, item.blockIndex, 48);
								item.buf.copy(packet, 56);
								hash.copy(packet, 8, 0, 16);
								item.packet = packet;
								pending--;
								if(pending === 0) writeBatch(items, self._outputStream, readBatch);
							});
						});
					}

					readBatch();
				}
			};

			eventCb('start', { totalBlocks: totalBlocks });
			global.__par3ProfEnd('setup');
			global.__par3ProfStart('read');
			processFileFull(0);
			return;
		}

		// Per-batch path: kept for PAR3_BATCH_SIZE env-override, JS-kernel
		// fallback, and totalRecoverySize > 512 MiB cases.
		var BATCH_SIZE = Number(process.env.PAR3_BATCH_SIZE) || 64;
		if(!(BATCH_SIZE >= 1)) BATCH_SIZE = 64;
		var batchInputs = [];

		// P1 of issue #46: prefer compute_recovery_streaming when available
		// (single C++ call instead of N/BATCH_SIZE per-batch dispatchRecovery
		// roundtrips). Falls back to per-batch path if binding absent or
		// totalRecoverySize > MAX_IN_MEMORY (streaming entry has no chunking).
		var streamBinding = null;
		var batchBuffers = null;
		if(totalRecoverySize <= MAX_IN_MEMORY && !process.env.PAR3_GF64_USE_LEGACY_BATCH) {
			var probeBinding = getGf64Binding();
			if(probeBinding && typeof probeBinding.compute_recovery_streaming === 'function') {
				streamBinding = probeBinding;
				batchBuffers = [];
			}
		}

		function flushBatch() {
			if(batchInputs.length === 0) return;
			ensureAccumulator();
			if(streamBinding) {
				var batchBuf = Buffer.concat(batchInputs, batchInputs.length * o.blockSize);
				batchBuffers.push(batchBuf);
				batchInputs.length = 0;
			} else {
				var batchStart = processedBlocks - batchInputs.length;
				self._processRecoveryBatch(batchInputs, batchStart, firstRecovery, numRecovery, accumulator);
				batchInputs.length = 0;
				flushCompletedBlocks();
			}
		}

		var processNextFile = function(idx) {
			if(idx >= self.files.length) {
				flushBatch();
				if(streamBinding) {
					// Cauchy/Fenger/Barycentric are all linear in inputs, so a
					// single call with all batches is mathematically identical
					// to per-batch dispatchRecovery calls XORed into accumulator.
					// numInputs must be the actual padded-block count in
					// batchBuffers (each file < blockSize contributes one full
					// padded block, so this can exceed this.totalBlocks).
					var streamingInputs = 0;
					for(var bi = 0; bi < batchBuffers.length; bi++) {
						streamingInputs += batchBuffers[bi].length / o.blockSize;
					}
					global.__par3ProfStart('kernel');
					streamBinding.compute_recovery_streaming(
						batchBuffers,
						accumulator,
						streamingInputs,
						numRecovery,
						o.blockSize,
						0,
						firstRecovery,
						self.opts.numThreads || 0
					);
					global.__par3ProfEnd('kernel');
					flushCompletedBlocks();
					batchBuffers = null;
				}
				eventCb('generating_recovery', { inputBlocks: totalInputBlocks, recoveryBlocks: numRecovery });
				var finalAcc = tmpFilePath || accumulator;
				self._finalizeRecoveryBlocks(finalAcc, totalInputBlocks, eventCb, function(err) {
					// B2: release accumulator back to the pool (data already .copy()'d into packets).
					releaseRecoveryBuffer(accumulator);
					self._outputStream.end(function() {
						self._outputStream = null;
						eventCb('complete', { processedBlocks: processedBlocks });
						// v2: profile dump (PAR3_PROFILE=1) -- BEFORE completeCb
						if (PROFILE) {
							var totalMs = Date.now() - (global.__par3Profile.start || Date.now());
							var keys = Object.keys(profilePhases).filter(function(k){return k!=='_t';});
							var lines = ['[PAR3_PROFILE] create took ' + totalMs + 'ms total'];
							keys.forEach(function(k){
								var pct = totalMs > 0 ? (profilePhases[k]/totalMs*100).toFixed(1) : '0.0';
								lines.push('  ' + k.padEnd(8) + ' ' + profilePhases[k].toFixed(1).padStart(8) + 'ms  (' + pct + '%)');
							});
							try {
								var fs = require('fs');
								var dumpPath = path.join(process.cwd(), '.par3-profile-dump.txt');
								fs.writeFileSync(dumpPath, lines.join('\n') + '\n');
							} catch (e) { /* ignore */ }
						}

						completeCb(err);
					});
				});
				return;
			}

			var file = self.files[idx];
			if(!file.size) {
				// A3: pool reset between files (mirrors the useComputeRecoveryFull path).
				_resetInputBufferPool();
				processNextFile(idx + 1);
				return;
			}

			eventCb('processing_file', file, idx);

			// A3: reset before reading the next file (mirrors useComputeRecoveryFull).
			_resetInputBufferPool();

			var fd;
			var remaining = file.size;
			var blockSize = o.blockSize;

			fd = fs.openSync(file.name, 'r');
			readBlock();

			function readBlock() {
				// B3: batch reads + dispatch hashes in parallel (batchSize=poolSize)
				// when enabled; otherwise batchSize=1 preserves one-block-at-a-time
				// behavior.
				var batchSize = HASH_POOL_ENABLED ? HASH_POOL_SIZE * HASH_BATCH_MULT : 1;

				function readBatch() {
					if(self._streamError) {
						if(fd) { try{fs.closeSync(fd);}catch(e){} }
						completeCb(self._streamError);
						return;
					}
					if(remaining <= 0) {
						fs.closeSync(fd);
						eventCb('file_complete', file, processedBlocks);
						// A3: pool reset after each file completes (per spec).
						_resetInputBufferPool();
						processNextFile(idx + 1);
						return;
					}

					var items = [];
					while(items.length < batchSize && remaining > 0) {
						var toRead = Math.min(remaining, blockSize);
						var buf;
						if(HASH_POOL_ENABLED) {
							// B3: per-block allocation — A3 pool slices alias
							// the pool and would be overwritten before the
							// async worker callbacks fire for previous blocks.
							buf = Buffer.alloc(toRead);
						} else {
							// A3: pool-backed buf (slice shares memory with inputBufferPool).
							buf = acquireInputBuffer(toRead);
						}
						var bytesRead = fs.readSync(fd, buf, 0, toRead, file.size - remaining);
						if(bytesRead === 0) { remaining = 0; break; }

						processedBlocks++;
						remaining -= bytesRead;
						var blockIndex = processedBlocks - 1;

						if(buf.length < blockSize) {
							var padded = Buffer.alloc(blockSize);
							buf.copy(padded);
							batchInputs.push(padded);
						} else {
							batchInputs.push(buf);
						}

						if(batchInputs.length >= BATCH_SIZE) {
							flushBatch();
						}

						items.push({ buf: buf, blockIndex: blockIndex });

						if(processedBlocks % 100 === 0) {
							eventCb('progress', processedBlocks, totalBlocks);
						}
					}

					if(items.length === 0) {
						fs.closeSync(fd);
						eventCb('complete', { processedBlocks: processedBlocks });
						completeCb(null);
						return;
					}

					if(!HASH_POOL_ENABLED || batchSize === 1) {
						items.forEach(function(item) {
							item.packet = self._createDataPacket(item.buf, item.blockIndex);
						});
						writeBatch(items, self._outputStream, readBatch);
						return;
					}

					var pending = items.length;
					items.forEach(function(item) {
						var bodySize = 8 + item.buf.length;
						var afterChecksum = self._getDataAfterChecksumFor(bodySize);
						var workerInput = Buffer.alloc(item.buf.length);
						item.buf.copy(workerInput);
						_hashBlockParallel(item.blockIndex, workerInput, afterChecksum, function(err, result) {
							var hash;
							if(err) {
								var bib = Buffer.alloc(8);
								bib.writeBigUInt64LE(BigInt(item.blockIndex), 0);
								hash = blake3.createHash()
									.update(afterChecksum)
									.update(bib)
									.update(item.buf)
									.digest()
									.slice(0, 16);
							} else {
								hash = result.hash;
							}
							var packet = createPacketHeader(PAR3_PKT_TYPE.DATA, bodySize, self.inputSetId);
							writeUInt64LE(packet, item.blockIndex, 48);
							item.buf.copy(packet, 56);
							hash.copy(packet, 8, 0, 16);
							item.packet = packet;
							pending--;
							if(pending === 0) writeBatch(items, self._outputStream, readBatch);
						});
					});
				}

				readBatch();
			}
		};

		eventCb('start', { totalBlocks: totalBlocks });
		processNextFile(0);
	},

	_createDataPacket: function(data, blockIndex) {
		var bodySize = 8 + data.length;
		var packet = createPacketHeader(PAR3_PKT_TYPE.DATA, bodySize, this.inputSetId);
		var bodyOffset = PAR3_PKT_HDR_SIZE;
		writeUInt64LE(packet, blockIndex, bodyOffset);
		data.copy(packet, bodyOffset + 8);
		finalizePacketHeader(packet, packet.slice(48));
		return packet;
	},

	// B3: cached 24-byte post-checksum header fragment for DATA packets.
	// Body size varies (full block vs trailing partial block), so cache by size.
	_getDataAfterChecksumFor: function(bodySize) {
		var cache = this._dataAfterChecksumCache || (this._dataAfterChecksumCache = {});
		var cached = cache[bodySize];
		if(cached) return cached;
		var ac = Buffer.alloc(24);
		writeUInt64LE(ac, PAR3_PKT_HDR_SIZE + bodySize, 0);
		this.inputSetId.copy(ac, 8);
		ac.write(PAR3_PKT_TYPE.DATA, 16, 8, 'ascii');
		cache[bodySize] = ac;
		return ac;
	},

	_writePacket: function(packet) {
		if(!this._outputStream) return false;
		return this._outputStream.write(packet);
	},

	// Close and cleanup
	close: function(cb) {
		if(this.encoder) {
			try {
				var binding = getGf64Binding();
				if(binding && binding.Gf64Encoder_destroy) {
					binding.Gf64Encoder_destroy(this.encoder);
				}
			} catch(e) {}
			this.encoder = null;
		}
		if(cb) process.nextTick(cb);
	}
};

// D5: Batched output writes using stream.cork() + pre-allocated buffer.
// Accumulates up to 64 packets then flushes; handles backpressure via drain.
var WRITE_BATCH_SIZE = 64;

function writeBatch(items, stream, cb) {
	var idx = 0;
	var total = items.length;

	function flush() {
		if(idx >= total) { cb(); return; }

		var end = Math.min(idx + WRITE_BATCH_SIZE, total);
		var batchStart = idx;
		idx = end;

		// Measure total bytes (skip null packets)
		var totalLen = 0;
		for(var i = batchStart; i < end; i++) {
			var pkt = items[i].packet;
			if(pkt) totalLen += pkt.length;
		}

		if(totalLen === 0) {
			process.nextTick(flush);
			return;
		}

		// Pre-allocate single buffer, copy packets into it
		var merged = Buffer.allocUnsafe(totalLen);
		var offset = 0;
		for(var i = batchStart; i < end; i++) {
			var pkt = items[i].packet;
			if(pkt) {
				pkt.copy(merged, offset);
				offset += pkt.length;
			}
		}

		stream.cork();
		var ok = stream.write(merged);
		stream.uncork();

		if(!ok) {
			stream.once('drain', flush);
		} else {
			process.nextTick(flush);
		}
	}

	flush();
}

// ============================================================================
// High-level API Functions
// ============================================================================

function run_par3(files, blockSize, opts, cb) {
	if(typeof opts === 'function' && cb === undefined) {
		cb = opts;
		opts = {};
	}
	if(!files) {
		process.nextTick(function() {
			cb(new Error('No input files supplied'));
		});
		return;
	}
	if(!files.length) files = [];

	var ee = new emitter();

	// Get file info (first 16KB for hash)
	module.exports.fileInfo(files, function(err, info) {
		if(err) return cb(err);

		try {
			var par = new PAR3Gen(info, blockSize, opts);
		} catch(e) {
			return cb(e);
		}

		ee.emit('info', par);

		// Run the PAR3 generation
		par.run(function(event) {
			var args = Array.prototype.slice.call(arguments, 1);
			ee.emit.apply(ee, [event, par].concat(args));
		}, function(err) {
			par.close();
			cb(err);
		});
	});

	return ee;
}

function fileInfo(files, recurse, skipSymlinks, concurrency, cb) {
	if(!cb) {
		cb = concurrency;
		concurrency = null;
		if(!cb) {
			cb = skipSymlinks;
			skipSymlinks = false;
			if(!cb) {
				cb = recurse;
				recurse = false;
			}
		}
	}
	if(!concurrency) concurrency = 2;

	var results = [];
	var scanFiles;
	var bufs = new (require('./bufferpool'))([], 16384, concurrency);
	var statFn = skipSymlinks ? fs.lstat : fs.stat;

	scanFiles = function(files, recurse, cbDoneScan) {
		var filesLeft = files.length;
		if(filesLeft == 0) return cbDoneScan();
		var doneCalled = false;
		var procErr = null;

		async.eachSeries(files, function(file, cbNextFile) {
			bufs.get(function(buf) {
				var info = { name: file, size: 0, sha256_16k: null };
				var fd;

				async.waterfall([
					statFn.bind(fs, file),
					function(stat, cb) {
						info.mode = stat.mode;
						if(stat.isDirectory()) {
							info = null;
							if(recurse) {
								fs.readdir(file, function(err, dirFiles) {
									bufs.put(buf);
									if(err) return cb(err);
									scanFiles(dirFiles.map(function(fn) {
										return path.join(file, fn);
									}), typeof recurse === 'number' ? recurse - 1 : recurse, function(err) {
										cb(err || true);
									});
								});
							} else {
								bufs.put(buf);
								cb(true);
							}
							return;
						}
						if(stat.isSymbolicLink()) {
							info = null;
							bufs.put(buf);
							return cb(true);
						}
						if(!stat.isFile()) return cb(new Error(file + ' is not a valid file'));

						info.size = stat.size;
						if(!info.size) {
							info.sha256_16k = Buffer.alloc(32);
							return cb(true);
						}
						fs.open(file, 'r', cb);
					},
					function(_fd, cb) {
						fd = _fd;
						fs.read(fd, buf, 0, 16384, null, cb);
					},
					function(bytesRead, buffer, cb) {
						// For PAR3, use Blake3-like hash (SHA-256 for now since Blake3 not available)
						info.sha256_16k = crypto.createHash('sha256').update(bufferSlice.call(buffer, 0, bytesRead)).digest();
						fs.close(fd, cb);
					}
				], function(err) {
					if(err && err !== true)
						procErr = err;
					if(info) {
						results.push(info);
						bufs.put(buf);
					}
					if(--filesLeft == 0 && !doneCalled) {
						doneCalled = true;
						cbDoneScan(procErr);
					}
				});
				cbNextFile(procErr);
			});
		}, function(err) {
			if(err && !doneCalled) {
				doneCalled = true;
				cbDoneScan(err);
			}
		});
	};

	scanFiles(files, recurse, function(err) {
		bufs.end(function() {
			cb(err, results);
		});
	});

	return results;
}

// ============================================================================
// Command-line Interface Support
// ============================================================================

function par3_create(inputFiles, outputBase, opts, cb) {
	if(typeof opts === 'function') {
		cb = opts;
		opts = {};
	}

	var blockSize = opts.blockSize || PAR3_BLOCK_SIZE_DEFAULT;
	var recoverySlices;
	if(opts.recoverySlices === undefined) {
		// Default to 10% ratio for backward compatibility
		recoverySlices = { unit: 'ratio', value: 0.1 };
	} else if(typeof opts.recoverySlices === 'number') {
		// Number is interpreted as slice count
		recoverySlices = { unit: 'slices', value: opts.recoverySlices };
	} else {
		// Object — pass through (supports ratio, bytes, etc.)
		recoverySlices = opts.recoverySlices;
	}

	run_par3(inputFiles, blockSize, {
		outputBase: outputBase,
		recoverySlices: recoverySlices,
		numThreads: opts.numThreads,
		gfMethod: opts.gfMethod,
		matrixType: opts.matrixType || 'cauchy',
		memoryLimit: opts.memoryLimit
	}, cb);
}

function par3_verify(par3File, cb) {
	if(typeof par3File === 'function') {
		cb = par3File;
		par3File = null;
	}

	var verifyState = {
		startPacket: null,
		files: [],
		matrix: null,
		recoveryBlocks: [],
		dataBlocks: [],
		availableBlocks: [],
		missingBlocks: [],
		totalBlocks: 0,
		recoveryCount: 0,
		inputCount: 0
	};

	var packetCallback = function(type, body, body_length, header) {
		if(header) {
			validatePacketChecksum(header, body);
		}
		var offset = 0;

		switch(type) {
			case PAR3_PKT_TYPE.START: {
				var startData = {
					gf_size: body[offset],
					block_size: readUInt64LE(body, offset + 16),
					block_pow: body.readUInt32LE(offset + 24)
				};
				verifyState.startPacket = startData;
				verifyState.totalBlocks = Math.ceil(verifyState.inputCount / startData.block_size);
				break;
			}
			case PAR3_PKT_TYPE.FILE: {
				verifyState.inputCount++;
				break;
			}
			case PAR3_PKT_TYPE.CAUCHY:
			case PAR3_PKT_TYPE.MATRIX: {
				// par3cmdline emits 24-byte CAU bodies (first_input,
				// last_input, first_recovery) WITHOUT recovery_count;
				// the 40-byte body adds it. Guard the read and derive
				// the count from the recovery-block packets later
				// (Todo 5 / unit-cross-compat-cau.js).
				var hasRecoveryCount = (body.length - offset) >= 32;
				verifyState.matrix = {
					first_input: readUInt64LE(body, offset),
					last_input: readUInt64LE(body, offset + 8),
					first_recovery: readUInt64LE(body, offset + 16),
					recovery_count: hasRecoveryCount ? readUInt64LE(body, offset + 24) : undefined
				};
				if(hasRecoveryCount) {
					verifyState.recoveryCount = verifyState.matrix.recovery_count;
				}
				verifyState.inputCount = verifyState.matrix.last_input - verifyState.matrix.first_input + 1;
				break;
			}
			case PAR3_PKT_TYPE.RECOVERY: {
				var recoveryBlock = {
					first_block: readUInt64LE(body, offset),
					block_count: readUInt64LE(body, offset + 8)
				};
				verifyState.recoveryBlocks.push(recoveryBlock);
				for(var i = 0; i < recoveryBlock.block_count; i++) {
					verifyState.availableBlocks.push(recoveryBlock.first_block + i);
				}
				break;
			}
			case PAR3_PKT_TYPE.DATA: {
				var dataBlock = {
					block_index: readUInt64LE(body, offset)
				};
				verifyState.dataBlocks.push(dataBlock);
				verifyState.availableBlocks.push(dataBlock.block_index);
				break;
			}
		}
		return 0;
	};

	par3_parse_stream(par3File, packetCallback, function(err) {
		if(err) return cb(new Error('Failed to parse PAR3 file: ' + err.message));

		// Verify structure
		if(!verifyState.startPacket) {
			return cb(new Error('No Start packet found'));
		}
		if(!verifyState.matrix) {
			return cb(new Error('No Matrix packet found'));
		}
		// A 24-byte (par3cmdline) CAU body carries no recovery_count —
		// derive the count from the recovery-block packets. Only reject
		// when the matrix EXPLICITLY declared zero recovery blocks.
		if(verifyState.matrix.recovery_count === undefined) {
			verifyState.recoveryCount = 0;
			verifyState.recoveryBlocks.forEach(function(rb) {
				verifyState.recoveryCount += rb.block_count;
			});
		}
		if(verifyState.matrix.recovery_count === 0) {
			return cb(new Error('No recovery blocks available'));
		}

		// Calculate expected blocks
		var expectedBlocks = verifyState.inputCount + verifyState.recoveryCount;
		var availableSet = {};
		verifyState.availableBlocks.forEach(function(idx) {
			availableSet[idx] = true;
		});

		// Find missing blocks
		verifyState.missingBlocks = [];
		for(var i = 0; i < expectedBlocks; i++) {
			if(!availableSet[i]) {
				verifyState.missingBlocks.push(i);
			}
		}

		// Determine repairability
		var canRepair = verifyState.missingBlocks.length <= verifyState.recoveryCount;

		cb(null, {
			verified: true,
			archiveOk: verifyState.missingBlocks.length === 0,
			canRepair: canRepair,
			inputBlocks: verifyState.inputCount,
			recoveryBlocks: verifyState.recoveryCount,
			missingBlocks: verifyState.missingBlocks.length,
			missingBlockList: verifyState.missingBlocks,
			blockSize: verifyState.startPacket.block_size
		});
	});
}

// Compute MD5 checksum of data
function computeMD5(data) {
	var hash = crypto.createHash('md5');
	hash.update(data);
	return hash.digest();
}

// Atomic write: write to temp file, flush/sync, then rename to final path
function atomicWriteFile(filePath, data, mode, cb) {
	var tmpPath = filePath + '.tmp';
	var fd;

	async.waterfall([
		function(cb) {
			fs.open(tmpPath, 'w', mode, cb);
		},
		function(_fd, cb) {
			fd = _fd;
			fs.write(fd, data, 0, data.length, 0, cb);
		},
		function(bytesWritten, buffer, cb) {
			fs.fsync(fd, function(err) {
				if(err) {
					fs.close(fd, function() { cb(err); });
					return;
				}
				fs.close(fd, function(err2) {
					if(err2) { cb(err2); return; }
					// Rename temp to final path
					fs.rename(tmpPath, filePath, cb);
				});
			});
		}
	], function(err) {
		if(err) {
			// Clean up temp file on error
			fs.unlink(tmpPath, function() {});
			cb(err);
		} else {
			cb(null);
		}
	});
}

// Verify data matches expected checksum
function verifyChecksum(data, expectedChecksum) {
	if(!expectedChecksum) return { verified: true, match: true };
	var actual = computeMD5(data);
	var match = actual.equals(expectedChecksum);
	return {
		verified: true,
		match: match,
		expected: expectedChecksum.toString('hex'),
		actual: actual.toString('hex')
	};
}

// B2 repair gate: may the native Cauchy matrix build serve this repair?
// - footprint: n x totalInputs entries; the native call makes a NAPI copy
//   plus internal denom buffers, so the cap is budgeted ~2x (32M entries =
//   ~256 MiB native + ~256 MiB copy + internals, still under 512 MiB)
// - disjoint ranges: [firstRec, firstRec+n) and [firstInp, firstInp+total)
//   must not overlap — the native build maps denom==0 -> 1 (the create
//   contract) while the JS fallback historically skipped those terms;
//   overlap is unreachable for project-created archives (first_recovery ==
//   totalInputs) but keeps the historical semantics for malformed input.
function repairNativeMatrixAllowed(totalInputs, n, firstInp, firstRec, capEntries) {
	if (n <= 0 || totalInputs <= 0) return false;
	if (firstInp < firstRec + n && firstRec < firstInp + totalInputs) return false;
	if (n * totalInputs > capEntries) return false;
	return true;
}

function par3_repair(par3File, outputDir, opts, cb) {
	if(typeof opts === 'function') {
		cb = opts;
		opts = {};
	}
	if(!cb) cb = function() {};

	var verbose = opts.verbose || 0;
	var repairState = {
		startPacket: null,
		files: [],
		matrix: null,
		recoveryData: [],  // Available blocks for solving
		dataBlocks: [],    // Input data blocks
		missingBlocks: [], // Blocks that need to be reconstructed
		blockSize: 0,
		totalInputBlocks: 0,
		totalRecoveryBlocks: 0
	};

	// Progress reporting
	var reportProgress = function(msg) {
		if(verbose) {
			process.stderr.write(msg + '\n');
		}
	};

	reportProgress('Parsing PAR3 archive: ' + par3File);

	// Callback for parsing packets
	var packetCallback = function(type, body, body_length) {
		var offset = 0;

		switch(type) {
			case PAR3_PKT_TYPE.START: {
				repairState.startPacket = {
					gf_size: body[offset],
					block_size: readUInt64LE(body, offset + 16),
					block_pow: body.readUInt32LE(offset + 24)
				};
				repairState.blockSize = repairState.startPacket.block_size;
				break;
			}
			case PAR3_PKT_TYPE.CAUCHY:
			case PAR3_PKT_TYPE.MATRIX: {
				// par3cmdline emits 24-byte CAU bodies without
				// recovery_count (and without rows/cols); guard the
				// reads and derive the count from the recovery-block
				// packets later (Todo 5 / unit-cross-compat-cau.js).
				var matrixBodyLen = body.length - offset;
				repairState.matrix = {
					first_input: readUInt64LE(body, offset),
					last_input: readUInt64LE(body, offset + 8),
					first_recovery: readUInt64LE(body, offset + 16),
					recovery_count: matrixBodyLen >= 32 ? readUInt64LE(body, offset + 24) : undefined,
					rows: matrixBodyLen >= 40 ? body.readUInt32LE(offset + 32) : undefined,
					cols: matrixBodyLen >= 40 ? body.readUInt32LE(offset + 36) : undefined
				};
				repairState.totalInputBlocks = repairState.matrix.last_input - repairState.matrix.first_input + 1;
				repairState.totalRecoveryBlocks = repairState.matrix.recovery_count;
				break;
			}
			case PAR3_PKT_TYPE.RECOVERY: {
				var block_count = readUInt64LE(body, offset + 8);
				var recBlock = {
					first_block: readUInt64LE(body, offset),
					block_count: block_count,
					// Each recovery block is 8 bytes (GF(2^64) coefficient)
					data: body.slice(offset + 16, offset + 16 + 8 * block_count)
				};
				repairState.recoveryData.push(recBlock);
				break;
			}
			case PAR3_PKT_TYPE.DATA: {
				var dataBlock = {
					block_index: readUInt64LE(body, offset),
					data: body.slice(offset + 8, offset + 8 + repairState.blockSize)
				};
				repairState.dataBlocks.push(dataBlock);
				break;
			}
		}
		return 0;
	};

	// PASS 1: Stream-parse PAR3 file to extract metadata and record body offsets (no bodies kept)
	var dataBlockOffsets = [];  // { block_index, fileOffset }
	var recoveryOffsets = [];   // { first_block, block_count, fileOffset }

	var pass1Callback = function(type, body, bodyLen, absBodyOffset, header) {
		var offset = 0;
		// Validate packet checksum. If a DATA or RECOVERY packet fails validation,
		// its body is corrupt and the block must be treated as MISSING (not recorded
		// in dataBlockOffsets / recoveryOffsets). Without this filter, a damaged
		// archive still appears "complete" because the file offsets are valid even
		// though the contents are wrong, and the no-repair path writes corrupt data.
		var checksumValid = true;
		if(header) {
			checksumValid = validatePacketChecksum(header, body);
		}

		switch(type) {
			case PAR3_PKT_TYPE.START: {
				repairState.startPacket = {
					gf_size: body[offset],
					block_size: readUInt64LE(body, offset + 16),
					block_pow: body.readUInt32LE(offset + 24)
				};
				repairState.blockSize = repairState.startPacket.block_size;
				break;
			}
			case PAR3_PKT_TYPE.CAUCHY:
			case PAR3_PKT_TYPE.MATRIX: {
				// par3cmdline emits 24-byte CAU bodies without
				// recovery_count (and without rows/cols); guard the
				// reads and derive the count from the recovery-block
				// packets later (Todo 5 / unit-cross-compat-cau.js).
				var matrixBodyLen = body.length - offset;
				repairState.matrix = {
					first_input: readUInt64LE(body, offset),
					last_input: readUInt64LE(body, offset + 8),
					first_recovery: readUInt64LE(body, offset + 16),
					recovery_count: matrixBodyLen >= 32 ? readUInt64LE(body, offset + 24) : undefined,
					rows: matrixBodyLen >= 40 ? body.readUInt32LE(offset + 32) : undefined,
					cols: matrixBodyLen >= 40 ? body.readUInt32LE(offset + 36) : undefined
				};
				repairState.totalInputBlocks = repairState.matrix.last_input - repairState.matrix.first_input + 1;
				repairState.totalRecoveryBlocks = repairState.matrix.recovery_count;
				break;
			}
			case PAR3_PKT_TYPE.DATA: {
				if(!checksumValid) break; // treat damaged DATA packet as missing
				var blockIndex = readUInt64LE(body, offset);
				// Record file offset of this block's data (starts at body offset + 8)
				dataBlockOffsets.push({
					block_index: blockIndex,
					fileOffset: absBodyOffset + 8
				});
				break;
			}
			case PAR3_PKT_TYPE.RECOVERY: {
				if(!checksumValid) break; // treat damaged RECOVERY packet as missing
				var block_count = readUInt64LE(body, offset + 8);
				// Record file offset of recovery coefficients (starts at body offset + 16)
				recoveryOffsets.push({
					first_block: readUInt64LE(body, offset),
					block_count: block_count,
					fileOffset: absBodyOffset + 16
				});
				break;
			}
		}
		return 0;
	};

		par3_parse_stream_pass1(par3File, pass1Callback, function(err) {
			if(err) return cb(new Error('Failed to parse PAR3 file: ' + err));

			if(!repairState.startPacket) return cb(new Error('No Start packet found'));
			if(!repairState.matrix) return cb(new Error('No Matrix packet found'));

			// 24-byte (par3cmdline) CAU bodies omit recovery_count —
			// derive it from the recovery-block packets seen in pass 1
			// (recoveryOffsets carries block_count from pass 1;
			// recoveryData is only populated in pass 2, so it is
			// still empty here).
			if(repairState.matrix.recovery_count === undefined) {
				repairState.totalRecoveryBlocks = 0;
				recoveryOffsets.forEach(function(ro) {
					repairState.totalRecoveryBlocks += ro.block_count;
				});
			}

			// Sort DATA block offsets by block_index so pass 2 reads them in sequential order
			// This enables streaming writes in the no-repair-needed path
			dataBlockOffsets.sort(function(a, b) { return a.block_index - b.block_index; });

			reportProgress('Verifying blocks...');
		reportProgress('Input blocks: ' + repairState.totalInputBlocks);
		reportProgress('Recovery blocks: ' + repairState.totalRecoveryBlocks);

		// PASS 2: Open file again, seek to each recorded offset, read body data
		fs.open(par3File, 'r', function(err2, fd) {
			if(err2) return cb(new Error('Cannot open PAR3 file for body reading: ' + err2.message));

			async.eachSeries(dataBlockOffsets, function(entry, next) {
				var buf = allocBuffer(repairState.blockSize);
				fs.read(fd, buf, 0, repairState.blockSize, entry.fileOffset, function(err3, bytesRead) {
					if(err3) return next(err3);
					repairState.dataBlocks.push({
						block_index: entry.block_index,
						data: bytesRead < repairState.blockSize ? buf.slice(0, bytesRead) : buf
					});
					next();
				});
			}, function(err3) {
				if(err3) {
					fs.close(fd, function() { cb(new Error('Failed to read DATA blocks: ' + err3.message)); });
					return;
				}

				// Read recovery body data
				async.eachSeries(recoveryOffsets, function(entry, next) {
					var recDataLen = entry.block_count * repairState.blockSize;
					var buf = allocBuffer(recDataLen);
					fs.read(fd, buf, 0, recDataLen, entry.fileOffset, function(err4, bytesRead) {
						if(err4) return next(err4);
						repairState.recoveryData.push({
							first_block: entry.first_block,
							block_count: entry.block_count,
							data: buf
						});
						next();
					});
				}, function(err4) {
					fs.close(fd, function() {
						if(err4) return cb(new Error('Failed to read recovery blocks: ' + err4.message));

						// Build available block map
						var availableMap = {};
						repairState.dataBlocks.forEach(function(block) {
							availableMap[block.block_index] = block.data;
						});
						repairState.recoveryData.forEach(function(rec) {
							var blockSize = repairState.blockSize;
							for(var bi = 0; bi < rec.block_count; bi++) {
								availableMap[rec.first_block + bi] = rec.data.slice(bi * blockSize, (bi + 1) * blockSize);
							}
						});

						// Identify missing blocks
						var totalBlocks = repairState.totalInputBlocks + repairState.totalRecoveryBlocks;
						repairState.missingBlocks = [];
						for(var i = 0; i < totalBlocks; i++) {
							if(!availableMap[i]) {
								repairState.missingBlocks.push(i);
							}
						}

						reportProgress('Missing blocks: ' + repairState.missingBlocks.length);

						if(repairState.missingBlocks.length === 0) {
							reportProgress('No repair needed - archive is complete');
							reportProgress('Extracting data to block_0.dat');

							var outputFile = path.join(outputDir, 'block_0.dat');
							var outStream = fs.createWriteStream(outputFile);
							var writeError = null;

							// Write DATA blocks in sequential order (repairState.dataBlocks already sorted by block_index due to pre-sorted dataBlockOffsets)
							async.eachSeries(repairState.dataBlocks, function(block, next) {
								var ok = outStream.write(block.data);
								if(!ok) {
									outStream.once('drain', next);
								} else {
									next();
								}
							}, function(err) {
								outStream.end(function(endErr) {
									if(writeError) return cb(new Error('Failed to write output: ' + writeError.message));
									if(endErr) return cb(new Error('Failed to finalize output: ' + endErr.message));

									reportProgress('Wrote reconstructed file: ' + outputFile);

									return cb(null, {
										repaired: true,
										blocksRepaired: repairState.dataBlocks.length,
										missingBlocks: 0,
										missingBlockList: [],
										availableRecoveryBlocks: repairState.recoveryData.length,
										recoveryBlockList: repairState.recoveryData.map(function(r) {
											return { first_block: r.first_block, block_count: r.block_count };
										}),
										matrixInfo: repairState.matrix ? {
											firstInput: repairState.matrix.first_input,
											lastInput: repairState.matrix.last_input,
											firstRecovery: repairState.matrix.first_recovery,
											recoveryCount: repairState.matrix.recovery_count
										} : null,
										outputFile: outputFile,
										reconstructed: true
									});
								});
							});
						} else {

							if(repairState.missingBlocks.length > repairState.totalRecoveryBlocks) {
								cb(new Error('Cannot repair: too many missing blocks (' + repairState.missingBlocks.length + ' > ' + repairState.totalRecoveryBlocks + ')'));
								return;
							}

						reportProgress('Repair capability verified - reconstructing missing blocks');

						var binding = getGf64Binding();

						function invert64(val) {
							val = val & 0xFFFFFFFFFFFFFFFFn;
							if(val === 0n) return 0n;
							if(val === 1n) return 1n;

							var u = val;
							var POLY = 0x1000000000000001Bn;
							var v = POLY;
							var x1 = 1n;
							var x2 = 0n;

							while(u !== 1n && u !== 0n) {
								while((u & 1n) === 0n) {
									u >>= 1n;
									if((x1 & 1n) !== 0n) {
										x1 = ((x1 ^ POLY) >> 1n) & 0xFFFFFFFFFFFFFFFFn;
									} else {
										x1 >>= 1n;
									}
								}
								if(u === 1n) continue;
								while((v & 1n) === 0n) {
									v >>= 1n;
								}
								if(u < v) {
									var t = u; u = v; v = t;
									t = x1; x1 = x2; x2 = t;
								}
								u ^= v;
								x1 ^= x2;
							}
							return x1;
						}

						try {
							var n = repairState.missingBlocks.length;
							var recCount = repairState.totalRecoveryBlocks;
							var firstRec = repairState.matrix.first_recovery;
							var firstInp = repairState.matrix.first_input;
							var blockSize = repairState.blockSize;

							if(repairState.recoveryData.length < n) {
								cb(new Error('Not enough recovery blocks (' + repairState.recoveryData.length + ') to repair ' + n + ' missing blocks'));
								return;
							}

							var hasNativeSolve = binding && typeof binding.solve_and_reconstruct === 'function';
							// B2: one native batched Cauchy build covers BOTH the A
							// submatrix (missing x missing) and the RHS coefficients
							// (known x missing): M[eq][k] = invert((firstInp + k) ^
							// (firstRec + eq)). Footprint is n x N_total x 8 bytes —
							// capped so huge repairs stay on the JS fallback;
							// PAR3_REPAIR_NATIVE_MATRIX=0 forces the fallback
							// (native-vs-JS coefficient parity test).
							var nativeM = null;
							var N_total = repairState.totalInputBlocks;
							if (binding && typeof binding.build_coefficient_matrix === 'function'
							    && process.env.PAR3_REPAIR_NATIVE_MATRIX !== '0'
							    && repairNativeMatrixAllowed(N_total, n, firstInp, firstRec, 32 * 1024 * 1024)) {
								try {
									nativeM = binding.build_coefficient_matrix(N_total, n, firstInp, firstRec);
								} catch (e) {
									nativeM = null;
								}
							}
							var recByIdx = {};
								repairState.recoveryData.forEach(function(rec) {
									recByIdx[rec.first_block] = rec.data;
								});

								var A = Buffer.allocUnsafe(n * n * 8);
								if (nativeM) {
									for(var eq = 0; eq < n; eq++) {
										for(var col = 0; col < n; col++) {
											var missingIdx = repairState.missingBlocks[col];
											A.writeBigUInt64LE(nativeM.readBigUInt64LE((eq * N_total + missingIdx) * 8), eq * n * 8 + col * 8);
										}
									}
								} else {
									for(var eq = 0; eq < n; eq++) {
										for(var col = 0; col < n; col++) {
											var missingIdx = repairState.missingBlocks[col];
											var xi = BigInt(firstRec + eq);
											var yj = BigInt(firstInp + missingIdx);
											var denom = xi ^ yj;
											if(denom === 0n) denom = 1n;
											var coeff = invert64(denom);
											A.writeBigUInt64LE(coeff, eq * n * 8 + col * 8);
										}
									}
								}

								// Build RHS: rhs[eq] = recovery[eq] XOR (sum over KNOWN input blocks of coeff * input)
								// Each recovery block was computed as: recovery[eq] = sum_over_all_inputs(M[eq][j] * input[j])
								// To solve only for the missing subset, we must move the known contributions to the RHS:
								//   M_sub * missing = rhs  =>  rhs[eq] = recovery[eq] - sum_known(M[eq][j] * input[j])
								// Mirrors test/par3-repair-parity.js:348-377. Without this subtraction the solved
								// "missing blocks" include leaked contributions from the known inputs and the hash mismatches.
								var rhsBlocks = Buffer.allocUnsafe(n * blockSize);
								for(var eq = 0; eq < n; eq++) {
									var recBlockNum = firstRec + eq;
									var recData = recByIdx[recBlockNum];
									if(recData) {
										recData.copy(rhsBlocks, eq * blockSize);
									} else {
										rhsBlocks.fill(0, eq * blockSize, (eq + 1) * blockSize);
									}
								}

								// Identify KNOWN input blocks (available in availableMap, NOT in missingBlocks, NOT recovery blocks).
								var missingSet = {};
								repairState.missingBlocks.forEach(function(idx) { missingSet[idx] = true; });
								var knownInputIndices = [];
								for(var k = 0; k < repairState.totalInputBlocks; k++) {
									if(!missingSet[k] && availableMap[k]) knownInputIndices.push(k);
								}

								if(knownInputIndices.length > 0) {
									var numWords = blockSize / 8;
									// Pick the mul_arr backend: native (via binding) if available, else JS fallback
									var mulArrFn = (binding && typeof binding.mul_arr === 'function')
										? binding.mul_arr.bind(binding)
										: gf64Js.mul_arr;
									var coeffBuf = Buffer.alloc(8);
									var tmp = Buffer.allocUnsafe(blockSize);

									for(var ai = 0; ai < knownInputIndices.length; ai++) {
										var inputIdx = knownInputIndices[ai];
										var inputData = availableMap[inputIdx];
										for(var eq = 0; eq < n; eq++) {
											var coeff;
											if (nativeM) {
												// Same M[eq][inputIdx] entry the A-build reads —
												// the native build maps denom==0 to 1 (disjoint
												// ranges make denom==0 unreachable in practice).
												coeff = nativeM.readBigUInt64LE((eq * N_total + inputIdx) * 8);
											} else {
												var xi = BigInt(firstInp + inputIdx);
												var yk = BigInt(firstRec + eq);
												var denom = xi ^ yk;
												// Aligned with the native/create contract
												// (denom 0 -> 1): the term contributes coeff
												// invert(1) = 1. Overlapping ranges are gated
												// out of the native path, so both paths agree
												// on this case.
												coeff = (denom === 0n) ? 1n : invert64(denom);
											}
											coeffBuf.writeBigUInt64LE(coeff, 0);
											tmp.fill(0);
											mulArrFn(tmp, inputData, coeffBuf, numWords, 1);
											// XOR tmp into rhsBlocks[eq]
											var off = eq * blockSize;
											for(var b = 0; b < blockSize; b++) {
												rhsBlocks[off + b] ^= tmp[b];
											}
										}
									}
								}

								var ok;
								if(hasNativeSolve) {
									ok = binding.solve_and_reconstruct(A, rhsBlocks, n, blockSize);
								} else {
									ok = gf64Js.solve_and_reconstruct(A, rhsBlocks, n, blockSize, 0);
								}
								// The JS gf64_js.solve_and_reconstruct returns 0 on
								// success / -1 on failure, while the native
								// binding returns true/false. `!ok` treats JS
								// success (0) as failure — accept both return
								// conventions (Todo 6 / unit-par3-dispatch.js).
								if(ok !== 0 && ok !== true) {
									cb(new Error('Singular matrix - cannot solve'));
									return;
								}

								var repairedBlocks = [];
								for(var i = 0; i < n; i++) {
									var blockIdx = repairState.missingBlocks[i];
									var blockData = rhsBlocks.slice(i * blockSize, (i + 1) * blockSize);
									availableMap[blockIdx] = Buffer.from(blockData);
									repairedBlocks.push(blockIdx);
								}

								reportProgress('Successfully reconstructed ' + repairedBlocks.length + ' blocks');

								var allVerified = true;
								var verificationResults = [];
								var writeErrors = [];

								async.eachSeries(repairedBlocks, function(blockIdx, cb) {
									var blockData = availableMap[blockIdx];
									var outputPath = path.join(outputDir, 'block_' + blockIdx + '.dat');

									atomicWriteFile(outputPath, blockData, 0o644, function(err) {
										if(err) {
											writeErrors.push({ block: blockIdx, error: err.message });
											allVerified = false;
											return cb(null);
										}

										var verification = verifyChecksum(blockData, null);
										verificationResults.push({
											block: blockIdx,
											verified: verification.verified,
											match: verification.match
										});

										cb(null);
									});
								}, function() {
									// Write the full reconstructed file to block_0.dat so callers that
									// hash a single "repaired file" get the entire input back (matches
									// the no-repair-needed path's convention). Individual block_N.dat
									// files are still emitted above for block-level inspection.
									reportProgress('Writing reconstructed file to block_0.dat');
									var fullPath = path.join(outputDir, 'block_0.dat');
									var fullStream = fs.createWriteStream(fullPath);
									var fullWriteError = null;
									var blockIndices = [];
									for(var bi = 0; bi < repairState.totalInputBlocks; bi++) blockIndices.push(bi);
									async.eachSeries(blockIndices, function(k, nextBlock) {
										var blockBuf = availableMap[k];
										if(!blockBuf) {
											fullWriteError = 'Missing block ' + k + ' in availableMap';
											return nextBlock(fullWriteError);
										}
										var ok = fullStream.write(blockBuf);
										if(!ok) fullStream.once('drain', nextBlock);
										else nextBlock();
									}, function(streamErr) {
										if(streamErr) fullWriteError = streamErr;
										fullStream.end(function(endErr) {
											if(fullWriteError || endErr) {
												writeErrors.push({ block: 'full', error: (fullWriteError || endErr).message || String(fullWriteError || endErr) });
												allVerified = false;
											}
											var result = {
												repaired: true,
												blocksRepaired: repairedBlocks.length,
												missingBlocks: repairState.missingBlocks.length,
												missingBlockList: repairState.missingBlocks,
												repairedBlockList: repairedBlocks,
												availableRecoveryBlocks: repairState.recoveryData.length,
												recoveryBlockList: repairState.recoveryData.map(function(r) {
													return { first_block: r.first_block, block_count: r.block_count };
												}),
												usedNativeMatrix: !!nativeM,
												matrixInfo: repairState.matrix ? {
													firstInput: repairState.matrix.first_input,
													lastInput: repairState.matrix.last_input,
													firstRecovery: repairState.matrix.first_recovery,
													recoveryCount: repairState.matrix.recovery_count
												} : null,
												verification: {
													allVerified: allVerified,
													verifiedBlocks: verificationResults.filter(function(v) { return v.match; }).length,
													totalBlocks: repairedBlocks.length,
													verificationResults: verificationResults,
													writeErrors: writeErrors
												}
											};

											if(writeErrors.length > 0) {
												result.repaired = false;
												result.error = 'Failed to write ' + writeErrors.length + ' blocks';
											}

											cb(null, result);
										});
									});
								});
						} catch(e) {
							cb(new Error('Repair failed: ' + e.message));
						}
						}
					});
				});
			});
		});
	});
}

// Streaming parser (pass 1): reads a PAR3 file in 1MB chunks, tracks absolute file offsets,
// and calls packetCallback(type, body, bodyLen, absBodyOffset) for each packet.
// absBodyOffset is the absolute byte position of the packet body in the file.
// Used by par3_repair to record body offsets without loading the whole file.
function par3_parse_stream_pass1(filePath, packetCallback, done) {
	var CHUNK_SIZE = 1024 * 1024;

	fs.open(filePath, 'r', function(err, fd) {
		if(err) return done(new Error('Cannot open PAR3 file: ' + err.message));

		var leftover = null;
		var currentPos = 0;  // Absolute file position of the next read

		// Pre-allocated buffer for chunk accumulation — avoids per-read Buffer.concat.
		// Grows by CHUNK_SIZE if leftover is larger than the current allocation.
		var accumBuf = null;
		var accumLen = 0;

		function readChunk() {
			var buf = allocBuffer(CHUNK_SIZE);

			fs.read(fd, buf, 0, CHUNK_SIZE, currentPos, function(err2, bytesRead) {
				if(err2) {
					fs.close(fd, function() {
						done(new Error('Error reading PAR3 file: ' + err2.message));
					});
					return;
				}

				if(bytesRead === 0) {
					fs.close(fd, function() {
						if(leftover && leftover.length > 0) {
							done(new Error('Truncated PAR3 file: incomplete packet at end'));
						} else {
							done();
						}
					});
					return;
				}

				var bytes = bytesRead < CHUNK_SIZE ? bytesRead : CHUNK_SIZE;

				// Total space needed for leftover + new chunk
				var needed = (leftover ? leftover.length : 0) + bytes;

				// Grow pre-allocated buffer if current one is too small
				if(!accumBuf || accumBuf.length < needed) {
					accumBuf = allocBuffer(Math.max(needed, CHUNK_SIZE));
				}

				var writeOffset = 0;
				if(leftover) {
					leftover.copy(accumBuf, 0);
					writeOffset += leftover.length;
				}

				buf.copy(accumBuf, writeOffset, 0, bytes);
				accumLen = needed;

				// accumulated[0] is at file offset: currentPos - (leftover ? leftover.length : 0)
				var accumulatedStart = currentPos - (leftover ? leftover.length : 0);

				var parseOffset = 0;
				var lastCompleteEnd = 0;

				while(parseOffset + 48 <= accumLen) {
					var magic = accumBuf.slice(parseOffset, parseOffset + 8);

					if(!magic.equals(PAR3_MAGIC)) {
						parseOffset += 8;
						continue;
					}

					var pktLen = readUInt64LE(accumBuf, parseOffset + 24);
					if(pktLen < 48 || parseOffset + pktLen > accumLen) {
						break;
					}

					var bodyLen = pktLen - 48;
					var body = accumBuf.slice(parseOffset + 48, parseOffset + pktLen);
					var typeStr = accumBuf.slice(parseOffset + 40, parseOffset + 48).toString('ascii');

					var type = PAR3_PKT_TYPE.START;
					for(var k in PAR3_PKT_TYPE) {
						if(PAR3_PKT_TYPE[k] === typeStr) {
							type = PAR3_PKT_TYPE[k];
							break;
						}
					}

					// Absolute file offset of this packet's body
					var absBodyOffset = accumulatedStart + parseOffset + 48;

					try {
						var pktHeader = accumBuf.slice(parseOffset, parseOffset + 48);
						packetCallback(type, body, bodyLen, absBodyOffset, pktHeader);
					} catch(e) {
						fs.close(fd, function() {
							done(e);
						});
						return;
					}

					parseOffset += pktLen;
					lastCompleteEnd = parseOffset;
				}

				leftover = lastCompleteEnd < accumLen ? accumBuf.slice(lastCompleteEnd, accumLen) : null;
				currentPos += bytesRead;
				readChunk();
			});
		}

		readChunk();
	});
}

// Streaming parser: reads a PAR3 file in 1MB chunks and calls packetCallback(type, body, bodyLen) for each packet.
function par3_parse_stream(filePath, packetCallback, done) {
	var CHUNK_SIZE = 1024 * 1024;

	fs.open(filePath, 'r', function(err, fd) {
		if(err) return done(new Error('Cannot open PAR3 file: ' + err.message));

		var leftover = null;

		var accumBuf = null;
		var accumLen = 0;

		function readChunk() {
			var buf = allocBuffer(CHUNK_SIZE);

			fs.read(fd, buf, 0, CHUNK_SIZE, null, function(err2, bytesRead) {
				if(err2) {
					fs.close(fd, function() {
						done(new Error('Error reading PAR3 file: ' + err2.message));
					});
					return;
				}

				if(bytesRead === 0) {
					fs.close(fd, function() {
						if(leftover && leftover.length > 0) {
							done(new Error('Truncated PAR3 file: incomplete packet at end'));
						} else {
							done();
						}
					});
					return;
				}

				var bytes = bytesRead < CHUNK_SIZE ? bytesRead : CHUNK_SIZE;
				var needed = (leftover ? leftover.length : 0) + bytes;
				if(!accumBuf || accumBuf.length < needed) {
					accumBuf = allocBuffer(Math.max(needed, CHUNK_SIZE));
				}

				var writeOffset = 0;
				if(leftover) {
					leftover.copy(accumBuf, 0);
					writeOffset += leftover.length;
				}
				buf.copy(accumBuf, writeOffset, 0, bytes);
				accumLen = needed;

				var parseOffset = 0;
				var lastCompleteEnd = 0;

				while(parseOffset + 48 <= accumLen) {
					var magic = accumBuf.slice(parseOffset, parseOffset + 8);

					if(!magic.equals(PAR3_MAGIC)) {
						parseOffset += 8;
						continue;
					}

					var pktLen = readUInt64LE(accumBuf, parseOffset + 24);
					if(pktLen < 48 || parseOffset + pktLen > accumLen) {
						break;
					}

					var bodyLen = pktLen - 48;
					var body = accumBuf.slice(parseOffset + 48, parseOffset + pktLen);
					var typeStr = accumBuf.slice(parseOffset + 40, parseOffset + 48).toString('ascii');

					var type = PAR3_PKT_TYPE.START;
					for(var k in PAR3_PKT_TYPE) {
						if(PAR3_PKT_TYPE[k] === typeStr) {
							type = PAR3_PKT_TYPE[k];
							break;
						}
					}

					try {
						var pktHeader = accumBuf.slice(parseOffset, parseOffset + 48);
						packetCallback(type, body, bodyLen, pktHeader);
					} catch(e) {
						fs.close(fd, function() {
							done(e);
						});
						return;
					}

					parseOffset += pktLen;
					lastCompleteEnd = parseOffset;
				}

				leftover = lastCompleteEnd < accumLen ? accumBuf.slice(lastCompleteEnd, accumLen) : null;
				readChunk();
			});
		}

		readChunk();
	});
}

// Helper to parse buffer (simple version of par3_parse_buffer)
// In production this would call into native code
function par3_parse_buffer(buffer, callback, done) {
	var offset = 0;
	var PAR3_MAGIC = Buffer.from('PAR3\x00PKT');

	while(offset + 48 <= buffer.length) {
		// Check magic
		var magic = buffer.slice(offset, offset + 8);
		if(!magic.equals(PAR3_MAGIC)) {
			offset += 8;
			continue;
		}

		var pktLen = readUInt64LE(buffer, offset + 24);
		if(pktLen < 48 || offset + pktLen > buffer.length) {
			offset += 48;
			continue;
		}

		var bodyLen = pktLen - 48;
		var body = buffer.slice(offset + 48, offset + pktLen);
		var typeStr = buffer.slice(offset + 40, offset + 48).toString('ascii');

		// Map type string to PAR3_PKT_TYPE
		var type = PAR3_PKT_TYPE.START; // default
		for(var k in PAR3_PKT_TYPE) {
			if(PAR3_PKT_TYPE[k] === typeStr) {
				type = PAR3_PKT_TYPE[k];
				break;
			}
		}

		try {
			callback(type, body, bodyLen);
		} catch(e) {
			return done(e);
		}

		offset += pktLen;
	}

	done();
}

// ============================================================================
// Module Exports
// ============================================================================

module.exports = {
	PAR3Gen: PAR3Gen,

	// High-level API
	run: run_par3,
	run_par3: run_par3,

	// File info
	fileInfo: fileInfo,

	// CLI helpers
	create: par3_create,
	verify: par3_verify,
	repair: par3_repair,

	// B2 repair-matrix gate (test-facing; used by par3_repair)
	repairNativeMatrixAllowed: repairNativeMatrixAllowed,

	// Constants
	BLOCK_SIZE_DEFAULT: PAR3_BLOCK_SIZE_DEFAULT,
	GF_SIZE: PAR3_GF_SIZE,

	// Packet types
	PACKET_TYPE: PAR3_PKT_TYPE,

	// Version
	version: require('../package').version,

	// Packet header helpers
	finalizePacketHeader: finalizePacketHeader,

	// Binding reference for verification
	gf64Binding: gf64Binding,

	// Test hook for deterministic worker-pool shutdown; beforeExit handles it
	// for normal exit.
	shutdownHashPool: _shutdownHashPool,

	// T12: expose the Barycentric dispatch gate for the isolated dispatch test.
	dispatchRecovery: dispatchRecovery,
	// T12: env-tunable threshold (read once per dispatch).
	BARY_MIN_INPUTS_DEFAULT: BARY_MIN_INPUTS_DEFAULT
};