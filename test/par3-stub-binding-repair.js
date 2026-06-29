#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 Stub-Binding Repair E2E Test (T3 - TDD red - failing on master HEAD)
//
// Exercises the *repair* path of PAR3 with a fully stub binding — the same
// shape observed on macOS arm64 and Windows builds of `parpar_gf64.node`:
// a .node file that exports only `gf64_info` and lacks every other native
// function (no Gf64Encoder_create, no compute_recovery, no
// compute_recovery_full, no solve_and_reconstruct).
//
// The repair path must therefore fall through to the pure-JS kernel
// (lib/gf64_js.js) via the typeof guards wired in by commits 0141609,
// f9ee716, 65bf4ac, and 475d5b6.
//
// Two pre-existing production bugs in lib/par3gen.js cause this test to FAIL
// on the current branch HEAD (475d5b6):
//
//   Bug B (start packet offset): _createStartPacket writes blockSize at body
//     offset 24 as UInt32LE, but parsers in par3_repair read it at body
//     offset 16 as UInt64LE. Repair parses block_size = 0, allocates
//     n*0 bytes for rhsBlocks, and produces empty blocks.
//
//   Bug A (repair RHS adjustment): par3_repair builds rhsBlocks from raw
//     recovery data, missing the XOR-with-known-contribution step required
//     by Cauchy recovery (b' = rec[k] XOR sum over known i of
//     1/(rec_k + i) * data[i]). The reconstructed blocks are wrong even
//     when block_size parses correctly.
//
// Test will PASS after both fixes land (T4).
//
// Note on corruption: the repair path is driven entirely from the DATA /
// RECOVERY packets inside the archive (it does not look at source files on
// disk). To exercise the missing-block reconstruction path we corrupt the
// 8-byte magic of the last DATA packet header in-place. This makes pass1's
// packet scanner skip it entirely, so the block's index never enters
// availableMap and the repair path invokes solve_and_reconstruct to recover
// it from the RECOVERY packets (which still follow the DATA packets per
// Amendment 9 and are left untouched).
// ============================================================================

var path = require('path');
var fs = require('fs');
var os = require('os');
var Module = require('module');
var crypto = require('crypto');

console.log('PAR3 Stub-Binding Repair E2E Test (TDD red)');
console.log('============================================\n');

var passed = 0;
var failed = 0;
var total = 0;
var failures = [];

function check(condition, msg) {
    total++;
    if (condition) {
        console.log('  PASS: ' + msg);
        passed++;
    } else {
        console.error('  FAIL: ' + msg);
        failed++;
        failures.push(msg);
        process.exitCode = 1;
    }
}

function restore() {
    Module._load = origLoad;
    fs.existsSync = origExistsSync;
}

// ============================================================================
// Stub binding — only gf64_info is present. Mirrors the macOS arm64 / Windows
// build where the native addon loads but exports only the dispatch-info
// helper. The repair path therefore takes the pure-JS fallback
// (gf64Js.solve_and_reconstruct) which exposes Bug A and Bug B clearly.
// ============================================================================
var stubBinding = {
    gf64_info: function() { return { method: 3, name: 'scalar' }; }
};

var requiredMissingFns = [
    'Gf64Encoder_create', 'Gf64Encoder_destroy',
    'compute_recovery', 'compute_recovery_full',
    'solve_and_reconstruct', 'mul_arr', 'invert'
];
for (var i = 0; i < requiredMissingFns.length; i++) {
    if (typeof stubBinding[requiredMissingFns[i]] === 'function') {
        console.error('FATAL: stubBinding.' + requiredMissingFns[i] +
                      ' must be undefined to faithfully simulate the stub build');
        process.exit(2);
    }
}

// ============================================================================
// Install Module._load interception so that any .node load originating from
// inside par3gen.js returns the stub instead of trying to dlopen the real
// (real-but-buggy-from-our-perspective) addon. Also patch fs.existsSync so
// getGf64Binding() believes a candidate .node file is present, otherwise it
// short-circuits without ever calling require() and our interception never
// fires.
// ============================================================================
var buildDir = path.resolve(__dirname, '..', 'build', 'Release');
var candidates = ['parpar_gf64.node', 'parpar_gf64_native.node', 'gf64_addon.node'];
var origExistsSync = fs.existsSync;
fs.existsSync = function (p) {
    for (var j = 0; j < candidates.length; j++) {
        if (p === path.join(buildDir, candidates[j])) {
            return true;
        }
    }
    return origExistsSync.apply(fs, arguments);
};

var origLoad = Module._load;
Module._load = function (request, parent, isMain) {
    if (parent && parent.filename &&
        parent.filename.indexOf('par3gen.js') !== -1 &&
        typeof request === 'string' && /\.node$/.test(request)) {
        return stubBinding;
    }
    return origLoad.apply(this, arguments);
};

// ============================================================================
// Load par3gen.js with the stub binding installed. This is what makes the
// create + repair code paths take the pure-JS fallback. Mirrors the pattern
// from test/par3-stub-binding-create.js.
// ============================================================================
var par3genPath = path.resolve(__dirname, '..', 'lib', 'par3gen.js');
delete require.cache[par3genPath];
var par3;
try {
    par3 = require(par3genPath);
} catch (e) {
    console.error('FATAL: could not require lib/par3gen.js: ' + (e && e.stack || e));
    restore();
    process.exit(1);
}

// ============================================================================
// Set up an isolated tmp directory with real input files (small + distinct).
// 4 files x 4096 bytes each at blockSize=4096 -> 4 input blocks (indices
// 0..3). Recovery slices = 2 gives RECOVERY blocks 4..5.
// ============================================================================
var tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), 'par3-stub-repair-'));
console.log('Setup: tmpdir = ' + tmpdir + '\n');

var BLOCK_SIZE = 4096;
var NUM_INPUT_BLOCKS = 4;
var RECOVERY_SLICES = 2;

var inputFiles = [];
function makeInputFile(name, seed) {
    var p = path.join(tmpdir, name);
    var buf = Buffer.alloc(BLOCK_SIZE);
    for (var i = 0; i < buf.length; i++) {
        buf[i] = ((i + seed * 31) * 7 + 11) & 0xff;
    }
    buf.write('PAR3-STUB-REPAIR-' + name, 0, 'ascii');
    fs.writeFileSync(p, buf);
    inputFiles.push(p);
    return { path: p, data: buf };
}

var f1 = makeInputFile('file1.bin', 1);
var f2 = makeInputFile('file2.bin', 2);
var f3 = makeInputFile('file3.bin', 3);
var f4 = makeInputFile('file4.bin', 4);  // last file -> this DATA packet will be dropped

var archiveBase = path.join(tmpdir, 'archive');
var archiveFile = archiveBase + '.par3';

function clearArchive() {
    try { fs.unlinkSync(archiveFile); } catch (e) { /* ok */ }
}

function runCreate() {
    return new Promise(function(resolve, reject) {
        par3.create(
            inputFiles.slice(),
            archiveBase,
            {
                blockSize: BLOCK_SIZE,
                recoverySlices: RECOVERY_SLICES,
                gfMethod: 'scalar'
            },
            function(err) {
                if (err) reject(err); else resolve();
            }
        );
    });
}

// Corrupt the magic bytes of the archive's last DATA packet so the parser
// skips it entirely. Truncating before the packet would also clip the
// RECOVERY packets (which follow the DATA packets per Amendment 9); a
// targeted magic corruption drops exactly one DATA packet and leaves the
// RECOVERY packets intact so solve_and_reconstruct can run.
function corruptLastDataPacketMagic() {
    var buf = fs.readFileSync(archiveFile);
    var magic = Buffer.from('PAR3\0PKT', 'ascii');
    var dataType = Buffer.from('PAR DAT\0', 'ascii');

    var positions = [];
    var pos = 0;
    while ((pos = buf.indexOf(dataType, pos)) !== -1) {
        var hs = pos - 40;
        if (hs >= 0 && buf.slice(hs, hs + 8).equals(magic)) {
            positions.push(hs);
        }
        pos += dataType.length;
    }
    if (positions.length < NUM_INPUT_BLOCKS) {
        throw new Error('Expected at least ' + NUM_INPUT_BLOCKS +
                        ' DATA packets in archive, found ' + positions.length);
    }
    var lastDataStart = positions[positions.length - 1];
    buf.write('XXX3\0PKT', lastDataStart, 8, 'ascii');
    fs.writeFileSync(archiveFile, buf);
    return positions.length;
}

function runRepair() {
    return new Promise(function(resolve, reject) {
        var outputDir = path.join(tmpdir, 'repaired');
        try { fs.mkdirSync(outputDir, { recursive: true }); } catch (e) {}
        par3.repair(archiveFile, outputDir, { verbose: 1 }, function(err, result) {
            if (err) return reject(err);
            resolve({ outputDir: outputDir, result: result });
        });
    });
}

function readBlock(outputDir, blockIndex) {
    var p = path.join(outputDir, 'block_' + blockIndex + '.dat');
    return fs.readFileSync(p);
}

async function main() {
    var createE = null, repairE = null;

    // ===== Step 1: Create with stub binding =====
    console.log('Step 1: Create PAR3 archive (stub binding via par3.create)...');
    clearArchive();
    try {
        await runCreate();
    } catch (e) { createE = e; }
    check(createE === null,
        'par3.create() completes with stub binding' +
            (createE ? ' (caught: ' + (createE.name || 'Error') + ': ' + (createE.message || createE) + ')' : ''));
    var archiveExists = fs.existsSync(archiveFile);
    check(archiveExists, 'archive created at ' + archiveFile);
    if (!archiveExists) {
        return summarize();
    }
    var archiveSize = fs.statSync(archiveFile).size;
    console.log('  archive size: ' + archiveSize + ' bytes (' + NUM_INPUT_BLOCKS +
                ' input blocks, ' + RECOVERY_SLICES + ' recovery slices)');

    console.log('\nStep 2: Corrupt magic of last DATA packet (simulate missing block)...');
    var dataPackets;
    try {
        dataPackets = corruptLastDataPacketMagic();
    } catch (e) {
        check(false, 'corruptLastDataPacketMagic failed: ' + e.message);
        return summarize();
    }
    console.log('  corrupted 1 of ' + dataPackets + ' DATA packet headers');
    check(dataPackets === NUM_INPUT_BLOCKS,
        'found expected number of DATA packets (' + dataPackets + ')');

    console.log('\nStep 3: par3.repair() with stub binding...');
    var repairResult;
    try {
        repairResult = await runRepair();
    } catch (e) { repairE = e; }
    check(repairE === null,
        'par3.repair() completes without error' +
            (repairE ? ' (caught: ' + (repairE.name || 'Error') + ': ' + (repairE.message || repairE) + ')' : ''));

    if (repairE) {
        console.log('\nRepair raised: ' + repairE.message);
        console.log('This is the TDD-red failure mode for Bug A or Bug B.');
        return summarize();
    }

    var outputDir = repairResult.outputDir;
    console.log('  repair result: ' + JSON.stringify(repairResult.result && {
        blocksRepaired: repairResult.result.blocksRepaired,
        missingBlocks: repairResult.result.missingBlocks,
        missingBlockList: repairResult.result.missingBlockList,
        availableRecoveryBlocks: repairResult.result.availableRecoveryBlocks
    }));

    // ===== Step 4: Read the reconstructed block =====
    var missingIdx = NUM_INPUT_BLOCKS - 1;  // 3, the truncated block
    var blockPath = path.join(outputDir, 'block_' + missingIdx + '.dat');
    var blockExists = fs.existsSync(blockPath);
    check(blockExists, 'repaired block_' + missingIdx + '.dat exists at ' + blockPath);

    if (!blockExists) {
        console.log('\nNo repaired block file produced — repair path failed to write any output.');
        return summarize();
    }

    var blockBuf = readBlock(outputDir, missingIdx);
    var blockSha = crypto.createHash('sha256').update(blockBuf).digest('hex');
    var originalSha = crypto.createHash('sha256').update(f4.data).digest('hex');
    console.log('  Reconstructed block_' + missingIdx + '.dat SHA256: ' + blockSha);
    console.log('  Original       file4.bin         SHA256: ' + originalSha);
    console.log('  Reconstructed  size: ' + blockBuf.length + ' bytes');
    console.log('  Original       size: ' + f4.data.length + ' bytes');

    // ===== Step 5: SHA256 compare =====
    console.log('\nStep 4: Compare reconstructed block against original file4.bin...');
    check(blockSha === originalSha,
        'reconstructed block SHA256 matches original file4.bin SHA256 ' +
            '(got ' + blockSha + ', want ' + originalSha + ')');

    check(blockBuf.length === f4.data.length,
        'reconstructed block size matches original (' + blockBuf.length +
            ' vs ' + f4.data.length + ')');

    return summarize();
}

function summarize() {
    console.log('\n---');
    if (failed > 0) {
        console.log('FAILED (' + failed + ' failure(s), ' + passed + ' passed, ' + total + ' total)');
        console.log('Failure list:');
        failures.forEach(function(f) { console.log('  - ' + f); });
        console.log('');
        console.log('This is EXPECTED on master HEAD 475d5b6. The repair path is');
        console.log('exercising a fully stub binding which exposes two pre-existing');
        console.log('bugs in lib/par3gen.js:');
        console.log('  Bug B: START packet writes block_size at wrong body offset');
        console.log('         so repair parses block_size = 0 (empty/wrong blocks).');
        console.log('  Bug A: repair builds rhsBlocks without XORing in the');
        console.log('         contribution of known input blocks (Cauchy RHS bug).');
        console.log('After T4 applies both fixes, this test should PASS.');
        restore();
        try { fs.rmSync(tmpdir, { recursive: true, force: true }); } catch (e) {}
        process.exit(1);
    } else {
        console.log('PASS (' + passed + ' passed, ' + total + ' total)');
        console.log('Stub-binding repair round-trip is byte-correct.');
        restore();
        try { fs.rmSync(tmpdir, { recursive: true, force: true }); } catch (e) {}
    }
}

main().catch(function(e) {
    console.error('FATAL in main(): ' + (e && e.stack || e));
    restore();
    try { fs.rmSync(tmpdir, { recursive: true, force: true }); } catch (e) {}
    process.exit(1);
});