#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR3 chunked-recovery end-to-end parity test
// ----------------------------------------------------------------------------
// Verifies that par3gen.create() with bounded PAR3_MEMORY_LIMIT (chunked input
// accumulation) produces 100% bit-identical recovery packets to the unconstrained
// single-buffer create path, while never exhausting memory or buffer pools.
// ============================================================================

var fs = require("fs");
var path = require("path");
var os = require("os");
var crypto = require("crypto");
var par3gen = require("../lib/par3gen.js");

var passed = 0;
var failed = 0;

function assert(condition, msg) {
    if (condition) {
        console.log("  PASS: " + msg);
        passed++;
    } else {
        console.error("  FAIL: " + msg);
        failed++;
        process.exitCode = 1;
    }
}

// Cubic review P3 (round 4): collapse the near-duplicate extractDataBodies
//   and extractRecBodies parsers into a single parameterized helper. The two
//   functions differed only by (a) the matched type string and (b) the body
//   slice offset — exactly the kind of DRY violation that lets a parser fix
//   land in one and not the other. Single source of truth: extractBodies(buf,
//   type, bodyOffset).
function extractBodies(buf, type, bodyOffset) {
    var out = [];
    var off = 0;
    while (off + 48 <= buf.length) {
        if (buf.slice(off, off + 8).toString("latin1") !== "PAR3\u0000PKT") break;
        var len = Number(buf.readBigUInt64LE(off + 24));
        var t = buf.slice(off + 40, off + 48).toString("latin1");
        if (t === type) out.push(buf.slice(off + bodyOffset, off + len));
        off += len;
    }
    return Buffer.concat(out);
}

console.log("PAR3 Chunked Recovery End-to-End Parity Test");
console.log("=============================================\n");

var tmp = fs.mkdtempSync(path.join(os.tmpdir(), "par3-chunk-test-"));
var inFile = path.join(tmp, "test-input.bin");

var BLOCK_SIZE = 4096;
var NUM_BLOCKS = 2048; // 8 MiB
var RECOVERY = 8;     // 8 recovery blocks (small-R Zen4 benchmark shape)

var fd = fs.openSync(inFile, "w");
var pattern = crypto.randomBytes(BLOCK_SIZE);
for (var i = 0; i < NUM_BLOCKS; i++) {
    fs.writeSync(fd, pattern, 0, BLOCK_SIZE);
}
fs.closeSync(fd);

var basePar3 = path.join(tmp, "base.par3");
var chunkedPar3 = path.join(tmp, "chunked.par3");

// 1. Create unconstrained baseline
delete process.env.PAR3_MEMORY_LIMIT;
delete process.env.PAR3_SIMULATED_BUFFER_CAP;

par3gen.create([inFile], path.join(tmp, "base"), {
    blockSize: BLOCK_SIZE,
    recoverySlices: RECOVERY
}, function(err1) {
    assert(!err1, "Baseline create completed without error");
    var baseRecs = extractBodies(fs.readFileSync(basePar3), "PAR REC\u0000", 48);
    assert(baseRecs.length === RECOVERY * (16 + BLOCK_SIZE), "Baseline generated " + RECOVERY + " recovery blocks");

    // 2. Create with strict 128 KiB memory limit (forces chunked accumulation)
    process.env.PAR3_MEMORY_LIMIT = "131072"; // 128 KiB chunk budget

    par3gen.create([inFile], path.join(tmp, "chunked"), {
        blockSize: BLOCK_SIZE,
        recoverySlices: RECOVERY
    }, function(err2) {
        delete process.env.PAR3_MEMORY_LIMIT;
        assert(!err2, "Chunked create completed without error");
        var chunkedRecs = extractBodies(fs.readFileSync(chunkedPar3), "PAR REC\u0000", 48);
        assert(chunkedRecs.length === RECOVERY * (16 + BLOCK_SIZE), "Chunked generated " + RECOVERY + " recovery blocks");

        var eq = baseRecs.equals(chunkedRecs);
        assert(eq, "Chunked recovery packets are 100% bit-identical to baseline");

        // Leg 2: Unaligned file with partial trailing block
        var inFile2 = path.join(tmp, "unaligned.bin");
        var UNALIGNED_SIZE = 123456; // non-power-of-2, not a multiple of BLOCK_SIZE (4096)
        var fd2 = fs.openSync(inFile2, "w");
        fs.writeSync(fd2, crypto.randomBytes(UNALIGNED_SIZE), 0, UNALIGNED_SIZE);
        fs.closeSync(fd2);

        delete process.env.PAR3_MEMORY_LIMIT;
        par3gen.create([inFile2], path.join(tmp, "unaligned_base"), {
            blockSize: BLOCK_SIZE,
            recoverySlices: 4
        }, function(err3) {
            assert(!err3, "Unaligned baseline create completed without error");
            var baseRecs2 = extractBodies(fs.readFileSync(path.join(tmp, "unaligned_base.par3")), "PAR REC\u0000", 48);

            process.env.PAR3_MEMORY_LIMIT = "16384"; // 16 KiB limit = 4 blocks per chunk
            par3gen.create([inFile2], path.join(tmp, "unaligned_chunked"), {
                blockSize: BLOCK_SIZE,
                recoverySlices: 4
            }, function(err4) {
                delete process.env.PAR3_MEMORY_LIMIT;
                assert(!err4, "Unaligned chunked create completed without error");
                var chunkedRecs2 = extractBodies(fs.readFileSync(path.join(tmp, "unaligned_chunked.par3")), "PAR REC\u0000", 48);

                assert(baseRecs2.equals(chunkedRecs2), "Unaligned chunked recovery packets are bit-identical to unaligned baseline");

                // Cubic review P2: verify DATA packets match file size exactly without trailing garbage padding
                var baseData2 = extractBodies(fs.readFileSync(path.join(tmp, "unaligned_base.par3")), "PAR DAT\u0000", 56);
                var chunkedData2 = extractBodies(fs.readFileSync(path.join(tmp, "unaligned_chunked.par3")), "PAR DAT\u0000", 56);
                assert(baseData2.length === UNALIGNED_SIZE, "Baseline DATA packets total length equals exact unaligned file size (" + UNALIGNED_SIZE + " bytes)");
                assert(chunkedData2.length === UNALIGNED_SIZE, "Chunked DATA packets total length equals exact unaligned file size (" + UNALIGNED_SIZE + " bytes)");
                assert(baseData2.equals(chunkedData2), "Unaligned chunked DATA packets are bit-identical to baseline");

                // Leg 3 (Cubic review P1): Fenger chunked accumulation parity
                console.log("\n=== Leg 3: Fenger chunked accumulation ===");
                var fengerIn = path.join(tmp, "fenger_in.bin");
                var fengerBlocks = 64;
                var fengerBlockSize = 1024;
                var fengerRecovery = 16;
                var fd3 = fs.openSync(fengerIn, "w");
                for (var i = 0; i < fengerBlocks; i++) {
                    fs.writeSync(fd3, crypto.randomBytes(fengerBlockSize));
                }
                fs.closeSync(fd3);

                // Cubic review P2: Force Fenger kernel routing and assert routing decision
                process.env.PAR3_GF64_USE_FENGER = "1";
                delete process.env.PAR3_MEMORY_LIMIT;

                var routing = par3gen.decideRecoveryKernel(fengerBlocks, fengerRecovery, fengerBlockSize, require("../build/Release/parpar_gf64.node"));
                assert(routing.kernel === "fenger", "decideRecoveryKernel selected fenger kernel (" + routing.reason + ")");

                par3gen.create([fengerIn], path.join(tmp, "fenger_base"), {
                    blockSize: fengerBlockSize,
                    recoverySlices: fengerRecovery
                }, function(err5) {
                    assert(!err5, "Fenger baseline create completed without error");
                    var fengerBaseRecs = extractBodies(fs.readFileSync(path.join(tmp, "fenger_base.par3")), "PAR REC\u0000", 48);

                    process.env.PAR3_MEMORY_LIMIT = "49152"; // 48 KiB limit forces chunked accumulation (leaves 16 KiB input chunk after 32 KiB 2x recovery reserve)
                    par3gen.create([fengerIn], path.join(tmp, "fenger_chunked"), {
                        blockSize: fengerBlockSize,
                        recoverySlices: fengerRecovery
                    }, function(err6) {
                        delete process.env.PAR3_MEMORY_LIMIT;
                        delete process.env.PAR3_GF64_USE_FENGER;
                        assert(!err6, "Fenger chunked create completed without error");
                        var fengerChunkRecs = extractBodies(fs.readFileSync(path.join(tmp, "fenger_chunked.par3")), "PAR REC\u0000", 48);

                        assert(fengerBaseRecs.equals(fengerChunkRecs), "Fenger chunked recovery matches Fenger single-pass bit-identically");

                        // Cleanup
                        try {
                            fs.readdirSync(tmp).forEach(function(f) { fs.unlinkSync(path.join(tmp, f)); });
                            fs.rmdirSync(tmp);
                        } catch (e) {}

                        console.log("\n---");
                        console.log("RESULT: " + passed + " passed, " + failed + " failed");
                        process.exitCode = failed === 0 ? 0 : 1;
                        if (typeof par3gen.shutdownHashPool === 'function') {
                            par3gen.shutdownHashPool();
                        }
                    });
                });
            });
        });
    });
});
