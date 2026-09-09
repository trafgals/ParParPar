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

function extractRecBodies(buf) {
    var out = [];
    var off = 0;
    while (off + 48 <= buf.length) {
        if (buf.slice(off, off + 8).toString("latin1") !== "PAR3\u0000PKT") break;
        var len = Number(buf.readBigUInt64LE(off + 24));
        var type = buf.slice(off + 40, off + 48).toString("latin1");
        if (type === "PAR REC\u0000") out.push(buf.slice(off + 48, off + len));
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
    var baseRecs = extractRecBodies(fs.readFileSync(basePar3));
    assert(baseRecs.length === RECOVERY * (16 + BLOCK_SIZE), "Baseline generated " + RECOVERY + " recovery blocks");

    // 2. Create with strict 256 KiB memory limit (forces 64 chunks of 32 blocks each)
    process.env.PAR3_MEMORY_LIMIT = "131072"; // 128 KiB chunk budget

    par3gen.create([inFile], path.join(tmp, "chunked"), {
        blockSize: BLOCK_SIZE,
        recoverySlices: RECOVERY
    }, function(err2) {
        delete process.env.PAR3_MEMORY_LIMIT;
        assert(!err2, "Chunked create completed without error");
        var chunkedRecs = extractRecBodies(fs.readFileSync(chunkedPar3));
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
            var baseRecs2 = extractRecBodies(fs.readFileSync(path.join(tmp, "unaligned_base.par3")));

            process.env.PAR3_MEMORY_LIMIT = "16384"; // 16 KiB limit = 4 blocks per chunk
            par3gen.create([inFile2], path.join(tmp, "unaligned_chunked"), {
                blockSize: BLOCK_SIZE,
                recoverySlices: 4
            }, function(err4) {
                delete process.env.PAR3_MEMORY_LIMIT;
                assert(!err4, "Unaligned chunked create completed without error");
                var chunkedRecs2 = extractRecBodies(fs.readFileSync(path.join(tmp, "unaligned_chunked.par3")));

                assert(baseRecs2.equals(chunkedRecs2), "Unaligned chunked recovery packets are bit-identical to unaligned baseline");

                // Cleanup
                try {
                    fs.unlinkSync(inFile);
                    fs.unlinkSync(inFile2);
                    fs.unlinkSync(basePar3);
                    fs.unlinkSync(chunkedPar3);
                    fs.unlinkSync(path.join(tmp, "unaligned_base.par3"));
                    fs.unlinkSync(path.join(tmp, "unaligned_chunked.par3"));
                    fs.rmdirSync(tmp);
                } catch (e) {}

                console.log("\n---");
                console.log("RESULT: " + passed + " passed, " + failed + " failed");
                process.exit(failed === 0 ? 0 : 1);
            });
        });
    });
});
