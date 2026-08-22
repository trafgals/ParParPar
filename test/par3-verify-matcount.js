"use strict";

// Regression test for issue #97: verify double-counting bug.
//
// In par3_verify's packet callback the FILE case did `inputCount++` AFTER
// the CAU/MATRIX body had already set inputCount = last_input - first_input + 1.
// Packet order is CRE -> STA -> CAU -> FIL, so every FILE packet added a
// phantom input block on top of the matrix count. A 16-block archive therefore
// reported 17 input blocks, 1 missing block, and archiveOk === false.
//
// This test:
//   1. Creates a 16 MiB file (16 x 1 MiB blocks) with 8 recovery slices.
//   2. Parses the MATRIX/CAUCHY body straight from the raw .par3 bytes to
//      establish ground truth: expected = last_input - first_input + 1.
//   3. Runs par3.verify and asserts inputBlocks === expected,
//      archiveOk === true, missingBlocks === 0.
//
// On the unfixed code this FAILS (inputBlocks = 17, missingBlocks = 1,
// archiveOk = false). On the fixed code it PASSES.

var path = require('path');
var fs = require('fs');
var os = require('os');
var crypto = require('crypto');
var par3 = require('../lib/par3gen.js');

var BLOCK_SIZE = 1024 * 1024;   // 1 MiB (default)
var NUM_BLOCKS = 16;           // 16 MiB / 1 MiB
var RECOVERY_SLICES = 8;       // pow2 (Fenger fast path, #87)

var PAR3_MAGIC = Buffer.from('PAR3\0PKT');
var PAR3_PKT_HDR_SIZE = 48;
var PKT_MATRIX = 'PAR MAT\0';
var PKT_CAUCHY = 'PAR CAU\0';

function makeTempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'parpar-verify-matcount-'));
}

function rmrf(dir) {
  if (!dir || dir === '/') return;
  try {
    fs.readdirSync(dir).forEach(function(name) {
      var p = path.join(dir, name);
      var stat = fs.statSync(p);
      if (stat.isDirectory()) rmrf(p);
      else fs.unlinkSync(p);
    });
    fs.rmdirSync(dir);
  } catch (e) { /* ignore */ }
}

// Parse the first MATRIX/CAUCHY packet body from the raw .par3 bytes and
// return its input/recovery bounds. This is the ground truth that
// par3_verify's inputBlocks must equal. Mirrors par3_parse_stream framing.
function parseMatrix(par3File) {
  var buf = fs.readFileSync(par3File);
  var offset = 0;
  while (offset + PAR3_PKT_HDR_SIZE <= buf.length) {
    var magic = buf.slice(offset, offset + 8);
    if (!magic.equals(PAR3_MAGIC)) {
      offset += 8;
      continue;
    }
    var pktLen = Number(buf.readBigUInt64LE(offset + 24));
    if (pktLen < PAR3_PKT_HDR_SIZE || offset + pktLen > buf.length) {
      break;
    }
    var body = buf.slice(offset + PAR3_PKT_HDR_SIZE, offset + pktLen);
    var typeStr = buf.slice(offset + 40, offset + 48).toString('ascii');
    if (typeStr === PKT_MATRIX || typeStr === PKT_CAUCHY) {
      return {
        first_input: Number(body.readBigUInt64LE(0)),
        last_input: Number(body.readBigUInt64LE(8)),
        first_recovery: Number(body.readBigUInt64LE(16)),
        recovery_count: body.length >= 32 ? Number(body.readBigUInt64LE(24)) : undefined
      };
    }
    offset += pktLen;
  }
  throw new Error('No MATRIX/CAUCHY packet found in ' + par3File);
}

function assertEq(actual, expected, msg) {
  if (actual !== expected) {
    throw new Error(msg + '\n  Expected: ' + expected + '\n  Got:      ' + actual);
  }
}

function run() {
  var tempDir = makeTempDir();
  var testFile = path.join(tempDir, 'test.bin');
  var outputBase = path.join(tempDir, 'out');
  var par3File = outputBase + '.par3';

  try {
    // 1) 16 MiB random test file -> 16 x 1 MiB blocks
    var data = crypto.randomBytes(NUM_BLOCKS * BLOCK_SIZE);
    fs.writeFileSync(testFile, data);

    // 2) Create a PAR3 archive with 8 recovery slices
    par3.create([testFile], outputBase, {
      outputBase: outputBase,
      recoverySlices: RECOVERY_SLICES
    }, function(err) {
      // P3 cubic-review fix: callback body runs after the outer try has
      // returned, so an uncaught throw here would skip rmrf and exit
      // via uncaughtException instead of the 'TEST FAILED:' handler.
      try {
        if (err) {
          console.error('par3.create failed:', err.message);
          rmrf(tempDir);
          process.exit(2);
          return;
        }

        // 3) Ground truth straight from the archive bytes
        var matrix = parseMatrix(par3File);
        var expectedInputBlocks = matrix.last_input - matrix.first_input + 1;
        console.log('Matrix: first_input=' + matrix.first_input +
          ' last_input=' + matrix.last_input +
          ' first_recovery=' + matrix.first_recovery +
          ' recovery_count=' + matrix.recovery_count);

        // Sanity: the 16 MiB file must have produced exactly 16 blocks.
        assertEq(expectedInputBlocks, NUM_BLOCKS,
          'archive matrix does not reflect the 16-block input file');

        // 4) Verify
        par3.verify(par3File, function(err2, verifyResult) {
          try {
            rmrf(tempDir);
            if (err2) {
              console.error('par3.verify failed:', err2.message);
              process.exit(2);
              return;
            }

            console.log('Verify result:');
            console.log('  inputBlocks:    ' + verifyResult.inputBlocks);
            console.log('  recoveryBlocks: ' + verifyResult.recoveryBlocks);
            console.log('  missingBlocks:  ' + verifyResult.missingBlocks);
            console.log('  archiveOk:      ' + verifyResult.archiveOk);
            console.log('  missingBlockList: ' + JSON.stringify(verifyResult.missingBlockList));

            // The assertions that pin the #97 contract:
            assertEq(verifyResult.inputBlocks, expectedInputBlocks,
              'verify inputBlocks must equal matrix (last_input - first_input + 1)');
            assertEq(verifyResult.recoveryBlocks, RECOVERY_SLICES,
              'verify recoveryBlocks must equal the recovery slice count');
            assertEq(verifyResult.missingBlocks, 0,
              'intact archive must report zero missing blocks');
            if (verifyResult.archiveOk !== true) {
              throw new Error('intact archive must report archiveOk === true (got ' + verifyResult.archiveOk + ')');
            }

            console.log('TEST PASSED — verify input count matches matrix (no phantom block)');
            process.exit(0);
          } catch (cbErr) {
            rmrf(tempDir);
            console.error('TEST FAILED:', cbErr.message);
            process.exit(1);
          }
        });
      } catch (cbErr) {
        rmrf(tempDir);
        console.error('TEST FAILED:', cbErr.message);
        process.exit(1);
      }
    });
  } catch (err) {
    rmrf(tempDir);
    console.error('TEST FAILED:', err.message);
    process.exit(1);
  }
}

run();
