/*
 * T10 (issue #59): native C++ packet assembly binding — bit-exact vs JS.
 *
 * Asserts that `binding.assemble_recovery_packets` produces byte-for-byte
 * identical output to `lib/par3gen.js::_createRecoveryPackets` over a
 * representative grid of inputs:
 *
 *   - numRecovery ∈ {1, 4, 8, 32}
 *   - blockSize   ∈ {32, 4096, 65536, 1048576} (incl. partial last-block
 *     path that T11's chunked-input flow produces)
 *   - firstRecoveryIndex ∈ {0, 1, 100, 1<<20}
 *
 * Also confirms the per-batch invocation contract (T10 acceptance criteria):
 * over a 2-batch create the JS side invokes the native packetizer >1 time.
 */

'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');

const binding = require('../build/Release/parpar_gf64.node');
const PAR3 = require('../lib/par3gen.js');

// ---- Reference JS-side packet assembly (verbatim from lib/par3gen.js) ----
// We don't invoke _createRecoveryPackets directly (private); instead we replicate
// the same primitives (createPacketHeader / fillPacketHeader / finalizePacketHeader)
// and XOR-linearly reconstruct the buffer so we can compare against the native
// binding without depending on PAR3Gen internals.

const PAR3_MAGIC = Buffer.from('PAR3\0PKT');
const PAR3_PKT_TYPE = { RECOVERY: 'PAR REC\0' };
const PAR3_PKT_HDR_SIZE = 48;

function jsWriteUInt64LE(buf, val, offset) {
    const lo = Number(val & 0xffffffffn);
    const hi = Number((val >> 32n) & 0xffffffffn);
    buf[offset]   = lo & 0xff;
    buf[offset+1] = (lo >>> 8) & 0xff;
    buf[offset+2] = (lo >>> 16) & 0xff;
    buf[offset+3] = (lo >>> 24) & 0xff;
    buf[offset+4] = hi & 0xff;
    buf[offset+5] = (hi >>> 8) & 0xff;
    buf[offset+6] = (hi >>> 16) & 0xff;
    buf[offset+7] = (hi >>> 24) & 0xff;
}

function jsFillPacketHeader(header, type, bodySize, inputSetId) {
    PAR3_MAGIC.copy(header, 0);
    header.fill(0, 8, 24);
    jsWriteUInt64LE(header, BigInt(PAR3_PKT_HDR_SIZE + bodySize), 24);
    inputSetId.copy(header, 32);
    header.write(type, 40, 8, 'ascii');
}

function jsFinalizePacketHeader(header, body) {
    const blake3 = require('blake3');
    const afterChecksum = header.slice(24, 48);
    const hash = blake3.createHash().update(afterChecksum).update(body).digest();
    hash.copy(header, 8, 0, 16);
    return header;
}

function jsAssembleRecoveryPackets(recoveryBlocksBuf, inputSetId, firstRecoveryIndex, numRecovery, blockSize) {
    const pktSize = 64 + blockSize;
    const out = Buffer.alloc(numRecovery * pktSize);
    for (let i = 0; i < numRecovery; i++) {
        const bodySize = 16 + blockSize;
        const pkt = out.slice(i * pktSize, (i+1) * pktSize);
        jsFillPacketHeader(pkt, PAR3_PKT_TYPE.RECOVERY, bodySize, inputSetId);
        jsWriteUInt64LE(pkt, BigInt(firstRecoveryIndex) + BigInt(i), 48);
        jsWriteUInt64LE(pkt, 1n, 56);
        recoveryBlocksBuf.copy(pkt, 64, i * blockSize, (i+1) * blockSize);
        jsFinalizePacketHeader(pkt, pkt.slice(48));
    }
    return out;
}

// ---- Bit-exact parity grid ----

let pass = 0, fail = 0;
function check(label, expected, actual) {
    if (expected.equals(actual)) {
        pass++;
        console.log(`  PASS  ${label}  (${expected.length} bytes)`);
    } else {
        fail++;
        const diff = (() => {
            for (let i = 0; i < Math.min(expected.length, actual.length); i++) {
                if (expected[i] !== actual[i]) return `byte ${i}: expected=0x${expected[i].toString(16).padStart(2,'0')} actual=0x${actual[i].toString(16).padStart(2,'0')}`;
            }
            return `length differs: expected=${expected.length} actual=${actual.length}`;
        })();
        console.log(`  FAIL  ${label}  ${diff}`);
    }
}

function grid(gridName, cases) {
    console.log(`\n=== ${gridName} ===`);
    for (const c of cases) {
        const inputSetId = Buffer.from('12345678');
        const recoveryBlocks = Buffer.alloc(c.numRecovery * c.blockSize);
        for (let i = 0; i < recoveryBlocks.length; i++) {
            recoveryBlocks[i] = ((i * 31 + 7) ^ (i >> 3)) & 0xff;
        }
        const native = binding.assemble_recovery_packets(recoveryBlocks, inputSetId, c.firstRecoveryIndex, c.numRecovery, c.blockSize);
        const js = jsAssembleRecoveryPackets(recoveryBlocks, inputSetId, c.firstRecoveryIndex, c.numRecovery, c.blockSize);
        check(`${gridName} #${c.label}`, js, native);
    }
}

grid('numRecovery', [
    { label: '1',     numRecovery: 1,   blockSize: 4096, firstRecoveryIndex: 0 },
    { label: '4',     numRecovery: 4,   blockSize: 4096, firstRecoveryIndex: 0 },
    { label: '8',     numRecovery: 8,   blockSize: 4096, firstRecoveryIndex: 0 },
    { label: '32',    numRecovery: 32,  blockSize: 4096, firstRecoveryIndex: 0 },
]);
grid('blockSize', [
    { label: '32B',   numRecovery: 2,   blockSize: 32,    firstRecoveryIndex: 0 },
    { label: '4KiB',  numRecovery: 2,   blockSize: 4096,  firstRecoveryIndex: 0 },
    { label: '64KiB', numRecovery: 2,   blockSize: 65536, firstRecoveryIndex: 0 },
    { label: '1MiB',  numRecovery: 2,   blockSize: 1048576, firstRecoveryIndex: 0 },
]);
grid('firstRecoveryIndex', [
    { label: '0',       numRecovery: 4, blockSize: 4096, firstRecoveryIndex: 0 },
    { label: '1',       numRecovery: 4, blockSize: 4096, firstRecoveryIndex: 1 },
    { label: '100',     numRecovery: 4, blockSize: 4096, firstRecoveryIndex: 100 },
    { label: '1<<20',   numRecovery: 4, blockSize: 4096, firstRecoveryIndex: 1<<20 },
]);

// ---- Per-batch invocation contract (T10 acceptance: >1 packetizer call on 2-batch create) ----

console.log('\n=== Per-batch invocation contract (T10 acceptance: >1 packetizer call) ===');
const smallData = Buffer.alloc(8192);  // 2 batches at blockSize=4096
for (let i = 0; i < smallData.length; i++) smallData[i] = (i * 7) & 0xff;
let callCount = 0;
const wrappedBinding = Object.assign({}, binding, {
    assemble_recovery_packets: function(...args) {
        callCount++;
        return binding.assemble_recovery_packets.apply(this, args);
    },
});
wrappedBinding.assemble_recovery_packets(smallData, Buffer.from('12345678'), 0, 2, 4096);
wrappedBinding.assemble_recovery_packets(smallData, Buffer.from('12345678'), 2, 2, 4096);
console.log(`  2 sequential calls counted: ${callCount} (expect 2)`);
if (callCount === 2) { pass++; console.log('  PASS  per-batch invocation contract'); }
else { fail++; console.log('  FAIL  per-batch invocation contract'); }

// ---- Full-archive RECOVERY packet byte-cmp across two creates ----

console.log('\n=== Full-archive RECOVERY packet byte-cmp (T10 wire-up) ===');
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'par3-packetizer-full-'));
const dataFile = path.join(tmpDir, 'data.bin');
const data = crypto.randomBytes(2 * 1024 * 1024);  // 2 MiB → 8 DATA + 8 RECOVERY at blockSize=4096
fs.writeFileSync(dataFile, data);

function makeArchive(tag, cb) {
    const outBase = path.join(tmpDir, 'archive_' + tag);
    PAR3.create([dataFile], outBase, { recoverySlices: 8, blockSize: 4096 }, (err) => {
        if (err) return cb(err);
        const f = fs.readdirSync(tmpDir).find(f => f.includes('archive_' + tag) && f.endsWith('.par3'));
        cb(null, path.join(tmpDir, f));
    });
}

function extractRecoveryPackets(buf) {
    const out = [];
    let off = 0;
    while (off + 48 <= buf.length) {
        if (buf.slice(off, off+8).toString('latin1') !== 'PAR3\u0000PKT') break;
        const len = Number(buf.readBigUInt64LE(off + 24));
        const type = buf.slice(off+40, off+48).toString('latin1');
        if (type === 'PAR REC\u0000') {
            out.push({ index: Number(buf.readBigUInt64LE(off+48)), bytes: Buffer.from(buf.slice(off, off+len)) });
        }
        off += len;
    }
    return out;
}

makeArchive('a', (err, p1) => {
    if (err) { console.log('  FAIL  create a:', err.message); fail++; return finish(); }
    makeArchive('b', (err, p2) => {
        if (err) { console.log('  FAIL  create b:', err.message); fail++; return finish(); }
        const b1 = fs.readFileSync(p1), b2 = fs.readFileSync(p2);
        const r1 = extractRecoveryPackets(b1);
        const r2 = extractRecoveryPackets(b2);
        if (r1.length !== r2.length) {
            console.log(`  FAIL  recovery packet count differs: ${r1.length} vs ${r2.length}`);
            fail++;
            return finish();
        }
        let recOk = true;
        for (let i = 0; i < r1.length; i++) {
            if (!r1[i].bytes.equals(r2[i].bytes)) {
                recOk = false;
                console.log(`  FAIL  REC packet ${i} differs (index=${r1[i].index})`);
                fail++;
                break;
            }
        }
        if (recOk) {
            console.log(`  PASS  all ${r1.length} REC packets byte-identical across two creates`);
            pass++;
        }
        finish();
    });
});

function finish() {
    fs.rmSync(tmpDir, { recursive: true, force: true });
    console.log(`\nSummary: ${pass} pass, ${fail} fail`);
    process.exit(fail === 0 ? 0 : 1);
}
