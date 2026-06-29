#!/usr/bin/env node
"use strict";

// ============================================================================
// PAR2 Stub-Binding Source Test (par3-finish-ci-fixes — T1, TDD red)
//
// Asserts that src/gf_stub.cc exists and conforms to the minimal NAPI stub
// shape used as the Windows fallback for the parpar_gf target. Mirrors
// src/gf64_stub.cc:
//
//   #include <node_api.h>
//   napi_value Init(napi_env env, napi_value exports) { return exports; }
//   NAPI_MODULE(<name>, Init)
//
// The stub must NOT include any OpenCL / Windows headers — including them
// would re-trigger the MSVC error:
//
//   "CL/cl.h": No such file or directory
//   fatal error C1083: Cannot open include file: 'CL/cl.hpp'
//
// Reference template: src/gf64_stub.cc (3 lines).
// Companion fix (T2): wraps binding.gyp:67-111 in `OS!="win"` and adds this
// stub as the Windows source.
// ============================================================================

var fs = require('fs');
var path = require('path');

var STUB_PATH = path.resolve(__dirname, '..', 'src', 'gf_stub.cc');

console.log('PAR2 Stub-Binding Source Test');
console.log('==============================\n');
console.log('Stub path: ' + STUB_PATH + '\n');

var passed = 0;
var failed = 0;
var failures = [];

function check(condition, msg) {
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

var source;
try {
    source = fs.readFileSync(STUB_PATH, 'utf8');
} catch (e) {
    // TDD red path: file does not exist yet. T2 must create it from the
    // src/gf64_stub.cc template.
    check(false, 'src/gf_stub.cc not found — T2 must create it before this test passes (readFileSync threw: ' +
        (e && e.code ? e.code : (e && e.message ? e.message : String(e))) + ')');

    console.log('\n---');
    console.log('FAILED (' + failed + ' failure(s), ' + passed + ' passed, ' + (passed + failed) + ' total)');
    failures.forEach(function(f) { console.log('  - ' + f); });
    process.exit(1);
}

// --- File exists ---
check(true, 'src/gf_stub.cc exists and is readable (' + source.length + ' bytes)');

// --- NAPI stub shape (mirrors src/gf64_stub.cc) ---
check(/^\s*#\s*include\s*<\s*node_api\.h\s*>/m.test(source),
    'stub includes <node_api.h>');

check(/\bNAPI_MODULE\s*\(\s*parpar_gf\s*,/.test(source),
    'stub registers NAPI_MODULE(parpar_gf, ...)');

check(/\bnapi_value\s+Init\s*\(\s*napi_env\s+env\s*,\s*napi_value\s+exports\s*\)\s*\{\s*return\s+exports\s*;?\s*\}/m.test(source),
    'stub defines napi_value Init(napi_env, napi_value) { return exports; }');

// --- Excludes Windows-specific / OpenCL headers that would re-trigger the
//     MSVC error on Windows builds. The stub is the FALLBACK source, so it
//     must not drag in any platform-specific GPU headers. ---
var forbiddenIncludes = [
    '<CL/cl.h>',
    '<CL/cl.hpp>',
    '<CL/opencl.h>',
    '<CL/opencl.hpp>',
    '<windows.h>',
    '<d3d11.h>',
    '<d3d12.h>'
];
for (var i = 0; i < forbiddenIncludes.length; i++) {
    var hdr = forbiddenIncludes[i];
    var re = new RegExp('^\\s*#\\s*include\\s*["<]\\s*' +
        hdr.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/"/g, '').replace(/[<>]/g, '') +
        '\\s*[">]', 'm');
    check(!re.test(source), 'stub does NOT include ' + hdr);
}

// --- Summary ---
console.log('\n---');
if (failed > 0) {
    console.log('FAILED (' + failed + ' failure(s), ' + passed + ' passed, ' + (passed + failed) + ' total)');
    failures.forEach(function(f) { console.log('  - ' + f); });
    process.exit(1);
} else {
    console.log('PASS (' + passed + ' passed, ' + (passed + failed) + ' total)');
    console.log('src/gf_stub.cc conforms to the minimal NAPI stub shape — safe Windows fallback for parpar_gf.');
}