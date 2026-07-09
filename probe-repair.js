var path = require('path');
var a = require(path.join(__dirname, 'build/Release/parpar_gf64.node'));
var helpers = require(path.join(__dirname, 'test/bench/bench-helpers'));
var par3 = require(path.join(__dirname, 'lib/par3gen.js'));
var fs = require('fs');
var e2eHelpers = require(path.join(__dirname, 'test/e2e/helpers'));
var tmp = helpers.getTempDir('r');
var src = path.join(tmp, 'src.bin');
var outBase = path.join(tmp, 'arc');
helpers.createBenchSource(1024*1024*1024, src); // 1 GiB
console.log('1: src written');
var t = Date.now();
par3.create([src], outBase, { recoverySlices: 10000 }, function(err) {
  if (err) { console.log('create failed:', err); process.exit(1); }
  console.log('2: create done in', Date.now()-t, 'ms');
  var sliceSize = Math.ceil(1024*1024*1024/10000);
  e2eHelpers.deleteRandomSlices(src, sliceSize, 500);
  console.log('3: corrupt 500 of 10000 slices');
  var t2 = Date.now();
  par3.repair(outBase + '.par3', tmp, {}, function(err2, result) {
    var dt = Date.now()-t2;
    var mbps = (1024/dt)*1000;
    console.log('4: repair done in', dt, 'ms,', mbps.toFixed(1), 'MB/s, hash match='+(result && result.reconstructed));
    process.exit(err2 ? 1 : 0);
  });
});
