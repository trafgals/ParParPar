var path = require('path');
var a = require(path.join(__dirname, 'build/Release/parpar_gf64.node'));
var helpers = require(path.join(__dirname, 'test/bench/bench-helpers'));
var par3 = require(path.join(__dirname, 'lib/par3gen.js'));

var tmp = helpers.getTempDir('p');
var src = path.join(tmp, 'src.bin');
helpers.createBenchSource(4*1024*1024, src);
console.log('1: source written');

console.log('2: creating small archive, method=', a.gf64_info(0));
var t = Date.now();
par3.create([src], path.join(tmp, 'ar'), { recoverySlices: 50, blockSize: 4096 }, function(err) {
  console.log('3: create done in', Date.now()-t, 'ms', err ? 'err:'+err.message : 'ok');
  process.exit(err ? 1 : 0);
});
