var path = require('path');
var a = require(path.join(__dirname, 'build/Release/parpar_gf64.node'));
console.log('LOAD OK');
var keys = Object.keys(a).sort();
console.log('keys:', keys.join(','));
var info = a.gf64_info ? a.gf64_info(0) : null;
console.log('gf64_info(0):', JSON.stringify(info));
// Quick kernel test
if (a.Gf64Encoder && a.Gf64Encoder_create) {
  var enc = a.Gf64Encoder_create(0);
  var inBuf = a.allocAlignedBuffer(4096);
  var outBuf = a.allocAlignedBuffer(4096);
  console.log('Gf64Encoder: ok, allocAlignedBuffer: ok');
}
process.exit(0);
