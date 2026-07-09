var path = require('path');
var a = require(path.join(__dirname, 'build/Release/parpar_gf64.node'));
console.log('gf64_info(0):', JSON.stringify(a.gf64_info(0)));
console.log('gf64_info(1):', JSON.stringify(a.gf64_info(1)));
console.log('gf64_info(2):', JSON.stringify(a.gf64_info(2)));
console.log('gf64_info(3):', JSON.stringify(a.gf64_info(3)));
console.log('gf64_info(4):', JSON.stringify(a.gf64_info(4)));
console.log('gf64_info(5):', JSON.stringify(a.gf64_info(5)));
