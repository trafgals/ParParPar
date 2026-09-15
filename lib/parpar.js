"use strict";

var Par2 = require('./par2');
var filescan = require('./filescan');
var utils = require('./utils');

module.exports = Par2._extend({
	version: require('../package').version,
	par2: Par2,
	get par3() { return require('./par3gen'); },
	get PAR3Gen() { return require('./par3gen').PAR3Gen; },
	fileScan: filescan.fileScan,
	utils: utils
}, Par2, require('./par2gen'));

