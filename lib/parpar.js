"use strict";

var Par2 = require('./par2');
var Par3 = require('./par3gen');
var filescan = require('./filescan');
var utils = require('./utils');

module.exports = Par2._extend({
	version: require('../package').version,
	par2: Par2,
	par3: Par3,
	PAR3Gen: Par3.PAR3Gen,
	fileScan: filescan.fileScan,
	utils: utils
}, Par2, require('./par2gen'));
