"use strict";

var path = require('path');

var friendlySize = function(s) {
	var units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB'];
	for(var i=0; i<units.length; i++) {
		if(s < 10000) break;
		s /= 1024;
	}
	return (Math.round(s * 100) / 100) + ' ' + units[i];
};

var friendlyCount = function(n) {
	var units = ['', 'K', 'M', 'G', 'T'];
	for(var i=0; i<units.length; i++) {
		if(n < 10000) break;
		n /= 1000;
	}
	return (Math.round(n * 100) / 100) + units[i];
};

var sumSize = function(ar) {
	return ar.reduce(function(sum, e) {
		return sum + e.size;
	}, 0);
};

// normalize path for comparison purposes; cross-platform path separator handling
var pathNormalize, pathToPar;
if(path.sep === '\\') {
	pathNormalize = function(p) {
		return p.replace(/\//g, '\\').toLowerCase();
	};
	pathToPar = function(p) {
		return p.replace(/\\/g, '/');
	};
} else {
	pathToPar = pathNormalize = function(p) {
		return p;
	};
}

module.exports = {
	friendlySize: friendlySize,
	friendlyCount: friendlyCount,
	sumSize: sumSize,
	pathNormalize: pathNormalize,
	pathToPar: pathToPar
};
