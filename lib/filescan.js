"use strict";

var fs = require('fs');
var path = require('path');
var crypto = require('crypto');
var async = require('async');
var BufferPool = require('./bufferpool');

var bufferSlice = Buffer.prototype.readBigInt64BE ? Buffer.prototype.subarray : Buffer.prototype.slice;

var EMPTY_MD5 = Buffer.from('d41d8cd98f00b204e9800998ecf8427e', 'hex');
var EMPTY_SHA256 = Buffer.from('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'hex');

function fileScan(files, recurse, skipSymlinks, concurrency, hashAlgo, cb) {
	if(typeof hashAlgo === 'function') {
		cb = hashAlgo;
		hashAlgo = 'md5';
	}
	if(!cb) {
		cb = concurrency;
		concurrency = null;
		if(!cb) {
			cb = skipSymlinks;
			skipSymlinks = false;
			if(!cb) {
				cb = recurse;
				recurse = false;
			}
		}
	}
	if(!concurrency) concurrency = 2;
	if(!hashAlgo) hashAlgo = 'md5';

	var bufs = new BufferPool([], 16384, concurrency);
	var statFn = skipSymlinks ? fs.lstat : fs.stat;
	var results = [];
	var scanFiles;

	scanFiles = function(fileList, recDepth, cbDoneScan) {
		if(!fileList || fileList.length === 0) return cbDoneScan(null, []);
		var subResults = new Array(fileList.length);
		var items = fileList.map(function(f, idx) {
			return { file: f, idx: idx };
		});

		async.eachLimit(items, concurrency, function(item, cbNextFile) {
			var file = item.file;
			var fileIdx = item.idx;

			bufs.get(function(buf) {
				var info = { name: file, size: 0 };
				if(hashAlgo === 'sha256') {
					info.sha256_16k = null;
				} else {
					info.md5_16k = null;
				}
				var bufHeld = true;
				var fd;

				async.waterfall([
					statFn.bind(fs, file),
					function(stat, cbWf) {
						info.mode = stat.mode;
						if(stat.isDirectory()) {
							info = null;
							if(recDepth) {
								fs.readdir(file, function(err, dirFiles) {
									if(bufHeld) { bufs.put(buf); bufHeld = false; }
									if(err) return cbWf(err);
									scanFiles(dirFiles.map(function(fn) {
										return path.join(file, fn);
									}), typeof recDepth === 'number' ? recDepth - 1 : recDepth, function(err2, dirRes) {
										if(!err2 && dirRes) subResults[fileIdx] = dirRes;
										cbWf(err2 || true);
									});
								});
							} else {
								if(bufHeld) { bufs.put(buf); bufHeld = false; }
								cbWf(true);
							}
							return;
						}
						if(stat.isSymbolicLink()) {
							info = null;
							if(bufHeld) { bufs.put(buf); bufHeld = false; }
							return cbWf(true);
						}
						if(!stat.isFile()) {
							info = null;
							if(bufHeld) { bufs.put(buf); bufHeld = false; }
							return cbWf(new Error(file + ' is not a valid file'));
						}

						info.size = stat.size;
						if(!info.size) {
							if(hashAlgo === 'sha256') {
								info.sha256_16k = Buffer.alloc(32);
							} else {
								info.md5 = info.md5_16k = EMPTY_MD5;
							}
							if(bufHeld) { bufs.put(buf); bufHeld = false; }
							return cbWf(true);
						}
						fs.open(file, 'r', cbWf);
					},
					function(_fd, cbWf) {
						fd = _fd;
						fs.read(fd, buf, 0, 16384, null, cbWf);
					},
					function(bytesRead, buffer, cbWf) {
						var slice = bufferSlice.call(buffer, 0, bytesRead);
						if(hashAlgo === 'sha256') {
							info.sha256_16k = crypto.createHash('sha256').update(slice).digest();
						} else {
							info.md5_16k = crypto.createHash('md5').update(slice).digest();
							if(info.size < 16384) info.md5 = info.md5_16k;
						}
						fs.close(fd, cbWf);
					}
				], function(err) {
					if(bufHeld) {
						bufs.put(buf);
						bufHeld = false;
					}
					if(err && err !== true) {
						return cbNextFile(err);
					}
					if(info) {
						subResults[fileIdx] = [info];
					}
					cbNextFile();
				});
			});
		}, function(err) {
			if(err) return cbDoneScan(err);
			var flattened = [];
			for(var i = 0; i < subResults.length; i++) {
				if(subResults[i]) {
					for(var j = 0; j < subResults[i].length; j++) {
						flattened.push(subResults[i][j]);
					}
				}
			}
			cbDoneScan(null, flattened);
		});
	};

	scanFiles(files, recurse, function(err, fileList) {
		bufs.end(function() {
			if(fileList) {
				for(var i = 0; i < fileList.length; i++) {
					results.push(fileList[i]);
				}
			}
			cb(err, results);
		});
	});

	return results;
}

module.exports = {
	fileScan: fileScan
};
