"use strict";
// Run the library's OWN verify + repair on a known-correct archive and
// compare against the e2e's reported numbers. This isolates whether the
// 4101/1-missing is a verify miscount or an archive-content difference.
var par3 = require('./lib/par3gen.js');
var ARCHIVE = process.argv[2];
if (!ARCHIVE) { console.error('usage: node diag_verify.js <path.par3>'); process.exit(2); }

console.log('=== VERIFY (lib) on', ARCHIVE, '===');
par3.verify(ARCHIVE, function(err, r) {
  if (err) { console.error('VERIFY ERR:', err.message); process.exit(1); }
  console.log('  verified      =', r.verified);
  console.log('  archiveOk     =', r.archiveOk);
  console.log('  canRepair     =', r.canRepair);
  console.log('  inputBlocks   =', r.inputBlocks);
  console.log('  recoveryBlocks=', r.recoveryBlocks);
  console.log('  missingBlocks =', r.missingBlocks);
  console.log('  missingList   =', JSON.stringify((r.missingBlockList||[]).slice(0,10)));
  console.log('  blockSize     =', r.blockSize);
  process.exit(0);
});
