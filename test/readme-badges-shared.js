"use strict";
/* Shared helpers for the README badge contract tests
 * (test/readme-badge-urls.js, test/readme-pending-row-footnote.js).
 *
 * Centralizes the badge branch raw URL prefix and the URL-decoded
 * badge-id regex. Without this, a change to the branch name or URL
 * scheme would have to be applied in two test files (cubic review
 * 95db4ba2 P3 on PR #95).
 *
 * Tests that need to fetch the JSON should `require('https')` and
 * write their own fetch — the response shape (string body vs parsed
 * object) differs enough between tests that a shared fetch is more
 * hindrance than help.
 */

var BADGE_BRANCH_RAW = 'https://raw.githubusercontent.com/trafgals/ParParPar/feat/ci-benchmark-badge/';

// Match the URL-ENCODED raw.githubusercontent.com form (what shields.io
// actually fetches) so the test asserts the live URL the badge uses.
// Captures the badge id (e.g. "par3-1g-32k-zen4").
// Use the /g flag when iterating all matches; the /g-less form is fine
// for single .test() calls.
var ZEN4_BADGE_ID_RE = /https%3A%2F%2Fraw\.githubusercontent\.com%2Ftrafgals%2FParParPar%2Ffeat%2Fci-benchmark-badge%2Fbenchmarks%2Fbadges%2F([a-z0-9-]+)\.json/;

module.exports = {
  BADGE_BRANCH_RAW: BADGE_BRANCH_RAW,
  ZEN4_BADGE_ID_RE: ZEN4_BADGE_ID_RE
};
