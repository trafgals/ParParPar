# AGENTS.md — AI Agent Instructions

## Repository Targeting Rules

**NEVER open a pull request against the upstream `animetosho/ParPar` repository.**

**ONLY open pull requests against `trafgals/parparpar`** (case-insensitive; this repo's GitHub remote).

### Rationale

- This is a **fork** of `animetosho/ParPar` with PAR3 extensions (GF(2^64) recovery, 4 KiB–4 GiB input blocks, AVX-512/AVX-2 kernel).
- Upstream `animetosho/ParPar` does not accept PAR3 patches — only PAR2 + GF(16) work.
- All work in this fork lives on `trafgals/parparpar` and merges to `master` there.

### Enforcement

When running `gh pr create`:

```bash
# CORRECT — explicit repo, lowercase
gh pr create --repo trafgals/parparpar --base master --head <branch> ...

# CORRECT — when already in the trafgals/parparpar clone
gh pr create --base master --head <branch> ...

# WRONG — never do this
gh pr create --repo animetosho/ParPar ...
```

When pushing:

```bash
# CORRECT — push to trafgals/parparpar
git push origin <branch>

# WRONG — never push to animetosho/ParPar
git push upstream <branch>
```

### Git Remotes

- `origin` → `https://github.com/trafgals/ParParPar.git` (PR target)
- `upstream` → `https://github.com/animetosho/ParPar.git` (read-only reference; do not push, do not PR)

### Verification

Before opening any PR, confirm:

```bash
git remote -v
# origin  https://github.com/trafgals/parparpar.git (fetch)
# origin  https://github.com/trafgals/parparpar.git (push)
# upstream        https://github.com/animetosho/ParPar.git (fetch)
# upstream        https://github.com/animetosho/ParPar.git (push)
```

If the PR target looks like `animetosho/ParPar`, STOP and switch to `trafgals/parparpar`.

## CI Watching Rules

**Always watch CI with an exit-status flag so a failing run fails the command — never fire-and-forget a PR push or sleep-poll.**

```bash
# CORRECT — watch a PR's CI checks, exit watch mode (non-zero) on the first failure
gh pr checks <number> --watch --fail-fast --interval 10

# CORRECT — watch a specific workflow run; exit non-zero if it fails
gh run watch <run-id> --exit-status
```

Verified against gh 2.85.0 (2026-01-14): there is **no** `gh watch` command and **no** `--exit-code` flag. The exit-status flags are `--fail-fast` (on `gh pr checks --watch`) and `--exit-status` (on `gh run watch`) — do not write `gh watch ... --exit-code` in scripts, docs, or PR instructions.

- `gh pr checks --watch` polls until all checks finish; `--fail-fast` aborts the watch the moment a check fails (the command then exits non-zero) instead of waiting for every check to complete.
- `gh run watch <run-id>` streams a run to completion; `--exit-status` makes the exit code reflect the run's success/failure, so it can gate `&&`/`||` chains and CI steps.
- Interval flag on both commands: `-i/--interval` (`gh pr checks` default 10 s, `gh run watch` default 3 s).
- Before merging a PR: `gh pr checks <number> --watch --fail-fast` must exit 0 AND the review threads must be resolved — threads anchor to commit SHAs and can stay open after the code is fixed.

## Test Discipline for PR Comments

**Whenever a comment is raised in a PR (review bot, human reviewer, follow-up note, or self-spotted during the fix), write a test for it as part of the fix.**

### Rationale

A code change without a regression test will silently rot. A reviewer flagging "X could break here" is a contract assertion; the corresponding test pins the contract down so future refactors cannot quietly violate it. PR #53's cubic-AI review cycles (4910826158, 4910960162, 4914681432, 4915459866, 4915648282) each surfaced a fix that was completed in the same commit — but without a test, the only signal that the fix regressed would be a future cubic round-trip. Adding the test at fix time costs almost nothing and locks the contract in immediately.

### Rule (applies to every fix on every PR)

For every comment, finding, or follow-up that results in a code change:

1. **Identify the test surface.** Is it a C kernel? A JS dispatch? An engine integration? A binding contract?
2. **Add a test case (or extend an existing test) that exercises the exact contract the comment was about.** Examples:
   - "Release/acquire missing on zmm_probe cache" → multi-thread test that fans out N threads racing the probe and asserts all observe the same value.
   - "Cap query not exercised" → boundary test at the cap, one past the cap, and at the exact-cap value.
   - "Dispatch preservation not tested" → test that sets a workload-rebind method, calls the entry point, and asserts the method is unchanged.
   - "Cap-aware gate missing" → test that asserts the gate falls through to the slow path when the padded size exceeds the cap.
3. **Place the test in the right place.** C-level kernel contracts → `gf64/test/test_*.c` plus a Makefile rule. JS-level engine contracts → `test/*.js`. Binding contracts → both, depending on which side they apply to.
4. **Run it before committing.** A green test alongside the green fix is the bar.

### What this does NOT mean

- Do NOT skip the test because "the fix is obvious" or "the cubic review would catch a regression next round" — that reasoning is exactly what produced the ratchet of unfixed regressions across review cycles 4910826158 → 4915648282.
- Do NOT bundle the test into a separate "test cleanup" PR. The test belongs with the fix.
- Do NOT only add unit tests for integration contracts — the JS test layer (`test/*.js`) is the integration tier, and many cubic findings (e.g., 4910826158 P1) live there.

### Enforcement: the audit checklist before pushing a PR commit

When a PR addresses a comment, before committing, confirm:

```text
[ ] I have identified the file + line + behavior the comment targets.
[ ] I have added a test (or test case) that exercises that exact behavior.
[ ] The test runs in CI (C tests via `make` in gf64/test/, JS tests via `node test/foo.js`).
[ ] The test fails on the unfixed code and passes on the fixed code.
[ ] The test name / file references the review ID (e.g., "cubic review 4915648282 P1").
```

If any box is unchecked, the fix is not complete.