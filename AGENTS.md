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