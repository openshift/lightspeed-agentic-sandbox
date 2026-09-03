---
name: raise-pr
description: Step-by-step workflow for committing staged changes and opening a pull request for lightspeed-agentic-sandbox.
disable-model-invocation: true
---

# Create a PR

## Step 1: Branch

```bash
git fetch upstream   # or origin, depending on fork setup
git branch --show-current
git log --oneline upstream/main..HEAD   # adjust main ref if needed
git diff upstream/main --stat
gh pr list --head "$(git branch --show-current)" --state open
```

Present a short summary and ask before proceeding when ambiguous:

> You are on branch `<branch>`, `<N>` commit(s) ahead of main touching: …
> [If an open PR exists]: Pushing would update `<url>`.

**Do not continue until the user answers** when creating a new branch vs using
the current one is unclear.

If creating a new branch from main:

```bash
git fetch upstream
git checkout -b <type>/<short-description> upstream/main
# cherry-pick from old branch if needed
```

Prefer rebasing onto the canonical main (`upstream/main` for forks of
`openshift/lightspeed-agentic-sandbox`) so the PR does not include stale fork
commits.

## Step 2: Pre-commit checks

Run in order and fix failures:

1. `make test`
2. `make verify`

Optional if the change touches live cluster behavior: `make e2e <provider>` (slow,
needs credentials — only when requested).

## Step 3: Commit and push

Commit message: short imperative description. Use conventional prefixes if the
team prefers (`fix:`, `chore:`, `feat:`).

```bash
git add -A   # or only relevant files
git commit -m "<message>"
git push -u origin HEAD
```

## Step 4: Open the PR

If `.github/PULL_REQUEST_TEMPLATE.md` exists, follow it. Otherwise use a clear
title and body:

- What changed and why
- How you tested (`make test`, `make verify`, `make e2e` if run)
- Links to related issues or specs (e.g. `.ai/spec/...`)

```bash
gh pr create --title "<title>" --body "$(cat <<'EOF'
## Summary
...

## Testing
- [ ] make test
- [ ] make verify
EOF
)"
```

Return the PR URL when done.
