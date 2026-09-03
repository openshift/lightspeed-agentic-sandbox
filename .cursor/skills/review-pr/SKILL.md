---
name: review-pr
description: Review PR with structured approach covering architecture, naming, patterns, and critical questions
disable-model-invocation: true
---

# Review PR

When asked to review a PR, follow this structured approach.

## 1. Fetch Latest Changes

**Always** fetch the latest PR state before reviewing. Cached data may be stale.

Default GitHub location: `openshift/lightspeed-agentic-sandbox`. Adjust org/repo
if the user specifies another fork.

```bash
git fetch upstream pull/<PR_NUMBER>/head:pr-<PR_NUMBER>   # or: origin pull/...
git log pr-<PR_NUMBER> --oneline -10
git diff upstream/main...pr-<PR_NUMBER> --stat
```

For follow-up reviews, re-fetch:

```bash
git fetch upstream pull/<PR_NUMBER>/head:pr-<PR_NUMBER> --force
```

Read diffs per area (`src/lightspeed_agentic/`, `tests/`, routes, providers)
rather than one massive diff.

## 2. Understand What It Implements

- Summarize the feature/fix in 2-3 sentences
- Identify the flow: entry point → processing → output
- Map which files serve which role (route, model, provider adapter, test)

## 3. Evaluate How It Was Implemented

Only raise issues if you have a concrete concern:

- **Architecture**: Thin provider adapters vs duplicated SDK logic (`CLAUDE.md`)
- **Error handling**: FastAPI responses, streaming edge cases
- **Tests**: Offline unit tests; live behavior belongs in `tests/e2e/` (`make e2e`)
- **Dependencies**: Optional extras, no top-level provider SDK imports in
  `providers/*.py`

## 4. Tests and Risk

- Are route/provider paths covered without hitting real APIs?
- Do changes need Konflux lockfile or `make requirements` updates?

## 5. Output Format

- Short summary (what + why)
- **Must-fix** vs **nit** vs **questions**
- Avoid generic praise; be specific
