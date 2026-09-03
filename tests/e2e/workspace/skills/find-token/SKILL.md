---
name: find-token
description: Find the hidden verification token. Run the find-token script to retrieve unique DIAG and VERIFY tokens.
allowed-tools: Bash(bash:*)
---

# Find Token Skill

Retrieve hidden verification tokens by running the find-token script.

## Usage

From the skill directory:

```bash
bash scripts/find-token.sh
```

Or use `run_skill_script` with `file_path=scripts/find-token.sh`.

## Output

The script prints a full structured analysis JSON object (actionRequired, options with
remediationPlan.actions, components with DIAG_/VERIFY_ tokens). Use that JSON as
the basis for your structured response.
