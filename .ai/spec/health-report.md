# Spec health report

Last evaluated: 2026-08-31
Trigger: post-milestone (drift audit — staleness + accuracy vs current code)
Layout: software (.ai/spec/)

## Stale

**e2e-testing.md — OLS-3739 troubleshooting section overstates implementation.**
The spec describes the cluster troubleshooting BDD suite in present tense as if
fully landed. Verified against code (git: `ebec6ee` added the spec, `dfade58`
implemented only the scenario scripts):

- **Implemented:** `scenarios/troubleshooting/` with all 11 scenario dirs
  (`setup.sh`/`cleanup.sh`) + `scenario_metadata.yaml`; conftest fixtures
  `k8s_client`, `k8s_core_client`, `scenario_cleanup` (`tests/e2e/conftest.py`).
- **NOT implemented (referenced as if present):**
  - `tests/e2e/features/troubleshooting.feature` — does not exist.
  - Troubleshooting BDD test module (no `test_troubleshooting_bdd.py`).
  - LLM judge utility module — does not exist.
  - `scripts/e2e-cluster.sh` and `make e2e-cluster` target — do not exist
    (Makefile has only the `e2e` target → `scripts/e2e-containers.sh`).
- The Verification map (line ~181) lists `troubleshooting.feature` as a current
  feature file, which is misleading — the file does not exist.

## Missing

None. All `what/` and `how/` spec files referenced in `README.md` exist, and all
code symbols named in `how/` specs are present and correctly named
(`resolve_sdk`, `parse_reasoning_config`, `parse_mcp_servers`,
`resolve_router_model`, `resolve_startup_model`, `create_provider`,
`run_readiness_checks`, `check_provider_env`, `DEFAULT_ALLOWED_TOOLS`,
`has_skills`, `DEFAULT_MAX_TURNS`).

## Structural concerns

None. `what/` and `how/` separation is clean; behavioral rules stay in `what/`,
code navigation in `how/`.

## Findability issues

None significant. The README quick-start and cross-reference tables cover all
spec files. The `decisions/` directory holds only a README template (no ADRs
yet); not a defect.

## Verified current (no drift — [PLANNED] markers accurate)

Checked against code and confirmed these still-planned items are genuinely
unimplemented, so their `[PLANNED]` markers are correct and must stay:

- **OLS-3743** (agent timeout / max-turns): code still uses `LIGHTSPEED_TIMEOUT_MS`
  via `batch._resolve_timeout_ms()` and `DEFAULT_MAX_TURNS = 200`
  (`batch.py:43,202,274`). No `LIGHTSPEED_AGENT_TIMEOUT_SECONDS` /
  `LIGHTSPEED_AGENT_MAX_TURNS` handling, no `AgentTimeout` reason in `status.py`.
  Recent commits `f72f325` / `5ee7656` were spec-only. Markers accurate.
- **OLS-3661** (Result CR `status.tokenUsage`): token counts flow through
  `run_agent`/`types`/`audit`/`metrics`, but `publish_results/status.py` does not
  set `status.tokenUsage`. Commit `f3efc07` was spec-only. Marker accurate.
- **OLS-3033** (`allowedTools` override): `run_agent_query()` always passes
  `DEFAULT_ALLOWED_TOOLS`; no input/env override path. Marker accurate.
- **OLS-3500** (Claude adapter removed): no `providers/claude.py`; only
  `deepagents.py`, `gemini.py`, `openai.py`. `[Removed]` rules accurate.
- **OLS-3491** (system-prompt file): `/input/system-prompt` read with
  `DEFAULT_SYSTEM_PROMPT = "You are an AI agent."` default (`batch.py:41,79`) —
  matches run-api rule 5; forward-looking marker for the full contract retained.
</content>
</invoke>
