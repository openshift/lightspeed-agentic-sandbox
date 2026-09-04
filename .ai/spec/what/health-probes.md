# Behavioral spec: readiness checks

Audience: AI agents (Claude). Precision over narrative.

Cross-references: batch lifecycle → `run-api.md`. Credential env mapping → `configuration.md`.

> **HTTP probes superseded (OLS-3066).** There is no `GET /health` or `GET /ready`.
> Readiness runs in-process at batch startup before the LLM is invoked.

## Behavioral Rules

1. **Fail-fast before LLM.** After `resolve_sdk()`, `batch.main()` MUST call `run_readiness_checks(sdk)` before `create_provider()` or `run_agent_query()`. When any check fails, the sandbox MUST use the sandbox failure path (`run-api.md` rule 23): write a termination log with per-check status strings and exit non-zero.

2. **R1 — Credential env.** `check_provider_env(expected_envs, credential_file_envs)` — required env vars from `ResolvedSDK` MUST be set and non-empty. For env vars listed in `credential_file_envs` (Vertex: `GOOGLE_APPLICATION_CREDENTIALS`), the path MUST exist, be readable, and non-empty.

2a. **R1 — Azure credential alternatives** [OLS-3050]. For `azure`, readiness MUST pass when **either** an API key is present (`AZURE_OPENAI_API_KEY` env or `apitoken` file) **or** all three Entra ID service-principal files (`client_id`, `tenant_id`, `client_secret`) exist under `/var/run/secrets/llm-credentials/` and are non-empty — the two modes defined in `configuration.md` rule 9a. When neither complete set is present, readiness MUST fail with a descriptive error naming the missing credential set. Readiness validates **presence** only; token acquisition happens at adapter init / first request (see `provider-contract.md` rules 29, 38), and a definitive token-acquisition failure terminates the run there rather than at readiness.

2b. **R1 — Bedrock credential alternatives** [OLS-4092]. For `bedrock`, readiness MUST pass when `aws_access_key_id` and `aws_secret_access_key` are present and non-empty (as env vars or files under `/var/run/secrets/llm-credentials/`); an optional `role_arn` file additionally selects STS assume-role — the two modes defined in `configuration.md` rule 9b. When the required key pair is absent, readiness MUST fail with a descriptive error. Readiness validates **presence** only; STS assume-role and credential refresh are performed by `botocore` at first request (see `provider-contract.md` rule 38), not by a readiness network call.

3. **No endpoint network probe.** The sandbox MUST NOT HTTP-probe provider base URLs before the agent run. Endpoint reachability is established when the provider SDK invokes the LLM API. Readiness makes no network call for Azure token acquisition either.

| Backend | Required credential(s) |
|---------|-------------------|
| `anthropic` (direct) | `ANTHROPIC_API_KEY` |
| `vertex/*` | `GOOGLE_APPLICATION_CREDENTIALS` (file path) |
| `openai` (direct) | `OPENAI_API_KEY` |
| `azure` | `AZURE_OPENAI_API_KEY`/`apitoken` **or** `client_id`+`tenant_id`+`client_secret` files (rule 2a) |
| `bedrock` | `aws_access_key_id`+`aws_secret_access_key` (env or files), optional `role_arn` for STS (rule 2b) |

4. **MCP reachability.** Not implemented; no Jira story.

## Verification

| Artifact | Rules exercised |
|----------|-----------------|
| [test_ready.py](../../../tests/test_ready.py) | R1, `run_readiness_checks()` |
| [test_batch.py](../../../tests/test_batch.py) | Rule 1 (fail-fast path) |

Live batch Jobs run in-process readiness (rule 1) before the LLM is invoked; see
[e2e-testing.md](e2e-testing.md). HTTP probe BDD scenarios were removed.
