# Behavioral spec: readiness checks

Audience: AI agents (Claude). Precision over narrative.

Cross-references: batch lifecycle → `run-api.md`. Credential env mapping → `configuration.md`.

> **HTTP probes superseded (OLS-3066).** There is no `GET /health` or `GET /ready`.
> Readiness runs in-process at batch startup before the LLM is invoked.

## Behavioral Rules

1. **Fail-fast before LLM.** `batch.main()` MUST resolve the model family once, reuse it for model routing, and call `run_readiness_checks(model_family, model_route)` before MCP discovery or provider creation. Routing MUST NOT construct model clients or perform readiness network I/O. When any check fails, the sandbox MUST use the sandbox failure path (`run-api.md` rule 23): write a termination log with per-check status strings and exit non-zero.

2. **R1 — Credential sources.** `check_provider_env(expected_envs, credential_file_envs)` validates required environment variables for legacy `ResolvedSDK` configurations. Batch passes the resolved family and model route; readiness validates the migrated routes' typed credential inputs: direct Anthropic and native/custom OpenAI require non-empty resolved API keys (including rejection of whitespace-only OpenAI keys), Vertex requires a readable non-empty mounted credential file, and Bedrock requires a complete non-empty access/secret key pair. `run_readiness_checks()` performs presence checks only and makes no provider network calls.

2a. **R1 — Azure credential alternatives** [OLS-3050]. For `azure`, readiness MUST pass when **either** an API key is present (`AZURE_OPENAI_API_KEY` env or `apitoken` file) **or** all three Entra ID service-principal files (`client_id`, `tenant_id`, `client_secret`) exist under `/var/run/secrets/llm-credentials/` and are non-empty — the two modes defined in `configuration.md` rule 9a. When neither complete set is present, readiness MUST fail with a descriptive error naming the missing credential set. For a resolved Azure family, readiness also validates the typed credential source, nonempty API key, HTTPS endpoint, and nonempty resolved API version, reporting `provider_configuration` before MCP discovery or client construction. These checks belong in readiness, not the prepared Azure model constructor, and validate the resolved family used by the shared Azure DeepAgents route. Legacy `ResolvedSDK` callers retain their environment-presence check. Readiness makes no authentication request; token acquisition happens at adapter init / first request (see `provider-contract.md` rules 29, 38), and a definitive token-acquisition failure terminates the run there rather than at readiness.

2b. **R1 — Bedrock credential alternatives** [OLS-4092]. For `bedrock`, readiness MUST pass when `aws_access_key_id` and `aws_secret_access_key` are present and non-empty (as environment values or files under `/var/run/secrets/llm-credentials/`); optional `aws_session_token` supplies temporary source credentials, and optional `role_arn` selects STS assume-role. When the required key pair is absent, readiness MUST fail with a descriptive error. Readiness validates **presence** only; STS assume-role and credential refresh are performed by `botocore` at first request (see `provider-contract.md` rule 38), not by a readiness network call.

2c. **R2 — Model route parameters.** When a model route is supplied, readiness MUST validate its required construction parameters before MCP discovery: `vertex/anthropic` and `vertex/google` require a non-empty project and region, `bedrock/anthropic` requires a non-empty region, and `openai-compatible` requires a non-empty resolved base URL. Vertex defaults to `global`; Bedrock resolves region from `LIGHTSPEED_PROVIDER_REGION`, then `AWS_REGION`, then `us-east-1`. This check MUST NOT construct model clients or make network calls.

3. **No endpoint network probe.** The sandbox MUST NOT HTTP-probe provider base URLs before the agent run. Endpoint reachability is established when the provider SDK invokes the LLM API. Readiness makes no network call for Azure token acquisition either.

| Backend | Required credential(s) |
|---------|-------------------|
| `anthropic` (direct) | Non-empty resolved `AnthropicCredentialSource.api_key` (from `ANTHROPIC_API_KEY`) |
| `vertex/*` | Non-empty mounted Vertex credential file referenced in typed parameters |
| `openai` / `openai-compatible` | Non-empty resolved `ModelParameters.api_key` (from `OPENAI_API_KEY`) |
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
