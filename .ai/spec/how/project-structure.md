# Project Structure

Package tree (authoritative for agents): see `AGENTS.md` Architecture section.
Do not maintain a duplicate path inventory here.

## Key Entry Points

| Entry point | How invoked |
|---|---|
| `lightspeed_agentic.batch` | Container CMD: `python -m lightspeed_agentic.batch` |
| `batch.main()` | Read `/input/`, run agent, publish Result CR, exit |
| `config.resolve_sdk()` | Called at start of each batch run before readiness checks and provider construction |
| `run_readiness_checks()` | R1 credential env (+ file paths) before LLM (`readiness.py`) |
| `create_provider(sdk.name)` | Factory lazy-imports the selected adapter |
| `run_agent_query()` | Shared agent execution (`run_agent.py`) |
| `publish_agent_result()` | Result CR create + status via Kubernetes API |
| `init_tracer` / `shutdown_tracer` | OTel TracerProvider setup/teardown in `batch.main()` |

## Naming Conventions

- **Package:** `lightspeed_agentic` under `src/` (hatchling src-layout).
- **Provider modules:** one file per provider in `providers/`, named after the SDK (`deepagents.py`, `gemini.py`, `openai.py`). Each exports a single `XProvider` class.
- **Batch / agent:** `batch.py` (entrypoint + input reading), `run_agent.py` (provider query loop).
- **Publish results:** `publish_results/publish.py`, `publish_results/status.py`.
- **Observability modules:** `audit.py` (span events), `metrics.py` (histograms), `tracing.py` (TracerProvider + traceparent parsing).
- **Config / MCP / readiness:** `config.py` maps `LIGHTSPEED_*` → SDK env; `mcp.py` parses `LIGHTSPEED_MCP_SERVERS`; `readiness.py` runs `run_readiness_checks()` at batch startup (see `health-probes.md`).
- **Test layout:** `tests/` mirrors source. `tests/e2e/` holds batch cluster BDD (feature files, `batch_runner.py`, `suite_setup.py`). Live provider coverage runs via `make e2e` (see [e2e-testing.md](../what/e2e-testing.md)).

## Proposed runtime structure [PLANNED: OLS-4434]

[DeepAgents migration](../what/deepagents-migration.md) replaces the three agent
adapters with one execution path plus provider-specific model construction. The
current entry points/extras below are not the target dependency inventory. The
migration removes `google-adk` and `openai-agents`, declares needed LangChain model
integrations directly, and retains required vendor/auth libraries. Lazy optional
imports, image completeness, and local/hermetic graph consistency remain required.
No dependency or source-file layout is changed by this spec proposal.

## Dependency Organization

The project uses optional extras to gate provider SDKs:

| Extra | Packages |
|---|---|
| `deepagents` | `deepagents`, `langchain-anthropic`, `langchain-google-vertexai`, `langchain-aws`, `langchain-mcp-adapters` |
| `gemini` | `google-adk` |
| `openai` | `openai-agents` |
| `all` | All three provider extras |
| `dev` | All providers + test/lint tools |
| `eval` | Eval-specific test dependencies |
| `e2e` | BDD test dependencies |

Runtime dependencies include `kubernetes` (Result CR publishing) and `prometheus-client` (metrics histograms). FastAPI and Uvicorn are **not** runtime dependencies.

Provider SDK imports are always lazy (inside methods or guarded by the factory) so the base package imports cleanly without any extras installed.
