# Project Structure

Package tree (authoritative for agents): see `AGENTS.md` Architecture section.
Do not maintain a duplicate path inventory here.

## Key Entry Points

| Entry point | How invoked |
|---|---|
| `lightspeed_agentic.batch` | Container CMD: `python -m lightspeed_agentic.batch` |
| `batch.main()` | Read `/input/`, run agent, publish Result CR, exit |
| `config.resolve_model_family()` | Resolve family, SDK env mapping, and credentials once per batch run |
| `providers.model_router.resolve_model_route(model_family=...)` | Reuse that family; return an Anthropic, Vertex/Google, native/custom OpenAI, or Azure model route without constructing clients; unsupported Vertex families are rejected earlier by `resolve_model_family()` |
| `run_readiness_checks(model_family, model_route)` | R1 credential env (+ file paths) and route parameters before LLM (`readiness.py`) |
| `providers.deepagents.DeepAgentsProvider(model_route)` | Construct the shared adapter directly with the resolved model route |
| `run_agent_query()` | Shared agent execution (`run_agent.py`) |
| `publish_agent_result()` | Result CR create + status via Kubernetes API |
| `init_tracer` / `shutdown_tracer` | OTel TracerProvider setup/teardown in `batch.main()` |

## Naming Conventions

- **Package:** `lightspeed_agentic` under `src/` (hatchling src-layout).
- **Provider adapter:** `providers/deepagents.py` exports the single active provider class, receiving a resolved model route for Anthropic, Vertex/Google, native/custom OpenAI, and Azure. The legacy OpenAI adapter is removed; Vertex supports only Anthropic and Google models.
- **Model routing/construction:** `config.py` defines provider settings, resolved Anthropic credential values, cloud credential references, and `ModelParameters`; `providers/model_router.py` defines `ResolvedModelRoute` and the `ModelCreator` callable alias. `providers/models/anthropic.py` provides direct, Bedrock, and Vertex Anthropic constructors; `providers/models/gemini.py` constructs the Vertex Gemini model; `providers/models/openai.py` constructs native Responses and compatible Chat Completions models with their respective schema translators. Optional SDK imports remain lazy.
- **Batch / agent:** `batch.py` (entrypoint + input reading), `run_agent.py` (provider query loop).
- **Publish results:** `publish_results/publish.py`, `publish_results/status.py`.
- **Observability modules:** `audit.py` (span events), `metrics.py` (histograms), `tracing.py` (TracerProvider + traceparent parsing).
- **Config / MCP / readiness:** `config.py` maps `LIGHTSPEED_*` → SDK env; `mcp.py` parses `LIGHTSPEED_MCP_SERVERS`; `readiness.py` runs `run_readiness_checks()` at batch startup (see `health-probes.md`).
- **Test layout:** `tests/` mirrors source. `tests/e2e/` holds batch cluster BDD (feature files, `batch_runner.py`, `suite_setup.py`). Live provider coverage runs via `make e2e` (see [e2e-testing.md](../what/e2e-testing.md)).

## Dependency Organization

> [PLANNED: OLS-4671] Provider extras move to the shared core library; this project depends on the core (hash-pinned sdist direct URL) with the batch routes' extras. See [provider-contract.md rule 73](../what/provider-contract.md#shared-execution-core-planned-ols-4671).

The project uses optional extras to gate provider SDKs:

| Extra | Packages |
|---|---|
| `deepagents` | Shared harness, directly imported LangChain/framework and cloud-credential libraries, JSON Schema validation, and Anthropic/Bedrock/Vertex Gemini model integrations |
| `openai` | Includes `deepagents`, plus `langchain-openai`, the native `openai` SDK, `azure-core`, `azure-identity`, and `aiohttp`; supports native/custom OpenAI and Azure |
| `all` | Both active provider extras |
| `dev` | `all` providers + test/lint tools |
| `e2e` | BDD test dependencies and the existing mock MCP server's MCP/Starlette/Uvicorn libraries |

Base dependencies include `kubernetes` (Result CR publishing), `prometheus-client` (metrics), HTTPX (shared TLS/transport), and directly imported OTLP encoding/Protobuf and urllib3 interfaces. The obsolete `gemini` extra and the `google-adk`/`openai-agents` harness dependencies are removed. Vertex Gemini remains in `deepagents`. FastAPI is no longer in the project dependency graph; Starlette and Uvicorn remain transitive MCP dependencies, not a sandbox HTTP server.

Provider SDK imports remain lazy (inside methods or narrow model-construction helpers), so the base package imports without provider extras. Local and Konflux dependency inputs exclude the retired harnesses; generated requirements and Tekton package lists are updated together.
