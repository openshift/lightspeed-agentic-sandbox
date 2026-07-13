# Project Structure

## Module Map

| File/Directory | Key Symbols | Responsibility |
|---|---|---|
| `src/lightspeed_agentic/__init__.py` | re-exports | Public API surface: `create_provider`, `AgentProvider`, event types, `ProviderQueryOptions`, `EventLogger` |
| `src/lightspeed_agentic/app.py` | `app` | FastAPI entry point, wires provider + router + health routes |
| `src/lightspeed_agentic/factory.py` | `ProviderName`, `create_provider()` | Provider selection via env var, lazy SDK imports |
| `src/lightspeed_agentic/types.py` | `AgentProvider`, `ProviderQueryOptions`, event dataclasses, `DEFAULT_MODEL` | Core abstractions: provider ABC, query options bundle, normalized event union |
| `src/lightspeed_agentic/tools.py` | `DEFAULT_ALLOWED_TOOLS` | Shared tool allowlist constant |
| `src/lightspeed_agentic/logging.py` | `EventLogger` | Thinking-buffer and structured event logging |
| `src/lightspeed_agentic/health.py` | `register_health_routes()`, `health_payload()`, `register_ready_route()`, `ready_response()` | `GET /health` liveness probe, `GET /ready` readiness probe |
| `src/lightspeed_agentic/routes/__init__.py` | `build_router()` | Router factory: model resolution, default params, mounts query routes |
| `src/lightspeed_agentic/routes/models.py` | `RunRequest`, `RunResponse` | Pydantic request/response models |
| `src/lightspeed_agentic/routes/query.py` | `run_endpoint`, `_format_context_prefix()` | `POST /run` handler: context prefixing, timeout, JSON parse/fallback |
| `src/lightspeed_agentic/providers/deepagents.py` | `DeepAgentsProvider` | DeepAgents (LangChain) adapter for Anthropic models |
| `src/lightspeed_agentic/providers/gemini.py` | `GeminiProvider` | Google ADK adapter |
| `src/lightspeed_agentic/providers/openai.py` | `OpenAIProvider` | OpenAI Agents SDK adapter |

## Key Entry Points

| Entry point | How invoked |
|---|---|
| `lightspeed_agentic.app:app` | Uvicorn ASGI target (`uvicorn lightspeed_agentic.app:app --host 0.0.0.0 --port 8080`) |
| `create_provider()` | Called once at module load in `app.py` |
| `build_router(provider, ...)` | Called once at module load in `app.py`, mounted at `/v1/agent` |

## Naming Conventions

- **Package:** `lightspeed_agentic` under `src/` (hatchling src-layout).
- **Provider modules:** one file per provider in `providers/`, named after the SDK (`deepagents.py`, `gemini.py`, `openai.py`). Each exports a single `XProvider` class.
- **Route modules:** `routes/` contains `models.py` (Pydantic shapes), `query.py` (endpoint registration), `__init__.py` (router builder).
- **Test layout:** `tests/` mirrors source structure. `tests/e2e/` holds BDD feature files and step definitions. `evals/` is a separate integration test suite run in containers.

## Dependency Organization

The project uses optional extras to gate provider SDKs:

| Extra | Packages |
|---|---|
| `deepagents` | `deepagents`, `langchain-google-vertexai`, `langchain-aws` |
| `gemini` | `google-adk` |
| `openai` | `openai-agents` |
| `all` | All three provider extras |
| `dev` | All providers + test/lint tools |
| `eval` | Eval-specific test dependencies |
| `e2e` | BDD test dependencies |

Provider SDK imports are always lazy (inside methods or guarded by the factory match) so the base package imports cleanly without any extras installed.
