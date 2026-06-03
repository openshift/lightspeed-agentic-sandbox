# Architecture: modules, data flow, SDK integration

Audience: AI agents (Claude). File paths and symbols allowed here.

Cross-references: behavioral rules → `what/run-api.md`, `what/provider-contract.md`, `what/configuration.md`.

## Module Map

| Path | Contents |
|------|----------|
| `src/lightspeed_agentic/__init__.py` | Re-exports `create_provider`, `EventLogger`, `AgentProvider`, event dataclasses, `ProviderQueryOptions`. |
| `src/lightspeed_agentic/app.py` | `app: FastAPI` — constructs provider via `create_provider()`, builds router with `build_router(...)`, `include_router(..., prefix="/v1/agent")`, registers `GET /health` and `GET /ready` via health module. |
| `src/lightspeed_agentic/health.py` | `GET /health` liveness (no external calls) + `GET /ready` readiness (R1 credential env + R2 provider endpoint reachability per `health-probes.md`). |
| `src/lightspeed_agentic/factory.py` | `ProviderName`, `create_provider()` — `match` on name/env, lazy-imports `ClaudeProvider`, `GeminiProvider`, `OpenAIProvider`. |
| `src/lightspeed_agentic/types.py` | `stringify()`, truncation-related constants, event datclasses (`TextDeltaEvent`, `ThinkingDeltaEvent`, `ContentBlockStopEvent`, `ToolCallEvent`, `ToolResultEvent`, `ResultEvent`), `ProviderQueryOptions`, abstract `AgentProvider`. |
| `src/lightspeed_agentic/tools.py` | `DEFAULT_ALLOWED_TOOLS`. |
| `src/lightspeed_agentic/logging.py` | `EventLogger` — thinking buffer, flush thresholds, truncation caps, structured log lines per event type. |
| `src/lightspeed_agentic/routes/__init__.py` | `build_router()` — resolves model from env via provider name map and `DEFAULT_MODEL`, calls `register_query_routes`. |
| `src/lightspeed_agentic/routes/models.py` | Pydantic `RunRequest`, `RunResponse` (`extra="allow"`). |
| `src/lightspeed_agentic/routes/query.py` | `_format_context_prefix()`, `register_query_routes()` — async `run_endpoint`, `asyncio.wait_for`, JSON parse / fallback, registers `POST /run`. |
| `src/lightspeed_agentic/providers/claude.py` | `ClaudeProvider` — `_ensure_skills_link()`, `query()` using `ClaudeAgentOptions`, `query` stream, maps `StreamEvent` / `AssistantMessage` / `ResultMessage` to `ProviderEvent`, `output_format` from schema. |
| `src/lightspeed_agentic/providers/gemini.py` | `_load_skills_toolset()`, `GeminiProvider` — `ExecuteBashTool` with auto-confirm wrapper and `bash -c` wrapping, optional search tools, `SkillToolset`, `Agent` + `Runner` + `InMemorySessionService`, `response_schema` when output schema set, usage aggregation. |
| `src/lightspeed_agentic/providers/openai.py` | `_RawJsonSchema`, `_ensure_openai_init()`, `OpenAIProvider` — `AsyncOpenAI`, `OpenAIResponsesModel`, `SandboxAgent`, `UnixLocalSandboxClient`, capabilities `Shell`/`Filesystem`/`Skills`, `Runner.run_streamed`, stream event mapping. |

## Data Flow

1. Client (operator) `POST /v1/agent/run` with JSON body.
2. FastAPI validates `RunRequest`; `run_endpoint` computes timeout, system prompt, optional context prefix + query.
3. Handler calls `provider.query(ProviderQueryOptions(...))` with model, turns, budget, tools, cwd, schema.
4. Handler async-iterates events, `EventLogger.log` side effects, stops at first `result` event.
5. Handler parses `result.text` as JSON object or falls back to plain summary; returns `RunResponse`.

## Key Abstractions

- **Factory:** `create_provider` centralizes backend choice and lazy imports so optional SDKs are not loaded unless selected.
- **Events:** Normalized `ProviderEvent` union decouples route layer from vendor streaming models.
- **Options:** `ProviderQueryOptions` is the single bundle passed into every adapter.
- **Router builder:** Encapsulates env-based model resolution and default router parameters.

## Integration Points

- **FastAPI / Uvicorn:** ASGI entry `lightspeed_agentic.app:app`.
- **claude-agent-sdk:** `query`, `ClaudeAgentOptions`, message types, `structured_output` on result.
- **google-adk / google.genai:** `Agent`, `Runner`, `InMemorySessionService`, `ExecuteBashTool`, `SkillToolset`, `GenerateContentConfig`.
- **openai-agents (+ openai):** `SandboxAgent`, `Runner`, `UnixLocalSandboxClient`, stream item types, `AgentOutputSchemaBase` subclass for raw schema.

## Implementation Notes

- **Claude skills layout:** `_ensure_skills_link` creates `.claude/skills` with symlinks to child dirs of the skill root, preferring writable cwd else temp workspace; module-level cache keyed by cwd string.
- **Gemini bash:** Monkey-patches `run_async` to force confirmation and shell wrapping via `shlex.quote`.
- **OpenAI init:** One-time verbose logging and tracing disable.
- **Containerfile:** Multi-stage build — builder installs hashed requirements into target site-packages and installs Claude Code via npm (`npm ci` under cachi2 or global install fallback); runtime copies site-packages, `node_modules`, app `src`, sets user `agent`, `tini` entrypoint, Uvicorn CMD on port 8080.
- **Makefile:** `uv sync` targets, `requirements` generates hashed per-arch requirements; `rpm-lockfile` regenerates RPM lock via fedora tool image; `image` builds local container.
- **Tests / evals:** HTTP clients target `POST /v1/agent/run` (see `tests/` and `evals/`).
