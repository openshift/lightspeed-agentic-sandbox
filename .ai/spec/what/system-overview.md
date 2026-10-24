# System Overview

The lightspeed-agentic-sandbox is a multi-provider agent runtime that runs inside ephemeral Kubernetes pods as a **one-shot batch process**. The OpenShift Lightspeed operator mounts input files, starts the container, and reads the Result CR and exit status. Anthropic, Vertex Gemini, native/custom OpenAI, and Azure use the shared DeepAgents harness; Vertex/OpenAI is unimplemented pending OLS-4472; the legacy OpenAI adapter is removed.

## Single-Harness Migration [PARTIALLY IMPLEMENTED: OLS-4434]

Anthropic and Vertex/Google routes use the shared DeepAgents adapter. Vertex Gemini is provided through the `ChatGoogleGenerativeAI` model integration; DeepAgents owns tools/skills, admitted MCP integration, safety, and the common event path. Native OpenAI also uses DeepAgents with `ChatOpenAI` on the Responses API; custom/vLLM endpoints use `ChatOpenAI` on Chat Completions. Their endpoint boundary is defined in [configuration rule 3a](configuration.md). Azure uses the shared DeepAgents harness with native Azure clients and both API-key/Entra auth; Vertex/OpenAI is unimplemented pending OLS-4472. Azure's routing/workflows are offline-tested; live authentication acceptance remains outstanding. This split is implementation status, not a claim of completed live parity or telemetry compatibility review. Operator inputs, credential contracts, batch lifecycle, and Result publishing remain unchanged. The Google ADK provider adapter is removed; cleanup of its declared dependency is deferred. The broader target and remaining migration gates are described in [provider-contract.md](provider-contract.md#deepagents-consolidation-partially-implemented-ols-4434) and [decision 0003](../decisions/0003-single-deepagents-harness.md).

## Behavioral Rules

### System Role

1. The sandbox is a stateless, one-shot worker. Each pod runs one batch: read `/input/`, run the agent, publish a Result CR, exit. No session state persists between runs.

2. The operator is the sole orchestrator. The sandbox does not interpret workflow semantics (phases, retries, step ordering).

3. The sandbox delegates tool execution, command invocation, and skill discovery to the underlying provider SDK.

### Component Inventory

4. Major components: batch entrypoint (`batch.py`), agent execution (`run_agent.py`), provider abstraction (factory, events, options), the shared DeepAgents adapter, model routing and construction (`providers/model_router.py`, `providers/models/anthropic.py`, `providers/models/gemini.py`, `providers/models/openai.py`), configuration (`config.py`), MCP (`mcp.py`), Result CR publishing (`publish_results/`), readiness checks (`readiness.py`), and observability (`audit.py`, `metrics.py`, `tracing.py`).

5. Component behavioral rules: `run-api.md`, `provider-contract.md`, `configuration.md`, `health-probes.md`, `audit-logging.md`, `e2e-testing.md`.

### Lifecycle

6. `batch.main()` lifecycle: read `/input/` and resolve step from `result-template.kind`; resolve the model family once and pass it to `resolve_model_route(model_family=...)`; parse reasoning config, MCP servers, and execution limits; `run_readiness_checks(model_family, model_route)` (fail-fast before the LLM); `init_tracer()` when OTEL is configured; discover/admit MCP tools; construct DeepAgents with the resolved route for Anthropic, Vertex/Google, native/custom OpenAI, or Azure; Vertex/OpenAI fails route selection pending OLS-4472; `run_agent_query()`; publish the Result CR via the Kubernetes API; `shutdown_tracer()`; exit.

7. Provider is selected once per run through `resolve_model_family()` and `resolve_model_route()`. All implemented providers return typed model routes for DeepAgents; Vertex/OpenAI remains unimplemented pending OLS-4472. The same family object supplies credential readiness requirements.

8. Every provider route requires a resolved model: explicit override, canonical `LIGHTSPEED_MODEL`, or route-specific legacy model input. No route silently selects a package default when model configuration is missing. The injected route's model is authoritative in DeepAgents.

### Integration Boundaries

9. **Operator → Sandbox:** ConfigMap volume at `/input/` with query, output-schema, context, result-template, optional system-prompt. Environment variables for provider, audit, MCP, skills.

10. **Sandbox → Kubernetes API:** On agent completion (success or agent failure), create Result CR from template and replace `status` (`kubernetes` Python client). Sandbox infrastructure failures exit non-zero with `terminated.message` only — no Result CR.

11. **Sandbox → Provider SDK:** `ProviderQueryOptions` into the selected adapter; async event stream until terminal `result` event.

12. **Provider SDK → External:** Each SDK manages API auth, tools, and skills. Sandbox supplies credentials via env and file mounts.

## Configuration Surface

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `LIGHTSPEED_PROVIDER` | string | `anthropic` | Hosting backend → model family and DeepAgents route; Vertex/OpenAI unimplemented (OLS-4472) |
| `LIGHTSPEED_SKILLS_DIR` | string | `/app/skills` | Skill root and provider `cwd` |
| `LIGHTSPEED_MODEL` | string | required (or route-specific legacy model input) | Canonical model id |

See `configuration.md` for the full environment variable reference.

## Constraints

- Not a general-purpose API server. One operator, one batch run per pod.
- Provider SDK packages are optional extras; the container image ships all three.
- Hermetic Konflux builds (no network during image build).

## Planned Changes

| Ticket | Summary |
|---|---|
| OLS-4434 | Verify migrated DeepAgents routes (including Azure API key/Entra) and preserve external contracts and shared tools/safety; dependency cleanup remains deferred |
| OLS-4472 | Implement Vertex/OpenAI, currently unimplemented; support may instead be removed in a subsequent decision |
| OLS-3033 | Align operator-passed `allowedTools` and `llm` with provider options |
| OLS-3491 | System instructions via `/input/system-prompt` only |
