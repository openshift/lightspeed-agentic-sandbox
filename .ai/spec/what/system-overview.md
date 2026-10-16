# System Overview

The lightspeed-agentic-sandbox is a multi-provider agent runtime that runs inside ephemeral Kubernetes pods as a **one-shot batch process**. The OpenShift Lightspeed operator mounts input files, starts the container, and reads the Result CR and exit status. The runtime wraps DeepAgents (Anthropic/Claude), Gemini, and OpenAI SDKs behind a unified provider abstraction.

## Planned Single-Harness Architecture [PLANNED: OLS-4434]

The target runtime uses one shared DeepAgents adapter for every existing model route, with narrow LangChain model integrations for provider-specific API/authentication, reasoning, structured output, and metadata. DeepAgents owns tools/skills, admitted MCP tool integration, safety, and the common event path. Operator inputs, credential contracts, batch lifecycle, and Result publishing remain unchanged. ADK/OpenAI Agents adapters and tool implementations retire only after the route-specific verification gates pass; vendor client libraries may remain. This section describes a target, not implemented parity. Normative rules and migration gates: [provider-contract.md](provider-contract.md#deepagents-consolidation-planned-ols-4434). Decision: [0003](../decisions/0003-single-deepagents-harness.md).

## Behavioral Rules

### System Role

1. The sandbox is a stateless, one-shot worker. Each pod runs one batch: read `/input/`, run the agent, publish a Result CR, exit. No session state persists between runs.

2. The operator is the sole orchestrator. The sandbox does not interpret workflow semantics (phases, retries, step ordering).

3. The sandbox delegates tool execution, command invocation, and skill discovery to the underlying provider SDK.

### Component Inventory

4. Major components: batch entrypoint (`batch.py`), agent execution (`run_agent.py`), provider abstraction (factory, events, options), provider adapters (DeepAgents, Gemini, OpenAI), configuration (`config.py`), model routing and construction (`providers/model_router.py`, `providers/models/anthropic.py`), MCP (`mcp.py`), Result CR publishing (`publish_results/`), readiness checks (`readiness.py`), and observability (`audit.py`, `metrics.py`, `tracing.py`).

5. Component behavioral rules: `run-api.md`, `provider-contract.md`, `configuration.md`, `health-probes.md`, `audit-logging.md`, `e2e-testing.md`.

### Lifecycle

6. `batch.main()` lifecycle: read `/input/` and resolve step from `result-template.kind`; resolve the model family once and pass it to `resolve_model_route(model_family=...)`; parse reasoning config, MCP servers, and execution limits; `run_readiness_checks(model_family)` (fail-fast before the LLM); `init_tracer()` when OTEL is configured; discover/admit MCP tools; `create_provider()` with the Anthropic route or legacy SDK name; `run_agent_query()`; publish the Result CR via the Kubernetes API; `shutdown_tracer()`; exit.

7. Provider is selected once per run through `resolve_model_family()` and `resolve_model_route()`. Anthropic returns a typed model route for DeepAgents; Google/OpenAI retain legacy Gemini/OpenAI adapter configuration. The same family object supplies credential readiness requirements.

8. Anthropic model resolution uses the explicit override, canonical `LIGHTSPEED_MODEL` mapping, legacy `ANTHROPIC_MODEL`, then the Anthropic package default. The injected route's model is authoritative in DeepAgents. Google/OpenAI retain `resolve_startup_model()` / `resolve_router_model()` until migration.

### Integration Boundaries

9. **Operator → Sandbox:** ConfigMap volume at `/input/` with query, output-schema, context, result-template, optional system-prompt. Environment variables for provider, audit, MCP, skills.

10. **Sandbox → Kubernetes API:** On agent completion (success or agent failure), create Result CR from template and replace `status` (`kubernetes` Python client). Sandbox infrastructure failures exit non-zero with `terminated.message` only — no Result CR.

11. **Sandbox → Provider SDK:** `ProviderQueryOptions` into the selected adapter; async event stream until terminal `result` event.

12. **Provider SDK → External:** Each SDK manages API auth, tools, and skills. Sandbox supplies credentials via env and file mounts.

## Configuration Surface

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `LIGHTSPEED_PROVIDER` | string | `anthropic` | Hosting backend → model family and model route or legacy adapter |
| `LIGHTSPEED_SKILLS_DIR` | string | `/app/skills` | Skill root and provider `cwd` |
| `LIGHTSPEED_MODEL` | string | package default | Canonical model id |

See `configuration.md` for the full environment variable reference.

## Constraints

- Not a general-purpose API server. One operator, one batch run per pod.
- Provider SDK packages are optional extras; the container image ships all three.
- Hermetic Konflux builds (no network during image build).

## Planned Changes

| Ticket | Summary |
|---|---|
| OLS-4434 | Consolidate every existing provider route on DeepAgents; preserve external contracts and apply shared DeepAgents tools/safety |
| OLS-3033 | Align operator-passed `allowedTools` and `llm` with provider options |
| OLS-3491 | System instructions via `/input/system-prompt` only |
