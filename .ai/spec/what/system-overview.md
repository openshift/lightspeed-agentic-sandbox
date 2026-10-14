# System Overview

The lightspeed-agentic-sandbox is a multi-provider agent runtime that runs inside ephemeral Kubernetes pods as a **one-shot batch process**. The OpenShift Lightspeed operator mounts input files, starts the container, and reads the Result CR and exit status. The runtime wraps DeepAgents (Anthropic/Claude), Gemini, and OpenAI SDKs behind a unified provider abstraction.

## Proposed runtime consolidation [PLANNED: OLS-4434]

[DeepAgents migration](deepagents-migration.md) proposes one DeepAgents agent
runtime with provider-specific LangChain model clients. It removes Google ADK and
OpenAI Agents agent SDKs, not the needed vendor model/authentication libraries.
The current component inventory and lifecycle below remain in effect until the
migration is implemented. The operator/batch/Result integration boundary is not
changed; route qualification and the unresolved Vertex/OpenAI case are explicit
in the target design.

## Behavioral Rules

### System Role

1. The sandbox is a stateless, one-shot worker. Each pod runs one batch: read `/input/`, run the agent, publish a Result CR, exit. No session state persists between runs.

2. The operator is the sole orchestrator. The sandbox does not interpret workflow semantics (phases, retries, step ordering).

3. The sandbox delegates tool execution, command invocation, and skill discovery to the underlying provider SDK.

### Component Inventory

4. Major components: batch entrypoint (`batch.py`), agent execution (`run_agent.py`), provider abstraction (factory, events, options), provider adapters (DeepAgents, Gemini, OpenAI), configuration (`config.py`), MCP (`mcp.py`), Result CR publishing (`publish_results/`), readiness checks (`readiness.py`), and observability (`audit.py`, `metrics.py`, `tracing.py`).

5. Component behavioral rules: `run-api.md`, `provider-contract.md`, `configuration.md`, `health-probes.md`, `audit-logging.md`, `e2e-testing.md`.

### Lifecycle

6. `batch.main()` lifecycle: read `/input/` and resolve step from `result-template.kind`; `resolve_sdk()`; parse reasoning config and MCP servers; `run_readiness_checks()` (fail-fast before the LLM); `init_tracer()` when OTEL is configured; `create_provider()` and `resolve_router_model()`; `run_agent_query()`; publish the Result CR via the Kubernetes API; `shutdown_tracer()`; exit.

7. Provider is selected once per run via `LIGHTSPEED_PROVIDER` (mapped by `resolve_sdk()`).

8. Model resolution uses `LIGHTSPEED_MODEL` (mapped to SDK env vars), with package default fallback via `resolve_router_model()`.

### Integration Boundaries

9. **Operator → Sandbox:** ConfigMap volume at `/input/` with query, output-schema, context, result-template, optional system-prompt. Environment variables for provider, audit, MCP, skills.

10. **Sandbox → Kubernetes API:** On agent completion (success or agent failure), create Result CR from template and replace `status` (`kubernetes` Python client). Sandbox infrastructure failures exit non-zero with `terminated.message` only — no Result CR.

11. **Sandbox → Provider SDK:** `ProviderQueryOptions` into the selected adapter; async event stream until terminal `result` event.

12. **Provider SDK → External:** Each SDK manages API auth, tools, and skills. Sandbox supplies credentials via env and file mounts.

## Configuration Surface

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `LIGHTSPEED_PROVIDER` | string | `anthropic` | Hosting backend → SDK name |
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
| OLS-4434 | Proposed [single DeepAgents runtime](deepagents-migration.md) with explicit model-client/API routing and qualification gates |
| OLS-3033 | Align operator-passed `allowedTools` and `llm` with provider options |
| OLS-3491 | System instructions via `/input/system-prompt` only |
