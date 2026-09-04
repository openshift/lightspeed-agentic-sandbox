# Audit Logging

Implementation spec for compliance audit logging in the agentic sandbox. Parent spec: `ols/.ai/spec/what/audit-logging.md` (authoritative for cross-repo requirements, event semantics, correlation contract, and OTel GenAI attribute reference).

Telemetry aligns with [OTel GenAI Semantic Conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/README.md) (v1.41).

## Behavioral Rules

### Span Naming and Kinds

1. The sandbox MUST create a `chat {gen_ai.request.model}` span (e.g., `chat claude-sonnet-4-20250514`) as a child of the operator's span (using the received trace context). This is the **inference span** covering the full SDK inference call. Span kind MUST be `CLIENT`.

2. The sandbox MUST create an `execute_tool {gen_ai.tool.name}` span (e.g., `execute_tool Bash`) for each tool call/result pair. These are children of the inference span. Span kind MUST be `INTERNAL`.

3. The sandbox does not run its own agent loop — it consumes events from the provider SDK's internal agentic loop. Spans are created alongside the existing event normalization in each provider adapter.

### GenAI Attributes — Inference Span

4. The inference span (`chat {gen_ai.request.model}`) MUST carry the following attributes:

| Attribute | Requirement | Description |
|---|---|---|
| `gen_ai.operation.name` | Required | `"chat"` |
| `gen_ai.request.model` | Required | Model name requested (e.g., `claude-sonnet-4-20250514`) |
| `gen_ai.response.model` | Recommended | Actual model from SDK response |
| `gen_ai.provider.name` | Required | Provider name (e.g., `anthropic`, `openai`, `google`) |
| `gen_ai.usage.input_tokens` | Recommended | Input token count for this operation |
| `gen_ai.usage.output_tokens` | Recommended | Output token count for this operation |
| `agenticrun.uid` | Recommended (custom) | AgenticRun CR metadata.uid as received via `x-agenticrun-uid` (hyphens preserved) when the operator sends it — cross-trace correlation key |
| `server.address` | Recommended | LLM API endpoint hostname |

5. When `x-agenticrun-uid` (or equivalent request context) is present, the sandbox MUST propagate that value as the `agenticrun.uid` span attribute on spans it creates. When absent, the sandbox MUST NOT invent a uid.

### GenAI Attributes — Tool Span

6. Each tool execution span (`execute_tool {gen_ai.tool.name}`) MUST carry the following attributes:

| Attribute | Requirement | Description |
|---|---|---|
| `gen_ai.operation.name` | Required | `"execute_tool"` |
| `gen_ai.tool.name` | Required | Tool name (e.g., `Bash`, `ReadFile`) |
| `gen_ai.tool.call.id` | Recommended | Tool call ID from SDK |
| `gen_ai.tool.type` | Recommended | `"function"` |

### Span Events

7. The sandbox MUST emit `gen_ai.choice` span events attached to the inference span:
   - **Text output**: a `gen_ai.choice` event with a `gen_ai.completion` attribute containing the text content.
   - **Thinking/reasoning output**: a gen_ai.choice event with gen_ai.reasoning_content when the adapter emits thinking (DeepAgents, and Gemini/OpenAI when reasoning is configured per provider-contract.md). When the model emits both completion and thinking content, they MAY be combined into a single gen_ai.choice event with both attributes.

8. There are no separate `audit.agent.started` or `audit.agent.completed` events. The data previously captured by those events (phase, model, provider, success/failure, and total tokens) MUST be recorded as span attributes on the inference span instead.

### Content Capture Policy

9. The `gen_ai.completion` and `gen_ai.reasoning_content` span event attributes contain LLM output that may include PII or sensitive data. Recording these attributes MUST be opt-in via the audit content capture setting. When `LIGHTSPEED_AUDIT_ENABLED=true` and `LIGHTSPEED_CAPTURE_CONTENT` is unset, the sandbox MUST default content capture to on (operator does not set the env today). Set `LIGHTSPEED_CAPTURE_CONTENT=false` to emit `gen_ai.choice` events without content attributes. When audit is disabled, content capture MUST be off. This aligns with the OTel GenAI semantic convention requirement level of Opt-In for content attributes.

### Trace Context Reception

10. The sandbox establishes trace context during each batch run. When audit or OTLP export is enabled (`otel_runtime_enabled()`), `batch.main()` calls `init_tracer()` before `run_agent_query()`. When the operator sets W3C `TRACEPARENT` on the pod (from the active phase span), `batch.main()` passes it to `run_agent_query()` so the inference span is a child of the operator phase span.

11. If `TRACEPARENT` is unset or invalid, the sandbox MUST generate a new trace ID for the run (graceful degradation).

### Single-Emission Rule

12. Each audit-significant datum MUST be recorded exactly once as an OTel span or span event. Multiple exporters / processors on the same TracerProvider produce views of that single emission:
    - **OTLP span exporter** sends spans to the collector (when `OTEL_EXPORTER_OTLP_ENDPOINT` is set).
    - **Stdout exporter** serializes the same span data as OTLP JSON to stdout (when audit is enabled).
    - **Span-event → log processor** forwards the same span events as OTLP log records to the collector (when the endpoint is set **and** audit is enabled) — templog destination, not a second audit write at call sites.

13. Python `logging` MUST emit developer-debugging messages and MUST NOT be used at AuditLogger call sites to re-record span/event data. When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, stdlib logging is dual-shipped to stderr and OTLP (`LoggingHandler` on the root logger). The span-event → log bridge (rule 23) also emits through that same stdlib path so templog gets dual-ship without a separate OTel Logs API emit. This collapses into:
    - OTel spans/events for audit (stdout + OTLP traces), with templog OTLP logs (and stderr) via the bridge → LoggingHandler.
    - Standard logging for developer debugging (stderr + OTLP when the endpoint is set).

### Structured Log Format

14. The stdout exporter MUST emit OTLP JSON — the OTel standard wire format. There is no custom JSON format. Both the stdout and OTLP exporters are destinations for the same TracerProvider spans.

15. The stdout exporter MUST NOT truncate span attributes or event attributes. Full fidelity is preserved. The stdout signal is the compliance record.

### Provider-Specific Instrumentation

16. **DeepAgents / Anthropic** (`providers/deepagents.py`): Emit `gen_ai.choice` span event with `gen_ai.completion` from `AIMessage` text content. Emit `gen_ai.choice` span event with `gen_ai.reasoning_content` from `AIMessage.content_blocks` entries with `type == "reasoning"`. Create `execute_tool {name}` spans from `AIMessage.tool_calls` and `ToolMessage` content. Set `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` from accumulated `usage_metadata`.

17. **OpenAI** (`providers/openai.py`): Emit `gen_ai.choice` with `gen_ai.completion` from stream text deltas (buffered). Emit `gen_ai.reasoning_content` from reasoning delta items when present. Create `execute_tool {name}` spans from tool call/output items. Set token usage from the stream end.

18. **Gemini** (`providers/gemini.py`): Emit `gen_ai.choice` with `gen_ai.completion` from text parts (buffered). Emit `gen_ai.reasoning_content` from thought parts when present. Create `execute_tool {name}` spans from function_call/response parts. Set token usage from the stream end.

### Metrics

19. The sandbox MUST record the following `gen_ai.*` Prometheus histograms during agent execution (`metrics.py`). Histograms are **in-process only** (`prometheus_client`); the batch entrypoint MUST NOT expose a `/metrics` HTTP scrape endpoint and MUST NOT export histograms to OTLP or Pushgateway at shutdown. Short-lived one-shot pods are a poor fit for pull-based Prometheus scraping; **OTLP traces** (with `gen_ai.usage.*` on inference spans) are the operational signal when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Unit tests (`tests/test_metrics.py`) verify histogram recording.

| Metric | Type | Unit | Labels |
|---|---|---|---|
| `gen_ai_client_token_usage` | Histogram | `{token}` | `gen_ai_token_type`, `gen_ai_request_model`, `gen_ai_provider_name`, `gen_ai_operation_name` |
| `gen_ai_client_operation_duration_seconds` | Histogram | `s` | `gen_ai_request_model`, `gen_ai_provider_name`, `gen_ai_operation_name` |
| `gen_ai_execute_tool_duration_seconds` | Histogram | `s` | `gen_ai_tool_name` |

20. Token usage histogram bucket boundaries MUST be `[1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]` (per semconv recommendation). Reasoning tokens are tracked separately via `gen_ai.usage.reasoning_tokens` span attribute on inference spans, not as a `gen_ai.token.type` value.

### Configuration

21. The sandbox receives audit config from the operator via environment variables (`LIGHTSPEED_AUDIT_ENABLED`, `LIGHTSPEED_CAPTURE_CONTENT`, `OTEL_EXPORTER_OTLP_ENDPOINT`, and when OTEL is enabled also `LIGHTSPEED_AGENTICRUN_UID` / `LIGHTSPEED_AGENTICRUN_STEP`). Audit is enabled only when `LIGHTSPEED_AUDIT_ENABLED` is `"true"` after strip and lowercasing (same parsing as `configuration.md` / `batch.py`). Unset and every other value disable audit. When audit is disabled, the sandbox MUST NOT emit `gen_ai.choice` content events and MUST NOT use the stdout audit exporter path gated by that flag. Inference and tool spans may still be created for the agent path (current code and unit tests). When audit is enabled, spans and span events emit per the rules above.

22. When `OTEL_EXPORTER_OTLP_ENDPOINT` is configured (passed from the operator), the sandbox MUST configure OTLP exporters for **both** traces and logs targeting that endpoint. The span-event → log processor MUST be attached only when the endpoint is set **and** audit is enabled (`LIGHTSPEED_AUDIT_ENABLED`), matching the stdout audit exporter gate. When the endpoint is absent, no OTLP exporters and no span-event log forwarding. The stdout span exporter emits OTLP JSON when audit is enabled.

### OTLP Log Emission (Templog) [OLS-3515]

23. When `OTEL_EXPORTER_OTLP_ENDPOINT` is set **and** audit is enabled, the sandbox MUST emit audit span events as OTLP log records to that endpoint, in addition to the stdout and OTLP trace exporters. The same endpoint is used for traces and logs (operator does not set a separate logs endpoint).

24. Each forwarded span-event OTLP log record MUST carry log **record** attributes matching lightspeed-otel-collector postgresexporter: `agenticrun.uid` and `agenticrun.phase` (from `LIGHTSPEED_AGENTICRUN_UID` / `LIGHTSPEED_AGENTICRUN_STEP` when set), and `event` (span event name, e.g. `gen_ai.choice`). These MUST be stamped via stdlib `logging` `extra` so `LoggingHandler` preserves them on the OTel log record. The span event attributes are the log record body (JSON). When content capture is disabled, `gen_ai.choice` events are still forwarded and the body MAY be `{}` (no content attributes). TraceID on the log record MUST come from the ended span's context (bridge attaches that context before logging so `LoggingHandler` correlates). TracerProvider and LoggerProvider share one Resource with pinned `service.name` (not agenticrun uid/phase). Records without `agenticrun.uid` are skipped by the collector. When audit and the OTLP endpoint are enabled but `LIGHTSPEED_AGENTICRUN_UID` and/or `LIGHTSPEED_AGENTICRUN_STEP` cannot be resolved, the sandbox MUST log a warning at startup (do not fail startup). The span-event → log processor MUST NOT forward OTel automatic `exception` events (stack traces); other intentional span events remain eligible for templog.

25. When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, stdlib Python logging MUST be exported as OTLP logs via `LoggingHandler` (dual-ship with stderr). Templog audit records use that same path: the span-event processor logs through stdlib (rules 23–24), not a separate OTel Logs API emit.

26. When `OTEL_EXPORTER_OTLP_ENDPOINT` is absent, no OTLP log records are emitted. Graceful degradation.

## Verification

- Unit: `tests/test_tracing.py` — shared Resource, span-event → OTLP log forwarding (record attrs; audit gate; unresolved AgenticRun env warning), LoggingHandler dual-ship when endpoint set
- Unit: `tests/test_audit.py` — AuditLogger span/event emission (unchanged call sites)
- Unit: `tests/test_metrics.py` — in-process histogram recording (no export path in batch)
- Live (batch cluster): `tests/e2e/features/sandbox_e2e.feature` scenario **Batch run exports traces and audit logs to OTEL** — minimal batch Job with audit enabled; asserts the e2e OTEL collector debug exporter received spans and bridged audit log records carrying `agenticrun.uid` / `agenticrun.phase` for the run (requires `scripts/e2e-install-fixtures.sh`, `E2E_BATCH_VERIFY_FIXTURES=1`)

### MCP Semantic Conventions [UNTRACKED]

27. MCP tool connectivity is implemented. Additional MCP span attributes (`mcp.method.name`, `mcp.session.id`, `mcp.protocol.version`, `network.transport`) are not implemented and have no Jira story. Do not treat this table as a current MUST until a ticket exists. Prefer `gen_ai.tool.*` on tool spans today.

## Cross-References

- `run-api.md` — batch entrypoint where tracing and agent execution run
- `provider-contract.md` — provider adapter event streams where spans and span events are created
- Parent workspace `ols/.ai/spec/what/templog.md` — Temporary audit log storage (cross-repo); sandbox emission tracked by OLS-3515
- `ols/.ai/spec/what/audit-logging.md` — parent spec (authoritative for correlation model, event semantics, OTel GenAI attribute reference)
- [OTel GenAI Semantic Conventions v1.41](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/README.md)
- [OTel MCP Semantic Conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/mcp.md)
