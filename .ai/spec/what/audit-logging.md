# Audit Logging

Implementation spec for compliance audit logging in the agentic sandbox. Parent spec: `ols/.ai/spec/what/audit-logging.md` (authoritative for cross-repo requirements, event semantics, and correlation contract).

## Behavioral Rules

### Audit Events

1. The sandbox MUST emit the following structured JSON audit events to stdout during agent execution. Each event carries `trace_id` (received from the operator via `traceparent` header).

| Event | When | Payload | Notes |
|---|---|---|---|
| `audit.agent.started` | Before SDK agent call begins | Phase, model, provider | |
| `audit.agent.text` | SDK yields complete text block | Text content | LLM's visible reasoning between tool calls. Buffered per-message, not per-token. |
| `audit.agent.thinking` | SDK yields thinking delta | Thinking content | Claude only; OpenAI/Gemini do not expose thinking blocks. |
| `audit.agent.tool.call` | SDK yields tool call event | Tool name, input arguments | All three SDKs expose this. |
| `audit.agent.tool.result` | SDK yields tool result event | Tool name, output, success/failure | All three SDKs expose this. |
| `audit.agent.completed` | SDK run finishes | Success/failure, total tokens, total cost | Token counts at run level only. |

2. The sandbox does not run its own agent loop — it consumes events from the provider SDK's internal agentic loop. Audit events are emitted alongside the existing event normalization in each provider adapter.

### Trace Context Reception

3. The sandbox MUST extract the W3C `traceparent` header from incoming `/v1/agent/run` requests. The trace ID from this header becomes the `trace_id` on all audit events for that run.

4. If no `traceparent` header is present, the sandbox MUST generate a new trace ID for the run (graceful degradation).

### OTEL Spans

5. The sandbox MUST create an `agent.run` span as a child of the operator's span (using the received trace context). Child spans: `agent.turn` (per SDK turn boundary if detectable), `tool.{name}` (per tool call).

6. Since the SDKs own the agent loop and do not expose explicit turn boundaries with usage stats, the sandbox creates `tool.{name}` spans from the tool call/result events it already extracts.

### Provider-Specific Instrumentation

7. **DeepAgents / Anthropic** (`providers/deepagents.py`): Emit `audit.agent.text` from `AIMessage` text content. Emit `audit.agent.thinking` from thinking blocks extracted from `AIMessage.content` (entries with `type="thinking"`). Emit `audit.agent.tool.call` from `AIMessage.tool_calls`. Emit `audit.agent.tool.result` from `ToolMessage` content. Emit `audit.agent.completed` when the stream ends, with token counts from accumulated `usage_metadata`.

8. **OpenAI** (`providers/openai.py`): Emit `audit.agent.text` from `RawResponsesStreamEvent` text deltas (buffered per-message). Emit `audit.agent.tool.call` from `ToolCallItem` events. Emit `audit.agent.tool.result` from `ToolCallOutputItem` events. No thinking events (OpenAI does not expose reasoning). Emit `audit.agent.completed` when the stream ends.

9. **Gemini** (`providers/gemini.py`): Emit `audit.agent.text` from text parts in event content (buffered per-message). Emit `audit.agent.tool.call` from `function_call` parts. Emit `audit.agent.tool.result` from `function_response` parts. No thinking events (Gemini does not expose reasoning). Emit `audit.agent.completed` when the stream ends.

### Configuration

10. The sandbox receives audit config from the operator via environment variable or config mount. When audit is enabled, all events emit. When disabled, no audit events emit.

11. When an OTEL tracing endpoint is configured (passed from operator), the sandbox configures an OTLP span exporter. When absent, a no-op exporter is used. Structured JSON to stdout always emits when audit is enabled.

### OTLP Log Emission (Templog)

12. When the OTLP log endpoint environment variable is set (wired by the lightspeed-operator when `spec.templog` is enabled), the sandbox MUST also emit all audit events as OTLP log records to that endpoint. This is in addition to stdout — dual emission.

13. Each OTLP log record MUST carry: `trace_id` in the log record's trace context (received via `traceparent` header), `event` as a log record attribute, and the full structured JSON audit event as the log record body.

14. The OTLP log endpoint is independent of the OTEL tracing endpoint. Both can be active simultaneously.

15. When the OTLP log endpoint is absent, no OTLP log records are emitted. Graceful degradation.

### Structured JSON Format

16. All audit events MUST be single JSON lines to stdout with at minimum: `timestamp`, `level`, `event`, `trace_id`. Additional fields vary by event type per the catalog above.

17. The `phase` field (analysis/execution/verification/escalation) MUST be included on every event. The sandbox can derive this from the request context (the operator's `/v1/agent/run` request carries phase context).

## Cross-References

- `run-api.md` — `/v1/agent/run` endpoint where trace context arrives
- `provider-contract.md` — provider adapter event streams where audit hooks are added
- Parent `templog.md` — Temporary audit log storage: OTLP log emission architecture
