# Audit Logging

Implementation spec for compliance audit logging in the agentic sandbox. Parent spec: `ols/.ai/spec/what/audit-logging.md` (authoritative for cross-repo requirements, standard agent/model/tool spans, correlation, and the product-data boundary).

Telemetry conforms to the official [OpenTelemetry GenAI Semantic Conventions v1.41.0](https://github.com/open-telemetry/semantic-conventions/tree/v1.41.0/docs/gen-ai). The Tracer instrumentation scope uses schema URL `https://opentelemetry.io/schemas/1.41.0`; this does not change the shared Resource.

## Behavioral Rules

### Span Hierarchy and Kinds

1. `run_agent_query()` MUST create one `invoke_agent lightspeed` span per sandbox agent invocation. It MUST be kind `INTERNAL`, carry `gen_ai.operation.name=invoke_agent` and `gen_ai.agent.name=lightspeed`, and cover the SDK's complete agent loop. The incoming operator phase span is its parent when valid W3C trace context is supplied.

2. Each actual SDK inference request MUST have its own `{gen_ai.operation.name} {gen_ai.request.model}` span of kind `CLIENT`, parented directly to `invoke_agent lightspeed`. Use `chat` for DeepAgents/OpenAI and `generate_content` for Gemini. Span boundaries MUST match the actual request start and completion or error, not the complete provider query, normalized stream, or terminal `ResultEvent`. Multiple SDK requests in one agent invocation produce multiple inference spans.

3. Each actual tool execution MUST have one `execute_tool {gen_ai.tool.name}` `INTERNAL` span, parented directly to the agent invocation (a sibling of inference spans). Start and end it at the actual execution boundaries, not when a normalized stream event is emitted. Use the provider's tool call ID only when actually supplied; do not substitute a framework run ID or fabricated ID.

4. Adapters MUST observe provider-native SDK lifecycle callbacks (or a thin wrapper at the actual request boundary) for inference and tool spans. Normalized `ProviderEvent` messages remain solely for application results/output and developer logging; they MUST NOT produce duplicate telemetry spans. If an SDK already emits a matching source span, instrumentation MUST reuse or adapt it rather than add a duplicate.

### Span Attributes and Endpoint Routing

5. `gen_ai.operation.name` MUST identify the operation on each GenAI span. Agent spans use `invoke_agent`, `gen_ai.agent.name=lightspeed`, the endpoint provider, and the requested model when known; inference spans use their actual operation and `gen_ai.provider.name`; tool spans use `execute_tool` and `gen_ai.tool.name`. Other attribute requirement levels follow v1.41.0.

6. `gen_ai.provider.name` MUST identify the service serving the configured endpoint, not the adapter or framework. Examples are `anthropic` for the direct Anthropic API, `openai` for OpenAI, `gcp.gemini` for Gemini API, and `gcp.vertex_ai` for Vertex AI. The shared `resolve_provider_name()` helper in `audit.py` resolves it from the actual provider/endpoint route (including overrides); names such as `deepagents`, `google-adk`, or `openai-agents` MUST NOT be used as provider values.

7. `gen_ai.request.model` MUST use the actual requested model when known. `gen_ai.response.model` MUST be set only when the actual returned model is exposed by the SDK callback (including native Responses streaming); if SDK normalization drops it, leave it absent. Never copy `gen_ai.request.model` into `gen_ai.response.model`. Record `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, and `gen_ai.usage.reasoning.output_tokens` only when actually observed. When provenance is preserved, an actual SDK-reported `reasoning_tokens=0` MUST be recorded as `gen_ai.usage.reasoning.output_tokens=0` (native Responses usage can preserve it). The installed OpenAI ChatCompletions normalization reconstructs `0` for both absent and explicit zero and erases provenance, so its normalized zero MUST be omitted as unavailable; synthetic default zeros and absent usage MUST also be omitted. Record observed `gen_ai.response.finish_reasons` and recommended `server.address` when available; `server.port` is required when `server.address` is set.
Gemini output usage is the observed candidate count plus thoughts when supplied. Missing thoughts MUST NOT suppress an observed candidate count or fabricate a reasoning-token count.

8. Tool spans MUST set `gen_ai.tool.name`; `gen_ai.tool.call.id`, `gen_ai.tool.type`, and `gen_ai.tool.description` SHOULD be set when actually available. `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` are Opt-In content attributes. Adapters pass structured Python values to `AuditLogger`; the shared recorder JSON-serializes them only when the span is recording. Record available arguments and the complete raw callback result when native execution succeeds, regardless of later inspection; omit a result only on native execution failure. Source-value serialization MUST use lossless JSON escaping so decoded raw values survive protobuf-backed OTLP export unchanged.
OpenAI observes `RunHooks.on_tool_start` / `on_tool_end` and wraps `FunctionTool` invocation and failure handlers because RunHooks has no failure callback. It MUST NOT patch shell sessions, MCP extractors, filesystem editors, or individual built-in tool types to recover signals hidden inside the SDK. A failure already converted to a normal return by the SDK is recorded as the hook's result, not inferred from its text or tool name. An ordinary nonzero command exit is not itself reclassified as failure.
When OpenAI exposes a tool call ID, pending-span lookup MUST require exactly one match for that tool and ID; a missing or ambiguous match MUST NOT fall back to another call's context or arguments. Heuristic matching is permitted only when no call ID is supplied.

### Standard Message and Tool Content

9. When a source span is recording, `invoke_agent lightspeed` and every non-classifier inference span, including main-agent, shape, and subagent requests, MUST retain the full available input, output, instruction, and tool-definition content in standard v1.41.0 attributes. The agent span MUST record the exact post-shaped `AgentResult.output` whenever a terminal result is normally produced, including domain `success=false`; it MUST omit output only when invocation ends before a terminal result, such as an exception, timeout, cancellation, or guardrail abort. Each successful native tool span MUST retain available arguments and the complete raw callback result at actual SDK completion, independently of inspection; inspection gates model and application-result delivery, not source-span retention. These source fields are independent of `LIGHTSPEED_CAPTURE_CONTENT`; captured content MUST NOT be redacted or truncated beyond upstream SDK/adapter limits. Adapters pass structured Python objects to `AuditLogger`, which JSON-serializes them only when the span is recording. The content fields are `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.definitions`, `gen_ai.tool.call.arguments`, and `gen_ai.tool.call.result`. A standalone safety-classifier inference span is the privacy exception: it MUST still record actual timing, endpoint provider, requested/observed model, observed usage, and error status/type, but MUST omit `gen_ai.input.messages`, `gen_ai.output.messages`, and `gen_ai.system_instructions` even when recording. Do not expose classifier prompts or outputs through inspection telemetry or developer logs; see the parent tool-result-inspection contract rules 105, 105d, and 105f.
Inference message attributes MUST reflect the available request content actually visible to the provider, including a SAFE-02 model-facing wrapper only if and when that separately planned OLS-3929 behavior is implemented; this telemetry rule does not implement or imply that wrapper. `gen_ai.tool.call.result` MUST contain only the complete raw result of successful native execution, never a model-facing wrapped copy.

10. `gen_ai.input.messages` MUST be a JSON-string array conforming to the pinned [input-message schema](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-input-messages.json). Preserve provider message order; each message has a `role` and ordered `parts`.

11. `gen_ai.output.messages` MUST be a JSON-string array conforming to the pinned [output-message schema](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-output-messages.json). Emit one complete message per response choice/candidate, with `role`, ordered `parts`, and `finish_reason`; use the standard `unknown` value when the SDK does not expose a finish reason. Reasoning is a part such as `{"type":"reasoning","content":"..."}`. Tool calls use `{"type":"tool_call","id":"...","name":"...","arguments":...}` and tool responses use `{"type":"tool_call_response","id":"...","response":...}`. Include IDs only when supplied; never invent a provider call ID.

12. `gen_ai.system_instructions` MUST remain separate from user/model messages and use the standard JSON array of instruction parts, not a role-bearing message array. Record `gen_ai.tool.definitions` separately when available. The agent span's `gen_ai.output.messages` MUST record the exact post-shaped result when a terminal `AgentResult` is normally produced, including domain `success=false`; it MUST be absent for pre-terminal invocation failures.

### Status and Correlation

13. Normally completed agent invocations, inference requests, and native tool executions MUST leave OTel status `UNSET`; they MUST NOT set status `OK`. Native agent, inference, or tool failures MUST set status `ERROR` and a low-cardinality `error.type`; do not use sensitive error text. A successful native tool span MUST retain `gen_ai.tool.call.result` even if later inspection rejects it, the classifier fails, or a sibling fails; completed parallel siblings retain their own results and native end times. Only native execution failures and operations still open when interrupted are tool-span errors; interruption MUST close only unfinished operations. `tool_result.inspection` spans MUST be `UNSET` for valid `benign` or `malicious` decisions and `ERROR` with controlled failure metadata for `classifier_error`, including cancellation. A fail-closed rejection or classifier failure leaves the enclosing `invoke_agent` span `ERROR` with controlled metadata and no terminal output, while a normally produced domain `success=false` result remains a completed invocation. These rules classify telemetry only; they MUST NOT alter SDK execution, model-visible error replies, or final-success/application behavior.

14. When `LIGHTSPEED_AGENTICRUN_UID` and `LIGHTSPEED_AGENTICRUN_STEP` are supplied, the sandbox MUST put their literal values in `agenticrun.uid` and `agenticrun.phase` on the agent, inference, and tool spans. Each inspection span MUST receive only those literal values from the existing invocation configuration when supplied; missing values remain absent and MUST NOT be inferred from a Resource, process environment, parent context, or another span. Product eligibility requires both values on the span itself; Resource attributes are not a fallback.

### Compliance Content Policy

15. The compliance stdout and templog projections MUST be derived from completed source spans. `LIGHTSPEED_CAPTURE_CONTENT` filters only the six standard content fields listed in rule 9 on those compliance copies; it MUST NOT mutate source spans or change OTLP trace export. When unset, compliance content capture follows `LIGHTSPEED_AUDIT_ENABLED`; setting it to false omits those fields from the copies. Content-enabled compliance copies may retain successful raw tool output later rejected by inspection.

### Trace Context Reception

16. When audit or OTLP export is enabled (`otel_runtime_enabled()`), `batch.main()` calls `init_tracer()` before `run_agent_query()`. When the operator supplies W3C `TRACEPARENT` from its active phase span, `invoke_agent lightspeed` is a child of that phase span, and model/tool spans are its children.

17. If `TRACEPARENT` is unset or invalid, the sandbox MUST create a new trace for the agent invocation (graceful degradation).

### Single-Emission Rule

18. Each actual agent, inference, or tool operation MUST have one source span. The OTLP trace exporter sends full source spans through the shared endpoint whenever configured; the stdout compliance exporter emits its derived view when audit is enabled; and the completed GenAI span-to-log projection is enabled only when both the endpoint and audit are enabled. Audit-disabled MUST NOT suppress trace export. Source span attributes MUST NOT be re-recorded through `AuditLogger` call-site Python logging.

    Stdout emits OTLP JSON and MUST NOT truncate its compliance view. Each completed GenAI span-derived OTLP log record uses `event` for the `gen_ai.operation.name`, puts the filtered JSON span attributes in the log body, and preserves the ended span's TraceID/SpanID plus `agenticrun.uid` and `agenticrun.phase` record attributes when supplied. The TracerProvider and LoggerProvider share the same Resource and `service.name`; run correlation remains on spans and log records, not the Resource. The bridge uses stdlib `logging` and its `LoggingHandler`, dual-shipped with stderr when the endpoint is configured; it is not a separate OTel Logs API emission. Unrelated developer logs and non-GenAI span events keep their existing behavior. With `http/protobuf`, the shared `OTEL_EXPORTER_OTLP_ENDPOINT` is a base URL: trace and log exporters append `/v1/traces` and `/v1/logs`, trimming a trailing slash and preserving any configured base path; `grpc` continues to use the shared endpoint unchanged.

### Tool-Result Inspection

18a. Sandbox telemetry MUST conform to `openshift/ols/.ai/spec/what/tool-result-inspection.md`.

18b. Each DeepAgents inspection that reaches classification MUST create the existing per-chunk `tool_result.inspection` span and attach it to the parent agent trace. Do not fabricate a span or verdict if inspection was disabled, not reached, interrupted, or serialization failure prevented classifier dispatch.

18c. The sandbox can add controlled tool, provider, model, and tool-call correlation identifiers to the contract-defined attributes. Each inspection span MUST include `gen_ai.tool.call.id` only when the inspected `ToolMessage` exposes a non-empty call ID; use that ID unchanged. Each inspection span MUST also include the literal `agenticrun.uid` and `agenticrun.phase` values supplied in the existing invocation configuration, when present, without Resource, environment, or parent-context fallback; absent values remain absent.

18d. Existing generic inference instrumentation can observe classifier calls. The sandbox MUST add no feature-specific Prometheus metric.

18e. Developer logs and `tool_result.inspection` telemetry MUST NOT contain tool arguments, tool results, or tool-generated errors.

18f. At successful native tool completion, `AuditLogger` MUST retain the complete raw callback result on that source span's `gen_ai.tool.call.result`, when the span is recording, regardless of any later inspection outcome. Inspection rejection does not retroactively alter the completed tool span.

18g. Compliance projections apply their six-field content filter after source spans are recorded. Content-enabled copies may retain rejected raw tool output; `LIGHTSPEED_CAPTURE_CONTENT=false` removes the six standard content fields from copies only and MUST NOT change the source span or trace export.

18h. A result serialization or encoding failure before classifier dispatch MUST fail closed through the controlled `ToolResultSafetyInspectionFailed` path, without fabricating an inspection span or outcome and without marking an already completed successful native tool span as a native execution failure or removing its raw result. Raw payloads and exception text MUST NOT appear in inspection telemetry or developer logs.
18i. Classifier-task cancellation MUST mark the `tool_result.inspection` span `ERROR` with controlled `inspection.outcome=classifier_error` and `error.type=cancelled`; raw tool-result content and exception/error text MUST NOT be recorded in attributes, events, or logs.

### Metrics

19. The sandbox MUST record the following Prometheus histograms in-process during actual operations (`metrics.py`). Samples are recorded at the actual request/execution boundary even when a trace span is not recording; token usage is recorded only from observed counts. Histograms are not exported to OTLP or Pushgateway, and the batch entrypoint MUST NOT expose a `/metrics` endpoint. OTLP traces remain the operational token/duration signal when configured.

| Metric | Type | Unit | Labels / scope |
|---|---|---|---|
| `gen_ai_client_token_usage` | Histogram | `{token}` | `gen_ai_token_type`, `gen_ai_request_model`, `gen_ai_provider_name`, `gen_ai_operation_name` |
| `gen_ai_client_operation_duration_seconds` | Histogram | `s` | `gen_ai_request_model`, `gen_ai_provider_name`, `gen_ai_operation_name`; `error_type` only for failures |
| `gen_ai_execute_tool_duration_seconds` | Histogram | `s` | `gen_ai_tool_name`; project-specific extension, not a v1.41.0 GenAI metric |

20. Token-usage bucket boundaries MUST be `[1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]` (the v1.41.0 recommendation). The standard `gen_ai.token.type` values are `input` and `output`; observed reasoning-token counts use the `gen_ai.usage.reasoning.output_tokens` span attribute, not another token type.

20a. The standard `gen_ai.client.operation.duration` histogram uses recommended boundaries `[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92]`. The `error_type` Prometheus label represents OTel `error.type` and is populated only when the operation fails; successful operations leave it unset. Measure each actual operation, not an aggregate provider query. v1.41.0 defines no tool-duration histogram; the existing tool histogram is a project extension.
Duration metric error labels MUST use a bounded vocabulary: `TimeoutError`, `CancelledError`, `operation_cancelled`, `empty_response`, `error`, `response.error`, `response.failed`, `response.incomplete`, and `response_incomplete`. Other exception types or strings map to `_OTHER` for metrics; source spans retain their diagnostic error type.

### Configuration

21. The sandbox receives audit and shared tracing configuration through `LIGHTSPEED_AUDIT_ENABLED`, `LIGHTSPEED_CAPTURE_CONTENT`, and `OTEL_EXPORTER_OTLP_ENDPOINT`; run correlation uses `LIGHTSPEED_AGENTICRUN_UID` and `LIGHTSPEED_AGENTICRUN_STEP`. Audit is enabled only when `LIGHTSPEED_AUDIT_ENABLED` is `"true"` after strip and lowercasing. Audit-disabled suppresses compliance stdout and span-derived log copies, but MUST NOT suppress source tracing when the shared OTLP endpoint is configured.

22. When `OTEL_EXPORTER_OTLP_ENDPOINT` is configured, OTLP trace export and stdlib OTLP log export target that endpoint. Trace export is active whenever the endpoint is set. The completed GenAI span-to-log bridge is attached only when the endpoint is set and audit is enabled; the stdout compliance exporter emits when audit is enabled. When the endpoint is absent, no OTLP exporters or span-derived log forwarding are configured.

### OTLP Log Emission (Templog) [OLS-3515]

23. When `OTEL_EXPORTER_OTLP_ENDPOINT` is set and audit is enabled, the sandbox MUST emit the compliance view of each completed GenAI span as an OTLP log record to that endpoint, in addition to stdout and OTLP trace export.

24. Each forwarded log record MUST carry record attributes `agenticrun.uid` and `agenticrun.phase` from the environment when supplied, plus `event` equal to the span's `gen_ai.operation.name`. Its body is the JSON compliance view of the span attributes after filtering. TraceID and SpanID MUST come from the ended span's context. The log and trace providers share the Resource with pinned `service.name`; run UID and phase remain record/span attributes. If either configured correlation value cannot be resolved while audit and the OTLP endpoint are enabled, the sandbox MUST log a startup warning without failing startup. The bridge MUST NOT forward automatic OTel `exception` events; unrelated intentional non-GenAI span events remain unchanged.

25. When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, stdlib Python logging MUST be exported as OTLP logs via `LoggingHandler` (dual-ship with stderr). Span-derived compliance records use that same path, not a separate OTel Logs API emit.

26. When `OTEL_EXPORTER_OTLP_ENDPOINT` is absent, no OTLP log records are emitted. Graceful degradation.

### Agentic Product Trace Boundary

27. The sandbox MUST emit standard v1.41.0 agent, inference, and tool spans through the existing shared trace runtime. `data-collection.md` defines only the sandbox producer boundary; the parent `ols/.ai/spec/what/agentic-data-collection.md` owns cross-repository collection eligibility and downstream interpretation. The sandbox MUST NOT add a product-specific input/output event catalog or a second content-event source.

## Verification

- Focused offline suite (controller-reported: 295 passed): `tests/test_audit.py`, `tests/test_deepagents_telemetry.py`, `tests/test_deepagents.py`, `tests/test_tool_result_inspection_client.py`, `tests/test_tool_result_inspection_middleware.py`, `tests/test_tool_result_inspection_telemetry.py`, `tests/test_run_agent.py`, `tests/test_gemini_telemetry.py`, `tests/test_tracing.py`, and `tests/test_logging.py` cover native result timing/status, inspection outcomes/correlation, model and event rejection, exact terminal output, payload-free diagnostics, capture filtering, and call IDs.
- Offline native SDK proof: ADK Runner/FunctionTool dispatch preserves supplied and SDK-assigned IDs in finalized model output and actual tools, while IDs stripped from later effective requests remain absent. Cleanup smokes exercised the actual OpenAI Runner with parallel same-name FunctionTools, the ADK Runner with a local subprocess tool, and native LangChain model/tool dispatch. OpenAI retained separate successful and failed spans; ADK retained observed model/usage and joined the finalized call ID to the raw result; LangChain retained the completed raw tool span while model-boundary rejection blocked delivery.
- CodeRabbit regressions: `tests/test_gemini_telemetry.py` covers candidate-only usage and omitted reasoning; `tests/test_metrics.py` covers bounded error labels on completion and cleanup with preserved exported span diagnostics; `tests/test_openai_telemetry.py` covers overlapping same-tool failures and ambiguous call IDs. A separate offline smoke used the actual OpenAI Runner with a deterministic model and concurrent FunctionTool calls: the failed call retained `ERROR` and no result, while its sibling retained its own subprocess output and native end time. Importing OpenAI telemetry also succeeded with `agents` imports deliberately blocked.
- Local HTTP OTLP trace/log and stdout proof: actual exports were exercised with content capture both enabled and disabled for benign/offloaded, malicious, `classifier_error`, and invalid-encoding cases. Each run emitted 21 source spans; rejected raw results remained on successful `UNSET` tool spans, model continuation was blocked, content-disabled compliance copies omitted the six fields without mutating source spans, payloads stayed out of inspection/classifier/developer telemetry, and lossless JSON escaping preserved an unpaired surrogate's decoded value through protobuf-backed OTLP export. The emitted `gen_ai.input.messages` (18), `gen_ai.output.messages` (12), `gen_ai.system_instructions` (16), and `gen_ai.tool.definitions` (8) attributes were validated against the official v1.41.0 schemas across both runs.
- Repository verification uses Ruff formatting/lint, mypy, hermetic dependency accounting, and the full offline unit suite. GNU `patch` is required by the existing filesystem-tool success test; on workstations without it, a genuine distribution binary can be extracted into a temporary directory and added to the test subprocess `PATH` without installing packages or changing repository dependencies. RHOAI curated-version skew warnings are non-failing when all runtime packages are accounted for.
- Controller-reported final receiver review: HTTP routing passed compliance and code-quality re-review. The path-plus-query regression failed before the URL-path fix and passed afterward; all 26 tracing tests passed, including real per-signal request targets, protobuf delivery/correlation, and guaranteed receiver cleanup.
- Live (batch cluster): `tests/e2e/features/sandbox_e2e.feature` scenario **Batch run exports traces and audit logs to OTEL** remains the live-cluster check (requires `scripts/e2e-install-fixtures.sh`, `E2E_BATCH_VERIFY_FIXTURES=1`); it was not exercised by the evidence above.

### MCP Semantic Conventions [UNTRACKED]

28. MCP tool connectivity is implemented. Additional MCP span attributes (`mcp.method.name`, `mcp.session.id`, `mcp.protocol.version`, `network.transport`) are not implemented and have no Jira story. Do not treat this table as a current MUST until a ticket exists. Prefer `gen_ai.tool.*` on tool spans today.

## Cross-References

- `run-api.md` — batch entrypoint where tracing and agent execution run
- `provider-contract.md` — native SDK lifecycle hooks used for GenAI spans
- Parent workspace `ols/.ai/spec/what/templog.md` — temporary audit log storage (cross-repo); sandbox emission tracked by OLS-3515
- `ols/.ai/spec/what/audit-logging.md` — parent spec authoritative for correlation and OTel GenAI semantics
- `data-collection.md` — sandbox source-span producer boundary
- `ols/.ai/spec/what/agentic-data-collection.md` — canonical cross-repository collection contract
- [OTel GenAI Semantic Conventions v1.41.0](https://github.com/open-telemetry/semantic-conventions/tree/v1.41.0/docs/gen-ai)
- [OTel GenAI agent spans v1.41.0](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-agent-spans.md)
- [OTel GenAI inference and tool spans v1.41.0](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-spans.md)
- [OTel GenAI input-message schema v1.41.0](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-input-messages.json)
- [OTel GenAI output-message schema v1.41.0](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-output-messages.json)
- [OTel GenAI metrics v1.41.0](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-metrics.md)
- [OTel MCP Semantic Conventions v1.41.0](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/mcp.md)
- Parent workspace `ols/.ai/spec/what/tool-result-inspection.md` — cross-repository tool-result inspection contract
