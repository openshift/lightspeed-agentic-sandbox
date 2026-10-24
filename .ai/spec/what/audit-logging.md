# Audit Logging

Implementation spec for compliance logging and the named three-model-family trace profile. Parent spec: `ols/.ai/spec/what/audit-logging.md` remains authoritative for cross-repository audit/logging and correlation requirements; the sandbox producer exception is scoped in `data-collection.md`, without changing the parent contract.

Telemetry follows the named [OTel GenAI profile pinned at commit `4f85037ef86e92c510d2ef881a58f1076f6fc0e4`](https://github.com/open-telemetry/semantic-conventions-genai/tree/4f85037ef86e92c510d2ef881a58f1076f6fc0e4/docs/gen-ai). Upstream status at that revision is Development; this does not claim every optional convention or add a dependency.

## Harness Consolidation [PARTIALLY IMPLEMENTED: OLS-4434]

Anthropic, Vertex/Google, native/custom OpenAI, and Azure routes use shared DeepAgents/LangChain generation capture. The legacy OpenAI adapter and its hooks are removed. Vertex/OpenAI is unimplemented pending OLS-4472. Azure routing/workflows are offline-tested; live acceptance and the parent telemetry compatibility/owner-review gate remain separate. The Gemini ADK adapter and its capture path are removed. For migrated routes, [provider-contract.md rules 57 and 60–62](provider-contract.md#deepagents-consolidation-partially-implemented-ols-4434) and [data-collection.md](data-collection.md#harness-consolidation-partially-implemented-ols-4434) describe current capture; Agents SDK-specific requirements below (`ModelResponse`, `requests > 0`, hooks, and `openai.api.type`) document the retired harness only and no longer apply to runtime routes. Native/custom OpenAI and Azure follow the shared DeepAgents callback/result path, with provider `openai`. Keep normalized event types, approved audit/log projections, run correlation, real provider/call metadata, usage, and configured capture/export gates; DeepAgents built-in tools apply to Anthropic, Vertex Gemini, and native/custom OpenAI. Inspected tool paths use payload-free developer logging for all DeepAgents routes.

The parent inspection/collection contract remains authoritative for native source evidence: a successful native tool source span retains its raw callback result independently of later inspection. Rejection suppresses normalized result events and Result CRs, not completed source evidence; compliance copies follow the parent content policy. This exception takes precedence over older statements below that could otherwise be read to suppress all canonical source-result attributes. Changed native span/event/message behavior requires the parent component/Dataverse compatibility review and version decision before implementation merges; migration grants no waiver.

## Behavioral Rules

### Span Naming and Kinds

1. The sandbox MUST create an `invoke_agent` span with `gen_ai.operation.name="invoke_agent"` and `SpanKind.INTERNAL`, as a child of the received operator context. It carries the configured request model and is not a provider inference span: it MUST NOT claim `gen_ai.provider.name` or `gen_ai.response.model`.

2. The sandbox MUST create an `execute_tool {gen_ai.tool.name}` span for each local tool execution. Tool spans are `INTERNAL` children of the invocation span and use only actual SDK tool-call IDs.

3. `run_agent_query()` and `AuditLogger` create canonical invocation/tool spans; DeepAgents and OpenAI create canonical main-agent generation spans. Each accepted generation and each local `execute_tool` span is a direct child of the canonical `invoke_agent` span, which is itself a child of the received operator context. Spans under the exact `gcp.vertex.agent` instrumentation scope are omitted only by the stdout and OTLP trace exporters (rule 4h); this filter does not imply an active ADK provider adapter.

### GenAI Attributes — Invocation Span

4. The `invoke_agent` span MUST carry the profile attributes below and retain native span context, start/end time, and OTel status.

| Attribute | Requirement | Description |
|---|---|---|
| `gen_ai.operation.name` | Required | `"invoke_agent"` |
| `gen_ai.request.model` | Required | Configured request model; not an actual response model |
| `gen_ai.input.messages` | Required | JSON message array containing the effective post-context-prefix user prompt |
| `gen_ai.system_instructions` | Required | JSON instruction array containing the configured supplied system string |
| `gen_ai.output.type` | Conditional | `"json"` iff `output_schema is not None` |
| `gen_ai.output.messages` | Conditional | Exact observed `ResultEvent.text` before parsing/shaping; empty is present, no `ResultEvent` is omitted |
| `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` | Existing aggregate | Preserve the existing terminal `ResultEvent` usage behavior |
| `gen_ai.usage.reasoning.output_tokens` | Nonzero only | Existing aggregate reasoning count; do not write zero when absent/zero |
| `error.type` | On operational root error | `timeout`, exception class name (including `CancelledError`), or `empty_response`; cancellation propagates unchanged. Unsuccessful domain outcomes remain output content, not span errors. |
| `agenticrun.uid` / `agenticrun.phase` | When available | Existing run correlation values; do not invent missing values |

Message attributes are compact JSON strings matching the pinned schemas. The invocation captures the effective prompt and configured instructions supplied to the run, not all SDK-added instructions or repeated request history. The root output is a result projection, not another generation. Root usage remains aggregate and MUST NOT be summed with child-generation usage.

### GenAI Attributes — Provider Generation Span

4a. DeepAgents and OpenAI MUST create `SpanKind.CLIENT` `chat {gen_ai.request.model}` child spans with operation `chat`. This includes Vertex Gemini through the shared DeepAgents route; it does not create a separate `generate_content` span. Each span uses the configured request model, actual provider name, and available invocation `agenticrun.uid`/`agenticrun.phase`. Generation spans use the invocation context captured before the provider query and MUST NOT become current.

4b. DeepAgents provider names come from the resolved route: `anthropic`, `aws.bedrock`, `gcp.vertex_ai` (including Vertex Gemini), or `openai` (native/custom OpenAI and Azure). Vertex/OpenAI has no runtime route pending OLS-4472. Azure's version-selected API uses the shared DeepAgents/LangChain callback and observed metadata, not legacy hooks.

4c. Generation `gen_ai.output.messages` MUST contain only SDK-observed output and preserve source message/item/content/part order. The DeepAgents structured-output shaping request is a separate generation whose output is the raw model response. Generation spans MUST NOT repeat input/system histories or reconstruct output from `ProviderEvent`/legacy choice deltas. Provider-specific part mapping is defined in `data-collection.md`.

4d. Set `gen_ai.response.model`, `gen_ai.response.id`, and `gen_ai.response.finish_reasons` only from actual SDK evidence. DeepAgents and OpenAI MUST NOT substitute configured models, LangChain run IDs, or transport/item IDs. Vertex Gemini follows the DeepAgents LangChain result-metadata path; ADK `Event.id` and `model_version` fields are not used by this route.

4e. DeepAgents MUST preserve present input/output/reasoning usage counts, including explicit zero, for its migrated routes, and omit absent counts. The legacy OpenAI adapter MUST read usage only when `ModelResponse.usage.requests > 0`, preserving present zero input/output counts and omitting missing counts; emit reasoning output tokens only for supplied nonzero reasoning detail. Child-generation usage MUST NOT be summed with invocation aggregate usage.

4f. DeepAgents retains failure output only when the callback exposes an `LLMResult`, including for Vertex Gemini; OpenAI records output only from a completed response and does not recover stream deltas. Observed generation failures are ERROR; exceptions/cancellation propagate unchanged after span closure, and an early close without an exception uses `generation_interrupted`. No provider adds delta recovery.

4g. Vertex Gemini uses the same DeepAgents LangChain callback as Anthropic. The callback selects main-agent chat-model requests, records the observed `LLMResult`/`AIMessage` output and metadata, and closes the span at model completion. It does not register ADK callbacks, consume Runner events, or classify ADK `function_response` values. Failure and interruption behavior follows rule 4f.

4h. The stdout and OTLP trace exporters MUST exclude spans whose instrumentation scope name is exactly `gcp.vertex.agent`; no rename, projection, or attribute normalization is applied. The exclusion affects only these two trace exporters: native spans continue through TracerProvider processors, their events, IDs, parentage, status, resources, scope, and dropped counters remain unmodified, and every other instrumentation scope is unaffected. Existing log correlation, event contents, processors, and log gates remain unchanged. `invoke_agent` remains a child of the received operator context, and each accepted canonical generation and local `execute_tool` span remains directly parented to `invoke_agent`, so excluding vendor-scope spans MUST NOT orphan or reparent canonical spans.

4i. Vertex Gemini generation capture follows the shared DeepAgents/LangChain `chat` callback path. The runtime records only LangChain-exposed results; it does not reconstruct provider deltas or claim token chronology.

### GenAI Attributes — Tool Span

6. Each local tool span MUST retain native span context, start/end time, and OTel status.

| Attribute | Requirement | Description |
|---|---|---|
| `gen_ai.operation.name` | Required | `"execute_tool"` |
| `gen_ai.tool.name` | Required | Tool name |
| `gen_ai.tool.call.id` | Optional | Actual SDK ID only; omit when unavailable |
| `gen_ai.tool.call.arguments` | When observed | JSON string encoding a JSON object |
| `gen_ai.tool.call.result` | When observed | JSON string encoding a JSON object |
| `agenticrun.uid` / `agenticrun.phase` | When available | Same existing correlation values as the invocation span |

Use strict JSON parsing for tool strings: accept standards-compliant finite JSON only; treat `NaN`, `Infinity`, `-Infinity`, exponent overflow, and parser/encoder-limit failures as non-JSON. Pass decoded dictionaries unchanged and wrap other decoded values as `{"content": value}`. On parse, encode, or limit rejection, preserve the complete original raw string in `{"content": raw_string}`; this is sandbox normalization, not a provider-native field. Never truncate it or raise a telemetry-only provider error. Preserve observed empty values and omit missing arguments/results. A missing-ID result may match only one pending call; ambiguous results stay unmatched. Unresolved calls end ERROR as `error.type="missing_tool_result"`; on cancellation, close pending tool spans without result or tool-duration histogram observation.

### Legacy Choice Events

7. Existing `gen_ai.choice` span events remain attached to the invocation span as a legacy audit projection. Preserve their existing text/reasoning mapping, emission/content gates, and buffer flush order. The buffered text is flushed before reasoning, so these events are not a canonical cross-type chronology. Cancellation does not force another choice/developer-log buffer flush or emit additional choice-derived templog records.

8. Do not add separate `audit.agent.started`/`audit.agent.completed` events or custom transcript events. Canonical invocation status, output, and usage follow the span-attribute profile above; provider identity and actual response model are not inferred for the root.

### Content Capture Policy

9. Whenever an invocation, tool, or generation span is recording, canonical profile attributes ignore `LIGHTSPEED_AUDIT_ENABLED` and `LIGHTSPEED_CAPTURE_CONTENT`. Existing `gen_ai.choice` emission remains audit-gated; its text/reasoning payload retains the prior content-capture policy. Developer logs and the span-event → templog bridge retain their existing gates, payloads, and flush order; none is a canonical transcript.

### Trace Context Reception

10. When audit or OTLP export is enabled (`otel_runtime_enabled()`), `batch.main()` calls `init_tracer()` before `run_agent_query()`. When the operator sets W3C `TRACEPARENT` on the pod, `batch.main()` passes it to `run_agent_query()` so `invoke_agent` is a child of the operator phase span.

11. If `TRACEPARENT` is unset or invalid, the sandbox MUST generate a new trace ID for the run (graceful degradation).

### Trace and Log Projections

12. Canonical invocation, local-tool, and provider-generation spans flow through the shared TracerProvider. `invoke_agent` is a child of the received operator context; each accepted generation and local `execute_tool` span is a direct child of `invoke_agent`. Legacy choice events remain a separate audit projection. At the export boundary only, the stdout and OTLP trace exporters exclude spans whose instrumentation-scope name is exactly `gcp.vertex.agent`; the filter does not rename or normalize spans, mutate native spans/events/parents or processors, or affect other scopes.
    - **OTLP trace exporter** omits only the exact `gcp.vertex.agent` scope and sends remaining trace data when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
    - **Stdout exporter** omits that same scope and serializes remaining spans as OTLP JSON when audit is enabled.
    - **Span-event → log processor** continues forwarding original audit events only when the endpoint and audit are enabled; existing span-event log correlation and event contents remain unchanged.

13. Python `logging` MUST emit developer-debugging messages and MUST NOT be used at AuditLogger call sites to re-record span/event data. When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, stdlib logging is dual-shipped to stderr and OTLP (`LoggingHandler` on the root logger). The span-event → log bridge also emits through that same stdlib path so templog gets dual-ship without a separate OTel Logs API emit. This collapses into:
    - OTel spans and legacy events for audit (stdout + OTLP traces), with templog OTLP logs (and stderr) via the bridge → LoggingHandler.
    - Standard logging for developer debugging (stderr + OTLP when the endpoint is set).

### Structured Log Format

14. The stdout exporter MUST emit OTLP JSON — the OTel standard wire format — for the spans it exports. The span data comes from the shared TracerProvider and includes the enabled three-model-family canonical profile attributes; only the exact `gcp.vertex.agent` instrumentation scope is excluded. This is not a custom transcript format.

15. The stdout exporter MUST NOT truncate span attributes or event attributes. Full fidelity is preserved; downstream size limits remain best-effort collection constraints.

### Legacy Provider Event Projection

16. **DeepAgents / Anthropic, Vertex Gemini, and native/custom OpenAI** (`providers/deepagents.py`): Preserve existing `gen_ai.choice` text events from `AIMessage` and reasoning events from reasoning `content_blocks`; create tool spans from `AIMessage.tool_calls` and `ToolMessage` content. Vertex Gemini uses this same shared event projection. Existing aggregate usage remains on the invocation span through `ResultEvent`.

17. **[RETIRED] Legacy OpenAI Agents** (removed `providers/openai.py`; historical requirements only): Preserve buffered `gen_ai.choice` text events from stream deltas and reasoning events when present; create tool spans from existing tool call/output items. Existing aggregate usage remains on the invocation span through `ResultEvent`.

18. **Vertex Gemini route.** There is no separate Gemini provider event projection; Vertex Gemini uses the DeepAgents event mapping in rule 16.

The existing `gen_ai.choice` projection and its gates remain unchanged. DeepAgents captures canonical main-agent generations for all implemented routes through its model callbacks; spans under the exact `gcp.vertex.agent` scope are omitted only from stdout and OTLP trace exports. Developer logs and templog projections are not changed.

### Tool-Result Inspection [PLANNED: OLS-3928]

18a. Sandbox telemetry MUST conform to `openshift/ols/.ai/spec/what/tool-result-inspection.md`.

18b. Each DeepAgents inspection MUST create the contract's `tool_result.inspection` span and attach it to the parent agent trace.

18c. The sandbox can add controlled tool, provider, model, and tool-call correlation identifiers to the contract-defined attributes. Each inspection span MUST include `gen_ai.tool.call.id` when the inspected `ToolMessage` provides a non-empty call ID. This value MUST match the correlated `execute_tool` span.

18d. Existing generic inference instrumentation can observe classifier calls. The sandbox MUST add no feature-specific Prometheus metric.

18e. Developer logs and `tool_result.inspection` telemetry MUST NOT contain tool arguments, tool results, or tool-generated errors. Rejected classifier replies emit a warning per attempt, even if an existing retry later succeeds. `inspection.response_issue` uses fixed codes (`output_limit`, `incomplete_response`, `provider_response_error`, `refusal`, `invalid_envelope`, `schema_mismatch`, `non_json`, `missing_text`), never raw provider messages/statuses. Attempt warnings carry `inspection.attempt_count`, `inspection.chunk_index`, and `inspection.chunk_count`; terminal warnings retain the final response issue and attempt count. The issue and attempt count also appear in the warning message for stderr-only monitoring. Inspection spans retain the reserved `inspection.output_token_limit`; existing outcome/failure attributes and retry/deadline policy remain unchanged. These are diagnostics, not an additional feature-specific metric.

18f. After inspection passes, `AuditLogger` MUST retain the complete normalized result for the approved `gen_ai.tool.call.result` span attribute.

18g. The canonical tool-result span attribute is recorded whenever its span is recording, independent of audit/content flags. Existing developer-log redaction and legacy choice-event/log gates remain unchanged.

18h. If inspection fails, `AuditLogger` MUST receive no approved result payload; the rejected result MUST NOT appear in a canonical tool-result attribute.

18i. The DeepAgents adapter MUST hold normalized `ToolResultEvent` records until the model-boundary middleware accepts the associated results. If inspection rejects any result, the adapter MUST release no pending result events from that boundary. Rejected output and pending sibling output MUST NOT enter canonical trace attributes or legacy audit payloads.

### Metrics

19. The sandbox MUST record the following `gen_ai.*` Prometheus histograms during agent execution (`metrics.py`). Histograms are **in-process only** (`prometheus_client`); the batch entrypoint MUST NOT expose a `/metrics` HTTP scrape endpoint and MUST NOT export histograms to OTLP or Pushgateway at shutdown. Short-lived one-shot pods are a poor fit for pull-based Prometheus scraping; the aggregate `gen_ai.usage.*` values on the invocation span remain a separate OTLP trace usage signal. Child-generation usage is available on provider-generation spans and MUST NOT be summed with the invocation aggregate. Unit tests (`tests/test_metrics.py`) verify histogram recording.

| Metric | Type | Unit | Labels |
|---|---|---|---|
| `gen_ai_client_token_usage` | Histogram | `{token}` | `gen_ai_token_type`, `gen_ai_request_model`, `gen_ai_provider_name`, `gen_ai_operation_name` |
| `gen_ai_client_operation_duration_seconds` | Histogram | `s` | `gen_ai_request_model`, `gen_ai_provider_name`, `gen_ai_operation_name` |
| `gen_ai_execute_tool_duration_seconds` | Histogram | `s` | `gen_ai_tool_name` |

20. Token usage histogram bucket boundaries MUST be `[1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]` (per semconv recommendation). Root reasoning usage is recorded as `gen_ai.usage.reasoning.output_tokens` only when the existing aggregate value is nonzero; it is not a `gen_ai.token.type` value.

### Configuration

21. The sandbox receives audit and shared tracing configuration through `LIGHTSPEED_AUDIT_ENABLED`, `LIGHTSPEED_CAPTURE_CONTENT`, and `OTEL_EXPORTER_OTLP_ENDPOINT`; run correlation uses `LIGHTSPEED_AGENTICRUN_UID` and `LIGHTSPEED_AGENTICRUN_STEP`. Audit is enabled only when `LIGHTSPEED_AUDIT_ENABLED` is `"true"` after strip and lowercasing. Audit-disabled suppresses stdout and span-event log copies, but MUST NOT suppress tracing when the shared OTLP endpoint is configured. Canonical profile attributes bypass audit/content payload gates only on spans the existing runtime records.

22. When `OTEL_EXPORTER_OTLP_ENDPOINT` is configured, the sandbox MUST configure OTLP exporters for traces and logs targeting that same endpoint. Trace export is active whenever the endpoint is set. The span-event → log processor is attached only when the endpoint is set and audit is enabled; the stdout span exporter emits when audit is enabled. When the endpoint is absent, no OTLP exporters or span-event log forwarding are configured.

### OTLP Log Emission (Templog) [OLS-3515]

23. When `OTEL_EXPORTER_OTLP_ENDPOINT` is set and audit is enabled, the sandbox MUST emit the compliance view of audit span events as OTLP log records to that endpoint, in addition to stdout and OTLP trace export.

24. Each forwarded span-event OTLP log record MUST carry log record attributes `agenticrun.uid` and `agenticrun.phase` (from `LIGHTSPEED_AGENTICRUN_UID` and `LIGHTSPEED_AGENTICRUN_STEP` when set), plus `event` (the span event name). These MUST be stamped via stdlib `logging` `extra` so `LoggingHandler` preserves them. The span event attributes are the JSON log-record body; when content capture is disabled, `gen_ai.choice` copies MAY have an empty body. TraceID MUST come from the ended span's context. TracerProvider and LoggerProvider share one Resource with pinned `service.name`; run UID and phase remain record/span attributes. When audit and the OTLP endpoint are enabled but either correlation value cannot be resolved, the sandbox MUST log a startup warning without failing startup. The span-event → log processor MUST NOT forward automatic OTel `exception` events; other intentional span events remain eligible.

25. When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, stdlib Python logging MUST be exported as OTLP logs via `LoggingHandler` (dual-ship with stderr). Templog audit records use that same path: the span-event processor logs through stdlib, not a separate OTel Logs API emit.

26. When `OTEL_EXPORTER_OTLP_ENDPOINT` is absent, no OTLP log records are emitted. Graceful degradation.

### Agentic Trace Profile

27. The audit/tracing layer MUST emit the invocation, tool, and three-model-family generation attributes defined in `data-collection.md` through the existing shared trace runtime. Existing `gen_ai.choice` events and their log projections remain legacy outputs with their old gates and ordering. The stdout and OTLP trace exporters exclude only spans with the exact `gcp.vertex.agent` instrumentation-scope name; native span processing and log behavior/gates remain unchanged, and other scopes are unaffected. The sandbox producer exception does not change the parent collection contract outside this scope.

## Verification

- Native OpenAI offline integration: [test_openai_models.py](../../../tests/test_openai_models.py) exercises a real DeepAgents/ChatOpenAI file-tool/shaping loop with scripted HTTP/SSE. It checks all three response IDs and individual generation usage, plus terminal input/output totals including shaping exactly once. This does not verify live exports or satisfy the parent telemetry compatibility/owner-review gate.
- Exported regressions: [test_run_agent.py](../../../tests/test_run_agent.py), [test_audit.py](../../../tests/test_audit.py), [test_tracing.py](../../../tests/test_tracing.py), and [test_deepagents_generation_spans.py](../../../tests/test_deepagents_generation_spans.py) cover invocation/tool spans, shared DeepAgents capture, and retained audit/log projections. [test_gemini_models.py](../../../tests/test_gemini_models.py) covers offline Vertex Gemini construction. The Vertex/Google case in `test_deepagents_generation_spans.py` covers shared DeepAgents capture with a fake model and a directly constructed route.
- DeepAgents/OpenAI offline trace smoke proof and detailed producer/wire scope: [data-collection.md Verification](data-collection.md#verification). It exercised in-memory exports, all four audit/content-gate combinations, OTLP protobuf reconstruction, failure/cancellation, and inspection rejection; no live provider API or deployed collector/FileExporter/Dataverse path was exercised.
- Historical pre-migration ADK 2.11 exporter smoke facts and limits: [data-collection.md Verification](data-collection.md#verification); these do not describe the current Vertex Gemini route.
- Cancellation boundary/OTLP smoke: root ERROR/`CancelledError` and pending-tool ERROR/`missing_tool_result` survived the wire with no tool result, duration-histogram observation, or cancellation-triggered choice/log flush.
- Existing log/metric regression checks: [test_run_agent.py](../../../tests/test_run_agent.py) verifies cancellation does not flush legacy event buffers or log cancellation details; [test_tracing.py](../../../tests/test_tracing.py) verifies LoggingHandler/span-event log forwarding and correlation gates; [test_metrics.py](../../../tests/test_metrics.py) verifies in-process histogram recording.
- Existing live batch coverage: [sandbox_e2e.feature](../../../tests/e2e/features/sandbox_e2e.feature) checks trace and bridged audit-log export; it is not deployed FileExporter/Dataverse proof for this profile.

### MCP Semantic Conventions [UNTRACKED]

28. MCP tool connectivity is implemented. Additional MCP span attributes (`mcp.method.name`, `mcp.session.id`, `mcp.protocol.version`, `network.transport`) are not implemented and have no Jira story. Do not treat this table as a current MUST until a ticket exists. Prefer `gen_ai.tool.*` on tool spans today.

## Cross-References

- `run-api.md` — batch tracing lifecycle and invocation span
- `provider-contract.md` — shared provider-event behavior, optional trace-only tool metadata, and route-based generation capture
- Parent workspace `ols/.ai/spec/what/templog.md` — temporary audit log storage; sandbox emission tracked by OLS-3515
- `ols/.ai/spec/what/audit-logging.md` — parent cross-repository audit/logging and correlation requirements; sandbox producer exception is scoped in `data-collection.md`
- `data-collection.md` — named three-model-family profile and its parent-contract scope
- `ols/.ai/spec/what/agentic-data-collection.md` — parent collection contract, unchanged outside the sandbox producer exception
- [Pinned OTel GenAI profile](https://github.com/open-telemetry/semantic-conventions-genai/tree/4f85037ef86e92c510d2ef881a58f1076f6fc0e4/docs/gen-ai) — Development snapshot; named alignment profile
- [OTel MCP Semantic Conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/mcp.md)
- Parent workspace `ols/.ai/spec/what/tool-result-inspection.md` — cross-repository tool-result inspection contract
