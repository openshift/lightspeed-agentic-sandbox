# Agentic Data Collection

Sandbox-owned contract for producing literal agent-interaction events as OTLP trace span events. The parent cross-repository contract is `ols/.ai/spec/what/agentic-data-collection.md`; it is authoritative for canonical correlation, event semantics, and all downstream collection behavior. This file defines only `lightspeed-agentic-sandbox` runtime and provider-adapter behavior. All behavior in this file is planned under OLS-3569.

## Producer Boundary

1. [PLANNED: OLS-3569] When the trace runtime described by `run-api.md` is active, the sandbox MUST emit the content events in this specification as OTLP trace span events. Trace export MUST use the existing shared `OTEL_EXPORTER_OTLP_ENDPOINT`; the sandbox MUST NOT introduce a product-specific endpoint or handoff key.
2. [PLANNED: OLS-3569] The sandbox MUST NOT write product-data files or receive or evaluate downstream collection state. Its collection responsibility ends after emitting correlated traces through the ordinary OTLP runtime.
3. [PLANNED: OLS-3569] The sandbox MUST emit these content atoms through the tracing API only and MUST NOT add an OTel Logs API emission path for them. The existing compliance audit bridge remains governed by `audit-logging.md`.
4. [PLANNED: OLS-3569] Content events MUST retain complete literal values. The sandbox and provider adapters MUST NOT redact or truncate PII, secrets, prompts, context, tool payloads, skill content, reasoning, or final responses. Existing developer-log truncation MAY remain because developer logs are not these trace events.

## Content Event Contract

5. [PLANNED: OLS-3569] The sandbox MUST implement the exact provider-neutral event names, required attributes, literal meanings, and optional-value rules in the parent contract's Transcript candidate interface. This repository owns how its runtime and provider adapters obtain and emit those values; it MUST NOT define a second event schema.

6. [PLANNED: OLS-3569] `gen_ai.input` MUST be emitted once on the inference span immediately before provider invocation. Completion, reasoning, tool, and skill events MUST be emitted when their normalized provider signals are observed. `gen_ai.output` MUST be emitted once on the inference span after provider processing succeeds and before the span ends.
7. [PLANNED: OLS-3569] `gen_ai.input.prompt` MUST contain the effective prompt after `run-api.md` context-prefix formatting. `gen_ai.input.context` and `gen_ai.input.output_schema` MUST contain the complete canonical JSON source values separately; the event MUST NOT include unrelated process environment.
8. [PLANNED: OLS-3569] Tool call/result events MUST coexist with the operational `execute_tool {name}` span and use the same tool name and call ID. When the SDK omits a call ID, the adapter MUST assign one stable ID to the matching call, result, and tool span. Skill events MUST come only from explicit load/use signals available to the sandbox or provider adapter and MUST NOT be inferred from completion text or generic tool output.
9. [PLANNED: OLS-3569] `gen_ai.output` is distinct from preceding `gen_ai.choice` events: choices preserve provider event granularity, while output records the exact terminal value consumed by Result shaping. The sandbox MUST emit both without synthesizing output by concatenating choices.

## Correlation and Ordering

10. [PLANNED: OLS-3569] Every product-eligible sandbox inference or tool span MUST carry non-empty literal `agenticrun.uid` and valid `agenticrun.phase` as span attributes. The sandbox receives those values through the existing batch correlation configuration (`LIGHTSPEED_AGENTICRUN_UID` and `LIGHTSPEED_AGENTICRUN_STEP`) and MUST NOT use resource attributes as a fallback or invent missing values.
11. [PLANNED: OLS-3569] Every content event MUST be attached to the run's `chat {gen_ai.request.model}` inference span, which also carries `gen_ai.provider.name` and `gen_ai.request.model`. Events MUST be added sequentially in normalized provider observation order and retain their native OTEL event timestamps.
12. [PLANNED: OLS-3569] Each literal content atom MUST be emitted exactly once. Provider delta buffering MAY combine contiguous deltas only at the existing semantic flush boundary and MUST preserve their content and position relative to intervening completion, reasoning, tool, skill, and result signals.

## Provider Normalization

13. [PLANNED: OLS-3569] DeepAgents, Gemini, and OpenAI MUST produce the same event names and required values. Adapter-specific normalization, terminal-value handling, model/token fallbacks, tool correlation, and explicit skill-signal rules are defined by `provider-contract.md` rules 39–46; provider SDK object shapes MUST stop at the adapter boundary.

## Cross-References

- Parent contract: `ols/.ai/spec/what/agentic-data-collection.md`
- Accepted architecture decision: `ols/.ai/spec/decisions/0042-agentic-data-collection-via-otel.md`
- `audit-logging.md` — inference/tool spans and the shared OTLP trace runtime
- `provider-contract.md` — provider-specific normalization and fallbacks
- `run-api.md` — effective input construction and batch trace lifecycle
