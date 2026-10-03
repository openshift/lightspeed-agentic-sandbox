# Agentic Data Collection

Sandbox-owned boundary for the source spans used by cross-repository Agentic product collection. The parent contract is `ols/.ai/spec/what/agentic-data-collection.md`; it owns product eligibility, collection state, and downstream interpretation. This file defines only the sandbox producer behavior.

## Producer Boundary

1. The sandbox MUST represent its agent invocation, actual model requests, and actual tool executions with the standard v1.41.0 spans described in `audit-logging.md`, and export them through the existing shared `OTEL_EXPORTER_OTLP_ENDPOINT`. It MUST NOT add a product-specific endpoint or handoff key.

2. The sandbox MUST NOT write product-data files, receive or evaluate downstream collection state, or assemble product Actions/Transcripts. Its producer responsibility ends after exporting correlated source spans through the normal OTLP trace path.

3. When a source span is recording, `invoke_agent lightspeed` and every non-classifier inference span, including main-agent, shape, and subagent requests, MUST retain full available input, output, instruction, and tool-definition content in standard span attributes, independent of `LIGHTSPEED_CAPTURE_CONTENT`. The agent span MUST record the exact post-shaped `AgentResult.output` whenever a terminal result is normally produced, including domain `success=false`; pre-terminal failures have no fabricated output. Each successful native tool span MUST retain available arguments and the complete raw callback result at actual SDK completion, independently of inspection; only native execution failures omit the success-only result. Do not redact or truncate captured content beyond upstream SDK/adapter limits. Adapters pass structured Python values to `AuditLogger`, which JSON-serializes them only when the span is recording. Missing model/usage observations remain absent; an observed zero is valid.
Standalone safety-classifier inference MUST retain timing, endpoint provider, requested/observed model, observed usage, and error status/type but MUST omit input messages, output messages, and system instructions even when recording, as required by the parent tool-result-inspection contract. Valid benign or malicious `tool_result.inspection` outcomes are `UNSET`; `classifier_error`, including cancellation, is `ERROR` with controlled metadata. A fail-closed rejection or classifier failure leaves the enclosing invocation `ERROR` with no terminal output and MUST keep rejected content out of model context, application result events, Result CRs, and termination messages. Inference span inputs reflect the available content actually visible to the provider; SAFE-02 wrappers appear there only if and when the separately planned OLS-3929 behavior is implemented. This telemetry contract does not implement or imply SAFE-02, and tool spans retain the complete raw native result, not a model-facing wrapper.

4. GenAI input, output, instructions, tools, and content belong in the v1.41.0 span attributes described by `audit-logging.md`. The sandbox MUST NOT emit parallel `gen_ai.choice` or other GenAI content span events, define a product-specific input/output event catalog, or derive a second telemetry source from normalized `ProviderEvent` messages.

5. Compliance stdout and templog content are separate filtered projections governed by `audit-logging.md`. Compliance filters MUST NOT mutate the source spans used by the existing trace endpoint.

## Correlation

6. When supplied through the existing invocation configuration, `LIGHTSPEED_AGENTICRUN_UID` and `LIGHTSPEED_AGENTICRUN_STEP` MUST be translated to the literal `agenticrun.uid` and `agenticrun.phase` span attributes on agent, inference, tool, and `tool_result.inspection` spans. Inspection spans MUST copy only these supplied values; missing values remain absent and MUST NOT be filled from a Resource, re-read process environment, parent context, or another span. Product eligibility requires both values on each span; Resource attributes are not a fallback.

7. The sandbox MUST use the incoming W3C `TRACEPARENT` as the parent of `invoke_agent lightspeed` when valid. Each actual model request and tool execution is parented to that agent span; missing or invalid trace context creates a new trace.
8. The sandbox MUST reuse provider/SDK-assigned tool-call IDs unchanged only when exposed. Gemini observes finalized ADK Events to capture late SDK-assigned IDs on model output and actual tool spans; IDs that ADK strips from later effective requests remain absent, and supplied provider IDs remain unchanged. Missing IDs MUST NOT be synthesized; exact joins require a common exposed ID.

## Cross-References

- Parent product contract: `ols/.ai/spec/what/agentic-data-collection.md`
- `audit-logging.md` — v1.41.0 agent/model/tool spans, content attributes, and compliance projections
- `provider-contract.md` — provider SDK lifecycle instrumentation
- `run-api.md` — effective input construction and batch trace lifecycle
- [Official OpenTelemetry GenAI Semantic Conventions v1.41.0](https://github.com/open-telemetry/semantic-conventions/tree/v1.41.0/docs/gen-ai)
