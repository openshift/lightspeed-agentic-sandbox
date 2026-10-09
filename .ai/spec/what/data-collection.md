# Agentic Data Collection

Sandbox producer contract for the named three-provider GenAI trace profile, aligned to the [pinned OTel GenAI conventions](https://github.com/open-telemetry/semantic-conventions-genai/tree/4f85037ef86e92c510d2ef881a58f1076f6fc0e4/docs/gen-ai). The conventions were Development at that revision; this named profile does not claim every optional convention or add a semantic-conventions dependency.

For the sandbox producer only, this profile supersedes conflicting candidate/event producer requirements in `ols/.ai/spec/what/agentic-data-collection.md`. The parent contract and downstream collection requirements remain unchanged. Invocation/tool spans, DeepAgents/OpenAI generation capture, Gemini generation capture, and exact-scope `gcp.vertex.agent` exclusion at stdout and OTLP trace exporters are implemented.

## Producer Boundary

1. When the existing trace runtime records spans, canonical profile content is stored in span attributes. Do not add a custom transcript attribute/event or skill events, and do not infer skill use from text or generic tool output.
2. Full profile attributes are recorded on applicable recording spans independently of audit/content flags. Existing runtime and exporter activation remain as defined in `run-api.md` and `audit-logging.md`; this profile does not activate a previously disabled trace path or change developer logs, templog, or legacy choice events.
3. Preserve existing safety inspection, execution limits, and prior result redaction. Capture only values exposed by the existing normalized execution path; do not recover pre-redaction output or bypass inspection.
4. Export only through the existing OTLP trace runtime. FileExporter, rotation/retention, upload, and Dataverse behavior are downstream and outside the sandbox runtime. Delivery is best-effort; FileExporter may reject an over-limit batch. The producer adds no truncation to fit that limit and does not write product-data files or read collection state.

## Invocation and Tool Spans

Structured message and tool values use compact JSON strings with `ensure_ascii=False` and separators `(",", ":")`, matching the schemas at the pinned upstream revision. Escape unpaired surrogate code points for UTF-8-safe OTLP export while preserving ordinary Unicode and JSON-decoded content.

### Invocation Span

- The invocation span is `invoke_agent`, `SpanKind.INTERNAL`, with `gen_ai.operation.name="invoke_agent"`. It is a child of the received operator context and retains available `agenticrun.uid` and `agenticrun.phase` values; missing correlation is not invented.
- `gen_ai.request.model` is the configured request model. The invocation span does not claim `gen_ai.provider.name` or `gen_ai.response.model`.
- `gen_ai.input.messages` is `[{"role":"user","parts":[{"type":"text","content":prompt}]}]`, where `prompt` is the effective post-context-prefix input, including an observed empty string.
- `gen_ai.system_instructions` is `[{"type":"text","content":system_prompt}]`, using the configured supplied string, including an empty string.
- Set `gen_ai.output.type="json"` iff `output_schema is not None`.
- When a `ResultEvent` is observed, set `gen_ai.output.messages` to `[{"role":"assistant","parts":[{"type":"text","content":event.text}]}]` before parsing or result shaping. Preserve an observed empty string; omit the attribute when no `ResultEvent` arrives. Do not concatenate choice events.
- Keep existing aggregate `ResultEvent` usage on this span. Record `gen_ai.usage.reasoning.output_tokens` only for a nonzero aggregate reasoning count, as today. This root usage remains a separate aggregate signal and MUST NOT be summed with child-generation usage.
- Set ERROR status and `error.type` to `timeout`, the exception class name, or `empty_response` for operational invocation failures. Cancellation is propagated unchanged after setting root ERROR and `error.type="CancelledError"`. An unsuccessful analysis/execution outcome in a normal response does not set ERROR or `error.type`; preserve it in the exact terminal output and Result payload, independently of span status.

### Tool Spans

- Each local tool execution uses an `execute_tool {name}` `SpanKind.INTERNAL` child span with `gen_ai.operation.name="execute_tool"` and `gen_ai.tool.name`.
- Set `gen_ai.tool.call.id` only from an actual SDK call ID; never publish a fabricated ID. Set `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` only when their respective values were observed.
- Normalize string payloads with strict JSON parsing: accept standards-compliant finite JSON only; treat `NaN`, `Infinity`, `-Infinity`, exponent overflow, and parser/encoder-limit failures as non-JSON. Pass decoded dictionaries through unchanged and wrap other successfully decoded values as `{"content": value}`.
- On any parse, encode, or limit rejection, preserve the complete original raw string in `{"content": raw_string}`. This is sandbox normalization, not a provider-native field; do not truncate it or raise a telemetry-only error into the provider path. Preserve observed empty values; absent arguments/results remain absent.
- A result without an ID may match only when exactly one call is pending. Do not attach an ambiguous result to an arbitrary call. Unresolved calls end with ERROR and `error.type="missing_tool_result"`; do not invent a result. On cancellation, pending tool spans close ERROR with that type and no result or tool-duration histogram observation.

## Provider Generation Spans

DeepAgents, OpenAI, and Gemini capture accepted main-agent model generations on standard `SpanKind.CLIENT` child spans of `invoke_agent`. DeepAgents and OpenAI use `chat {gen_ai.request.model}` with `gen_ai.operation.name="chat"`; Gemini uses `generate_content {gen_ai.request.model}` with `gen_ai.operation.name="generate_content"`. Provider names are `anthropic`, `aws.bedrock`, or `gcp.vertex_ai` for DeepAgents, `openai` for OpenAI, and `gcp.vertex_ai` or `gcp.gemini` for Gemini according to the existing Vertex selection. `start_generation_span` uses the invocation context captured before the provider query, copies available `agenticrun.uid` and `agenticrun.phase`, and does not make the generation span current.

The DeepAgents callback selects main-agent requests and excludes named nested agents, `nostream` classifier calls, and summarization calls. It also captures the separate structured-output shaping generation from its raw `AIMessage`, not the parsed JSON result. OpenAI hooks select only the main `SandboxAgent` and record its completed `ModelResponse`; `openai.api.type` follows the existing `uses_responses_api` routing decision, including Azure API-version selection.

Gemini registers the public `before_model_callback` and `after_model_callback` on the existing main ADK `Agent` only. The before callback starts the generation span; the after callback records a completion timestamp only when `llm_response.partial` is false and does not end the span. In the existing `runner.run_async` loop, capture finalized main-agent model events (`event.author == agent.name` with model-role content, or an empty terminal model response). This Runner output carries ADK-finalized function-call IDs. End the span with the saved callback timestamp, so its end precedes local tool execution even when the final event is consumed later.

ADK can queue partial Runner events after the final model callback. Use `event.partial` to distinguish them: partial events may update the open span but do not end it; the finalized aggregate replaces the partial value and closes the span at the callback timestamp. Both callbacks return `None`; no SDK event/response is replaced and existing `ProviderEvent` yields remain unchanged.

Generation spans contain SDK-observed output, not `gen_ai.input.messages` or `gen_ai.system_instructions`; SDK input histories are not repeated.

### Ordered Output, Metadata, and Usage

`gen_ai.output.messages` is compact JSON and preserves the observed message, item, content, and part order:

- DeepAgents maps observed `LLMResult.generations` choices using `message.content_blocks`; plain string content becomes text when no blocks exist. Complete `tool_call` blocks retain observed ID, name, and arguments. Partial `tool_call_chunk` blocks remain `GenericPart` values with their native discriminator and observed fields; they are not promoted or reassembled.
- OpenAI maps `ModelResponse.output` and nested content arrays in source order. Text/refusal remain literal, and reasoning content and summary text become reasoning parts. Function calls use actual `call_id`/name and decode arguments only when they are valid finite JSON; malformed or nonfinite argument strings remain literal. Custom calls use the actual `call_id`/name and literal `.input`.
- Gemini maps ordered `Event.content.parts` text and thought parts to text and reasoning. A finalized `function_call` becomes `tool_call` with its ADK-finalized ID, name, and args. A local `function_response` uses only the shared tool-result path after `_trim_tool_response`; it is not duplicated in generation output.
- For a Gemini local `function_response`, set trace-only `ToolResultEvent.error_type` to `error_code` only when the response is a `Mapping` with a truthy `error` and a nonempty string `error_code`. A generic `error` field alone, other error-like keys, or `returncode` MUST NOT imply failure, and no synthetic `TOOL_ERROR` fallback response may be created. This classification MUST preserve the existing result payload and actual function-call ID on the shared tool-result path and MUST NOT change `EventLogger` or developer-log contents.
- Hosted `tool_call` and `tool_response` parts stay inside generation messages as `server_tool_call` and `server_tool_call_response`, with nested `server_tool_call={"type": tool_type.value, "args": args}` and `server_tool_call_response={"type": tool_type.value, "response": response}` respectively. Do not create local `execute_tool` spans or durations for hosted tools. If the SDK omits the tool type, preserve the raw JSON fields in an upstream `GenericPart` with the source discriminator `tool_call` or `tool_response`; do not invent a server-tool name.
- Set `gen_ai.response.model`, `gen_ai.response.id`, and `gen_ai.response.finish_reasons` only from actual provider evidence. DeepAgents may use message/generation metadata or observed `LLMResult.llm_output`; OpenAI uses actual response IDs and does not substitute requested models or transport/item IDs. Gemini uses actual `model_version` and `finish_reason` when present; an ADK `Event.id` is not a response ID.
- DeepAgents and Gemini preserve present input/output/reasoning usage counts, including explicit zero, and omit absent counts. OpenAI records usage only when `requests > 0`, preserving present zero input/output counts and omitting missing counts; reasoning output tokens are emitted only for supplied nonzero reasoning detail. Never synthesize zero from missing evidence, and do not sum generation usage with root aggregate usage.

On DeepAgents failure, retain partial output/metadata/usage only when the callback exposes an `LLMResult`; OpenAI records output only from a completed response callback and does not recover stream deltas. Gemini retains only output exposed by Runner events. Observed provider failures close the generation with ERROR; Gemini exceptions/cancellation propagate unchanged after closing with the exception class, explicit `error_code` is recorded as the error type, and early closure without an exception uses `generation_interrupted`. Do not synthesize output or add delta-recovery buffers.

The supported accurate Gemini boundary covers batch and ADK's default progressive SSE. If progressive SSE is explicitly disabled, the legacy SDK aggregator can split, reorder, or discard aggregates, leaving canonical generation parts or tool-call links incomplete. Do not recover deltas or mutate SDK flags. Part order is the SDK-observed order; there is no token-chronology guarantee.

### ADK Span Export Exclusion

The stdout and OTLP trace exporters MUST exclude spans whose instrumentation-scope name is exactly `gcp.vertex.agent`; exclusion drops those spans from these exports without renaming, projecting, or normalizing them. Native ADK spans and events remain in TracerProvider/span-processor processing, while all other instrumentation scopes are unaffected by this filter and follow existing exporter gates.

Canonical parenting is independent of ADK spans: `invoke_agent` is a child of the received operator context, and each accepted provider-generation and local `execute_tool` span is a direct child of `invoke_agent`. These canonical spans are not children of native ADK spans, so exporter filtering cannot orphan or reparent them.

The exclusion MUST NOT affect span-event-to-log processing or stdlib `LoggingHandler`; native ADK log correlation, event payloads, and existing log gates remain unchanged.

## Reconstruction and Ordering

1. Select the `invoke_agent` trace subtree using native trace/span IDs and parent relationships; retain available `agenticrun.uid`/phase and native span IDs.
2. Read invocation system instructions and the initial user message once. Expose the exact terminal root output separately, not as another model generation.
3. Read DeepAgents/OpenAI `chat` and Gemini `generate_content` outputs in native span start-time order. Preserve each response's observed message/part order, including the raw DeepAgents shaping generation. Do not append root output or reconstruct repeated input histories.
4. Join complete local tool-call parts to `execute_tool` spans by trace ID and actual call ID. A partial `tool_call_chunk` is not a complete call. Retain concurrent tool intervals; hosted tool calls/results remain in generation parts.
5. Use per-generation usage for model analysis and existing invocation usage separately; never sum the root aggregate and child usage. Ignore legacy `gen_ai.choice` events for canonical reconstruction.
6. Export-batch and file order are not execution order. Do not infer token chronology or a causal total order for concurrent tools.
7. Missing spans or files are missing evidence, not reconstructed successes.

## Verification

- Permanent regressions: [test_tracing.py](../../../tests/test_tracing.py) covers exact-scope `gcp.vertex.agent` exclusion at stdout and OTLP, native processing/log correlation, and other-scope passthrough; [test_gemini_telemetry.py](../../../tests/test_gemini_telemetry.py) covers Gemini generation and local function-response error classification with preserved result payload, call ID, and provider log output. [test_run_agent.py](../../../tests/test_run_agent.py) and [test_audit.py](../../../tests/test_audit.py) cover invocation/tool spans and legacy log/event gates; [test_deepagents_generation_spans.py](../../../tests/test_deepagents_generation_spans.py) and [test_openai_generation_spans.py](../../../tests/test_openai_generation_spans.py) cover the other provider-generation scenarios. Existing Gemini paths are covered by [test_gemini_skills.py](../../../tests/test_gemini_skills.py) and [test_reasoning_config.py](../../../tests/test_reasoning_config.py).
- Offline DeepAgents/OpenAI trace smoke passed: it exercised the actual `run_agent_query()` through a DeepAgents fake-model graph (including nested-agent exclusion and raw shaping), the OpenAI Agents `Runner` with a scripted `Model`, and `init_tracer()` with in-memory trace/log exporters. All four audit/content-flag combinations retained the prior developer/templog and choice-event behavior. The smoke exercised protobuf encode/decode and shuffled-span reconstruction, inspection rejection, failure, and cancellation. The wire decoded 51 spans (33,029 bytes); five pinned schemas validated 74 values.
- DeepAgents/OpenAI smoke scope: producer/wire proof only. It did not contact live model APIs or exercise a deployed collector, FileExporter pipeline, or Dataverse delivery. It did not exercise the DeepAgents `LLMResult.llm_output` metadata fallback or partial `tool_call_chunk` GenericPart mapping; those mappings preserve only SDK-observed values/fields. Collection remains best-effort and producer-side truncation is not added.
- Historical ADK 2.11 Runner/export-view smoke evidence (superseded policy; not current export behavior): the actual Runner was exercised through `run_agent_query()` with a scripted `BaseLlm`, a real local Bash echo tool, and scripted hosted-tool parts. Batch and default progressive SSE passed; four audit/content combinations retained developer/templog replay and native ADK log/span-ID correlation. Checks included SDK local-tool call IDs, protobuf encode/decode, shuffled-span reconstruction, stdout/OTLP projection equality, and failure-before-output.
- Historical evidence from that prior run: five schemas at pinned revision `4f85037` validated; focused tests passed (50, then 40 in the corrective run); its final suite was 727 tests with 17 warnings, and `make lint` passed. These results predate canonical-only filtering and do not verify current exporter behavior.
- Historical scope limit: hosted parts were scripted, not live; that smoke did not exercise cancellation, partial failure, or oversized trimming and did not test live provider, collector, rotation, upload, Dataverse, SQL, or deployed end-to-end data delivery.
- Current ADK 2.11 canonical-only exporter smoke passed with a scripted `BaseLlm`, the real ADK `Runner` and tools, and an in-memory exporter. It exported eight canonical spans (one `invoke_agent`, four generations, three tools); 13 native `gcp.vertex.agent` spans were processed but none exported. Stdout and OTLP exports had equal trace/span IDs, and all seven canonical child spans had `invoke_agent` as parent. Gemini `RESOURCE_NOT_FOUND` and `INVALID_RESOURCE_PATH` function responses produced canonical tool spans with ERROR status; the corrected Bash invocation and overall run succeeded, and the legacy reasoning choice log remained. This is scripted-model/in-memory-exporter proof, not live-provider API or deployed-collector proof.

## Cross-References

- Parent contract (unchanged outside the sandbox producer exception): `ols/.ai/spec/what/agentic-data-collection.md`
- `audit-logging.md` — shared invocation/tool/generation spans and legacy logging projections
- `provider-contract.md` — unchanged provider-event behavior and three-provider generation capture
- `run-api.md` — effective input construction and trace lifecycle
