# Agentic Data Collection

Sandbox producer contract for the named three-model-family GenAI trace profile, aligned to the [pinned OTel GenAI conventions](https://github.com/open-telemetry/semantic-conventions-genai/tree/4f85037ef86e92c510d2ef881a58f1076f6fc0e4/docs/gen-ai). The conventions were Development at that revision; this named profile does not claim every optional convention or add a semantic-conventions dependency.

For the sandbox producer only, this profile supersedes conflicting candidate/event producer requirements in `ols/.ai/spec/what/agentic-data-collection.md`. The parent contract and downstream collection requirements remain unchanged. Invocation/tool spans, DeepAgents generation capture for Anthropic, Vertex/Google, and native/custom OpenAI, and exact-scope `gcp.vertex.agent` exclusion at stdout and OTLP trace exporters are implemented.

## Harness Consolidation [PARTIALLY IMPLEMENTED: OLS-4434]

Anthropic, Vertex/Google, native/custom OpenAI, and Azure routes use shared DeepAgents/LangChain generation capture; the legacy OpenAI adapter and its hooks are removed. Vertex/OpenAI is unimplemented pending OLS-4472. Azure routing/workflows have offline evidence, not live auth or deployed telemetry compatibility/owner-review acceptance. The Gemini ADK adapter and its event-capture path are removed. For the migrated routes, [provider-contract.md rules 57 and 60–62](provider-contract.md#deepagents-consolidation-partially-implemented-ols-4434) govern current behavior; Agents SDK-specific requirements below (`SandboxAgent`, `ModelResponse`, `requests > 0`, hooks, and `openai.api.type`) document the retired harness only and no longer apply to runtime routes. Native/custom OpenAI and Azure use the shared DeepAgents callback/result path with provider `openai`. Preserve the data contract's observed content, usage, actual IDs/provider identity, ancestry, and failure semantics, including the parent contract's independent native tool-source evidence policy. No source-result suppression may be inferred solely from a rejected normalized event.

The change from Vertex Gemini `generate_content` spans/ADK parts to DeepAgents `chat` spans/LangChain messages, and from native/custom OpenAI Agents hooks/`ModelResponse` to shared DeepAgents callbacks, can affect downstream interpretation. The workspace collection contract's rules 41–42 require compatibility classification, an explicit schema/data-contract version decision, component/Dataverse-owner approval, and downstream readiness before a breaking cutover. This spec does not choose a new version, change the pinned conventions, waive that review, or claim that pre-migration three-harness smoke evidence verifies the current route. Existing capture/export controls and collection topology are unchanged.

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

DeepAgents captures accepted main-agent model generations on standard `SpanKind.CLIENT` child spans of `invoke_agent`. DeepAgents uses `chat {gen_ai.request.model}` with `gen_ai.operation.name="chat"` for all its migrated routes. Provider names come from the resolved DeepAgents route (`anthropic`, `aws.bedrock`, `gcp.vertex_ai`, or native `openai`); OpenAI uses `openai`. `start_generation_span` uses the invocation context captured before the provider query, copies available `agenticrun.uid` and `agenticrun.phase`, and does not make the generation span current.

The DeepAgents callback selects main-agent requests and excludes named nested agents, `nostream` classifier calls, and summarization calls. It also captures the separate structured-output shaping generation from its raw `AIMessage`, not the parsed JSON result. OpenAI hooks select only the main `SandboxAgent` and record its completed `ModelResponse`; `openai.api.type` follows the existing legacy Vertex/OpenAI `uses_responses_api` decision. Azure's version-selected API instead uses the same shared LangChain capture as other DeepAgents routes.

Vertex Gemini uses the same DeepAgents LangChain callback as Anthropic. It captures main-agent `on_chat_model_start`/`on_llm_end` results using `LLMResult` and `AIMessage` metadata; it does not register ADK callbacks, run an ADK `Runner`, or depend on ADK-finalized call IDs. The generation operation is `chat` and its provider identity comes from the route (`gcp.vertex_ai`).

The shared DeepAgents callback closes each accepted main-model generation when LangChain reports completion, records only SDK-exposed result data, and closes open spans with error status on failure or interruption. It does not buffer/reconstruct provider deltas or replace model responses.

Generation spans contain SDK-observed output, not `gen_ai.input.messages` or `gen_ai.system_instructions`; SDK input histories are not repeated.

### Ordered Output, Metadata, and Usage

`gen_ai.output.messages` is compact JSON and preserves the observed message, item, content, and part order:

- DeepAgents maps observed `LLMResult.generations` choices using `message.content_blocks`; plain string content becomes text when no blocks exist. Complete `tool_call` blocks retain observed ID, name, and arguments. Partial `tool_call_chunk` blocks remain `GenericPart` values with their native discriminator and observed fields; they are not promoted or reassembled.
- OpenAI maps `ModelResponse.output` and nested content arrays in source order. Text/refusal remain literal, and reasoning content and summary text become reasoning parts. Function calls use actual `call_id`/name and decode arguments only when they are valid finite JSON; malformed or nonfinite argument strings remain literal. Custom calls use the actual `call_id`/name and literal `.input`.
- Vertex Gemini uses the shared DeepAgents/LangChain mapping: `LLMResult.generations` and `AIMessage.content_blocks` are recorded in observed order. Text and reasoning blocks retain their observed values; complete LangChain tool calls use the observed IDs, names, and arguments. Tool results follow the common `ToolMessage` path, not ADK `function_response` events.
- Hosted `tool_call` and `tool_response` parts stay inside generation messages as `server_tool_call` and `server_tool_call_response`, with nested `server_tool_call={"type": tool_type.value, "args": args}` and `server_tool_call_response={"type": tool_type.value, "response": response}` respectively. Do not create local `execute_tool` spans or durations for hosted tools. If the SDK omits the tool type, preserve the raw JSON fields in an upstream `GenericPart` with the source discriminator `tool_call` or `tool_response`; do not invent a server-tool name.
- Set `gen_ai.response.model`, `gen_ai.response.id`, and `gen_ai.response.finish_reasons` only from actual provider evidence. DeepAgents may use message/generation metadata or observed `LLMResult.llm_output` for all its migrated routes; the legacy OpenAI adapter uses actual response IDs and does not substitute requested models or transport/item IDs. Do not interpret ADK `Event.id` as a response ID; the current Gemini route does not consume ADK events.
- DeepAgents preserves present input/output/reasoning usage counts, including explicit zero, for all its migrated routes and omits absent counts. The legacy OpenAI adapter records usage only when `requests > 0`, preserving present zero input/output counts and omitting missing counts; reasoning output tokens are emitted only for supplied nonzero reasoning detail. Never synthesize zero from missing evidence, and do not sum generation usage with root aggregate usage.

On DeepAgents failure, retain partial output/metadata/usage only when the callback exposes an `LLMResult`; this includes Vertex Gemini. OpenAI records output only from a completed response callback and does not recover stream deltas. Observed provider failures close the generation with ERROR; exceptions and cancellation propagate unchanged after span closure, and early closure without an exception uses `generation_interrupted`. Do not synthesize output or add delta-recovery buffers.

Vertex Gemini is captured through the same DeepAgents/LangChain message-stream and callback boundary as Anthropic. Output order and metadata reflect values exposed by those LangChain results; the runtime does not reconstruct provider deltas or claim token chronology.

### Google Vertex SDK Span Export Exclusion

The stdout and OTLP trace exporters MUST exclude spans whose instrumentation-scope name is exactly `gcp.vertex.agent`; exclusion drops those spans from these exports without renaming, projecting, or normalizing them. Span processors and all other instrumentation scopes are unaffected by this exporter filter and follow existing gates. The scope name is retained for compatibility; the active Vertex Gemini model route does not use the ADK provider adapter.

Canonical parenting is independent of spans under that vendor scope: `invoke_agent` is a child of the received operator context, and each accepted provider-generation and local `execute_tool` span is a direct child of `invoke_agent`. Exporter filtering cannot orphan or reparent them.

The exclusion MUST NOT affect span-event-to-log processing or stdlib `LoggingHandler`; existing log correlation, event payloads, and gates remain unchanged.

## Reconstruction and Ordering

1. Select the `invoke_agent` trace subtree using native trace/span IDs and parent relationships; retain available `agenticrun.uid`/phase and native span IDs.
2. Read invocation system instructions and the initial user message once. Expose the exact terminal root output separately, not as another model generation.
3. Read DeepAgents (including Vertex Gemini) and OpenAI `chat` outputs in native span start-time order. Preserve each response's observed message/part order, including the raw DeepAgents shaping generation. Do not append root output or reconstruct repeated input histories.
4. Join complete local tool-call parts to `execute_tool` spans by trace ID and actual call ID. A partial `tool_call_chunk` is not a complete call. Retain concurrent tool intervals; hosted tool calls/results remain in generation parts.
5. Use per-generation usage for model analysis and existing invocation usage separately; never sum the root aggregate and child usage. Ignore legacy `gen_ai.choice` events for canonical reconstruction.
6. Export-batch and file order are not execution order. Do not infer token chronology or a causal total order for concurrent tools.
7. Missing spans or files are missing evidence, not reconstructed successes.

## Verification

- Native OpenAI integration: [test_openai_models.py](../../../tests/test_openai_models.py) uses the real native constructor, DeepAgents loop, and filesystem backend with scripted HTTP/SSE. It verifies per-generation response IDs and input/output usage, and exact terminal totals with shaping counted once. It is offline producer evidence, not live API/export, collector/Dataverse delivery, or compatibility-review evidence.
- Current regressions: [test_tracing.py](../../../tests/test_tracing.py) covers exact-scope `gcp.vertex.agent` exclusion at stdout and OTLP, span processing/log correlation, and other-scope passthrough. [test_run_agent.py](../../../tests/test_run_agent.py) and [test_audit.py](../../../tests/test_audit.py) cover invocation/tool spans and legacy log/event gates; [test_deepagents_generation_spans.py](../../../tests/test_deepagents_generation_spans.py) covers shared DeepAgents generation capture. [test_gemini_models.py](../../../tests/test_gemini_models.py) covers offline Vertex Gemini construction. The Vertex/Google case in the generation-span test uses a fake model and a directly constructed route; it does not exercise route resolution or `create_vertex_gemini_model`. No Gemini ADK telemetry or skills test modules remain.
- Historical pre-retirement DeepAgents/OpenAI trace smoke (not current-route acceptance) passed: it exercised the actual `run_agent_query()` through a DeepAgents fake-model graph (including nested-agent exclusion and raw shaping), the OpenAI Agents `Runner` with a scripted `Model`, and `init_tracer()` with in-memory trace/log exporters. All four audit/content-flag combinations retained the prior developer/templog and choice-event behavior. The smoke exercised protobuf encode/decode and shuffled-span reconstruction, inspection rejection, failure, and cancellation. The wire decoded 51 spans (33,029 bytes); five pinned schemas validated 74 values.
- DeepAgents/OpenAI smoke scope: producer/wire proof only. It did not contact live model APIs or exercise a deployed collector, FileExporter pipeline, or Dataverse delivery. It did not exercise the DeepAgents `LLMResult.llm_output` metadata fallback or partial `tool_call_chunk` GenericPart mapping; those mappings preserve only SDK-observed values/fields. Collection remains best-effort and producer-side truncation is not added.
- Historical, pre-migration ADK 2.11 Runner/export-view smoke evidence (superseded implementation; not current route behavior): the actual Runner was exercised through `run_agent_query()` with a scripted `BaseLlm`, a real local Bash echo tool, and scripted hosted-tool parts. Batch and default progressive SSE passed; four audit/content combinations retained developer/templog replay and native ADK log/span-ID correlation. Checks included SDK local-tool call IDs, protobuf encode/decode, shuffled-span reconstruction, stdout/OTLP projection equality, and failure-before-output.
- Historical evidence from that prior run: five schemas at pinned revision `4f85037` validated; focused tests passed (50, then 40 in the corrective run); its final suite was 727 tests with 17 warnings, and `make lint` passed. These results predate canonical-only filtering and do not verify current exporter behavior.
- Historical scope limit: hosted parts were scripted, not live; that smoke did not exercise cancellation, partial failure, or oversized trimming and did not test live provider, collector, rotation, upload, Dataverse, SQL, or deployed end-to-end data delivery.
- Historical pre-migration ADK 2.11 canonical-only exporter smoke passed with a scripted `BaseLlm`, the ADK `Runner` and tools, and an in-memory exporter. It exported eight canonical spans (one `invoke_agent`, four generations, three tools); 13 native `gcp.vertex.agent` spans were processed but none exported. Stdout and OTLP exports had equal trace/span IDs, and all seven canonical child spans had `invoke_agent` as parent. Gemini `RESOURCE_NOT_FOUND` and `INVALID_RESOURCE_PATH` function responses produced canonical tool spans with ERROR status; the corrected Bash invocation and overall run succeeded, and the legacy reasoning choice log remained. This is scripted-model/in-memory-exporter proof, not live-provider API or deployed-collector proof.

## Cross-References

- Parent contract (unchanged outside the sandbox producer exception): `ols/.ai/spec/what/agentic-data-collection.md`
- `audit-logging.md` — shared invocation/tool/generation spans and legacy logging projections
- `provider-contract.md` — shared provider-event behavior and route-based generation capture
- `run-api.md` — effective input construction and trace lifecycle
