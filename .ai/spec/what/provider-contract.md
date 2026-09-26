# Behavioral spec: provider abstraction and events

Audience: AI agents (Claude). Precision over narrative.

Cross-references: batch agent invocation → `run-api.md`. Env and build → `configuration.md`. Provider-neutral product trace events → `data-collection.md`.

## Behavioral Rules

1. **AgentProvider.** Each backend implements a `name` property and a `query` method accepting `ProviderQueryOptions` and returning an async iterator of `ProviderEvent`.

2. **Text delta (`text_delta`).** Carries incremental natural-language or assistant text chunks for logging or streaming use.

3. **Thinking delta (`thinking_delta`).** Carries incremental chain-of-thought or reasoning text. When reasoning is configured and the SDK produces reasoning output, all adapters MUST emit `thinking_delta` events. DeepAgents emits from `AIMessage.content_blocks` with `type == "reasoning"`. Gemini MUST emit from `ThinkingConfig` thought parts when `include_thoughts` is enabled. OpenAI MUST emit from reasoning items in the response stream.

4. **Content block stop (`content_block_stop`).** Signals that a content or tool block has completed; used by logging to flush buffered thinking.

5. **Tool call (`tool_call`).** Carries the tool name and a complete string representation of inputs. Provider adapters MUST NOT length-truncate this value. `EventLogger` can truncate its developer-log rendering, subject to OLS-3928 rule 6.

6. **Tool result (`tool_result`).** Carries a complete string representation of tool output. Provider adapters MUST NOT length-truncate this value. `EventLogger` can truncate its developer-log rendering, subject to OLS-3928 rule 6.

7. **Result (`result`).** Terminal event: final text payload (may be JSON or plain text depending on structured-output path), input/output token counts, reasoning token count, and response model metadata.

8. **ProviderQueryOptions — `prompt`.** Full user message after any context prefix formatting in `run_agent.py` (see `run-api.md` context rules).

9. **ProviderQueryOptions — `system_prompt`.** System or developer instruction string.

10. **ProviderQueryOptions — `model`.** Model identifier resolved before the call (see `configuration.md`).

11. **ProviderQueryOptions — `max_turns`.** Upper bound on agent/SDK iteration. [PLANNED: OLS-3743] The value comes from required `LIGHTSPEED_AGENT_MAX_TURNS`, resolved by the operator from `Agent.spec.maxTurns` with a default of 200. Adapters MUST pass it to their native limit: DeepAgents/LangGraph `recursion_limit`, Gemini ADK `max_llm_calls`, and OpenAI Agents `max_turns`. These mechanisms are not semantically identical, but all enforce an upper bound; adapters MUST NOT substitute their SDK defaults.

12. **ProviderQueryOptions — `allowed_tools`.** List of tool names the SDK may use for that invocation.

13. **ProviderQueryOptions — `cwd`.** Directory used as skill root and/or workspace for filesystem and shell tools.

14. **ProviderQueryOptions — `output_schema`.** Optional JSON-schema dict; when set, adapters map it to the SDK's native structured-output mechanism.

15. **ProviderQueryOptions — `stream`.** When true, adapters that support partial streaming should yield deltas; when false, they may batch. The batch entrypoint does not set this flag from input files.

16. **ProviderQueryOptions — `mcp_servers`.** Optional list of provider-facing projections derived from the canonical admitted MCP server list. Each entry carries only `name`, `url`, `timeout`, resolved `headers`, and admitted tool names. Authentication classification, tool metadata, and RBAC declarations remain outside the provider boundary; batch derives the separate analysis policy projection from the canonical admission result. Adapters MAY convert headers to a dict at the SDK boundary and MAY reconnect or reload SDK tool objects, but they MUST filter those objects by the admitted names without independently admitting or reclassifying tools. When non-empty, adapters MUST initialize each admitted server and wire only admitted tools into their SDK mechanism (see rules 31–33). A reload or initialization error MUST fail the provider query rather than silently omit a server. A successful DeepAgents reload MAY return fewer tools than admission discovered; it MUST still filter out unadmitted tools. When empty or absent, no MCP servers are configured.

17. **ProviderQueryOptions — `reasoning_config`.** Optional dict (JSON object). When present, adapters MUST map it to their SDK's native reasoning/thinking parameters. When absent or `None`, adapters MUST NOT set any reasoning parameters and SDK defaults apply. DeepAgents passes only the `thinking` key through to `ChatAnthropic*`. Gemini constructs `ThinkingConfig(**config)` and OpenAI constructs `Reasoning(**rc)` — extra keys are forwarded to the SDK constructors (not stripped by the adapter); invalid values fail at SDK/API invocation time.

18. **[Removed]** *(Claude adapter was removed in OLS-3500; Anthropic reasoning is now handled by the DeepAgents adapter — see rule 34.)*

19. **Reasoning — Gemini.** When `reasoning_config` is present, the Gemini adapter MUST construct a `types.ThinkingConfig(**config)` and pass it via `GenerateContentConfig.thinking_config` on the Agent. Config keys (e.g. `thinking_budget`, `thinking_level`, `include_thoughts`) are forwarded into `ThinkingConfig`; the Gemini API validates at invocation time.

20. **Reasoning — OpenAI.** When `reasoning_config` is present, the OpenAI adapter MUST construct `ModelSettings(reasoning=Reasoning(**rc), verbosity=...)` from the config keys (e.g. `effort`, `mode`, `context`, `verbosity`) and pass it to `SandboxAgent(model_settings=...)`. Config keys are forwarded into `Reasoning`; the OpenAI API validates at invocation time.

21. **Thin-adapter principle.** Providers MUST delegate tool execution, command invocation, and skill discovery to their SDKs. Adapters MUST NOT implement custom tool executors that duplicate SDK behavior except for minimal glue (e.g., auto-confirm, path layout).

22. **Structured output.** When `output_schema` is set: DeepAgents converts the JSON schema to a Pydantic model and MUST NOT pass `response_format` to `create_deep_agent()` (native schema binding on the agent pass plus the deepagents tool surface exceeds Bedrock grammar limits and conflicts with extended thinking when enabled). After the agent run completes, the adapter MUST always run a second tool-free `with_structured_output(...)` call on a `ChatAnthropic*` model constructed **without** thinking, using the agent's text output as shaping input. The shape pass uses `method="json_schema"` on direct API and Vertex; on Bedrock it uses `method="function_calling"` because `json_schema` grammar compilation fails for large operator schemas. Phase 1 MAY use thinking when `reasoning_config.thinking` is set; the shape pass MUST NOT enable thinking. Schema conversion supports `properties`, `required`, `type`, `enum`, nested objects, and arrays; does not support `$ref`, `oneOf`, `allOf`, `additionalProperties`. Gemini sets native response MIME type and response schema on the content config. OpenAI uses two strategies based on endpoint type: **Native OpenAI endpoints** (api.openai.com) wrap the schema for the agents SDK output type with strict JSON-schema mode enabled. When strict mode is enabled, the schema is transformed to add `additionalProperties: false` and list all properties as required at every object level, as OpenAI's strict mode requires. Additionally, `oneOf` is rewritten to `anyOf` because OpenAI Structured Outputs rejects `oneOf`; `allOf` is left unchanged. **Non-native endpoints** (vLLM or OpenAI-compatible via `OPENAI_BASE_URL`) MUST use a two-pass pattern: Phase 1 runs the agent without `response_format` to allow tools to flow freely; Phase 2 runs a tool-free `with_structured_output(..., method="json_schema")` call using the agent's text output as shaping input, analogous to the DeepAgents strategy. Non-native endpoints do not support strict JSON-schema mode, so schema transformation (additionalProperties, oneOf rewrite) is skipped.

23. **Skills.** `cwd` is the skills root. Skill content lives at `cwd/<name>/SKILL.md`. DeepAgents and OpenAI MUST enable their SDK skills mechanism only when at least one immediate subdirectory of `cwd` contains a `SKILL.md` (`has_skills(cwd)`). DeepAgents then passes `skills=[cwd]` to `create_deep_agent()` (`SkillsMiddleware`). OpenAI registers the `Skills` capability with `LocalDirLazySkillSource` rooted at `cwd`; `skills_path="skills/.agents"` is the sandbox materialization path relative to the manifest root (`cwd.parent`), matching the operator emptyDir at `/app/skills/.agents` — it is not a host discovery path. An empty `cwd/.agents` directory MUST NOT enable skills. Gemini loads a skill toolset from the skill directory listing and omits it when none are found.

24. **Default allowed tools list.** Shared default names: `Bash`, `Read`, `Glob`, `Grep`, `Skill`. `run_agent_query()` always passes this list unless a future contract exposes overrides. [PLANNED: OLS-3033]

25. **Event logging.** A phase-tagged logger buffers `thinking_delta` events, flushes when buffer size exceeds an internal threshold or on `content_block_stop` or tool/result events, and logs truncated thinking. Tool calls and results are logged with separate input/output truncation caps. The `result` event logs the combined token count and truncated final text. [PLANNED: OLS-3928] DeepAgents MUST NOT log tool arguments or inspected tool-result content. It can log only controlled inspection fields and safe tool metadata.

26. **Stringifying tool I/O.** Non-string tool arguments and results are JSON-serialized for events when the SDK exposes structured objects.

27. **Gemini / Vertex.** When Vertex mode is enabled via environment, search-style tools MUST NOT be combined with non-search tools in the same agent tool list; the adapter omits those search tools in that mode.

28. **Gemini / exit loop.** When no `output_schema` is set, the adapter registers an SDK exit-loop tool; when `output_schema` is set, that tool is omitted.

29. **OpenAI client.** The OpenAI adapter selects its client and model wrapper from the provider type (see `configuration.md`):

    - **Native OpenAI / OpenAI-compatible** (`LIGHTSPEED_PROVIDER=openai`, or `vertex`/`OpenAI`): construct a plain `AsyncOpenAI` client with optional base URL override (`OPENAI_BASE_URL`) and wrap it in `OpenAIResponsesModel`.
    - **Azure OpenAI** (`LIGHTSPEED_PROVIDER=azure`): the adapter MUST use the OpenAI SDK's built-in Azure support — construct the SDK's native `AsyncAzureOpenAI` client with the SDK's own Azure parameters (`azure_endpoint`, `api_version`, `azure_deployment`) and wrap it in `OpenAIChatCompletionsModel(openai_client=...)`. It MUST NOT point a plain `AsyncOpenAI` at an Azure base URL and MUST NOT hand-build the `Authorization` header. Authentication follows the mode resolved by `configuration.md` rule 9a: **Entra ID mode** passes the built-in `azure_ad_token_provider = get_bearer_token_provider(ClientSecretCredential(tenant_id, client_id, client_secret), "https://cognitiveservices.azure.com/.default")`; **API-key mode** passes the native `api_key`. Token minting and refresh are owned by the provider SDK per rule 38. This closes the OLS-3049 gap (config mapping landed; the Azure client-construction path did not) as part of OLS-3050.

    Provider SDK and `azure.identity` imports stay inside the method per the optional-extra import convention. `AsyncAzureOpenAI` and `azure_ad_token_provider` ship in the `openai` package (already present via `openai-agents`), but the Entra credential classes (`ClientSecretCredential`, `get_bearer_token_provider`) come from `azure-identity` — a new optional dependency added under the `openai` extra (see `how/provider-architecture.md`).

30. **[Removed]** *(Claude adapter was removed in OLS-3500; MCP for Anthropic models is now handled by the DeepAgents adapter — see rule 33.)*

31. **MCP — Gemini.** When admitted `mcp_servers` is non-empty, the Gemini adapter MUST create `McpToolset` instances with `StreamableHTTPConnectionParams` for each admitted server, including resolved headers, and apply the admitted tool-name filter before the toolset is exposed to the model. If the SDK cannot enforce that filter or materialize every admitted server before model exposure, the provider query MUST fail.

32. **MCP — OpenAI.** When admitted `mcp_servers` is non-empty, the OpenAI adapter MUST create `MCPServerStreamableHttp` instances for each admitted server, including resolved headers, and apply the admitted tool-name filter through the SDK's native mechanism before passing them to the agent. Native and OpenAI-compatible paths MUST preserve the complete admitted server set. Conversion failure, manager initialization failure, or an incomplete active-server set MUST fail the provider query rather than continue without MCP tools.

33. **MCP — DeepAgents.** When admitted `mcp_servers` is non-empty, the DeepAgents adapter MUST load MCP tools via `langchain-mcp-adapters` `MultiServerMCPClient`, filter the returned tool objects using the admitted names, and pass only those tools to `create_deep_agent(tools=...)` where they merge with built-in harness tools. It MUST NOT pass an unfiltered MCP server to the agent. A reload error for an admitted server MUST fail the provider query rather than silently remove that server. If a successful reload returns fewer tools than were admitted, the adapter MAY continue with the returned admitted tools; it MUST NOT expose newly returned unadmitted tools.

34. **Reasoning — DeepAgents.** When `reasoning_config` is present, the DeepAgents adapter MUST pass the `thinking` key from the config to the `ChatAnthropic*` model constructor on the agent pass unchanged. Structured-output shaping (rule 22) MUST use a separate model instance without thinking.

35. **DeepAgents / Anthropic model routing.** The adapter resolves the model string to the correct LangChain chat model instance based on the backend configuration (see `configuration.md`). Direct Anthropic API uses `ChatAnthropic`. Vertex AI uses `ChatAnthropicVertex` (from `langchain_google_vertexai.model_garden`) with project and location from env. Bedrock uses `ChatAnthropicBedrock`. The resolved instance is passed to `create_deep_agent(model=...)`.

36. **DeepAgents / tool execution.** The adapter uses `LocalShellBackend` with `virtual_mode=False`, which provides built-in shell (`execute`), filesystem (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`), and `delete` tools. Filesystem tools operate on the real filesystem: absolute paths are used as-is, and relative paths are resolved under `root_dir` (options.cwd); filesystem tools are not confined to that directory. The shell tool (`execute`) has unrestricted system access regardless of `virtual_mode`. `virtual_mode=False` is required because deepagents >=0.7.18 changed the default to `True`, whose `_resolve_path` doubles absolute paths and raises on legitimate path components. The thin-adapter principle (rule 21) applies — tool execution is delegated to the deepagents backend.

37. **DeepAgents / prompt caching.** `AnthropicPromptCachingMiddleware` is applied unconditionally by `create_deep_agent()` and no-ops for non-Anthropic models. No adapter-level configuration needed.

38. **SDK-delegated short-lived tokens.** For providers that authenticate with short-lived access tokens derived from a long-lived credential, the sandbox mounts only the **long-lived** credential and delegates all short-lived token minting and refresh to the provider SDK's own credential object. The sandbox MUST NOT implement a token cache, refresh timer, or manual expiry/leeway logic. Because the sandbox reads the long-lived credential once at startup (one-shot batch process, no credential hot-reload), only the short-lived token is refreshed in-run — which is all a single run needs. Instances:

    | Provider | Long-lived credential (mounted) | SDK that mints/refreshes the short-lived token |
    | --- | --- | --- |
    | Vertex (existing) | `GOOGLE_APPLICATION_CREDENTIALS` service-account key | google-auth |
    | Azure Entra ID (OLS-3050) | `client_id` / `tenant_id` / `client_secret` | `azure.identity` `ClientSecretCredential` via `azure_ad_token_provider` (rule 29) |
    | AWS Bedrock (OLS-4092) | `aws_access_key_id` / `aws_secret_access_key` + optional `role_arn` | `botocore` credential-provider chain: with `role_arn` it performs STS assume-role and refreshes the short-lived credentials (see `configuration.md` rule 9b). The Anthropic-on-Bedrock model path is unchanged. |

### Agentic product trace normalization

39. [PLANNED: OLS-3569] Provider adapters MUST expose the complete provider-neutral completion, reasoning, tool call/result, explicit skill load/use, and terminal-result values required by `data-collection.md`. Provider-specific SDK object shapes MUST stop at the adapter boundary and MUST NOT create alternate content-event names.

40. [PLANNED: OLS-3569] Tool input/result and assistant/reasoning values retained for content trace events MUST NOT be length-truncated. The existing `EventLogger` can truncate its developer-log rendering. For DeepAgents tool calls and results, OLS-3928 rule 6 prohibits payload content in that rendering.

41. [PLANNED: OLS-3569] Every adapter's terminal `result` MUST carry the exact final response, requested-model fallback or actual response model, input tokens, output tokens, and reasoning tokens. When an SDK does not expose the actual model or a token category, the adapter MUST use the requested model or zero respectively; it MUST NOT omit the field or invent usage.

42. [PLANNED: OLS-3569] Gemini MUST retain terminal text from non-streamed ADK responses and pass it through the terminal `result`; it MUST NOT leave the final value empty because the text arrived in a non-partial event. Gemini MUST also expose response-model and token metadata under rule 41.

43. [PLANNED: OLS-3569] DeepAgents structured output MUST preserve the first agent pass's ordered completion, reasoning, tool, and skill signals and pass the second tool-free shape result as terminal `result` text. Usage totals MUST include both passes, and response-model fallback follows rule 41.

44. [PLANNED: OLS-3569] OpenAI MUST serialize `result.final_output` as the terminal `result` value and expose model and token metadata under rule 41, including reasoning tokens from output-token details when available.

45. [PLANNED: OLS-3569] Adapters MUST emit skill-loaded and skill-used signals only when their SDK or sandbox integration explicitly exposes those facts. They MUST include identity and all available content or metadata without redaction or truncation and MUST NOT infer skill use from model text or generic tool output.

46. [PLANNED: OLS-3569] Adapters MUST preserve the same tool name and stable call ID across each tool call/result pair and the corresponding operational tool span, retain complete input and output, and normalize result status to `ok` or `error`. When the SDK omits a call ID, the adapter MUST generate one stable ID for the pair.

### Tool-Result Prompt-Injection Inspection [PLANNED: OLS-3928]

 1. **Normative source.** The sandbox MUST conform to `openshift/ols/.ai/spec/what/tool-result-inspection.md`.

 2. **Runtime coverage.** The guarded adapter is DeepAgents only. Gemini and OpenAI adapters remain unchanged. Selection of an unguarded adapter MUST NOT cause a runtime warning.

 3. **Interception point.** DeepAgents middleware, or an equivalent tool wrapper, MUST inspect each effective model-visible result. Inspection occurs after artifact offload and before delivery to the main model or `ToolResultEvent` emission.

 4. **Model integration.** The middleware MUST construct the isolated classifier from the resolved DeepAgents model configuration. It MUST omit the main agent's reasoning configuration.

 5. **Local paths.** The interception paths include normal results, tool-generated errors, shell output, MCP output, file reads, and search results. They also include offload previews and references. Each later model-visible artifact read or search result MUST pass through the same middleware.

 6. **Event boundary.** For DeepAgents tool calls and results, the adapter MUST send only controlled, payload-free metadata to `EventLogger`. After a pass, the adapter MUST send the complete normalized `ToolResultEvent` to `AuditLogger`. This path retains the full result required by rules 39–46. A failed inspection MUST raise `ToolResultSafetyInspectionFailed` and send no result event to either logger.

 7. **Disabled behavior.** When `LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED` is false, the middleware MUST skip inspection calls and inspection-based termination. The main-system safety instruction remains active for every provider.

## Configuration Surface

| Mechanism | Purpose |
| ----------- | --------- |
| `ProviderQueryOptions.*` | All option fields listed above (set by router, not raw HTTP for most fields). |
| `GOOGLE_GENAI_USE_VERTEXAI` | Gemini: Vertex vs consumer API behavior and tool mix. Set internally by configuration mapping (see `configuration.md` rule 2), not by operator. |
| `OPENAI_BASE_URL` | OpenAI-compatible API endpoint override. Set internally by configuration mapping, not by operator. |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` | Azure: `azure_endpoint` / `api_version` for `AsyncAzureOpenAI` (rule 29). Set internally by configuration mapping. |
| `AZURE_OPENAI_API_KEY` | Azure API-key credential (API-key mode only). Populated from credentials secret envFrom. |
| `/var/run/secrets/llm-credentials/{client_id,tenant_id,client_secret}` | Azure Entra ID service-principal files for `ClientSecretCredential` (Entra ID mode, rule 29). Mounted by operator. |
| `GOOGLE_API_KEY`, `GEMINI_API_KEY` | Gemini credential and routing. Populated from credentials secret envFrom. |
| `ANTHROPIC_API_KEY` | DeepAgents/Anthropic: direct API credential. Populated from credentials secret envFrom. |
| `CLAUDE_CODE_USE_VERTEX` | DeepAgents/Anthropic: when `"1"`, adapter builds `ChatAnthropicVertex` instead of `ChatAnthropic`. Set by configuration mapping. |
| `CLAUDE_CODE_USE_BEDROCK` | DeepAgents/Anthropic: when `"1"`, adapter builds Bedrock-compatible chat model. Set by configuration mapping. |

## Constraints

- Not every adapter emits `thinking_delta` when reasoning is unconfigured; absence does not imply failure. DeepAgents MUST emit `thinking_delta` for Anthropic models that support extended thinking.
- DeepAgents structured output via Pydantic model conversion does not support all JSON Schema features (`$ref`, `oneOf`, `allOf`, `additionalProperties`). Schemas used by the operator MUST stay within the supported subset.
- Anthropic extended thinking (when `reasoning_config.thinking` is set) is incompatible with schema binding on the agent pass. The DeepAgents adapter MUST use two-phase structured output whenever `output_schema` is set (rule 22); thinking applies only to phase 1.

## Verification

- Unit: [test_run_agent.py](../../../tests/test_run_agent.py) — event stream, structured output, context prefix; [test_deepagents.py](../../../tests/test_deepagents.py) — DeepAgents structured output and admitted-name filtering; [test_mcp.py](../../../tests/test_mcp.py) — canonical admission projections and Gemini/OpenAI native filters; [test_openai_schema.py](../../../tests/test_openai_schema.py) — OpenAI complete-set initialization and fail-closed behavior
- [PLANNED: OLS-3928] Fast mock tests verify contract conformance, offloaded read paths, disabled inspection, and controlled sandbox failure.
- [PLANNED: OLS-3928] Integration tests verify inspection before `ToolResultEvent` emission. They verify payload-free `EventLogger` records and full-fidelity `AuditLogger` events after a pass. They also verify rejected-event suppression and controlled termination without a Result CR.
- The cross-repository real-model corpus and reporting requirements are owned by `openshift/ols/.ai/spec/what/tool-result-inspection.md`.
- Live batch: [skills.feature](../../../tests/e2e/features/skills.feature), [structured_output.feature](../../../tests/e2e/features/structured_output.feature), [mcp.feature](../../../tests/e2e/features/mcp.feature), [reasoning_config.feature](../../../tests/e2e/features/reasoning_config.feature)
- Harness helpers: [test_batch_e2e_helpers.py](../../../tests/test_batch_e2e_helpers.py) (no cluster)

## Planned Changes

- Parity improvements across providers (tools, streaming, structured output edge cases). [PLANNED: OLS-3047–OLS-3053]
- BYOK and RAG integration hooks without breaking the thin-adapter rule. [PLANNED: OLS-3054–OLS-3057]
- Align operator-passed `allowedTools` and `llm` with `ProviderQueryOptions`. [PLANNED: OLS-3033]
- Wire operator-resolved `Agent.spec.maxTurns` through `LIGHTSPEED_AGENT_MAX_TURNS` to each provider-native iteration limit. [PLANNED: OLS-3743]
- DeepAgents: token-level streaming via `astream_events()` instead of batch `stream_mode="messages"`. [PLANNED: OLS-3500]
- [PLANNED: OLS-3928] DeepAgents-only inspection of every model-visible tool result and error.
