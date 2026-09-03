# Behavioral spec: provider abstraction and events

Audience: AI agents (Claude). Precision over narrative.

Cross-references: batch agent invocation → `run-api.md`. Env and build → `configuration.md`.

## Behavioral Rules

1. **AgentProvider.** Each backend implements a `name` property and a `query` method accepting `ProviderQueryOptions` and returning an async iterator of `ProviderEvent`.

2. **Text delta (`text_delta`).** Carries incremental natural-language or assistant text chunks for logging or streaming use.

3. **Thinking delta (`thinking_delta`).** Carries incremental chain-of-thought or reasoning text. When reasoning is configured and the SDK produces reasoning output, all adapters MUST emit `thinking_delta` events. DeepAgents emits from `AIMessage.content_blocks` with `type == "reasoning"`. Gemini MUST emit from `ThinkingConfig` thought parts when `include_thoughts` is enabled. OpenAI MUST emit from reasoning items in the response stream.

4. **Content block stop (`content_block_stop`).** Signals that a content or tool block has completed; used by logging to flush buffered thinking.

5. **Tool call (`tool_call`).** Carries the tool name and a string representation of inputs (length-truncated per internal adapter limits).

6. **Tool result (`tool_result`).** Carries stringified tool output (length-truncated per internal adapter limits).

7. **Result (`result`).** Terminal event: final text payload (may be JSON or plain text depending on structured-output path), USD cost (numeric; adapters may report zero when the SDK lacks cost), and input/output token counts.

8. **ProviderQueryOptions — `prompt`.** Full user message after any context prefix formatting in `run_agent.py` (see `run-api.md` context rules).

9. **ProviderQueryOptions — `system_prompt`.** System or developer instruction string.

10. **ProviderQueryOptions — `model`.** Model identifier resolved before the call (see `configuration.md`).

11. **ProviderQueryOptions — `max_turns`.** Upper bound on agent/SDK iteration. [PLANNED: OLS-3743] The value comes from required `LIGHTSPEED_AGENT_MAX_TURNS`, resolved by the operator from `Agent.spec.maxTurns` with a default of 200. Adapters MUST pass it to their native limit: DeepAgents/LangGraph `recursion_limit`, Gemini ADK `max_llm_calls`, and OpenAI Agents `max_turns`. These mechanisms are not semantically identical, but all enforce an upper bound; adapters MUST NOT substitute their SDK defaults.

12. **ProviderQueryOptions — `max_budget_usd`.** SDK-level spend ceiling in USD.

13. **ProviderQueryOptions — `allowed_tools`.** List of tool names the SDK may use for that invocation.

14. **ProviderQueryOptions — `cwd`.** Directory used as skill root and/or workspace for filesystem and shell tools.

15. **ProviderQueryOptions — `output_schema`.** Optional JSON-schema dict; when set, adapters map it to the SDK's native structured-output mechanism.

16. **ProviderQueryOptions — `stream`.** When true, adapters that support partial streaming should yield deltas; when false, they may batch. The batch entrypoint does not set this flag from input files.

17. **ProviderQueryOptions — `mcp_servers`.** Optional list of `ResolvedMCPServer` values from `mcp.parse_mcp_servers()`. Each entry carries `name`, `url`, `timeout`, and `headers` as a list of `ResolvedMCPHeader` (`name`, `value`). Adapters MAY convert headers to a dict at the SDK boundary. When non-empty, adapters MUST wire these servers into their SDK's native MCP client mechanism (see rules 31–34). When empty or absent, no MCP servers are configured.

18. **ProviderQueryOptions — `reasoning_config`.** Optional dict (JSON object). When present, adapters MUST map it to their SDK's native reasoning/thinking parameters. When absent or `None`, adapters MUST NOT set any reasoning parameters and SDK defaults apply. DeepAgents passes only the `thinking` key through to `ChatAnthropic*`. Gemini constructs `ThinkingConfig(**config)` and OpenAI constructs `Reasoning(**rc)` — extra keys are forwarded to the SDK constructors (not stripped by the adapter); invalid values fail at SDK/API invocation time.

19. **[Removed]** *(Claude adapter was removed in OLS-3500; Anthropic reasoning is now handled by the DeepAgents adapter — see rule 35.)*

20. **Reasoning — Gemini.** When `reasoning_config` is present, the Gemini adapter MUST construct a `types.ThinkingConfig(**config)` and pass it via `GenerateContentConfig.thinking_config` on the Agent. Config keys (e.g. `thinking_budget`, `thinking_level`, `include_thoughts`) are forwarded into `ThinkingConfig`; the Gemini API validates at invocation time.

21. **Reasoning — OpenAI.** When `reasoning_config` is present, the OpenAI adapter MUST construct `ModelSettings(reasoning=Reasoning(**rc), verbosity=...)` from the config keys (e.g. `effort`, `mode`, `context`, `verbosity`) and pass it to `SandboxAgent(model_settings=...)`. Config keys are forwarded into `Reasoning`; the OpenAI API validates at invocation time.

22. **Thin-adapter principle.** Providers MUST delegate tool execution, command invocation, and skill discovery to their SDKs. Adapters MUST NOT implement custom tool executors that duplicate SDK behavior except for minimal glue (e.g., auto-confirm, path layout).

23. **Structured output.** When `output_schema` is set: DeepAgents converts the JSON schema to a Pydantic model and MUST NOT pass `response_format` to `create_deep_agent()` (native schema binding on the agent pass plus the deepagents tool surface exceeds Bedrock grammar limits and conflicts with extended thinking when enabled). After the agent run completes, the adapter MUST always run a second tool-free `with_structured_output(...)` call on a `ChatAnthropic*` model constructed **without** thinking, using the agent's text output as shaping input. The shape pass uses `method="json_schema"` on direct API and Vertex; on Bedrock it uses `method="function_calling"` because `json_schema` grammar compilation fails for large operator schemas. Phase 1 MAY use thinking when `reasoning_config.thinking` is set; the shape pass MUST NOT enable thinking. Schema conversion supports `properties`, `required`, `type`, `enum`, nested objects, and arrays; does not support `$ref`, `oneOf`, `allOf`, `additionalProperties`. Gemini sets native response MIME type and response schema on the content config. OpenAI wraps the schema for the agents SDK output type with strict JSON-schema mode enabled for native OpenAI endpoints (api.openai.com) and disabled for custom endpoints (vLLM etc. via `OPENAI_BASE_URL`). When strict mode is enabled, the schema is transformed to add `additionalProperties: false` and list all properties as required at every object level, as OpenAI's strict mode requires. Additionally, `oneOf` is rewritten to `anyOf` because OpenAI Structured Outputs rejects `oneOf`; `allOf` is left unchanged.

24. **Skills.** `cwd` is the skills root. Skill content lives at `cwd/<name>/SKILL.md`. DeepAgents and OpenAI MUST enable their SDK skills mechanism only when at least one immediate subdirectory of `cwd` contains a `SKILL.md` (`has_skills(cwd)`). DeepAgents then passes `skills=[cwd]` to `create_deep_agent()` (`SkillsMiddleware`). OpenAI registers the `Skills` capability with `LocalDirLazySkillSource` rooted at `cwd`; `skills_path="skills/.agents"` is the sandbox materialization path relative to the manifest root (`cwd.parent`), matching the operator emptyDir at `/app/skills/.agents` — it is not a host discovery path. An empty `cwd/.agents` directory MUST NOT enable skills. Gemini loads a skill toolset from the skill directory listing and omits it when none are found.

25. **Default allowed tools list.** Shared default names: `Bash`, `Read`, `Glob`, `Grep`, `Skill`. `run_agent_query()` always passes this list unless a future contract exposes overrides. [PLANNED: OLS-3033]

26. **Event logging.** A phase-tagged logger buffers `thinking_delta` events, flushes when buffer size exceeds an internal threshold or on `content_block_stop` or tool/result events, and logs truncated thinking. Tool calls and results are logged with separate input/output truncation caps. The `result` event logs cost, combined token count, and truncated final text.

27. **Stringifying tool I/O.** Non-string tool arguments and results are JSON-serialized for events when the SDK exposes structured objects.

28. **Gemini / Vertex.** When Vertex mode is enabled via environment, search-style tools MUST NOT be combined with non-search tools in the same agent tool list; the adapter omits those search tools in that mode.

29. **Gemini / exit loop.** When no `output_schema` is set, the adapter registers an SDK exit-loop tool; when `output_schema` is set, that tool is omitted.

30. **OpenAI client.** The OpenAI adapter selects its client and model wrapper from the provider type (see `configuration.md`):

    - **Native OpenAI / OpenAI-compatible** (`LIGHTSPEED_PROVIDER=openai`, or `vertex`/`OpenAI`): construct a plain `AsyncOpenAI` client with optional base URL override (`OPENAI_BASE_URL`) and wrap it in `OpenAIResponsesModel`.
    - **Azure OpenAI** (`LIGHTSPEED_PROVIDER=azure`): the adapter MUST use the OpenAI SDK's built-in Azure support — construct the SDK's native `AsyncAzureOpenAI` client with the SDK's own Azure parameters (`azure_endpoint`, `api_version`, `azure_deployment`) and wrap it in `OpenAIChatCompletionsModel(openai_client=...)`. It MUST NOT point a plain `AsyncOpenAI` at an Azure base URL and MUST NOT hand-build the `Authorization` header. Authentication follows the mode resolved by `configuration.md` rule 9a: **Entra ID mode** passes the built-in `azure_ad_token_provider = get_bearer_token_provider(ClientSecretCredential(tenant_id, client_id, client_secret), "https://cognitiveservices.azure.com/.default")`; **API-key mode** passes the native `api_key`. Token minting and refresh are owned by the provider SDK per rule 39. This closes the OLS-3049 gap (config mapping landed; the Azure client-construction path did not) as part of OLS-3050.

    Provider SDK and `azure.identity` imports stay inside the method per the optional-extra import convention. `AsyncAzureOpenAI` and `azure_ad_token_provider` ship in the `openai` package (already present via `openai-agents`), but the Entra credential classes (`ClientSecretCredential`, `get_bearer_token_provider`) come from `azure-identity` — a new optional dependency added under the `openai` extra (see `how/provider-architecture.md`).

31. **[Removed]** *(Claude adapter was removed in OLS-3500; MCP for Anthropic models is now handled by the DeepAgents adapter — see rule 34.)*

32. **MCP — Gemini.** When `mcp_servers` is non-empty, the Gemini adapter MUST create `McpToolset` instances with `StreamableHTTPConnectionParams` for each server (including resolved headers) and add them to the agent's `tools` list alongside existing tools.

33. **MCP — OpenAI.** When `mcp_servers` is non-empty, the OpenAI adapter MUST create `MCPServerStreamableHttp` instances for each server (with resolved headers) and pass them to the agent's `mcp_servers` parameter.

34. **MCP — DeepAgents.** When `mcp_servers` is non-empty, the DeepAgents adapter MUST load MCP tools via `langchain-mcp-adapters` `MultiServerMCPClient` and pass them to `create_deep_agent(tools=...)` where they merge with built-in harness tools.

35. **Reasoning — DeepAgents.** When `reasoning_config` is present, the DeepAgents adapter MUST pass the `thinking` key from the config to the `ChatAnthropic*` model constructor on the agent pass unchanged. Structured-output shaping (rule 23) MUST use a separate model instance without thinking.

36. **DeepAgents / Anthropic model routing.** The adapter resolves the model string to the correct LangChain chat model instance based on the backend configuration (see `configuration.md`). Direct Anthropic API uses `ChatAnthropic`. Vertex AI uses `ChatAnthropicVertex` (from `langchain_google_vertexai.model_garden`) with project and location from env. Bedrock uses `ChatAnthropicBedrock`. The resolved instance is passed to `create_deep_agent(model=...)`.

37. **DeepAgents / tool execution.** The adapter uses `LocalShellBackend` which provides built-in shell (`execute`), filesystem (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`), and `delete` tools. The thin-adapter principle (rule 22) applies — tool execution is delegated to the deepagents backend.

38. **DeepAgents / prompt caching.** `AnthropicPromptCachingMiddleware` is applied unconditionally by `create_deep_agent()` and no-ops for non-Anthropic models. No adapter-level configuration needed.

39. **SDK-delegated short-lived tokens.** For providers that authenticate with short-lived access tokens derived from a long-lived credential, the sandbox mounts only the **long-lived** credential and delegates all short-lived token minting and refresh to the provider SDK's own credential object. The sandbox MUST NOT implement a token cache, refresh timer, or manual expiry/leeway logic. Because the sandbox reads the long-lived credential once at startup (one-shot batch process, no credential hot-reload), only the short-lived token is refreshed in-run — which is all a single run needs. Instances:

    | Provider | Long-lived credential (mounted) | SDK that mints/refreshes the short-lived token |
    |---|---|---|
    | Vertex (existing) | `GOOGLE_APPLICATION_CREDENTIALS` service-account key | google-auth |
    | Azure Entra ID (OLS-3050) | `client_id` / `tenant_id` / `client_secret` | `azure.identity` `ClientSecretCredential` via `azure_ad_token_provider` (rule 30) |
    | AWS Bedrock (OLS-4092) | `aws_access_key_id` / `aws_secret_access_key` + optional `role_arn` | `botocore` credential-provider chain: with `role_arn` it performs STS assume-role and refreshes the short-lived credentials (see `configuration.md` rule 9b). The Anthropic-on-Bedrock model path is unchanged. |

## Configuration Surface

| Mechanism | Purpose |
|-----------|---------|
| `ProviderQueryOptions.*` | All option fields listed above (set by router, not raw HTTP for most fields). |
| `GOOGLE_GENAI_USE_VERTEXAI` | Gemini: Vertex vs consumer API behavior and tool mix. Set internally by configuration mapping (see `configuration.md` rule 2), not by operator. |
| `OPENAI_BASE_URL` | OpenAI-compatible API endpoint override. Set internally by configuration mapping, not by operator. |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` | Azure: `azure_endpoint` / `api_version` for `AsyncAzureOpenAI` (rule 30). Set internally by configuration mapping. |
| `AZURE_OPENAI_API_KEY` | Azure API-key credential (API-key mode only). Populated from credentials secret envFrom. |
| `/var/run/secrets/llm-credentials/{client_id,tenant_id,client_secret}` | Azure Entra ID service-principal files for `ClientSecretCredential` (Entra ID mode, rule 30). Mounted by operator. |
| `GOOGLE_API_KEY`, `GEMINI_API_KEY` | Gemini credential and routing. Populated from credentials secret envFrom. |
| `ANTHROPIC_API_KEY` | DeepAgents/Anthropic: direct API credential. Populated from credentials secret envFrom. |
| `CLAUDE_CODE_USE_VERTEX` | DeepAgents/Anthropic: when `"1"`, adapter builds `ChatAnthropicVertex` instead of `ChatAnthropic`. Set by configuration mapping. |
| `CLAUDE_CODE_USE_BEDROCK` | DeepAgents/Anthropic: when `"1"`, adapter builds Bedrock-compatible chat model. Set by configuration mapping. |

## Constraints

- Not every adapter emits `thinking_delta` when reasoning is unconfigured; absence does not imply failure. DeepAgents MUST emit `thinking_delta` for Anthropic models that support extended thinking.
- Cost fields on `result` may be zero where the SDK does not report usage or price. DeepAgents reports `cost_usd=0`; token counts are available via LangChain `usage_metadata`.
- DeepAgents structured output via Pydantic model conversion does not support all JSON Schema features (`$ref`, `oneOf`, `allOf`, `additionalProperties`). Schemas used by the operator MUST stay within the supported subset.
- Anthropic extended thinking (when `reasoning_config.thinking` is set) is incompatible with schema binding on the agent pass. The DeepAgents adapter MUST use two-phase structured output whenever `output_schema` is set (rule 23); thinking applies only to phase 1.

## Verification

- Unit: [test_run_agent.py](../../../tests/test_run_agent.py) — event stream, structured output, context prefix; [test_deepagents.py](../../../tests/test_deepagents.py) — DeepAgents structured output strategy when thinking is configured
- Live batch: [skills.feature](../../../tests/e2e/features/skills.feature), [structured_output.feature](../../../tests/e2e/features/structured_output.feature), [mcp.feature](../../../tests/e2e/features/mcp.feature), [reasoning_config.feature](../../../tests/e2e/features/reasoning_config.feature)
- Harness helpers: [test_batch_e2e_helpers.py](../../../tests/test_batch_e2e_helpers.py) (no cluster)

## Planned Changes

- Parity improvements across providers (tools, streaming, structured output edge cases). [PLANNED: OLS-3047–OLS-3053]
- BYOK and RAG integration hooks without breaking the thin-adapter rule. [PLANNED: OLS-3054–OLS-3057]
- Align operator-passed `allowedTools` and `llm` with `ProviderQueryOptions`. [PLANNED: OLS-3033]
- Wire operator-resolved `Agent.spec.maxTurns` through `LIGHTSPEED_AGENT_MAX_TURNS` to each provider-native iteration limit. [PLANNED: OLS-3743]
- DeepAgents: token-level streaming via `astream_events()` instead of batch `stream_mode="messages"`. [PLANNED: OLS-3500]
- DeepAgents: cost tracking from token counts x model pricing. [PLANNED: OLS-3500]
- DeepAgents: `max_budget_usd` enforcement via adapter-level token cost tracking. [PLANNED: OLS-3500]
- DeepAgents: `allowed_tools` filtering at `create_deep_agent(tools=...)` construction. [PLANNED: OLS-3500]
