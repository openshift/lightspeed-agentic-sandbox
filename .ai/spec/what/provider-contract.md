# Behavioral spec: provider abstraction and events

Audience: AI agents (Claude). Precision over narrative.

Cross-references: HTTP mapping of prompts and timeouts → `run-api.md`. Env and build → `configuration.md`.

## Behavioral Rules

1. **AgentProvider.** Each backend implements a `name` property and a `query` method accepting `ProviderQueryOptions` and returning an async iterator of `ProviderEvent`.

2. **Text delta (`text_delta`).** Carries incremental natural-language or assistant text chunks for logging or streaming use.

3. **Thinking delta (`thinking_delta`).** Carries incremental chain-of-thought or reasoning text where the SDK exposes it.

4. **Content block stop (`content_block_stop`).** Signals that a content or tool block has completed; used by logging to flush buffered thinking.

5. **Tool call (`tool_call`).** Carries the tool name and a string representation of inputs (length-truncated per internal adapter limits).

6. **Tool result (`tool_result`).** Carries stringified tool output (length-truncated per internal adapter limits).

7. **Result (`result`).** Terminal event: final text payload (may be JSON or plain text depending on structured-output path), USD cost (numeric; adapters may report zero when the SDK lacks cost), and input/output token counts.

8. **ProviderQueryOptions — `prompt`.** Full user message after any HTTP-layer prefixing (see `run-api.md` context rules).

9. **ProviderQueryOptions — `system_prompt`.** System or developer instruction string.

10. **ProviderQueryOptions — `model`.** Model identifier resolved before the call (see `configuration.md`).

11. **ProviderQueryOptions — `max_turns`.** Upper bound on agent/SDK turn or LLM-call loops.

12. **ProviderQueryOptions — `max_budget_usd`.** SDK-level spend ceiling in USD.

13. **ProviderQueryOptions — `allowed_tools`.** List of tool names the SDK may use for that invocation.

14. **ProviderQueryOptions — `cwd`.** Directory used as skill root and/or workspace for filesystem and shell tools.

15. **ProviderQueryOptions — `output_schema`.** Optional JSON-schema dict; when set, adapters map it to the SDK’s native structured-output mechanism.

16. **ProviderQueryOptions — `stream`.** When true, adapters that support partial streaming should yield deltas; when false, they may batch. The HTTP `POST /run` path does not set this flag from the request body.

17. **Thin-adapter principle.** Providers MUST delegate tool execution, command invocation, and skill discovery to their SDKs. Adapters MUST NOT implement custom tool executors that duplicate SDK behavior except for minimal glue (e.g., auto-confirm, path layout).

18. **Structured output.** When `output_schema` is set: DeepAgents converts the JSON schema to a Pydantic model via `response_format` (supports `properties`, `required`, `type`, `enum`, nested objects, and arrays; does not support `$ref`, `oneOf`, `allOf`, `additionalProperties`). Gemini sets native response MIME type and response schema on the content config. OpenAI wraps the schema for the agents SDK output type with strict JSON-schema mode enabled for native OpenAI endpoints (api.openai.com) and disabled for custom endpoints (vLLM etc. via `OPENAI_BASE_URL`). When strict mode is enabled, the schema is transformed to add `additionalProperties: false` and list all properties as required at every object level, as OpenAI’s strict mode requires.

19. **Skills.** DeepAgents loads skills from the `cwd` directory via `skills=[cwd]` parameter passed to `create_deep_agent()`; the `SkillsMiddleware` handles discovery and progressive disclosure. Gemini loads a skill toolset from the skill directory listing. OpenAI uses lazy skill loading from a local directory source rooted at `cwd`.

20. **Default allowed tools list.** Shared default names: `Bash`, `Read`, `Glob`, `Grep`, `Skill`. The HTTP route always passes this list unless a future contract exposes overrides. [PLANNED: OLS-3033]

21. **Event logging.** A phase-tagged logger buffers `thinking_delta` events, flushes when buffer size exceeds an internal threshold or on `content_block_stop` or tool/result events, and logs truncated thinking. Tool calls and results are logged with separate input/output truncation caps. The `result` event logs cost, combined token count, and truncated final text.

22. **Stringifying tool I/O.** Non-string tool arguments and results are JSON-serialized for events when the SDK exposes structured objects.

23. **Gemini / Vertex.** When Vertex mode is enabled via environment, search-style tools MUST NOT be combined with non-search tools in the same agent tool list; the adapter omits those search tools in that mode.

24. **Gemini / exit loop.** When no `output_schema` is set, the adapter registers an SDK exit-loop tool; when `output_schema` is set, that tool is omitted.

25. **OpenAI client.** The OpenAI adapter constructs an async OpenAI client with optional base URL override from environment (see `configuration.md`).

26. **DeepAgents / Anthropic model routing.** The adapter resolves the model string to the correct LangChain chat model instance based on the backend configuration (see `configuration.md`). Direct Anthropic API uses `ChatAnthropic`. Vertex AI uses `ChatAnthropicVertex` with project and location from env. Bedrock uses `ChatBedrockConverse` or equivalent `langchain-aws` client. The resolved instance is passed to `create_deep_agent(model=...)`.

27. **DeepAgents / extended thinking.** The adapter MUST configure the `ChatAnthropic` (or Vertex/Bedrock variant) model instance with thinking enabled for models that support it. Thinking content blocks in `AIMessage.content` (entries with `type="thinking"`) MUST be extracted and yielded as `ThinkingDeltaEvent`. This is required for `audit.agent.thinking` compliance (see `audit-logging.md` rule 7).

28. **DeepAgents / tool execution.** The adapter uses `LocalShellBackend` which provides built-in shell (`execute`), filesystem (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`), and `delete` tools. The thin-adapter principle (rule 17) applies — tool execution is delegated to the deepagents backend.

29. **DeepAgents / prompt caching.** `AnthropicPromptCachingMiddleware` is applied unconditionally by `create_deep_agent()` and no-ops for non-Anthropic models. No adapter-level configuration needed.

## Configuration Surface

| Mechanism | Purpose |
|-----------|---------|
| `ProviderQueryOptions.*` | All option fields listed above (set by router, not raw HTTP for most fields). |
| `GOOGLE_GENAI_USE_VERTEXAI` | Gemini: Vertex vs consumer API behavior and tool mix. Set internally by configuration mapping (see `configuration.md` rule 2), not by operator. |
| `OPENAI_BASE_URL` | OpenAI-compatible API endpoint override. Set internally by configuration mapping, not by operator. |
| `GOOGLE_API_KEY`, `GEMINI_API_KEY` | Gemini credential and routing. Populated from credentials secret envFrom. |
| `ANTHROPIC_API_KEY` | DeepAgents/Anthropic: direct API credential. Populated from credentials secret envFrom. |
| `CLAUDE_CODE_USE_VERTEX` | DeepAgents/Anthropic: when `"1"`, adapter builds `ChatAnthropicVertex` instead of `ChatAnthropic`. Set by configuration mapping. |
| `CLAUDE_CODE_USE_BEDROCK` | DeepAgents/Anthropic: when `"1"`, adapter builds Bedrock-compatible chat model. Set by configuration mapping. |

## Constraints

- Not every adapter emits `thinking_delta`; absence does not imply failure. DeepAgents MUST emit `thinking_delta` for Anthropic models that support extended thinking.
- Cost fields on `result` may be zero where the SDK does not report usage or price. DeepAgents reports `cost_usd=0`; token counts are available via LangChain `usage_metadata`.
- DeepAgents structured output via Pydantic model conversion does not support all JSON Schema features (`$ref`, `oneOf`, `allOf`, `additionalProperties`). Schemas used by the operator MUST stay within the supported subset.

## Planned Changes

- Parity improvements across providers (tools, streaming, structured output edge cases). [PLANNED: OLS-3047–OLS-3053]
- BYOK and RAG integration hooks without breaking the thin-adapter rule. [PLANNED: OLS-3054–OLS-3057]
- Align operator-passed `allowedTools` and `llm` with `ProviderQueryOptions`. [PLANNED: OLS-3033]
- DeepAgents: token-level streaming via `astream_events()` instead of batch `stream_mode="values"`. [PLANNED: OLS-3500]
- DeepAgents: cost tracking from token counts × model pricing. [PLANNED: OLS-3500]
- DeepAgents: `max_budget_usd` enforcement via adapter-level token cost tracking. [PLANNED: OLS-3500]
- DeepAgents: `allowed_tools` filtering at `create_deep_agent(tools=...)` construction or via 0.7.x tool allowlist. [PLANNED: OLS-3500]
