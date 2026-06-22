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

17. **ProviderQueryOptions — `mcp_servers`.** Tuple of `McpServerConfig` (name, URL, optional headers) for external MCP server endpoints. When non-empty, each adapter wires MCP tools into its SDK: Claude passes an `mcp_servers` dict on `ClaudeAgentOptions` with `type: "http"` entries and adds `mcp__{name}__*` patterns to `allowed_tools`; Gemini appends `McpToolset` instances with `StreamableHTTPConnectionParams` to the agent tools list; OpenAI creates `MCPServerStreamableHttp` instances, manages their connect/cleanup lifecycle, and passes them to `SandboxAgent`. Only streamable HTTP transport is supported.

18. **Thin-adapter principle.** Providers MUST delegate tool execution, command invocation, and skill discovery to their SDKs. Adapters MUST NOT implement custom tool executors that duplicate SDK behavior except for minimal glue (e.g., auto-confirm, path layout).

19. **Structured output.** When `output_schema` is set: Claude uses the SDK’s JSON-schema output format; Gemini sets native response MIME type and response schema on the content config; OpenAI wraps the schema for the agents SDK output type with strict JSON-schema mode enabled for native OpenAI endpoints (api.openai.com) and disabled for custom endpoints (vLLM etc. via `OPENAI_BASE_URL`). When strict mode is enabled, the schema is transformed to add `additionalProperties: false` and list all properties as required at every object level, as OpenAI’s strict mode requires.

20. **Skills.** Claude discovers skills via SDK skill settings and a writable symlink layout under the effective cwd when the skill root is read-only. Gemini loads a skill toolset from the skill directory listing. OpenAI uses lazy skill loading from a local directory source rooted at `cwd`.

21. **Default allowed tools list.** Shared default names: `Bash`, `Read`, `Glob`, `Grep`, `Skill`. The HTTP route always passes this list unless a future contract exposes overrides. [PLANNED: OLS-3033]

22. **Event logging.** A phase-tagged logger buffers `thinking_delta` events, flushes when buffer size exceeds an internal threshold or on `content_block_stop` or tool/result events, and logs truncated thinking. Tool calls and results are logged with separate input/output truncation caps. The `result` event logs cost, combined token count, and truncated final text.

23. **Stringifying tool I/O.** Non-string tool arguments and results are JSON-serialized for events when the SDK exposes structured objects.

24. **Gemini / Vertex.** When Vertex mode is enabled via environment, search-style tools MUST NOT be combined with non-search tools in the same agent tool list; the adapter omits those search tools in that mode.

25. **Gemini / exit loop.** When no `output_schema` is set, the adapter registers an SDK exit-loop tool; when `output_schema` is set, that tool is omitted.

26. **OpenAI client.** The OpenAI adapter constructs an async OpenAI client with optional base URL override from environment (see `configuration.md`).

## Configuration Surface

| Mechanism | Purpose |
|-----------|---------|
| `ProviderQueryOptions.*` | All option fields listed above (set by router, not raw HTTP for most fields). |
| `GOOGLE_GENAI_USE_VERTEXAI` | Gemini: Vertex vs consumer API behavior and tool mix. Set internally by configuration mapping (see `configuration.md` rule 2), not by operator. |
| `OPENAI_BASE_URL` | OpenAI-compatible API endpoint override. Set internally by configuration mapping, not by operator. |
| `GOOGLE_API_KEY`, `GEMINI_API_KEY` | Gemini credential and routing. Populated from credentials secret envFrom. |
| `CLAUDE_CODE_USE_VERTEX`, `ANTHROPIC_VERTEX_PROJECT_ID`, `CLOUD_ML_REGION` | Claude via Vertex. Set internally by configuration mapping, not by operator. |

## Constraints

- Not every adapter emits `thinking_delta`; absence does not imply failure.
- Cost fields on `result` may be zero where the SDK does not report usage or price.

## Planned Changes

- Parity improvements across providers (tools, streaming, structured output edge cases). [PLANNED: OLS-3047–OLS-3053]
- BYOK and RAG integration hooks without breaking the thin-adapter rule. [PLANNED: OLS-3054–OLS-3057]
- Align operator-passed `allowedTools` and `llm` with `ProviderQueryOptions`. [PLANNED: OLS-3033]
