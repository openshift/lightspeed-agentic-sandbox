# Architecture: data flow, SDK integration

Audience: AI agents. File paths and symbols allowed here.
Package tree: `AGENTS.md`. Behavioral rules: `what/run-api.md`, `what/provider-contract.md`, `what/configuration.md`, `what/audit-logging.md`.

## Data Flow

1. Startup: `batch.main()` reads `/input/`, then calls `resolve_sdk()`, `parse_reasoning_config()`, and `parse_mcp_servers()` (fail-fast on bad env), `run_readiness_checks()`, `init_tracer()` when `otel_runtime_enabled()`, and `create_provider()`. [PLANNED: OLS-3743] Startup also requires and validates `LIGHTSPEED_AGENT_TIMEOUT_SECONDS` and `LIGHTSPEED_AGENT_MAX_TURNS` before provider invocation.
2. `run_agent_query()` applies context prefix, passes pre-parsed `mcp_servers` and operator-resolved maximum turns into `ProviderQueryOptions`, and calls `provider.query(...)`. [PLANNED: OLS-3743] The outer agent invocation is bounded by the operator-resolved timeout; timeout returns a structured classification used by Result status assembly.
3. Handler async-iterates events; `EventLogger` and `AuditLogger` side effects; metrics histograms updated; stops at first `result` event.
4. `publish_agent_result()` builds status from agent output, creates Result CR via Kubernetes API (`create_namespaced_custom_object`), replaces status (`replace_namespaced_custom_object_status`).
5. `shutdown_tracer()`; exit 0 on sandbox success (including agent failure), non-zero on infrastructure failure with termination log.

## Key Abstractions

- **Config mapping:** `resolve_sdk()` owns env → SDK name; factory does not read provider env vars.
- **Factory:** `create_provider(name)` lazy-imports the selected adapter.
- **Events:** Normalized `ProviderEvent` union decouples agent layer from vendor streaming models.
- **Options:** `ProviderQueryOptions` is the single bundle passed into every adapter (includes `mcp_servers`, `reasoning_config`).
- **Model resolution:** `resolve_router_model()` / `resolve_startup_model()` in `config.py`.
- **Result publishing:** `publish_results/publish.py` + `status.py` — Kubernetes client, no `oc` subprocess.

## Integration Points

- **Batch entrypoint:** `python -m lightspeed_agentic.batch` (`batch.py`).
- **Kubernetes API:** `kubernetes` Python client for Result CR create + status update (ServiceAccount token).
- **deepagents (+ langchain-anthropic, langchain-google-vertexai, langchain-aws, langchain-mcp-adapters):** `create_deep_agent`, `LocalShellBackend`, MCP via `MultiServerMCPClient`.
- **google-adk / google.genai:** `Agent`, `Runner`, `ExecuteBashTool`, `SkillToolset`. MCP via `McpToolset` + `StreamableHTTPConnectionParams`.
- **openai-agents (+ openai):** `SandboxAgent`, `Runner`, `UnixLocalSandboxClient`. MCP via `MCPServerStreamableHttp`.
- **OpenTelemetry:** `tracing.py` TracerProvider; `audit.py` GenAI spans/events; `metrics.py` in-process Prometheus histograms (no `/metrics` route).

## Implementation Notes

- **DeepAgents model routing:** `_resolve_model()` checks `CLAUDE_CODE_USE_VERTEX` and `CLAUDE_CODE_USE_BEDROCK`.
- **DeepAgents streaming:** `astream(stream_mode="messages")`.
- **Gemini bash:** Monkey-patches `run_async` for confirmation and `bash -c` wrapping.
- **MCP Secret headers:** First file (sorted by name) under `/var/secrets/mcp/<secretName>/`.
- **Containerfile:** Multi-stage hermetic build; `oc`/`kubectl` in image for **agent tools** (not Result CR publishing); user `agent`; `catatonit`; batch CMD.
- **Unit tests:** `test_run_agent.py`, `test_batch.py`, `test_ready.py`, `test_publish_results_*.py`, `test_batch_e2e_helpers.py` (harness helpers, no cluster).
- **[PLANNED: OLS-3743] Execution limits:** Parse timeout/max-turn environment values once in `batch.py`; pass the parsed values to `run_agent_query()`. Provider adapters continue to receive maximum turns only through `ProviderQueryOptions`. Preserve timeout as structured internal state through `publish_results/status.py` so Result condition selection never depends on matching summary text.
- **Live batch BDD:** `tests/e2e/` feature files via `scripts/e2e-containers.sh` — see [e2e-testing.md](../what/e2e-testing.md).
- **Evals:** separate HTTP integration suite (`evals/`); not migrated to the batch entrypoint.
