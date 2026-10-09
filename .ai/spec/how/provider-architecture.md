# Architecture: data flow, SDK integration

Audience: AI agents. File paths and symbols allowed here.
Package tree: `AGENTS.md`. Behavioral rules: `what/run-api.md`, `what/provider-contract.md`, `what/configuration.md`, `what/audit-logging.md`.

## Data Flow

1. Startup: `batch.main()` reads `/input/`, then calls `resolve_sdk()`, `parse_reasoning_config()`, and `parse_mcp_servers()` (fail-fast on bad env), `run_readiness_checks()`, `init_tracer()` when `otel_runtime_enabled()`, and `create_provider()`. [PLANNED: OLS-3743] Startup also requires and validates `LIGHTSPEED_AGENT_TIMEOUT_SECONDS` and `LIGHTSPEED_AGENT_MAX_TURNS` before provider invocation.
2. `run_agent_query()` applies context prefix, passes pre-parsed `mcp_servers` and operator-resolved maximum turns into `ProviderQueryOptions`, and calls `provider.query(...)`. [PLANNED: OLS-3743] The outer agent invocation is bounded by the operator-resolved timeout; timeout returns a structured classification used by Result status assembly.
2a. [PLANNED: OLS-3928] The DeepAgents adapter installs result-inspection middleware around model-visible tool results and errors. The middleware uses the resolved DeepAgents model for isolated classifier calls.
2b. DeepAgents installs model-boundary middleware even when inspection is disabled. Output limits and artifact offload run first. Enabled inspection runs before wrapping. The middleware escapes external closing markers, HTML-escapes tool names, and wraps results before model delivery. The adapter appends the trust instruction to the main-agent and general-purpose-subagent system prompts.
3. `run_agent_query()` starts an INTERNAL `invoke_agent` span beneath the received operator context, with effective prompt/system attributes and exact terminal `ResultEvent.text` before shaping; it uses the configured request model without claiming a provider or response model.
4. `AuditLogger` creates local `execute_tool` INTERNAL children. Existing normalized ProviderEvents still drive `EventLogger`, legacy `gen_ai.choice`/templog projections, metrics, and result handling; choice-event flush order is not transcript chronology.
5. DeepAgents and OpenAI capture accepted main-agent `chat` generations; Gemini captures main-agent `generate_content` through callbacks plus finalized Runner events, preserving ADK call IDs and ending at the model-completion timestamp before local tool execution. All reuse captured invocation context without making generation spans current. `invoke_agent` is a child of the received operator context, and each accepted generation and local `execute_tool` span is a direct child of `invoke_agent`. The stdout and OTLP trace exporters exclude only spans whose instrumentation scope name is exactly `gcp.vertex.agent`; they do not rename/project spans, change native ADK span processing or event-to-log processing, or alter other scopes/gates. Existing provider events, developer logs, and templog projections are unchanged.
6. `publish_agent_result()` builds status from agent output, creates Result CR via Kubernetes API (`create_namespaced_custom_object`), replaces status (`replace_namespaced_custom_object_status`).
7. `shutdown_tracer()`; exit 0 on sandbox success (including agent failure), non-zero on infrastructure failure with termination log.

## Key Abstractions

- **Config mapping:** `resolve_sdk()` owns env → SDK name; factory does not read provider env vars.
- **Factory:** `create_provider(name)` lazy-imports the selected adapter.
- **Events:** Normalized `ProviderEvent` union decouples agent layer from vendor streaming models.
- **Options:** `ProviderQueryOptions` is the single bundle passed into every adapter (includes `mcp_servers`, `reasoning_config`).
- **Model resolution:** `resolve_router_model()` / `resolve_startup_model()` in `config.py`.
- **Result publishing:** `publish_results/publish.py` + `status.py` — Kubernetes client, no `oc` subprocess.
- **Result inspector [PLANNED: OLS-3928]:** A focused module implements `openshift/ols/.ai/spec/what/tool-result-inspection.md` and exposes `ToolResultSafetyInspectionFailed` to the batch path.
- **Tool-output boundary:** A focused formatter escapes external closing markers and tool names, then supplies the fixed markers. DeepAgents middleware applies the formatter at the model boundary, not in the provider-event consumer. See `../decisions/0001-tool-output-boundary.md`.

## Provider-egress CA bundle [PLANNED: OLS-3042]

The batch startup/configuration path discovers `.crt` and `.pem` files below
`/var/run/secrets/lightspeed/tls/`, combines them with the platform/system
trust store, and exposes one runtime bundle to Python and provider TLS code.
Provider adapters MUST consume the shared bundle and MUST NOT contain
operator Secret names, source-specific CA paths, or per-source trust logic.
The generic TLS values passed by the agentic operator are parsed once at the
runtime boundary; adapters receive configured TLS behavior through shared
configuration rather than independent CA arguments.

## Integration Points

- **Batch entrypoint:** `python -m lightspeed_agentic.batch` (`batch.py`).
- **Kubernetes API:** `kubernetes` Python client for Result CR create + status update (ServiceAccount token).
- **deepagents (+ langchain-anthropic, langchain-google-vertexai, langchain-aws, langchain-mcp-adapters):** `create_deep_agent`, `LocalShellBackend`, MCP via `MultiServerMCPClient`.
- **google-adk / google.genai:** `Agent`, `Runner`, `ExecuteBashTool`, `SkillToolset`. MCP via `McpToolset` + `StreamableHTTPConnectionParams`.
- **openai-agents (+ openai):** `SandboxAgent`, `Runner`, `UnixLocalSandboxClient`. MCP via `MCPServerStreamableHttp`. Client selection by provider (`what/provider-contract.md` rule 29): native `api.openai.com` → `AsyncOpenAI` + `OpenAIResponsesModel`; custom/vLLM OpenAI-compatible endpoint → `AsyncOpenAI` + `OpenAIChatCompletionsModel`; Azure → native `AsyncAzureOpenAI` + `OpenAIResponsesModel` or `OpenAIChatCompletionsModel` according to API-version support.
- **azure.identity (Azure Entra ID):** `ClientSecretCredential` + `get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")` supplies the `azure_ad_token_provider` passed to `AsyncAzureOpenAI`; the library owns token caching/refresh (`what/provider-contract.md` rule 38). Imported inside the adapter method (optional-extra convention). **New dependency** [OLS-3050]: `azure-identity` (pulls `azure-core`) is added to the `openai` optional extra — `AsyncAzureOpenAI` and its `azure_ad_token_provider` param ship in `openai` (via `openai-agents`), but the credential classes do not. Adding it requires regenerating the Konflux hashed requirements/lockfiles.
- **OpenTelemetry:** `tracing.py` shared runtime and `start_generation_span`; `run_agent.py` invocation span; `audit.py` tool spans and legacy choice events; DeepAgents, OpenAI, and Gemini capture canonical main-agent generations; stdout/OTLP trace exporters exclude the exact `gcp.vertex.agent` instrumentation scope (native ADK spans are not exported by either); `metrics.py` in-process Prometheus histograms (no `/metrics` route).

## Container Build and Regeneration

- **Release inputs:** `.konflux/build.args` selects RHOAI 3.5 CPU base images and hermetic mode. `.konflux/profiles.toml` selects the RHOAI 3.5 CPU package index, Python 3.12, `x86_64`/`aarch64`, extras, and bootstrap packages. Non-hermetic builds use the UBI Python defaults in `Containerfile` and install from `uv.lock`.
- **Independent dependency resolution:** `make requirements` (alias for `make konflux-requirements`) runs `scripts/konflux_resolve.py --profile cpu`. Containerized `uv pip compile` resolves `pyproject.toml` independently of `uv.lock`, preferring the RHOAI index over PyPI. RHOAI-sourced packages use wheels. PyPI packages use wheels when listed in `.konflux/pypi_wheel_only.txt` or when no source distribution exists; otherwise they use source distributions. `.konflux/requirements.overrides.txt` supports optional version overrides; it currently contains no pins.
- **Generated Python outputs:** Shared `.konflux/requirements.hashes.wheel.txt`, `.konflux/requirements.hashes.source.txt`, and `.konflux/requirements.hashes.wheel.pypi.txt` contain hashes covering the configured architectures. `.konflux/requirements.hermetic.txt` contains hashed bootstrap wheels; `.konflux/requirements-build.txt` contains source-build dependencies. Regeneration also patches binary package lists in both `.tekton/` pipelines. Review and commit all generated changes together.
- **Regeneration prerequisites:** Network access, Podman or Docker, access to `quay.io/syedriko/uv:prefer-index`, host Python 3.12+ with `packaging`, and `pybuild-deps` on `PATH`. These are not supplied by the ordinary development setup; see [AGENTS.md](../../../AGENTS.md#konflux-hermetic-builds) for contributor instructions.
- **RPM regeneration:** `make rpm-lockfile` runs `scripts/generate-rpm-lock.sh`, reading the base image from `.konflux/build.args`, packages from `.konflux/rpms.in.yaml`, and subscribed RHEL/RHEL AI repositories from `.konflux/redhat.repo`; it writes `.konflux/rpms.lock.yaml`. It requires Podman or Docker and Red Hat subscription credentials (`ACTIVATION_KEY`, `ORG_ID`).
- **Dependency upgrades and verification:** `make bump-deps` upgrades `uv.lock` and regenerates Konflux requirements. `make verify-hermetic-requirements` checks runtime package coverage against `uv.lock`, accounting for explicit platform/graph and build-only exceptions. Missing packages, stale allowlist entries, and orphans fail; version skew for either RHOAI or PyPI is nonfatal.

## Implementation Notes

- **DeepAgents model routing:** `_resolve_model()` checks `CLAUDE_CODE_USE_VERTEX` and `CLAUDE_CODE_USE_BEDROCK`.
- **OpenAI/Azure adapter (`providers/openai.py`):** branches on `LIGHTSPEED_PROVIDER`. For `azure`, builds `AsyncAzureOpenAI` from `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_VERSION` / deployment; Entra ID mode (per `_resolve_azure()` in `config.py`, `what/configuration.md` rule 9a) reads `client_id`/`tenant_id`/`client_secret` from `/var/run/secrets/llm-credentials/` and passes `azure_ad_token_provider`; API-key mode passes `api_key`. Fail-fast on definitive token-acquisition failure — do not construct or use a broken client.
- **Bedrock credentials (`config.py::_resolve_bedrock`):** [OLS-4092] the Anthropic-on-Bedrock model path (`ChatBedrockConverse` via `deepagents`) is unchanged; only credential resolution grows. Reads `aws_access_key_id` / `aws_secret_access_key` / optional `role_arn` from `/var/run/secrets/llm-credentials/` (`what/configuration.md` rule 9b). With `role_arn`, `botocore` performs STS assume-role and owns short-lived-credential refresh (delegated-token principle, `what/provider-contract.md` rule 38); without it, static keys are used. `boto3`/`botocore` are already present via `langchain-aws` — no new dependency.
- **DeepAgents streaming:** `astream(stream_mode="messages")`.
- **DeepAgents result inspection [PLANNED: OLS-3928]:** Install middleware after artifact offload. Run inspection before result delivery to the model and before `ToolResultEvent` emission. Route each model-visible preview, normal result, error, file read, and search result through it.
- **DeepAgents result boundary:** `inspection/middleware.py` applies the boundary at `awrap_model_call()`, after effective-result transformations. Boundary installation stays independent of `options.tool_output_inspection_enabled`. The adapter constructs classifier resources only when inspection is enabled. The middleware also applies to the general-purpose subagent model path.
- **Model/event separation:** Produce model-facing wrapped message copies or equivalent request-local representations. Preserve original content for `is_passed()` correlation and normalized events. Track sandbox-owned wrapping through internal identity/state, never through external marker text. Preserve message metadata and do not modify stored artifacts.
- **DeepAgents boundary tests:** Offline middleware and adapter tests cover the requirements in `what/provider-contract.md`. They exercise actual model-request interception and normalized events. A formatter-only assertion does not prove that the model receives wrapped content.
- **Classifier integration [PLANNED: OLS-3928]:** Construct the contract-defined isolated invocation from the resolved model and bind its schema through the existing LangChain structured-output interface.
- **Inspection failure [PLANNED: OLS-3928]:** Do not emit a rejected `ToolResultEvent`. Propagate `ToolResultSafetyInspectionFailed` to `batch.py`, which exits nonzero without publishing a Result CR.
- **Gemini bash:** Monkey-patches `run_async` for confirmation and `bash -c` wrapping.
- **MCP Secret headers:** `Secret` sources resolve one mounted Secret value per header. No Secret key selection is performed, and resolved values are used as complete header values.
- **Containerfile:** Multi-stage hermetic build; `oc`/`kubectl` in image for **agent tools** (not Result CR publishing); user `agent`; `catatonit`; batch CMD.
- **Unit tests:** `test_run_agent.py`, `test_deepagents.py`, `test_openai_generation_spans.py`, `test_gemini_telemetry.py`, `test_audit.py`, `test_tracing.py`, `test_batch.py`, `test_ready.py`, `test_publish_results_*.py`, `test_batch_e2e_helpers.py` (harness helpers, no cluster).
- **DeepAgents/OpenAI trace smoke:** [data-collection.md verification](../what/data-collection.md#verification) records the offline producer/wire proof and its limits (no live model APIs, deployed collector/FileExporter, or Dataverse).
- **ADK 2.11 canonical-only exporter smoke:** [data-collection.md verification](../what/data-collection.md#verification) records current span counts/parentage and limits (scripted model, real Runner/tools, in-memory exporter; no live provider API or deployed collector).
- **[PLANNED: OLS-3743] Execution limits:** Parse timeout/max-turn environment values once in `batch.py`; pass the parsed values to `run_agent_query()`. Provider adapters continue to receive maximum turns only through `ProviderQueryOptions`. Preserve timeout as structured internal state through `publish_results/status.py` so Result condition selection never depends on matching summary text.
- **Live batch BDD:** `tests/e2e/` feature files via `scripts/e2e-containers.sh` — see [e2e-testing.md](../what/e2e-testing.md).
- **Live cluster tests:** batch BDD (`tests/e2e/`); shares `run_batch_query` with the e2e harness. See [e2e-testing.md](../what/e2e-testing.md).
