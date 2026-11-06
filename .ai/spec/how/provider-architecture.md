# Architecture: data flow, SDK integration

Audience: AI agents. File paths and symbols allowed here.
Package tree: `AGENTS.md`. Behavioral rules: `what/run-api.md`, `what/provider-contract.md`, `what/configuration.md`, `what/audit-logging.md`.

## DeepAgents Migration Status [PARTIALLY IMPLEMENTED: OLS-4434]

Behavioral authority: [provider-contract.md rules 50–67](../what/provider-contract.md#deepagents-consolidation-partially-implemented-ols-4434). Decision: [0003](../decisions/0003-single-deepagents-harness.md). Anthropic, Vertex/Google, native/custom OpenAI, and Azure routes use DeepAgents and narrow model integrations; Vertex supports only Anthropic and Google models. The legacy OpenAI adapter and factory selection are removed. Azure routing/workflows have offline evidence, not live-authentication acceptance. The native OpenAI boundary is hosting `openai` plus the default endpoint or an HTTPS `api.openai.com` hostname; see [configuration rule 3a](../what/configuration.md) for precedence, normalization, and HTTPS rejection. The Gemini ADK adapter and Google ADK/OpenAI Agents dependency roots are removed from local and Konflux inputs; the Tekton package lists are regenerated. The rules below describe the remaining consolidation target; they are not evidence that all OpenAI routes have migrated or all route-specific parity gates are complete.

1. Resolve generic operator configuration into a provider route/model and native credential settings, independently of the fixed DeepAgents harness. Keep readiness, TLS, input construction, and MCP discovery/admission at their existing boundaries.
2. Construct the route-appropriate LangChain chat model through a narrow model integration. Route-specific API selection, auth, reasoning, shaping capability, and metadata handling belong here; do not grow an Anthropic-only model resolver into multiple full agent implementations.
3. Run one shared DeepAgents adapter behind the existing options/events boundary. Use native DeepAgents shell/filesystem and skills, admitted MCP tool objects, shared inspection/wrapping, and a common LangChain event/callback path. Do not retain ADK/OpenAI Agents tool managers or parallel instrumentation hooks.
4. Shape required structured output using the route-appropriate mechanism, then preserve existing batch Result/status, cancellation, resource cleanup, and trace shutdown behavior. All required limits remain in force.
5. The Vertex/Google ADK provider adapter and factory selection have been removed. The legacy OpenAI adapter, tools, MCP conversion, factory entry, and batch fallback have also been removed. Vertex/OpenAI support is removed. The local manifest/lock and regenerated Konflux dependency/build inputs no longer include the retired OpenAI Agents/Google ADK harnesses. Exact model integrations are recorded below.

Keep model integration responsibilities separate from common harness behavior; no new plugin registry or external proxy is required. Historical telemetry smoke results below do not prove this target. Provider trace changes require the parent collection contract's review/versioning gate.

## Data Flow

1. Startup: `batch.main()` reads `/input/`, resolves `model_family = resolve_model_family()` once, and passes it into `resolve_model_route(model_family=model_family)`. It parses reasoning/MCP configuration (fail-fast on bad env), calls `run_readiness_checks(model_family, model_route)`, initializes tracing when `otel_runtime_enabled()`, admits MCP tools, and constructs the provider. Anthropic, Vertex/Google, and native/custom OpenAI use `DeepAgentsProvider(model_route)` directly and `model_route.model`; Azure now uses that same route-driven startup path, with no legacy credential handoff. The SDK-name dispatch and legacy model-resolution helpers are removed; Vertex/OpenAI fails family resolution before routing, readiness, MCP discovery, or provider creation. [PLANNED: OLS-3743] Startup also requires and validates `LIGHTSPEED_AGENT_TIMEOUT_SECONDS` and `LIGHTSPEED_AGENT_MAX_TURNS` before provider invocation.
2. `run_agent_query()` applies context prefix, passes pre-parsed `mcp_servers` and operator-resolved maximum turns into `ProviderQueryOptions`, and calls `provider.query(...)`. [PLANNED: OLS-3743] The outer agent invocation is bounded by the operator-resolved timeout; timeout returns a structured classification used by Result status assembly.
2a. [PLANNED: OLS-3928] The DeepAgents adapter installs result-inspection middleware around model-visible tool results and errors. The middleware uses the resolved DeepAgents model for isolated classifier calls.
2b. DeepAgents installs model-boundary middleware even when inspection is disabled. Output limits and artifact offload run first. Enabled inspection runs before wrapping. The middleware escapes external closing markers, HTML-escapes tool names, and wraps results before model delivery. The adapter appends the trust instruction to the main-agent and general-purpose-subagent system prompts.
3. `run_agent_query()` starts an INTERNAL `invoke_agent` span beneath the received operator context, with effective prompt/system attributes and exact terminal `ResultEvent.text` before shaping; it uses the configured request model without claiming a provider or response model.
4. `AuditLogger` creates local `execute_tool` INTERNAL children. Existing normalized ProviderEvents still drive `EventLogger`, legacy `gen_ai.choice`/templog projections, metrics, and result handling; choice-event flush order is not transcript chronology.
5. DeepAgents and OpenAI capture accepted main-agent `chat` generations. Vertex Gemini uses the shared DeepAgents/LangChain callback; it does not capture ADK `generate_content` events or ADK call IDs. All reuse captured invocation context without making generation spans current. `invoke_agent` is a child of the received operator context, and each accepted generation and local `execute_tool` span is a direct child of `invoke_agent`. The stdout and OTLP trace exporters exclude only spans whose instrumentation scope name is exactly `gcp.vertex.agent`; they do not rename/project spans or alter other scopes/gates. Existing provider events, developer logs, and templog projections are unchanged.
6. `publish_agent_result()` builds status from agent output, creates Result CR via Kubernetes API (`create_namespaced_custom_object`), replaces status (`replace_namespaced_custom_object_status`).
7. `shutdown_tracer()`; exit 0 on sandbox success (including agent failure), non-zero on infrastructure failure with termination log.

## Key Abstractions

- **Config mapping:** `resolve_model_family()` owns env/credential resolution and returns full `ResolvedModelFamily` metadata (`anthropic`, `google`, or `openai`) along with provider-specific settings and resolved Anthropic credentials (other cloud credentials remain lazy references). `config.py` defines these configuration values and `ModelParameters`; Batch reuses the resolved family for routing and readiness; adapter construction does not read provider env vars.
- **Model routes:** `providers/model_router.py` defines the `ModelCreator` callable alias and `ResolvedModelRoute`. The route optionally carries the model integration's `translate_output_schema` and `bind_structured_output` callables; shared shaping applies translation before binding without provider-specific branches. With no custom binder, it uses the SDK's `with_structured_output` directly. Native OpenAI selects raw-schema mode plus its strict translator; custom/vLLM selects raw-schema mode plus `openai_compatible_structured_output_schema`, preserving schema contents with `strict: false`. `resolve_model_route(model=None, *, model_family=None)` reuses a supplied family or resolves one for standalone callers. Anthropic, Vertex/Google, native/custom OpenAI, and Azure return `ResolvedModelRoute` directly; Vertex/OpenAI cannot resolve to a model family. For the OpenAI family, configuration guarantees hosting is Azure or OpenAI; routing selects the Azure constructor explicitly and the native/compatible OpenAI constructors in the default branch. Azure uses one constructor for both auth modes, with raw schema mode and a lazily bound API-version translator plus the explicit Azure shaping binder. Startup validation stays in readiness. Routing constructs no clients and imports no optional provider SDKs.
- **Adapter construction:** Batch constructs `DeepAgentsProvider(model_route)` directly for every implemented route. The provider constructor imports no optional SDKs and creates no model clients; no provider factory or name dispatch remains.
- **Events:** Normalized `ProviderEvent` union decouples agent layer from vendor streaming models.
- **Options:** `ProviderQueryOptions` is the single bundle passed into every adapter (includes `mcp_servers`, `reasoning_config`).
- **Model resolution:** Every provider requires a resolved model from an explicit override, canonical model mapping, or route-specific legacy model input; no package default is applied. The route model is authoritative in DeepAgents.
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
- **Vertex Gemini model integration:** `langchain_google_genai.ChatGoogleGenerativeAI` is constructed in Vertex mode by `providers/models/gemini.py` and injected through `ResolvedModelRoute`. The model integration is supplied by the `deepagents` extra; the obsolete `gemini`/Google ADK extra is removed.
- **Native OpenAI model integration:** `providers/models/openai.py` constructs `langchain_openai.ChatOpenAI(use_responses_api=True, output_version="responses/v1")` with the resolved API key/base URL and shared sync/async HTTP clients. DeepAgents owns its tools, skills, MCP, inspection/wrapping, and event loop. The native route uses `raw_json_schema` with `native_openai_structured_output_schema` as its translator and the `json_schema` method for separate tool-free shaping; its schema wrapper preserves descriptions and schema dictionaries while applying strict transformations, bypassing the Anthropic Pydantic subset.
- **Compatible OpenAI model integration:** Hosting `openai` with a custom endpoint selects `openai-compatible` and `create_openai_compatible_model()`, constructing `ChatOpenAI(use_responses_api=False, output_version="v1", stream_usage=True)`. The local `CompatibleChatOpenAI` produced by `providers/models/openai.py::_chat_model_with_observed_metadata()` keeps SDK imports lazy and delegates SDK chunk conversion, copies observed server completion IDs into response metadata, and prepends nonempty string `delta.reasoning_content` as standard reasoning blocks while preserving SDK text/tool blocks. This compensates for the installed SDK's omitted compatible fields without parallel capture hooks or fabricated IDs/reasoning; the shared event mapper remains unchanged. Its private conversion override must remain covered when upgrading LangChain. The exact served model, resolved URL/path and actual or placeholder API key are passed explicitly with shared TLS; no configuration env vars are rewritten. Readiness requires a nonblank resolved URL and key. The model maps reasoning effort to `reasoning_effort`, maps verbosity separately, and preserves other options for SDK/API validation; classifier/shaping receive no reasoning. Shared shaping uses the same route with the non-strict translator and `bind_openai_compatible_output`, which binds `response_format` explicitly and retains the SDK parser's raw/parsed/error envelope. This avoids native OpenAI model-name heuristics without overriding the SDK's general structured-output method; classifier function calling and native OpenAI binding remain unchanged. Endpoint schema support remains version/backend-specific; routing does not establish live support.
- **OpenAI/Azure:** native OpenAI and compatible endpoints use `ChatOpenAI`; Azure uses `AzureChatOpenAI` and native `AsyncAzureOpenAI` through the shared DeepAgents route. The `openai` extra declares the native OpenAI SDK directly and includes the shared DeepAgents harness; the retired `openai-agents` dependency is removed.
- **azure.identity (Azure Entra ID):** the existing `azure-identity` optional dependency supplies the native async `ClientSecretCredential` and `get_bearer_token_provider` callback for the shared Azure model constructor. Identity owns acquisition/cache/refresh for `https://cognitiveservices.azure.com/.default`; the integration forwards the callback to native `AsyncAzureOpenAI`. Imports stay lazy, TLS is shared, and model cleanup closes the credential owner.
- **OpenTelemetry:** `tracing.py` shared runtime and `start_generation_span`; `run_agent.py` invocation span; `audit.py` tool spans and legacy choice events; DeepAgents captures main-agent generations for Anthropic, Vertex/Google, and native/custom OpenAI routes (provider `openai` for native/custom OpenAI); the retired OpenAI Agents hooks are removed. Stdout/OTLP trace exporters exclude the exact `gcp.vertex.agent` instrumentation scope; `metrics.py` records in-process Prometheus histograms (no `/metrics` route).

## Model HTTP Attempt Observation [OLS-4192]

- `connection_audit.py` owns frozen model/classifier/destination metadata,
  ContextVar restoration, the per-attempt start/completion latch, safe failure
  categories, scalar-field allowlisting and the dedicated nonpropagating JSON
  logger. `emit()` projects private adapter bookkeeping onto the shared v1
  `outbound.request.started` / `outbound.request.finished` allowlist, omitting
  invalid or unavailable optional values. `tracing.py` supplies resolved run
  identity and attaches/detaches its existing OTLP handler without overwriting
  validated outbound-record attributes. Both outputs honor the existing audit
  gate. Valid captured trace/span IDs, including unsampled context, are reused;
  otherwise both IDs remain absent and `attempt_id` correlates the pair. No
  fallback IDs or audit-only spans are generated. HTTPX, aiohttp and urllib3
  omit `request.body.size`: buffered lengths do not prove complete transmission.
- `providers/audit_support.py::audited_model_class` decorates native `_generate`, `_agenerate`, `_stream`
  and `_astream` methods, retaining explicit `run_manager` signatures. Nested
  fallbacks reuse a logical call, but distinct known call IDs are not merged.
  Streaming activates its captured call scope only while advancing/closing the
  native iterator, not across caller-visible yields; cross-context closure must
  not leak metadata or reset another Context's token. Factories instrument only
  their explicit/lazy model-owned HTTP clients.
- `providers/audit_support.py::instrument_httpx_client` decorates existing default/mounted HTTPX transports and
  byte streams, retaining pools, context entry/exit, TLS/proxy/auth configuration,
  native retry behavior and ownership. Private client transport fields and the
  selected httpcore proxy authority are isolated here and tested against native
  routing. Request streams are never pre-read; HTTPX phase timeouts are not
  misrepresented as an overall timeout.
- `providers/audit_support.py::instrument_aiohttp_session` uses native middleware for each physical attempt,
  before connection work and inside native redirect/retry loops. It retains the
  actual ClientResponse and delegates body reads/iteration/release/close; EOF
  received into an unread buffer is not consumption. Known native total timeouts
  are scoped at request dispatch. Model-only request kwargs also reach Google
  authorized sessions without attaching audit middleware to token requests.
- The Google API client's per-instance lazy session getter reinstalls observation
  for closed/loop-local recreated sessions and SDK-reconstructed request kwargs.
  Do not force HTTPX when the SDK selects aiohttp. No SDK implementation is copied.
- `run_agent_query()` scopes explicit caller run/phase identity even without
  tracing, with ContextVar isolation for concurrent invocations.
  `DeepAgentsProvider.query()` scopes the existing deadline during each iterator
  advance/close; `_query()` retains native graph/tool lifecycle. Classifier/shaping use immutable operation-tagged
  `ModelParameters` copies. `inspection/inspector.py` scopes actual retry identity.
  Existing canonical GenAI callbacks/spans and content gates are unchanged.

Behavioral authority and tests: [audit-logging.md](../what/audit-logging.md#llm-request-metadata-audit-ols-4192).
SDK upgrades must retain the tested private transport/session integration points.

### MCP reuse [OLS-4192]

`connection_audit.py` also scopes frozen `MCPRequestMetadata` through the same call
context and emits shared records with `request.kind=mcp` through the same
`RequestAttempt` lifecycle. Only a specific remote MCP tool contributes
`gen_ai.tool.name`; server labels and tool-call IDs are not public wire fields.
`mcp.py::create_mcp_http_client` delegates to the shared TLS factory and installs
the existing `providers/audit_support.py` HTTPX observer. Canonical admission and
DeepAgents runtime reload scope discovery; `mcp.py::audit_mcp_tool_call` is the
native adapter's tool interceptor and scopes the complete native tool session.
AnyIO background tasks inherit this context, including GET reconnects and DELETE
termination. No request-body parsing, separate transport implementation, SDK
patching or new production module is needed. LLM call interfaces/events remain
unchanged; MCP deliberately does not inherit classifier-only metadata.

Behavioral authority: [MCP request metadata audit](../what/audit-logging.md#mcp-request-metadata-audit-ols-4192).

### Credential transport reuse [OLS-4192]

`AuthRequestMetadata` and `auth_request()` reuse the existing call context and
`RequestAttempt` lifecycle, projected as `request.kind=credential`. Same-provider
nested exchanges reuse their internal logical ID; scope exit restores prior
LLM/MCP identity. Provider identity (`azure`, `google`, `aws`) remains private
adapter bookkeeping; model/classifier labels are omitted from the wire.

`providers/cloud_credentials.py` decorates only owned native credential instances:
Azure token methods scope the native cache/refresh calls and its lazy native
`AioHttpTransport.open()` installs the existing aiohttp observer. Google final
credential `refresh(request)` scopes the supplied native request; ADC loading uses
a lazy native Request delegate to include federation/project discovery and closes
only that loader-owned session without replacing a primary failure. Source
credentials are not decorated before native impersonation copies/scopes them.
Botocore's private session still creates its STS clients; their selected native
connection managers receive observation, and the fetcher retains its native cache.

The urllib3 section of `providers/audit_support.py` shares one observer for Google
requests and botocore. This module groups HTTPX, aiohttp, urllib3 and model-scope
support without changing their native transport behavior or adding compatibility
shim modules.
Per-instance requests adapter/manager hooks retain native mTLS and proxy selection;
manager-selected pools start before native `urlopen`, observe `_make_request` and
proxy preparation before native retries can swallow failures, and decorate native
response reads/streams/close without changing their class or preloading. Stream
exhaustion, not compressed socket EOF or response headers, completes observation.
Typed native error causes map to existing safe categories without inspecting
messages or implicit context. Native pools, retries, auth, TLS and ownership stay
unchanged. These private hooks are covered against `uv.lock`; SDK upgrades and the
separately resolved hermetic runtime still require compatibility validation.

Behavioral authority: [credential metadata audit](../what/audit-logging.md#credential-request-metadata-audit-ols-4192).

## Container Build and Regeneration

- **Release inputs:** `.konflux/build.args` selects RHOAI 3.5 CPU base images and hermetic mode. `.konflux/profiles.toml` selects the RHOAI 3.5 CPU package index, Python 3.12, `x86_64`/`aarch64`, extras, and bootstrap packages. Non-hermetic builds use the UBI Python defaults in `Containerfile` and install from `uv.lock`.
- **Independent dependency resolution:** `make bump` (alias `make konflux-requirements`) runs `scripts/konflux_resolve.py --profile cpu` and refreshes `uv.lock` in the same run. Containerized `uv pip compile` resolves `pyproject.toml` independently of `uv.lock`, preferring the RHOAI index over PyPI. RHOAI-sourced packages use wheels. PyPI packages use wheels when listed in `.konflux/pypi_wheel_only.txt` or when no source distribution exists; otherwise they use source distributions. `.konflux/requirements.overrides.txt` supports optional version overrides.
- **Generated Python outputs:** Shared `.konflux/requirements.hashes.wheel.txt`, `.konflux/requirements.hashes.source.txt`, and `.konflux/requirements.hashes.wheel.pypi.txt` contain hashes covering the configured architectures. `.konflux/requirements.hermetic.txt` contains hashed RHOAI wheels for bootstrap tools and source-build dependencies; `.konflux/requirements-build.txt` contains the PyPI build-time dependencies, index-free. Regeneration also patches binary package lists in both `.tekton/` pipelines. Review and commit all generated changes together.
- **Regeneration prerequisites:** A Linux environment, network access, Podman or Docker, access to `quay.io/syedriko/uv:prefer-index`, and network access to the RHOAI index and PyPI. `make bump` provisions Python 3.12 and the resolver's tools (`packaging`, `tomlkit`, `pybuild-deps`, `pip-tools`, `pip`) via `uv run --no-project`. Build-dependency markers follow the host OS; macOS generation omits Linux-only requirements. See [AGENTS.md](../../../AGENTS.md#konflux-hermetic-builds) for contributor instructions.
- **RPM regeneration:** `make rpm-lockfile` runs `scripts/generate-rpm-lock.sh`, reading the base image from `.konflux/build.args`, packages from `.konflux/rpms.in.yaml`, and subscribed RHEL/RHEL AI repositories from `.konflux/redhat.repo`; it writes `.konflux/rpms.lock.yaml`. It requires Podman or Docker and Red Hat subscription credentials (`ACTIVATION_KEY`, `ORG_ID`).
- **Dependency upgrades and verification:** `make bump` refreshes `uv.lock` and regenerates Konflux requirements together; `make bump PKG=<name> VERSION=<minimum>` targets a single package. `make verify-hermetic-requirements` checks runtime package coverage against `uv.lock`, accounting for explicit platform/graph and build-only exceptions. Missing packages, stale allowlist entries, and orphans fail; version skew for either RHOAI or PyPI is nonfatal.

## Implementation Notes

- **DeepAgents model construction:** `DeepAgentsProvider(model_route)` delegates fresh main/classifier/shaping model creation to `model_route.create_model(model_route.model_parameters, reasoning_config=...)`. Only the main receives query reasoning; classifier/shaping use `None`. Classifier models reuse the resolved model parameters without a classifier-specific output-token override; SDK/provider defaults apply when no limit is configured. Input chunking subtracts instruction overhead from the context estimate without a fixed output reservation; main and shaping limits remain unchanged. Shared shaping rejects SDK parsing errors/missing typed output and validates raw results against the original schema using the explicitly declared `jsonschema`/`referencing` dependencies; local references work, unavailable external references fail without fetching, and rejected payloads are omitted from validation error messages. Telemetry provider/model and shaping method also come from the route. No backend detection or vendor-client construction remains in the adapter; initialized model clients are still closed by it.
- **Anthropic integrations:** `providers/models/anthropic.py` exposes `create_direct_model()`, `create_bedrock_model()`, and `create_vertex_model()`. Each accepts `ModelParameters` and optional reasoning; it preserves native SDK authentication/refresh and shared TLS. Vertex region comes from `LIGHTSPEED_PROVIDER_REGION` and defaults to `global`; readiness validates provider settings and mounted credentials before model construction. Anthropic, Vertex, and Bedrock route resolution does not translate settings into provider-specific environment variables. Constructors do not select hosting from `LIGHTSPEED_*` or `CLAUDE_CODE_USE_*`.
- **Retired OpenAI adapter:** `providers/openai.py`, its legacy tool/MCP helpers, and corresponding tests are removed. No runtime route can select it or use it as a fallback.
- **Azure credential preparation (OLS-4470/OLS-4471):** `cloud_credentials.create_azure_credentials()` accepts the typed source without rereading credentials or mutating environment. API-key mode returns the resolved key without importing identity SDKs. Entra mode returns a native async `ClientSecretCredential` and its `get_bearer_token_provider` callback, with the native OpenAI missing-key sentinel to prevent ambient API-key selection. The Azure Identity request hook passes the shared SSL context to its native aiohttp transport; the SDK creates its session lazily and owns token caching/refresh. The caller must await the result's `aclose()` on success, failure, or cancellation. The shared Azure model constructor consumes this helper through the Azure DeepAgents route.
- **Azure model integration (OLS-4470/OLS-4471):** `providers/models/openai.py::create_azure_model()` builds `AzureChatOpenAI` with explicit native Azure endpoint, deployment, API version, and resolved auth, without consulting ambient SDK credentials or endpoint settings. Startup readiness owns HTTPS endpoint, API-version presence, credential-source type, and nonempty API-key validation on the resolved Azure family; the constructor trusts those validated settings rather than repeating startup checks. Both auth modes share the constructor; Entra uses the integration's native async-only callback support. API versions before `2025-03-01` select Chat Completions; later versions select Responses. The existing Chat stream correction above retains observed IDs/reasoning on Azure too. Version-aware schema preparation preserves the original non-strict Chat schema from `2024-08-01`, uses the native strict translator for Responses, and rejects earlier API versions; explicit binding avoids deployment-name capability guesses. Constructor-owned async resources are kept outside LangChain metadata/serialization and closed by shared model cleanup, including when SDK client closure raises. Offline request/stream/auth/refresh/schema/classifier/cleanup checks live in `tests/test_azure_models.py`. Runtime routing selects this constructor and its shaping helpers for both auth modes. `tests/test_azure_deepagents.py` additionally exercises real SDK/DeepAgents skill/file/shell/admitted-MCP execution, inspection before wrapping, malicious rejection, shaping, exact usage/IDs, and cleanup on success/401/setup failure/task cancellation for both APIs and auth modes. Only external HTTP is scripted. These checks do not establish live Azure acceptance.
- **Bedrock credentials (`config.py::_resolve_bedrock`):** [OLS-4092] the Anthropic-on-Bedrock model path (`ChatAnthropicBedrock` via DeepAgents) is unchanged. It resolves the static key pair and optional session token/role ARN from environment values or mounted credential files. Without a role ARN, the constructor sets the resolved static credentials for boto3's standard chain. With a role ARN, `cloud_credentials.create_bedrock_credentials()` returns botocore-managed refreshable credentials from a private session; the Bedrock model signs requests with the current assumed credentials, and botocore owns STS refresh (delegated-token principle, `what/provider-contract.md` rule 38). `boto3`/`botocore` are already present via `langchain-aws` — no new dependency.
- **DeepAgents skills:** When `has_skills()` succeeds, `providers/deepagents.py` passes `skills=["/skills/"]` to `create_deep_agent()`. A `CompositeBackend` routes `/skills/` to a virtual `FilesystemBackend` rooted at the physical skills directory; `LocalShellBackend` remains the default. DeepAgents' filesystem middleware injects the virtual-to-host path mapping used by the model to invoke native `execute`. Artifacts use the physical `HOME/.deepagents` `artifacts_root`. See [0002-sdk-native-skills-route.md](../decisions/0002-sdk-native-skills-route.md) for route behavior and limits.
- **DeepAgents streaming:** `astream(stream_mode=["messages", "custom"])`. Internal `lightspeed.accepted_tool_results` custom records carry exact-result signatures after an entire model boundary passes. The adapter consumes each record against currently pending normalized results, then discards it; custom records are not provider events, logs, or reusable trust grants. Pending results without an acknowledgement fail closed before later model output or terminal completion.
- **DeepAgents result inspection [PLANNED: OLS-3928]:** Install middleware after artifact offload. Run inspection before result delivery to the model and before `ToolResultEvent` emission. Route each model-visible preview, normal result, error, file read, and search result through it.
- **DeepAgents result boundary:** `inspection/middleware.py` applies the boundary at `awrap_model_call()`, after effective-result transformations. Boundary installation stays independent of `options.tool_output_inspection_enabled`. The adapter constructs classifier resources only when inspection is enabled. The middleware also applies to the general-purpose subagent model path.
- **Model/event separation:** Produce model-facing wrapped message copies or equivalent request-local representations. Preserve original content for exact-signature event-release correlation and normalized events. Cache only actual classifier passes for historical-result deduplication within a query. Re-evaluate source trust at every boundary; trusted-skill exemptions never populate the classifier cache. The middleware publishes boundary-local acceptance notifications only after every result passes, without retaining an acceptance set. Track sandbox-owned wrapping through internal identity/state, never through external marker text. Preserve message metadata and do not modify stored artifacts.
- **DeepAgents boundary tests:** Offline middleware and adapter tests cover the requirements in `what/provider-contract.md`. They exercise actual model-request interception and normalized events. A formatter-only assertion does not prove that the model receives wrapped content.
- **Classifier integration [PLANNED: OLS-3928]:** Construct the contract-defined isolated invocation from the resolved model and bind its schema through the existing LangChain structured-output interface. Validate raw completion/refusal/error signals before accepting parsed decisions or strict JSON text; parsed arguments cannot override a cutoff. Completed native tool-call states remain supported. Rejected attempts log fixed response-issue codes through the existing logger, including attempts followed by a successful retry; final failures retain the issue on the inspection span and warning. The span does not claim a fixed output-token limit. No response/tool payload or arbitrary provider status is logged.
- **Inspection failure [PLANNED: OLS-3928]:** Do not emit a rejected `ToolResultEvent`. Propagate `ToolResultSafetyInspectionFailed` to `batch.py`, which exits nonzero without publishing a Result CR.
- **Vertex credentials:** `providers/cloud_credentials.py` optionally wraps ADC credentials in Google Auth impersonated credentials when the sibling `target_service_account` file is present. Both the Anthropic and Gemini Vertex constructors pass those credentials to their SDK integrations, which own token refresh. The mounted Secret shape, IAM requirements, and interaction with ADC-configured impersonation are defined in [configuration.md](../what/configuration.md) rule 9d.
- **Vertex Gemini construction:** `providers/models/gemini.py` applies Vertex project/region, optional custom base URL, shared TLS, and reasoning config. Its calls use the same DeepAgents tool and event path as other routed models.
- **MCP Secret headers:** `Secret` sources resolve one mounted Secret value per header. No Secret key selection is performed, and resolved values are used as complete header values.
- **Containerfile:** Multi-stage hermetic build; `oc`/`kubectl` in image for **agent tools** (not Result CR publishing); user `agent`; `catatonit`; batch CMD.
- **Unit tests:** `test_model_router.py`, `test_anthropic_models.py`, `test_gemini_models.py`, and `test_openai_models.py` cover typed routes and offline model construction. The OpenAI tests additionally resolve native/custom routes and capture Responses/Chat Completions and schema-shaping requests. Native OpenAI tests also exercise a real DeepAgents file-tool/shaping loop with scripted HTTP/SSE, response IDs, and per-call/aggregate usage. The compatible integration uses real resolution/readiness, direct adapter construction, ChatOpenAI, DeepAgents skill discovery/file reads/shell execution, admitted MCP tool loading/execution, and isolated classification before wrapping; only LLM/MCP HTTP is scripted. It verifies the script's file effect and separate tool-free/reasoning-free shaping of the agent's answer. The compatible loop also checks actual completion IDs, distinct observed response-model identity, finish metadata, per-generation usage, and exact terminal totals with shaping once. Real-SDK invalid/missing/schema-invalid output and agent/shaping HTTP 401 cases exercise `run_agent_query()` failure classification, absence of a terminal output projection, canonical generation ancestry/error metadata, and client closure. These checks do not establish exhaustive failure/cancellation coverage, live-provider support, exported-OTLP reconstruction, or downstream telemetry compatibility. `test_batch.py` verifies single family resolution and route/provider/model propagation across hosting/family paths. Other coverage: `test_run_agent.py`, `test_deepagents.py`, `test_deepagents_generation_spans.py`, `test_audit.py`, `test_tracing.py`, `test_batch.py`, `test_ready.py`, `test_publish_results_*.py`, `test_batch_e2e_helpers.py` (harness helpers, no cluster). The Vertex/Google cases in `test_deepagents.py` and `test_deepagents_generation_spans.py` exercise shared DeepAgents behavior with a fake model and directly constructed route; they do not exercise route resolution or `create_vertex_gemini_model`.
- **Azure workflow tests:** `test_azure_models.py` and `test_azure_deepagents.py` cover native Azure request/auth mapping and actual shared-harness execution. The routed workflow tests exercise flat and nested schemas (including nested objects and array items), field descriptions, API-version-specific strictness, tools/skills/admitted MCP, inspection, main-only reasoning, schema-valid results, usage/events, and cleanup for both auth modes and API-selection branches. They use scripted external HTTP; existing CI batch scenarios own live acceptance.
- **DeepAgents/OpenAI trace smoke:** [data-collection.md verification](../what/data-collection.md#verification) records the offline producer/wire proof and its limits (no live model APIs, deployed collector/FileExporter, or Dataverse).
- **Historical ADK 2.11 exporter smoke:** [data-collection.md verification](../what/data-collection.md#verification) records pre-migration scripted-model evidence; it does not describe or verify the current Vertex Gemini route.
- **[PLANNED: OLS-3743] Execution limits:** Parse timeout/max-turn environment values once in `batch.py`; pass the parsed values to `run_agent_query()`. Provider adapters continue to receive maximum turns only through `ProviderQueryOptions`. Preserve timeout as structured internal state through `publish_results/status.py` so Result condition selection never depends on matching summary text.
- **Live batch BDD:** `tests/e2e/` feature files via `scripts/e2e-containers.sh` — see [e2e-testing.md](../what/e2e-testing.md).
- **Live cluster tests:** batch BDD (`tests/e2e/`); shares `run_batch_query` with the e2e harness. See [e2e-testing.md](../what/e2e-testing.md).
