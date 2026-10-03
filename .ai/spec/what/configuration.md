# Behavioral spec: configuration, environment, deployment

Audience: AI agents (Claude). Precision over narrative.

Cross-references: how options are consumed in code → `how/provider-architecture.md`. Batch input and agent behavior → `run-api.md`. Provider options → `provider-contract.md`.

## Behavioral Rules

1. **Operator env var contract.** The operator sets generic `LIGHTSPEED_*` env vars on the sandbox pod template. The sandbox MUST NOT depend on the operator setting any SDK-specific env vars. The operator sets:

    | Env var | Required | Description |
    | --- | --- | --- |
    | `LIGHTSPEED_PROVIDER` | Yes | Hosting backend: `anthropic`, `vertex`, `openai`, `azure`, `bedrock` |
    | `LIGHTSPEED_MODEL` | Yes | Model identifier (e.g. `claude-sonnet-4-20250514`) |
    | `LIGHTSPEED_MODEL_PROVIDER` | When provider=`vertex` | Model family on Vertex: `Anthropic`, `Google`, `OpenAI` |
    | `LIGHTSPEED_PROVIDER_URL` | When URL set on provider config | Optional API endpoint override |
    | `LIGHTSPEED_PROVIDER_PROJECT` | When provider=`vertex` | Cloud project ID |
    | `LIGHTSPEED_PROVIDER_REGION` | When provider=`vertex` or `bedrock` | Cloud region |
    | `LIGHTSPEED_PROVIDER_API_VERSION` | When provider=`azure` | API version |
    | `LIGHTSPEED_REASONING_CONFIG` | No | JSON-serialized reasoning config from `Agent.spec.reasoningConfig`. When absent, SDK defaults apply. |
    | `LIGHTSPEED_AGENT_TIMEOUT_SECONDS` | Yes [PLANNED: OLS-3743] | Whole-agent invocation budget resolved from the selected Agent step timeout or the operator default. |
    | `LIGHTSPEED_AGENT_MAX_TURNS` | Yes [PLANNED: OLS-3743] | Provider iteration cap resolved from `Agent.spec.maxTurns` or the operator default of 200. |
    | `LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED` | No | Enable DeepAgents tool-result inspection. Default: `true`. |

    Credentials are mounted via `envFrom` (all secret keys as env vars) AND as files at `/var/run/secrets/llm-credentials/`.

    The operator also sets audit, observability, and MCP env vars:

    | Env var | Required | Description |
    | --- | --- | --- |
    | `LIGHTSPEED_AUDIT_ENABLED` | No | When `"true"`, enables compliance stdout and completed-GenAI-span log projections. Default: disabled. It does not gate source OTLP trace export. |
    | `LIGHTSPEED_CAPTURE_CONTENT` | No | Filters the six standard content attributes from compliance copies only; when unset, follows `LIGHTSPEED_AUDIT_ENABLED`. It never filters source spans. The operator does not set this env today. [DEFERRED] Separate CRD field for user-controllable opt-in/out planned per parent spec. |
    | `OTEL_EXPORTER_OTLP_ENDPOINT` | No | Shared base endpoint for source trace and stdlib log export. With `http/protobuf`, exporters append `/v1/traces` and `/v1/logs` to the URL path, preserving any configured base path and query; `grpc` uses the shared endpoint unchanged. When absent, OTLP export is off (stdout compliance JSON still applies when audit is enabled). |
    | `OTEL_EXPORTER_OTLP_PROTOCOL` | No | `grpc` (default) or `http/protobuf`. |
    | `LIGHTSPEED_AGENTICRUN_UID` | No | AgenticRun `metadata.uid` for this sandbox pod. Stamped on source spans and bridged OTLP log record attributes when set; required by collector templog INSERT. Set by operator with the OTEL endpoint. |
    | `LIGHTSPEED_AGENTICRUN_STEP` | No | AgenticRun step/phase for this pod (`analysis`, `execution`, …). Mapped to `agenticrun.phase` on source spans and bridged OTLP log records when set. Set by operator with the OTEL endpoint. |
    | `TRACEPARENT` | No | W3C trace context from the operator phase span. When set, `invoke_agent lightspeed` is its child; model/tool spans are children of the agent span. When absent or invalid, the sandbox starts a new trace. |
    | `LIGHTSPEED_MCP_SERVERS` | No | JSON array of MCP server configs. See rule 20a. When absent, no MCP servers are configured. |
    | `LIGHTSPEED_TLS_PROFILE` | No | Optional resolved OpenShift TLS profile type from the operator handoff. Runtime defaults apply when unset. |
    | `LIGHTSPEED_TLS_MIN_VERSION` | No | Optional resolved minimum TLS version from the operator handoff. Runtime defaults apply when unset. |
    | `LIGHTSPEED_TLS_CIPHER_SUITES` | When the resolved profile has an explicit cipher list | JSON array of resolved cipher-suite names from the operator handoff. |

2. **Provider configuration mapping.** On startup (in `batch.main()`), the sandbox MUST read the generic env vars from rule 1 and set the SDK-specific env vars required by each provider SDK. The mapping logic:

    | `LIGHTSPEED_PROVIDER` | `LIGHTSPEED_MODEL_PROVIDER` | SDK | SDK env vars set |
    | --- | --- | --- | --- |
    | `anthropic` | *(derived)* | `deepagents` | `ANTHROPIC_MODEL`, `ANTHROPIC_BASE_URL` |
    | `vertex` | `anthropic` | `deepagents` | `ANTHROPIC_MODEL`, `CLAUDE_CODE_USE_VERTEX=1`, `ANTHROPIC_VERTEX_PROJECT_ID`, `CLOUD_ML_REGION`, `GOOGLE_APPLICATION_CREDENTIALS`, `ANTHROPIC_BASE_URL` |
    | `vertex` | `google` | `gemini` | `GEMINI_MODEL`, `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` |
    | `vertex` | `openai` | `openai` | `OPENAI_MODEL`, `OPENAI_BASE_URL`, `GOOGLE_APPLICATION_CREDENTIALS` |
    | `openai` | *(derived)* | `openai` | `OPENAI_MODEL` |
    | `azure` | *(derived)* | `openai` | `OPENAI_MODEL`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` (credential handling per rule 9a) |
    | `bedrock` | *(derived)* | `deepagents` | `ANTHROPIC_MODEL`, `CLAUDE_CODE_USE_BEDROCK=1`, `AWS_REGION`, `ANTHROPIC_BASE_URL` (credential handling per rule 9b) |

    `LIGHTSPEED_PROVIDER_URL` MUST be mapped to the SDK-appropriate URL env var when set (e.g. `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`). `LIGHTSPEED_PROVIDER_PROJECT` and `LIGHTSPEED_PROVIDER_REGION` MUST be mapped to the provider-specific project/region vars. Credential files at `/var/run/secrets/llm-credentials/` MUST be referenced via `GOOGLE_APPLICATION_CREDENTIALS` for Vertex providers that require file-based credentials.

3. **Provider selection.** `resolve_sdk()` returns a `ResolvedSDK` whose `name` field selects the backend SDK (`deepagents`, `gemini`, or `openai`). This is determined by the configuration mapping (rule 2), not by the operator. Unknown values are rejected at startup.

4. **Default provider.** When `LIGHTSPEED_PROVIDER` is unset, the provider defaults to `anthropic`, which resolves to SDK name `deepagents`.

5. **Model resolution.** `LIGHTSPEED_MODEL` is the canonical model input. The provider configuration mapping (rule 2) sets the SDK-specific model var (`ANTHROPIC_MODEL`, `GEMINI_MODEL`, or `OPENAI_MODEL`) from `LIGHTSPEED_MODEL`. SDK-specific model vars MAY also be read directly for backward compatibility when `LIGHTSPEED_MODEL` is unset; if all are unset, use the package default model constant.

6. **Model override.** `resolve_router_model(provider_name, model=None)` accepts an explicit `model` string; when provided, it overrides environment-based resolution for that run.

7. **Skills directory.** `LIGHTSPEED_SKILLS_DIR` sets the filesystem root for skills and provider `cwd`. Default when unset is the container default path under `/app`.

8. **[PLANNED: OLS-3743] Agent timeout.** `LIGHTSPEED_AGENT_TIMEOUT_SECONDS` is required and sets the wall-clock budget for the complete `run_agent_query()` invocation. It MUST be a positive integer. Missing, zero, negative, or malformed values fail sandbox startup; the sandbox has no independent default. `LIGHTSPEED_TIMEOUT_MS` is removed without an alias. See `run-api.md` rule 9.

9. **Provider credentials.** API authentication uses the conventional env vars expected by each vendor SDK (Anthropic, Google/Gemini, OpenAI). These are populated from the credentials secret mounted via `envFrom` by the operator, and optionally from the file mount at `/var/run/secrets/llm-credentials/` for file-based credentials. The sandbox configuration mapping (rule 2) sets any additional credential-related env vars (e.g. `GOOGLE_APPLICATION_CREDENTIALS` path).

9a. **Azure OpenAI credential resolution (Entra ID / API key).** For `LIGHTSPEED_PROVIDER=azure`, the sandbox MUST select the authentication mode from the mounted credential files at `/var/run/secrets/llm-credentials/`:

    - **Entra ID (service principal)** when `client_id`, `tenant_id`, **and** `client_secret` files are all present and non-empty. The sandbox MUST NOT set `AZURE_OPENAI_API_KEY` in this mode.
    - **API key** otherwise, when `apitoken` (file) or `AZURE_OPENAI_API_KEY` (`envFrom`) is present and non-empty.
    - When neither a complete Entra ID set nor an API key is available, readiness MUST fail at startup with a descriptive error naming the missing credential set (see `health-probes.md`).

    Both modes MUST use the OpenAI SDK's built-in Azure support — the adapter constructs the SDK's native `AsyncAzureOpenAI` client (see `provider-contract.md` rule 29 and `how/provider-architecture.md`). In Entra ID mode, token minting and refresh are owned by the provider SDK's credential object (`ClientSecretCredential` via `azure_ad_token_provider`), per the short-lived-token principle in `provider-contract.md` rule 38; the sandbox performs no manual token caching. This mirrors the classic OLS service's Azure Entra ID behavior ([OLS-3050]); key names (`client_id`, `tenant_id`, `client_secret`, `apitoken`) match the classic credential-secret shape.
10a. [PLANNED: OLS-3472] **Gemma 4 on RHOAI/vLLM.** A Gemma 4 model served by RHOAI/vLLM MUST use the existing `openai` provider path. `LIGHTSPEED_PROVIDER=openai`, `LIGHTSPEED_MODEL=<the identifier exposed by vLLM>`, and `LIGHTSPEED_PROVIDER_URL=<the cluster-internal OpenAI-compatible API URL>` MUST resolve to the existing OpenAI SDK adapter and `OPENAI_BASE_URL`. The sandbox MUST NOT require a Gemma-specific provider value or SDK adapter.

11. **Anthropic via Vertex.** When `LIGHTSPEED_PROVIDER=vertex` and `LIGHTSPEED_MODEL_PROVIDER=anthropic`, the configuration mapping resolves to SDK name `deepagents` and sets Vertex env vars for `ChatAnthropicVertex`.

9b. **AWS Bedrock credential resolution (static keys / STS assume-role).** For `LIGHTSPEED_PROVIDER=bedrock`, the sandbox resolves AWS credentials from the mounted files at `/var/run/secrets/llm-credentials/` (`aws_access_key_id`, `aws_secret_access_key`, and optional `role_arn`), matching the classic OLS service's IAM credential shape ([OLS-1895]). This does **not** change the existing Bedrock model path (Anthropic models via the `deepagents` SDK / `ChatBedrockConverse`); it governs credentials only.

    - **Static keys** when `aws_access_key_id` and `aws_secret_access_key` are present and non-empty and `role_arn` is absent: use them directly as long-lived credentials (no refresh needed).
    - **STS assume-role (short-lived)** when `role_arn` is also present: the AWS SDK (`botocore` credential-provider chain) performs the assume-role and mints/refreshes the short-lived credentials transparently for the life of the run — the sandbox performs no manual token acquisition, caching, or refresh, per the short-lived-token principle in `provider-contract.md` rule 38.
    - When no usable AWS credential set is present, readiness MUST fail at startup with a descriptive error (see `health-probes.md` rule 2b).

    Token/credential lifecycle is owned by `botocore` (already present via `langchain-aws`); the sandbox adds no AWS credential-handling dependency.

9c. **Anthropic bearer token auth (vLLM, Anthropic-compatible endpoints).** For vLLM or other Anthropic-compatible endpoints that require bearer token auth, set `ANTHROPIC_AUTH_TOKEN`. The adapter sends it as `Authorization: Bearer <token>` via `default_headers`. `ANTHROPIC_API_KEY` must still be set to any non-empty value to pass the sandbox readiness check (rule 9, `check_provider_env`). Otherwise use `ANTHROPIC_API_KEY` alone.

10. **Vertex / Google GenAI.** `GOOGLE_GENAI_USE_VERTEXAI` toggles Vertex behavior for the Gemini adapter (tool composition rules per `provider-contract.md`). Set by the configuration mapping when `LIGHTSPEED_PROVIDER=vertex` and `LIGHTSPEED_MODEL_PROVIDER=Google`.

10a. **Reasoning configuration.** When `LIGHTSPEED_REASONING_CONFIG` is set, the sandbox MUST parse it as a JSON object and make it available to provider adapters via `ProviderQueryOptions.reasoning_config`. When the env var is absent or empty, `reasoning_config` MUST be `None` and adapters MUST use SDK defaults. When the value is present but is not valid JSON or parses to a non-object type (e.g. array, string, number), the sandbox MUST fail at startup with a descriptive error — it MUST NOT silently fall back to `None`. The sandbox MUST NOT validate the object's keys or values — the upstream SDK and model API validate at invocation time. When a run also has structured output (`output-schema` on the batch input), DeepAgents adapter behavior is defined in [provider-contract.md](provider-contract.md) rule 22 (Anthropic thinking vs forced tool choice). This field is aligned with the classic OLS `reasoning_config` model parameter ([OLS-3452]).

 1. **OpenAI base URL.** `OPENAI_BASE_URL` overrides the OpenAI client base URL when set. Mapped from `LIGHTSPEED_PROVIDER_URL` by the configuration mapping for `openai` and `vertex`/`OpenAI` providers.

11a. **OpenAI-compatible endpoints (vLLM, local services, RHOAI, RHEL AI).** ([OLS-3053]) When `OPENAI_BASE_URL` is set to a non-api.openai.com URL (e.g. vLLM, local service, RHOAI/RHEL AI deployment), the OpenAI provider adapts endpoint selection and configuration:
    - **Endpoint selection.** Native OpenAI (api.openai.com or unset) uses `OpenAIResponsesModel` (streaming via `/v1/responses`). Non-native endpoints use `OpenAIChatCompletionsModel` (streaming via `/v1/chat/completions`) to avoid endpoint-specific bugs and ensure compatibility with vLLM and similar strict OpenAI-compatible implementations.
    - **Structured output.** For non-native endpoints, strict JSON-schema mode is disabled; the model produces plain JSON conforming to the schema without OpenAI's strict-mode enforcement at the first token. Native OpenAI continues to use strict mode for schema compliance guarantees.
    - **Configuration via operator.** The LLMProvider CRD `url` field maps to `LIGHTSPEED_PROVIDER_URL` (set by the operator), which the sandbox configuration mapping converts to `OPENAI_BASE_URL` (see `provider-contract.md` rule 29). Credentials are mounted from the credentials secret referenced in the LLMProvider, with keys `api_key` (mapped to `OPENAI_API_KEY`) and optional `model` and `base_url` overrides (mapped to `OPENAI_MODEL`, `OPENAI_BASE_URL`). For vLLM and local deployments that do not require authentication, the secret may contain a placeholder value (e.g. `api_key: "EMPTY"` or a dummy token).
    - **Model support and caveats.** vLLM and other OpenAI-compatible endpoints support tool calling and streaming. Model variants (gpt-3.5-turbo, gpt-4, gpt-4o, or open-weights equivalents) must support function calling to participate in agentic flows. Models without function-calling support will fail at runtime when tools are available. Reasoning configuration (`LIGHTSPEED_REASONING_CONFIG`) is supported on compatible models but may not be available on all vLLM-served models; unsupported reasoning configs fail at API invocation time, not at startup. Structured output validation happens at API time; if a model does not support the requested schema format or produces invalid JSON, the agent fails with a clear error message.
    - **Example: RHOAI vLLM endpoint.** When an organization deploys a vLLM instance on RHOAI serving a tool-capable model (e.g. Granite 3.x, Llama 2 70B), the secret contains the endpoint URL and model, and the LLMProvider references it:
        ```yaml
        apiVersion: agentic.openshift.io/v1alpha1
        kind: LLMProvider
        metadata:
          name: rhoai-vllm
        spec:
          type: OpenAI
          openAI:
            credentialsSecret:
              name: rhoai-vllm-creds  # Secret with OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
        ```
        The credentials secret contains:
        ```
OPENAI_API_KEY: <token-or-placeholder>
        OPENAI_BASE_URL: https://<rhoai-vllm-host>/v1
        OPENAI_MODEL: granite-3-8b-instruct  # or other tool-capable model
        ```
        The sandbox resolves these env vars from the secret, detects `OPENAI_BASE_URL` is non-native, and uses `OpenAIChatCompletionsModel` for request/response handling.

 1. **Anthropic via Vertex.** When `LIGHTSPEED_PROVIDER=vertex` and `LIGHTSPEED_MODEL_PROVIDER=anthropic`, the configuration mapping resolves to SDK name `deepagents` and sets Vertex env vars for `ChatAnthropicVertex`.

 2. **[PLANNED: OLS-3743] Maximum turns.** `LIGHTSPEED_AGENT_MAX_TURNS` is required, parsed as an integer from 1 through 500, and passed to `ProviderQueryOptions.max_turns`. Missing, out-of-range, or malformed values fail sandbox startup. The operator resolves omitted `Agent.spec.maxTurns` to 200; the sandbox does not maintain a second default.

 3. **Process entry.** The container process runs `python -m lightspeed_agentic.batch` under `catatonit` as PID 1. There is no HTTP listener.

 4. **Container filesystem layout.** `/app` is the agent workspace (skills only). Application source lives at `/opt/lightspeed/src/`, outside the agent-visible tree to prevent context pollution. A read-only skills mount path, a writable per-pod workspace path under system temp, and a writable home directory path for the non-root runtime user are provisioned with ownership for that UID. LLM credential files are mounted read-only at `/var/run/secrets/llm-credentials/`.

 5. **Python load path.** Runtime sets process environment so application source under `/opt/lightspeed/src` and installed site-packages are on `PYTHONPATH` as defined in the image.

 6. **Hermetic / Konflux build inputs.** Release images are built with network isolation after prefetch: per-architecture Python requirements files with hashes and RPM lockfile input. The generic binary artifacts lockfile may be empty when binaries are copied from other image stages (e.g. `oc`/`kubectl` from `ose-cli`). Regeneration of Python/RPM artifacts is via project automation commands (see `how/provider-architecture.md`).

 7. **Non-hermetic fallback.** When prefetch directories are absent, the container build recipe may fetch selected binaries from external URLs for developer builds.

 8. **System packages — minimum expectations.** Runtime image includes Bash, Git, OpenShift CLI (`oc`), Kubernetes CLI (`kubectl`), and supporting OS utilities per the container recipe. Ripgrep is not currently installed in the image.

17a. [PLANNED: OLS-3472] **Disconnected runtime.** A released sandbox image running the Gemma 4/vLLM path MUST start and execute a batch step without downloading Python packages, model artifacts, tools, skills, or binaries from an external network. The model is remote to the sandbox and reachable at the cluster-internal URL; model preparation is the serving platform's responsibility.

18. **Non-hermetic fallback.** When prefetch directories are absent, the container build recipe may fetch selected binaries from external URLs for developer builds.
 9. **MCP server configuration.** When `LIGHTSPEED_MCP_SERVERS` is set, the sandbox MUST parse it as a JSON array of MCP server entries. When the value is present but is not valid JSON or parses to a non-array type, the sandbox MUST fail at startup with a descriptive error (sandbox failure path, `run-api.md` rule 23) — it MUST NOT silently continue with no MCP servers. Each entry has the shape `{"name": string, "url": string, "timeout": int | float, "headers": [{"name": string, "source": string, "secretName"?: string}]}`. JSON booleans MUST NOT be accepted as `timeout` (fall back to default). Invalid server entries (wrong type, missing `name`/`url`, non-array `headers`, malformed header objects, unsupported `source` other than `ServiceAccountToken` or `Secret`) MUST fail at startup with a descriptive error — the sandbox MUST NOT skip entries and continue. When `source` is `Secret` and `secretName` is missing, empty, or not a string, the sandbox MUST omit that server (warn) and continue with other server entries. A Secret mount MUST contain exactly one readable non-empty value; missing, empty, unreadable, multiple-value, or path-traversal Secret mounts MUST omit that server (warn). A missing service-account token MAY omit only that header, but the server MUST remain classified as Kubernetes. MCP server names MUST be unique within the array; duplicates are an invalid top-level configuration. After discovery and admission, the sandbox MUST derive provider-facing projections containing connection data and admitted tool names and pass those projections via `ProviderQueryOptions.mcp_servers` (see `provider-contract.md`). When the env var is absent or empty, no MCP servers are configured.

20a. **MCP tool admission and RBAC metadata** [OLS-4059]. The sandbox MUST perform one-shot MCP tool admission after `tools/list` and before exposing tools to the LLM. Invalid top-level MCP configuration MUST fail the complete execution, but a per-server connection, authentication, timeout, or `tools/list` failure MUST be isolated to that server: the sandbox MUST log a safe failure reason, omit the failed server, and continue processing other servers. A failed or undiscovered server MUST never be passed to a provider. If no MCP servers remain after discovery or admission, the sandbox MUST continue without MCP tools. An MCP server is Kubernetes-authenticated when any configured `LIGHTSPEED_MCP_SERVERS[].headers[].source` is `ServiceAccountToken`; the sandbox MUST compute this from the raw configuration before resolving header values and carry the classification into the canonical admission result. Resolved header values MUST NOT be used to infer the authentication class. This classification applies to the entire server. Servers using only `Secret` header sources are outside this Kubernetes RBAC filter and their tools are allowed. For a Kubernetes-authenticated server, a tool with `readOnlyHint=true` and no contradictory destructive indication MAY be admitted without RBAC metadata. This relies on the explicit trust assumption that configured MCP servers classify read-only tools honestly; the annotation is not an API-server security boundary. A non-read-only tool MUST contain structurally valid `_meta["openshift.io/rbac"]` metadata using `rules`, `deriveFromArgs`, or `deriveFromManifest`. Missing, empty, malformed, `noRbac: true`, or `unbounded: true` declarations MUST cause the tool to be filtered. The sandbox validates top-level shape and supported form only; the analysis agent resolves admitted declarations against call arguments and reports standard `PolicyRule`s. There is no oc-IR fallback. Filtered tools MUST NOT be exposed to the LLM. If all tools from a server are filtered, the sandbox MUST remove that server from the provider configuration without automatically failing or escalating the workflow. The same omission behavior applies when a server cannot be reached or its `tools/list` discovery fails. The sandbox MUST emit structured application logs for each filtered tool and removed server, including names, classification, reason, and run/step correlation where available, while excluding tokens, secret values, authorization headers, and complete RBAC metadata payloads. See the workspace-level spec `ols/.ai/spec/what/mcp-tool-rbac.md`.

20b. **Provider MCP tool filtering** [OLS-4059]. The sandbox owns one canonical admission result per server with authentication classification and admitted tool metadata. Before provider invocation, batch derives a provider-facing projection containing only connection data and admitted tool names; authentication classification and RBAC metadata MUST NOT cross the provider boundary. Adapters MAY reconnect or reload SDK tool objects but MUST NOT independently admit or reclassify them. Gemini and OpenAI MUST apply admitted names through SDK-native filtering before model exposure. DeepAgents MUST load MCP tool objects through `MultiServerMCPClient`, filter those objects by admitted names, and pass only admitted tools to `create_deep_agent(tools=...)`. Once a non-empty projection reaches a provider, failure to enforce its filters or initialize its complete admitted server set MUST fail the provider query rather than silently omit an admitted server. A successful DeepAgents reload MAY return fewer tools than admission discovered; the adapter MUST still filter out unadmitted tools. All providers MUST apply the same admission policy from rule 20a; provider-specific mechanisms are implementation details, and the LLM MUST receive only admitted tools.

20c. **Analysis prompt boundary** [OLS-4059]. The sandbox MUST consume the operator-provided system and user prompts from `/input/system-prompt` and `/input/query` without adding an oc-IR fallback or weakening the admitted-tool contract. The agentic operator owns the built-in analysis instructions that explain admitted MCP tool availability and RBAC metadata resolution. For the analysis step only, the sandbox MUST append a separate tagged JSON data block derived from admitted mutating-tool RBAC metadata; it MUST contain no connection details or sandbox-generated behavioral instructions. MCP-derived values MUST remain JSON data and MUST NOT escape the policy block delimiters. The sandbox owns tool discovery, admission, provider enforcement, and runtime exposure; it MUST NOT generate or rewrite the analysis instructions to make a filtered tool available.

 1. **MCP header resolution.** For each header in an MCP server entry, the sandbox MUST resolve the value based on the `source` field:

    | `source` | Resolution |
    | --- | --- |
    | `ServiceAccountToken` | Read the projected service-account token from `/var/run/secrets/kubernetes.io/serviceaccount/token` and format it as `Bearer <token>`. This source makes the entire MCP server Kubernetes-authenticated. |
    | `Secret` | Resolve the named mounted Secret and use its single credential value as the complete header value. One Secret is associated with one header; no Secret key selection is supported. |
    `secretName` MUST identify the Secret used by `Secret` sources. The sandbox MUST read the Secret value as UTF-8 and trim surrounding whitespace, including the trailing newline commonly present in mounted Secret files. It MUST NOT add prefixes such as `Bearer` or otherwise parse or interpret the credential. The resulting non-empty value is used as the complete header value. `Secret` sources do not make a server Kubernetes-authenticated. A Secret mount MUST contain exactly one readable non-empty value. If a Secret cannot be resolved, is missing, empty, unreadable, contains multiple values, or uses a path outside the mount root, the sandbox MUST omit that server and continue with other servers; it MUST not silently choose one of multiple values.

 2. **MCP transport.** The sandbox MUST use Streamable HTTP as the MCP transport when connecting to remote MCP servers. SSE transport (deprecated in MCP spec since 2025-03-26) MUST NOT be used for new connections.

22a. **Tool-result inspection.** Configuration MUST conform to `openshift/ols/.ai/spec/what/tool-result-inspection.md`. `LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED` controls the local DeepAgents middleware.

22b. The value MUST default to `true` when the variable is absent or empty.

22c. The sandbox MUST accept only case-insensitive `true` and `false` values. Another non-empty value MUST fail startup.

22d. A `false` value MUST skip classifier calls and inspection-based termination. Main-model tool-safety instructions remain active.

22e. The value MUST NOT change Gemini ADK or OpenAI Agents behavior.

22f. [PLANNED: OLS-3929] DeepAgents tool-output wrapping and its system-prompt trust instruction MUST remain active regardless of this value. SAFE-02 requires no new environment variable, operator setting, or configurable delimiter format. See `provider-contract.md`, Tool-Output Content Boundary.

### Provider-egress TLS and CA

 1. **Stable CA mount root.** The sandbox MUST treat `/var/run/secrets/lightspeed/tls/` as the common read-only root for operator-provided CA sources. It MUST scan regular `.crt` and `.pem` files below this root, including files from the additional CA ConfigMap and integration CA Secrets. When the additional CA reference is absent, no `additional-ca/` source is expected and the system trust store remains the base trust source.
 2. **Generic runtime code.** Sandbox code MUST NOT contain individual OTEL, MCP, RHOKP, or additional-CA Secret names, source-specific CA filenames, or per-source CA selection logic. Those names belong only to the operator handoff and PodSpec mount layers.
 3. **Combined runtime bundle.** At startup, the sandbox MUST combine valid mounted certificates with the platform/system trust store and configure Python and provider/runtime TLS code to use the resulting bundle. It MUST NOT replace or mutate the system trust store or pass individual mounted CA paths to provider clients.
 4. **Certificate failures.** Malformed required certificate material MUST fail sandbox startup or connection configuration with a descriptive error. Files with unrelated extensions are ignored.
 5. **Resolved TLS settings.** The sandbox MUST consume `LIGHTSPEED_TLS_PROFILE`, `LIGHTSPEED_TLS_MIN_VERSION`, and `LIGHTSPEED_TLS_CIPHER_SUITES` passed by the agentic operator. The classic operator is expected to resolve these values even when the user does not configure a profile. If the values are absent, the sandbox preserves existing runtime/provider defaults. It MUST apply supplied values using native runtime/provider behavior and MUST NOT infer OpenShift defaults or translate cipher names.
 6. **Separate Secret classes.** Provider credentials, client certificates/keys, and MCP authentication Secrets are not CA-bundle inputs and remain separate when required by their protocols.
 7. **No per-server trust selection.** Per-MCP-server CA configuration and trust selection are not implemented. OLS-3857 is out of scope.

## Configuration Surface

| Variable / field | Role |
| ------------------ | ------ |
| `LIGHTSPEED_PROVIDER` | Hosting backend from operator (see rule 1). Replaces direct SDK selection. |
| `LIGHTSPEED_MODEL` | Model identifier from operator (see rule 1). |
| `LIGHTSPEED_MODEL_PROVIDER` | Model family on Vertex from operator (see rule 1). |
| `LIGHTSPEED_PROVIDER_URL` | Optional API endpoint override from operator (see rule 1). |
| `LIGHTSPEED_PROVIDER_PROJECT` | Cloud project ID from operator (see rule 1). |
| `LIGHTSPEED_PROVIDER_REGION` | Cloud region from operator (see rule 1). |
| `LIGHTSPEED_PROVIDER_API_VERSION` | API version from operator (see rule 1). |
| `GEMINI_MODEL`, `OPENAI_MODEL` | Internal: SDK-specific model vars. Set by configuration mapping (rule 2), not operator. |
| `LIGHTSPEED_SKILLS_DIR` | Skill root and provider working directory default. |
| `GOOGLE_API_KEY`, `GEMINI_API_KEY` | Google GenAI credential (from credentials secret envFrom). |
| `OPENAI_API_KEY` | OpenAI SDK credential (from credentials secret envFrom). |
| `ANTHROPIC_AUTH_TOKEN` | Bearer token for Anthropic-compatible endpoints (vLLM, custom services). Sent as `Authorization: Bearer <token>` header (rule 9c). `ANTHROPIC_API_KEY` must also be set to any non-empty value. |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API-key credential (from credentials secret envFrom). Used only in API-key mode; omitted in Entra ID mode (rule 9a). |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` | Internal: Azure client config. Set by configuration mapping (rule 2). |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Bedrock IAM credential (from credentials secret envFrom or `/var/run/secrets/llm-credentials/`; rule 9b). |
| `AWS_REGION` | Internal: Bedrock region. Set by configuration mapping (rule 2). |
| `/var/run/secrets/llm-credentials/{aws_access_key_id,aws_secret_access_key,role_arn}` | Bedrock IAM files; `role_arn` (optional) selects STS assume-role, refreshed by botocore (rule 9b). Mounted by operator. |
| `GOOGLE_GENAI_USE_VERTEXAI` | Internal: Vertex mode for Gemini adapter. Set by configuration mapping. |
| `OPENAI_BASE_URL` | Internal: OpenAI-compatible endpoint. Set by configuration mapping. |
| `LIGHTSPEED_AUDIT_ENABLED` | Compliance stdout and completed-GenAI-span log projection toggle. Set by operator from `AgenticOLSConfig`; source trace export remains endpoint-controlled. |
| `LIGHTSPEED_CAPTURE_CONTENT` | Filters standard content fields from compliance copies only; when unset, follows audit enablement. It does not filter source spans. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Shared base endpoint for source span and stdlib log export. With `http/protobuf`, exporters append `/v1/traces` and `/v1/logs` to the URL path, preserving any configured base path and query; `grpc` uses the shared endpoint unchanged. Set by operator from `AgenticOLSConfig`. |
| `LIGHTSPEED_AGENTICRUN_UID` | AgenticRun UID on source span and bridged OTLP log-record attributes when set. |
| `LIGHTSPEED_AGENTICRUN_STEP` | AgenticRun step → `agenticrun.phase` on source spans and bridged OTLP log records when set. |
| `LIGHTSPEED_MCP_SERVERS` | JSON array of MCP server configs with URLs, timeouts, and header sources. Set by operator from `ToolsSpec.mcpServers` and auto-injected defaults. |
| `LIGHTSPEED_TLS_PROFILE` | Resolved OpenShift TLS profile type from the operator handoff. |
| `LIGHTSPEED_TLS_MIN_VERSION` | Resolved minimum TLS version from the operator handoff. |
| `LIGHTSPEED_TLS_CIPHER_SUITES` | JSON array of resolved cipher suites from the operator handoff. |
| `/var/run/secrets/lightspeed/tls/` | Common read-only root for additional and integration CA files. |
| `LIGHTSPEED_REASONING_CONFIG` | JSON reasoning config from operator. Parsed at startup, passed to adapters via `ProviderQueryOptions`. |
| `/var/run/secrets/llm-credentials/` | LLM credential files mounted by operator (unconditional). |
| `/var/run/secrets/kubernetes.io/serviceaccount/token` | Projected SA token for MCP `ServiceAccountToken` header resolution. |
| `/var/secrets/mcp/<secretName>` or `/var/secrets/mcp/<secretName>/` | MCP header Secret value mounted by the operator for `Secret`-sourced headers; the runtime supports a direct single-value file or a directory containing the single value. |
| `LIGHTSPEED_AGENT_TIMEOUT_SECONDS` | [PLANNED: OLS-3743] Required whole-agent invocation timeout from the operator. |
| `LIGHTSPEED_AGENT_MAX_TURNS` | [PLANNED: OLS-3743] Required provider iteration cap from the operator. |
| `LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED` | DeepAgents tool-result inspection; defaults to `true`. |
| `resolve_router_model()`, `resolve_startup_model()` | Model resolution from env (see `config.py`). |

## Constraints

- Input files and env vars carry query, schema, and context; provider name and model are environment-driven. [PLANNED: OLS-3743] Agent timeout and maximum turns are required environment values resolved by the operator, not sandbox defaults.
- Optional Python extras gate which provider SDKs are installed in a given environment; the image recipe installs all extras.
- Bedrock resolves to SDK name `deepagents` via `ChatAnthropicBedrock`. When Bedrock support for other model families is needed, a `modelProvider` field should be added to the `AWSBedrockConfig` CRD (similar to `googleCloudVertex.modelProvider`).
- [PLANNED: OLS-3472] Gemma 4 compatibility is scoped to a vLLM OpenAI-compatible API. The sandbox does not install RHOAI, provision GPUs, or load model weights.

## Verification

- Unit: [test_config.py](../../../tests/test_config.py), [test_model_resolution.py](../../../tests/test_model_resolution.py) — env mapping, model resolution, reasoning/MCP parse errors
- [test_mcp.py](../../../tests/test_mcp.py) covers raw `source` classification, read-only admission, valid and invalid RBAC metadata, `noRbac`/`unbounded` filtering, server removal when no tools remain, canonical projections, policy rendering, and Gemini/OpenAI native filters.
- [test_deepagents.py](../../../tests/test_deepagents.py), [test_openai_schema.py](../../../tests/test_openai_schema.py), and [test_openai_vllm_structured_output.py](../../../tests/unit/providers/test_openai_vllm_structured_output.py) verify admitted-name filtering and fail-closed provider materialization.
- [test_mcp.py](../../../tests/test_mcp.py) verifies filtered-tool and removed-server logs include server/tool names, authentication classification, and safe reason codes without exception details, descriptions, credentials, or RBAC payloads.
- Live batch: [mcp.feature](../../../tests/e2e/features/mcp.feature) (`LIGHTSPEED_MCP_SERVERS`), [reasoning_config.feature](../../../tests/e2e/features/reasoning_config.feature) (`LIGHTSPEED_REASONING_CONFIG`)
- [PLANNED: OLS-3743] Unit tests cover required timeout/max-turn parsing, invalid values, and propagation into `run_agent_query()`.
- Controller-reported integrated focused offline suite: 295 passed. The compliance-copy/source-span behavior is covered by [test_tracing.py](../../../tests/test_tracing.py) and [test_logging.py](../../../tests/test_logging.py); the existing configuration and provider tests above retain their narrower env-mapping and selection coverage.
- The real local HTTP OTLP proof exercised the shared base endpoint and trace/log signal paths. A receiver regression also covers a base path with a query and verifies exact per-signal request targets plus decoded protobuf delivery/correlation; offline native SDK and HTTP evidence is detailed in [audit-logging.md](audit-logging.md), Verification. No live cluster was exercised; the batch BDD links above require a configured cluster.
- Controller-reported final repository checks: `make verify` passed (hermetic requirements alignment, Ruff formatting/lint, mypy on 33 source files); `make test` passed all 669 tests with 15 warnings. The GNU patch workaround and warning details are recorded in [audit-logging.md](audit-logging.md), Verification. No live cluster was exercised.

## Planned Changes

- OLS-3857 per-MCP-server trust selection remains out of scope.
- Konflux pipeline and lockfile policy updates as Red Hat platform requirements evolve. [PLANNED: OLS-2894]
- [PLANNED: OLS-3743] Require operator-resolved `LIGHTSPEED_AGENT_TIMEOUT_SECONDS` and `LIGHTSPEED_AGENT_MAX_TURNS`; remove the sandbox-owned timeout and turn defaults.
