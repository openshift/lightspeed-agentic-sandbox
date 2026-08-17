# Behavioral spec: configuration, environment, deployment

Audience: AI agents (Claude). Precision over narrative.

Cross-references: how options are consumed in code → `how/provider-architecture.md`. HTTP fields → `run-api.md`. Provider options → `provider-contract.md`.

## Behavioral Rules

1. **Operator env var contract.** The operator sets generic `LIGHTSPEED_*` env vars on the sandbox pod template. The sandbox MUST NOT depend on the operator setting any SDK-specific env vars. The operator sets:

    | Env var | Required | Description |
    |---|---|---|
    | `LIGHTSPEED_PROVIDER` | Yes | Hosting backend: `anthropic`, `vertex`, `openai`, `azure`, `bedrock` |
    | `LIGHTSPEED_MODEL` | Yes | Model identifier (e.g. `claude-sonnet-4-20250514`) |
    | `LIGHTSPEED_MODEL_PROVIDER` | When provider=`vertex` | Model family on Vertex: `Anthropic`, `Google`, `OpenAI` |
    | `LIGHTSPEED_PROVIDER_URL` | When URL set on provider config | Optional API endpoint override |
    | `LIGHTSPEED_PROVIDER_PROJECT` | When provider=`vertex` | Cloud project ID |
    | `LIGHTSPEED_PROVIDER_REGION` | When provider=`vertex` or `bedrock` | Cloud region |
    | `LIGHTSPEED_PROVIDER_API_VERSION` | When provider=`azure` | API version |
    | `LIGHTSPEED_REASONING_CONFIG` | No | JSON-serialized reasoning config from `Agent.spec.reasoningConfig`. When absent, SDK defaults apply. |

    Credentials are mounted via `envFrom` (all secret keys as env vars) AND as files at `/var/run/secrets/llm-credentials/`.

    The operator also sets audit, observability, and MCP env vars:

    | Env var | Required | Description |
    |---|---|---|
    | `LIGHTSPEED_AUDIT_ENABLED` | No | When `"true"`, structured audit event logging is enabled. Default: disabled. |
    | `LIGHTSPEED_CAPTURE_CONTENT` | No | When `"true"`, `gen_ai.completion` and `gen_ai.reasoning_content` attributes are recorded on `gen_ai.choice` span events. Currently hardcoded to `"true"` by the operator when audit is enabled. [DEFERRED] Separate CRD field for user-controllable opt-in/out planned per parent spec. |
    | `OTEL_EXPORTER_OTLP_ENDPOINT` | No | Shared OTLP collector endpoint for span **and** log export. When absent, OTLP export is off (stdout audit JSON still applies when audit is enabled). |
    | `OTEL_EXPORTER_OTLP_CERTIFICATE` | No | Optional path to CA cert for the collector (set by operator when mTLS/TLS is configured). |
    | `OTEL_EXPORTER_OTLP_PROTOCOL` | No | `grpc` (default) or `http/protobuf`. |
    | `LIGHTSPEED_AGENTICRUN_UID` | No | AgenticRun `metadata.uid` for this sandbox pod. Stamped on bridged OTLP log **record** attributes. Required by collector templog INSERT. Set by operator with the OTEL endpoint. |
    | `LIGHTSPEED_AGENTICRUN_STEP` | No | AgenticRun step/phase for this pod (`analysis`, `execution`, …). Mapped to `agenticrun.phase` on bridged OTLP log records. Set by operator with the OTEL endpoint. |
    | `LIGHTSPEED_MCP_SERVERS` | No | JSON array of MCP server configs. See rule 20. When absent, no MCP servers are configured. |

2. **Provider configuration mapping.** On startup, the sandbox MUST read the generic env vars from rule 1 and set the SDK-specific env vars required by each provider SDK. This mapping runs before the FastAPI app starts. The mapping logic:

    | `LIGHTSPEED_PROVIDER` | `LIGHTSPEED_MODEL_PROVIDER` | SDK | SDK env vars set |
    |---|---|---|---|
    | `anthropic` | *(derived)* | `deepagents` | `ANTHROPIC_MODEL`, `ANTHROPIC_BASE_URL` |
    | `vertex` | `anthropic` | `deepagents` | `ANTHROPIC_MODEL`, `CLAUDE_CODE_USE_VERTEX=1`, `ANTHROPIC_VERTEX_PROJECT_ID`, `CLOUD_ML_REGION`, `GOOGLE_APPLICATION_CREDENTIALS`, `ANTHROPIC_BASE_URL` |
    | `vertex` | `google` | `gemini` | `GEMINI_MODEL`, `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` |
    | `vertex` | `openai` | `openai` | `OPENAI_MODEL`, `OPENAI_BASE_URL`, `GOOGLE_APPLICATION_CREDENTIALS` |
    | `openai` | *(derived)* | `openai` | `OPENAI_MODEL` |
    | `azure` | *(derived)* | `openai` | `OPENAI_MODEL`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` |
    | `bedrock` | *(derived)* | `deepagents` | `ANTHROPIC_MODEL`, `CLAUDE_CODE_USE_BEDROCK=1`, `AWS_REGION`, `ANTHROPIC_BASE_URL` |

    `LIGHTSPEED_PROVIDER_URL` MUST be mapped to the SDK-appropriate URL env var when set (e.g. `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`). `LIGHTSPEED_PROVIDER_PROJECT` and `LIGHTSPEED_PROVIDER_REGION` MUST be mapped to the provider-specific project/region vars. Credential files at `/var/run/secrets/llm-credentials/` MUST be referenced via `GOOGLE_APPLICATION_CREDENTIALS` for Vertex providers that require file-based credentials.

3. **Provider selection.** `resolve_sdk()` returns a `ResolvedSDK` whose `name` field selects the backend SDK (`deepagents`, `gemini`, or `openai`). This is determined by the configuration mapping (rule 2), not by the operator. Unknown values are rejected at startup.

4. **Default provider.** When `LIGHTSPEED_PROVIDER` is unset, the provider defaults to `anthropic`, which resolves to SDK name `deepagents`.

5. **Model resolution.** `LIGHTSPEED_MODEL` is the canonical model input. The provider configuration mapping (rule 2) sets the SDK-specific model var (`ANTHROPIC_MODEL`, `GEMINI_MODEL`, or `OPENAI_MODEL`) from `LIGHTSPEED_MODEL`. SDK-specific model vars MAY also be read directly for backward compatibility when `LIGHTSPEED_MODEL` is unset; if all are unset, use the package default model constant.

6. **Router override.** Callers of the library `build_router` may pass an explicit `model` string; when provided, it overrides environment-based resolution for that router instance.

7. **Skills directory.** `LIGHTSPEED_SKILLS_DIR` sets the filesystem root for skills and provider `cwd`. Default when unset is the container default path under `/app`.

8. **Provider credentials.** API authentication uses the conventional env vars expected by each vendor SDK (Anthropic, Google/Gemini, OpenAI). These are populated from the credentials secret mounted via `envFrom` by the operator, and optionally from the file mount at `/var/run/secrets/llm-credentials/` for file-based credentials. The sandbox configuration mapping (rule 2) sets any additional credential-related env vars (e.g. `GOOGLE_APPLICATION_CREDENTIALS` path).

9. **Vertex / Google GenAI.** `GOOGLE_GENAI_USE_VERTEXAI` toggles Vertex behavior for the Gemini adapter (tool composition rules per `provider-contract.md`). Set by the configuration mapping when `LIGHTSPEED_PROVIDER=vertex` and `LIGHTSPEED_MODEL_PROVIDER=Google`.

9a. **Reasoning configuration.** When `LIGHTSPEED_REASONING_CONFIG` is set, the sandbox MUST parse it as a JSON object and make it available to provider adapters via `ProviderQueryOptions.reasoning_config`. When the env var is absent or empty, `reasoning_config` MUST be `None` and adapters MUST use SDK defaults. When the value is present but is not valid JSON or parses to a non-object type (e.g. array, string, number), the sandbox MUST fail at startup with a descriptive error — it MUST NOT silently fall back to `None`. The sandbox MUST NOT validate the object's keys or values — the upstream SDK and model API validate at invocation time. This field is aligned with the classic OLS `reasoning_config` model parameter ([OLS-3452]).

10. **OpenAI base URL.** `OPENAI_BASE_URL` overrides the OpenAI client base URL when set. Mapped from `LIGHTSPEED_PROVIDER_URL` by the configuration mapping for `openai` and `vertex`/`OpenAI` providers.

11. **Anthropic via Vertex.** When `LIGHTSPEED_PROVIDER=vertex` and `LIGHTSPEED_MODEL_PROVIDER=anthropic`, the configuration mapping resolves to SDK name `deepagents` and sets Vertex env vars for `ChatAnthropicVertex`.

12. **Router defaults — `max_turns`.** The router supplies a built-in default maximum turn count to provider options when routes are registered (not exposed on `RunRequest`).

13. **Router defaults — `default_timeout_ms`.** The router supplies a built-in default milliseconds timeout for the run handler when `RunRequest.timeout_ms` is null.

14. **Process entry.** The container process invokes Uvicorn serving the FastAPI app on TCP port `8080` on all interfaces.

15. **Container filesystem layout.** `/app` is the agent workspace (skills only). Application source lives at `/opt/lightspeed/src/`, outside the agent-visible tree to prevent context pollution. A read-only skills mount path, a writable per-pod workspace path under system temp, and a writable home directory path for the non-root runtime user are provisioned with ownership for that UID. LLM credential files are mounted read-only at `/var/run/secrets/llm-credentials/`.

16. **Python load path.** Runtime sets process environment so application source under `/opt/lightspeed/src` and installed site-packages are on `PYTHONPATH` as defined in the image.

17. **Hermetic / Konflux build inputs.** Release images are built with network isolation after prefetch: per-architecture Python requirements files with hashes and RPM lockfile input. The generic binary artifacts lockfile may be empty when binaries are copied from other image stages (e.g. `oc`/`kubectl` from `ose-cli`). Regeneration of Python/RPM artifacts is via project automation commands (see `how/provider-architecture.md`).

18. **Non-hermetic fallback.** When prefetch directories are absent, the container build recipe may fetch selected binaries from external URLs for developer builds.

19. **System packages — minimum expectations.** Runtime image includes Bash, Git, OpenShift CLI (`oc`), Kubernetes CLI (`kubectl`), and supporting OS utilities per the container recipe. Ripgrep is not currently installed in the image.

20. **MCP server configuration.** When `LIGHTSPEED_MCP_SERVERS` is set, the sandbox MUST parse it as a JSON array of MCP server entries. Each entry has the shape `{"name": string, "url": string, "timeout": int | float, "headers": [{"name": string, "source": string, "secretName"?: string}], "caFile"?: string}`. JSON booleans MUST NOT be accepted as `timeout` (fall back to default). When `source` is `Secret` and `secretName` is missing, empty, or not a string, the sandbox MUST skip that header (warn) and continue — it MUST NOT reject the entire server entry. When `caFile` is present, it MUST be a non-empty absolute path to a readable PEM CA file; invalid CA configuration MUST skip the entire server rather than silently use system trust. The sandbox MUST build SDK-native MCP client configs from this array and pass them into provider adapters via `ProviderQueryOptions.mcp_servers` (see `provider-contract.md`). When the env var is absent or empty, no MCP servers are configured.

21. **MCP header resolution.** For each header in an MCP server entry, the sandbox MUST resolve the value based on the `source` field:

    | `source` | Resolution |
    |---|---|
    | `ServiceAccountToken` | Read the projected SA token from `/var/run/secrets/kubernetes.io/serviceaccount/token` and format as `Bearer <token>`. |
    | `Secret` | List files under `/var/secrets/mcp/<secretName>/`, sort by name, and read the first regular file. If the directory is missing, empty, unreadable, or `secretName` is invalid, skip the header (warn). Path traversal outside the mount root MUST be rejected. |
    | `Client` | Skip — not resolved by the sandbox. Reserved for future client-passthrough flows. |

    Note: A stricter deterministic path (`.../<secretName>/<secretName>`) and reject-on-missing-`secretName` were written into this spec via review-only commit `1508589` without code or a follow-up ticket (orphan promise). Current behavior is first-file as above.

22. **MCP transport.** The sandbox MUST use Streamable HTTP as the MCP transport when connecting to remote MCP servers. SSE transport (deprecated in MCP spec since 2025-03-26) MUST NOT be used for new connections.

23. **MCP server CA trust.** When an MCP server has `caFile`, the sandbox MUST create a server-specific TLS context that loads normal system trust roots and then adds the configured CA. The same verified context MUST be supplied through each provider SDK's native HTTP client factory. The sandbox MUST NOT disable certificate or hostname verification, globally replace public trust roots, or log CA contents.

## Configuration Surface

| Variable / field | Role |
|------------------|------|
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
| `GOOGLE_GENAI_USE_VERTEXAI` | Internal: Vertex mode for Gemini adapter. Set by configuration mapping. |
| `OPENAI_BASE_URL` | Internal: OpenAI-compatible endpoint. Set by configuration mapping. |
| `LIGHTSPEED_AUDIT_ENABLED` | Audit event logging toggle. Set by operator from `AgenticOLSConfig`. |
| `LIGHTSPEED_CAPTURE_CONTENT` | Content capture toggle for `gen_ai.completion`/`gen_ai.reasoning_content` on choice events. Set by operator from `AgenticOLSConfig`. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Shared OTLP endpoint for span and log export. Set by operator from `AgenticOLSConfig`. |
| `LIGHTSPEED_AGENTICRUN_UID` | AgenticRun UID on bridged OTLP log record attrs (templog). Set by operator with OTEL endpoint. |
| `LIGHTSPEED_AGENTICRUN_STEP` | AgenticRun step → `agenticrun.phase` on bridged OTLP log records. Set by operator with OTEL endpoint. |
| `LIGHTSPEED_MCP_SERVERS` | JSON array of MCP server configs with URLs, timeouts, header sources, and optional per-server `caFile`. Set by operator from `ToolsSpec.mcpServers` and auto-injected defaults. |
| `LIGHTSPEED_REASONING_CONFIG` | JSON reasoning config from operator. Parsed at startup, passed to adapters via `ProviderQueryOptions`. |
| `/var/run/secrets/llm-credentials/` | LLM credential files mounted by operator (unconditional). |
| `/var/run/secrets/kubernetes.io/serviceaccount/token` | Projected SA token for MCP `ServiceAccountToken` header resolution. |
| `/var/secrets/mcp/<secretName>/` | MCP header secret files mounted by operator for `Secret`-sourced headers. |
| `build_router(..., skills_dir=..., model=..., max_turns=..., default_timeout_ms=...)` | Library-level defaults when embedding the router. |

## Constraints

- `RunRequest` does not carry provider name, model, max turns, or budget; changing those requires env vars, router constructor args, or future API extensions.
- Optional Python extras gate which provider SDKs are installed in a given environment; the image recipe installs all extras.
- Bedrock resolves to SDK name `deepagents` via `ChatAnthropicBedrock`. When Bedrock support for other model families is needed, a `modelProvider` field should be added to the `AWSBedrockConfig` CRD (similar to `googleCloudVertex.modelProvider`).

## Planned Changes

- TLS termination, mTLS, and network policies for operator-to-sandbox traffic. [PLANNED: OLS-3038–OLS-3043]
- Konflux pipeline and lockfile policy updates as Red Hat platform requirements evolve. [PLANNED: OLS-2894]
- `Client` header source type resolution when client-passthrough MCP auth flows are implemented.
