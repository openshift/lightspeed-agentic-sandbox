# DeepAgents execution consolidation — OLS-4434

- **Status:** Proposed spec design; not approved or implemented by this document.
- **Tracking:** [OLS-4434](https://redhat.atlassian.net/browse/OLS-4434).
- **Evidence:** Grounded in the current repository implementation and the supplied migration brief. No live-model parity is claimed.
- **Decision:** [0002-single-deepagents-runtime.md](../decisions/0002-single-deepagents-runtime.md).

## Purpose and scope

Consolidate agent execution under DeepAgents while removing the Google ADK and
OpenAI Agents **agent SDKs**. Retain provider-specific **model clients** and their
necessary authentication/transport libraries. One agent runtime does not mean one
model API, one authentication mechanism, or identical model capabilities.

This document specifies the proposed target and its acceptance gates, not an
implementation plan. All target requirements below are **[PLANNED: OLS-4434]**.
Existing rules in [provider-contract.md](provider-contract.md),
[configuration.md](configuration.md), and [audit-logging.md](audit-logging.md)
continue to describe the current runtime until migration is implemented. Their
ADK/OpenAI Agents mechanisms are replaced by the target mechanisms here only as
part of that migration; their SDK-independent integration and security contracts
remain in force.

In scope: model routing, shared execution, reasoning/output handling, tools and
skills, safety, credentials/TLS, normalized events, telemetry, dependency removal,
and verification. Out of scope: changing operator CRDs, workflow/approval semantics,
Result CR schemas, deploying model servers, new model families, Classic OLS runtime
changes, and a second agent execution engine as a fallback.

## Current baseline

- `providers/deepagents.py::_resolve_model()` constructs `ChatAnthropic`,
  `ChatAnthropicVertex`, or `ChatAnthropicBedrock`. These routes already use
  DeepAgents; that fact is not evidence of new cross-provider parity.
- Vertex Gemini resolves to the Google ADK adapter. Native OpenAI, compatible
  endpoints, and Azure resolve to the OpenAI Agents adapter.
- Native OpenAI uses Responses; custom OpenAI-compatible endpoints use Chat
  Completions. Azure selects Responses for parsed API-version dates on or after
  `2025-03-01`, otherwise Chat Completions. This is the sandbox's current selection
  policy, not a guarantee that every Azure deployment/version supports a feature.
- Vertex/OpenAI is declared by the operator and mapped by sandbox configuration.
  Those declarations do not establish a working endpoint or OAuth refresh path.
- The Classic OLS service's `google_vertex.py` demonstrates the
  `ChatGoogleGenerativeAI(vertexai=True)` model-client pattern, not its compatibility
  with this DeepAgents execution flow.

## Target architecture

```text
operator environment + credential files + TLS configuration
    -> configuration/readiness and model-route resolution
    -> configured LangChain chat model instance
    -> shared DeepAgents agent + admitted MCP tools + local backend + middleware
    -> normalized ProviderEvent / ResultEvent
    -> existing run processing, audit/telemetry, and Result CR publication
```

1. **Single runtime.** Every established model route MUST use the same DeepAgents
   execution path. Provider selection identifies the model host/family and client
   configuration, not a different agent loop. No ADK Runner, OpenAI Agents Runner,
   SandboxAgent, or SDK-specific tool executor remains in the migrated runtime.
2. **Explicit models.** Every route MUST pass a configured chat model instance to
   `create_deep_agent(model=...)`. The runtime MUST NOT pass a provider-prefixed
   model string as a substitute for endpoint/API/authentication configuration.
3. **Separation of responsibilities.** Configuration owns host/family/model,
   endpoint, project/region, API version/mode, and credential selection. Model
   construction owns the corresponding LangChain client and model parameters.
   DeepAgents owns execution, tool dispatch, skills, and agent state. Existing
   batch/result layers own invocation lifecycle and Kubernetes publication.
4. **One route identity.** The resolved host/family/API choice MUST remain available
   for readiness, auxiliary model construction, error reporting, and telemetry.
   These consumers MUST NOT infer that all DeepAgents executions use Anthropic.
5. **No automatic fallback.** Model-client or API failures MUST NOT silently switch
   provider, endpoint, API mode, credentials, or agent SDK. Existing configuration
   and invocation failure paths remain applicable.

### Model-client and API matrix

| Current model route | Target model client | Model API | Status / remaining check |
| --- | --- | --- | --- |
| Direct Anthropic / Claude | `ChatAnthropic` | Anthropic Messages | Already used with DeepAgents in this sandbox; retain behavior and regression coverage. |
| Vertex AI / Claude | `ChatAnthropicVertex` | Vertex-hosted Anthropic API | Already used with DeepAgents in this sandbox; retain behavior and regression coverage. |
| AWS Bedrock / Claude | `ChatAnthropicBedrock` | Bedrock-hosted Anthropic API | Already used with DeepAgents in this sandbox; retain behavior and regression coverage. |
| Vertex AI / Gemini | `ChatGoogleGenerativeAI(vertexai=True)` | Vertex Gemini API | Model-client pattern exists in `lightspeed-service`; verify DeepAgents tools, reasoning, streaming, and output behavior. |
| Native OpenAI | `ChatOpenAI(use_responses_api=True)` | Responses API | Preserve native-OpenAI API selection; verify DeepAgents behavior. |
| OpenAI-compatible / vLLM | `ChatOpenAI(base_url=…, use_responses_api=False)` | Chat Completions API | Explicitly avoid Responses; verify deployed-model tool calls/parser, streaming, reasoning fields, and structured output. |
| Azure OpenAI | `AzureChatOpenAI` with explicit version-based API selection | Chat Completions for older API versions; Responses for newer versions | Preserve the sandbox's selection policy; verify LangChain Responses routing, API-key and Entra authentication, tools, and output. |
| Vertex AI / OpenAI-compatible model | Candidate: `ChatOpenAI` with a Vertex-compatible URL and Google OAuth authentication | Likely Chat Completions; unconfirmed | Unresolved: endpoint, token refresh, and a working deployment have not been established in these repositories. |

6. **API selection.** DeepAgents does not require Responses. Native OpenAI MUST
   retain Responses; custom/vLLM OpenAI-compatible endpoints MUST explicitly select
   Chat Completions. Merely configuring `base_url` is insufficient. An `openai:...`
   string MUST NOT replace the configured vLLM client because that route may default
   to Responses. Azure MUST preserve its existing version-based decision, with
   transport tests proving which API is actually invoked.
7. **Endpoint overrides.** Existing provider URL, model identifier, project, and
   region inputs MUST retain their meaning. The migration MUST NOT rewrite model
   identifiers or redirect custom endpoints to a native provider. Existing
   Anthropic-compatible endpoint behavior remains separate from the OpenAI/vLLM
   Chat Completions route.
8. **Vertex/OpenAI uncertainty.** The last matrix row is a candidate, not a support
   commitment or a decision to remove the route. No default URL, static OAuth-token
   workaround, or claimed deployment may be invented. The final support/release
   disposition remains open pending endpoint and credential-refresh evidence.

## Execution and feature contracts

9. **Stable integration boundary.** Preserve `/input/` handling, operator-provided
   prompts/context, `ProviderQueryOptions` and normalized event semantics, Result CR
   creation/status, and the distinction between agent failure and sandbox failure
   in [run-api.md](run-api.md). Consolidation MUST NOT add an HTTP server.
10. **Tools and MCP.** Use DeepAgents' native local-shell/filesystem backend and
    `langchain-mcp-adapters` for all migrated routes. Preserve canonical admission,
    admitted-name filtering, connection headers/timeouts, and fail-closed provider
    materialization. Never bypass admission to compensate for a model's inability
    to call a tool. Preserve existing result limits/offload before safety handling;
    do not add event-only truncation. Changing default tool allowlists or adding
    provider-hosted search/browser tools is not part of this design.
11. **Skills and workspace.** Use the existing DeepAgents skill-presence gate,
    physical skills paths, shell working-directory semantics, writable agent-home
    artifacts, and pod isolation. Do not carry forward OpenAI Agents skill
    materialization or ADK-specific script/tool execution. Existing skill behavior
    must be demonstrated through the shared backend for migrated model routes.
12. **Execution limits.** Use the existing DeepAgents/LangGraph iteration mechanism
    and preserve the operator-resolved invocation budget/maximum-turn contract
    tracked by OLS-3743. SDK iteration counts are not interchangeable; the migration
    MUST NOT claim that the same numeric limit implies identical model-call counts.
    Cancellation and deadline handling also cover shaping and inspection calls.
13. **Streaming and events.** Normalize LangChain model/tool events into the existing
    text, reasoning, tool-call, tool-result, and terminal-result event types. Emit
    complete tool arguments only after streamed call assembly, preserve real call
    IDs, and retain observed usage. Do not manufacture missing reasoning, usage,
    tool results, or response metadata. No token-chronology guarantee is introduced.

### Reasoning and structured output

14. **Provider-specific reasoning.** Preserve the public JSON-object reasoning
    configuration, with absent configuration leaving model defaults untouched.
    Map Anthropic `thinking`, Gemini thinking budget/level/thought options, and
    OpenAI/Azure reasoning settings to their selected model client's native
    parameters. Do not feed Anthropic `thinking` to every client or silently discard
    configured fields. Exact client-version support and nonstandard vLLM fields
    require verification; API selection alone does not prove reasoning support.
15. **Two-phase output.** Keep the agent/tool pass separate from a tool-free output
    shaping pass when `output_schema` is set. This extends the existing DeepAgents
    pattern to the shared runtime rather than retaining agent-SDK-specific schema
    binding. The shaping client MUST use the same resolved provider/endpoint/auth/TLS
    configuration but omit main-agent reasoning settings, and MUST receive the
    completed agent text, not another executable tool loop.
16. **Model-specific schema strategy.** Preserve the existing Anthropic method
    selection and supported schema subset. Preserve native OpenAI strict-schema
    transformations where required, while avoiding strict-native assumptions for
    vLLM. Gemini, Azure, and compatible endpoints need a verified client-native
    structured-output method for their deployment/API. No universal schema method
    or expanded JSON Schema support is assumed. Unsupported schemas, invalid
    shaped output, and shaping failures MUST follow the existing failure contract,
    not silently return success or switch APIs.

### Safety, authentication, and transport

17. **Shared safety boundary.** Every model using the shared runtime MUST use the
    existing DeepAgents tool-output wrapping/trust instruction and inspection
    interception points, including general-purpose subagents and summarization.
    Keep result limits/offload -> enabled inspection -> wrapping -> model delivery
    order. Keep stored artifacts and approved normalized/audit results unwrapped.
    Rejecting inspection MUST suppress pending result payloads and preserve the
    controlled failure/no-Result behavior. Existing authority remains the parent
    tool-result-inspection contract and local SAFE-02 decision.
18. **Auxiliary clients.** Classifier and shaping clients MUST be derived from the
    same resolved route, without main-agent reasoning configuration. A Gemini,
    OpenAI, or Azure run MUST NOT silently invoke an Anthropic classifier. Verify
    structured classifier output with each client. Disabling inspection still
    leaves wrapping and trust instructions active.
19. **Credentials and refresh.** Preserve mounted credential inputs and existing
    readiness failure behavior. Retain Google service-account credentials, AWS
    static/assume-role handling, Azure API-key/Entra mode precedence, and native or
    compatible endpoint credentials. Delegate short-lived-token minting/refresh to
    provider credential libraries; do not introduce a token cache/refresh timer or
    treat a startup OAuth token as a complete long-running solution. Verify both
    Azure authentication modes on both API paths.
20. **TLS and cleanup.** Model, authentication, and MCP clients MUST consume the
    shared provider-egress CA/TLS configuration. Adding LangChain clients MUST NOT
    bypass the shared SSL context, disable certificate verification, or add
    provider-specific CA mounts. Close owned sync/async model and credential
    resources on success, failure, cancellation, and auxiliary-call failure.

### Observability and packaging

21. **Shared telemetry, actual provider identity.** Capture generations at the
    LangChain model boundary through the shared runtime. Retain invocation/tool
    parentage, main-agent-only capture plus raw shaping generations, exclusion of
    classifier/subagent/summary generations from that canonical generation stream,
    real call IDs, observed response metadata, and separate aggregate/generation
    usage accounting. All shared LangChain chat generations use operation `chat`;
    migrated Gemini therefore changes from ADK `generate_content` to `chat`.
    Preserve actual hosting identity (`anthropic`, `aws.bedrock`, `gcp.vertex_ai`,
    or existing OpenAI/Azure identity `openai`), not `deepagents` as the model vendor.
    Retain `openai.api.type` from the actual selected OpenAI/Azure API path.
22. **No synthetic legacy SDK telemetry.** Do not fabricate ADK/OpenAI Agents events,
    spans, IDs, hosted-tool parts, or logs after removing those SDKs. Preserve
    normalized developer/audit/templog projections and their activation/content
    gates through the shared events. Safety-rejected content stays suppressed.
    Verify the mapping from each client's observed content blocks, metadata, and
    usage; existing three-adapter test results are baseline evidence, not proof
    of the replacement. Parent collection/export contracts are unchanged.
23. **Dependency boundary.** Remove `google-adk` and `openai-agents` and runtime
    imports of their agent APIs. Retain DeepAgents, needed LangChain integrations
    (`langchain-anthropic`, `langchain-google-vertexai`, `langchain-aws`,
    `langchain-google-genai`, `langchain-openai`, `langchain-mcp-adapters`), and
    necessary vendor/auth libraries such as `google-genai`, `openai`, Anthropic,
    Google auth, AWS, and Azure identity dependencies. Removal of an agent SDK is
    not removal of its vendor's model SDK. Declare model integrations directly
    rather than relying on incidental transitive installation.
24. **Build and optional imports.** Preserve lazy optional model-client imports and
    clean base-package importability. Dependency extras, image inputs, `uv.lock`,
    Konflux profiles/hashed requirements, and pipeline package lists MUST agree
    when implementation changes dependencies. The released image MUST contain its
    runtime dependencies without startup downloads, including the disconnected
    vLLM path. This spec change does not modify dependency artifacts.

## Verification and acceptance gates

No migration checks in this section have been executed by writing this spec.
Existing tests listed in the linked specs remain baseline evidence only.

| Gate | Required evidence |
| --- | --- |
| Route construction | Offline constructor tests cover every established matrix row, explicit model-instance injection, endpoint/model/project/region propagation, correct API flags, Azure version boundaries, and no fallback to prefixed model strings. |
| Transport/API selection | Client/transport tests show native OpenAI Responses, custom/vLLM Chat Completions with no Responses requests, and each Azure version path using the intended endpoint. |
| Tools and skills | Shared-runtime tests cover local execution, actual skill scripts, admitted MCP names, complete admitted-server initialization, tool failures, result bounds/offload, and real call-ID correlation for each migrated family. |
| Reasoning and output | Tests cover absent/present reasoning, client-native mapping, no silent key loss, tool use followed by tool-free shaping, supported schemas, invalid output, and reasoning-free auxiliary clients. |
| Safety | Existing inspection/wrapping tests run against migrated client families, including rejection before event release, summary/subagent paths, offloaded reads, and wrapping with inspection disabled. |
| Credentials/TLS | Tests cover Azure key/Entra on both APIs, SDK-delegated refresh, Google/AWS configuration, shared CA/profile propagation, and cleanup on all terminal paths. |
| Events and telemetry | Scripted model/runtime and OTLP wire tests verify normalized events, observed metadata/usage, raw shaping capture, excluded auxiliary generations, cancellation/errors, provider/API identity, parentage, and unchanged audit/content gates. |
| SDK removal and image | Dependency/import checks establish no ADK/OpenAI Agents runtime dependency, retained model clients, optional-import isolation, and coherent local/hermetic dependency graphs. |
| Live route qualification | Batch BDD runs the configured deployed model through DeepAgents for tools/MCP, skills, reasoning where supported, structured output, streaming/event handling, and Result publication. Record model ID, endpoint type, client versions, configuration, observed limitations, and evidence location. |

For vLLM, record the served model and tool-call parser configuration. A usable
Chat Completions endpoint does not establish tool capability, nonstandard reasoning
field handling, or structured-output support. For Azure, qualify the LangChain
Responses path and API-key/Entra behavior rather than inferring them from an API
version date. Scripted transports and offline tests do not establish live parity;
live provider proof does not establish downstream collector/Dataverse delivery.

## Open design issues

- **Vertex/OpenAI-compatible:** identify the actual deployment and endpoint shape,
  confirm API mode, and demonstrate Google OAuth acquisition/refresh through the
  model client. Its final support/release disposition is not selected here.
- **Azure Responses:** verify that the selected `AzureChatOpenAI` version supports
  the existing endpoint/deployment/authentication contract on the Responses path.
  A failed check requires a reviewed design revision, not silent API substitution.
- **Client capability details:** establish Gemini/Azure/compatible shaping methods,
  Gemini/OpenAI reasoning parameter mappings, and nonstandard vLLM reasoning
  preservation for the chosen client/model versions. Do not assume Anthropic-only
  helper code or metadata parsers are portable unchanged.
- **SDK-native capabilities:** the migration does not promise reproduction of every
  ADK/OpenAI Agents feature, including provider-hosted tools. Any required capability
  absent from the shared backend needs an explicit follow-up design decision.

These are named evidence gaps, not placeholder requirements or claims of support.
Approval of this proposed design and successful qualification are distinct gates;
neither is implied by committing the document.

## Related specifications

- [Provider contract](provider-contract.md): current events, models, tools, skills,
  reasoning, output shaping, MCP, inspection, and SAFE-02.
- [Configuration](configuration.md): operator inputs, credentials, model resolution,
  compatible endpoints, TLS, and dependency/build policy.
- [Run API](run-api.md): unchanged batch/Result boundary and execution failures.
- [Provider architecture](../how/provider-architecture.md): current mechanisms and
  the target separation between model clients and the shared runtime.
- [Audit logging](audit-logging.md) and [data collection](data-collection.md):
  existing telemetry contracts and historical verification boundaries.
- [E2E testing](e2e-testing.md): batch BDD harness and current provider matrix.
