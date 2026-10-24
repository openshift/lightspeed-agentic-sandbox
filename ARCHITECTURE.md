# Architecture

The lightspeed-agentic-sandbox is a multi-provider agent runtime for OpenShift Lightspeed. It runs as a **one-shot batch process** inside ephemeral Kubernetes pods: the operator mounts input files, the sandbox runs the agent, publishes a Result CR via the Kubernetes API, and exits.

## System Context

The sandbox sits between the operator (workflow engine) and the LLM provider APIs. Each pod processes one AgenticRun step and is disposable.

```mermaid
graph LR
    Operator["Lightspeed Operator<br/>(workflow engine)"]
    Sandbox["Agentic Sandbox<br/>(batch process)"]
    K8s["Kubernetes API<br/>(Result CR)"]
    Anthropic["Anthropic API<br/>(via DeepAgents)"]
    Gemini["Gemini API<br/>(Google)"]
    OpenAI["OpenAI API"]
    Skills["Skills<br/>(mounted volume)"]

    Operator -->|"ConfigMap /input/*"| Sandbox
    Operator -->|"creates pod"| Sandbox
    Sandbox -->|"kubernetes client"| K8s
    K8s -->|"Result CR status"| Operator
    Sandbox -->|provider SDK| Anthropic
    Sandbox -->|provider SDK| Gemini
    Sandbox -->|provider SDK| OpenAI
    Sandbox -->|filesystem| Skills
```

There is **no HTTP server** in the sandbox — no FastAPI, no `/health`, `/ready`, or `/v1/agent/run`.

## Internal Architecture

```mermaid
graph TD
    subgraph "Batch entrypoint"
        Batch["batch.py<br/>read /input, publish Result"]
        RunAgent["run_agent.py<br/>run_agent_query()"]
        Readiness["readiness.py<br/>run_readiness_checks()"]
    end

    subgraph "Configuration & Cross-Cutting"
        Config["config.py<br/>resolve_model_family()<br/>env mapping + credentials"]
        ModelRouter["providers/model_router.py<br/>resolve_model_route()"]
        MCP["mcp.py<br/>parse_mcp_servers()"]
        Audit["audit.py<br/>AuditLogger"]
        Metrics["metrics.py<br/>in-process histograms"]
        Tracing["tracing.py<br/>TracerProvider"]
    end

    subgraph "Result publishing"
        Publish["publish_results/publish.py<br/>CustomObjectsApi"]
        Status["publish_results/status.py<br/>status assembly"]
    end

    subgraph "Provider Abstraction"
        Factory["factory.py<br/>create_provider()"]
        Types["types.py<br/>AgentProvider ABC<br/>ProviderEvent union"]
        Logger["logging.py<br/>EventLogger"]
    end

    subgraph "Provider Adapters"
        DeepAgentsP["deepagents.py"]
        OpenAIP["openai.py"]
    end

    subgraph "DeepAgents Model Integrations"
        AnthropicModel["models/anthropic.py"]
        GeminiModel["models/gemini.py"]
        OpenAIModel["models/openai.py"]
    end

    Batch --> Config
    Batch --> ModelRouter
    Batch --> Readiness
    Batch --> Factory
    Batch --> RunAgent
    Batch --> Publish
  Publish --> Status
    RunAgent --> Types
    RunAgent --> Logger
    RunAgent --> MCP
    RunAgent --> Audit
    RunAgent --> Metrics
    Factory -->|route-driven| DeepAgentsP
    Factory -->|legacy OpenAI path| OpenAIP
    ModelRouter -->|ResolvedModelRoute| DeepAgentsP
    DeepAgentsP -->|route constructor| AnthropicModel
    DeepAgentsP -->|route constructor| GeminiModel
    DeepAgentsP -->|route constructor| OpenAIModel
```

## Batch Run Flow

```mermaid
sequenceDiagram
    participant Op as Operator
    participant Batch as batch.py
    participant Agent as run_agent_query()
    participant Provider as Provider Adapter
    participant SDK as Vendor SDK
    participant LLM as LLM API
    participant K8s as Kubernetes API

    Op->>Batch: mount /input/*, start pod
    Batch->>Batch: resolve_model_family(), resolve_model_route(model_family=...)
    Batch->>Batch: run_readiness_checks(model_family)
    Batch->>Batch: create_provider(name, model_route for DeepAgents)
    Batch->>Agent: system prompt, query, schema, context
    Agent->>Provider: query(ProviderQueryOptions)
    Provider->>SDK: SDK-specific invocation
    SDK->>LLM: API calls (multi-turn)

    loop Event stream
        SDK-->>Provider: SDK events
        Provider-->>Agent: ProviderEvent
        Agent->>Agent: EventLogger.log()
    end

    SDK-->>Provider: Final result
    Provider-->>Agent: ResultEvent
    Agent-->>Batch: AgentRunResult
    Batch->>K8s: create_namespaced_custom_object (Result)
    Batch->>K8s: replace_namespaced_custom_object_status
    Batch-->>Op: exit 0 (or termination-log + exit 1 on infra failure)
```

Result CR lifecycle uses the **`kubernetes` Python client** (`CustomObjectsApi`), not `oc` subprocesses. The image still ships `oc` and `kubectl` for **agent tool execution** (delegated to provider SDKs).

## Provider Adapter Design

Each adapter is a thin wrapper. The SDK owns tool execution, skill discovery, and multi-turn orchestration. Adapters map SDK events to normalized `ProviderEvent` objects and extract token usage.

| Harness | Model integrations | Structured Output | Skills | Tools |
|---|---|---|---|---|
| DeepAgents | Anthropic direct, Bedrock, and Vertex; Gemini on Vertex; native OpenAI and custom/vLLM endpoints via `langchain-openai` | Tool-free second pass without reasoning; native OpenAI uses strict JSON schema, compatible endpoints preserve the supplied schema | `/skills/` virtual `CompositeBackend` route when skills exist | `LocalShellBackend` (default) + `/skills/` `FilesystemBackend` route + MCP tools |

Batch resolves the model family once and reuses it for routing and credential
readiness. Anthropic, Vertex/Google, and native/custom OpenAI return a `ResolvedModelRoute` with
the selected model, typed constructor parameters, telemetry identity, and
structured-output method. The factory injects this route into `DeepAgentsProvider`;
the provider creates fresh main, inspection-classifier, and output-shaping models
through it. Only the main model receives query reasoning. Anthropic constructors
live in `providers/models/anthropic.py`; the Vertex Gemini constructor lives in
`providers/models/gemini.py`; native OpenAI uses `providers/models/openai.py` to
construct `ChatOpenAI(use_responses_api=True)` with resolved credentials. These
integrations retain lazy SDK imports and shared TLS. Native OpenAI is selected
only for hosting `openai` with no endpoint override or an HTTPS URL whose hostname
is `api.openai.com`; a non-HTTPS URL with that hostname is rejected. Endpoint
precedence and normalization are defined in
[configuration rule 3a](.ai/spec/what/configuration.md). Custom/vLLM endpoints use
`ChatOpenAI(use_responses_api=False)` through the `openai-compatible` route, with
raw schema mode and a non-strict schema wrapper. Azure uses the shared DeepAgents
harness with native Azure clients, API-key or Entra ID auth, and version-aware
Chat Completions/Responses schema handling. Vertex/OpenAI is unimplemented pending OLS-4472; the legacy adapter and dispatch are removed. The overall single-harness migration
remains partially implemented; live parity and telemetry compatibility review are
still required.

DeepAgents exposes skills at virtual `/skills/` paths and relies on its SDK
middleware to provide the corresponding host paths to shell commands. Scripts
run through native `execute`; dependent setup/check commands must share one shell
call. Artifacts use writable `HOME/.deepagents` storage. See the
[skills-route decision](.ai/spec/decisions/0002-sdk-native-skills-route.md)
for path mapping and execution limits.

## Container & Deployment

The sandbox ships as a container image built with Konflux hermetic builds (dependencies prefetched, no network during build). Release builds use RHOAI 3.5 CPU base images configured in `.konflux/build.args`; non-hermetic developer builds default to UBI 9 Python 3.12 images.

```mermaid
graph TD
    subgraph "Container Image"
        direction TB
        Base["RHOAI 3.5 CPU base (release)<br/>UBI 9 Python base (development)"]
        Sys["System packages<br/>(bash, git, oc, kubectl, catatonit)"]
        Py["Python 3.12 + site-packages<br/>(kubernetes, provider SDKs)"]
        AppSrc["Application source<br/>/opt/lightspeed/src/"]
        SkillMount["Skills mount<br/>/app/skills/ (read-only)"]
        InputMount["Input ConfigMap<br/>/input/ (read-only)"]
    end

    Base --> Sys --> Py --> AppSrc
    AppSrc -.-> SkillMount
    AppSrc -.-> InputMount

    subgraph "Runtime"
        Catatonit["catatonit (PID 1)"]
        BatchCmd["python -m lightspeed_agentic.batch"]
    end

    Catatonit --> BatchCmd
```

The container runs as non-root user `agent`. `catatonit` is PID 1. The batch module reads `/input/`, runs the agent, publishes the Result CR, and exits — no listener port.

## Key Decisions

- **One provider per pod:** Selected at startup through `config.resolve_model_family()` and `providers.model_router.resolve_model_route()`. Anthropic, Vertex/Google, native/custom OpenAI, and Azure receive an explicit DeepAgents model route; Vertex/OpenAI is unimplemented pending OLS-4472.

- **Thin adapters:** Provider modules map SDK events to a normalized union type; they do not re-implement SDK behavior.

- **Lazy SDK imports:** The factory uses lazy imports so the base package loads without any vendor SDK installed.

- **Kubernetes API for Result CRs:** `publish_results/publish.py` uses the `kubernetes` Python client — not `oc create` / `oc patch`. Inside a pod (`KUBERNETES_SERVICE_HOST` set), authentication MUST use in-cluster config only; if that fails, publish fails and kubeconfig MUST NOT be consulted. Outside a cluster (local dev/tests), kubeconfig MAY be used when in-cluster config is unavailable.

- **Hermetic builds:** Python wheels and RPMs are lockfile-prefetched. `oc`/`kubectl` are copied from Red Hat image stages for agent tools, not for sandbox infrastructure.
