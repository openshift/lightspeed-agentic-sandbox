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
        Config["config.py<br/>resolve_sdk()<br/>env mapping"]
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
        GeminiP["gemini.py"]
        OpenAIP["openai.py"]
    end

    Batch --> Config
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
    Factory -->|lazy import| DeepAgentsP
    Factory -->|lazy import| GeminiP
    Factory -->|lazy import| OpenAIP
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
    Batch->>Batch: resolve_sdk(), run_readiness_checks()
    Batch->>Batch: create_provider()
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

| Provider | SDK | Structured Output | Skills | Tools |
|---|---|---|---|---|
| DeepAgents | `deepagents` + `langchain-anthropic` | `ProviderStrategy` when thinking configured, else `ToolStrategy` via `response_format` | Skills dirs passed to `create_deep_agent()` | `LocalShellBackend` + MCP tools |
| Gemini | `google-adk` | Response schema on content config | `SkillToolset` from directory | `ExecuteBashTool` + web tools |
| OpenAI | `openai-agents` | `output_type` wrapper | `Skills` capability | `SandboxAgent` shell/filesystem |

## Container & Deployment

The sandbox ships as a container image built with Konflux hermetic builds (dependencies prefetched, no network during build).

```mermaid
graph TD
    subgraph "Container Image"
        direction TB
        Base["UBI 9 base"]
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

- **One provider per pod:** Selected at startup via `LIGHTSPEED_PROVIDER` (mapped to an SDK name by `config.resolve_sdk()`).

- **Thin adapters:** Provider modules map SDK events to a normalized union type; they do not re-implement SDK behavior.

- **Lazy SDK imports:** The factory uses lazy imports so the base package loads without any vendor SDK installed.

- **Kubernetes API for Result CRs:** `publish_results/publish.py` uses the `kubernetes` Python client — not `oc create` / `oc patch`. Inside a pod (`KUBERNETES_SERVICE_HOST` set), authentication MUST use in-cluster config only; if that fails, publish fails and kubeconfig MUST NOT be consulted. Outside a cluster (local dev/tests), kubeconfig MAY be used when in-cluster config is unavailable.

- **Hermetic builds:** Python wheels and RPMs are lockfile-prefetched. `oc`/`kubectl` are copied from Red Hat image stages for agent tools, not for sandbox infrastructure.
