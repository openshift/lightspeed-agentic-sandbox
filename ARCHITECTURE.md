# Architecture

The lightspeed-agentic-sandbox is a multi-provider agent runtime for OpenShift Lightspeed. It runs as a FastAPI application inside ephemeral Kubernetes pods, accepting structured queries from the Lightspeed operator and delegating execution to one of three LLM provider SDKs.

## System Context

The sandbox sits between the operator (workflow engine) and the LLM provider APIs. It is a stateless worker — each pod processes one query and is disposable.

```mermaid
graph LR
    Operator["Lightspeed Operator<br/>(workflow engine)"]
    Sandbox["Agentic Sandbox<br/>(FastAPI)"]
    Claude["Claude API<br/>(Anthropic)"]
    Gemini["Gemini API<br/>(Google)"]
    OpenAI["OpenAI API"]
    Skills["Skills<br/>(mounted volume)"]

    Operator -->|"POST /v1/agent/run<br/>RunRequest JSON"| Sandbox
    Sandbox -->|RunResponse JSON| Operator
    Sandbox -->|provider SDK| Claude
    Sandbox -->|provider SDK| Gemini
    Sandbox -->|provider SDK| OpenAI
    Sandbox -->|filesystem| Skills
```

## Internal Architecture

The application has a layered design: HTTP routes parse requests and format responses, the provider abstraction normalizes query options and events, and thin adapters map between the normalized interface and each vendor SDK.

```mermaid
graph TD
    subgraph "HTTP Layer"
        App["app.py<br/>FastAPI entry"]
        Router["routes/<br/>build_router()"]
        Query["query.py<br/>POST /run handler"]
        Health["health.py<br/>GET /health"]
    end

    subgraph "Provider Abstraction"
        Factory["factory.py<br/>create_provider()"]
        Types["types.py<br/>AgentProvider ABC<br/>ProviderEvent union<br/>ProviderQueryOptions"]
        Logger["logging.py<br/>EventLogger"]
    end

    subgraph "Provider Adapters"
        ClaudeP["claude.py<br/>ClaudeProvider"]
        GeminiP["gemini.py<br/>GeminiProvider"]
        OpenAIP["openai.py<br/>OpenAIProvider"]
    end

    App --> Factory
    App --> Router
    Router --> Query
    App --> Health
    Query --> Types
    Query --> Logger
    Factory -->|lazy import| ClaudeP
    Factory -->|lazy import| GeminiP
    Factory -->|lazy import| OpenAIP
```

## Request Flow

A single request flows through the system as follows:

```mermaid
sequenceDiagram
    participant Op as Operator
    participant Route as POST /run
    participant Provider as Provider Adapter
    participant SDK as Vendor SDK
    participant LLM as LLM API

    Op->>Route: RunRequest JSON
    Route->>Route: Resolve timeout, system prompt
    Route->>Route: Format context prefix + query
    Route->>Provider: query(ProviderQueryOptions)
    Provider->>SDK: SDK-specific invocation
    SDK->>LLM: API calls (multi-turn)

    loop Event stream
        SDK-->>Provider: SDK events
        Provider-->>Route: ProviderEvent (text_delta, tool_call, etc.)
        Route->>Route: EventLogger.log()
    end

    SDK-->>Provider: Final result
    Provider-->>Route: ResultEvent (text, cost, tokens)
    Route->>Route: Parse JSON or text fallback
    Route-->>Op: RunResponse JSON
```

## Provider Adapter Design

Each adapter is a thin wrapper. The SDK owns tool execution, skill discovery, and multi-turn orchestration. Adapters are responsible only for:

1. Mapping `ProviderQueryOptions` to SDK-specific configuration
2. Consuming SDK event streams and yielding normalized `ProviderEvent` objects
3. Extracting cost and token usage from SDK results

| Provider | SDK | Structured Output | Skills | Tools |
|---|---|---|---|---|
| Claude | `claude-agent-sdk` | `output_format` JSON schema | Native `skills="all"` | Built-in SDK tools |
| Gemini | `google-adk` | Response schema on content config | `SkillToolset` from directory | `ExecuteBashTool` + web tools |
| OpenAI | `openai-agents` | `output_type` wrapper | `Skills` capability | `SandboxAgent` shell/filesystem |

## Container & Deployment

The sandbox ships as a container image built with Konflux hermetic builds (all dependencies prefetched, no network during build).

```mermaid
graph TD
    subgraph "Container Image"
        direction TB
        Base["UBI 9 base"]
        Sys["System packages<br/>(bash, git, oc, kubectl, ripgrep, tini)"]
        Py["Python 3.12 + site-packages<br/>(FastAPI, provider SDKs)"]
        Node["Node.js + claude-code CLI"]
        AppSrc["Application source<br/>/app/src/"]
        SkillMount["Skills mount<br/>/app/skills/ (read-only)"]
    end

    Base --> Sys --> Py --> Node --> AppSrc
    AppSrc -.-> SkillMount

    subgraph "Runtime"
        Tini["tini (PID 1)"]
        Uvicorn["uvicorn :8080"]
    end

    Tini --> Uvicorn
```

The container runs as a non-root `agent` user. `tini` is the init process (PID 1). Uvicorn serves the FastAPI app on port 8080.

## Key Decisions

- **One provider per pod:** The provider is selected at startup via `LIGHTSPEED_AGENT_PROVIDER`. This keeps pods simple and disposable — the operator chooses which provider to target when creating the pod.

- **Thin adapters over abstraction layers:** Provider modules map SDK events to a normalized union type but do not re-implement SDK behavior. This keeps maintenance cost proportional to SDK surface, not to a custom abstraction.

- **Lazy SDK imports:** Provider SDK packages are optional extras. The factory uses `match`-based lazy imports so the base package loads without any vendor SDK installed.

- **Hermetic builds:** All dependencies (Python wheels, RPMs, npm packages, external binaries) are declared in lockfiles and prefetched before the build starts. This ensures reproducible, auditable images.
