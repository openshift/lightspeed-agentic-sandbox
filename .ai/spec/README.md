# Lightspeed Agentic Sandbox — Specifications

These specs define the behavioral rules and codebase navigation for the lightspeed-agentic-sandbox, a multi-provider agent runtime that runs inside ephemeral Kubernetes pods for OpenShift Lightspeed.

## Structure

| Layer | Path | Purpose |
|---|---|---|
| **what/** | `.ai/spec/what/` | Behavioral rules. What the system must do. Implementation-agnostic. |
| **how/** | `.ai/spec/how/` | Codebase navigation. How the code is organized. Implementation-specific. |
| **decisions/** | `.ai/spec/decisions/` | Architectural choices, alternatives, and consequences. |

### what/ — Behavioral Specifications

| Spec | Description |
|------|-------------|
| [system-overview.md](what/system-overview.md) | System role, component inventory, lifecycle, integration boundaries |
| [run-api.md](what/run-api.md) | Batch entrypoint: `/input/` files, agent run, Result CR publishing, context prefix, timeouts |
| [provider-contract.md](what/provider-contract.md) | AgentProvider ABC, event model, structured output, thin-adapter principle, skills delegation; DeepAgents routes including native/custom OpenAI, and deferred Vertex/OpenAI support |
| [configuration.md](what/configuration.md) | Environment variables, provider selection, model resolution, container layout, build system |
| [health-probes.md](what/health-probes.md) | Readiness checks at batch startup (R1); HTTP probes superseded |
| [audit-logging.md](what/audit-logging.md) | OTel GenAI invocation, tool, and generation spans; legacy choice events and audit-log projections |
| [data-collection.md](what/data-collection.md) | Three model-family GenAI invocation/tool/generation profile and exact `gcp.vertex.agent` exclusion from stdout/OTLP trace exports |
| [e2e-testing.md](what/e2e-testing.md) | Batch cluster BDD harness: OpenShift Jobs, fixtures, live vs unit split |

### how/ — Architecture Specifications

| Spec | Description |
|------|-------------|
| [project-structure.md](how/project-structure.md) | Entry points, naming conventions, dependency extras (package tree in AGENTS.md) |
| [provider-architecture.md](how/provider-architecture.md) | Data flow, abstractions, SDK integration points, implementation notes |

### decisions/ — Architectural Decision Records

| Decision | Description |
|---|---|
| [0001-tool-output-boundary.md](decisions/0001-tool-output-boundary.md) | DeepAgents-only SAFE-02 boundaries, trust instruction, and model/event separation |
| [0002-sdk-native-skills-route.md](decisions/0002-sdk-native-skills-route.md) | DeepAgents virtual `/skills/` route and SDK-provided host-path mapping for native shell execution (OLS-4413) |
| [0003-single-deepagents-harness.md](decisions/0003-single-deepagents-harness.md) | Historical OLS-4434 decision for consolidating model routes on DeepAgents; Anthropic, Vertex/Google, native/custom OpenAI, and Azure use it; Vertex/OpenAI migration and route-specific live acceptance remain |

## Scope

These specs cover the **lightspeed-agentic-sandbox** Python agent runtime only. The operator (which calls this runtime), console plugin, and skills packaging are separate projects with their own specs.

## Audience

AI agents. Content is optimized for precision and machine consumption.

## Quick Start

| Task | Start here |
|---|---|
| Understand the system | `what/system-overview.md` |
| Understand the batch entrypoint | `what/run-api.md` |
| Add or modify a provider | `what/provider-contract.md` + `how/provider-architecture.md` |
| Understand the DeepAgents migration and remaining target | `what/provider-contract.md` rules 50–67 + `decisions/0003-single-deepagents-harness.md` |
| Understand DeepAgents tool-output boundaries | `what/provider-contract.md` (SAFE-02) + `decisions/0001-tool-output-boundary.md` |
| Understand DeepAgents skill execution | `what/provider-contract.md` (rules 23, 33, 36) + `decisions/0002-sdk-native-skills-route.md` |
| Understand env vars and deployment | `what/configuration.md` |
| Navigate the codebase | `how/project-structure.md` |
| Understand readiness checks | `what/health-probes.md` |
| Understand audit logging | `what/audit-logging.md` |
| Understand sandbox Agentic content traces | `what/data-collection.md` |
| Understand E2E testing | `what/e2e-testing.md` |
| Run live cluster BDD | `what/e2e-testing.md` + `make e2e` |

## Cross-Reference

| what/ | how/ |
|---|---|
| `what/system-overview.md` | `how/project-structure.md` |
| `what/run-api.md` | `how/provider-architecture.md` (data flow section) |
| `what/provider-contract.md` | `how/provider-architecture.md` |
| `what/configuration.md` | `how/provider-architecture.md` (container build, implementation notes) |
| `what/health-probes.md` | `how/project-structure.md` (readiness.py) |
| `what/audit-logging.md` | `how/provider-architecture.md` (observability integration) |
| `what/data-collection.md` | `how/provider-architecture.md` (shared invocation/tool capture and provider-generation capture) |

## Conventions

- **Rule numbering:** behavioral rules are numbered sequentially within each what/ file.
- **Planned changes:** unimplemented behavior is marked with `[PLANNED]` or `[PLANNED: OLS-XXXX]` inline next to the rule it affects.
- **Environment variables:** reference the actual env var (e.g., `LIGHTSPEED_PROVIDER`).
- **Constraints:** component-specific and cross-cutting constraints go in the relevant what/ file's Constraints section, co-located with behavioral rules. Development conventions go in CLAUDE.md.
- **Authority:** what/ specs are authoritative for behavior. how/ specs are authoritative for implementation. When they conflict, what/ wins.
- **When to create a new file vs. extend an existing one:** if the new concern has its own lifecycle, configuration surface, and can be understood independently, it gets its own file. If it's a capability added to an existing component, it goes in that component's file.

## Project Context

This is the agent runtime that runs inside ephemeral sandbox pods. The operator mounts input files at `/input/`, starts the batch container, and reads the Result CR and pod exit status. Anthropic, Vertex Gemini, native/custom OpenAI, and Azure use the shared DeepAgents harness; Vertex/OpenAI is unimplemented pending OLS-4472. The legacy OpenAI adapter and factory entry are removed; harness dependency cleanup remains deferred. Azure's routing and SDK workflows are offline-tested, not live-authentication acceptance. The overall migration remains partial; see configuration rule 3a for the native OpenAI endpoint boundary. Result CR publishing uses the Kubernetes Python client, not `oc`.

Jira tracking: Feature OCPSTRAT-3095, Epic OLS-2894.
