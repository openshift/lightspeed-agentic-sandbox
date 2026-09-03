# Lightspeed Agentic Sandbox — Specifications

These specs define the behavioral rules and codebase navigation for the lightspeed-agentic-sandbox, a multi-provider agent runtime that runs inside ephemeral Kubernetes pods for OpenShift Lightspeed.

## Structure

| Layer | Path | Purpose |
|---|---|---|
| **what/** | `.ai/spec/what/` | Behavioral rules. What the system must do. Implementation-agnostic. |
| **how/** | `.ai/spec/how/` | Codebase navigation. How the code is organized. Implementation-specific. |

### what/ — Behavioral Specifications

| Spec | Description |
|------|-------------|
| [system-overview.md](what/system-overview.md) | System role, component inventory, lifecycle, integration boundaries |
| [run-api.md](what/run-api.md) | Batch entrypoint: `/input/` files, agent run, Result CR publishing, context prefix, timeouts |
| [provider-contract.md](what/provider-contract.md) | AgentProvider ABC, event model, structured output, thin-adapter principle, skills delegation |
| [configuration.md](what/configuration.md) | Environment variables, provider selection, model resolution, container layout, build system |
| [health-probes.md](what/health-probes.md) | Readiness checks at batch startup (R1); HTTP probes superseded |
| [audit-logging.md](what/audit-logging.md) | OTel GenAI semantic conventions, span events for LLM calls and tool execution, compliance audit trail |
| [e2e-testing.md](what/e2e-testing.md) | Batch cluster BDD harness: OpenShift Jobs, fixtures, live vs unit split |

### how/ — Architecture Specifications

| Spec | Description |
|------|-------------|
| [project-structure.md](how/project-structure.md) | Entry points, naming conventions, dependency extras (package tree in AGENTS.md) |
| [provider-architecture.md](how/provider-architecture.md) | Data flow, abstractions, SDK integration points, implementation notes |

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
| Understand env vars and deployment | `what/configuration.md` |
| Navigate the codebase | `how/project-structure.md` |
| Understand readiness checks | `what/health-probes.md` |
| Understand audit logging | `what/audit-logging.md` |
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

## Conventions

- **Rule numbering:** behavioral rules are numbered sequentially within each what/ file.
- **Planned changes:** unimplemented behavior is marked with `[PLANNED]` or `[PLANNED: OLS-XXXX]` inline next to the rule it affects.
- **Environment variables:** reference the actual env var (e.g., `LIGHTSPEED_PROVIDER`).
- **Constraints:** component-specific and cross-cutting constraints go in the relevant what/ file's Constraints section, co-located with behavioral rules. Development conventions go in CLAUDE.md.
- **Authority:** what/ specs are authoritative for behavior. how/ specs are authoritative for implementation. When they conflict, what/ wins.
- **When to create a new file vs. extend an existing one:** if the new concern has its own lifecycle, configuration surface, and can be understood independently, it gets its own file. If it's a capability added to an existing component, it goes in that component's file.

## Project Context

This is the agent runtime that runs inside ephemeral sandbox pods. The operator mounts input files at `/input/`, starts the batch container, and reads the Result CR and pod exit status. The runtime wraps multiple LLM provider SDKs (DeepAgents, Gemini, OpenAI) behind a single interface. Result CR publishing uses the Kubernetes Python client, not `oc`.

Jira tracking: Feature OCPSTRAT-3095, Epic OLS-2894.
