# 0003 — Consolidate on One DeepAgents Harness

- **Jira:** [OLS-4434](https://redhat.atlassian.net/browse/OLS-4434)
- **Status:** Accepted design; implementation [PLANNED: OLS-4434]
- **Date:** 2026-10-15
- **Applies to:** lightspeed-agentic-sandbox
- **Behavior:** [provider-contract.md](../what/provider-contract.md#deepagents-consolidation-planned-ols-4434), rules 50–67

## Intent

Reduce duplicated implementation and testing of MCP exposure, tools/skills,
structured output, streaming events, reasoning, safety, usage accounting, and
telemetry across DeepAgents, Gemini ADK, and OpenAI Agents SDK. Success is one
agent harness supporting all existing model routes without requiring changes to
operator configuration or credential Secrets.

DeepAgents was selected by the product/design owner during this specification.
This is not an SDK-selection spike or a claim that every route already works.

## Context

The sandbox is an ephemeral, one-shot batch worker behind a normalized provider
boundary. Today its three adapters each delegate agent behavior to a different
harness. DeepAgents already provides the local shell/filesystem backend, skills,
MCP tool integration, model-boundary inspection, and tool-output wrapping for
Anthropic routes. Gemini and OpenAI routes have separate tools, event mapping,
structured-output paths, and telemetry hooks.

The provider and its credentials are independent of the harness. Preserving
Azure, Vertex, Bedrock, OpenAI-compatible endpoints, and native model capabilities
still requires provider client libraries and model integrations.

## Decision

### One adapter, narrow model integrations

Use one DeepAgents adapter behind the existing `AgentProvider` boundary. Keep
batch execution, normalized events, Result CR publishing, and operator inputs
stable. Separate route/model resolution from the fixed harness identity.

Provider-specific model integrations construct/configure LangChain chat models
and map native metadata, reasoning, and structured-output behavior where needed.
They do not own a second agent loop, tool executor, safety implementation, or
parallel telemetry pipeline. Exact integration versions/classes must satisfy the
behavioral contract and be verified during implementation; this decision does
not assert untested library support.

Preserve direct Anthropic, Vertex Anthropic/Google, Bedrock Anthropic,
native OpenAI, Azure OpenAI, and OpenAI-compatible/vLLM routes. Native credential
objects continue to own short-lived-token minting/refresh and retries.

The original Vertex/OpenAI migration requirement is superseded by the decision
not to implement that integration. Sandbox configuration accepts only Anthropic
and Google for Vertex; `openai` is rejected during family resolution. Native
OpenAI, Azure OpenAI, and custom OpenAI-compatible endpoints remain supported.
Removal of the operator's Vertex/OpenAI CRD choice is separate follow-up work.

### DeepAgents tools and safety for every model

All models use DeepAgents' native shell/filesystem and skills mechanisms, plus
configured MCP tools exposed through DeepAgents. Do not retain or recreate the
Gemini ADK or OpenAI Agents tool systems. ADK-hosted Google Search and URL Context
are explicitly not required for migration parity. Built-in tool names follow
DeepAgents rather than preserving legacy SDK aliases.

MCP discovery/admission stays outside the adapter. Each model gets only admitted
tools through the existing DeepAgents materialization/filtering path.

Every migrated model receives the existing DeepAgents inspection and SAFE-02
contracts, including their main/subagent interception, summary wrapping, and
offloaded-result rules. Inspection remains configurable; wrapping and trust
instructions remain active when inspection is disabled. Shared classifier
semantics and source-evidence policy remain owned by the workspace inspection
and collection contracts, not redefined by a model integration.

The original [SAFE-02 decision](0001-tool-output-boundary.md) remains valid. Its
unchanged-Gemini/OpenAI statements describe the legacy harnesses; OLS-4434 extends
the same DeepAgents behavior to their model routes when those harnesses retire.

### Preserve capabilities, not retired SDK internals

Keep provider-specific reasoning and endpoint-compatible structured output,
including required tool-free shaping. Do not impose the current Anthropic schema
conversion subset on every model family. Preserve normalized events and observed
usage/metadata, but do not emulate legacy SDK tool names or exact stream timing.

Keep the downstream trace/content contract. Replacing native SDK hooks and
changing Google generation representation can affect collected data even if
attributes retain their names. Implementation merges require the workspace's
component/Dataverse-owner compatibility review, explicit version decision, and
supporting evidence. A breaking producer rollout waits for compatible downstream
handling and a consumer-visible contract discriminator. This spec grants no
compatibility waiver and invents no implemented version field.

### Gated retirement

Develop model integrations and shared contract tests before retiring old routes.
Require offline and live batch evidence for every route and preserved credential
mode, shared tools/skills/MCP, schemas/reasoning, safety, events/telemetry, and
lifecycle/limits. Missing credentials or skipped tests mean unverified, not pass.
The normative verification matrix is in `provider-contract.md`.

The final runtime contains only DeepAgents: no permanent migration selector,
legacy-harness fallback, ADK adapter, or OpenAI Agents adapter. Remove direct
harness dependencies and update local/hermetic lock/build inputs together;
retain vendor clients only where needed. Rollback is to a compatible prior
sandbox release through existing deployment controls.

## Alternatives Considered

| Approach | Trade-off / outcome |
| --- | --- |
| One DeepAgents adapter with narrow model integrations (selected) | Centralizes cross-cutting behavior; still isolates necessary provider/API differences and permits shared contract tests. |
| Separate provider adapters, each using DeepAgents | Easier incremental isolation, but retains places to duplicate tools, safety, events, and telemetry; not the final architecture. Temporary unmigrated adapters are development checkpoints only. |
| Keep the three existing harnesses | Preserves native SDK behavior but does not meet the consolidation objective. |

## Consequences

- Cross-cutting agent changes have one implementation and shared tests.
- Provider-specific auth, reasoning, API selection, output schemas, and metadata
  still need narrow integrations and real-provider verification.
- Gemini/OpenAI model routes gain DeepAgents safety and tools; old ADK/OpenAI
  Agents tool implementations are intentionally not preserved.
- Consolidation can change native trace shape, tool names, timing, and iteration
  semantics; compatibility evidence is required, not assumed.
- [PLANNED: OLS-4671] The "no Classic OLS migration" scope below is superseded by [0004](0004-consume-shared-agent-core.md): this harness moves into the shared core used by chat and batch; the route, tool and safety decisions here remain.
- No operator/CRD change, Classic OLS migration, proxy service, new model family,
  or additional tool system is included.
- This is a planned contract. It does not establish successful builds, live
  provider parity, inspection accuracy, or downstream Dataverse delivery.
