# 0002 — Single DeepAgents execution runtime

- **Status:** Proposed; not approved or implemented by this record.
- **Tracking:** [OLS-4434](https://redhat.atlassian.net/browse/OLS-4434).
- **Target contract:** [DeepAgents migration](../what/deepagents-migration.md).

## Context

The sandbox maintains separate DeepAgents, Google ADK, and OpenAI Agents execution
paths. They duplicate tools/skills integration, event normalization, reasoning and
output handling, and generation telemetry. DeepAgents already handles direct,
Vertex-hosted, and Bedrock-hosted Anthropic models through explicit LangChain chat
clients. DeepAgents accepts configured chat models; it does not require OpenAI's
Responses API.

Consolidating the agent loop must not collapse provider-specific model APIs,
credentials, or capabilities into one assumed protocol. In particular, deployed
vLLM services may support Chat Completions but not Responses, and tool calling
still depends on the served model and parser.

## Proposed decision

Use one DeepAgents execution path with a separate provider-specific model resolver.
Pass configured chat model instances to `create_deep_agent(model=...)`, never
provider-prefixed model strings as substitutes for explicit endpoint/API selection.
Retain native OpenAI Responses, explicit compatible-endpoint Chat Completions, and
Azure's existing version-based API selection, subject to the client-path checks in
the target contract.

Remove Google ADK and OpenAI Agents agent SDKs. Keep necessary LangChain model
integrations, vendor model SDKs, and credential/transport libraries. Preserve the
batch/Result boundary, admitted MCP policy, shared DeepAgents safety middleware,
credential refresh delegation, and provider-egress TLS.

Use separate agent and tool-free shaping phases when structured output is requested.
Keep reasoning and schema strategies model-specific. Rebase event/telemetry capture
on the shared LangChain boundary without fabricating legacy agent-SDK evidence.

## Alternatives

1. **Retain all three agent SDKs.** Reduces immediate model-client migration risk,
   but retains duplicated execution and safety/telemetry integration; does not
   meet the requested consolidation.
2. **Separate DeepAgents adapter per model vendor.** Removes the other agent SDKs,
   but invites duplicated runtime setup and divergent safety/tool behavior.
   Model construction is the required variation, not separate execution loops.
3. **One runtime with model-string auto-initialization.** Less explicit wiring, but
   can select unwanted Responses defaults and obscure endpoint, authentication,
   and TLS requirements. Rejected for the proposed target.

## Consequences and approval boundaries

- Model-specific behavior remains real work: reasoning, tool streaming, shaping,
  credentials, client lifecycle, and content/usage metadata need qualification.
- All migrated models inherit DeepAgents tools, skills, and safety boundaries;
  this is not a promise to reproduce every removed SDK capability.
- Gemini generation telemetry moves from ADK `generate_content` to shared chat
  generations. Provider identity and actual API mode remain independent of runtime.
- Dependency and hermetic-build inputs must be regenerated during implementation,
  not by this documentation-only change.
- Vertex/OpenAI-compatible remains unresolved. This proposal neither declares it
  working nor decides to remove it. Endpoint, OAuth refresh, deployment evidence,
  and final support/release disposition remain explicit open design issues.
- Azure Responses, cross-client reasoning/shaping, and live-model qualification
  remain acceptance gates. Existing Anthropic use and offline tests do not prove
  new-route parity.

This record proposes the architecture. It is not an implementation plan, an
accepted decision, or evidence that live verification has passed.
