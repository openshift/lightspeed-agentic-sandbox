# 0004 — Move the Agent Implementation into the Shared Execution Core

- **Jira:** [OLS-4671](https://redhat.atlassian.net/browse/OLS-4671)
- **Status:** Proposed; implementation [PLANNED: OLS-4671]
- **Date:** 2026-11-23
- **Applies to:** lightspeed-agentic-sandbox
- **Parent:** workspace `.ai/spec/decisions/0052-shared-deepagents-execution-core.md`; contract workspace `.ai/spec/what/unified-execution-core.md`
- **Behavior:** [provider-contract.md](../what/provider-contract.md#shared-execution-core-planned-ols-4671), rules 68–74

## Context

The sandbox's DeepAgents harness (decision 0003) and Classic chat's custom loop in lightspeed-service implement the same concerns twice. OLS-4671 moves agent execution for both into one library, `lightspeed-agent-core` (placeholder name), maintained in a new repository. Where both implementations exist, the core takes the sandbox implementation.

## Decision

- Move the sandbox's agent implementation into the core: the DeepAgents adapter, the model router and model integrations, cloud credentials, the MCP client, tool-result inspection, wrapping, redaction, summarization handling, usage accounting, GenAI tracing, reasoning configuration, structured-output shaping, and the cooperative agent timeout. Their tests move with them.
- The sandbox stays the batch entry point: `/input/` handling, readiness, batch lifecycle and deadlines, MCP tool admission, Result CR publishing, and termination-log handling.
- The migration is a pure refactor (workspace contract rule 37 step 2). Batch behavior, operator inputs, environment variables, Result CRs, and the existing offline and live suites stay unchanged.
- Chat-only core capabilities (approval, turn-limit continuation, tool retries, the chat profile) are opt-in. The sandbox MUST NOT enable them.
- The sandbox consumes the core as a hash-pinned HTTPS direct-URL requirement to a GitHub release sdist, with only the provider extras for its routes.

## Alternatives Considered

| Approach | Outcome |
| --- | --- |
| Keep the harness in the sandbox and add a chat entrypoint to the sandbox image | Rejected at the workspace level (parent ADR 0052): it moves the retrieval stack and an HTTP server into every batch pod. |
| Host the core inside this repository as a second distribution | Rejected in favor of a neutral repository with independent releases. |

## Consequences

- Changes to the agent harness happen in the core repository. Sandbox changes are limited to the batch entry point and version bumps.
- Decision 0003's statement that it includes no Classic OLS migration is superseded by this decision. Its route, tool and safety decisions remain and now apply inside the core.
- Sandbox image size is expected to stay about the same (workspace contract rule 42 limits growth to 5%).
- Hermetic builds gain one direct-URL requirement and the core's build backend.
