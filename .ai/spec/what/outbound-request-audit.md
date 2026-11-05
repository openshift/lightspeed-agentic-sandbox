# Agentic Sandbox Outbound Request Audit

[PLANNED: OLS-4192] Python-level outbound request instrumentation for lightspeed-agentic-sandbox. The workspace specification `ols/.ai/spec/what/outbound-request-audit.md` is authoritative for the shared version-1 JSON schema, lifecycle, privacy boundary, and scope. This child specification defines sandbox integration and coverage; it does not claim instrumentation is implemented.

## Intent and Explicit Coverage Limit

Operators can correlate sandbox application-level Python/SDK outbound requests with an AgenticRun and observed outcomes without collecting prompts, credentials, command text, or personal user identifiers in the new records. Application traffic includes LLM and remote MCP requests and their SDK credential exchanges. Kubernetes API request auditing is excluded: Result CR publication and other cluster API lifecycle operations are infrastructure traffic and a separate concern.

The sandbox is permitted to execute oc and other commands. That permission does not make subprocess network activity observable through Python client hooks. Existing native tool-execution evidence remains available under its own contract, but it is not a connection audit. Potential additional network observation is tracked separately by OLS-4848 and introduces no dependency for this Python-only work.

## Behavioral Rules

### In-Scope Python Requests

1. Instrument Python/SDK model requests for every supported runtime/provider route, including streaming, shaping, inspection/auxiliary model calls, and SDK credential/token acquisition. Shared transport factories are useful integration points but do not prove that SDK-managed clients or non-HTTP transports are covered.
2. Instrument Python-initiated remote MCP operations: initialization, discovery, resources, tool calls, and connection/session cleanup requests where they issue transport requests. Long-lived SSE receives and separate POSTs are distinct transport lifetimes, not one record per model tool turn. Local stdio IPC is not itself an outbound network request.
3. Kubernetes API requests, including Result CR create/read/status-update requests and cluster API requests during readiness or other batch lifecycle stages, are infrastructure traffic and MUST NOT enter this application-level audit stream. Their auditing is a separate concern, regardless of their Python client implementation. A request to a remote MCP tool that accesses Kubernetes remains in scope; the tool server's downstream Kubernetes request does not.
4. Cover direct production application-level Python HTTP requests outside the provider/MCP helpers as well. Preserve an explicit transport inventory for HTTP, cloud SDKs, and gRPC where used. Unsupported routes are not made supported by this specification; instrumentation MUST not revive retired adapters or add providers. Python client usage alone does not bring infrastructure traffic into scope.
5. In-scope application requests before or after agent invocation remain covered, including SDK credential exchanges and MCP cleanup requests. Audit initialization MUST precede the first in-scope application request; audit teardown MUST follow their cleanup and close still-active request lifetimes within existing shutdown constraints. Result publication does not require outbound instrumentation.

### Excluded Traffic

6. Do not instrument shell commands or infer their destinations by parsing command text. oc, curl, wget, Python child processes, subprocess-based credential helpers, and external stdio MCP processes may initiate their own connections; those connections are outside this implementation even when the parent Python process starts the command.
7. A shell-tool span or successful oc exit code MUST NOT be represented as a Python-observed outbound request. Do not invent an API request for each shell tool call, copy command arguments into metadata, or claim that Kubernetes API audit covers attempts that fail before reaching the server.
8. External MCP servers' downstream traffic, other workloads, database-wire protocols, and the audit/telemetry exporters' own traffic are excluded under the shared contract. No packet observer, egress proxy, sidecar, extra privilege, or new deployment dependency is introduced.

### Record Production and Context

9. Emit the shared `outbound.request.started` and `outbound.request.finished` JSON records for each Python transport attempt, with a fresh shared `attempt_id` for the pair. SDK retries and redirects require separate IDs at their actual transport boundaries; layered client hooks MUST not double-count them.
10. Use a dedicated allowlisted metadata emitter. Do not project existing GenAI, shell-tool, or CR-content spans into this stream. This narrowly scoped record type is an exception to [audit-logging.md](audit-logging.md)'s existing span-only/Python-logging restrictions; it does not authorize another content transcript or modify trace/templog payloads.
11. Capture the active valid OTel trace/span context and validated `agenticrun.uid`/`agenticrun.phase` at attempt start, and reuse them at finish. Retain context across SDK callbacks, thread work, concurrent tools, stream iteration, and in-scope post-invocation application requests. Missing run metadata remains absent, while standalone execution establishes local OTel context.
12. Use the existing batch correlation handoff (`TRACEPARENT`, `LIGHTSPEED_AGENTICRUN_UID`, `LIGHTSPEED_AGENTICRUN_STEP`) without changing operator APIs. Run UID MUST be a valid Kubernetes UUID; phase MUST be one of the existing controlled phase values. No personal user ID, run name/namespace, arbitrary baggage, or credentials may be added to the shared record schema.
13. Set `gen_ai.tool.name` only for a specific registered remote MCP tool execution whose request attribution is known. Discovery, LLM inference, and credential refresh MUST NOT inherit a tool name. Shell execution and Result publication do not produce records in this stream. Ambiguous attribution stays absent; do not infer it by parsing prompts, request bodies, or shell commands.

### Gates, Outcomes, and Streams

14. Honor the existing `LIGHTSPEED_AUDIT_ENABLED` parsing/default behavior in [audit-logging.md](audit-logging.md). No new environment variable or CR field is introduced. `LIGHTSPEED_CAPTURE_CONTENT` MUST never make these records content-bearing. Audit-disabled mode suppresses new records but does not suppress independent native tracing or product collection.
15. Enabled outbound records MUST emit without `OTEL_EXPORTER_OTLP_ENDPOINT` and independently of span sampling. Context establishment must supply valid trace/span IDs even where no source span is recorded/exported. New records must not depend on the span-event-to-templog bridge to appear on stdout.
16. Emit observed outcomes, not policy decisions. TLS failures, timeouts, DNS/connect errors, HTTP 403, and non-HTTP protocol errors use only the shared controlled categories. Never log exception messages, tracebacks, URLs, SDK error dictionaries, or provider response text in these records.
17. Start before transport invocation; finish at stream EOF/close, failure, or cancellation, including callbacks after the model invocation returns. Do not mistake response headers for successful streaming completion. Explicit early close ends the transport lifetime without claiming full application consumption. Preserve cancellation and SDK retry behavior, and never drain/buffer streams for logging or byte counting.
18. Preserve existing TLS/CA/proxy/authentication handling and thin SDK wrappers. Do not replace a provider transport with httpx merely to make instrumentation easy. Count a complete encoded request body only when naturally observable; partial or unknown sizes remain absent.
19. Auditing MUST not recurse on OTLP/telemetry requests or install duplicate hooks on repeated initialization. Emitter failures MUST not replace the original SDK exception, change domain outcomes, prevent Result publication, or block batch termination indefinitely.

### Compatibility

20. Preserve all existing native invocation, generation, tool, inspection, compliance, templog, and product-collection contracts, including independent content-capture/export gates. Existing content-bearing shell/tool evidence remains governed by those contracts; it MUST NOT be copied into these metadata-only records.
21. Gateway audit remains active when present and is neither replaced nor required. New records supplement Gateway visibility into Python attempts before a remote service can receive them.
22. Use Kubernetes logging enrichment for source pod/namespace/container identity. No API calls solely to enrich audit logs, new storage system, fixed search latency, or guarantee of complete sandbox network visibility is introduced.

## Implementation Boundaries and Inventory

The implementation PR MUST map each supported provider route to its actual transport and tests. Route consolidation may change source files; inventory the supported runtime at implementation time rather than adding instrumentation to dead adapters.

| Client family | Existing integration locations | Required evidence |
| --- | --- | --- |
| Shared HTTP/TLS clients | `src/lightspeed_agentic/tls.py` | Sync/async lifetimes, unchanged TLS/proxy behavior, hook deduplication |
| Provider SDK clients | `src/lightspeed_agentic/providers/` | Supported cloud/native/custom routes, SDK-managed clients, retries, credential requests, and non-HTTP transports where used |
| Remote MCP | `src/lightspeed_agentic/mcp.py` and provider MCP setup | Discovery versus call context, session transports, concurrent tools, no shell-tool attribution |
| Kubernetes API — excluded | `src/lightspeed_agentic/publish_results/publish.py` and any other Python API clients | Result publication and other cluster API infrastructure requests emit no application audit records |
| Batch lifecycle/correlation | `src/lightspeed_agentic/batch.py`, `run_agent.py`, `tracing.py` | In-scope application requests across startup/execution/cleanup, run context lifetime, audit gates, bounded shutdown |
| Existing tool/compliance evidence | `src/lightspeed_agentic/audit.py`, `logging.py` | Existing payload/gate behavior unchanged; no conversion of shell events into connection records |

## Verification — Required for Implementation

No outbound-audit test coverage is claimed by this specification-only change.

- Shared schema/privacy tests: destination URL userinfo/path/query/fragment, tokens/headers, prompts, manifests, tool arguments/results, exception messages, and personal identities contain sentinel values; assert none appear anywhere in the serialized outbound JSON record.
- Provider-route tests: each supported route, auxiliary/shaping/inspection calls, credential/token traffic, sync/async clients, HTTP and any non-HTTP transports, and SDK retries. A fake model callback without transport requests is not coverage proof.
- MCP tests: initialization/discovery/resource traffic, different concurrent tools, SSE/streamable HTTP lifetimes, no tool attribution on LLM or credential calls, and no duplicate pair from nested factories.
- Infrastructure exclusion tests: Result publication success/failure/retry and other Kubernetes lifecycle API calls emit no application outbound records, including when they share transport libraries with instrumented clients. Existing Result publication and failure handling remain unchanged.
- Streaming tests: EOF, explicit early close, midstream failure, cancellation, known/unknown/partial request body sizes, and no audit-induced stream consumption or changed exception propagation.
- Correlation tests: supplied parent trace and run UID/phase, standalone/missing/invalid metadata, parallel operations, callback/thread boundaries, and in-scope application cleanup after invocation ends. Start and finish must retain identical correlation.
- Gate/resilience tests: audit on/off, content capture on/off, OTLP endpoint present/absent, unsampled contexts, duplicate initialization, exporter recursion, emitter failure, and shutdown flush under existing limits.
- Subprocess negative test: execute a shell/oc fixture and confirm no Python outbound-request record is synthesized for its subprocess connection; ordinary existing tool evidence remains unchanged. Python Kubernetes-client publication in the same run also emits no application outbound records, while instrumented LLM/MCP requests still do.
- Deployment checks: JSON parsing, source workload enrichment, selection of only metadata-only records, and compatibility with supported OpenShift cluster logging/SIEM forwarding. No five-second searchability or network-observer prerequisite is asserted.

## References

- Workspace `ols/.ai/spec/what/outbound-request-audit.md` — shared schema and lifecycle.
- [Audit logging](audit-logging.md), [data collection](data-collection.md) — independent existing evidence streams.
- [Run API](run-api.md), [configuration](configuration.md), [provider contract](provider-contract.md) — batch and SDK integration.
- [OLS-4848](https://redhat.atlassian.net/browse/OLS-4848) — complementary subprocess network-observation follow-up, not an OLS-4192 dependency. It may observe connections also used by Kubernetes Python clients, but does not promise equivalent Kubernetes API request-level auditing.
