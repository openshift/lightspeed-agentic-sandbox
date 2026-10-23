# 0004 — Go Migration Feasibility Study

- **Jira:** none (exploratory study; commits/PRs use OLS-0000)
- **Status:** Proposed study — records feasibility analysis only; no migration decision
- **Date:** 2026-10-23
- **Applies to:** lightspeed-agentic-sandbox (cross-repo context: lightspeed-agentic-operator, lightspeed-service)
- **Behavior:** none changed; relates to [provider-contract.md](../what/provider-contract.md) rules 21–22 and 50–67, [0001-tool-output-boundary.md](0001-tool-output-boundary.md), [0003-single-deepagents-harness.md](0003-single-deepagents-harness.md)

## Intent

Record a feasibility analysis for rewriting the sandbox runtime in Go, motivated by Python packaging fragility, and assess the consequences for the provider contract, cross-repo unification, and the OLS Go ecosystem. This study does not approve a migration.

## Context

The motivation is Python's dependency fragility, which this tree exhibits in two distinct classes:

- **Packaging fragility (pip/PyPI/Konflux).** 159-entry `uv.lock`; 2,048 lines across 12 `.konflux/` requirement/RPM files plus Tekton package lists; dual dependency graphs (uv.lock vs Konflux RHOAI-preferred resolution) whose versions can diverge between development and shipped images; RHOAI/PyPI index skew reported only as warnings; wheel-vs-sdist duality (`pypi_wheel_only.txt`, `requirements-build.txt`); forced security floors on transitive dependencies. lightspeed-service documents the same class as CVE-pin comments fighting RHOAI index lag.
- **Agent-stack churn.** 18 lockfile entries from the deepagents/langchain/langgraph/adk/openai-agents family plus their transitive web; frequent breaking releases.

Relationship to ADR 0003: the accepted single-DeepAgents-harness consolidation ([PLANNED: OLS-4434]) removes the Gemini ADK and OpenAI Agents SDK dependencies, and OpenAI-compatible routes have already migrated on main (OLS-4469). This study assumes the post-0003 shape: one harness, many model clients. The consolidation is a prerequisite for the sequencing below.

## Findings

### Package mapping (availability verified 2026-10-23 via proxy.golang.org)

| Current dependency | Go equivalent | Status |
| --- | --- | --- |
| `kubernetes` (Result CR publishing) | `k8s.io/client-go` | Official |
| `opentelemetry-*` (traces, OTLP gRPC/HTTP, log bridge) | `go.opentelemetry.io/otel` + otlp exporters | Official; GenAI spans are hand-built today and port mechanically |
| `prometheus-client` | `prometheus/client_golang` | Official |
| `pydantic` (schema coercion) | struct tags + `santhosh-tekuri/jsonschema` or `invopop/jsonschema` | Adequate; different idiom |
| `httpx` (inspection classifier client) | `net/http` | Stdlib |
| `mcp`, `langchain-mcp-adapters` | `github.com/modelcontextprotocol/go-sdk` | Official |
| `deepagents` (harness) | none | Must be owned; see below |
| `langchain-anthropic` / `-google-vertexai` / `-aws` (model layer) | `github.com/anthropics/anthropic-sdk-go` + `/vertex` + `/bedrock` subpackages | Official |
| Google model routes | `github.com/googleapis/go-genai` | Official |
| `openai` / Azure / vLLM routes | `github.com/openai/openai-go` (Azure via options) | Official |
| `pytest` / `pytest-bdd` | `go test` + `github.com/cucumber/godog` | Gherkin scenarios in `tests/e2e/features/` run unchanged |

Pre-consolidation gaps (no `adk-go` skilltoolset parity need, no OpenAI Agents SDK port) are mooted by ADR 0003: those adapters are retired, not ported.

### Port surface

Against origin/main (847408b): 6,027 src lines total, of which providers are 2,089. The remaining ~3,900 lines (batch, config, types, readiness, mcp, publish_results, audit/metrics/tracing, tls, skills, tools, logging, run_agent, inspection) port mechanically to Go structs, `context.Context`, and channels for the async event streams.

### What must be owned in Go

The DeepAgents-equivalent harness: tool-calling loop, streaming event mapping to the normalized event model, shell/filesystem backends, skills loading, subagents, thinking-aware structured output (two-pass shape strategy, provider-contract rule 22), and tool-output boundaries (ADR 0001). Model clients remain official SDKs behind a thin model interface, replacing the LangChain chat-model construction in `_resolve_model`. Post-0003 there is exactly one harness to own — the consolidation replaces the three SDK stacks with one as it completes.

### Contract impact

Rule 21 (thin-adapter principle — SDKs own tool execution, command invocation, and skill discovery) has no Go counterpart to delegate to: a Go sandbox becomes the harness owner. A migration decision would require revising provider-contract.md rules 21–22 and re-anchoring ADRs 0001–0003 to the owned harness. The event model, Result CR contract, operator input files, env-var surface, and readiness checks can be preserved as-is.

### Unification (why Go, not Rust)

- **Compile-time CRD contract.** `publish_results/status.py` duplicates operator constants by hand (`CONDITION_STARTED = "Started"` …) that also exist in `lightspeed-agentic-operator/api/v1alpha1/result_types.go`. A Go sandbox imports the operator's `v1alpha1` package; drift becomes a compile error, and the typed client replaces dict-based `CustomObjectsApi` bodies.
- **Shared modules.** One model-client/harness library consumable by sandbox and lightspeed-service (which today maintains a parallel, larger provider matrix). TLS profile resolution, OTel-GenAI span builders, and MCP admission/policy are additional candidates. Shared surface must stay small and versioned (an API-types module, not a common library).
- **Ecosystem alignment.** lightspeed-agentic-operator, lightspeed-agentic-alerts-adapter, lightspeed-hub, lightspeed-operator are Go. Rust fixes packaging fragility equally well but offers no official Anthropic/OpenAI/Google SDKs and would leave every harness to own; Go maximizes official SDK coverage and org alignment.

### Ecosystem note

Python's agent-framework abundance originates in vendor SDKs following the pre-LLM ML population; Go's infrastructure population is arriving at agent workloads now. Protocol standardization (MCP, JSON-schema structured outputs, uniform tool-calling APIs) has externalized knowledge previously locked inside Python frameworks, which is what makes an owned Go harness tractable at this point rather than earlier.

## Consequences (if pursued)

1. **Prerequisite:** ADR 0003 lands first; it is the scope reduction.
2. **Sequencing:** port the core with an identical Result CR contract, env-var surface, and event model, validated by the existing BDD suite (godog runs the same `.feature` files against the Go binary); then port the single harness; then delete the `.konflux` requirement machinery, `make requirements`, and the RHOAI/PyPI dual graph. A batch runtime is all-or-nothing — no strangler path exists for the sandbox itself.
3. **Cross-repo:** lightspeed-service (a server, not a batch Job) can follow strangler-style per endpoint. Its hard parts are local embeddings (ONNX via `onnxruntime_go` or `hugot`; the two shipped models export to ONNX), llama-index orchestration (hand-rolled), FAISS (`chromem-go` or `pgvector-go`), and Solr hybrid search (`solr-go` or Postgres FTS). A sandbox-first port produces the shared modules the service migration would consume.
4. **Trade-offs:** the RHOAI curated channel for LLM libraries disappears (the sandbox owns its upgrade and CVE cadence); rented framework churn converts to owned maintenance; the CVE-vs-RHOAI-lag treadmill documented in both repos' dependency files ends.
5. **Upstream potential:** the owned harness is an eventual open-source candidate only after two internal consumers exist (sandbox plus service); extract and generalize then, not before. OTel-GenAI span helpers are a better early upstream contribution (opentelemetry-go-contrib) than a standalone project.

## Verification

Analysis-only; no code or behavior change. Package availability was verified 2026-10-23 against proxy.golang.org, crates.io, and GitHub repository checks. Repo statistics were measured against origin/main 847408b. Operator constant duplication was verified against `lightspeed-agentic-operator/api/v1alpha1/result_types.go` in the workspace.
