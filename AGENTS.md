# Lightspeed Agentic Sandbox

Multi-provider agentic sandbox for OpenShift Lightspeed. This repo runs as a
one-shot batch process with a DeepAgents harness for Anthropic, Vertex Gemini,
native or custom OpenAI-compatible endpoints (including vLLM), and Azure OpenAI.
Vertex/OpenAI is unimplemented pending OLS-4472; the legacy OpenAI adapter is removed.
When editing it, optimize for thin provider wrappers, consistent event mapping,
and tests that stay offline unless you are intentionally running live cluster BDD.

## General coding behavior

### Think before you implement

**Don't assume. Don't hide confusion. Surface tradeoffs.**
Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### Simplicity first

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### Surgical Changes

**Touch only what you must. Clean up only your own mess.**
When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### Goal-driven execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Specs

All specifications live in `.ai/spec/`. Start with [`.ai/spec/README.md`](.ai/spec/README.md) for project overview, reading order, and structure guide. Human-readable architecture overview: [`ARCHITECTURE.md`](ARCHITECTURE.md).

Before changing code, read the relevant spec:

| Working on | Read |
| --- | --- |
| System overview, integration boundaries | [system-overview.md](.ai/spec/what/system-overview.md) |
| Provider adapters | [provider-contract.md](.ai/spec/what/provider-contract.md) |
| Batch entrypoint | [run-api.md](.ai/spec/what/run-api.md) |
| Readiness checks | [health-probes.md](.ai/spec/what/health-probes.md) |
| Deployment, env vars, or defaults | [configuration.md](.ai/spec/what/configuration.md) |
| Audit / OTel / metrics | [audit-logging.md](.ai/spec/what/audit-logging.md) |
| E2E harness or BDD scenarios | [e2e-testing.md](.ai/spec/what/e2e-testing.md) |

Specs capture invariants, design decisions, and known quirks that the code
cannot express about itself. The code and this file cover the "how" — specs
cover the "why" and the "must."

### Component Specs

Each spec has a Verification section linking to tests that exercise its rules.
Use this table to navigate from component → spec → executable tests:

| Spec | Description | Verification |
| --- | --- | --- |
| [run-api.md](.ai/spec/what/run-api.md) | Batch entrypoint: input files, context prefix, timeouts, Result CR publishing | [test_batch.py](tests/test_batch.py), [test_batch_input.py](tests/test_batch_input.py), [test_run_agent.py](tests/test_run_agent.py), [test_publish_results_publish.py](tests/test_publish_results_publish.py), [test_publish_results_status.py](tests/test_publish_results_status.py) |
| [health-probes.md](.ai/spec/what/health-probes.md) | Readiness checks at batch startup (R1) | [test_ready.py](tests/test_ready.py), [test_batch.py](tests/test_batch.py) (readiness fail-fast) |
| [provider-contract.md](.ai/spec/what/provider-contract.md) | Provider adapter rules: events, structured output, thin-adapter principle | [test_run_agent.py](tests/test_run_agent.py), [test_deepagents.py](tests/test_deepagents.py); live batch: [skills.feature](tests/e2e/features/skills.feature), [structured_output.feature](tests/e2e/features/structured_output.feature), [analysis_output.feature](tests/e2e/features/analysis_output.feature), [mcp.feature](tests/e2e/features/mcp.feature), [reasoning_config.feature](tests/e2e/features/reasoning_config.feature) |
| [configuration.md](.ai/spec/what/configuration.md) | Provider selection, model resolution, skills directory, env vars | [test_model_router.py](tests/test_model_router.py), [test_config.py](tests/test_config.py); live batch: [mcp.feature](tests/e2e/features/mcp.feature), [reasoning_config.feature](tests/e2e/features/reasoning_config.feature) |
| [e2e-testing.md](.ai/spec/what/e2e-testing.md) | Batch cluster BDD harness (OpenShift Jobs, Result CRs, fixtures) | Live batch: [sandbox_e2e.feature](tests/e2e/features/sandbox_e2e.feature), [structured_output.feature](tests/e2e/features/structured_output.feature), [skills.feature](tests/e2e/features/skills.feature), [analysis_output.feature](tests/e2e/features/analysis_output.feature), [mcp.feature](tests/e2e/features/mcp.feature), [reasoning_config.feature](tests/e2e/features/reasoning_config.feature); helpers: [test_batch_e2e_helpers.py](tests/test_batch_e2e_helpers.py), [test_e2e_credentials.py](tests/test_e2e_credentials.py), [test_analysis_schemas.py](tests/test_analysis_schemas.py) |

## Quick Commands

```bash
make install                           # create/update .venv with dev dependencies via uv
make install-all                       # install all providers + dev + e2e extras
make lock                              # refresh uv.lock after dependency changes
make test                              # unit tests only; mocked providers, no API calls
make e2e openai-agents                 # live batch BDD on OpenShift (see e2e-testing.md)
make lint                              # ruff check src/ tests/
make format                            # ruff format + autofix
```

## Architecture

```text
src/lightspeed_agentic/
├── batch.py              # Batch entrypoint: read /input, run agent, publish Result CR
├── run_agent.py          # run_agent_query(), format_context_prefix()
├── config.py             # resolve_model_family(); SDK env/credential mapping; reasoning parse
├── factory.py            # create_provider(name, model_route=None); DeepAgents requires a route
├── readiness.py          # R1 credential checks; run_readiness_checks() at batch startup
├── mcp.py                # parse_mcp_servers(); header resolution
├── audit.py              # AuditLogger GenAI spans/events
├── metrics.py            # Prometheus histograms (in-process; no /metrics route)
├── tracing.py            # TracerProvider, traceparent helpers
├── logging.py            # EventLogger (debug thinking buffer)
├── skills.py             # has_skills() — SKILL.md presence under cwd
├── tools.py              # DEFAULT_ALLOWED_TOOLS only
├── types.py              # Provider events, query options, AgentProvider ABC
├── publish_results/
│   ├── publish.py        # Result CR via kubernetes CustomObjectsApi
│   └── status.py         # Status assembly (conditions, failureReason)
├── providers/
│   ├── deepagents.py     # Route-driven DeepAgents adapter
│   ├── model_router.py   # Typed DeepAgents routes; Vertex/OpenAI deferred to OLS-4472
│   └── models/
│       ├── anthropic.py  # Explicit direct, Bedrock, and Vertex model constructors
│       ├── gemini.py     # Vertex Gemini model constructor
│       └── openai.py     # OpenAI Responses/Chat Completions models and schema shaping
```

| Feature | DeepAgents |
| --- | --- |
| Models | Anthropic direct, Bedrock, and Vertex; Gemini on Vertex; native/custom OpenAI; Azure API key/Entra ID |
| Tools | `LocalShellBackend` + `/skills/` `FilesystemBackend` route + MCP tools |
| Skills | `/skills/` virtual route passed to `create_deep_agent()` when skills exist |
| Structured output | Tool-free second pass without reasoning; native OpenAI uses strict JSON schema, compatible endpoints preserve the supplied schema |
| Streaming | `astream(stream_mode="messages")` |

Keep provider adapters thin. The SDK should own tool execution and skill
discovery; `tools.py` holds the shared allowlist constant only. The SKILL.md
presence gate lives in `skills.py` (`has_skills`) — do not duplicate it in
adapters or put path helpers in `tools.py`.
DeepAgents exposes skills through its native `/skills/` virtual route. The SDK
adds the virtual-to-host path mapping needed by shell commands; scripts run through
native `execute`, with setup and dependent commands combined in one shell call.
See [the skills-route decision](.ai/spec/decisions/0002-sdk-native-skills-route.md)
for the route, artifact storage, and execution limits.

## Code Conventions

- Keep provider SDK imports inside methods or narrow helpers in provider modules.
  These SDKs are optional extras, so top-level imports must not break the base
  package import path.
- Batch resolves the model family once and reuses it for routing and readiness.
  Inject `ResolvedModelRoute` into DeepAgents for Anthropic, Vertex Gemini, and
  native/custom OpenAI; hosting selection and model construction belong in
  `model_router.py` and `providers/models/`, not the adapter. Native OpenAI means
  hosting `openai` with no endpoint override or an HTTPS `api.openai.com` hostname.
  Custom/vLLM endpoints use Chat Completions; native OpenAI uses Responses.
  Azure uses the same DeepAgents harness through its native Azure model integration;
  Vertex/OpenAI is unimplemented pending OLS-4472.
  See configuration rule 3a for endpoint precedence and HTTPS validation.
- `types.py` event objects are frozen dataclasses. New event types should follow
  the same pattern and stay simple to serialize/log.
- Providers yield async event streams; `run_agent_query()` consumes async
  iterators and waits for the final result event.
- Preserve the "thin adapter" shape when touching provider files: map SDK
  events into `ProviderEvent`, do not re-implement SDK behavior locally unless a
  testable workaround is required.

## Testing Conventions

- `make test` is the default verification path for code changes. Unit tests use
  mocked providers and must not require live credentials.
- Put reusable fake providers and event fixtures in `tests/conftest.py`.
  Prefer exercising real batch/run_agent glue over deep mocking of SDK internals.
- Unit tests cover `batch.py`, `run_agent.py`, and `readiness.py` with mocked
  providers and Kubernetes client — no live API calls.
- `make e2e` runs live batch BDD on an OpenShift cluster (`scripts/e2e-containers.sh`);
  see [e2e-testing.md](.ai/spec/what/e2e-testing.md).
- `make e2e` runs live batch BDD on an OpenShift cluster (`scripts/e2e-containers.sh`);
  see [e2e-testing.md](.ai/spec/what/e2e-testing.md).
- If you change e2e workspace fixtures or skills, verify batch mount paths in
  `tests/e2e/batch_runner.py` and `tests/e2e/skills_fixtures.py`.

## Konflux Hermetic Builds

The container image is built in [Konflux](https://konflux-ci.dev/) with hermetic
builds enabled — all dependencies are prefetched and verified before the build
starts, with no network access during the build itself.

### Dependency files

| File | Purpose | How to regenerate |
| --- | --- | --- |
| `.konflux/requirements.hashes.wheel.txt` | RHOAI wheel deps with hashes | `make requirements` |
| `.konflux/requirements.hashes.source.txt` | PyPI sdist deps with hashes | `make requirements` |
| `.konflux/requirements.hashes.wheel.pypi.txt` | PyPI wheel deps with hashes | `make requirements` |
| `.konflux/requirements-build.txt` | Build-time deps for source distributions | `make requirements` |
| `.konflux/requirements.hermetic.txt` | Hashed bootstrap wheels | `make requirements` |
| `.konflux/requirements.overrides.txt` | Optional RHOAI compatibility overrides (currently no pins) | Edit manually, then `make requirements` |
| `.konflux/pypi_wheel_only.txt` | Packages that must use PyPI wheels instead of source distributions | Edit manually, then `make requirements` |
| `.konflux/profiles.toml` | Package index, Python version, architectures, extras, and bootstrap packages | Edit manually, then `make requirements` |
| `.konflux/build.args` | Release base images (RHOAI 3.5) and hermetic build arguments | Edit manually |
| `.konflux/rpms.in.yaml` | System RPM package list | Edit manually |
| `.konflux/rpms.lock.yaml` | Resolved RPM lockfile | `make rpm-lockfile` |
| `.konflux/redhat.repo` | RHEL 9.6 EUS and RHEL AI 3.5 repositories for RPM resolution | Edit when repository configuration changes |
| `artifacts.lock.yaml` | Generic binary lockfile (may be empty; `oc`/`kubectl` come from image stages) | Edit manually when used |

### Bumping dependencies

Local development uses `uv.lock`. Konflux resolves a separate dependency graph
from `pyproject.toml`, preferring the RHOAI index over PyPI. RHOAI packages use
wheels; PyPI packages use source distributions unless explicitly listed in
`.konflux/pypi_wheel_only.txt` or no source distribution exists, in which case
they use wheels. Versions can differ from `uv.lock`.
`make verify-hermetic-requirements` checks package coverage, stale allowlist
entries, and orphan packages; RHOAI and PyPI version skew produces warnings.

Requirements regeneration needs network access, Podman or Docker, access to
`quay.io/syedriko/uv:prefer-index`, Python 3.12+ with `packaging` available to
`python3`, and `pybuild-deps` on `PATH`. These are separate from the ordinary
`make install` setup. For example, install the host tools with
`uv tool install pybuild-deps` and install `packaging` in the Python environment
used by `python3`.

RPM regeneration needs Podman or Docker and a Red Hat subscription; set
`ACTIVATION_KEY` and `ORG_ID`. It reads the release base image from
`.konflux/build.args` and repository definitions from `.konflux/redhat.repo`.

```bash
make bump-deps          # upgrade uv.lock + regenerate Konflux requirements and Tekton package lists
make rpm-lockfile       # regenerate .konflux/rpms.lock.yaml
```

Review and commit all changed lockfiles, hashed runtime/bootstrap requirements,
build requirements, and `.tekton/` pipeline package lists together. The Konflux
pipeline will prefetch the new versions on the next PR.

### Adding a new system package

1. Add the package name to `.konflux/rpms.in.yaml`
2. Run `make rpm-lockfile` to regenerate `.konflux/rpms.lock.yaml`
3. Add the `dnf install` line to the appropriate section in `Containerfile`

### Adding a new external binary

1. Add an entry to `artifacts.lock.yaml` with the download URL, checksum, and
   filename (per-arch if needed)
2. Add the install logic to the generic-fetcher section in `Containerfile`

## What To Avoid

- Do not add top-level imports of provider SDK packages in `src/lightspeed_agentic/providers/`.
- Do not make unit tests hit real model APIs. Live coverage belongs in `tests/e2e/`.
- Do not edit `tests/e2e/workspace/skills` without
  checking how batch Jobs mount them (`tests/e2e/batch_runner.py`, `tests/e2e/skills_fixtures.py`).
- Keep runtime dependencies aligned with the shipped container entrypoint. If
  the image invokes a module directly, declare it in `pyproject.toml`.
- Do not turn this file back into a long-form architecture tutorial. It should
  stay focused on how an agent works in this repo.

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `LIGHTSPEED_PROVIDER` | Provider type from operator (`anthropic`, `vertex`, `openai`, `azure`, `bedrock`) |
| `LIGHTSPEED_MODEL` | Model name from operator |
| `LIGHTSPEED_MODEL_PROVIDER` | Model provider for Vertex (`anthropic`, `google`, `openai`) |
| `LIGHTSPEED_PROVIDER_URL` | Optional API endpoint override |
| `LIGHTSPEED_PROVIDER_PROJECT` | Cloud project ID (Vertex) |
| `LIGHTSPEED_PROVIDER_REGION` | Cloud region (Vertex, Bedrock) |
| `LIGHTSPEED_PROVIDER_API_VERSION` | API version (Azure) |
| `LIGHTSPEED_AUDIT_ENABLED` | Enable audit span exporters / choice events (see audit-logging.md) |
| `LIGHTSPEED_CAPTURE_CONTENT` | Opt-out (`false`) for content on `gen_ai.choice` events; defaults on when audit is enabled |
| `LIGHTSPEED_MCP_SERVERS` | JSON array of MCP server configs |
| `LIGHTSPEED_REASONING_CONFIG` | JSON object with reasoning/thinking params, parsed at startup, passed to adapters |
| `LIGHTSPEED_SKILLS_DIR` | Skills root mounted in the sandbox pod, default `/app/skills` |
| `LIGHTSPEED_AGENTICRUN_UID` | AgenticRun UID on bridged OTLP log record attrs (templog); set by operator with OTEL endpoint |
| `LIGHTSPEED_AGENTICRUN_STEP` | AgenticRun step → `agenticrun.phase` on bridged OTLP log records |
| `ANTHROPIC_MODEL` | E2E harness input copied to `LIGHTSPEED_MODEL`; not a runtime fallback for DeepAgents routes |
| `GEMINI_MODEL` | E2E harness input copied to `LIGHTSPEED_MODEL`; Vertex Gemini runtime requires `LIGHTSPEED_MODEL` |
| `OPENAI_MODEL` | Legacy OpenAI model input when `LIGHTSPEED_MODEL` is unset |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible endpoint override |
| `CLAUDE_CODE_USE_BEDROCK` | Legacy E2E credential compatibility flag; not set by runtime config or used to select the DeepAgents model route |
| `CLAUDE_CODE_USE_VERTEX` | Legacy E2E credential compatibility flag; not set by runtime config or used to select the DeepAgents model route |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Shared OTLP endpoint for traces and logs |
| `SSL_CERT_FILE` | Combined system and operator-provided CA bundle |
| `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH` | Combined CA bundle used by gRPC |
| `AWS_CA_BUNDLE` | Combined CA bundle used by AWS/Botocore |
| `GRPC_SSL_CIPHER_SUITES` | Resolved TLS cipher suites for gRPC |
| `LIGHTSPEED_TLS_PROFILE` | Resolved OpenShift TLS profile |
| `LIGHTSPEED_TLS_MIN_VERSION` | Resolved minimum TLS version |
| `LIGHTSPEED_TLS_CIPHER_SUITES` | JSON array of resolved TLS cipher suites |
| `/var/run/secrets/lightspeed/tls/` | Shared read-only root for operator-provided CA files |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` (default) or `http/protobuf` |
| `ANTHROPIC_VERTEX_PROJECT_ID` | Vertex project for Anthropic via Vertex |
| `CLOUD_ML_REGION` | E2E harness input copied to `LIGHTSPEED_PROVIDER_REGION`; runtime Vertex region defaults to `global` |

Provider credentials such as `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
`GEMINI_API_KEY`, and `OPENAI_API_KEY` are expected by the underlying SDKs or
passed through by cluster LLM credential Secrets mounted on batch Jobs.

## Git and PR Workflow

### Commit Messages

- Start with the Jira ticket reference: `OLS-XXXX: description`
- Keep the first line under 72 characters
- Use imperative mood

### Pull Requests

This repo uses a **fork-based workflow**:

1. **Push to your fork**, not to `origin` (openshift/lightspeed-agentic-sandbox)
2. **Create the PR** against `origin/main` using your fork's branch:

   ```bash
   git push <your-fork-remote> <branch>
   gh pr create --repo openshift/lightspeed-agentic-sandbox --head <your-github-user>:<branch> --base main
   ```

3. **PR title** must start with the Jira reference: `OLS-XXXX: description`
4. **Squash commits** before pushing
