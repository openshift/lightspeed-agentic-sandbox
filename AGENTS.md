# Lightspeed Agentic Sandbox

Multi-provider agentic sandbox for OpenShift Lightspeed. This repo exposes a
FastAPI app plus provider adapters for Claude, Gemini, and OpenAI.
When editing it, optimize for thin provider wrappers, consistent event mapping,
and tests that stay offline unless you are intentionally running containerized
evals.

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
|---|---|
| System overview, integration boundaries | [system-overview.md](.ai/spec/what/system-overview.md) |
| Provider adapters | [provider-contract.md](.ai/spec/what/provider-contract.md) |
| /run endpoint | [run-api.md](.ai/spec/what/run-api.md) |
| Deployment, env vars, or defaults | [configuration.md](.ai/spec/what/configuration.md) |
| Health probes | [health-probes.md](.ai/spec/what/health-probes.md) |

### Component Specs

Each spec has a Verification section linking to BDD feature files that exercise
its rules. Use this table to navigate from component → spec → executable tests:

| Spec | Description | Feature files |
|---|---|---|
| [run-api.md](.ai/spec/what/run-api.md) | `/run` endpoint: parsing, timeouts, context prefix, streaming | [structured_output.feature](tests/e2e/features/structured_output.feature) |
| [provider-contract.md](.ai/spec/what/provider-contract.md) | Provider adapter rules: events, structured output, thin-adapter principle | [structured_output.feature](tests/e2e/features/structured_output.feature) |
| [configuration.md](.ai/spec/what/configuration.md) | Provider selection, model resolution, skills directory, env vars | [structured_output.feature](tests/e2e/features/structured_output.feature) (implicit) |

## Quick Commands

```bash
make install                           # create/update .venv with dev dependencies via uv
make install-all                       # install all providers + dev + eval extras
make lock                              # refresh uv.lock after dependency changes
make test                              # unit tests only; mocked providers, no API calls
make lint                              # ruff check src/ tests/ evals/
make format                            # ruff format + autofix
make eval                              # build image and run live evals in containers
make eval EVAL_ARGS="-k claude"        # run a subset of evals
make eval-report                       # write evals/report.json
```

## Architecture

```text
src/lightspeed_agentic/
├── app.py                # FastAPI entry point, mounts router at /v1/agent
├── factory.py            # create_provider(name) — receives SDK name from config.resolve_sdk()
├── logging.py            # Event logging helpers for query flows
├── tools.py              # Shared tool/skill utilities and defaults
├── types.py              # Provider events, query options, AgentProvider ABC
├── providers/
│   ├── claude.py         # claude-agent-sdk adapter
│   ├── gemini.py         # google-adk adapter
│   └── openai.py         # openai-agents adapter
└── routes/
    ├── __init__.py       # build_router(...)
    ├── query.py          # POST /run endpoint
    └── models.py         # Pydantic request/response models
```

| Feature | Claude (`claude-agent-sdk`) | Gemini (`google-adk`) | OpenAI (`openai-agents`) |
| --- | --- | --- | --- |
| Tools | Built-in SDK tools | Native `ExecuteBashTool` plus built-in web tools | Native `SandboxAgent` shell/filesystem/skills |
| Skills | Native `skills="all"` | Native `SkillToolset` | Native `Skills` capability |
| Structured output | `output_format` JSON schema | Native response schema path | `output_type` wrapper |
| Streaming | Partial message stream events | `StreamingMode.SSE` | `Runner.run_streamed()` |

Keep provider adapters thin. The SDK should own tool execution and skill
discovery; shared path logic belongs in `tools.py`, not in duplicated provider
helpers.

## Code Conventions

- Keep provider SDK imports inside methods or narrow helpers in provider modules.
  These SDKs are optional extras, so top-level imports must not break the base
  package import path.
- `types.py` event objects are frozen dataclasses. New event types should follow
  the same pattern and stay simple to serialize/log.
- Route payloads use Pydantic models in `routes/models.py`.
  Prefer extending those models over passing around untyped dicts.
- Providers yield async event streams; the query handler consumes async
  iterators and waits for the final result event.
- Preserve the "thin adapter" shape when touching provider files: map SDK
  events into `ProviderEvent`, do not re-implement SDK behavior locally unless a
  testable workaround is required.

## Testing Conventions

- `make test` is the default verification path for code changes. Unit tests use
  mocked providers and must not require live credentials.
- Put reusable fake providers and event fixtures in `tests/conftest.py`.
  Prefer exercising real route/provider glue over deep mocking of SDK internals.
- Route tests should build a FastAPI app with `build_router(...)` and use
  `httpx.AsyncClient` plus `ASGITransport`.
- `make eval` and `make eval-report` are integration-only checks. They build the
  container image, start one container per provider, and run evals against live
  `/v1/agent/analyze` endpoints.
- See [`evals/README.md`](evals/README.md) for eval setup, credential handling,
  provider coverage, and report details.
- Evals are container-only. If you change eval workspace fixtures, skills, or
  mounted tool behavior, verify the corresponding assumptions in `evals/run.sh`.

## Konflux Hermetic Builds

The container image is built in [Konflux](https://konflux-ci.dev/) with hermetic
builds enabled — all dependencies are prefetched and verified before the build
starts, with no network access during the build itself.

### Dependency files

| File | Purpose | How to regenerate |
|---|---|---|
| `requirements.x86_64.txt` | Python deps with hashes (x86_64) | `make requirements` |
| `requirements.aarch64.txt` | Python deps with hashes (aarch64) | `make requirements` |
| `requirements-build.txt` | Build-time deps (empty — we use wheels) | N/A |
| `rpms.in.yaml` | System RPM package list | Edit manually |
| `rpms.lock.yaml` | Resolved RPM lockfile | `make rpm-lockfile` |
| `ubi.repo` | UBI 9 repo definitions for RPM resolution | Rarely changes |
| `artifacts.lock.yaml` | External binaries (oc, ripgrep) | Edit manually, update checksums |
| `package.json` / `package-lock.json` | npm deps (claude-code CLI) | `npm install --package-lock-only` |

### Bumping dependencies

```bash
make bump-deps          # upgrade uv.lock + regenerate requirements.{arch}.txt
make rpm-lockfile       # regenerate rpms.lock.yaml (needs podman)
npm update              # update package-lock.json for claude-code
```

After bumping, commit all changed lockfiles and requirements files together.
The Konflux pipeline will prefetch the new versions on the next PR.

### Adding a new system package

1. Add the package name to `rpms.in.yaml`
2. Run `make rpm-lockfile` to regenerate `rpms.lock.yaml`
3. Add the `dnf install` line to the appropriate section in `Containerfile`

### Adding a new external binary

1. Add an entry to `artifacts.lock.yaml` with the download URL, checksum, and
   filename (per-arch if needed)
2. Add the install logic to the generic-fetcher section in `Containerfile`

## What To Avoid

- Do not add top-level imports of provider SDK packages in `src/lightspeed_agentic/providers/`.
- Do not make unit tests hit real model APIs. Live coverage belongs in `evals/`.
- Do not edit `evals/workspace/skills` or `evals/workspace/tools` without
  checking how `evals/run.sh` copies and mounts them.
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
| `LIGHTSPEED_SKILLS_DIR` | Skills root mounted by the FastAPI app, default `/app/skills` |
| `ANTHROPIC_MODEL` | Default Claude model for query routes |
| `GEMINI_MODEL` | Default Gemini model for query routes |
| `OPENAI_MODEL` | Default OpenAI model for query routes |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible endpoint override |
| `CLAUDE_CODE_USE_VERTEX` | When set to `1`, Claude uses Vertex-backed Anthropic (Claude agent SDK / Claude Code) |
| `ANTHROPIC_VERTEX_PROJECT_ID` | Vertex project for Claude via Vertex |
| `CLOUD_ML_REGION` | Vertex region for Claude via Vertex (default `us-east5`) |
| `EVAL_SERVER_URLS` | Provider-to-URL map exported by `evals/run.sh` for eval pytest fixtures |
| `EVAL_WORKSPACES` | Provider-to-output-workspace map exported by `evals/run.sh` for eval pytest fixtures |

Provider credentials such as `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
`GEMINI_API_KEY`, and `OPENAI_API_KEY` are expected by the underlying SDKs or
passed through by the eval container launcher.
