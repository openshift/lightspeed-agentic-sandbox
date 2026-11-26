# lightspeed-agentic-sandbox

Multi-provider agentic sandbox library for OpenShift Lightspeed.

See [ARCHITECTURE.md](ARCHITECTURE.md) for architecture and
[AGENTS.md](AGENTS.md) for development guidance.

Local development uses `uv`. Run `make install` for dev dependencies,
`make install-all` for all providers plus e2e extras, and `make lock` to refresh
`uv.lock` after dependency changes.

## Dependency management

The retired Google ADK and OpenAI Agents harness libraries are removed from
`pyproject.toml`, `uv.lock`, and the generated Konflux inputs. Use the `deepagents`
extra for Anthropic (including on Bedrock) and Vertex models, `openai` for the
shared harness plus OpenAI integrations (including Azure and Bedrock Mantle), or
`all` for every supported model integration. The obsolete ADK-only `gemini`
extra is removed; Vertex Gemini remains supported through `deepagents`.

Konflux runtime/bootstrap/build requirements and both Tekton binary package
lists are regenerated for the active dependency set. Run `make test`,
`make verify`, and `uv pip check` to check the local environment and hermetic
package coverage. Version skew between the independently resolved graphs is
reported as warnings; missing packages and orphans fail verification.

## OpenAI models on AWS Bedrock Mantle

Set `LIGHTSPEED_PROVIDER=bedrock` and `LIGHTSPEED_MODEL=openai.gpt-6-luna`.
Set `LIGHTSPEED_PROVIDER_REGION` if needed; it defaults to `AWS_REGION` or
`us-east-1`. The sandbox uses `https://bedrock-mantle.<region>.api.aws` unless
`LIGHTSPEED_PROVIDER_URL` specifies a Mantle gateway. It sends Responses API
requests under `/openai/v1` using the existing Bedrock IAM credentials; no
OpenAI API key is required. Anthropic model IDs continue to use the existing
Bedrock route.

## Detecting Unused Dependencies

Run `make check-unused-deps` for an on-demand dependency audit. It runs pinned
[deptry](https://deptry.com/) through `uvx` in an isolated tool environment,
without changing application dependencies, lockfiles, or Konflux requirements.
The first run needs access to the package index.

Only the unused-dependency rule (`DEP002`) is enabled. Application, test, and
script imports count as usage; notebooks are excluded. Provider extras are
checked, while the `dev` and `e2e` extras are treated as development dependencies
and are not checked for unused packages. The target exits nonzero on findings
and is intentionally separate from `make verify`.

Treat findings as candidates, not automatic removals. Check dependency parents,
CLI entrypoints, dynamic imports, and security/compatibility constraints before
removing anything. A redundant direct declaration does not remove a package
that is still needed transitively. Non-installing uv constraints are not unused
dependency declarations.

Package/import-name mappings are configured in `pyproject.toml`. Shared
namespaces such as `opentelemetry` and `google` make usage detection approximate;
unmapped packages use installed metadata when available, otherwise normalized
package names with a warning.

## Bumping Dependencies

The release image uses RHOAI 3.5 and is built hermetically in Konflux.
Local development uses `uv.lock`; Konflux resolves independently from
`pyproject.toml` against the RHOAI and PyPI indexes, so versions can differ.

Requirements regeneration needs Podman or Docker, access to
`quay.io/syedriko/uv:prefer-index`, and network access to the RHOAI index and
PyPI. The target provisions Python 3.12 and the resolver's tools via
`uv run --no-project`. Run generation in a Linux environment: build-dependency
markers follow the host OS, so macOS generation omits Linux-only requirements.
RPM regeneration also needs a Red Hat subscription with `ACTIVATION_KEY` and
`ORG_ID` set.

```bash
make bump                              # refresh uv.lock + regenerate .konflux/requirements.* and Tekton package lists
make bump PKG=<package> VERSION=<minimum>  # single package (e.g. a CVE bump)
make rpm-lockfile                      # regenerate .konflux/rpms.lock.yaml
make verify-hermetic-requirements      # check package coverage; version skew is a warning
```

Review and commit all changed lockfiles, requirements, and Tekton files together.
See [AGENTS.md](AGENTS.md#konflux-hermetic-builds) for the build inputs,
prerequisites, and instructions for adding dependencies.

## Live vLLM integration scenario (OLS-4640)

Provision the `vllm-openai-compatible` Secret in the Konflux tenant namespace before enabling `sandbox-integration-vllm-openai-compatible`. Set `token` to the bearer credential, `endpoint` to the custom HTTPS OpenAI-compatible base URL ending in `/v1`, and `model` to the served tool-calling model ID (tested with `redhataigemma-4-31b-it-fp8-blo`). Rotate the token through the CI secret store, never through Git. The ephemeral cluster must be able to reach the endpoint with TLS verification enabled.

The scenario uses the candidate snapshot image and runs only the benign and malicious MCP inspection tests through the `openai-agents` entry. Both must pass without skips. The existing artifact uploader collects the pytest log, JUnit XML, image reference and route/model summary.

For a local reproduction on a dev cluster, configure `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY` and `OPERATOR_REPO` (the base URL ending in `/v1`, the served model ID, the bearer token, and the local operator checkout path, respectively), then run:

```bash
export E2E_REQUIRE_VLLM_ROUTE=1
export ARTIFACT_DIR="$(mktemp -d)"
bash scripts/e2e-containers.sh openai-agents -k test_tool_result_inspection_bdd --tb=short
```

Use `SANDBOX_IMAGE` to select the candidate image. Confirm both tests pass without
skips and `ARTIFACT_DIR` contains `vllm-benign-observed.txt` and
`vllm-malicious-observed.txt`. Konflux checks the no-skips requirement automatically.
