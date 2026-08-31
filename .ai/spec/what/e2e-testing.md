# E2E batch BDD harness

Meta-spec for **how** live end-to-end tests run in this repository. Behavioral
rules for the batch entrypoint, readiness helpers, and providers live in the other
`what/` specs; this document maps scenarios to those rules and records spike decisions for
[OLS-3220](https://redhat.atlassian.net/browse/OLS-3220) spike decisions and batch
harness behavior ([OLS-3926](https://redhat.atlassian.net/browse/OLS-3926) migration complete).

The sandbox no longer exposes HTTP (`app.py`, `/health`, `/ready`, `/v1/agent/run`
removed). Live BDD runs **batch Jobs** on an OpenShift cluster using the published
sandbox image (`SANDBOX_IMAGE`, default Konflux `:main` tag). Pytest on the host
creates input ConfigMaps + Jobs per scenario; step definitions assert on Result CR
status and, when needed, pod log enrichment (see [Response bodies](#response-bodies)).

## Spike findings (OLS-3220)

Investigation goal: add BDD coverage for context prefix, MCP, reasoning, structured
output, and OTEL without flaky free-text LLM assertions.

### Feasible in live batch BDD

| Area | Approach | Artifact |
|------|----------|----------|
| Context reaches the model | **Structured echo**: prepared `context` (`targetNamespaces`, `previousAttempts`, `approvedOption`) + `outputSchema`; model echoes back as response fields | [sandbox_e2e.feature](../../../tests/e2e/features/sandbox_e2e.feature) |
| OTEL traces and audit logs | Batch Job with audit enabled; poll in-cluster collector debug exporter | [sandbox_e2e.feature](../../../tests/e2e/features/sandbox_e2e.feature) |
| Structured output / skills | Batch Job per scenario | [structured_output.feature](../../../tests/e2e/features/structured_output.feature), [skills.feature](../../../tests/e2e/features/skills.feature) |
| MCP connectivity | In-cluster mock MCP (`scripts/e2e-install-fixtures.sh`); `LIGHTSPEED_MCP_SERVERS` on Jobs | [mcp.feature](../../../tests/e2e/features/mcp.feature) |
| Reasoning config | `LIGHTSPEED_REASONING_CONFIG` on Jobs (defaults per provider in `e2e-containers.sh`) | [reasoning_config.feature](../../../tests/e2e/features/reasoning_config.feature) |

Context proof is **semantic** (model output reflects injected context), not
inspection of the composed `[context]` prefix string. Exact prefix formatting
belongs in unit tests.

### Not feasible / intentionally unit-only

| Area | Reason | Artifact |
|------|--------|----------|
| Liveness / readiness HTTP probes | HTTP server removed; readiness runs in-process at batch startup | [test_ready.py](../../../tests/test_ready.py), [test_batch.py](../../../tests/test_batch.py) |
| Agent timeout (`LIGHTSPEED_AGENT_TIMEOUT_SECONDS`) | Provider SDKs do not propagate asyncio cancellation reliably enough for a deterministic live assertion | [test_run_agent.py](../../../tests/test_run_agent.py), [test_batch.py](../../../tests/test_batch.py) |
| Exact `[context]` prefix text | Deterministic formatting; no need for live LLM | [test_run_agent.py](../../../tests/test_run_agent.py) (`format_context_prefix`) |
| Empty provider result (run-api rule 23) | Requires a mocked provider; unreliable with live models | [test_run_agent.py](../../../tests/test_run_agent.py) |
| Readiness R1 when credentials missing | Needs deliberately misconfigured runtime; covered without live network | [test_ready.py](../../../tests/test_ready.py) |
| Adversarial schema hard failure (rule 22) | Live suite asserts structured envelope; unit tests cover batch failure path | [structured_output.feature](../../../tests/e2e/features/structured_output.feature), [test_run_agent.py](../../../tests/test_run_agent.py) |

### Unimplemented / uncovered

| Area | Reason | Artifact |
|------|--------|----------|
| Readiness rule R3 (MCP reachability) | Not implemented; no tracked story | — |

### Design decisions

- **Batch Job per scenario** — `tests/e2e/batch_runner.py` creates input ConfigMap +
  Job, waits for completion, reads Result CR, optionally enriches from pod logs.
  [PLANNED: OLS-3743] Every Job sets valid `LIGHTSPEED_AGENT_TIMEOUT_SECONDS`
  and `LIGHTSPEED_AGENT_MAX_TURNS` values because both become required.
- **Cluster fixtures** — `scripts/e2e-install-fixtures.sh` installs Result CRDs,
  sandbox ServiceAccount/RBAC, OTEL collector, mock MCP server. LLM creds synced via
  `scripts/e2e-install-openai-creds.sh` (or Vertex secret scripts).
- **One feature file** for OLS-3220 context/OTEL scenarios: `sandbox_e2e.feature`.
  Legacy HTTP probe and timeout scenarios removed (not applicable to batch).
- **Skills mounts** — one ConfigMap volume per skill (source at ``/mnt/e2e-skills-src/{basename}``);
  an init container copies files with ``cp -aL`` into an emptyDir at ``/app/skills/{basename}``
  so ``SKILL.md`` is a regular file (ConfigMap symlinks break OpenAI lazy skill discovery);
  writable ``.agents`` under the same emptyDir; ``E2E_OUTPUT_DIR`` for echo-token only.
- **Multi-provider matrix** — `anthropic-vertex-deepagents`, `anthropic-bedrock-deepagents`,
  `gemini-vertex-adk`, `openai-agents`; OpenAI validated most frequently on cluster.
  Bedrock Jobs MUST use a Bedrock model or inference profile ID (e.g.
  `global.anthropic.claude-haiku-4-5-20251001-v1:0` from `tests/e2e/config.env` as
  `ANTHROPIC_BEDROCK_MODEL`); Anthropic API shorthand such as `claude-haiku-4-5` is invalid on Bedrock.
- **Reasoning on every Job** — `e2e-containers.sh` sets provider-specific
  `LIGHTSPEED_REASONING_CONFIG` on all batch Jobs (not only
  `reasoning_config.feature`). Anthropic Jobs therefore combine thinking with
  structured-output scenarios (MCP, skills, echo schemas). The DeepAgents adapter
  MUST use two-phase structured output when `output_schema` is set (see
  [provider-contract.md](provider-contract.md) rule 23): agent pass with optional
  thinking, then a tool-free shape pass without thinking.

## Relationship to behavioral specs

| Behavioral spec | This harness exercises |
|-----------------|------------------------|
| [run-api.md](run-api.md) | Context wiring (rules 4, 7, 12–16); rules 21–23 via unit tests |
| [health-probes.md](health-probes.md) | Batch startup readiness (R1) in-process; no HTTP probe BDD |
| [provider-contract.md](provider-contract.md) | Structured output, skills, MCP, reasoning via feature files |
| [configuration.md](configuration.md) | Model/env/MCP/reasoning resolution via `e2e-containers.sh` and Job env |
| [audit-logging.md](audit-logging.md) | OTEL trace/log export from batch Jobs |

Do **not** duplicate behavioral rules here. When adding a scenario, update the
relevant `what/` spec Verification table first, then the feature file.

## Harness

### Layout

```text
tests/e2e/
├── features/              # Gherkin scenarios
├── steps/                 # given / when / then step definitions
├── batch_runner.py        # batch Job lifecycle + response mapping
├── batch_log_contract.py  # pod log prefix contract (EventLogger)
├── suite_setup.py         # session config + cluster preflight
├── run_result.py          # batch run result envelope for BDD steps
├── skills_fixtures.py     # skills ConfigMap helpers
├── otel_verify.py         # collector log polling
├── conftest.py            # k8s clients, batch_e2e_config, run_runner
├── credentials.py         # preflight credential checks per provider
├── config.env             # default models for e2e (sourced in clean env)
└── pytest.ini             # e2e collection config

scripts/e2e-containers.sh       # fixture install, cred sync, pytest driver
scripts/e2e-install-fixtures.sh # cluster fixtures (CRDs, OTEL, mock MCP)
scripts/e2e-install-openai-creds.sh
```

### Run mode

**OpenShift cluster (default)** — requires `oc` + `KUBECONFIG`:

```bash
make e2e openai-agents
# or: bash scripts/e2e-containers.sh openai-agents [model-override]
# matrix ids: anthropic-vertex-deepagents | anthropic-bedrock-deepagents | gemini-vertex-adk | openai-agents
# pytest filter: bash scripts/e2e-containers.sh openai-agents -- -k reasoning
```

Installs cluster fixtures (unless `E2E_SKIP_FIXTURES=1`), syncs LLM credential
Secrets, exports provider/model/MCP/reasoning env, runs pytest. Each scenario creates
a labeled batch Job in `E2E_NAMESPACE` (default `openshift-lightspeed`) using
`SANDBOX_IMAGE`.

### Environment exports

| Variable | Set by | Purpose |
|----------|--------|---------|
| `E2E_PROVIDER` | `e2e-containers.sh` | Provider matrix id for credential checks |
| `E2E_NAMESPACE` | `e2e-containers.sh` | Target namespace for Jobs and fixtures |
| `SANDBOX_IMAGE` | `e2e-containers.sh` | Batch Job container image (default Konflux `:main`) |
| `E2E_BATCH_VERIFY_FIXTURES` | `e2e-containers.sh` | When `1`, pytest verifies SA/OTEL/mock MCP fixtures |
| `LIGHTSPEED_MCP_SERVERS` | `e2e-containers.sh` | In-cluster mock MCP URL on every Job |
| `LIGHTSPEED_REASONING_CONFIG` | `e2e-containers.sh` | Provider-specific reasoning defaults on every Job |
| `LIGHTSPEED_AGENT_TIMEOUT_SECONDS` | `e2e-containers.sh` [PLANNED: OLS-3743] | Required whole-agent timeout for every Job |
| `LIGHTSPEED_AGENT_MAX_TURNS` | `e2e-containers.sh` [PLANNED: OLS-3743] | Required provider iteration cap for every Job |
| `E2E_ARGS` | user / `--` passthrough | Extra pytest args (e.g. `-v`, `-k`, single file) |
| `E2E_SKIP_FIXTURES` | user | Skip `e2e-install-fixtures.sh` when fixtures already present |
| `ARTIFACT_DIR` | CI | Pytest tee to `e2e-<provider>-pytest.log` and summary file |

### Response bodies

BDD steps assert a response envelope (`run_result.py`) built from:

1. **Result CR status** — primary source (`batch_runner._body_from_result_cr`).
2. **Pod log enrichment** — when CR status is generic (e.g. `summary: Step completed`),
   merge agent output from the `[provider:run] output:` log line emitted by
   `EventLogger` in `lightspeed_agentic.logging` (see `batch_log_contract.py`).

Then steps use batch-oriented wording:

- **`the batch job completes`** — Job finished without a harness error.
- **`the run completes successfully`** — Job finished and the agent reported `success: true`.

### Flake policy

- Prefer **structured output** and **unguessable echo values** over free-text LLM output.
- MCP scenarios must prove MCP was used (not local shell workarounds) via summary content.
- Live tests require provider credentials and a reachable cluster; skip/defer scenarios
  that need main-code changes not yet in `SANDBOX_IMAGE`.
- Clean up labeled Jobs/ConfigMaps/Result CRs between long runs (`agentic.openshift.io/component=sandbox-e2e`).

## Verification map

Feature files and unit tests are also listed under each behavioral spec. Summary:

| Feature file | Primary spec | Scenarios |
|--------------|--------------|-----------|
| [sandbox_e2e.feature](../../../tests/e2e/features/sandbox_e2e.feature) | run-api, audit-logging | Context echo, OTEL export |
| [structured_output.feature](../../../tests/e2e/features/structured_output.feature) | run-api, provider-contract | JSON schema, text fallback, adversarial schema |
| [skills.feature](../../../tests/e2e/features/skills.feature) | provider-contract | Skills mount, echo-token skill, nonskill query |
| [mcp.feature](../../../tests/e2e/features/mcp.feature) | provider-contract, configuration | MCP wiring, tool invoke, MCP tool failure envelope |
| [reasoning_config.feature](../../../tests/e2e/features/reasoning_config.feature) | provider-contract, configuration | Reasoning config passthrough |
| `troubleshooting.feature` [PLANNED: OLS-3739] | e2e-testing (troubleshooting) | Cluster-level troubleshooting scenario validation — feature file not yet implemented (see Troubleshooting section) |

Unit tests: [test_run_agent.py](../../../tests/test_run_agent.py),
[test_batch.py](../../../tests/test_batch.py),
[test_ready.py](../../../tests/test_ready.py),
[test_batch_e2e_helpers.py](../../../tests/test_batch_e2e_helpers.py) (harness helpers, no cluster),
[test_e2e_credentials.py](../../../tests/test_e2e_credentials.py) (credential preflight, including Bedrock Konflux mount).

## Troubleshooting scenario tests (OLS-3739)

Cluster-level BDD tests that exercise the full AgenticRun lifecycle against
real OpenShift clusters with injected broken states. These tests verify the
**quality and correctness of sandbox output** — phase transition testing is the
operator's responsibility (see lightspeed-agentic-operator specs).

> **Implementation status (2026-08-31).** Landed: the scenario scripts under
> `scenarios/troubleshooting/` (11 scenario dirs with `setup.sh`/`cleanup.sh`
> plus `scenario_metadata.yaml`) and the cluster conftest fixtures
> `k8s_client`, `k8s_core_client`, `scenario_cleanup` (`tests/e2e/conftest.py`).
> [PLANNED: OLS-3739] Not yet implemented — the `troubleshooting.feature` file,
> its BDD step/test module, the LLM judge module, and the `make e2e-cluster` /
> `scripts/e2e-cluster.sh` run mode. The subsections below describe the target
> design for those pieces; treat them as planned until the artifacts exist.

### Scope

- **In scope:** BDD scenarios that inject broken cluster state, create
  AgenticRun CRs via the `kubernetes` Python client, wait for completion, read
  AnalysisResult/ExecutionResult/VerificationResult CRs and sandbox pod logs,
  assert domain-keyword presence, and run an LLM judge for output relevance.
- **Out of scope:** Phase transition assertions (operator product-e2e),
  behavioral correctness of execution fixes (future work).

### Prerequisites

- Running OpenShift cluster with batch sandbox fixtures installed (see [Harness](#harness))
- Operator deployed on a live OpenShift cluster
- `kubernetes` Python client (new test dependency)
- KUBECONFIG with permissions to create/delete namespaces, deployments, and
  AgenticRun CRs
- LLM provider credentials (same as existing e2e)

### Scenarios

Troubleshooting scenario scripts live in `scenarios/troubleshooting/` at the
repository root. Each scenario directory contains `setup.sh` (inject broken
state), `cleanup.sh` (restore). A shared `scenario_metadata.yaml` at the
`scenarios/troubleshooting/` root maps scenario IDs to AgenticRun request text
and expected domain keywords.

| Scenario ID | AgenticRun request | Expected keywords |
|---|---|---|
| `envvar_missing` | Diagnose CrashLoopBackOff in warehouse-ops | `CrashLoopBackOff`, `DEPLOY_ENV` |
| `batch_failure` | Diagnose job failure | `job`, `fail` |
| `storage_binding` | Diagnose PVC issue | `PersistentVolumeClaim`, `bound` |
| `namespace_pod_count` | Count pods in fleet-alpha | `fleet-alpha`, `pod` |
| `scheduled_outage_detection` | Detect API outage window | `outage`, `03:00` |
| `periodic_failure_window` | Detect periodic failure | `failure`, `03:00` |
| `config_drift_analysis` | Diagnose connection refused | `connection refused`, `config` |
| `readiness_probe_diagnosis` | Diagnose readiness probe failure | `readiness`, `probe` |
| `ingress_rule_mismatch` | Diagnose NetworkPolicy blocking | `NetworkPolicy`, `traffic` |
| `oom` | Diagnose OOMKilled | `OOMKilled` |
| `wrong_networkpolicy` | Diagnose and fix NetworkPolicy | `NetworkPolicy` |

Setup/cleanup scripts run on the test host via `subprocess` — they are Bash scripts
that manipulate cluster state with `kubectl` (see `scenarios/troubleshooting/lib.sh`).
AgenticRun and Result CR lifecycle in tests use the `kubernetes` Python client, not shell.

### BDD structure

Feature file: `tests/e2e/features/troubleshooting.feature`

Scenario outline parametrized over the 11 scenarios. Each scenario:
1. Injects broken cluster state via `setup.sh`
2. Creates AgenticRun CR via `kubernetes.client.CustomObjectsApi`
3. Polls until AgenticRun reaches `Completed` phase (or timeout)
4. Reads AnalysisResult/ExecutionResult/VerificationResult CRs
5. Reads sandbox pod logs via `kubernetes.client.CoreV1Api`
6. Asserts expected domain keywords in result content
7. Calls LLM judge to verify output relevance
8. Runs `cleanup.sh` (always, even on failure)

Step definitions extend the existing `tests/e2e/steps/` modules. New fixtures
in `tests/e2e/conftest.py`:
- `k8s_client` — authenticated `CustomObjectsApi` from KUBECONFIG
- `k8s_core_client` — `CoreV1Api` for pod log retrieval
- `scenario_cleanup` — yield fixture ensuring cleanup.sh runs

### LLM judge

A utility module that takes scenario context (ID, request, expected keywords)
plus sandbox output (analysis/execution/verification results, pod logs) and
asks the configured LLM whether the output correctly identifies and addresses
the scenario's problem. Returns pass/fail with reasoning.

- Model configurable via `E2E_JUDGE_MODEL` env var
- Defaults to the same provider/model that ran the AgenticRun
- Uses provider credentials already available in the e2e environment

### Run mode

Cluster troubleshooting BDD is a separate entry point from batch sandbox BDD.
It requires a live cluster with the operator deployed.

```bash
make e2e-cluster <provider>
# e.g.: make e2e-cluster openai-agents
```

`scripts/e2e-cluster.sh` handles environment setup, runs only the
troubleshooting feature file, and collects artifacts.

### Environment exports

| Variable | Set by | Purpose |
|----------|--------|---------|
| `KUBECONFIG` | User/CI | Cluster access for kubernetes client |
| `E2E_JUDGE_MODEL` | Optional | Override LLM judge model (default: run provider model) |
| `E2E_SCENARIOS_DIR` | `e2e-cluster.sh` | Path to scenario scripts (default: `scenarios/troubleshooting/`) |
| `E2E_OPERATOR_NAMESPACE` | `e2e-cluster.sh` | Namespace where operator is deployed (default: `openshift-lightspeed`) |

### Flake policy

- Scenario setup/cleanup scripts MUST be idempotent
- AgenticRun polling uses configurable timeout (default: 20m per scenario)
- LLM judge assertions are logged but SHOULD NOT cause hard test failure in
  initial rollout — keyword assertions are the primary gate
- Cleanup runs in a finally/yield block regardless of test outcome

### Future work

- [PLANNED] **Behavioral correctness assertions:** verify that ExecutionResult
  actually attempted a fix (e.g., patched a resource, adjusted limits) and
  VerificationResult confirmed or denied the fix worked
- [PLANNED] **LLM judge as hard gate:** once confidence in judge reliability is
  established, promote judge pass/fail to a hard test assertion

## Commands

```bash
make install-all                         # providers + e2e extras (first time)
make test                                # unit only; no credentials
make e2e openai-agents                   # live batch BDD on OpenShift cluster
E2E_SKIP_FIXTURES=1 E2E_ARGS="-v -k mcp" bash scripts/e2e-containers.sh openai-agents
make e2e-cluster openai-agents           # cluster BDD, troubleshooting scenarios
```
