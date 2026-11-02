# Real-model inspection evaluation (OLS-4477)

An ordinary BDD scenario measures the existing shared tool-result inspector on
labeled examples. Unlike the focused MCP enforcement scenarios, it invokes only
the classifier inside the sandbox image, without running the main agent.
It introduces no classifier algorithm, runtime dependency, or accuracy threshold.

## Versioned baseline

`corpus_full.yaml` is a byte-for-byte snapshot of Classic OLS's 18-case full corpus
(6 benign, 12 malicious). `corpus-source.json` records the upstream repository,
source path, revision, SHA256, and case count. The loader rejects mismatched
hashes/counts, duplicate YAML keys/case IDs, extra fields, invalid categories, and
inconsistent labels. Change the snapshot and manifest deliberately together;
review label changes rather than silently relabeling outcomes to match a model.

The baseline covers attack categories, OpenShift output, multilingual output,
errors, JSON, quoted attacks, and a chunking-tagged fixture. **Important limits:**

- Its quoted attacks are labeled **malicious**. There are no benign quoted-attack
  controls; this is not evidence of correct handling of benign quoted attacks.
- The chunking fixture may fit in a single chunk at a provider's context budget.
  Actual chunk totals/visited chunks and classifier attempts are recorded. A tag
  is not proof that a live multi-chunk boundary was exercised.
- Eighteen examples are a small regression baseline, not a representative safety
  benchmark, an accuracy guarantee, or complete provider acceptance.

## Execution

Use an already configured provider/credential setup and reachable OpenShift
cluster. No local-cluster setup is needed or performed by offline tests.

```bash
SANDBOX_IMAGE=quay.io/openshift-lightspeed/lightspeed-agentic-sandbox@sha256:<digest> \
ARTIFACT_DIR=/tmp/inspection-evaluation \
bash scripts/e2e-containers.sh openai-agents -k inspection_evaluation
```

Select the other existing provider matrix IDs in the same way. A custom/vLLM
endpoint uses the existing OpenAI-compatible endpoint/model configuration; it is
not verified by a native OpenAI run. Azure API-key/Entra, API-version-selected
Chat Completions/Responses, and Bedrock static/assumed-role configurations require
separate runs. Do not count missing credentials, endpoints, or auth variants as
passing coverage. The corpus scenario uses normal pytest selection and `E2E_ARGS`.
Without `-k`, the usual local/CI BDD suite includes it automatically. To exclude
its paid-model calls locally, use `-k 'not inspection_evaluation'`.

The runner and corpus are mounted from a ConfigMap into the **digest-pinned
sandbox image** by a small BDD helper using the existing Kubernetes clients,
provider configuration, and credential/TLS mounts. The image entrypoint, production
TLS initialization (before model resolution/construction), configuration/model
resolver, readiness checks, native model constructor, route-specific classifier factory
(including Azure normalization), shared inspector, UTF-8 byte codec, instruction
allowance (512), context sizing/fallback, retries, and client cleanup are reused.
There is no host-side model SDK execution, main agent, MCP call, response-shaping
pass, or Result publication in this scenario. Each case has a positive finite wall
clock budget, including the shared retries. Inspection is explicitly enabled;
main-request reasoning and conversation history are not inherited. No agent tools
are attached; SDK function-calling for the `ClassifierDecision` schema is still
permitted. SDK retries and actual chunks can multiply calls and costs.

`tests/e2e/features/inspection_evaluation.feature` and its thin BDD steps call
`tests/evaluation/batch.py` as a helper, not a second host CLI or CI phase. The
in-image worker remains necessary to exercise the image's production classifier
rather than a host-side SDK. Each case has a 60-second budget including shared
retries. The Job active deadline is 20 minutes for this 18-case corpus. Its BDD
timeout is 1,350 seconds, allowing the helper's 1,230-second monitoring budget and
bounded observation/cleanup to finish instead of inheriting the suite's 600 seconds.

## Ordinary provider CI

Normal pytest BDD discovery includes the corpus scenario for all seven existing
provider/credential configurations: direct Anthropic, Anthropic on Vertex and
Bedrock, Gemini on Vertex, native OpenAI, and Azure API-key/Entra credentials.
There are no evaluation-specific pipeline parameters, dispatch modes, credentials,
or extra setup. One cluster and fixture/credential setup serves the whole BDD
suite. A complete seven-configuration cycle evaluates 126 cases; chunks and retries
can multiply paid-model calls.

An ordinary earlier BDD failure does not stop later scenarios under pytest's
normal defaults. Explicit fail-fast selection, cancellation, or bootstrap failures
can prevent execution and must not count as coverage. A failed OpenAI native
precheck still fails the provider runner without suppressing the BDD suite.
Operational evaluation failures fail its BDD scenario; misclassifications alone
remain measurements, not an invented accuracy gate. Existing scenario optionality,
`onError: continue`/final failure propagation, and secret scanning are unchanged.

The same artifact bundle retains BDD JUnit/logs, focused safety reports, and corpus
JSON. The updated pipeline uses
`quay.io/openshift-lightspeed/ols-operator-artifacts:ols-agentic-sandbox-integration-<provider>-<PipelineRun-UID>`;
its `artifact-reference` task result identifies the bundle. Unique attempts also
separate Azure credential variants. Older deployed pipeline definitions keep their
existing commit-based tag and still include the new report in that bundle.

Existing `main`-resolved pipelines already check out the tested SNAPSHOT's source
and run its ordinary BDD suite, which discovers the additional scenario. No special
PipelineRun or scenario configuration is needed. Local runs without `ARTIFACT_DIR`
retain corpus reports in the scenario's pytest temporary directory.

Inspect the ordinary provider CI results and retained reports to establish live
proof. Missing/failed/unscheduled runs remain unverified; default-on wiring is not
itself evidence of correct model decisions. Custom endpoints and auth/route
variants outside these seven configurations still require separate evidence.

## Reports and interpretation

`inspection-evaluation-<provider>-<run>.json` records:

- Corpus provenance/hash, exact mounted runner hash, requested image reference,
  observed agent container image ID, Kubernetes
  run identity, timeout, credential Secret **name**, and rerun command.
- Runtime-resolved hosting/route, requested model ID, auth mechanism, region,
  project/API version, context budget, and sanitized endpoint origin plus an
  allowlisted standard path. Hosted aliases/served revisions are not inferred.
  Google ADC is not proof of a specific principal or impersonation chain.
- Per-case labels, observed outcome/category, controlled error classification,
  duration, actual classifier attempts/chunk counts, and aggregate TP/TN/FP/FN,
  category mismatches, errors, missing coverage, and explicit denominators.

FP rate = FP / classified benign cases; FN rate = FN / classified malicious cases.
Rates are **conditional on a classification**, not on all planned cases: errors
and missing results are reported separately and never become invented benign or
malicious decisions. A missing denominator/rate is `null`, not zero. Partial runs
retain valid completed decisions and mark missing cases `unverified`.

Test-code revision is available in the surrounding CI run metadata. Report
helpers do not access Git or record a local checkout revision/dirty state.

`category_mismatches` counts wrong category labels only among true positives
(detected attacks) with an expected category. False positives and false negatives
are counted separately, not again as category mismatches.

`operational_success` means complete, valid observations and run identity—not
acceptable safety accuracy. Misclassifications alone do not fail the process;
provider/inspection errors, missing or malformed/duplicate protocol records,
unknown labels, missing image/auth metadata, redacted nonstandard endpoint
configuration, or inconsistent exit status do. A green evaluation Job does not
approve provider parity or final-image cutover; review its measurements alongside
focused MCP enforcement evidence and the outstanding acceptance matrix.

The host retries read-only Job status polling after transport errors or HTTP
429/500/502/503/504, allowing at most three consecutive failed status reads within
the original monitoring budget. Other API failures fail fast. Polling retries
never recreate the Job or restart the corpus/model evaluation. Monitoring,
observation, and abort-cleanup requests use explicit connection/read timeouts;
other batch BDD requests retain their existing defaults.

If monitoring fails, expires, receives Ctrl-C (SIGINT), or is interrupted by
pytest's signal-based timeout, the host attempts a bounded-request snapshot of
partial logs/status, then requests cascading deletion of the outstanding Job,
even after input ownership was attached. It writes an unsuccessful redacted
report with valid completed measurements and missing cases `unverified`. The
original Ctrl-C or pytest failure is re-raised after reporting, including when
interrupted during observation or cleanup; interruptions use the safe `cancelled`
failure tag, never raw exception text. Confirmed terminal Jobs,
whether successful or failed, keep the normal TTL/debug lifecycle and owned input
ConfigMap. If the API is unavailable or denies deletion, stopping is not confirmed:
the Job's active deadline remains the backstop (20 minutes at default settings).
This is graceful SIGINT/signal-timeout handling, not new SIGTERM handling;
SIGKILL and hard-exit timeout methods cannot run cleanup or reporting.

Launch failures produce a full-corpus unverified report when setup got that far.
Earlier BDD fixture/configuration failures, failures in the surrounding credential/
bootstrap script, hard termination, or unwritable artifact storage may produce no
corpus report. Those failures remain visible in pytest/CI results; missing reports
are never successful evidence.

Reports exclude corpus content, raw tool/classifier responses, reasoning, arbitrary
environment values, credential contents, endpoint userinfo/query/fragment, and raw
exceptions. The host accepts only allowlisted framed records; raw SDK logs are
not copied. Evaluation ConfigMaps deliberately omit the session label used by
generic ConfigMap diagnostics; evaluation pods use a distinct component label so
BDD pod-log collection does not copy their raw stdout. ConfigMaps are owned by the
Job for shared TTL cleanup; partial launch/ownership failures attempt targeted
cleanup. Reports use the existing artifact upload and secret scanner unchanged.
Never disable scanning to retain an evaluation artifact.

Offline coverage: `tests/test_inspection_evaluation.py` exercises real inspector
chunking/retries/deadlines, real OpenAI/native-compatible SDK parsing with scripted
HTTP, strict corpus/protocol validation, metrics/partial coverage, resource
cleanup/cancellation, the checked-in shell dispatch and CI tag script, and report
redaction. `tests/test_inspection_evaluation_bdd.py` exercises actual BDD discovery,
steps, report retention, operational failures, preceding test failures, SIGINT
cleanup, and the real signal-based pytest timeout with only Kubernetes APIs
scripted. These checks require no cluster or live
model and do not constitute real-model acceptance evidence.
