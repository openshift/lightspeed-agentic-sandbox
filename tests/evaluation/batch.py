"""BDD helper for shared-inspector corpus Jobs and payload-free reports."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import time
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from kubernetes.client import (
    ApiException,
    BatchV1Api,
    CoreV1Api,
    V1ConfigMap,
    V1Job,
    V1ObjectMeta,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from urllib3.exceptions import HTTPError

from tests.e2e.batch_runner import (
    _build_job_spec,
    _delete_config_map_ignore_not_found,
    _fetch_agent_container_status,
    _fetch_job_pod_logs,
    _set_config_map_job_owner,
)
from tests.e2e.suite_setup import BatchE2EConfig
from tests.evaluation.runner import (
    PREFIX,
    CaseResult,
    InspectionCase,
    load_corpus,
    summarize,
)

HERE = Path(__file__).resolve().parent
_API_TIMEOUT = (3.0, 5.0)  # Connect/read timeouts for monitoring and abort cleanup.


def _wait_for_evaluation_job(
    batch_api: BatchV1Api,
    namespace: str,
    job_name: str,
    timeout_seconds: float,
) -> bool:
    """Retry only read-only polling, without resetting the overall wait budget."""
    deadline = time.monotonic() + timeout_seconds
    consecutive_errors = 0
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            job = batch_api.read_namespaced_job(
                job_name,
                namespace,
                _request_timeout=tuple(min(value, remaining) for value in _API_TIMEOUT),
            )
        except (ApiException, HTTPError) as exc:
            consecutive_errors += 1
            if (
                isinstance(exc, ApiException) and exc.status not in {429, 500, 502, 503, 504}
            ) or consecutive_errors >= 3:
                raise
        else:
            consecutive_errors = 0
            status = job.status
            if status and status.succeeded:
                return True
            if status and status.failed:
                return False
        time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
    raise TimeoutError("Evaluation monitoring deadline expired")


class RuntimeMetadata(BaseModel):
    """Reject unexpected runtime fields before writing any external report."""

    model_config = ConfigDict(extra="forbid", strict=True)
    route: Literal[
        "openai",
        "openai-compatible",
        "azure",
        "anthropic",
        "vertex/google",
        "vertex/anthropic",
        "bedrock/anthropic",
    ]
    provider: str
    hosting: Literal["openai", "azure", "anthropic", "vertex", "bedrock"]
    model: str = Field(min_length=1)
    auth_mode: Literal[
        "api_key",
        "entra_service_principal",
        "google_adc",
        "aws_sigv4",
        "aws_sigv4_assume_role",
        "bearer_token",
        "unverified",
    ]
    endpoint_origin: str | None
    endpoint_path: Literal["", "/", "/v1", "/v1/"] | None = None
    endpoint_details_redacted: bool
    region: str | None
    project: str | None
    api_version: str | None
    context_window_tokens: int = Field(gt=0)
    instruction_tokens: Literal[512]
    reasoning: None
    tools: Literal[False]
    history: Literal[False]

    @model_validator(mode="after")
    def safe_endpoint(self) -> RuntimeMetadata:
        if self.endpoint_origin:
            url = urlsplit(self.endpoint_origin)
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.path
                or url.query
                or url.fragment
            ):
                raise ValueError("unsafe endpoint metadata")
            if self.endpoint_path is None and not self.endpoint_details_redacted:
                raise ValueError("missing endpoint path metadata")
        elif self.endpoint_path is not None:
            raise ValueError("endpoint path without origin")
        return self


class EndRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    case_count: int = Field(ge=0)
    exit_code: Literal[0, 1, 2]


def build_evaluation_job(
    config: BatchE2EConfig,
    job_name: str,
    input_name: str,
    run_uid: str,
    *,
    case_count: int,
    case_timeout: float,
) -> V1Job:
    """Reuse credential/TLS mounts, but keep corpus/runner outside the shipped image."""
    if not re.search(r"@sha256:[0-9a-f]{64}$", config.sandbox_image):
        raise ValueError("Evaluation requires an immutable SANDBOX_IMAGE digest")
    if case_count < 1 or not math.isfinite(case_timeout) or case_timeout <= 0:
        raise ValueError("Evaluation requires cases and a finite positive timeout")
    labels = {
        "agentic.openshift.io/component": "sandbox-inspection-eval",
        "agentic.openshift.io/run": run_uid,
        "e2e.openshift.io/session": config.session_id,
    }
    job = _build_job_spec(
        config,
        job_name,
        input_name,
        labels,
        run_uid,
        "analysis",
        job_env_overrides={
            "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED": "true",
            "LIGHTSPEED_AUDIT_ENABLED": "false",
            "LIGHTSPEED_CAPTURE_CONTENT": "false",
        },
    )
    container = job.spec["template"]["spec"]["containers"][0]
    container["env"] = list({item["name"]: item for item in container["env"]}.values())
    container["args"] = [
        "python3.12",
        "/input/runner.py",
        "--dataset",
        "/input/corpus_full.yaml",
        "--source",
        "/input/corpus-source.json",
        "--case-timeout",
        f"{case_timeout:g}",
    ]
    job.spec["activeDeadlineSeconds"] = math.ceil(case_count * case_timeout + 120)
    return job


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len(pairs) != len({key for key, _ in pairs}):
        raise ValueError("duplicate protocol key")
    return dict(pairs)


def _valid_result(row: CaseResult, case: InspectionCase) -> bool:
    if (row.expected_outcome, row.expected_category, row.tags) != (
        case.expected_outcome,
        case.expected_category,
        case.tags,
    ):
        return False
    if row.observed_outcome in {"benign", "malicious"}:
        return (
            row.failure_type is None
            and row.observed_category is not None
            and (row.observed_outcome == "benign") == (row.observed_category == "none")
            and row.chunks_total is not None
            and 1 <= row.chunks_visited <= row.chunks_total
            and row.attempts >= row.chunks_visited
            and (row.observed_outcome != "benign" or row.chunks_visited == row.chunks_total)
        )
    return (
        row.observed_outcome == "error"
        and row.observed_category is None
        and row.failure_type
        in {"timeout", "provider_error", "invalid_response", "size_limit", "runner_error"}
    )


def parse_evaluation(
    cases: list[InspectionCase],
    logs: str,
    *,
    job_succeeded: bool,
    container_exit_code: int | None,
    image_id: str | None,
) -> dict[str, Any]:
    """Keep valid partial observations; missing/corrupted records cannot claim parity."""
    expected = {case.case_id: case for case in cases}
    observed: dict[str, CaseResult] = {}
    runtime = None
    end = None
    failures = []
    for line in logs.split("\n"):
        if not line.startswith(PREFIX):
            continue  # Never copy raw SDK logs or exception text into evidence.
        try:
            if end is not None:
                raise ValueError("protocol record after end")
            record = json.loads(line[len(PREFIX) :], object_pairs_hook=_unique_object)
            kind = record.pop("record_type")
            if kind == "metadata" and runtime is None:
                runtime = RuntimeMetadata.model_validate(record)
            elif kind == "case":
                if runtime is None:
                    raise ValueError("case without runtime identity")
                row = CaseResult.model_validate(record)
                if row.case_id not in expected or row.case_id in observed:
                    raise ValueError("unknown/duplicate case")
                if not _valid_result(row, expected[row.case_id]):
                    raise ValueError("invalid case observation")
                observed[row.case_id] = row
            elif kind == "end" and end is None:
                end = EndRecord.model_validate(record)
            elif (
                kind == "run_error"
                and set(record) == {"failure_type"}
                and record["failure_type"] in {"setup_error", "runner_error", "cleanup_error"}
            ):
                failures.append(record["failure_type"])
            else:
                raise ValueError("invalid protocol record")
        except (ValueError, TypeError, KeyError, AttributeError):
            failures.append("protocol_error")
    rows = [
        observed.get(case.case_id)
        or CaseResult(
            case_id=case.case_id,
            expected_outcome=case.expected_outcome,
            expected_category=case.expected_category,
            tags=case.tags,
            observed_outcome="unverified",
            failure_type="missing_result",
        )
        for case in cases
    ]
    if len(observed) != len(cases):
        failures.append("missing_results")
    if runtime is None or runtime.auth_mode == "unverified":
        failures.append("runtime_identity_unverified")
    if runtime and runtime.endpoint_details_redacted:
        failures.append("endpoint_details_redacted")
    if not image_id or not re.search(r"sha256:[0-9a-f]{64}$", image_id):
        failures.append("image_identity_unverified")
    if end is None or end.case_count != len(observed):
        failures.append("missing_or_invalid_end")
    if end and (end.exit_code != container_exit_code or job_succeeded != (end.exit_code == 0)):
        failures.append("exit_status_mismatch")
    if not job_succeeded or container_exit_code != 0:
        failures.append("job_execution_failed_or_unverified")
    metrics = summarize(rows)
    if metrics["errors"]:
        failures.append("case_errors")
    operational_success = not failures and job_succeeded and container_exit_code == 0
    return {
        "status": "complete" if operational_success else "incomplete_or_errors",
        "operational_success": operational_success,
        "runtime": runtime.model_dump() if runtime else None,
        "metrics": metrics,
        "results": [row.model_dump() for row in rows],
        "failures": sorted(set(failures)),
        "container_exit_code": container_exit_code,
        "image_id": image_id,
    }


def run_evaluation(
    config: BatchE2EConfig,
    core_api: CoreV1Api,
    batch_api: BatchV1Api,
    *,
    output_dir: Path,
    case_timeout: float,
) -> int:
    """Always retain a redacted report when cluster launch/execution fails."""
    cases, source = load_corpus(HERE / "corpus_full.yaml", HERE / "corpus-source.json")
    run_uid = secrets.token_hex(16)
    job_name = f"inspection-eval-{run_uid[:12]}"
    input_name = f"{job_name}-input"
    logs, image_id, exit_code = "", None, None
    job_ok = False
    launch_errors = []
    input_attempted = False
    job_attempted = False
    job_created = False
    owner_attached = False
    monitoring_complete = False
    interrupted: BaseException | None = None
    failure_type = "cluster_or_job_setup_error"
    runner_bytes = (HERE / "runner.py").read_bytes()
    try:
        job = build_evaluation_job(
            config,
            job_name,
            input_name,
            run_uid,
            case_count=len(cases),
            case_timeout=case_timeout,
        )
        input_attempted = True
        core_api.create_namespaced_config_map(
            config.namespace,
            V1ConfigMap(
                metadata=V1ObjectMeta(
                    name=input_name,
                    # Generic diagnostics select ConfigMaps by this session label.
                    # The corpus is a mounted fixture, not a developer-log artifact.
                    labels={
                        key: value
                        for key, value in job.metadata.labels.items()
                        if key != "e2e.openshift.io/session"
                    },
                ),
                data={
                    "runner.py": runner_bytes.decode("utf-8"),
                    "corpus_full.yaml": (HERE / "corpus_full.yaml").read_text(encoding="utf-8"),
                    "corpus-source.json": (HERE / "corpus-source.json").read_text(encoding="utf-8"),
                },
            ),
        )
        job_attempted = True
        created = batch_api.create_namespaced_job(config.namespace, job)
        job_created = True
        if created.metadata.uid:
            _set_config_map_job_owner(
                core_api, config.namespace, input_name, job_name, created.metadata.uid
            )
            owner_attached = True
        else:
            raise ValueError("Missing created Job identity")
        failure_type = "job_monitoring_error"
        job_ok = _wait_for_evaluation_job(
            batch_api,
            config.namespace,
            job_name,
            math.ceil(len(cases) * case_timeout + 150),
        )
        monitoring_complete = True
    except Exception:
        launch_errors.append(failure_type)
    except BaseException as exc:
        # pytest-timeout raises outside Exception. Retain evidence before re-raising
        # control-flow interruptions unchanged, just as for Ctrl-C.
        interrupted = exc
        launch_errors.append("cancelled")
    finally:
        if job_created:
            try:
                logs = _fetch_job_pod_logs(
                    core_api,
                    config.namespace,
                    job_name,
                    tail_lines=None,
                    request_timeout=_API_TIMEOUT,
                )
                status = _fetch_agent_container_status(
                    core_api,
                    config.namespace,
                    job_name,
                    request_timeout=_API_TIMEOUT,
                )
                if status:
                    image_id = status.image_id
                    if status.state and status.state.terminated:
                        exit_code = status.state.terminated.exit_code
            except Exception:
                launch_errors.append("pod_observation_error")
            except BaseException as exc:
                interrupted = interrupted or exc
                launch_errors.append("cancelled")
        if input_attempted:
            # Ownership provides TTL cleanup, not cancellation of a live Job.
            if job_attempted and (not owner_attached or not monitoring_complete):
                try:
                    batch_api.delete_namespaced_job(
                        job_name,
                        config.namespace,
                        propagation_policy="Background",
                        _request_timeout=_API_TIMEOUT,
                    )
                except ApiException as exc:
                    if exc.status != 404:
                        launch_errors.append("job_cleanup_error")
                except Exception:
                    launch_errors.append("job_cleanup_error")
                except BaseException as exc:
                    interrupted = interrupted or exc
                    launch_errors.extend(["cancelled", "job_cleanup_error"])
            if not owner_attached:
                try:
                    _delete_config_map_ignore_not_found(
                        core_api,
                        config.namespace,
                        input_name,
                        request_timeout=_API_TIMEOUT,
                    )
                except Exception:
                    launch_errors.append("input_cleanup_error")
                except BaseException as exc:
                    interrupted = interrupted or exc
                    launch_errors.extend(["cancelled", "input_cleanup_error"])
    report = parse_evaluation(
        cases,
        logs,
        job_succeeded=job_ok,
        container_exit_code=exit_code,
        image_id=image_id,
    )
    report.update(
        {
            "report_version": 1,
            "evaluation_scope": "shared_inspector_real_model_corpus",
            "corpus": source,
            "requested_image": config.sandbox_image,
            "provider_matrix_id": config.provider_name,
            "namespace": config.namespace,
            "job_name": job_name,
            "run_uid": run_uid,
            "case_timeout_seconds": case_timeout,
            "runner_sha256": hashlib.sha256(runner_bytes).hexdigest(),
            "accuracy_threshold": None,
            "credential_secret_name": config.llm_secret,
            "command": [
                "env",
                f"SANDBOX_IMAGE={config.sandbox_image}",
                "bash",
                "scripts/e2e-containers.sh",
                config.provider_name,
                "-k",
                "inspection_evaluation",
            ],
            "limitations": [
                "Small labeled baseline, not guaranteed prompt-injection resistance "
                "or full provider parity.",
                "Rates use classified cases only; denominators, errors "
                "and missing coverage are explicit.",
                "Quoted examples retain upstream malicious labels; "
                "there are no benign quoted-attack controls.",
                "The chunking-tagged fixture does not necessarily span chunks "
                "at this model's context budget.",
                "Main-agent enforcement is tested separately by the focused MCP BDD scenarios.",
                "Model identity is the resolved request; served revisions behind aliases "
                "are not inferred.",
                "Google ADC identifies the authentication mechanism, "
                "not a verified principal or impersonation chain.",
            ],
        }
    )
    if launch_errors:
        report["failures"] = sorted(set(report["failures"] + launch_errors))
        report["operational_success"] = False
        report["status"] = "incomplete_or_errors"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"inspection-evaluation-{config.provider_name}-{run_uid}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    finally:
        if interrupted is not None:
            raise interrupted from None
    return 0 if report["operational_success"] else 1
