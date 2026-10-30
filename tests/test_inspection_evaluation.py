"""Offline tests for live corpus execution/reporting, not safety accuracy claims."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from lightspeed_agentic.inspection.models import ClassifierDecision

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests/evaluation/corpus_full.yaml"
SOURCE = CORPUS.with_name("corpus-source.json")
PRIVATE = "PRIVATE-CORPUS-CONTENT-never-export"


@pytest.fixture(autouse=True)
def _restore_tls_process_state():
    """The worker normally owns its process; isolate its bootstrap side effects here."""
    from lightspeed_agentic import tls

    names = (
        "SSL_CERT_FILE",
        "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",
        "AWS_CA_BUNDLE",
        "GRPC_SSL_CIPHER_SUITES",
    )
    previous_env = {name: os.environ.get(name) for name in names}
    previous_context = tls._runtime_ssl_context
    yield
    tls._runtime_ssl_context = previous_context
    for name, value in previous_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _case(case_id: str = "sample", *, expected: str = "benign", content: str = PRIVATE) -> Any:
    from tests.evaluation.runner import InspectionCase

    return InspectionCase(
        case_id=case_id,
        tool_name="get_pods",
        result_type="result",
        content=content,
        expected_outcome=expected,
        expected_category="none" if expected == "benign" else "instruction_override",
        tags=["benign" if expected == "benign" else "attack"],
    )


def test_snapshot_preserves_existing_eighteen_case_labels_and_provenance() -> None:
    from tests.evaluation.runner import load_corpus

    cases, source = load_corpus(CORPUS, SOURCE)
    assert len(cases) == 18
    assert sum(case.expected_outcome == "benign" for case in cases) == 6
    assert sum(case.expected_outcome == "malicious" for case in cases) == 12
    assert source["sha256"] == hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    assert source["source_revision"] == "e7cfa54c46850d7d2fec7524541f740b77f5a436"
    assert {"quoted", "multilingual", "error", "cluster"} <= {
        tag for case in cases for tag in case.tags
    }
    assert (
        next(c for c in cases if c.case_id == "full-instruction-override-quoted").expected_outcome
        == "malicious"
    )


@pytest.mark.parametrize("bad", ["duplicate-key", "duplicate-id", "category", "empty", "hash"])
def test_invalid_corpus_is_rejected_without_payload_diagnostics(tmp_path: Path, bad: str) -> None:
    import yaml

    from tests.evaluation.runner import load_corpus

    item = {
        "case_id": "sample",
        "tool_name": "get_pods",
        "result_type": "result",
        "content": PRIVATE,
        "expected_outcome": "benign",
        "expected_category": "none",
        "tags": ["benign"],
    }
    cases = [item, item] if bad == "duplicate-id" else [item]
    if bad == "category":
        item["expected_category"] = "instruction_override"
    text = yaml.safe_dump([] if bad == "empty" else cases)
    if bad == "duplicate-key":
        text += "  content: " + PRIVATE + "\n"
    path = tmp_path / "corpus.yaml"
    path.write_text(text)
    manifest = json.loads(SOURCE.read_text())
    manifest.update(case_count=len(cases), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if bad == "hash":
        manifest["sha256"] = "0" * 64
    source = tmp_path / "source.json"
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="corpus") as exc:
        load_corpus(path, source)
    assert PRIVATE not in str(exc.value)
    assert exc.value.__cause__ is None


def test_confusion_rates_use_classified_denominators_and_report_missing_coverage() -> None:
    from tests.evaluation.runner import CaseResult, summarize

    results = [
        CaseResult(case_id="tp", expected_outcome="malicious", observed_outcome="malicious"),
        CaseResult(case_id="tn", expected_outcome="benign", observed_outcome="benign"),
        CaseResult(case_id="fp", expected_outcome="benign", observed_outcome="malicious"),
        CaseResult(case_id="fn", expected_outcome="malicious", observed_outcome="benign"),
        CaseResult(
            case_id="error",
            expected_outcome="benign",
            observed_outcome="error",
            failure_type="timeout",
        ),
        CaseResult(
            case_id="missing",
            expected_outcome="malicious",
            observed_outcome="unverified",
            failure_type="missing_result",
        ),
    ]
    metrics = summarize(results)
    assert metrics["true_positives"] == metrics["true_negatives"] == 1
    assert metrics["false_positives"] == metrics["false_negatives"] == 1
    assert metrics["benign_total"] == metrics["malicious_total"] == 3
    assert metrics["benign_evaluated"] == metrics["malicious_evaluated"] == 2
    assert metrics["false_positive_rate"] == metrics["false_negative_rate"] == 0.5
    assert metrics["evaluated"] == 4
    assert metrics["total"] == 6
    assert metrics["coverage"] == pytest.approx(4 / 6)
    assert metrics["errors"] == 1
    assert metrics["unverified"] == 1


@pytest.mark.parametrize(
    ("expected", "expected_category", "observed", "observed_category", "mismatches"),
    [
        ("malicious", "instruction_override", "malicious", "prompt_extraction", 1),
        ("malicious", "instruction_override", "malicious", "instruction_override", 0),
        ("benign", "none", "malicious", "instruction_override", 0),
        ("malicious", "instruction_override", "benign", "none", 0),
        ("benign", "none", "benign", "none", 0),
        ("malicious", None, "malicious", "prompt_extraction", 0),
        ("malicious", "instruction_override", "error", None, 0),
        ("malicious", "instruction_override", "unverified", None, 0),
    ],
    ids=["wrong-category", "correct-category", "fp", "fn", "tn", "unlabeled", "error", "missing"],
)
def test_category_mismatches_count_only_labeled_true_positives(
    expected: str,
    expected_category: str | None,
    observed: str,
    observed_category: str | None,
    mismatches: int,
) -> None:
    from tests.evaluation.runner import CaseResult, summarize

    result = summarize(
        [
            CaseResult(
                case_id="category-check",
                expected_outcome=expected,
                expected_category=expected_category,
                observed_outcome=observed,
                observed_category=observed_category,
            )
        ]
    )
    assert result["category_mismatches"] == mismatches


def test_no_classified_cases_have_unavailable_not_zero_error_rates() -> None:
    from tests.evaluation.runner import CaseResult, summarize

    result = summarize(
        [
            CaseResult(
                case_id="error",
                expected_outcome="benign",
                observed_outcome="error",
                failure_type="provider_error",
            )
        ]
    )
    assert result["false_positive_rate"] is None
    assert result["false_negative_rate"] is None
    assert result["coverage"] == 0
    assert summarize([])["coverage"] is None


@pytest.mark.asyncio
async def test_corpus_calls_real_inspector_without_exposing_labels_or_rejected_content() -> None:
    from tests.evaluation.runner import evaluate_cases

    observed = []

    class Client:
        async def classify(
            self, request: Any, *, deadline: float | None = None
        ) -> ClassifierDecision:
            observed.append(request)
            assert deadline is not None
            assert request.content == PRIVATE
            assert "expected_outcome" not in request.model_dump()
            return ClassifierDecision(
                injection_detected=len(observed) == 2,
                category="instruction_override" if len(observed) == 2 else "none",
            )

    records = []
    results = await evaluate_cases(
        [_case("benign"), _case("attack", expected="malicious")],
        Client(),
        context_window_tokens=100_000,
        case_timeout=1,
        emit=records.append,
    )
    assert [row.observed_outcome for row in results] == ["benign", "malicious"]
    assert [row.attempts for row in results] == [1, 1]
    assert all(row.chunks_total == 1 for row in results)
    assert len(records) == 2
    assert PRIVATE not in json.dumps(records)
    assert all("content" not in row for row in records)


@pytest.mark.asyncio
async def test_classification_failures_are_not_reported_as_detected_attacks() -> None:
    from tests.evaluation.runner import evaluate_cases

    class Client:
        async def classify(self, _request: Any, *, deadline: float | None = None) -> Any:
            del deadline
            raise ValueError(PRIVATE)

    records = []
    results = await evaluate_cases(
        [_case(expected="malicious")],
        Client(),
        context_window_tokens=100_000,
        case_timeout=3,
        emit=records.append,
    )
    assert results[0].observed_outcome == "error"
    assert results[0].failure_type == "invalid_response"
    assert PRIVATE not in json.dumps(records)


@pytest.mark.asyncio
async def test_hanging_classifier_times_out_and_later_cases_are_still_recorded() -> None:
    from tests.evaluation.runner import evaluate_cases

    class Client:
        calls = 0

        async def classify(
            self, _request: Any, *, deadline: float | None = None
        ) -> ClassifierDecision:
            del deadline
            self.calls += 1
            if self.calls == 1:
                await asyncio.Event().wait()
            return ClassifierDecision(injection_detected=False, category="none")

    async with asyncio.timeout(2):
        results = await evaluate_cases(
            [_case("first"), _case("second")],
            Client(),
            context_window_tokens=100_000,
            case_timeout=0.03,
            emit=lambda _record: None,
        )
    assert results[0].observed_outcome == "error"
    assert results[0].failure_type == "timeout"
    assert results[1].observed_outcome == "benign"


@pytest.mark.asyncio
async def test_case_measurements_preserve_real_chunking_not_one_case_per_request() -> None:
    from tests.evaluation.runner import evaluate_cases

    class Client:
        async def classify(
            self, _request: Any, *, deadline: float | None = None
        ) -> ClassifierDecision:
            del deadline
            return ClassifierDecision(injection_detected=False, category="none")

    results = await evaluate_cases(
        [_case(content="正常 " * 300)],
        Client(),
        context_window_tokens=1024,
        case_timeout=1,
        emit=lambda _record: None,
    )
    assert len(results) == 1
    assert results[0].chunks_total > 1
    assert results[0].attempts == results[0].chunks_total


@pytest.mark.asyncio
async def test_evaluation_cancellation_propagates() -> None:
    from tests.evaluation.runner import evaluate_cases

    class Client:
        async def classify(self, _request: Any, *, deadline: float | None = None) -> Any:
            del deadline
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await evaluate_cases(
            [_case()],
            Client(),
            context_window_tokens=100_000,
            case_timeout=1,
            emit=lambda _record: None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "profile", "context_tokens"),
    [
        ("none", {"max_input_tokens": 4096}, 4096),
        ("none", {"max_context_size": 8192}, 8192),
        ("none", {}, 100_000),
        ("wrapper", {}, 100_000),
        ("cleanup", {}, 100_000),
        ("cancel", {}, 100_000),
    ],
)
async def test_configured_run_uses_production_route_settings_and_closes_created_clients(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    profile: dict[str, int],
    context_tokens: int,
) -> None:
    from lightspeed_agentic.config import ModelParameters, ResolvedModelFamily
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute
    from tests.evaluation import runner

    closed = []
    construction = []

    class Resource:
        async def aclose(self) -> None:
            closed.append("closed")
            if failure == "cleanup":
                raise ValueError(PRIVATE)

    model = SimpleNamespace(profile=profile, _lightspeed_owned_resources=[Resource()])

    class Client:
        async def classify(
            self, _request: Any, *, deadline: float | None = None
        ) -> ClassifierDecision:
            del deadline
            if failure == "cancel":
                raise asyncio.CancelledError
            # Deliberate false positive: measurements must not introduce an accuracy gate.
            return ClassifierDecision(injection_detected=True, category="instruction_override")

    def create_model(parameters: Any, reasoning: Any) -> Any:
        construction.append((parameters.model, reasoning))
        return model

    def create_classifier(actual_model: Any) -> Client:
        assert actual_model is model
        assert actual_model.disable_streaming is True
        if failure == "wrapper":
            raise ValueError(PRIVATE)
        return Client()

    family = ResolvedModelFamily("openai", (), hosting="openai", model="test-model")
    route = ResolvedModelRoute(
        route="openai",
        model="test-model",
        telemetry_provider="openai",
        model_parameters=ModelParameters(model="test-model"),
        create_model=create_model,
        structured_output_method="function_calling",
        create_classifier_client=create_classifier,
    )
    monkeypatch.setattr(runner, "resolve_model_family", lambda: family)
    monkeypatch.setattr(runner, "resolve_model_route", lambda **_kwargs: route)
    monkeypatch.setattr(runner, "run_readiness_checks", lambda *_args: (True, {}))
    records = []
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await runner.run_configured([_case()], case_timeout=1, emit=records.append)
        assert closed == ["closed"]
        assert not any(row["record_type"] == "end" for row in records)
        return
    status = await runner.run_configured([_case()], case_timeout=1, emit=records.append)
    assert status == (0 if failure == "none" else 2)
    assert construction == [("test-model", None)]
    assert closed == ["closed"]
    assert PRIVATE not in json.dumps(records)
    assert records[-1]["record_type"] == "end"
    if failure == "none":
        assert (
            next(row for row in records if row["record_type"] == "metadata")[
                "context_window_tokens"
            ]
            == context_tokens
        )
        assert (
            next(row for row in records if row["record_type"] == "case")["observed_outcome"]
            == "malicious"
        )


def test_job_uses_pinned_runtime_image_and_mounts_evaluator_without_runtime_changes() -> None:
    from tests.evaluation.batch import build_evaluation_job
    from tests.test_inspection_e2e import _config

    job = build_evaluation_job(
        _config(), "eval-job", "eval-input", "run-uid", case_count=18, case_timeout=60
    )
    spec = job.spec["template"]["spec"]
    container = spec["containers"][0]
    assert container["image"] == "quay.io/example/sandbox@sha256:" + "a" * 64
    assert container["args"] == [
        "python3.12",
        "/input/runner.py",
        "--dataset",
        "/input/corpus_full.yaml",
        "--source",
        "/input/corpus-source.json",
        "--case-timeout",
        "60",
    ]
    assert "command" not in container  # Preserve the image's signal-forwarding init.
    assert spec["volumes"][0]["configMap"]["name"] == "eval-input"
    assert job.metadata.labels["agentic.openshift.io/component"] == "sandbox-inspection-eval"
    assert job.spec["activeDeadlineSeconds"] >= 18 * 60
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED"] == "true"
    assert env["LIGHTSPEED_AUDIT_ENABLED"] == "false"


def _framed_records(*, end: bool = True, extra: str = "") -> str:
    metadata = {
        "record_type": "metadata",
        "route": "openai",
        "provider": "openai",
        "hosting": "openai",
        "model": "test-model",
        "auth_mode": "api_key",
        "endpoint_origin": None,
        "endpoint_details_redacted": False,
        "region": None,
        "project": None,
        "api_version": None,
        "context_window_tokens": 100000,
        "instruction_tokens": 512,
        "reasoning": None,
        "tools": False,
        "history": False,
    }
    case = {
        "record_type": "case",
        "case_id": "sample",
        "expected_outcome": "benign",
        "expected_category": "none",
        "tags": ["benign"],
        "observed_outcome": "malicious",
        "observed_category": "instruction_override",
        "failure_type": None,
        "attempts": 1,
        "chunks_total": 1,
        "chunks_visited": 1,
        "duration_ms": 1.0,
    }
    lines = [
        "ignored SDK diagnostic " + PRIVATE,
        "INSPECTION_EVAL " + json.dumps(metadata),
        "INSPECTION_EVAL " + json.dumps(case),
    ]
    if extra:
        lines.append(extra)
    if end:
        lines.append('INSPECTION_EVAL {"record_type":"end","case_count":1,"exit_code":0}')
    return "\n".join(lines)


def test_full_false_positive_measurement_is_not_an_invented_accuracy_failure() -> None:
    from tests.evaluation.batch import parse_evaluation

    report = parse_evaluation(
        [_case()],
        _framed_records(),
        job_succeeded=True,
        container_exit_code=0,
        image_id="sandbox@sha256:" + "b" * 64,
    )
    assert report["operational_success"] is True
    assert report["metrics"]["false_positive_rate"] == 1
    assert report["metrics"]["false_positive_rate"] != 0
    assert report["metrics"]["false_negative_rate"] is None
    assert report["runtime"]["model"] == "test-model"
    assert PRIVATE not in json.dumps(report)


@pytest.mark.parametrize(
    "fault",
    [
        "missing-end",
        "duplicate",
        "unknown-case",
        "bad-label",
        "extra-payload",
        "bad-image",
        "exit-mismatch",
        "end-before-case",
        "case-before-metadata",
    ],
)
def test_incomplete_or_corrupted_protocol_cannot_claim_success(fault: str) -> None:
    from tests.evaluation.batch import parse_evaluation

    logs = _framed_records(end=fault != "missing-end")
    if fault in {"end-before-case", "case-before-metadata"}:
        lines = logs.splitlines()
        order = [0, 1, 3, 2] if fault == "end-before-case" else [0, 2, 1, 3]
        logs = "\n".join(lines[index] for index in order)
    if fault == "duplicate":
        logs += "\n" + next(line for line in logs.splitlines() if '"record_type": "case"' in line)
    if fault == "unknown-case":
        logs = logs.replace('"case_id": "sample"', '"case_id": "other"')
    if fault == "bad-label":
        logs = logs.replace('"expected_outcome": "benign"', '"expected_outcome": "malicious"')
    if fault == "extra-payload":
        logs = logs.replace(
            '"duration_ms": 1.0', '"duration_ms": 1.0, "content": "' + PRIVATE + '"'
        )
    report = parse_evaluation(
        [_case()],
        logs,
        job_succeeded=True,
        container_exit_code=1 if fault == "exit-mismatch" else 0,
        image_id=None if fault == "bad-image" else "sandbox@sha256:" + "b" * 64,
    )
    assert report["operational_success"] is False
    assert PRIVATE not in json.dumps(report)
    assert report["failures"]


def test_interrupted_corpus_keeps_completed_measurements_and_missing_cases_unverified() -> None:
    from tests.evaluation.batch import parse_evaluation

    report = parse_evaluation(
        [_case(), _case("pending", expected="malicious")],
        _framed_records(end=False),
        job_succeeded=False,
        container_exit_code=137,
        image_id="sandbox@sha256:" + "b" * 64,
    )
    assert report["metrics"]["false_positives"] == 1
    assert report["metrics"]["unverified"] == 1
    assert report["metrics"]["false_negatives"] == 0
    assert report["metrics"]["false_negative_rate"] is None
    assert report["operational_success"] is False


@pytest.mark.parametrize("failure", ["config_map_create", "job_create", "owner_attach"])
def test_cluster_failure_still_writes_redacted_full_corpus_unverified_report(
    tmp_path: Path,
    failure: str,
) -> None:
    from kubernetes.client import ApiException

    from tests.evaluation.batch import run_evaluation
    from tests.test_inspection_e2e import _config

    core, jobs = MagicMock(), MagicMock()
    jobs.create_namespaced_job.return_value.metadata.uid = "job-uid"
    operation = {
        "config_map_create": core.create_namespaced_config_map,
        "job_create": jobs.create_namespaced_job,
        "owner_attach": core.read_namespaced_config_map,
    }[failure]
    operation.side_effect = ApiException(status=403, reason=PRIVATE)
    status = run_evaluation(_config(), core, jobs, output_dir=tmp_path, case_timeout=1)
    # An API error can arrive after the resource was created; clean up by unique name.
    core.delete_namespaced_config_map.assert_called_once()
    if failure in {"job_create", "owner_attach"}:
        jobs.delete_namespaced_job.assert_called_once()
    assert status != 0
    raw = next(tmp_path.glob("inspection-evaluation-*.json")).read_text()
    assert PRIVATE not in raw
    report = json.loads(raw)
    assert report["metrics"]["total"] == report["metrics"]["unverified"] == 18
    assert report["metrics"]["false_positive_rate"] is None
    assert report["metrics"]["false_negative_rate"] is None
    assert report["corpus"]["sha256"] == hashlib.sha256(CORPUS.read_bytes()).hexdigest()


class _PollingClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


def _evaluation_cluster(
    *, complete: bool = False, exit_code: int | None = None
) -> tuple[Any, Any, dict[str, Any]]:
    """Script only Kubernetes resources/logs; retain the real launcher and parser."""
    from io import BytesIO

    from kubernetes.client import ApiException
    from urllib3.response import HTTPResponse

    from tests.evaluation.runner import PREFIX, CaseResult, load_corpus
    from tests.test_inspection_e2e import _container, _core_api

    cases, _source = load_corpus(CORPUS, SOURCE)
    selected = (
        cases if complete else [next(case for case in cases if case.expected_outcome == "benign")]
    )
    records = [
        line for line in _framed_records().splitlines() if '"record_type": "metadata"' in line
    ]
    for case in selected:
        row = CaseResult(
            case_id=case.case_id,
            expected_outcome=case.expected_outcome,
            expected_category=case.expected_category,
            tags=case.tags,
            observed_outcome="benign",
            observed_category="none",
            attempts=1,
            chunks_total=1,
            chunks_visited=1,
        )
        records.append(PREFIX + json.dumps({"record_type": "case", **row.model_dump()}))
    if complete:
        records.append(
            PREFIX + json.dumps({"record_type": "end", "case_count": 18, "exit_code": exit_code})
        )
    core = _core_api([_container("agent", exit_code, "")])
    core.read_namespaced_pod_log.side_effect = lambda **_kwargs: HTTPResponse(
        body=BytesIO("\n".join(records).encode()), status=200, preload_content=False
    )
    state: dict[str, Any] = {"exists": False, "delete_status": None, "reads": 0}
    jobs = MagicMock()

    def create(namespace: str, job: Any) -> Any:
        state.update(exists=True, name=job.metadata.name, namespace=namespace)
        job.metadata.uid = "job-uid"
        return job

    def delete(name: str, namespace: str, **kwargs: Any) -> None:
        assert name == state["name"]
        assert namespace == state["namespace"]
        assert kwargs["propagation_policy"] == "Background"
        assert all(0 < value <= 10 for value in kwargs["_request_timeout"])
        code = state["delete_status"]
        if code == 403:
            raise ApiException(status=403, reason=PRIVATE)
        state["exists"] = False
        if code == 404:
            raise ApiException(status=404, reason=PRIVATE)

    jobs.create_namespaced_job.side_effect = create
    jobs.delete_namespaced_job.side_effect = delete
    return core, jobs, state


def test_corpus_report_retains_identity_hashes_without_git_or_subprocesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.evaluation.batch import run_evaluation
    from tests.test_inspection_e2e import _config

    core, jobs, _state = _evaluation_cluster(complete=True, exit_code=0)
    jobs.read_namespaced_job.return_value.status = SimpleNamespace(succeeded=1, failed=None)

    def no_subprocess(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Corpus reporting must not launch subprocesses")

    monkeypatch.setattr(subprocess, "Popen", no_subprocess)
    status = run_evaluation(_config(), core, jobs, output_dir=tmp_path, case_timeout=1)
    assert status == 0
    report = json.loads(next(tmp_path.glob("inspection-evaluation-*.json")).read_text())
    assert report["operational_success"] is True
    assert report["metrics"]["evaluated"] == 18
    assert report["image_id"] == "quay.io/example/sandbox@sha256:" + "b" * 64
    assert report["corpus"]["sha256"] == hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    assert (
        report["runner_sha256"]
        == hashlib.sha256(CORPUS.with_name("runner.py").read_bytes()).hexdigest()
    )
    assert "test_source_revision" not in report
    assert "test_source_dirty" not in report


@pytest.mark.parametrize("fault", ["forbidden", "server", "transport", "deadline"])
def test_aborted_monitoring_stops_owned_job_and_retains_partial_measurements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    from kubernetes.client import ApiException
    from urllib3.exceptions import ProtocolError

    from tests.evaluation import batch
    from tests.test_inspection_e2e import _config

    core, jobs, state = _evaluation_cluster()
    clock = _PollingClock()
    monkeypatch.setattr(batch, "time", clock)

    def read(*_args: Any, **_kwargs: Any) -> Any:
        state["reads"] += 1
        if fault == "deadline":
            clock.now += 10_000
        if fault == "transport":
            raise ProtocolError(PRIVATE)
        raise ApiException(status=403 if fault == "forbidden" else 503, reason=PRIVATE)

    jobs.read_namespaced_job.side_effect = read
    status = batch.run_evaluation(_config(), core, jobs, output_dir=tmp_path, case_timeout=1)
    assert status == 1
    assert not state["exists"], "A failed monitor must not leave model calls running"
    raw = next(tmp_path.glob("inspection-evaluation-*.json")).read_text()
    assert PRIVATE not in raw
    report = json.loads(raw)
    assert report["operational_success"] is False
    assert report["metrics"]["true_negatives"] == report["metrics"]["evaluated"] == 1
    assert report["metrics"]["unverified"] == 17
    assert report["metrics"]["false_negatives"] == 0
    assert report["metrics"]["false_negative_rate"] is None
    assert "job_monitoring_error" in report["failures"]
    assert state["reads"] == (1 if fault in {"forbidden", "deadline"} else 3)
    if fault == "deadline":
        assert sum(clock.sleeps) == 0, "A retry must not extend an exhausted monitoring budget"
    for call in core.list_namespaced_pod.call_args_list:
        assert all(0 < value <= 10 for value in call.kwargs["_request_timeout"])
    assert all(
        0 < value <= 10
        for value in core.read_namespaced_pod_log.call_args.kwargs["_request_timeout"]
    )


@pytest.mark.parametrize("delete_status", [None, 403, 404])
@pytest.mark.parametrize("phase", ["poll", "observation"])
def test_ctrl_c_retains_partial_report_and_reraises_even_when_cleanup_fails(
    tmp_path: Path,
    delete_status: int | None,
    phase: str,
) -> None:
    from kubernetes.client import ApiException

    from tests.evaluation.batch import run_evaluation
    from tests.test_inspection_e2e import _config

    core, jobs, state = _evaluation_cluster()
    state["delete_status"] = delete_status
    interrupt = KeyboardInterrupt()
    if phase == "poll":
        jobs.read_namespaced_job.side_effect = interrupt
    else:
        jobs.read_namespaced_job.side_effect = ApiException(status=403, reason=PRIVATE)
        core.list_namespaced_pod.side_effect = [core.list_namespaced_pod.return_value, interrupt]
    with pytest.raises(KeyboardInterrupt) as caught:
        run_evaluation(_config(), core, jobs, output_dir=tmp_path, case_timeout=1)
    assert caught.value is interrupt
    reports = list(tmp_path.glob("inspection-evaluation-*.json"))
    assert len(reports) == 1, "Ctrl-C must retain an unverified partial report"
    raw = reports[0].read_text()
    assert PRIVATE not in raw
    report = json.loads(raw)
    assert report["metrics"]["evaluated"] == 1
    assert report["metrics"]["unverified"] == 17
    assert report["operational_success"] is False
    assert "cancelled" in report["failures"]
    assert ("job_cleanup_error" in report["failures"]) is (delete_status == 403)
    assert state["exists"] is (delete_status == 403)


@pytest.mark.parametrize(
    "phase", ["poll", "observation", "job-cleanup", "input-cleanup", "poll-and-cleanup"]
)
def test_pytest_failure_retains_partial_report_and_reraises_original_failure(
    tmp_path: Path, phase: str
) -> None:
    from kubernetes.client import ApiException

    from tests.evaluation.batch import run_evaluation
    from tests.test_inspection_e2e import _config

    core, jobs, state = _evaluation_cluster()
    failure = pytest.fail.Exception(PRIVATE)
    if phase in {"poll", "poll-and-cleanup"}:
        jobs.read_namespaced_job.side_effect = failure
        if phase == "poll-and-cleanup":
            jobs.delete_namespaced_job.side_effect = pytest.fail.Exception("secondary-" + PRIVATE)
    elif phase == "observation":
        jobs.read_namespaced_job.side_effect = ApiException(status=403, reason=PRIVATE)
        core.list_namespaced_pod.side_effect = [core.list_namespaced_pod.return_value, failure]
    elif phase == "job-cleanup":
        jobs.read_namespaced_job.side_effect = ApiException(status=403, reason=PRIVATE)
        jobs.delete_namespaced_job.side_effect = failure
    else:
        core.read_namespaced_config_map.side_effect = ApiException(status=403, reason=PRIVATE)
        core.delete_namespaced_config_map.side_effect = failure
    with pytest.raises(pytest.fail.Exception) as caught:
        run_evaluation(_config(), core, jobs, output_dir=tmp_path, case_timeout=1)
    assert caught.value is failure
    reports = list(tmp_path.glob("inspection-evaluation-*.json"))
    assert len(reports) == 1, "Pytest failures must not discard available measurements"
    raw = reports[0].read_text()
    assert PRIVATE not in raw
    report = json.loads(raw)
    assert report["metrics"]["true_negatives"] == report["metrics"]["evaluated"] == 1
    assert report["metrics"]["unverified"] == 17
    assert report["metrics"]["false_negatives"] == 0
    assert report["metrics"]["false_negative_rate"] is None
    assert report["operational_success"] is False
    assert "cancelled" in report["failures"]
    assert ("job_cleanup_error" in report["failures"]) is (
        phase in {"job-cleanup", "poll-and-cleanup"}
    )
    assert ("input_cleanup_error" in report["failures"]) is (phase == "input-cleanup")
    assert jobs.delete_namespaced_job.call_count == 1
    assert state["exists"] is (phase in {"job-cleanup", "poll-and-cleanup"})


def test_report_storage_failure_does_not_replace_original_pytest_failure(tmp_path: Path) -> None:
    from tests.evaluation.batch import run_evaluation
    from tests.test_inspection_e2e import _config

    core, jobs, state = _evaluation_cluster()
    failure = pytest.fail.Exception(PRIVATE)
    jobs.read_namespaced_job.side_effect = failure

    output_dir = tmp_path / "not-a-directory"
    output_dir.write_text("Blocked report storage")
    with pytest.raises(pytest.fail.Exception) as caught:
        run_evaluation(_config(), core, jobs, output_dir=output_dir, case_timeout=1)
    assert caught.value is failure
    assert state["exists"] is False


@pytest.mark.parametrize("fault", ["server", "transport", "interleaved"])
def test_transient_monitoring_errors_recover_without_relaunching_or_deleting_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    from kubernetes.client import ApiException
    from urllib3.exceptions import ProtocolError

    from tests.evaluation import batch
    from tests.test_inspection_e2e import _config

    core, jobs, state = _evaluation_cluster(complete=True, exit_code=0)
    clock = _PollingClock()
    monkeypatch.setattr(batch, "time", clock)
    replies = iter(
        [
            ApiException(status=503, reason=PRIVATE)
            if fault == "server"
            else ProtocolError(PRIVATE),
            ApiException(status=429, reason=PRIVATE),
            SimpleNamespace(status=SimpleNamespace(succeeded=1, failed=None)),
        ]
    )

    if fault == "interleaved":
        replies = iter(
            [
                ApiException(status=503, reason=PRIVATE),
                SimpleNamespace(status=SimpleNamespace(succeeded=None, failed=None)),
                ApiException(status=503, reason=PRIVATE),
                ApiException(status=429, reason=PRIVATE),
                SimpleNamespace(status=SimpleNamespace(succeeded=1, failed=None)),
            ]
        )

    def read(*_args: Any, **_kwargs: Any) -> Any:
        state["reads"] += 1
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    jobs.read_namespaced_job.side_effect = read
    status = batch.run_evaluation(_config(), core, jobs, output_dir=tmp_path, case_timeout=1)
    assert status == 0
    assert state["exists"], "Completed Jobs should retain their normal TTL/debug lifecycle"
    assert len(jobs.create_namespaced_job.call_args_list) == 1
    assert state["reads"] == (5 if fault == "interleaved" else 3)
    assert len(clock.sleeps) == (4 if fault == "interleaved" else 2)
    for call in jobs.read_namespaced_job.call_args_list:
        assert all(0 < value <= 10 for value in call.kwargs["_request_timeout"])
    report = json.loads(next(tmp_path.glob("inspection-evaluation-*.json")).read_text())
    assert report["operational_success"] is True
    assert report["metrics"]["evaluated"] == 18
    assert report["failures"] == []


def test_failed_partial_observation_does_not_prevent_owned_job_cleanup(tmp_path: Path) -> None:
    from kubernetes.client import ApiException

    from tests.evaluation.batch import run_evaluation
    from tests.test_inspection_e2e import _config

    core, jobs, state = _evaluation_cluster()
    jobs.read_namespaced_job.side_effect = ApiException(status=403, reason=PRIVATE)
    core.list_namespaced_pod.side_effect = ApiException(status=503, reason=PRIVATE)
    assert run_evaluation(_config(), core, jobs, output_dir=tmp_path, case_timeout=1) == 1
    assert not state["exists"]
    raw = next(tmp_path.glob("inspection-evaluation-*.json")).read_text()
    assert PRIVATE not in raw
    report = json.loads(raw)
    assert report["metrics"]["evaluated"] == 0
    assert report["metrics"]["unverified"] == 18
    assert report["metrics"]["false_positive_rate"] is None
    assert report["metrics"]["false_negative_rate"] is None
    assert {"job_monitoring_error", "pod_observation_error"} <= set(report["failures"])


@pytest.mark.parametrize("terminal_failure", [False, True])
def test_terminal_failure_retains_job_but_pending_job_is_stopped_at_monitor_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_failure: bool,
) -> None:
    from tests.evaluation import batch
    from tests.test_inspection_e2e import _config

    core, jobs, state = _evaluation_cluster(exit_code=1 if terminal_failure else None)
    clock = _PollingClock()
    monkeypatch.setattr(batch, "time", clock)

    def read(*_args: Any, **_kwargs: Any) -> Any:
        state["reads"] += 1
        assert state["reads"] == 1, "No polling after the remaining wait budget expires"
        # 18 cases * 1 second + 150 seconds host allowance = 168 seconds.
        clock.now += 167
        return SimpleNamespace(
            status=SimpleNamespace(succeeded=None, failed=1 if terminal_failure else None)
        )

    jobs.read_namespaced_job.side_effect = read
    status = batch.run_evaluation(_config(), core, jobs, output_dir=tmp_path, case_timeout=1)
    assert status == 1
    assert state["exists"] is terminal_failure
    assert clock.now == (167 if terminal_failure else 168)
    report = json.loads(next(tmp_path.glob("inspection-evaluation-*.json")).read_text())
    assert report["metrics"]["evaluated"] == 1
    assert report["metrics"]["unverified"] == 17
    assert ("job_monitoring_error" in report["failures"]) is (not terminal_failure)


def test_real_sigint_writes_partial_report_and_requests_job_deletion(tmp_path: Path) -> None:
    import signal
    import time

    ready = tmp_path / "ready"
    script = tmp_path / "interrupt.py"
    state_path = tmp_path / "cluster-state.json"
    script.write_text(
        "import json,time\nfrom pathlib import Path\n"
        "from tests.evaluation.batch import run_evaluation\n"
        "from tests.test_inspection_e2e import _config\n"
        "from tests.test_inspection_evaluation import _evaluation_cluster\n"
        "core,jobs,state = _evaluation_cluster()\n"
        "def read(*args, **kwargs):\n"
        f"    Path({str(ready)!r}).touch()\n"
        "    time.sleep(60)\n"
        "jobs.read_namespaced_job.side_effect = read\n"
        "try:\n"
        "    run_evaluation(_config(),core,jobs,\n"
        f"                   output_dir=Path({str(tmp_path)!r}),case_timeout=1)\n"
        "finally:\n"
        f"    Path({str(state_path)!r}).write_text(json.dumps(state))\n"
    )
    process = subprocess.Popen(  # noqa: S603 — actual runner, only Kubernetes scripted
        [sys.executable, str(script)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "Child must reach actual monitoring before sending SIGINT"
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode in {-signal.SIGINT, 130}
        assert PRIVATE not in stdout + stderr
        assert json.loads(state_path.read_text())["exists"] is False
        report = json.loads(next(tmp_path.glob("inspection-evaluation-*.json")).read_text())
        assert report["operational_success"] is False
        assert report["metrics"]["evaluated"] == 1
        assert report["metrics"]["unverified"] == 17
        assert "cancelled" in report["failures"]
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("provider_failure", [False, True])
async def test_evaluator_uses_actual_openai_sdk_and_isolated_structured_classifier(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    native: bool,
    provider_failure: bool,
) -> None:
    import httpx

    from lightspeed_agentic import tls
    from tests.evaluation.runner import run_configured
    from tests.test_openai_models import _openai_chat_completion, _openai_response

    captured = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        if provider_failure:
            return httpx.Response(
                400, json={"error": {"message": PRIVATE, "type": "invalid_request_error"}}
            )
        decision = {
            "injectionDetected": len(captured) == 2,
            "category": "instruction_override" if len(captured) == 2 else "none",
        }
        arguments = json.dumps(decision)
        if native:
            return httpx.Response(
                200,
                json=_openai_response(
                    "resp-eval",
                    {
                        "id": "fc-eval",
                        "type": "function_call",
                        "call_id": "call-eval",
                        "name": "ClassifierDecision",
                        "arguments": arguments,
                        "status": "completed",
                    },
                    input_tokens=3,
                    output_tokens=2,
                ),
            )
        return _openai_chat_completion(
            "chat-eval",
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-eval",
                        "type": "function",
                        "function": {"name": "ClassifierDecision", "arguments": arguments},
                    }
                ],
            },
            stream=False,
        )

    import ssl

    contexts = []
    clients = []

    def capture_transport(client: Any, **kwargs: Any) -> httpx.MockTransport:
        clients.append(client)
        contexts.append(kwargs["verify"])
        return httpx.MockTransport(respond)

    # Script only HTTP transport; keep the real TLS initialization/client factories.
    monkeypatch.setattr(httpx.Client, "_init_transport", capture_transport)
    monkeypatch.setattr(httpx.AsyncClient, "_init_transport", capture_transport)
    monkeypatch.setattr(tls, "_runtime_ssl_context", None)
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("LIGHTSPEED_TLS_MIN_VERSION", "VersionTLS13")
    monkeypatch.setenv("LIGHTSPEED_TLS_CIPHER_SUITES", '["ECDHE-RSA-AES128-GCM-SHA256"]')
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "openai")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-eval-key")
    monkeypatch.setenv(
        "LIGHTSPEED_PROVIDER_URL",
        "https://api.openai.com/v1" if native else "https://models.example/v1",
    )
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    records = []
    status = await run_configured(
        [_case("benign"), _case("attack", expected="malicious")],
        case_timeout=5,
        emit=records.append,
    )
    assert status == (1 if provider_failure else 0)
    assert len(captured) == (6 if provider_failure else 2)
    assert len(contexts) == 2
    assert all(context.minimum_version == ssl.TLSVersion.TLSv1_3 for context in contexts)
    assert all(
        [cipher["name"] for cipher in context.get_ciphers() if cipher["protocol"] != "TLSv1.3"]
        == ["ECDHE-RSA-AES128-GCM-SHA256"]
        for context in contexts
    )
    for body in captured:
        assert "reasoning" not in body
        assert "previous_response_id" not in body
        messages = body["input"] if native else body["messages"]
        request = json.loads(messages[-1]["content"])
        assert request["content"] == PRIVATE
        assert request["toolName"] == "get_pods"
        assert "expected_outcome" not in request
        assert len(messages) == 2
        tools = body["tools"]
        assert len(tools) == 1
        assert (
            tools[0]["name"] if native else tools[0]["function"]["name"]
        ) == "ClassifierDecision"
    assert all(client.is_closed for client in clients)
    assert PRIVATE not in json.dumps(records)
    assert PRIVATE not in caplog.text
    if provider_failure:
        results = [row for row in records if row["record_type"] == "case"]
        assert all(row["observed_outcome"] == "error" for row in results)
        assert all(row["failure_type"] == "provider_error" for row in results)
    metadata = next(row for row in records if row["record_type"] == "metadata")
    assert metadata["route"] == ("openai" if native else "openai-compatible")


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["LIGHTSPEED_TLS_MIN_VERSION", "LIGHTSPEED_TLS_CIPHER_SUITES"])
async def test_invalid_tls_policy_fails_safely_before_constructing_model_clients(
    monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    import httpx

    from tests.evaluation.runner import run_configured

    def unexpected_client(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Invalid TLS policy must fail before creating model transports")

    monkeypatch.setattr(httpx.Client, "_init_transport", unexpected_client)
    monkeypatch.setattr(httpx.AsyncClient, "_init_transport", unexpected_client)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "openai")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-eval-key")
    monkeypatch.setenv(setting, PRIVATE)
    records = []
    status = await run_configured([_case()], case_timeout=1, emit=records.append)
    assert status == 2
    assert records == [
        {"record_type": "run_error", "failure_type": "setup_error"},
        {"record_type": "end", "case_count": 0, "exit_code": 2},
    ]
    assert PRIVATE not in json.dumps(records)


@pytest.mark.parametrize(
    ("exit_code", "expected_status"),
    [(0, 0), (1, 1), (2, 2)],
)
def test_checked_in_shell_runs_bdd_once_without_hiding_failures(
    tmp_path: Path,
    exit_code: int,
    expected_status: int,
) -> None:
    text = (ROOT / "scripts/e2e-containers.sh").read_text()
    function = text[text.index("_run_e2e_pytest() {") : text.index("\nrun_one() {")]
    captured = tmp_path / "argv.json"
    uv = tmp_path / "fake-uv"
    uv.write_text(
        f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
        f"Path({str(captured)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        f"raise SystemExit({exit_code})\n"
    )
    uv.chmod(0o755)
    env = {
        **os.environ,
        "UV": str(uv),
        "SANDBOX_IMAGE": "sandbox@sha256:" + "a" * 64,
        "ARTIFACT_DIR": str(tmp_path),
    }
    result = subprocess.run(  # noqa: S603 — execute checked-in function against a stub UV
        ["bash", "-c", function + "\n_run_e2e_pytest openai-agents"],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected_status
    assert json.loads(captured.read_text())[:4] == ["run", "--extra", "e2e", "pytest"]


def _run_ci_dispatch(
    tmp_path: Path,
    provider: str,
    credential: str,
    *,
    precheck_exit: int = 0,
    bdd_exit: int = 0,
) -> tuple[Any, list[dict[str, Any]], Path]:
    """Execute CI's post-bootstrap script and real dispatcher; script external uv only."""
    import re

    import yaml

    pipeline = yaml.safe_load(
        (
            ROOT / ".tekton/integration-tests/pipelines/sandbox-integration-test-pipeline.yaml"
        ).read_text()
    )
    scenarios = list(
        yaml.safe_load_all(
            (
                ROOT
                / ".tekton/integration-tests/integration-test-scenarios"
                / "sandbox-integration-test-scenarios.yaml"
            ).read_text()
        )
    )
    matching = [
        scenario
        for scenario in scenarios
        if {item["name"]: item["value"] for item in scenario["spec"]["params"]}.items()
        >= {"provider": provider, "credential-secret": credential}.items()
    ]
    assert len(matching) == 1
    values = {item["name"]: item.get("default", "unused") for item in pipeline["spec"]["params"]}
    values.update({item["name"]: item["value"] for item in matching[0]["spec"]["params"]})
    values.update({"namespace": "ci-safety"})

    def resolve(value: str, params: dict[str, str]) -> str:
        return re.sub(r"\$\(params\.([^)]*)\)", lambda match: params[match[1]], value)

    task = next(task for task in pipeline["spec"]["tasks"] if task["name"] == "run-batch-e2e")
    params = {item["name"]: resolve(item["value"], values) for item in task["params"]}
    step = next(step for step in task["taskSpec"]["steps"] if step["name"] == "run-tests")
    forwarded = {
        item["name"]: resolve(item["value"], params)
        for item in step["env"]
        if item["name"]
        in {
            "PROVIDER",
            "E2E_NAMESPACE",
            "KUBECONFIG",
        }
    }
    text = (ROOT / "scripts/e2e-containers.sh").read_text()
    function = text[text.index("_run_e2e_pytest() {") : text.index("\nrun_one() {")]
    entry = tmp_path / "scripts/e2e-containers.sh"
    entry.parent.mkdir()
    entry.write_text(function + '\n_run_e2e_pytest "$1"\n')
    ledger = tmp_path / "calls.jsonl"
    artifacts = tmp_path / "artifacts"
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if args == ['sync', '--all-extras']: raise SystemExit(0)\n"
        f"if args[:3] == ['run','pytest','-q']: raise SystemExit({precheck_exit})\n"
        "if args[:4] == ['run','--extra','e2e','pytest']:\n"
        f"    phase,code = 'bdd',{bdd_exit}\n"
        "    junit = next(arg.split('=',1)[1] for arg in args if arg.startswith('--junitxml='))\n"
        "    Path(junit).write_text('<testsuite/>')\n"
        "else: raise SystemExit(99)\n"
        "row = {'phase':phase,'argv':args,\n"
        "       'provider':os.environ.get('PROVIDER'),\n"
        "       'namespace':os.environ.get('E2E_NAMESPACE'),\n"
        "       'kubeconfig':os.environ.get('KUBECONFIG')}\n"
        f"with Path({str(ledger)!r}).open('a') as stream: stream.write(json.dumps(row)+'\\n')\n"
        "raise SystemExit(code)\n"
    )
    uv.chmod(0o755)
    ci = (ROOT / ".tekton/integration-tests/scripts/run-sandbox-integration-tests.sh").read_text()
    # Credential/dependency bootstrap is external to this dispatch test.
    body = (
        ci[ci.index("set -euo pipefail") : ci.index("PROVIDER=")]
        + ci[ci.index("uv sync --all-extras\n") :]
    )
    env = {
        **os.environ,
        **forwarded,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "UV": str(uv),
        "ARTIFACT_DIR": str(artifacts),
        "SANDBOX_IMAGE": "sandbox@sha256:" + "a" * 64,
    }
    result = subprocess.run(  # noqa: S603 — checked-in CI/dispatch scripts, external process scripted
        ["bash", "-c", body],  # noqa: S607
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    calls = (
        [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    )
    return result, calls, artifacts


@pytest.mark.parametrize(
    ("provider", "credential"),
    [
        ("anthropic-vertex-deepagents", "vertex-apitoken"),
        ("anthropic-bedrock-deepagents", "bedrock-apitoken"),
        ("gemini-vertex-adk", "vertex-apitoken"),
        ("openai-agents", "openai"),
        ("anthropic-deepagents", "anthropic-apitoken"),
        ("azure-openai-agents", "azureopenai-apitoken"),
        ("azure-openai-agents", "azureopenai-entra-id"),
    ],
)
def test_every_configured_ci_provider_runs_only_the_normal_bdd_suite(
    tmp_path: Path,
    provider: str,
    credential: str,
) -> None:
    result, calls, artifacts = _run_ci_dispatch(tmp_path, provider, credential)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [call["phase"] for call in calls] == ["bdd"]
    assert [call["provider"] for call in calls] == [provider]
    assert [call["namespace"] for call in calls] == ["ci-safety"]
    assert [call["kubeconfig"] for call in calls] == ["/credentials/kubeconfig"]
    assert (artifacts / "junit_e2e.xml").exists()
    assert (artifacts / f"e2e-{provider}-summary.txt").exists()


@pytest.mark.parametrize(
    ("precheck", "bdd", "expected"),
    [(1, 0, 1), (0, 1, 1), (0, 2, 2), (1, 2, 1)],
)
def test_ci_runs_bdd_and_preserves_native_precheck_or_bdd_failure(
    tmp_path: Path,
    precheck: int,
    bdd: int,
    expected: int,
) -> None:
    result, calls, artifacts = _run_ci_dispatch(
        tmp_path,
        "openai-agents",
        "openai",
        precheck_exit=precheck,
        bdd_exit=bdd,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    assert [call["phase"] for call in calls] == ["bdd"]
    assert (artifacts / "junit_e2e.xml").exists()


def test_ci_bdd_dispatch_preserves_real_sigint(tmp_path: Path) -> None:
    import signal
    import time

    text = (ROOT / "scripts/e2e-containers.sh").read_text()
    function = text[text.index("_run_e2e_pytest() {") : text.index("\nrun_one() {")]
    ready, started = tmp_path / "bdd-ready", tmp_path / "unexpected-uv-command"
    test = tmp_path / "test_interrupt.py"
    test.write_text(
        "import time\nfrom pathlib import Path\ndef test_wait():\n"
        f"    Path({str(ready)!r}).touch()\n    time.sleep(60)\n"
    )
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\nimport os,sys\nfrom pathlib import Path\n"
        "if sys.argv[1:5] == ['run','--extra','e2e','pytest']:\n"
        f"    os.execv(sys.executable,[sys.executable,'-m','pytest','-q',{str(test)!r}])\n"
        f"Path({str(started)!r}).touch()\n"
    )
    uv.chmod(0o755)
    ci = (ROOT / ".tekton/integration-tests/scripts/run-sandbox-integration-tests.sh").read_text()
    preamble = ci[ci.index("set -euo pipefail") : ci.index("PROVIDER=")]
    process = subprocess.Popen(  # noqa: S603 — actual CI signal policy and pytest interrupted with SIGINT
        ["bash", "-c", preamble + "\n" + function + "\n_run_e2e_pytest openai-agents"],  # noqa: S607
        cwd=tmp_path,
        start_new_session=True,
        env={
            **os.environ,
            "UV": str(uv),
            "ARTIFACT_DIR": str(tmp_path),
            "SANDBOX_IMAGE": "sandbox@sha256:" + "a" * 64,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "BDD must be running before interrupting its process group"
        os.killpg(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode in {-signal.SIGINT, 130}, stdout + stderr
        assert not started.exists(), "CI must invoke only the normal BDD process"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)


def test_full_job_lifecycle_records_all_cases_without_accuracy_gating(tmp_path: Path) -> None:
    from io import BytesIO

    from urllib3.response import HTTPResponse

    from tests.evaluation.batch import run_evaluation
    from tests.evaluation.runner import PREFIX, CaseResult, load_corpus
    from tests.test_inspection_e2e import _config, _container, _core_api

    cases, _source = load_corpus(CORPUS, SOURCE)
    records = [
        line for line in _framed_records().splitlines() if '"record_type": "metadata"' in line
    ]
    for case in cases:
        row = CaseResult(
            case_id=case.case_id,
            expected_outcome=case.expected_outcome,
            expected_category=case.expected_category,
            tags=case.tags,
            observed_outcome="benign",
            observed_category="none",
            attempts=1,
            chunks_total=1,
            chunks_visited=1,
        )
        records.append(PREFIX + json.dumps({"record_type": "case", **row.model_dump()}))
    records.append(PREFIX + json.dumps({"record_type": "end", "case_count": 18, "exit_code": 0}))
    core = _core_api([_container("agent", 0, "")])
    core.read_namespaced_pod_log.side_effect = lambda **_kwargs: HTTPResponse(
        body=BytesIO("\n".join(records).encode()), status=200, preload_content=False
    )
    jobs = MagicMock()
    jobs.create_namespaced_job.return_value.metadata.uid = "job-uid"
    jobs.read_namespaced_job.return_value.status = SimpleNamespace(succeeded=1, failed=None)
    status = run_evaluation(_config(), core, jobs, output_dir=tmp_path, case_timeout=1)
    assert status == 0
    raw = next(tmp_path.glob("inspection-evaluation-*.json")).read_text()
    assert PRIVATE not in raw
    report = json.loads(raw)
    assert report["metrics"]["total"] == report["metrics"]["evaluated"] == 18
    assert report["metrics"]["false_negatives"] == 12
    assert report["metrics"]["false_negative_rate"] == 1
    assert report["accuracy_threshold"] is None
    assert report["command"] == [
        "env",
        "SANDBOX_IMAGE=quay.io/example/sandbox@sha256:" + "a" * 64,
        "bash",
        "scripts/e2e-containers.sh",
        "openai-agents",
        "-k",
        "inspection_evaluation",
    ]
    created_cm = core.create_namespaced_config_map.call_args.args[1]
    assert set(created_cm.data) == {"runner.py", "corpus_full.yaml", "corpus-source.json"}
    assert "e2e.openshift.io/session" not in created_cm.metadata.labels
    assert (
        report["runner_sha256"] == hashlib.sha256(created_cm.data["runner.py"].encode()).hexdigest()
    )
    assert jobs.create_namespaced_job.call_args.args[1].spec["activeDeadlineSeconds"] >= 18


def test_bdd_shell_setup_does_not_echo_credential_bearing_configuration(
    tmp_path: Path,
) -> None:
    text = (ROOT / "scripts/e2e-containers.sh").read_text()
    start = text.index("run_one() {")
    function = text[start : text.index("\n_install_e2e_fixtures\n", start)]
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/bash\nexit 0\n")
    uv.chmod(0o755)
    stubs = "\n".join(
        name + "() { :; }"
        for name in (
            "source_e2e_config",
            "prepare_host_llm_creds",
            "_install_provider_cluster_creds",
            "sync_provider_model",
            "_configure_batch_job_env",
            "_run_e2e_pytest",
        )
    )
    env = {
        **os.environ,
        "UV": str(uv),
        "E2E_SKIP_INSTALL": "1",
        "OPENAI_BASE_URL": f"https://user:{PRIVATE}@models.example/v1",
        "LIGHTSPEED_MCP_SERVERS": json.dumps({"headers": {"Authorization": PRIVATE}}),
    }
    result = subprocess.run(  # noqa: S603 — checked-in function, external setup commands stubbed
        ["bash", "-c", stubs + "\n" + function + "\nrun_one openai-agents"],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert PRIVATE not in result.stdout + result.stderr


@pytest.mark.parametrize("credential_url", [False, True])
def test_runtime_metadata_retains_safe_endpoint_path_without_url_credentials(
    credential_url: bool,
) -> None:
    from lightspeed_agentic.config import ResolvedModelFamily
    from lightspeed_agentic.providers.model_router import resolve_model_route
    from tests.evaluation.runner import runtime_metadata

    url = (
        f"https://user:{PRIVATE}@models.example/v1?key={PRIVATE}"
        if credential_url
        else "https://models.example/v1"
    )
    family = ResolvedModelFamily("openai", (), hosting="openai", model="test-model", base_url=url)
    result = runtime_metadata(family, resolve_model_route(model_family=family), 100_000)
    assert result["endpoint_origin"] == "https://models.example"
    assert result["endpoint_path"] == "/v1"
    assert result["endpoint_details_redacted"] is credential_url
    assert PRIVATE not in json.dumps(result)


def test_mounted_runner_has_no_host_test_imports_and_no_raw_validation_diagnostics(
    tmp_path: Path,
) -> None:
    script = tmp_path / "runner.py"
    script.write_bytes((ROOT / "tests/evaluation/runner.py").read_bytes())
    corpus = tmp_path / "corpus.yaml"
    corpus.write_text("- content: " + PRIVATE)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(  # noqa: S603 — run the checked-in standalone evaluator with invalid fixtures
        [sys.executable, str(script), "--dataset", str(corpus), "--source", str(SOURCE)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 2
    assert PRIVATE not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert json.loads(result.stdout.removeprefix("INSPECTION_EVAL ")) == {
        "record_type": "run_error",
        "failure_type": "setup_error",
    }


@pytest.mark.parametrize(
    "uid", ["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"]
)
def test_actual_ci_extract_step_preserves_unique_bdd_artifact_attempts(
    tmp_path: Path,
    uid: str,
) -> None:
    import yaml

    pipeline = yaml.safe_load(
        (
            ROOT / ".tekton/integration-tests/pipelines/sandbox-integration-test-pipeline.yaml"
        ).read_text()
    )
    task = next(task for task in pipeline["spec"]["tasks"] if task["name"] == "run-batch-e2e")
    step = next(step for step in task["taskSpec"]["steps"] if step["name"] == "extract-commit")
    commit_path, tag_path = tmp_path / "commit", tmp_path / "artifact-tag"
    script = (
        step["script"]
        .replace("$(step.results.commit.path)", str(commit_path))
        .replace("$(step.results.artifact-tag.path)", str(tag_path))
    )
    commit, provider = "a" * 40, "openai-agents"
    snapshot = json.dumps(
        {
            "components": [
                {
                    "name": "lightspeed-agentic-sandbox",
                    "source": {"git": {"revision": commit}},
                }
            ]
        }
    )
    substitutions = {
        "$(params.SNAPSHOT)": snapshot,
        "$(params.provider)": provider,
        "$(context.pipelineRun.uid)": uid,
    }
    env = dict(os.environ)
    for item in step["env"]:
        value = item["value"]
        for expression, replacement in substitutions.items():
            value = value.replace(expression, replacement)
        env[item["name"]] = value
    result = subprocess.run(  # noqa: S603 — actual checked-in CI script, controlled snapshot
        ["bash", "-c", script],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert commit_path.read_text() == commit
    assert tag_path.exists(), "CI must retain each BDD/corpus artifact attempt"
    expected_tag = f"ols-agentic-sandbox-integration-{provider}-{uid}"
    assert tag_path.read_text() == expected_tag
    push = next(step for step in task["taskSpec"]["steps"] if step["name"] == "push-artifacts")
    reference = next(param["value"] for param in push["params"] if param["name"] == "oci-ref")
    resolved = (
        reference.replace("$(params.provider)", provider)
        .replace("$(steps.extract-commit.results.commit)", commit)
        .replace("$(steps.extract-commit.results.artifact-tag)", tag_path.read_text())
    )
    assert resolved == "quay.io/openshift-lightspeed/ols-operator-artifacts:" + expected_tag


def test_mutable_image_cannot_be_counted_as_reproducible_evaluation() -> None:
    from tests.evaluation.batch import build_evaluation_job
    from tests.test_inspection_e2e import _config

    with pytest.raises(ValueError, match="immutable"):
        build_evaluation_job(
            replace(_config(), sandbox_image="sandbox:latest"),
            "job",
            "input",
            "run",
            case_count=18,
            case_timeout=60,
        )
