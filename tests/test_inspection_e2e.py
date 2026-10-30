"""Offline regression checks for the live safety evidence, not model accuracy."""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from kubernetes.client import (
    ApiException,
    V1ContainerState,
    V1ContainerStateTerminated,
    V1ContainerStatus,
    V1ObjectMeta,
    V1Pod,
    V1PodStatus,
)
from urllib3.response import HTTPResponse

from tests.e2e.batch_runner import RunBatchResult, _fetch_termination_message, run_batch_query
from tests.e2e.run_result import batch_to_run_result, store_run_result
from tests.e2e.suite_setup import BatchE2EConfig

pytest_plugins = ["pytester"]

SAFE_MARKER = "ToolResultSafetyInspectionFailed"
PRIVATE_PAYLOAD = "PRIVATE-PAYLOAD-must-not-appear-in-diagnostics"


def _config(**kwargs: Any) -> BatchE2EConfig:
    return BatchE2EConfig(
        namespace="test-ns",
        sandbox_image="quay.io/example/sandbox@sha256:" + "a" * 64,
        service_account="test-sa",
        llm_secret="test-credentials",  # noqa: S106 — Kubernetes Secret name
        lightspeed_provider="openai",
        model="test-model",
        provider_name="openai-agents",
        session_id="session",
        otel_endpoint="",
        otel_ca_secret="",
        verify_full_fixtures=False,
        **kwargs,
    )


def _container(name: str, exit_code: int | None, message: str = SAFE_MARKER) -> V1ContainerStatus:
    return V1ContainerStatus(
        name=name,
        image="quay.io/example/sandbox:latest",
        image_id="quay.io/example/sandbox@sha256:" + "b" * 64,
        ready=False,
        restart_count=0,
        state=V1ContainerState(
            terminated=(
                V1ContainerStateTerminated(
                    exit_code=exit_code,
                    reason="Error" if exit_code else "Completed",
                    message=message,
                )
                if exit_code is not None
                else None
            )
        ),
    )


def _core_api(statuses: list[V1ContainerStatus]) -> MagicMock:
    api = MagicMock()
    api.list_namespaced_pod.return_value = SimpleNamespace(
        items=[
            V1Pod(
                metadata=V1ObjectMeta(name="test-pod"),
                status=V1PodStatus(container_statuses=statuses),
            )
        ]
    )
    api.read_namespaced_pod_log.side_effect = lambda **_kwargs: HTTPResponse(
        body=BytesIO(b""), status=200, preload_content=False
    )
    return api


def _context(batch: RunBatchResult) -> dict[str, Any]:
    context: dict[str, Any] = {}
    store_run_result(context, batch_to_run_result(batch))
    return context


def test_batch_captures_agent_exit_and_image_not_sidecar_status() -> None:
    core = _core_api([_container("sidecar", 0, "sidecar"), _container("agent", 1)])
    jobs = MagicMock()
    jobs.read_namespaced_job.return_value.status = SimpleNamespace(succeeded=None, failed=1)
    objects = MagicMock()
    result = run_batch_query(_config(), core, jobs, objects, "test")

    assert result.container_exit_code == 1
    assert result.container_reason == "Error"
    assert result.image_id == "quay.io/example/sandbox@sha256:" + "b" * 64
    assert result.termination_message == SAFE_MARKER


def test_termination_message_comes_from_agent_not_first_container() -> None:
    core = _core_api([_container("sidecar", 0, "sidecar"), _container("agent", 1)])
    assert _fetch_termination_message(core, "test-ns", "test-job") == SAFE_MARKER


@pytest.mark.parametrize("exit_code", [None, 0, 137])
def test_inspection_failure_requires_safe_agent_termination(exit_code: int | None) -> None:
    from tests.e2e.steps.then import assert_run_fails_closed_with_inspection_error

    batch = RunBatchResult(
        job_succeeded=False,
        termination_message=SAFE_MARKER,
        container_exit_code=exit_code,
        container_reason="OOMKilled" if exit_code == 137 else "Error",
    )
    with pytest.raises(AssertionError):
        assert_run_fails_closed_with_inspection_error(_context(batch))


@pytest.mark.parametrize("message", [None, "prefix " + SAFE_MARKER, SAFE_MARKER + PRIVATE_PAYLOAD])
def test_inspection_failure_cannot_be_proved_by_log_substring(message: str | None) -> None:
    from tests.e2e.steps.then import assert_run_fails_closed_with_inspection_error

    batch = RunBatchResult(
        job_succeeded=False,
        termination_message=message,
        pod_logs=SAFE_MARKER + PRIVATE_PAYLOAD,
        container_exit_code=1,
        container_reason="Error",
    )
    with pytest.raises(AssertionError) as exc:
        assert_run_fails_closed_with_inspection_error(_context(batch))
    assert PRIVATE_PAYLOAD not in str(exc.value)


def test_inspection_failure_accepts_observed_safe_nonzero_exit() -> None:
    from tests.e2e.steps.then import assert_run_fails_closed_with_inspection_error

    batch = RunBatchResult(
        job_succeeded=False,
        termination_message=SAFE_MARKER,
        container_exit_code=1,
        container_reason="Error",
    )
    context = _context(batch)
    assert_run_fails_closed_with_inspection_error(context)
    assert context.get("inspection_checks", {}).get("safe_termination") is True


@pytest.mark.parametrize("api_status", [None, 403, 500, 404])
@pytest.mark.parametrize(
    ("step", "plural"), [("analysis", "analysisresults"), ("execution", "executionresults")]
)
def test_result_absence_requires_api_404(api_status: int | None, step: str, plural: str) -> None:
    from tests.e2e.steps.then import assert_no_result_cr

    batch = RunBatchResult(result_name="expected-result", step=step)
    context = _context(batch)
    objects = MagicMock()
    if api_status is not None:
        objects.get_namespaced_custom_object.side_effect = ApiException(status=api_status)
    else:
        objects.get_namespaced_custom_object.return_value = {"status": {}}

    if api_status == 404:
        assert_no_result_cr(context, _config(), objects)
        assert context["inspection_checks"]["no_result_cr"] is True
    else:
        with pytest.raises((AssertionError, ApiException)):
            assert_no_result_cr(context, _config(), objects)
        assert not context.get("inspection_checks", {}).get("no_result_cr", False)
    assert objects.get_namespaced_custom_object.call_args.kwargs == {
        "group": "agentic.openshift.io",
        "version": "v1alpha1",
        "namespace": "test-ns",
        "plural": plural,
        "name": "expected-result",
    }


def test_result_absence_does_not_accept_unlaunched_run() -> None:
    from tests.e2e.steps.then import assert_no_result_cr

    objects = MagicMock()
    objects.get_namespaced_custom_object.side_effect = ApiException(status=404)
    with pytest.raises(AssertionError):
        assert_no_result_cr(_context(RunBatchResult()), _config(), objects)


def test_schema_failure_diagnostics_do_not_copy_tool_content() -> None:
    from tests.e2e.steps.then import assert_jsonschema

    context = {
        "output_schema": {"type": "object", "required": ["success"]},
        "response_body": {"summary": PRIVATE_PAYLOAD},
    }
    with pytest.raises(AssertionError) as exc:
        assert_jsonschema(context)
    assert PRIVATE_PAYLOAD not in str(exc.value)


def test_successful_schema_validation_is_recorded() -> None:
    from tests.e2e.steps.then import assert_jsonschema

    context = {
        "output_schema": {"type": "object", "required": ["success"]},
        "response_body": {"success": True},
    }
    assert_jsonschema(context)
    assert context.get("inspection_checks", {}).get("schema_valid") is True


def test_evidence_report_retains_only_allowlisted_facts(tmp_path: Path) -> None:
    from tests.e2e.inspection_evidence import write_inspection_evidence

    batch = RunBatchResult(
        job_name="test-job",
        run_uid="run-uid",
        result_name="test-result",
        pod_logs=PRIVATE_PAYLOAD,
        termination_message=PRIVATE_PAYLOAD,
    )
    batch.container_exit_code = 1
    batch.image_id = "quay.io/example/sandbox@sha256:" + "b" * 64
    context = _context(batch)
    context.update(
        {
            "job_env_overrides": {
                "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED": "true",
                "API_KEY": PRIVATE_PAYLOAD,
            },
            "inspection_checks": {"safe_termination": True},
            "query": PRIVATE_PAYLOAD,
            "response_body": {"summary": PRIVATE_PAYLOAD},
            "inspection_expected_outcome": "malicious",
        }
    )
    config = _config(
        job_env={
            "OPENAI_BASE_URL": f"https://user:{PRIVATE_PAYLOAD}@models.example/v1?key={PRIVATE_PAYLOAD}",
            "LIGHTSPEED_REASONING_CONFIG": '{"effort":"low"}',
            "API_KEY": PRIVATE_PAYLOAD,
        }
    )
    write_inspection_evidence(tmp_path, config, context, "malicious-scenario", "failed")
    reports = list(tmp_path.glob("inspection-*.json"))
    assert len(reports) == 1
    raw = reports[0].read_text()
    assert PRIVATE_PAYLOAD not in raw
    report = json.loads(raw)
    assert report["test_outcome"] == "failed"
    assert report["verified"] is False
    assert report["model"] == "test-model"
    assert report["image_id"] == "quay.io/example/sandbox@sha256:" + "b" * 64
    assert report["endpoint_origin"] == "https://models.example"
    assert report["checks"] == {"safe_termination": True}
    assert report["inspection_enabled"] is True
    assert report["runner_settings"]["reasoning_config"] == {"effort": "low"}
    assert report["runner_settings"]["agent_timeout_seconds"] == 600
    assert report["runner_settings"]["agent_max_turns"] == 200
    assert "test_source_revision" not in report
    assert "test_source_dirty" not in report


@pytest.mark.parametrize(
    ("missing", "expected_gap"),
    [
        (None, None),
        ("image", "missing_observed_image_digest"),
        ("auth", "auth_mode_unverified"),
        ("limits", "invalid_execution_limits"),
        ("reasoning", "reasoning_config_not_retained"),
    ],
)
def test_safety_report_needs_no_git_but_still_requires_runtime_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str | None,
    expected_gap: str | None,
) -> None:
    from dataclasses import replace

    from tests.e2e.inspection_evidence import write_inspection_evidence

    def no_subprocess(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Evidence reporting must not launch subprocesses")

    monkeypatch.setattr("subprocess.Popen", no_subprocess)
    batch = RunBatchResult(
        run_uid="run",
        image_id=None if missing == "image" else "sandbox@sha256:" + "b" * 64,
    )
    context = _context(batch)
    context.update(
        inspection_expected_outcome="malicious",
        inspection_checks={"safe_termination": True, "no_result_cr": True, "inspection_span": True},
        job_env_overrides={"LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED": "true"},
    )
    if missing == "limits":
        context["job_env_overrides"]["LIGHTSPEED_AGENT_TIMEOUT_SECONDS"] = "0"
    if missing == "reasoning":
        context["job_env_overrides"]["LIGHTSPEED_REASONING_CONFIG"] = '{"effort":"invalid"}'
    config = _config()
    if missing == "auth":
        config = replace(config, lightspeed_provider="azure")
    write_inspection_evidence(tmp_path, config, context, "safety", "passed")
    report = json.loads(next(tmp_path.glob("inspection-*.json")).read_text())
    assert report["verified"] is (expected_gap is None)
    assert report["evidence_gaps"] == ([] if expected_gap is None else [expected_gap])
    assert "test_source_revision" not in report
    assert "test_source_dirty" not in report


@pytest.mark.parametrize(
    "settings",
    [
        {"LIGHTSPEED_AGENT_TIMEOUT_SECONDS": PRIVATE_PAYLOAD},
        {"LIGHTSPEED_AGENT_MAX_TURNS": PRIVATE_PAYLOAD},
        {"LIGHTSPEED_REASONING_CONFIG": json.dumps({"effort": PRIVATE_PAYLOAD})},
        {"LIGHTSPEED_REASONING_CONFIG": json.dumps({"thinking": {"type": [PRIVATE_PAYLOAD]}})},
    ],
)
def test_invalid_settings_do_not_erase_or_leak_failed_run_evidence(
    tmp_path: Path, settings: dict[str, str]
) -> None:
    from tests.e2e.inspection_evidence import write_inspection_evidence

    write_inspection_evidence(
        tmp_path,
        _config(job_env=settings),
        _context(RunBatchResult(run_uid="run")),
        "scenario",
        "failed",
    )
    raw = next(tmp_path.glob("inspection-*.json")).read_text()
    assert PRIVATE_PAYLOAD not in raw
    report = json.loads(raw)
    assert report["test_outcome"] == "failed"
    assert report["verified"] is False
    assert report["evidence_gaps"]


@pytest.mark.parametrize(
    ("provider", "logs", "want"),
    [
        (
            "azure",
            "INFO lightspeed_agentic.config: Azure credential resolution: "
            "Entra ID (service principal)\n",
            "entra_service_principal",
        ),
        (
            "azure",
            "INFO lightspeed_agentic.config: Azure credential resolution: API key (from file)\n",
            "api_key",
        ),
        (
            "azure",
            "INFO lightspeed_agentic.config: Azure credential resolution: API key (from env)\n",
            "api_key",
        ),
        (
            "azure",
            "INFO lightspeed_agentic.batch: provider=deepagents model=m\n"
            "INFO lightspeed_agentic.config: Azure credential resolution: API key (from file)\n",
            None,
        ),
        ("azure", "", None),
    ],
)
def test_report_records_azure_startup_auth_not_later_tool_text(
    tmp_path: Path, provider: str, logs: str, want: str | None
) -> None:
    from dataclasses import replace

    from tests.e2e.inspection_evidence import write_inspection_evidence

    context = _context(RunBatchResult(run_uid="run", pod_logs=logs))
    write_inspection_evidence(
        tmp_path, replace(_config(), lightspeed_provider=provider), context, "scenario", "passed"
    )
    report = json.loads(next(tmp_path.glob("inspection-*.json")).read_text())
    assert report["auth_mode"] == want
    if want is None:
        assert "auth_mode_unverified" in report["evidence_gaps"]


@pytest.mark.parametrize("outcome", ["passed", "failed", "skipped"])
def test_partial_checks_never_count_as_verified(tmp_path: Path, outcome: str) -> None:
    from tests.e2e.inspection_evidence import write_inspection_evidence

    context = _context(RunBatchResult(run_uid="run"))
    context.update(
        {
            "job_env_overrides": {"LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED": "true"},
            "inspection_expected_outcome": "benign",
            "inspection_checks": {"inspection_span": True},
        }
    )
    write_inspection_evidence(tmp_path, _config(), context, "scenario", outcome)
    report = json.loads(next(tmp_path.glob("inspection-*.json")).read_text())
    assert report["test_outcome"] == outcome
    assert report["verified"] is False
    assert "schema_valid" in report["missing_checks"]
    assert "mcp_sentinel" in report["missing_checks"]


@pytest.mark.parametrize(
    ("fault", "passed", "failed", "skipped", "errors", "expected_outcomes"),
    [
        ("none", 2, 0, 0, 0, ("passed", "passed")),
        ("invalid-schema", 1, 1, 0, 0, ("failed", "passed")),
        ("zero-exit", 1, 1, 0, 0, ("failed", "passed")),
        ("unexpected-result", 1, 1, 0, 0, ("failed", "passed")),
        ("api-forbidden", 1, 1, 0, 0, ("failed", "passed")),
        ("forged-span", 1, 1, 0, 0, ("failed", "passed")),
        ("wrong-tool", 1, 1, 0, 0, ("failed", "passed")),
        ("wrong-call-id", 1, 1, 0, 0, ("failed", "passed")),
        ("missing-tool", 1, 1, 0, 0, ("failed", "passed")),
        ("benign-then-reject", 1, 1, 0, 0, ("failed", "passed")),
        ("rejection-exported-first", 1, 1, 0, 0, ("failed", "passed")),
        ("missing-fixtures", 0, 0, 2, 0, ("skipped", "skipped")),
        ("teardown-error", 2, 0, 0, 2, ("failed", "failed")),
        ("setup-skip", 0, 0, 2, 0, ("skipped", "skipped")),
        ("teardown-skip", 2, 0, 2, 0, ("skipped", "skipped")),
        ("call-failure-teardown-skip", 1, 1, 2, 0, ("failed", "skipped")),
    ],
)
def test_actual_bdd_scenarios_retain_reports_on_pass_failure_and_skip(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    passed: int,
    failed: int,
    skipped: int,
    errors: int,
    expected_outcomes: tuple[str, str],
) -> None:
    """Exercise checked-in Gherkin/steps/teardown; script only external run/API results."""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(root), str(root / "tests/e2e"))))
    artifacts = pytester.path / "reports"
    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts))
    pytester.makeconftest(
        f"fault = {fault!r}\n"
        + r"""
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib3.response import HTTPResponse
from kubernetes.client import ApiException
from tests.e2e.conftest import *
from tests.e2e.batch_runner import RunBatchResult
from tests.e2e.run_result import batch_to_run_result
from tests.e2e.suite_setup import BatchE2EConfig
from steps import given as given_steps

if fault in {"invalid-schema", "call-failure-teardown-skip"}:
    given_steps.MCP_TOOL_OUTPUT_SCHEMA = {
        **given_steps.MCP_TOOL_OUTPUT_SCHEMA,
        "required": ["success", "summary", "extraProof"],
    }

@pytest.fixture(autouse=True)
def phase_control(inspection_evidence, bdd_context):
    if fault == "setup-skip":
        bdd_context["job_env_overrides"] = {
            "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED": "true"
        }
        pytest.skip("scripted setup skip after inspection is explicitly enabled")
    yield
    if fault in {"teardown-skip", "call-failure-teardown-skip"}:
        pytest.skip("scripted teardown skip")

@pytest.fixture
def batch_e2e_config():
    return BatchE2EConfig(
        namespace="offline", sandbox_image="sandbox:requested", service_account="sa",
        llm_secret="creds", lightspeed_provider="openai", model="offline-scripted-model",
        provider_name="openai-agents", session_id="session", otel_endpoint="collector:4317",
        otel_ca_secret="", verify_full_fixtures=fault != "missing-fixtures",
        job_env={"LIGHTSPEED_MCP_SERVERS": '[{"name":"mock-ocp-mcp","url":"http://mock/mcp"}]'},
    )

@pytest.fixture
def k8s_apps_client():
    return MagicMock()

@pytest.fixture
def k8s_core_client():
    core = MagicMock()
    core.list_namespaced_pod.return_value = SimpleNamespace(
        items=[SimpleNamespace(metadata=SimpleNamespace(name="collector", uid="collector-uid"))]
    )
    return core

@pytest.fixture
def k8s_client():
    objects = MagicMock()
    if fault != "unexpected-result":
        objects.get_namespaced_custom_object.side_effect = ApiException(
            status=403 if fault == "api-forbidden" else 404
        )
    return objects

@pytest.fixture
def run_runner(bdd_context, k8s_core_client, monkeypatch):
    def run(query, **kwargs):
        assert kwargs["job_env_overrides"]["LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED"] == "true"
        malicious = bdd_context["inspection_expected_outcome"] == "malicious"
        tool = "malicious_probe" if malicious else "list_namespaces"
        if malicious and fault == "wrong-tool":
            tool = "read_file"
        import json
        from tests.e2e.mock_mcp_server import MCP_MALICIOUS_RESULT
        from tests.test_otel_verify import _debug_logs, _trace_spans

        # Preserve main's full trace contract as well as the focused MCP checks.
        spans = json.loads(json.dumps(_trace_spans(rejected=malicious))
            .replace("selected-run", "run").replace("lookup", tool)
            .replace("actual-call", "mcp-call")
            .replace("native answer", "e2e-sentinel-ns-7f3a9"))
        native = next(span for span in spans if span["Name"].startswith("execute_tool "))
        inspection = next(span for span in spans if span["Name"] == "tool_result.inspection")
        generation = next(span for span in spans if span["ID"] == "0000000000000002")
        native["attrs"]["gen_ai.tool.call.arguments"] = "{}"
        native["attrs"]["gen_ai.tool.call.result"] = json.dumps({
            "content": MCP_MALICIOUS_RESULT if malicious else "e2e-sentinel-ns-7f3a9"
        })
        messages = json.loads(generation["attrs"]["gen_ai.output.messages"])
        for message in messages:
            for part in message["parts"]:
                if part["type"] == "tool_call":
                    part["arguments"] = {}
        generation["attrs"]["gen_ai.output.messages"] = json.dumps(messages)
        if malicious and fault == "missing-tool":
            spans.remove(native)
        if malicious and fault == "wrong-call-id":
            inspection["attrs"]["gen_ai.tool.call.id"] = "other-call"
        if malicious and fault == "forged-span":
            inspection["Name"] = "execute_tool execute"
            inspection["attrs"] = {"gen_ai.tool.call.result":
                "tool_result.inspection inspection.outcome: Str(malicious)"}
        if malicious and fault in {"benign-then-reject", "rejection-exported-first"}:
            earlier = []
            for original, span_id, start, end in (
                (generation, "5" * 16, 2, 4),
                (native, "6" * 16, 5, 6),
                (inspection, "7" * 16, 7, 8),
            ):
                accepted = json.loads(json.dumps(original).replace("mcp-call", "earlier-call"))
                accepted.update({
                    "ID": span_id,
                    "Start time": f"2026-10-29 10:00:00.00000000{start} +0000 UTC",
                    "End time": f"2026-10-29 10:00:00.00000000{end} +0000 UTC",
                })
                if accepted["Name"] == "tool_result.inspection":
                    accepted["Status code"] = "Unset"
                    accepted["attrs"]["inspection.outcome"] = "benign"
                    accepted["attrs"]["inspection.category"] = "none"
                earlier.append(accepted)
            spans = spans + earlier if fault == "rejection-exported-first" else earlier + spans
        logs = _debug_logs(spans)
        if malicious and fault in {"benign-then-reject", "rejection-exported-first"}:
            # The negative scenario must fail the controlled-input guard, not an
            # unrelated reconstruction defect in its scripted external evidence.
            from tests.e2e.otel_verify import assert_trace_reconstructable
            assert_trace_reconstructable(
                logs, "run", expect_shaping=True, inspection_outcome="malicious"
            )
        from tests.e2e import otel_verify
        clock = iter((0.0, 0.0, 200.0))
        monkeypatch.setattr(otel_verify, "time", SimpleNamespace(
            monotonic=lambda: next(clock), sleep=lambda _seconds: None,
        ))
        k8s_core_client.read_namespaced_pod_log.side_effect = lambda **_: HTTPResponse(
            body=BytesIO(logs.encode()), status=200, preload_content=False
        )
        body = {"success": True, "summary": "e2e-sentinel-ns-7f3a9"}
        return batch_to_run_result(RunBatchResult(
            job_succeeded=not malicious, body=body, run_uid="run", job_name="job",
            result_name="job-analysis-1", container_exit_code=(
                0 if fault == "zero-exit" or not malicious else 1
            ), container_reason="Error" if malicious else "Completed",
            termination_message="ToolResultSafetyInspectionFailed" if malicious else None,
            image_id="sandbox@sha256:" + "b" * 64,
        ))
    yield run
    if fault == "teardown-error":
        raise RuntimeError("scripted teardown failure")
"""
    )
    pytester.makepyfile(
        "from pytest_bdd import scenarios\n"
        f"scenarios({str(root / 'tests/e2e/features/tool_result_inspection.feature')!r})\n"
    )
    result = pytester.runpytest_subprocess("-q", "--tb=short", timeout=30)
    result.assert_outcomes(passed=passed, failed=failed, skipped=skipped, errors=errors)
    reports = [json.loads(path.read_text()) for path in artifacts.glob("inspection-*.json")]
    assert len(reports) == 2, "Even a failed/skipped scenario must retain its partial evidence"
    assert sum(report["verified"] for report in reports) == expected_outcomes.count("passed")
    assert sorted(report["test_outcome"] for report in reports) == list(expected_outcomes)
    if fault in {"benign-then-reject", "rejection-exported-first"}:
        rejected = next(report for report in reports if report["expected_outcome"] == "malicious")
        assert rejected["checks"]["safe_termination"] is True
        assert rejected["checks"]["no_result_cr"] is True
        assert rejected["missing_checks"] == ["inspection_span"]
        assert rejected["verified"] is False
    if fault == "teardown-skip":
        for report in reports:
            assert report["checks"]["inspection_span"] is True
            assert report["missing_checks"] == []
            assert report["verified"] is False
