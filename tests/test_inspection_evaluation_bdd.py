"""Exercise real corpus BDD discovery/steps/launcher; script only Kubernetes APIs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "fault", ["none", "partial", "job-error", "previous-test-failure", "interrupt", "timeout"]
)
def test_normal_bdd_runs_classifier_corpus_and_retains_measurements(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(ROOT), str(ROOT / "tests/e2e"))))
    artifacts = pytester.path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts))
    monkeypatch.setenv("OPENAI_API_KEY", "offline-bdd-key")
    monkeypatch.setenv("E2E_PROVIDER", "openai-agents")
    pytester.makeini("[pytest]\ntimeout = 600\nmarkers =\n    e2e: live cluster BDD\n")
    pytester.makeconftest(
        f"fault = {fault!r}\n"
        + r"""
import json
from pathlib import Path
from types import SimpleNamespace
from kubernetes.client import ApiException
from tests.e2e.conftest import *
from tests.test_inspection_e2e import _config
from tests.test_inspection_evaluation import _evaluation_cluster

core, jobs, state = _evaluation_cluster(
    complete=fault not in {"partial", "interrupt", "timeout"}, exit_code=0
)
jobs.read_namespaced_job.return_value.status = SimpleNamespace(succeeded=1, failed=None)
if fault == "job-error":
    jobs.create_namespaced_job.side_effect = ApiException(status=403, reason="PRIVATE-API-ERROR")
if fault == "interrupt":
    import signal
    def interrupt(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGINT)
    jobs.read_namespaced_job.side_effect = interrupt
if fault == "timeout":
    import time
    def block(*args, **kwargs):
        state["monitor_reached"] = True
        time.sleep(10)
    jobs.read_namespaced_job.side_effect = block

def pytest_collection_modifyitems(items):
    if fault == "timeout":
        for item in items:
            item.add_marker(pytest.mark.timeout(1, method="signal"), append=False)

@pytest.fixture(scope="session", autouse=True)
def _batch_suite_lifecycle():
    yield
    if jobs.create_namespaced_job.call_args is None:
        return
    created = jobs.create_namespaced_job.call_args.args[1]
    Path(os.environ["ARTIFACT_DIR"], "submissions.json").write_text(json.dumps({
        "jobs": jobs.create_namespaced_job.call_count,
        "exists": state["exists"],
        "delete_requests": jobs.delete_namespaced_job.call_count,
        "monitor_reached": state.get("monitor_reached", False),
        "test_timeout": state["test_timeout"],
        "job_deadline": created.spec["activeDeadlineSeconds"],
        "args": created.spec["template"]["spec"]["containers"][0]["args"],
    }))

@pytest.fixture(autouse=True)
def record_effective_timeout(request):
    import pytest_timeout
    state["test_timeout"] = pytest_timeout._get_item_settings(request.node).timeout

@pytest.fixture(scope="session")
def batch_e2e_config():
    return _config()

@pytest.fixture(scope="session")
def k8s_core_client():
    return core

@pytest.fixture(scope="session")
def k8s_batch_client():
    return jobs
"""
    )
    # Use the checked-in discovery module, not a duplicate test definition.
    pytester.makepyfile("from tests.e2e.test_inspection_evaluation_bdd import *")
    if fault == "previous-test-failure":
        pytester.makepyfile(
            test_aaa_failure='def test_failure():\n    assert False, "ordinary earlier failure"\n'
        )
    result = pytester.runpytest_subprocess("-q", "--tb=short", timeout=30)
    if fault == "interrupt":
        assert result.ret == pytest.ExitCode.INTERRUPTED
    else:
        result.assert_outcomes(
            passed=int(fault in {"none", "previous-test-failure"}),
            failed=int(fault != "none"),
        )
    if fault == "timeout":
        assert "Timeout (>1.0s) from pytest-timeout." in result.stdout.str()
    reports = list(artifacts.glob("inspection-evaluation-*.json"))
    assert len(reports) == 1
    raw = reports[0].read_text()
    assert "PRIVATE-API-ERROR" not in raw
    report = json.loads(raw)
    assert report["metrics"]["total"] == 18
    assert report["operational_success"] is (fault in {"none", "previous-test-failure"})
    if fault in {"none", "previous-test-failure"}:
        # Scripted benign decisions include twelve false negatives: measurements,
        # not an invented accuracy failure in the BDD scenario.
        assert report["metrics"]["evaluated"] == 18
        assert report["metrics"]["false_negatives"] == 12
        assert report["metrics"]["false_negative_rate"] == 1
        assert report["metrics"]["category_mismatches"] == 0
    else:
        assert report["metrics"]["unverified"] > 0
    assert report["accuracy_threshold"] is None
    submissions = json.loads((artifacts / "submissions.json").read_text())
    assert submissions["jobs"] == 1
    assert math.isfinite(submissions["test_timeout"])
    if fault == "timeout":
        assert submissions["test_timeout"] == 1
        assert submissions["monitor_reached"] is True
    else:
        assert submissions["test_timeout"] > submissions["job_deadline"] + 30
    if fault in {"interrupt", "timeout"}:
        assert submissions["delete_requests"] == 1
        assert submissions["exists"] is False
        assert report["metrics"]["evaluated"] == 1
        assert report["metrics"]["unverified"] == 17
        assert "cancelled" in report["failures"]
    assert submissions["args"] == [
        "python3.12",
        "/input/runner.py",
        "--dataset",
        "/input/corpus_full.yaml",
        "--source",
        "/input/corpus-source.json",
        "--case-timeout",
        "60",
    ]
