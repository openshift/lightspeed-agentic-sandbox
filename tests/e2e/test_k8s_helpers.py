"""Tests for k8s_helpers API wrappers (require kubernetes package)."""

from __future__ import annotations

from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

from urllib3.response import HTTPResponse

from tests.e2e.k8s_helpers import (
    CRD_GROUP,
    CRD_VERSION,
    create_agenticrun,
    fetch_pod_logs,
    poll_agenticrun_phase,
    read_result_cr,
)

import pytest


class TestCreateAgenticrun:
    """Tests for create_agenticrun helper."""

    def test_required_fields(self) -> None:
        api = MagicMock()
        api.create_namespaced_custom_object.return_value = {"metadata": {"name": "run1"}}

        result = create_agenticrun(
            api,
            namespace="ns",
            name="run1",
            request="diagnose pod crash",
        )

        api.create_namespaced_custom_object.assert_called_once()
        call_kwargs = api.create_namespaced_custom_object.call_args
        body: dict[str, Any] = call_kwargs.kwargs["body"]
        assert body["apiVersion"] == f"{CRD_GROUP}/{CRD_VERSION}"
        assert body["kind"] == "AgenticRun"
        assert body["metadata"]["name"] == "run1"
        assert body["metadata"]["namespace"] == "ns"
        assert body["spec"]["request"] == "diagnose pod crash"
        assert body["spec"]["analysis"]["agent"] == "default"
        assert "execution" not in body["spec"]
        assert "verification" not in body["spec"]
        assert "targetNamespaces" not in body["spec"]
        assert result == {"metadata": {"name": "run1"}}

    def test_optional_fields(self) -> None:
        api = MagicMock()
        api.create_namespaced_custom_object.return_value = {"metadata": {"name": "run2"}}

        create_agenticrun(
            api,
            namespace="ns",
            name="run2",
            request="fix oom",
            target_namespaces=["prod", "staging"],
            analysis_agent="custom-analyzer",
            execution=True,
            verification_agent="verify-bot",
        )

        call_kwargs = api.create_namespaced_custom_object.call_args
        body: dict[str, Any] = call_kwargs.kwargs["body"]
        assert body["spec"]["targetNamespaces"] == ["prod", "staging"]
        assert body["spec"]["analysis"]["agent"] == "custom-analyzer"
        assert body["spec"]["execution"] == {}
        assert body["spec"]["verification"]["agent"] == "verify-bot"


class TestPollAgenticrunPhase:
    """Tests for poll_agenticrun_phase polling behaviour."""

    def test_returns_when_target_phase_reached(self) -> None:
        api = MagicMock()
        run: dict[str, Any] = {
            "status": {
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
        }
        api.get_namespaced_custom_object.return_value = run

        result = poll_agenticrun_phase(
            api,
            namespace="ns",
            name="run1",
            target_phases={"Proposed"},
            timeout_seconds=2,
            poll_interval=0.1,
        )

        assert result is not None
        assert result == run

    def test_returns_none_on_timeout(self) -> None:
        api = MagicMock()
        run: dict[str, Any] = {
            "status": {
                "conditions": [{"type": "Analyzed", "status": "Unknown"}],
            },
        }
        api.get_namespaced_custom_object.return_value = run

        result = poll_agenticrun_phase(
            api,
            namespace="ns",
            name="run1",
            target_phases={"Completed"},
            timeout_seconds=0.3,
            poll_interval=0.1,
        )

        assert result is None

    def test_returns_on_terminal_failure(self) -> None:
        api = MagicMock()
        run: dict[str, Any] = {
            "status": {
                "conditions": [{"type": "Analyzed", "status": "False"}],
            },
        }
        api.get_namespaced_custom_object.return_value = run

        result = poll_agenticrun_phase(
            api,
            namespace="ns",
            name="run1",
            target_phases={"Completed"},
            timeout_seconds=2,
            poll_interval=0.1,
        )

        assert result is not None
        assert result == run


class TestReadResultCr:
    """Tests for read_result_cr kind-to-plural mapping."""

    @pytest.mark.parametrize(
        ("kind", "expected_plural"),
        [
            ("AnalysisResult", "analysisresults"),
            ("ExecutionResult", "executionresults"),
            ("VerificationResult", "verificationresults"),
        ],
    )
    def test_reads_correct_plural(self, kind: str, expected_plural: str) -> None:
        api = MagicMock()
        api.get_namespaced_custom_object.return_value = {"kind": kind}

        read_result_cr(api, namespace="ns", name="result1", kind=kind)

        api.get_namespaced_custom_object.assert_called_once_with(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace="ns",
            plural=expected_plural,
            name="result1",
        )

    def test_raises_on_unknown_kind(self) -> None:
        api = MagicMock()
        with pytest.raises(ValueError, match="Unknown kind"):
            read_result_cr(api, namespace="ns", name="x", kind="BogusKind")


class TestFetchPodLogs:
    """Tests for fetch_pod_logs helper."""

    def test_fetches_logs_by_label_selector(self) -> None:
        core_api = MagicMock()

        pod1 = MagicMock()
        pod1.metadata.name = "pod-a"
        pod2 = MagicMock()
        pod2.metadata.name = "pod-b"

        pod_list = MagicMock()
        pod_list.items = [pod1, pod2]
        core_api.list_namespaced_pod.return_value = pod_list

        core_api.read_namespaced_pod_log.side_effect = [
            HTTPResponse(body=BytesIO(b"log-a"), preload_content=False),
            HTTPResponse(body=BytesIO(b"log-b"), preload_content=False),
        ]

        result = fetch_pod_logs(
            core_api,
            namespace="ns",
            label_selector="app=test",
            tail_lines=50,
        )

        assert result == {"pod-a": "log-a", "pod-b": "log-b"}
        core_api.list_namespaced_pod.assert_called_once_with(
            namespace="ns", label_selector="app=test"
        )

    def test_returns_empty_dict_when_no_pods(self) -> None:
        core_api = MagicMock()
        pod_list = MagicMock()
        pod_list.items = []
        core_api.list_namespaced_pod.return_value = pod_list

        result = fetch_pod_logs(
            core_api,
            namespace="ns",
            label_selector="app=missing",
        )

        assert result == {}


class TestConstants:
    """Verify exported constants."""

    def test_crd_group(self) -> None:
        assert CRD_GROUP == "agentic.openshift.io"

    def test_crd_version(self) -> None:
        assert CRD_VERSION == "v1alpha1"
