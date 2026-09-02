"""Tests for Result CR publishing and sandbox termination log."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from kubernetes import config as k8s_config  # type: ignore[import-untyped]
from kubernetes.client.exceptions import ApiException  # type: ignore[import-untyped]

from lightspeed_agentic.publish_results.publish import (
    K8S_REQUEST_TIMEOUT,
    MAX_TERMINATION_LOG_BYTES,
    PublishError,
    _default_api,
    publish_agent_result,
    publish_result_cr,
    step_from_result_kind,
    step_from_result_template,
    write_termination_log,
)

_TEMPLATE: dict = {
    "apiVersion": "agentic.openshift.io/v1alpha1",
    "kind": "AnalysisResult",
    "metadata": {
        "name": "run-analysis-1",
        "namespace": "openshift-lightspeed",
        "labels": {"agentic.openshift.io/run": "uid"},
    },
    "spec": {"agenticRunName": "run"},
}

_STATUS: dict = {
    "conditions": [{"type": "Completed", "status": "True", "reason": "Succeeded"}],
    "options": [{"title": "fix"}],
}

_AGENT_OUTPUT: dict = {
    "actionRequired": True,
    "options": [{"title": "fix"}],
    "diagnosis": {"summary": "s", "rootCause": "r"},
}


def _api() -> MagicMock:
    return MagicMock()


def _dt() -> datetime:
    return datetime(2026, 8, 15, 12, 30, 0, tzinfo=UTC)


class TestPublishResultCR:
    def test_create_then_update_status(self) -> None:
        api = _api()
        api.create_namespaced_custom_object.return_value = {
            "metadata": {"resourceVersion": "12345"},
        }
        publish_result_cr(_TEMPLATE, _STATUS, api=api)

        api.create_namespaced_custom_object.assert_called_once()
        create_kwargs = api.create_namespaced_custom_object.call_args.kwargs
        assert create_kwargs["group"] == "agentic.openshift.io"
        assert create_kwargs["version"] == "v1alpha1"
        assert create_kwargs["plural"] == "analysisresults"
        assert create_kwargs["namespace"] == "openshift-lightspeed"
        assert create_kwargs["_request_timeout"] == K8S_REQUEST_TIMEOUT
        assert "status" not in create_kwargs["body"]

        api.replace_namespaced_custom_object_status.assert_called_once_with(
            group="agentic.openshift.io",
            version="v1alpha1",
            namespace="openshift-lightspeed",
            plural="analysisresults",
            name="run-analysis-1",
            body={
                "apiVersion": "agentic.openshift.io/v1alpha1",
                "kind": "AnalysisResult",
                "metadata": {
                    "name": "run-analysis-1",
                    "namespace": "openshift-lightspeed",
                    "resourceVersion": "12345",
                },
                "status": _STATUS,
            },
            _request_timeout=K8S_REQUEST_TIMEOUT,
        )

    def test_create_conflict_still_updates_status(self) -> None:
        api = _api()
        api.create_namespaced_custom_object.side_effect = ApiException(
            status=409,
            reason="AlreadyExists",
        )
        api.get_namespaced_custom_object.return_value = {
            "metadata": {"resourceVersion": "67890"},
        }

        publish_result_cr(_TEMPLATE, _STATUS, api=api)

        api.get_namespaced_custom_object.assert_called_once()
        api.replace_namespaced_custom_object_status.assert_called_once_with(
            group="agentic.openshift.io",
            version="v1alpha1",
            namespace="openshift-lightspeed",
            plural="analysisresults",
            name="run-analysis-1",
            body={
                "apiVersion": "agentic.openshift.io/v1alpha1",
                "kind": "AnalysisResult",
                "metadata": {
                    "name": "run-analysis-1",
                    "namespace": "openshift-lightspeed",
                    "resourceVersion": "67890",
                },
                "status": _STATUS,
            },
            _request_timeout=K8S_REQUEST_TIMEOUT,
        )

    def test_create_error_raises_publish_error(self) -> None:
        api = _api()
        api.create_namespaced_custom_object.side_effect = ApiException(
            status=403,
            reason="Forbidden",
        )

        with pytest.raises(PublishError, match=r"create analysisresults.*403 Forbidden"):
            publish_result_cr(_TEMPLATE, _STATUS, api=api)

        api.replace_namespaced_custom_object_status.assert_not_called()

    def test_create_error_without_reason_uses_body(self) -> None:
        api = _api()
        exc = ApiException(status=403, reason=None)
        exc.body = "forbidden"
        api.create_namespaced_custom_object.side_effect = exc

        with pytest.raises(PublishError, match=r"create analysisresults.*403 forbidden"):
            publish_result_cr(_TEMPLATE, _STATUS, api=api)

    def test_update_error_raises_publish_error(self) -> None:
        api = _api()
        api.replace_namespaced_custom_object_status.side_effect = ApiException(
            status=500,
            reason="Error",
        )

        with pytest.raises(PublishError, match="update status"):
            publish_result_cr(_TEMPLATE, _STATUS, api=api)

    def test_create_timeout_raises_publish_error(self) -> None:
        from urllib3.exceptions import ReadTimeoutError

        api = _api()
        api.create_namespaced_custom_object.side_effect = ReadTimeoutError(
            None,
            "/apis/",
            "Read timed out",
        )

        with pytest.raises(PublishError, match=r"create analysisresults.*request timed out"):
            publish_result_cr(_TEMPLATE, _STATUS, api=api)

        api.replace_namespaced_custom_object_status.assert_not_called()

    def test_invalid_template_raises_value_error(self) -> None:
        bad_template = {"apiVersion": "agentic.openshift.io/v1alpha1", "kind": "Foo"}
        with pytest.raises(ValueError, match="unsupported kind"):
            publish_result_cr(bad_template, _STATUS, api=_api())


class TestPublishAgentResult:
    def test_builds_status_and_publishes(self) -> None:
        api = MagicMock()
        publish_agent_result(
            _TEMPLATE,
            _AGENT_OUTPUT,
            started_at=_dt(),
            completed_at=_dt(),
            api=api,
        )

        api.create_namespaced_custom_object.assert_called_once()
        update_kwargs = api.replace_namespaced_custom_object_status.call_args.kwargs
        status = update_kwargs["body"]["status"]
        assert status["options"] == _AGENT_OUTPUT["options"]
        assert status["actionRequired"] == "True"
        assert status["conditions"][1]["reason"] == "Succeeded"

    def test_token_usage_propagated_to_status(self) -> None:
        api = MagicMock()
        publish_agent_result(
            _TEMPLATE,
            _AGENT_OUTPUT,
            started_at=_dt(),
            completed_at=_dt(),
            input_tokens=1000,
            output_tokens=400,
            api=api,
        )
        status = api.replace_namespaced_custom_object_status.call_args.kwargs["body"]["status"]
        assert status["tokenUsage"] == {"inputTokens": 1000, "outputTokens": 400}

    def test_missing_kind_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="missing kind"):
            publish_agent_result(
                {"apiVersion": "agentic.openshift.io/v1alpha1"},
                _AGENT_OUTPUT,
                started_at=_dt(),
                completed_at=_dt(),
                api=MagicMock(),
            )


class TestWriteTerminationLog:
    def test_writes_message(self, tmp_path) -> None:
        path = tmp_path / "termination-log"
        write_termination_log("sandbox input read failed", path=str(path))
        assert path.read_text(encoding="utf-8") == "sandbox input read failed"

    def test_truncates_to_max_bytes(self, tmp_path) -> None:
        path = tmp_path / "termination-log"
        write_termination_log("x" * 5000, path=str(path))
        assert len(path.read_bytes()) == MAX_TERMINATION_LOG_BYTES

    def test_truncates_on_utf8_boundary(self, tmp_path) -> None:
        path = tmp_path / "termination-log"
        # U+1F600 (😀) is 4 bytes; place one at the truncation edge.
        prefix = "a" * (MAX_TERMINATION_LOG_BYTES - 3)
        write_termination_log(prefix + "😀", path=str(path))
        data = path.read_bytes()
        assert len(data) <= MAX_TERMINATION_LOG_BYTES
        assert data.decode("utf-8") == prefix

    def test_write_failure_does_not_raise(self, tmp_path) -> None:
        path = tmp_path / "dir"
        path.mkdir()
        write_termination_log("readiness failed", path=str(path))


class TestDefaultApi:
    def test_in_cluster_failure_raises_without_kubeconfig_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        with (
            patch.object(
                k8s_config,
                "load_incluster_config",
                side_effect=k8s_config.ConfigException("no token"),
            ),
            patch.object(k8s_config, "load_kube_config") as load_kube,
            pytest.raises(PublishError, match="in-cluster Kubernetes config unavailable"),
        ):
            _default_api()
        load_kube.assert_not_called()

    def test_outside_cluster_falls_back_to_kubeconfig(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
        with (
            patch.object(
                k8s_config,
                "load_incluster_config",
                side_effect=k8s_config.ConfigException("not in cluster"),
            ),
            patch.object(k8s_config, "load_kube_config") as load_kube,
            patch(
                "lightspeed_agentic.publish_results.publish.CustomObjectsApi",
                return_value=MagicMock(),
            ),
        ):
            _default_api()
        load_kube.assert_called_once()

    def test_outside_cluster_kubeconfig_failure_raises_publish_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
        with (
            patch.object(
                k8s_config,
                "load_incluster_config",
                side_effect=k8s_config.ConfigException("not in cluster"),
            ),
            patch.object(
                k8s_config,
                "load_kube_config",
                side_effect=k8s_config.ConfigException("no kubeconfig"),
            ),
            pytest.raises(PublishError, match="local kubeconfig unavailable"),
        ):
            _default_api()


class TestStepFromResultKind:
    def test_maps_all_result_kinds(self) -> None:
        assert step_from_result_kind("AnalysisResult") == "analysis"
        assert step_from_result_kind("ExecutionResult") == "execution"
        assert step_from_result_kind("VerificationResult") == "verification"
        assert step_from_result_kind("EscalationResult") == "escalation"

    def test_template_uses_kind(self) -> None:
        step = step_from_result_template({"kind": "EscalationResult"})
        assert step == "escalation"

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported Result kind"):
            step_from_result_kind("UnknownResult")
