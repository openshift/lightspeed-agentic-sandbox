"""Unit tests for RBAC probe helper functions (OLS-5022)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client import (
    ApiClient,
    ApiException,
    AuthenticationV1TokenRequest,
    Configuration,
    CoreV1Api,
    CustomObjectsApi,
    V1TokenRequestStatus,
)

from tests.e2e.k8s_constants import CRD_GROUP, CRD_VERSION
from tests.e2e.rbac_probe import (
    cleanup_probe_result_cr,
    cleanup_probe_secret,
    create_sa_api_client,
    generate_probe_name,
    probe_create_result_cr,
    probe_create_secret,
    probe_read_result_cr,
)


class TestCreateSaApiClient:
    @patch("tests.e2e.rbac_probe.CoreV1Api")
    def test_creates_client_via_core_v1_token_request(
        self,
        mock_core_v1_cls: MagicMock,
    ) -> None:
        mock_core_api = MagicMock()
        mock_core_v1_cls.return_value = mock_core_api

        mock_token_resp = AuthenticationV1TokenRequest(
            status=V1TokenRequestStatus(
                token="test-sa-token-xyz",  # noqa: S106
                expiration_timestamp="2026-01-01T00:00:00Z",
            ),
        )
        mock_core_api.create_namespaced_service_account_token.return_value = mock_token_resp

        admin_cfg = Configuration()
        admin_cfg.host = "https://k8s-api.example.com:6443"
        admin_cfg.verify_ssl = False
        admin_client = ApiClient(configuration=admin_cfg)

        sa_client = create_sa_api_client(
            admin_client,
            namespace="openshift-lightspeed",
            sa_name="sandbox-e2e-sa",
        )

        mock_core_v1_cls.assert_called_once_with(admin_client)
        mock_core_api.create_namespaced_service_account_token.assert_called_once()
        call_kwargs = mock_core_api.create_namespaced_service_account_token.call_args.kwargs
        assert call_kwargs["name"] == "sandbox-e2e-sa"
        assert call_kwargs["namespace"] == "openshift-lightspeed"
        token_req = call_kwargs["body"]
        assert isinstance(token_req, AuthenticationV1TokenRequest)
        assert token_req.spec.expiration_seconds == 600

        assert sa_client.configuration.host == "https://k8s-api.example.com:6443"
        assert sa_client.configuration.api_key == {"BearerToken": "test-sa-token-xyz"}
        assert sa_client.configuration.api_key_prefix == {"BearerToken": "Bearer"}


class TestProbeCreateResultCr:
    def test_success_without_retries(self) -> None:
        mock_custom_api = MagicMock(spec=CustomObjectsApi)
        mock_custom_api.create_namespaced_custom_object.return_value = {"status": "created"}

        result = probe_create_result_cr(
            mock_custom_api,
            namespace="test-ns",
            result_name="probe-123",
        )

        assert result == {"status": "created"}
        mock_custom_api.create_namespaced_custom_object.assert_called_once()
        kwargs = mock_custom_api.create_namespaced_custom_object.call_args.kwargs
        assert kwargs["group"] == CRD_GROUP
        assert kwargs["version"] == CRD_VERSION
        assert kwargs["namespace"] == "test-ns"
        assert kwargs["plural"] == "analysisresults"
        assert kwargs["body"]["metadata"]["name"] == "probe-123"

    @patch("tests.e2e.rbac_probe.time.sleep")
    def test_retries_on_403_then_succeeds(self, mock_sleep: MagicMock) -> None:
        mock_custom_api = MagicMock(spec=CustomObjectsApi)
        mock_custom_api.create_namespaced_custom_object.side_effect = [
            ApiException(status=403, reason="Forbidden"),
            {"status": "created"},
        ]

        result = probe_create_result_cr(
            mock_custom_api,
            namespace="test-ns",
            result_name="probe-123",
        )

        assert result == {"status": "created"}
        assert mock_custom_api.create_namespaced_custom_object.call_count == 2
        mock_sleep.assert_called_once()

    def test_non_403_error_raises_immediately(self) -> None:
        mock_custom_api = MagicMock(spec=CustomObjectsApi)
        mock_custom_api.create_namespaced_custom_object.side_effect = ApiException(status=500)

        with pytest.raises(ApiException) as exc_info:
            probe_create_result_cr(
                mock_custom_api,
                namespace="test-ns",
                result_name="probe-123",
            )
        assert exc_info.value.status == 500


class TestProbeReadResultCr:
    def test_reads_custom_object(self) -> None:
        mock_custom_api = MagicMock(spec=CustomObjectsApi)
        mock_custom_api.get_namespaced_custom_object.return_value = {"kind": "AnalysisResult"}

        res = probe_read_result_cr(mock_custom_api, "test-ns", "probe-123")
        assert res == {"kind": "AnalysisResult"}
        mock_custom_api.get_namespaced_custom_object.assert_called_once_with(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace="test-ns",
            plural="analysisresults",
            name="probe-123",
        )


class TestProbeCreateSecret:
    def test_returns_403_when_forbidden(self) -> None:
        mock_core_api = MagicMock(spec=CoreV1Api)
        mock_core_api.create_namespaced_secret.side_effect = ApiException(status=403)

        status = probe_create_secret(mock_core_api, "test-ns", "test-sec")
        assert status == 403

    def test_returns_200_when_created(self) -> None:
        mock_core_api = MagicMock(spec=CoreV1Api)
        mock_core_api.create_namespaced_secret.return_value = MagicMock()

        status = probe_create_secret(mock_core_api, "test-ns", "test-sec")
        assert status == 200

    def test_raises_non_status_exception(self) -> None:
        mock_core_api = MagicMock(spec=CoreV1Api)
        mock_core_api.create_namespaced_secret.side_effect = RuntimeError("network down")

        with pytest.raises(RuntimeError):
            probe_create_secret(mock_core_api, "test-ns", "test-sec")


class TestCleanupProbeResources:
    def test_cleanup_result_cr_success(self) -> None:
        mock_custom_api = MagicMock(spec=CustomObjectsApi)
        cleanup_probe_result_cr(mock_custom_api, "test-ns", "probe-123")
        mock_custom_api.delete_namespaced_custom_object.assert_called_once()

    def test_cleanup_result_cr_swallows_404(self) -> None:
        mock_custom_api = MagicMock(spec=CustomObjectsApi)
        mock_custom_api.delete_namespaced_custom_object.side_effect = ApiException(status=404)
        cleanup_probe_result_cr(mock_custom_api, "test-ns", "probe-123")

    def test_cleanup_result_cr_propagates_non_404(self) -> None:
        mock_custom_api = MagicMock(spec=CustomObjectsApi)
        mock_custom_api.delete_namespaced_custom_object.side_effect = ApiException(status=500)
        with pytest.raises(ApiException):
            cleanup_probe_result_cr(mock_custom_api, "test-ns", "probe-123")

    def test_cleanup_secret_swallows_404(self) -> None:
        mock_core_api = MagicMock(spec=CoreV1Api)
        mock_core_api.delete_namespaced_secret.side_effect = ApiException(status=404)
        cleanup_probe_secret(mock_core_api, "test-ns", "probe-123")

    def test_cleanup_secret_propagates_non_404(self) -> None:
        mock_core_api = MagicMock(spec=CoreV1Api)
        mock_core_api.delete_namespaced_secret.side_effect = ApiException(status=500)
        with pytest.raises(ApiException):
            cleanup_probe_secret(mock_core_api, "test-ns", "probe-123")


class TestGenerateProbeName:
    def test_name_format(self) -> None:
        name = generate_probe_name()
        assert name.startswith("rbac-probe-")
        assert len(name) > len("rbac-probe-")
