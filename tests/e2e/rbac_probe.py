"""RBAC probe — validates SA authorization with real K8s API calls (OLS-5022).

Creates an API client authenticated as the fixture ServiceAccount (via
TokenRequest) and exercises one allowed operation (create + read
AnalysisResult) and one denied operation (create Secret).  Diagnostics
never print tokens or Secret contents.
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any

from kubernetes.client import (  # type: ignore[import-untyped]
    ApiClient,
    ApiException,
    AuthenticationV1Api,
    Configuration,
    CoreV1Api,
    CustomObjectsApi,
    V1ObjectMeta,
    V1Secret,
)

from tests.e2e.k8s_constants import CRD_GROUP, CRD_VERSION

logger = logging.getLogger(__name__)

RBAC_PROBE_LABEL = "agentic.openshift.io/rbac-probe"
_RBAC_PROPAGATION_TIMEOUT_S = 30
_RBAC_PROPAGATION_INTERVAL_S = 2


def create_sa_api_client(
    admin_client: ApiClient,
    namespace: str,
    sa_name: str,
) -> ApiClient:
    """Return an API client authenticated as *sa_name* via TokenRequest.

    The token is short-lived (600 s) and never logged.
    """
    auth_api = AuthenticationV1Api(admin_client)
    token_resp = auth_api.create_namespaced_service_account_token(
        name=sa_name,
        namespace=namespace,
        body={
            "apiVersion": "authentication.k8s.io/v1",
            "kind": "TokenRequest",
            "spec": {"expirationSeconds": 600},
        },
    )
    token: str = token_resp.status.token

    src = admin_client.configuration
    cfg = Configuration()
    cfg.host = src.host
    cfg.ssl_ca_cert = src.ssl_ca_cert
    cfg.verify_ssl = src.verify_ssl
    cfg.api_key = {"BearerToken": token}
    cfg.api_key_prefix = {"BearerToken": "Bearer"}
    return ApiClient(cfg)


def probe_create_result_cr(
    sa_custom_api: CustomObjectsApi,
    namespace: str,
    result_name: str,
) -> dict[str, Any]:
    """Create an AnalysisResult with SA credentials (bounded RBAC wait)."""
    body: dict[str, Any] = {
        "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
        "kind": "AnalysisResult",
        "metadata": {
            "name": result_name,
            "namespace": namespace,
            "labels": {RBAC_PROBE_LABEL: "true"},
        },
        "spec": {"agenticRunName": f"rbac-probe-{result_name}"},
    }
    deadline = time.monotonic() + _RBAC_PROPAGATION_TIMEOUT_S
    last_exc: ApiException | None = None
    while time.monotonic() < deadline:
        try:
            return sa_custom_api.create_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=namespace,
                plural="analysisresults",
                body=body,
            )
        except ApiException as exc:
            if exc.status == 403:
                last_exc = exc
                logger.info(
                    "RBAC not yet propagated for %s/%s, retrying in %ds…",
                    namespace,
                    result_name,
                    _RBAC_PROPAGATION_INTERVAL_S,
                )
                time.sleep(_RBAC_PROPAGATION_INTERVAL_S)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    msg = "RBAC propagation timed out without an exception"
    raise RuntimeError(msg)


def probe_read_result_cr(
    sa_custom_api: CustomObjectsApi,
    namespace: str,
    result_name: str,
) -> dict[str, Any]:
    """Read an AnalysisResult with SA credentials."""
    return sa_custom_api.get_namespaced_custom_object(
        group=CRD_GROUP,
        version=CRD_VERSION,
        namespace=namespace,
        plural="analysisresults",
        name=result_name,
    )


def probe_create_secret(
    sa_core_api: CoreV1Api,
    namespace: str,
    secret_name: str,
) -> int:
    """Try to create a Secret with SA credentials.  Returns the HTTP status.

    Expected: 403 Forbidden.  Diagnostics never log Secret contents.
    """
    try:
        sa_core_api.create_namespaced_secret(
            namespace=namespace,
            body=V1Secret(metadata=V1ObjectMeta(name=secret_name)),
        )
    except ApiException as exc:
        if exc.status is not None:
            return exc.status
        raise
    return 200


def cleanup_probe_result_cr(
    admin_custom_api: CustomObjectsApi,
    namespace: str,
    result_name: str,
) -> None:
    """Delete probe AnalysisResult via the admin client.  Ignores 404."""
    try:
        admin_custom_api.delete_namespaced_custom_object(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=namespace,
            plural="analysisresults",
            name=result_name,
        )
    except ApiException as exc:
        if exc.status != 404:
            raise


def cleanup_probe_secret(
    admin_core_api: CoreV1Api,
    namespace: str,
    secret_name: str,
) -> None:
    """Delete probe Secret via the admin client (unexpected-success path).  Ignores 404."""
    try:
        admin_core_api.delete_namespaced_secret(secret_name, namespace)
    except ApiException as exc:
        if exc.status != 404:
            raise


def generate_probe_name() -> str:
    """DNS-safe unique name for probe resources."""
    return f"rbac-probe-{secrets.token_hex(4)}"
