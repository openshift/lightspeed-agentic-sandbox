"""Publish Result CRs to the cluster and report sandbox infrastructure failures."""

from __future__ import annotations

import copy
import logging
import os
from datetime import datetime
from typing import Any

from kubernetes import config  # type: ignore[import-untyped]
from kubernetes.client import ApiClient, CustomObjectsApi  # type: ignore[import-untyped]
from kubernetes.client.exceptions import ApiException  # type: ignore[import-untyped]
from urllib3.exceptions import MaxRetryError
from urllib3.exceptions import TimeoutError as Urllib3TimeoutError

from lightspeed_agentic.publish_results.status import build_status

logger = logging.getLogger(__name__)

CRD_GROUP = "agentic.openshift.io"
CRD_VERSION = "v1alpha1"

TERMINATION_LOG_PATH = "/dev/termination-log"
MAX_TERMINATION_LOG_BYTES = 4096

K8S_CONNECT_TIMEOUT_SEC = 5.0
K8S_READ_TIMEOUT_SEC = 60.0
K8S_REQUEST_TIMEOUT = (K8S_CONNECT_TIMEOUT_SEC, K8S_READ_TIMEOUT_SEC)

_KIND_TO_PLURAL: dict[str, str] = {
    "AnalysisResult": "analysisresults",
    "ExecutionResult": "executionresults",
    "VerificationResult": "verificationresults",
    "EscalationResult": "escalationresults",
}

_KIND_TO_STEP: dict[str, str] = {
    "AnalysisResult": "analysis",
    "ExecutionResult": "execution",
    "VerificationResult": "verification",
    "EscalationResult": "escalation",
}


def step_from_result_kind(kind: str) -> str:
    """Map Result CR kind to workflow step (analysis, execution, …)."""
    step = _KIND_TO_STEP.get(kind)
    if step is None:
        msg = f"unsupported Result kind: {kind!r}"
        raise ValueError(msg)
    return step


def step_from_result_template(template: dict[str, Any]) -> str:
    """Resolve workflow step from ``result-template.kind``."""
    kind = template.get("kind")
    if not isinstance(kind, str):
        msg = "result-template missing kind"
        raise ValueError(msg)
    return step_from_result_kind(kind)


class PublishError(Exception):
    """Result CR could not be created or status could not be patched."""


def write_termination_log(message: str, *, path: str = TERMINATION_LOG_PATH) -> None:
    """Write a human-readable error to the container termination log (spec B6).

    Content is truncated to 4096 bytes so Kubernetes can surface it on
    pod.status.containerStatuses[].state.terminated.message.
    """
    data = _truncate_utf8_message(message, MAX_TERMINATION_LOG_BYTES)
    try:
        with open(path, "wb") as log_file:
            log_file.write(data)
    except OSError as exc:
        logger.warning("could not write termination log to %s: %s", path, exc)


def _truncate_utf8_message(message: str, max_bytes: int) -> bytes:
    """Encode and truncate without splitting a multi-byte UTF-8 character."""
    encoded = message.encode("utf-8")
    if len(encoded) <= max_bytes:
        return encoded

    result = bytearray()
    for char in message:
        piece = char.encode("utf-8")
        if len(result) + len(piece) > max_bytes:
            break
        result.extend(piece)
    return bytes(result)


def _api_error_detail(exc: ApiException) -> str:
    """Human-readable detail from a Kubernetes API error."""
    parts: list[str] = []
    if exc.reason:
        parts.append(str(exc.reason))
    if exc.body:
        body = exc.body
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        parts.append(str(body))
    return "; ".join(parts) if parts else "unknown error"


def _raise_publish_error(operation: str, path: str, exc: BaseException) -> None:
    """Raise PublishError for a failed Kubernetes API call."""
    if isinstance(exc, ApiException):
        raise PublishError(f"{operation} {path}: {exc.status} {_api_error_detail(exc)}") from exc
    if isinstance(exc, (Urllib3TimeoutError, MaxRetryError)):
        raise PublishError(f"{operation} {path}: request timed out") from exc
    raise exc


def publish_agent_result(
    template: dict[str, Any],
    agent_output: dict[str, Any],
    *,
    failure_reason: str | None = None,
    started_at: datetime,
    completed_at: datetime,
    input_tokens: int = 0,
    output_tokens: int = 0,
    api: CustomObjectsApi | None = None,
) -> None:
    """Assemble status from schema-driven agent output and publish the Result CR."""
    kind = template.get("kind")
    if not isinstance(kind, str):
        msg = "result-template missing kind"
        raise ValueError(msg)

    status = build_status(
        kind,
        agent_output,
        failure_reason=failure_reason,
        started_at=started_at,
        completed_at=completed_at,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    publish_result_cr(template, status, api=api)


def publish_result_cr(
    template: dict[str, Any],
    status: dict[str, Any],
    *,
    api: CustomObjectsApi | None = None,
) -> None:
    """Create a Result CR from template (metadata + spec) and update its status.

    Idempotent create: HTTP 409 AlreadyExists is ignored so status can be
    patched on retry.
    """
    group, version, plural, namespace, name = _parse_template(template)
    client = api if api is not None else _default_api()
    body = _create_body(template)

    try:
        created = client.create_namespaced_custom_object(
            group=group,
            version=version,
            namespace=namespace,
            plural=plural,
            body=body,
            _request_timeout=K8S_REQUEST_TIMEOUT,
        )
        resource_version = created["metadata"]["resourceVersion"]
    except (ApiException, Urllib3TimeoutError, MaxRetryError) as exc:
        path = _kind_path(plural, namespace, name)
        if isinstance(exc, ApiException) and exc.status == 409:
            resource_version = _get_resource_version(
                client,
                group,
                version,
                namespace,
                plural,
                name,
            )
        else:
            _raise_publish_error("create", path, exc)

    status_body = {
        "apiVersion": f"{group}/{version}",
        "kind": template["kind"],
        "metadata": {
            "name": name,
            "namespace": namespace,
            "resourceVersion": resource_version,
        },
        "status": status,
    }
    try:
        client.replace_namespaced_custom_object_status(
            group=group,
            version=version,
            namespace=namespace,
            plural=plural,
            name=name,
            body=status_body,
            _request_timeout=K8S_REQUEST_TIMEOUT,
        )
    except (ApiException, Urllib3TimeoutError, MaxRetryError) as exc:
        path = _kind_path(plural, namespace, name)
        _raise_publish_error("update status", path, exc)


def _default_api() -> CustomObjectsApi:
    """Build a CustomObjectsApi from in-cluster config or local kubeconfig.

    Inside a Kubernetes pod, only in-cluster config is used. A stray kubeconfig
    on disk MUST NOT be consulted — that could target the wrong cluster.
    """
    in_cluster = bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    try:
        config.load_incluster_config()
    except config.ConfigException as exc:
        if in_cluster:
            raise PublishError(f"in-cluster Kubernetes config unavailable: {exc}") from exc
        try:
            config.load_kube_config()
        except config.ConfigException as kube_exc:
            raise PublishError(f"local kubeconfig unavailable: {kube_exc}") from kube_exc
    return CustomObjectsApi(ApiClient())


def _get_resource_version(
    client: CustomObjectsApi,
    group: str,
    version: str,
    namespace: str,
    plural: str,
    name: str,
) -> str:
    """Fetch the current resourceVersion for an existing Result CR."""
    obj = client.get_namespaced_custom_object(
        group=group,
        version=version,
        namespace=namespace,
        plural=plural,
        name=name,
        _request_timeout=K8S_REQUEST_TIMEOUT,
    )
    if not isinstance(obj, dict):
        raise PublishError(f"get {plural}/{namespace}/{name}: unexpected response type")
    metadata = obj.get("metadata")
    if not isinstance(metadata, dict):
        raise PublishError(f"get {plural}/{namespace}/{name}: missing metadata")
    rv = metadata.get("resourceVersion")
    if not isinstance(rv, str) or not rv:
        raise PublishError(f"get {plural}/{namespace}/{name}: missing resourceVersion")
    return rv


def _parse_template(template: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Extract group, version, plural, namespace, and name from result-template."""
    api_version = template.get("apiVersion")
    if not isinstance(api_version, str) or "/" not in api_version:
        msg = "result-template missing valid apiVersion"
        raise ValueError(msg)

    group, version = api_version.split("/", 1)
    kind = template.get("kind")
    if not isinstance(kind, str) or kind not in _KIND_TO_PLURAL:
        msg = f"result-template has unsupported kind: {kind!r}"
        raise ValueError(msg)

    metadata = template.get("metadata")
    if not isinstance(metadata, dict):
        msg = "result-template missing metadata"
        raise ValueError(msg)

    name = metadata.get("name")
    namespace = metadata.get("namespace")
    if not isinstance(name, str) or not name:
        msg = "result-template metadata.name is required"
        raise ValueError(msg)
    if not isinstance(namespace, str) or not namespace:
        msg = "result-template metadata.namespace is required"
        raise ValueError(msg)

    return group, version, _KIND_TO_PLURAL[kind], namespace, name


def _create_body(template: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of the template without status for CR creation."""
    body = copy.deepcopy(template)
    body.pop("status", None)
    return body


def _kind_path(plural: str, namespace: str, name: str) -> str:
    """Format a plural/namespace/name path for error messages."""
    return f"{plural}/{namespace}/{name}"
