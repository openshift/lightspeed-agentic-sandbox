"""Helper functions for interacting with agentic.openshift.io/v1alpha1 CRDs."""

from __future__ import annotations

import time
from typing import Any

from kubernetes.client import CoreV1Api, CustomObjectsApi  # type: ignore[import-untyped]

from tests.e2e.k8s_constants import CRD_GROUP, CRD_VERSION, TERMINAL_PHASES, derive_phase

__all__ = [
    "CRD_GROUP",
    "CRD_VERSION",
    "TERMINAL_PHASES",
    "create_agenticrun",
    "derive_phase",
    "fetch_pod_logs",
    "poll_agenticrun_phase",
    "read_pod_logs",
    "read_result_cr",
]

_KIND_TO_PLURAL: dict[str, str] = {
    "AnalysisResult": "analysisresults",
    "ExecutionResult": "executionresults",
    "VerificationResult": "verificationresults",
}


def create_agenticrun(
    api: CustomObjectsApi,
    *,
    namespace: str,
    name: str,
    request: str,
    target_namespaces: list[str] | None = None,
    analysis_agent: str = "default",
    execution: bool = False,
    verification_agent: str | None = None,
) -> dict[str, Any]:
    """Create an AgenticRun custom resource."""
    spec: dict[str, Any] = {
        "request": request,
        "analysis": {"agent": analysis_agent},
    }
    if target_namespaces is not None:
        spec["targetNamespaces"] = target_namespaces
    if execution:
        spec["execution"] = {}
    if verification_agent is not None:
        spec["verification"] = {"agent": verification_agent}

    body: dict[str, Any] = {
        "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
        "kind": "AgenticRun",
        "metadata": {"name": name, "namespace": namespace},
        "spec": spec,
    }

    result: dict[str, Any] = api.create_namespaced_custom_object(
        group=CRD_GROUP,
        version=CRD_VERSION,
        namespace=namespace,
        plural="agenticruns",
        body=body,
    )
    return result


def poll_agenticrun_phase(
    api: CustomObjectsApi,
    *,
    namespace: str,
    name: str,
    target_phases: set[str],
    timeout_seconds: float = 300,
    poll_interval: float = 5,
) -> dict[str, Any] | None:
    """Poll an AgenticRun until it reaches a target or terminal phase.

    Return the run dict when the derived phase is in *target_phases* or
    TERMINAL_PHASES.  Return None on timeout.
    """
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        run: dict[str, Any] = api.get_namespaced_custom_object(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=namespace,
            plural="agenticruns",
            name=name,
        )
        phase = derive_phase(run.get("status", {}).get("conditions", []))
        if phase in target_phases or phase in TERMINAL_PHASES:
            return run
        time.sleep(poll_interval)

    return None


def read_result_cr(
    api: CustomObjectsApi,
    *,
    namespace: str,
    name: str,
    kind: str,
) -> dict[str, Any]:
    """Read a result CR (AnalysisResult, ExecutionResult, VerificationResult)."""
    plural = _KIND_TO_PLURAL.get(kind)
    if plural is None:
        msg = f"Unknown kind: {kind!r}"
        raise ValueError(msg)

    result: dict[str, Any] = api.get_namespaced_custom_object(
        group=CRD_GROUP,
        version=CRD_VERSION,
        namespace=namespace,
        plural=plural,
        name=name,
    )
    return result


def read_pod_logs(
    core_api: CoreV1Api,
    *,
    namespace: str,
    name: str,
    tail_lines: int | None = 200,
    timestamps: bool = False,
) -> str:
    """Decode actual log bytes, not the SDK's str(bytes) representation."""
    with core_api.read_namespaced_pod_log(
        name=name,
        namespace=namespace,
        tail_lines=tail_lines,
        timestamps=timestamps,
        _preload_content=False,
    ) as response:
        return response.data.decode("utf-8", errors="replace")  # type: ignore[no-any-return]


def fetch_pod_logs(
    core_api: CoreV1Api,
    *,
    namespace: str,
    label_selector: str,
    tail_lines: int = 200,
) -> dict[str, str]:
    """Fetch logs from pods matching a label selector."""
    pod_list = core_api.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
    logs: dict[str, str] = {}
    for pod in pod_list.items:
        pod_name: str = pod.metadata.name
        logs[pod_name] = read_pod_logs(
            core_api, name=pod_name, namespace=namespace, tail_lines=tail_lines
        )
    return logs
