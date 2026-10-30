"""Shared Kubernetes pod-log reader for batch E2E and collector verification."""

from __future__ import annotations

from kubernetes.client import CoreV1Api  # type: ignore[import-untyped]


def read_pod_logs(
    core_api: CoreV1Api,
    *,
    namespace: str,
    name: str,
    tail_lines: int | None = 200,
    timestamps: bool = False,
    request_timeout: tuple[float, float] | None = None,
) -> str:
    """Decode actual log bytes, not the SDK's str(bytes) representation."""
    with core_api.read_namespaced_pod_log(
        name=name,
        namespace=namespace,
        tail_lines=tail_lines,
        timestamps=timestamps,
        _preload_content=False,
        **({"_request_timeout": request_timeout} if request_timeout is not None else {}),
    ) as response:
        return response.data.decode("utf-8", errors="replace")  # type: ignore[no-any-return]
