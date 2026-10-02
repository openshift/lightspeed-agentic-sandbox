"""Verify sandbox batch runs exported OTLP traces and audit logs to the e2e collector."""

from __future__ import annotations

import re
import time
from collections.abc import Callable

from kubernetes.client import ApiException, CoreV1Api  # type: ignore[import-untyped]

from tests.e2e.suite_setup import DEFAULT_OTEL_DEPLOYMENT

OTEL_COLLECTOR_LABEL = "app=lightspeed-otel-collector"
DEFAULT_POLL_TIMEOUT_SECONDS = 90.0
DEFAULT_POLL_INTERVAL_SECONDS = 3.0
DEFAULT_LOG_TAIL_LINES = 10000
TOOL_RESULT_INSPECTION_TIMEOUT_SECONDS = 180.0
_TRACE_ID_RE = re.compile(r"\bTrace ID\s*:\s*(\S+)")


def fetch_otel_collector_logs(
    core_api: CoreV1Api,
    namespace: str,
    *,
    tail_lines: int | None = DEFAULT_LOG_TAIL_LINES,
) -> str:
    """Return recent stdout from OTEL collector pod(s) (debug exporter output)."""
    pods = core_api.list_namespaced_pod(
        namespace=namespace,
        label_selector=OTEL_COLLECTOR_LABEL,
    )
    if not pods.items:
        msg = f"no OTEL collector pods in {namespace} (selector {OTEL_COLLECTOR_LABEL})"
        raise RuntimeError(msg)

    chunks: list[str] = []
    for pod in pods.items:
        pod_name = pod.metadata.name
        pod_uid = pod.metadata.uid or "unknown"
        try:
            chunk = core_api.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                tail_lines=tail_lines,
                timestamps=True,
            )
        except ApiException as exc:
            raise RuntimeError(f"read logs for pod/{pod_name}: {exc.reason}") from exc
        chunks.append(f"=== collector pod/{pod_name} uid={pod_uid} ===\n{chunk}")
    return "\n".join(chunks)


def logs_contain_traces_for_run(logs: str, run_uid: str) -> bool:
    """True when debug exporter output includes spans correlated to ``run_uid``."""
    if run_uid not in logs:
        return False
    trace_markers = ("ResourceSpans", "Span #", "Trace ID")
    return any(marker in logs for marker in trace_markers)


def logs_contain_tool_result_inspection_for_run(logs: str, run_uid: str) -> bool:
    """True when an inspection span shares a trace with this run's span."""
    blocks = logs.split("Span #")[1:]
    run_trace_ids = {
        match.group(1)
        for block in blocks
        if f"agenticrun.uid: Str({run_uid})" in block
        if (match := _TRACE_ID_RE.search(block)) is not None
    }
    return any(
        "tool_result.inspection" in block
        and "inspection.outcome: Str(benign)" in block
        and (match := _TRACE_ID_RE.search(block)) is not None
        and match.group(1) in run_trace_ids
        for block in blocks
    )


def logs_contain_audit_logs_for_run(logs: str, run_uid: str, *, phase: str) -> bool:
    """True when debug exporter output includes bridged audit log records for the run."""
    if run_uid not in logs:
        return False
    if phase not in logs:
        return False
    audit_markers = ("LogRecord", "LogsExporter", "gen_ai.choice")
    if not any(marker in logs for marker in audit_markers):
        return False
    return "agenticrun" in logs


def wait_for_otel_traces(
    core_api: CoreV1Api,
    namespace: str,
    run_uid: str,
    *,
    timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> str:
    """Poll collector logs until trace export for ``run_uid`` is visible."""
    return _poll_collector_logs(
        core_api,
        namespace,
        run_uid,
        predicate=lambda logs: logs_contain_traces_for_run(logs, run_uid),
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        evidence_kind="traces",
    )


def wait_for_otel_tool_result_inspection(
    core_api: CoreV1Api,
    namespace: str,
    run_uid: str,
    *,
    timeout_seconds: float = TOOL_RESULT_INSPECTION_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> str:
    """Poll collector logs until a tool-result inspection span is visible."""
    return _poll_collector_logs(
        core_api,
        namespace,
        run_uid,
        predicate=lambda logs: logs_contain_tool_result_inspection_for_run(logs, run_uid),
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        evidence_kind="tool-result inspection span",
        tail_lines=None,
    )


def wait_for_otel_audit_logs(
    core_api: CoreV1Api,
    namespace: str,
    run_uid: str,
    *,
    phase: str,
    timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> str:
    """Poll collector logs until audit log export for ``run_uid`` is visible."""
    return _poll_collector_logs(
        core_api,
        namespace,
        run_uid,
        predicate=lambda logs: logs_contain_audit_logs_for_run(logs, run_uid, phase=phase),
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        evidence_kind=f"audit logs (phase={phase})",
    )


def _poll_collector_logs(
    core_api: CoreV1Api,
    namespace: str,
    run_uid: str,
    *,
    predicate: Callable[[str], bool],
    timeout_seconds: float,
    poll_interval_seconds: float,
    evidence_kind: str,
    tail_lines: int | None = DEFAULT_LOG_TAIL_LINES,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_logs = ""
    while time.monotonic() < deadline:
        last_logs = fetch_otel_collector_logs(
            core_api,
            namespace,
            tail_lines=tail_lines,
        )
        if predicate(last_logs):
            return last_logs
        time.sleep(poll_interval_seconds)
    snippet = last_logs[-2000:] if last_logs else "(empty collector logs)"
    msg = (
        f"OTEL collector missing {evidence_kind} for run_uid={run_uid} "
        f"after {timeout_seconds}s; recent collector log tail:\n{snippet}"
    )
    raise AssertionError(msg)


def assert_otel_deployment_present(core_api: CoreV1Api, namespace: str) -> None:
    """Raise if the e2e OTEL collector Deployment is missing."""
    pods = core_api.list_namespaced_pod(
        namespace=namespace,
        label_selector=OTEL_COLLECTOR_LABEL,
    )
    if pods.items:
        return
    msg = (
        f"OTEL collector pods not found in {namespace} "
        f"(expected deployment/{DEFAULT_OTEL_DEPLOYMENT})"
    )
    raise RuntimeError(msg)
