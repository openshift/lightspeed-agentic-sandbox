"""Verify sandbox batch runs exported OTLP traces and audit logs to the e2e collector."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

from kubernetes.client import ApiException, CoreV1Api  # type: ignore[import-untyped]

from tests.e2e.k8s_helpers import read_pod_logs
from tests.e2e.suite_setup import DEFAULT_OTEL_DEPLOYMENT

OTEL_COLLECTOR_LABEL = "app=lightspeed-otel-collector"
DEFAULT_POLL_TIMEOUT_SECONDS = 90.0
DEFAULT_POLL_INTERVAL_SECONDS = 3.0
DEFAULT_LOG_TAIL_LINES = None
TOOL_RESULT_INSPECTION_TIMEOUT_SECONDS = 180.0


@dataclass
class _CollectorSpan:
    fields: dict[str, str] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    scope: str = ""
    valid: bool = True


_LOG_PREFIX = re.compile(r"^(?:\[[^\]\r\n]+\]\s+)?(?:\d{4}-\d{2}-\d{2}T\S+\s+)?")
_SPAN_FIELD = re.compile(
    r"^\s*(Trace ID|Parent ID|ID|Name|Kind|Start time|End time|Status code|Status message"
    r"|DroppedAttributesCount)"
    r"\s*:\s*(.*)$"
)
_ATTRIBUTE = re.compile(r"^\s*-> ([\w.]+): (\w+)\((.*)\)$")


def _collector_spans(logs: str) -> list[_CollectorSpan]:
    """Read detailed Collector text, not payload markers or nested event attributes."""
    spans: list[_CollectorSpan] = []
    current: _CollectorSpan | None = None
    scope = ""
    in_attributes = False
    # splitlines() also splits valid Unicode separators inside JSON strings.
    for raw_line in logs.split("\n"):
        line = _LOG_PREFIX.sub("", raw_line).strip()
        if re.fullmatch(r"Span #\d+", line):
            current = _CollectorSpan(scope=scope)
            spans.append(current)
            in_attributes = False
        elif boundary := re.fullmatch(r"(?:\w+\t)?((?:Resource|Scope)\w+|LogRecord) #\d+", line):
            # Timestamp stripping leaves the console level on the first header.
            # Payload strings containing header-like text are not boundaries.
            current = None
            in_attributes = False
            if boundary[1].startswith("Resource"):
                scope = ""
        elif line == "InstrumentationScope" or line.startswith("InstrumentationScope "):
            scope = line.split()[1] if len(line.split()) > 1 else ""
        elif current is not None:
            if line == "Attributes:":
                in_attributes = True
            elif line in ("Events:", "Links:"):
                current = None
                in_attributes = False
            elif not in_attributes and (match := _SPAN_FIELD.fullmatch(line)):
                key, value = match.groups()
                current.valid &= key not in current.fields
                current.fields[key] = value
            elif in_attributes and (match := _ATTRIBUTE.fullmatch(line)):
                key, kind, value = match.groups()
                current.valid &= key not in current.attributes
                # Only these scalar kinds participate in reconstruction checks.
                # Other typed values remain literal instead of being guessed.
                current.attributes[key] = (
                    int(value)
                    if kind == "Int" and re.fullmatch(r"-?\d+", value)
                    else value
                    if kind == "Str"
                    else value == "true"
                    if kind == "Bool" and value in {"true", "false"}
                    else (kind, value)
                )
    return spans


def _timestamp(value: str) -> int:
    """Preserve nanoseconds in Collector Go timestamps and RFC3339 timestamps."""
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(?: \+0000 UTC|Z)",
        value,
    )
    assert match, "Missing or invalid span timestamp"
    seconds = int(datetime.fromisoformat(match[1]).replace(tzinfo=UTC).timestamp())
    return seconds * 1_000_000_000 + int((match[2] or "").ljust(9, "0"))


def _json_attribute(span: _CollectorSpan, name: str) -> Any:
    value = span.attributes.get(name)
    assert isinstance(value, str), f"Missing {name} on span {span.fields.get('ID')}"
    try:
        parsed = json.loads(value)
        json.dumps(parsed, allow_nan=False)  # Reject nonfinite values, including exponent overflow.
    except (ValueError, TypeError) as exc:
        raise AssertionError(f"Invalid JSON in {name} on span {span.fields.get('ID')}") from exc
    return parsed


def _output_parts(span: _CollectorSpan, *, terminal: bool = False) -> list[dict[str, Any]]:
    messages = _json_attribute(span, "gen_ai.output.messages")
    assert isinstance(messages, list), "Invalid output messages"
    assert messages, "Missing output messages"
    parts = []
    for message in messages:
        assert isinstance(message, dict), "Invalid output message"
        assert isinstance(message.get("role"), str), "Missing message role"
        assert isinstance(message.get("parts"), list), "Missing ordered message parts"
        for part in message["parts"]:
            assert isinstance(part, dict), "Invalid output part"
            assert isinstance(part.get("type"), str), "Missing output part discriminator"
            if part["type"] in ("text", "reasoning"):
                assert isinstance(part.get("content"), str), (
                    f"Missing or invalid {part['type']} content"
                )
            parts.append(part)
    if terminal:
        assert len(messages) == 1, "Invalid terminal output messages"
        assert messages[0]["role"] == "assistant", "Invalid terminal output role"
        assert len(parts) == 1, "Invalid terminal output parts"
        assert parts[0]["type"] == "text", "Invalid terminal output part type"
    return parts


def assert_trace_reconstructable(
    logs: str,
    run_uid: str,
    *,
    expect_shaping: bool,
    phase: str = "analysis",
    inspection_outcome: str | None = None,
) -> None:
    """Validate available live evidence without inventing missing SDK metadata."""
    exported = _collector_spans(logs)
    trace_ids = {
        span.fields.get("Trace ID")
        for span in exported
        if span.attributes.get("agenticrun.uid") == run_uid
    }
    assert len(trace_ids) == 1, "Expected one correlated trace"
    trace_id = trace_ids.pop()
    assert isinstance(trace_id, str), "Missing native trace ID"
    assert re.fullmatch(r"[0-9a-f]{32}", trace_id), "Invalid native trace ID"
    assert int(trace_id, 16), "Empty native trace ID"
    spans: dict[str, _CollectorSpan] = {}
    for span in exported:
        if span.fields.get("Trace ID") != trace_id:
            continue
        assert span.valid, "Duplicate span fields or attributes"
        assert span.scope != "gcp.vertex.agent", "Excluded SDK scope reached the collector"
        span_id = span.fields.get("ID", "")
        assert re.fullmatch(r"[0-9a-f]{16}", span_id), "Invalid native span ID"
        assert int(span_id, 16), "Empty native span ID"
        assert span_id not in spans or spans[span_id] == span, "Conflicting duplicate span evidence"
        spans[span_id] = span
    roots = [span for span in spans.values() if span.fields.get("Name") == "invoke_agent"]
    assert len(roots) == 1, "Expected one invocation span"
    root = roots[0]
    assert root.attributes.get("agenticrun.uid") == run_uid
    if "DroppedAttributesCount" in root.fields:
        assert root.fields["DroppedAttributesCount"] == "0", "Invocation attributes were dropped"
    assert root.attributes.get("agenticrun.phase") == phase, "Wrong invocation phase"
    assert isinstance(_json_attribute(root, "gen_ai.input.messages"), list)
    assert isinstance(_json_attribute(root, "gen_ai.system_instructions"), list)
    root_start, root_end = (
        _timestamp(root.fields.get("Start time", "")),
        _timestamp(root.fields.get("End time", "")),
    )
    assert root_start <= root_end, "Invalid invocation interval"
    rejected = inspection_outcome == "malicious"
    assert root.fields.get("Status code") in (("Error",) if rejected else ("Unset", "Ok")), (
        "Unexpected invocation status"
    )
    if rejected:
        assert root.attributes.get("error.type") == "ToolResultSafetyInspectionFailed"
        assert "gen_ai.output.messages" not in root.attributes, "Rejected run has terminal output"
    else:
        _output_parts(root, terminal=True)

    generations, tools, inspections = [], [], []
    for span in spans.values():
        name = span.fields.get("Name", "")
        if name.startswith("chat "):
            generations.append(span)
            assert span.fields.get("Kind") == "Client"
            assert span.attributes.get("gen_ai.request.model")
            assert span.attributes.get("gen_ai.provider.name")
            assert "gen_ai.input.messages" not in span.attributes
            assert "gen_ai.system_instructions" not in span.attributes
        elif name.startswith("execute_tool "):
            tools.append(span)
        elif name == "tool_result.inspection":
            inspections.append(span)
        else:
            continue
        if "DroppedAttributesCount" in span.fields:
            assert span.fields["DroppedAttributesCount"] == "0", "Canonical attributes were dropped"
        assert span.fields.get("Parent ID") == root.fields["ID"], "Orphaned canonical child"
        assert span.fields.get("Status code") in ("Unset", "Ok", "Error"), "Missing child status"
        start, end = (
            _timestamp(span.fields.get("Start time", "")),
            _timestamp(span.fields.get("End time", "")),
        )
        assert root_start <= start <= end <= root_end, "Invalid child interval"
        if name != "tool_result.inspection":
            assert span.attributes.get("agenticrun.uid") == run_uid, "Missing child run correlation"
            assert span.attributes.get("agenticrun.phase") == phase, "Wrong child phase"
    generations.sort(key=lambda span: _timestamp(span.fields["Start time"]))
    assert generations, "Missing model generations"
    next_generation_starts = {
        generation.fields["ID"]: _timestamp(following.fields["Start time"])
        for generation, following in pairwise(generations)
    }
    shaped = not rejected and expect_shaping
    assert not shaped or len(generations) >= 2, "Missing main/shaping generations"
    calls: dict[str, tuple[dict[str, Any], _CollectorSpan]] = {}
    for index, generation in enumerate(generations):
        parts = (
            _output_parts(generation)
            if generation.fields["Status code"] != "Error"
            or "gen_ai.output.messages" in generation.attributes
            else []
        )
        for key in (
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.reasoning.output_tokens",
        ):
            if key in generation.attributes:
                count = generation.attributes[key]
                assert type(count) is int, "Invalid observed generation usage"
                assert count >= 0, "Negative observed generation usage"
        # Structured-output tool calls are format binding, not local executions.
        if shaped and index == len(generations) - 1:
            continue
        for part in parts:
            if part["type"] == "tool_call":
                call_id = part.get("id")
                assert isinstance(call_id, str), "Missing actual tool-call ID"
                assert call_id, "Empty actual tool-call ID"
                assert call_id not in calls, "Ambiguous actual tool-call ID"
                calls[call_id] = (part, generation)
    tool_ids = [tool.attributes.get("gen_ai.tool.call.id") for tool in tools]
    assert len(tool_ids) == len(set(tool_ids)), "Ambiguous tool span IDs"
    assert set(tool_ids) == set(calls), "Unmatched tool-call evidence"
    for tool in tools:
        call, generation = calls[tool.attributes["gen_ai.tool.call.id"]]
        assert tool.attributes.get("gen_ai.tool.name") == call.get("name"), "Tool name mismatch"
        assert _json_attribute(tool, "gen_ai.tool.call.arguments") == call.get("arguments"), (
            "Tool arguments mismatch"
        )
        assert _timestamp(generation.fields["End time"]) <= _timestamp(tool.fields["Start time"]), (
            "Tool precedes its generation"
        )
        if tool.fields["Status code"] == "Error":
            assert "gen_ai.tool.call.result" not in tool.attributes, (
                "Failed tool has successful source result"
            )
        else:
            _json_attribute(tool, "gen_ai.tool.call.result")
        tool_end = _timestamp(tool.fields["End time"])
        next_start = next_generation_starts.get(generation.fields["ID"])
        if next_start is not None:
            assert tool_end <= next_start, "Model continued before tool completion"
        # Match only observed inspections: trusted skill reads legitimately have
        # none. Every observed chunk must finish before the model can continue.
        for inspection in inspections:
            if (
                inspection.attributes.get("gen_ai.tool.call.id")
                != tool.attributes["gen_ai.tool.call.id"]
            ):
                continue
            assert tool_end <= _timestamp(inspection.fields["Start time"]), (
                "Inspection precedes source completion"
            )
            if next_start is not None:
                assert _timestamp(inspection.fields["End time"]) <= next_start, (
                    "Model continued before inspection completion"
                )
    if inspection_outcome:
        matching = [
            span
            for span in inspections
            if span.attributes.get("inspection.outcome") == inspection_outcome
        ]
        assert matching, "Missing matching inspection outcome"
        assert tools, "Missing inspected tool evidence"
        for inspection in matching:
            expected_status = ("Error",) if rejected else ("Unset", "Ok")
            assert inspection.fields["Status code"] in expected_status, "Inspection status mismatch"
            assert isinstance(inspection.attributes.get("inspection.category"), str)
            attempts = inspection.attributes.get("inspection.attempt_count")
            assert type(attempts) is int, "Missing inspection attempt count"
            assert attempts > 0, "Invalid inspection attempt count"
        for inspection in inspections:
            assert inspection.attributes.get("gen_ai.tool.call.id") in calls, (
                "Unmatched inspection call"
            )
        if rejected:
            rejected_ids = {span.attributes["gen_ai.tool.call.id"] for span in matching}
            assert all(
                tool.fields["Status code"] in ("Unset", "Ok")
                for tool in tools
                if tool.attributes["gen_ai.tool.call.id"] in rejected_ids
            ), "Rejected source execution did not retain a successful result"
            rejected_start = min(_timestamp(span.fields["Start time"]) for span in matching)
            assert all(
                _timestamp(span.fields["Start time"]) <= rejected_start for span in generations
            ), "Model continued after rejection"
    if not rejected:
        for key in ("gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"):
            assert type(root.attributes.get(key)) is int, "Missing invocation usage"
            assert root.attributes[key] >= 0, "Negative invocation usage"
            if all(key in span.attributes for span in generations):
                assert root.attributes[key] == sum(span.attributes[key] for span in generations), (
                    "Invocation/generation usage mismatch"
                )


def fetch_otel_collector_logs(
    core_api: CoreV1Api,
    namespace: str,
    *,
    tail_lines: int | None = DEFAULT_LOG_TAIL_LINES,
) -> str:
    """Return available stdout from collector pod(s), without tailing by default."""
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
            chunk = read_pod_logs(
                core_api,
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


def logs_contain_tool_result_inspection_for_run(
    logs: str,
    run_uid: str,
    *,
    expected_tool: str,
    expected_outcome: str = "benign",
) -> bool:
    """Require the intended MCP execution and its isolated inspection, joined by ID."""
    exported = _collector_spans(logs)
    roots = {
        (span.fields.get("Trace ID", ""), span.fields.get("ID", ""))
        for span in exported
        if span.valid
        and span.fields.get("Name") == "invoke_agent"
        and span.attributes.get("agenticrun.uid") == run_uid
    }
    if len(roots) != 1:
        return False
    trace_id, root_id = roots.pop()
    if not re.fullmatch(r"[0-9a-f]{32}", trace_id) or not int(trace_id, 16):
        return False
    if not re.fullmatch(r"[0-9a-f]{16}", root_id) or not int(root_id, 16):
        return False
    spans: dict[str, _CollectorSpan] = {}
    for span in exported:
        if span.fields.get("Trace ID") != trace_id:
            continue
        span_id = span.fields.get("ID", "")
        if not span.valid or not re.fullmatch(r"[0-9a-f]{16}", span_id) or not int(span_id, 16):
            return False
        if span_id in spans and spans[span_id] != span:
            return False
        spans[span_id] = span  # Identical duplicate collector exports are harmless.
    matched = False
    for native in spans.values():
        attrs = native.attributes
        call_id = attrs.get("gen_ai.tool.call.id")
        if not (
            native.fields.get("Name") == f"execute_tool {expected_tool}"
            and native.fields.get("Parent ID") == root_id
            and native.fields.get("Status code") in {"Unset", "Ok"}
            and attrs.get("agenticrun.uid") == run_uid
            and attrs.get("gen_ai.operation.name") == "execute_tool"
            and attrs.get("gen_ai.tool.name") == expected_tool
            and isinstance(call_id, str)
            and call_id
            and "error.type" not in attrs
        ):
            continue
        if (
            sum(
                sibling.fields.get("Name") == native.fields["Name"]
                and sibling.fields.get("Parent ID") == root_id
                and sibling.attributes.get("gen_ai.tool.call.id") == call_id
                for sibling in spans.values()
            )
            != 1
        ):
            continue  # An inspection cannot identify one of two ambiguous executions.
        inspections = [
            span
            for span in spans.values()
            if span.fields.get("Name") == "tool_result.inspection"
            and span.fields.get("Parent ID") == root_id
            and span.attributes.get("tool.name") == expected_tool
            and span.attributes.get("gen_ai.tool.call.id") == call_id
        ]
        if not inspections:
            continue
        chunks = [span.attributes for span in inspections]
        if any(
            chunk.get("inspection.enabled") is not True
            or chunk.get("inspection.result_type") != "result"
            or type(chunk.get("inspection.chunk_count")) is not int
            or type(chunk.get("inspection.chunk_index")) is not int
            or not 0 <= chunk["inspection.chunk_index"] < chunk["inspection.chunk_count"]
            for chunk in chunks
        ):
            continue
        count = chunks[0]["inspection.chunk_count"]
        complete_benign = (
            len(inspections) == count
            and {chunk["inspection.chunk_index"] for chunk in chunks} == set(range(count))
            and all(chunk["inspection.chunk_count"] == count for chunk in chunks)
            and all(
                span.attributes.get("inspection.outcome") == "benign"
                and span.fields.get("Status code") in {"Unset", "Ok"}
                for span in inspections
            )
        )
        if expected_outcome == "malicious":
            # This controlled MCP test returns the same attack on every call.
            # A later rejection cannot rescue a completed benign inspection of it.
            if complete_benign:
                return False
            matched = matched or any(
                span.attributes.get("inspection.outcome") == "malicious"
                and span.fields.get("Status code") == "Error"
                for span in inspections
            )
        elif expected_outcome == "benign" and complete_benign:
            matched = True
    # Collector export order is not execution order: check every matching call.
    return matched


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
    expect_shaping: bool,
    timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    phase: str = "analysis",
) -> str:
    """Poll until the selected successful execution can be reconstructed."""

    def reconstructable(logs: str) -> bool:
        assert_trace_reconstructable(logs, run_uid, expect_shaping=expect_shaping, phase=phase)
        return True

    return _poll_collector_logs(
        core_api,
        namespace,
        run_uid,
        predicate=reconstructable,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        evidence_kind="reconstructable traces",
    )


def wait_for_otel_tool_result_inspection(
    core_api: CoreV1Api,
    namespace: str,
    run_uid: str,
    *,
    expect_shaping: bool,
    expected_tool: str,
    expected_outcome: str = "benign",
    timeout_seconds: float = TOOL_RESULT_INSPECTION_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    phase: str = "analysis",
) -> str:
    """Poll for joined generation/source/inspection evidence and the run outcome."""

    def reconstructable(logs: str) -> bool:
        if not logs_contain_tool_result_inspection_for_run(
            logs, run_uid, expected_tool=expected_tool, expected_outcome=expected_outcome
        ):
            return False
        assert_trace_reconstructable(
            logs,
            run_uid,
            expect_shaping=expect_shaping,
            phase=phase,
            inspection_outcome=expected_outcome,
        )
        return True

    return _poll_collector_logs(
        core_api,
        namespace,
        run_uid,
        predicate=reconstructable,
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
    last_validation_error = ""
    while time.monotonic() < deadline:
        last_logs = fetch_otel_collector_logs(
            core_api,
            namespace,
            tail_lines=tail_lines,
        )
        try:
            if predicate(last_logs):
                return last_logs
        except AssertionError as exc:
            last_validation_error = str(exc)
        time.sleep(poll_interval_seconds)
    snippet = last_logs[-2000:] if last_logs else "(empty collector logs)"
    msg = (
        f"OTEL collector missing {evidence_kind} for run_uid={run_uid} "
        f"after {timeout_seconds}s; "
        f"last validation: {last_validation_error or 'no matching evidence'}; "
        f"recent collector log tail:\n{snippet}"
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
