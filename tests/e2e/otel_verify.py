"""Verify sandbox batch runs exported OTLP traces and audit logs to the e2e collector."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from typing import Any
from uuid import UUID

from kubernetes.client import ApiException, CoreV1Api  # type: ignore[import-untyped]

from tests.e2e.k8s_helpers import read_pod_logs
from tests.e2e.suite_setup import DEFAULT_OTEL_DEPLOYMENT

OTEL_COLLECTOR_LABEL = "app=lightspeed-otel-collector"
DEFAULT_POLL_TIMEOUT_SECONDS = 90.0
DEFAULT_POLL_INTERVAL_SECONDS = 3.0
DEFAULT_LOG_TAIL_LINES = None
TOOL_RESULT_INSPECTION_TIMEOUT_SECONDS = 180.0


def _normalize_number(value: float) -> int | float:
    """Keep exact integers, fractions and booleans distinct; never truncate."""
    return int(value) if math.isfinite(value) and value.is_integer() else value


def _load_json(value: str) -> Any:
    # parse_float applies at every depth without changing integers or payload strings.
    return json.loads(value, parse_float=lambda text: _normalize_number(float(text)))


def _same_value(left: Any, right: Any) -> bool:
    """Compare normalized evidence without treating booleans as numeric aliases."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_value(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _same_uuid(left: Any, right: str) -> bool:
    """Compare UUID identities independent of hyphenation, preserving opaque test IDs."""
    if left == right:
        return True
    if not isinstance(left, str):
        return False
    try:
        return UUID(left) == UUID(right)
    except ValueError:
        return False


@dataclass(eq=False)
class _CollectorSpan:
    fields: dict[str, str] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    scope: str = ""
    valid: bool = True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _CollectorSpan) and _same_value(vars(self), vars(other))


def otlp_exports(evidence: str) -> list[dict[str, Any]]:
    """Read frames, never interpret text inside an AnyValue as record boundaries."""
    try:
        exports = [_load_json(line) for line in evidence.split("\n") if line.strip()]
    except ValueError:
        raise AssertionError("Invalid structured OTLP evidence") from None
    assert all(isinstance(export, dict) for export in exports), "Invalid OTLP export envelope"
    return exports


def otlp_value(value: dict[str, Any]) -> Any:
    if not value:
        return None
    assert len(value) == 1, "Ambiguous OTLP AnyValue"
    kind, data = next(iter(value.items()))
    if kind == "intValue":
        assert type(data) in (str, int), "Invalid OTLP integer"
        return int(data)
    if kind == "doubleValue":
        assert type(data) in (int, float) or data in ("NaN", "Infinity", "-Infinity"), (
            "Invalid OTLP double"
        )
        return _normalize_number(float(data))
    if kind == "arrayValue":
        return [otlp_value(item) for item in data.get("values", [])]
    if kind == "kvlistValue":
        attributes, valid = otlp_attributes(data.get("values", []))
        assert valid, "Duplicate OTLP map key"
        return attributes
    expected = {
        "stringValue": str,
        "boolValue": bool,
        "bytesValue": str,
    }
    assert kind in expected, "Invalid OTLP AnyValue"
    assert isinstance(data, expected[kind]), "Invalid OTLP AnyValue"
    return ("bytes", data) if kind == "bytesValue" else data


def otlp_attributes(items: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    attributes = {item["key"]: otlp_value(item.get("value", {})) for item in items}
    return attributes, len(attributes) == len(items)


def _collector_spans(evidence: str) -> list[_CollectorSpan]:
    """Decode only actual spans, never records embedded in attribute strings."""
    spans = []
    for export in otlp_exports(evidence):
        for resource in export.get("resourceSpans", []):
            for scope in resource.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    attributes, valid = otlp_attributes(span.get("attributes", []))
                    fields = {
                        name: str(span.get(key, ""))
                        for name, key in {
                            "Trace ID": "traceId",
                            "Parent ID": "parentSpanId",
                            "ID": "spanId",
                            "Name": "name",
                            "Start time": "startTimeUnixNano",
                            "End time": "endTimeUnixNano",
                        }.items()
                    }
                    fields["Kind"] = {
                        0: "Unspecified",
                        1: "Internal",
                        2: "Server",
                        3: "Client",
                        4: "Producer",
                        5: "Consumer",
                    }.get(span.get("kind", 0), "Unknown")
                    status = span.get("status", {})
                    fields["Status code"] = {0: "Unset", 1: "Ok", 2: "Error"}.get(
                        status.get("code", 0), "Unknown"
                    )
                    fields["Status message"] = status.get("message", "")
                    fields["DroppedAttributesCount"] = str(span.get("droppedAttributesCount", 0))
                    spans.append(
                        _CollectorSpan(
                            fields, attributes, scope.get("scope", {}).get("name", ""), valid
                        )
                    )
    return spans


def _timestamp(value: str) -> int:
    """OTLP encodes nanoseconds directly as decimal integers."""
    assert re.fullmatch(r"[0-9]+", value), "Missing or invalid span timestamp"
    return int(value)


def _json_attribute(span: _CollectorSpan, name: str) -> Any:
    value = span.attributes.get(name)
    assert isinstance(value, str), f"Missing {name} on span {span.fields.get('ID')}"
    try:
        parsed = _load_json(value)
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
        assert _same_value(
            _json_attribute(tool, "gen_ai.tool.call.arguments"), call.get("arguments")
        ), "Tool arguments mismatch"
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


# Independent schema check: do not import the producer's allowlist into its CI oracle.
_CONNECTION_FIELDS = frozenset(
    {
        "schema_version",
        "event",
        "timestamp",
        "service.name",
        "attempt_id",
        "request.kind",
        "server.address",
        "server.port",
        "network.peer.address",
        "network.transport",
        "http.request.method",
        "trace_id",
        "span_id",
        "gen_ai.conversation.id",
        "agenticrun.uid",
        "agenticrun.phase",
        "gen_ai.tool.name",
        "outcome",
        "error.type",
        "http.response.status_code",
        "duration_ms",
        "request.body.size",
    }
)
_REQUIRED_CONNECTION_FIELDS = frozenset(
    {
        "schema_version",
        "event",
        "timestamp",
        "service.name",
        "attempt_id",
        "request.kind",
        "network.transport",
        "agenticrun.uid",
        "agenticrun.phase",
    }
)
_COMPLETION_FIELDS = frozenset(
    {
        "outcome",
        "error.type",
        "duration_ms",
        "http.response.status_code",
        "request.body.size",
    }
)


_CONNECTION_LOG_ATTRIBUTES = frozenset(
    {
        "event",
        "agenticrun.uid",
        "agenticrun.phase",
        "code.file.path",
        "code.function.name",
        "code.line.number",
    }
)


@dataclass
class _CollectorLog:
    scope: str
    fields: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    valid: bool = True


def _collector_logs(evidence: str) -> list[_CollectorLog]:
    """Resource, scope, body and record attributes retain their OTLP boundaries."""
    records = []
    for export in otlp_exports(evidence):
        for resource in export.get("resourceLogs", []):
            for scope in resource.get("scopeLogs", []):
                for record in scope.get("logRecords", []):
                    attributes, valid = otlp_attributes(record.get("attributes", []))
                    records.append(
                        _CollectorLog(
                            scope.get("scope", {}).get("name", ""),
                            {
                                "Body": otlp_value(record.get("body", {})),
                                "Trace ID": record.get("traceId", ""),
                                "Span ID": record.get("spanId", ""),
                                "DroppedAttributesCount": str(
                                    record.get("droppedAttributesCount", 0)
                                ),
                            },
                            attributes,
                            valid,
                        )
                    )
    return records


def assert_connection_audit(
    logs: str,
    run_uid: str,
    *,
    phase: str,
    auth_stdout: str | None = None,
    mcp_addresses: set[str] | None = None,
    require_exported_spans: bool = False,
) -> None:
    """Validate shared v1 records; exported-span linkage is scenario-specific."""
    spans = _collector_spans(logs)
    trace_ids = {
        s.fields.get("Trace ID") for s in spans if s.attributes.get("agenticrun.uid") == run_uid
    }
    span_references = {(s.fields.get("Trace ID"), s.fields.get("ID")) for s in spans}
    tools = {
        s.attributes.get("gen_ai.tool.name")
        for s in spans
        if s.fields.get("Trace ID") in trace_ids
        and s.fields.get("Name", "").startswith("execute_tool ")
    }
    attempts: dict[str, dict[str, dict[str, Any]]] = {}
    observed_auth: list[dict[str, Any]] = []
    for record in _collector_logs(logs):
        if record.scope != "lightspeed_agentic.connection_audit":
            continue
        try:
            body = _load_json(record.fields["Body"])
        except (ValueError, TypeError):
            body = None
        attributed = _same_uuid(record.attributes.get("agenticrun.uid"), run_uid)
        if not attributed and (
            not isinstance(body, dict) or not _same_uuid(body.get("agenticrun.uid"), run_uid)
        ):
            continue
        assert isinstance(body, dict), "Invalid connection-audit JSON body"
        assert record.valid, "Duplicate connection-audit attributes"
        assert record.attributes.keys() <= _CONNECTION_LOG_ATTRIBUTES, (
            "Unexpected connection-audit attribute"
        )
        for name in ("code.file.path", "code.function.name"):
            if name in record.attributes:
                assert isinstance(record.attributes[name], str), "Invalid code-location attribute"
        if "code.line.number" in record.attributes:
            location = record.attributes["code.line.number"]
            assert type(location) is int, "Invalid code-location attribute"
            assert location >= 0, "Invalid code-location attribute"
        assert body.keys() <= _CONNECTION_FIELDS, "Unexpected connection-audit field"
        assert body.keys() >= _REQUIRED_CONNECTION_FIELDS, "Missing connection-audit metadata"
        assert all(
            type(value) in (str, bool, int, float)
            and (type(value) is not float or math.isfinite(value))
            for value in body.values()
        ), "Non-scalar, null or nonfinite connection-audit metadata"
        assert type(body["schema_version"]) is int
        assert body["schema_version"] == 1
        assert body["service.name"] == "lightspeed-agentic-sandbox", "Wrong audit service"
        kind = body["request.kind"]
        assert kind in {"llm", "mcp", "retrieval", "credential", "other"}, "Unknown request kind"
        event = body["event"]
        assert event in {"outbound.request.started", "outbound.request.finished"}, (
            "Unknown request event"
        )
        stage = event.rsplit(".", 1)[1]
        assert _same_uuid(body["agenticrun.uid"], run_uid), "Wrong audit run"
        assert _same_uuid(record.attributes.get("agenticrun.uid"), run_uid), "Wrong audit run"
        assert body["agenticrun.phase"] == record.attributes.get("agenticrun.phase") == phase, (
            "Wrong audit phase"
        )
        assert record.attributes.get("event") == event, "Wrong OTLP event attribute"
        assert record.fields.get("DroppedAttributesCount", "0") == "0", "Audit attributes dropped"
        try:
            timestamp = datetime.fromisoformat(body["timestamp"])
        except (TypeError, ValueError) as exc:
            raise AssertionError("Invalid audit timestamp") from exc
        assert timestamp.tzinfo is not None, "Audit timestamp lacks timezone"
        assert isinstance(body["attempt_id"], str)
        assert re.fullmatch(
            r"[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", body["attempt_id"]
        ), "Invalid attempt UUID"
        assert body["network.transport"] in {"tcp", "udp", "other"}, "Invalid network transport"
        if "http.request.method" in body:
            assert isinstance(body["http.request.method"], str)
            assert re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", body["http.request.method"]), (
                "Invalid HTTP method"
            )
        has_context = "trace_id" in body
        assert has_context == ("span_id" in body), "Incomplete audit trace context"
        for name, width in (("trace_id", 32), ("span_id", 16)):
            native = record.fields.get("Trace ID" if name == "trace_id" else "Span ID", "")
            if has_context:
                value = body[name]
                assert isinstance(value, str)
                assert re.fullmatch(rf"[0-9a-f]{{{width}}}", value)
                assert int(value, 16), "Empty native OTLP correlation"
                assert native == value, "Wrong native OTLP correlation"
            else:
                assert native in ("", "0" * width), "Unreported native OTLP correlation"
        if has_context and require_exported_spans:
            assert body["trace_id"] in trace_ids, "Audit belongs to another trace"
            assert (body["trace_id"], body["span_id"]) in span_references, (
                "Audit references missing span"
            )
        if "server.address" in body:
            host = body["server.address"]
            assert isinstance(host, str), "Invalid authority host"
            assert 0 < len(host) <= 253, "Invalid authority host"
            assert not any(c.isspace() or c in "/?#@" for c in host), "Not an authority-only host"
        if "server.port" in body:
            port = body["server.port"]
            assert type(port) is int, "Invalid authority port"
            assert 1 <= port <= 65535, "Invalid authority port"
        if "network.peer.address" in body:
            import ipaddress

            assert isinstance(body["network.peer.address"], str), "Invalid peer IP"
            try:
                ipaddress.ip_address(body["network.peer.address"])
            except ValueError as exc:
                raise AssertionError("Invalid peer IP") from exc
        if "gen_ai.tool.name" in body:
            name = body["gen_ai.tool.name"]
            assert kind == "mcp", "Tool name on non-MCP request"
            assert isinstance(name, str), "Invalid tool name"
            assert 0 < len(name) <= 256, "Invalid tool name"
            assert name.isprintable(), "Invalid tool name"
            if mcp_addresses is not None:
                assert name in tools, "Unmatched MCP tool name"
        if kind == "mcp" and mcp_addresses is not None:
            assert body.get("server.address") in mcp_addresses, "Unconfigured MCP address"
        if stage == "started":
            assert not _COMPLETION_FIELDS.intersection(body), "Premature completion metadata"
        else:
            assert {"outcome", "duration_ms"} <= body.keys(), "Missing completion metadata"
            assert type(body["duration_ms"]) in (int, float)
            assert body["duration_ms"] >= 0
            status = body.get("http.response.status_code")
            assert status is None or (type(status) is int and 100 <= status <= 599)
            if "request.body.size" in body:
                assert type(body["request.body.size"]) is int, "Invalid metadata count"
                assert body["request.body.size"] >= 0, "Invalid metadata count"
            category = body.get("error.type")
            assert category in {
                None,
                "dns_error",
                "connect_error",
                "tls_error",
                "timeout",
                "protocol_error",
                "http_error",
                "cancelled",
                "unknown_error",
            }, "Unknown error category"
            outcome = body["outcome"]
            assert outcome in {"success", "http_error", "transport_error", "cancelled"}, (
                "Unknown outcome"
            )
            if outcome == "success":
                assert category is None, "Invalid successful outcome"
                assert status is None or status < 400, "Invalid successful outcome"
            elif outcome == "http_error":
                assert category == "http_error", "Invalid HTTP error"
                assert status is not None, "Invalid HTTP error"
                assert status >= 400, "Invalid HTTP error"
            elif outcome == "cancelled":
                assert category == "cancelled", "Invalid cancellation"
            else:
                assert category is not None, "Invalid transport error"
                assert category not in {"http_error", "cancelled"}, "Invalid transport error"
        pair = attempts.setdefault(body["attempt_id"], {})
        assert stage not in pair or _same_value(pair[stage], body), (
            "Conflicting duplicate audit event"
        )
        pair[stage] = body
        if kind == "credential" and not any(
            _same_value(body, previous) for previous in observed_auth
        ):
            observed_auth.append(body)
    starts = []
    mutable = _COMPLETION_FIELDS | {"event", "timestamp"}
    for pair in attempts.values():
        assert pair.keys() == {"started", "finished"}, "Unpaired HTTP attempt"
        start, end = pair["started"], pair["finished"]
        assert _same_value(
            {key: value for key, value in start.items() if key not in mutable},
            {key: value for key, value in end.items() if key not in mutable},
        ), "Changed attempt identity"
        starts.append(start)
    assert any(body["request.kind"] == "llm" for body in starts), "Missing LLM request attempts"
    if auth_stdout is not None:
        emitted_auth: list[dict[str, Any]] = []
        for line in auth_stdout.split("\n"):
            try:
                emitted = _load_json(line)
            except ValueError:
                continue
            if (
                isinstance(emitted, dict)
                and emitted.get("event")
                in {"outbound.request.started", "outbound.request.finished"}
                and emitted.get("request.kind") == "credential"
                and _same_uuid(emitted.get("agenticrun.uid"), run_uid)
                and emitted.get("agenticrun.phase") == phase
                and not any(_same_value(emitted, previous) for previous in emitted_auth)
            ):
                emitted_auth.append(emitted)
        assert len(observed_auth) == len(emitted_auth), "Auth stdout/OTEL mismatch"
        assert all(
            any(_same_value(body, observed) for observed in observed_auth) for body in emitted_auth
        ), "Auth stdout/OTEL mismatch"
    if mcp_addresses is not None:
        assert mcp_addresses <= {
            body.get("server.address")
            for body in starts
            if body["request.kind"] == "mcp" and "gen_ai.tool.name" not in body
        }, "Missing MCP discovery/session requests"
        assert any(
            body["request.kind"] == "mcp" and "gen_ai.tool.name" in body for body in starts
        ), "Missing MCP tool session"


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


def fetch_otel_evidence(core_api: CoreV1Api, namespace: str) -> str:
    """Use the runner's Kubernetes credentials, without an external Route or port-forward."""
    with core_api.connect_get_namespaced_service_proxy_with_path(
        name="http:lightspeed-mock-mcp:mcp",
        namespace=namespace,
        path="otel",
        _preload_content=False,
        _request_timeout=(3.0, 10.0),
    ) as response:
        return response.data.decode("utf-8")


def logs_contain_traces_for_run(logs: str, run_uid: str) -> bool:
    return any(span.attributes.get("agenticrun.uid") == run_uid for span in _collector_spans(logs))


def _is_native_execution(
    native: _CollectorSpan, *, run_uid: str, root_id: str, expected_tool: str
) -> bool:
    """Identify a successful execution of the intended MCP tool under this root."""
    attrs = native.attributes
    call_id = attrs.get("gen_ai.tool.call.id")
    return bool(
        native.fields.get("Name") == f"execute_tool {expected_tool}"
        and native.fields.get("Parent ID") == root_id
        and native.fields.get("Status code") in {"Unset", "Ok"}
        and attrs.get("agenticrun.uid") == run_uid
        and attrs.get("gen_ai.operation.name") == "execute_tool"
        and attrs.get("gen_ai.tool.name") == expected_tool
        and isinstance(call_id, str)
        and call_id
        and "error.type" not in attrs
    )


def _is_unambiguous_call(
    native: _CollectorSpan, spans: dict[str, _CollectorSpan], *, root_id: str
) -> bool:
    """Require exactly one execution for this validated native call identity."""
    call_id = native.attributes["gen_ai.tool.call.id"]
    return (
        sum(
            sibling.fields.get("Name") == native.fields["Name"]
            and sibling.fields.get("Parent ID") == root_id
            and sibling.attributes.get("gen_ai.tool.call.id") == call_id
            for sibling in spans.values()
        )
        == 1
    )


def _is_complete_benign(inspections: list[_CollectorSpan]) -> bool:
    """Require the validated chunks to cover the entire result with benign outcomes."""
    chunks = [span.attributes for span in inspections]
    count = chunks[0]["inspection.chunk_count"]
    return (
        len(inspections) == count
        and {chunk["inspection.chunk_index"] for chunk in chunks} == set(range(count))
        and all(chunk["inspection.chunk_count"] == count for chunk in chunks)
        and all(
            span.attributes.get("inspection.outcome") == "benign"
            and span.fields.get("Status code") in {"Unset", "Ok"}
            for span in inspections
        )
    )


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
        if not _is_native_execution(
            native, run_uid=run_uid, root_id=root_id, expected_tool=expected_tool
        ):
            continue
        if not _is_unambiguous_call(native, spans, root_id=root_id):
            continue  # An inspection cannot identify one of two ambiguous executions.
        call_id = native.attributes["gen_ai.tool.call.id"]
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
        complete_benign = _is_complete_benign(inspections)
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
    """Require actual selected-run log attributes, not matching text in a body."""
    return any(
        _same_uuid(record.attributes.get("agenticrun.uid"), run_uid)
        and record.attributes.get("agenticrun.phase") == phase
        and isinstance(record.attributes.get("event"), str)
        for record in _collector_logs(logs)
    )


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

    return _poll_otel_evidence(
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
    mcp_addresses: set[str] | None = None,
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
        if mcp_addresses is not None:
            assert_connection_audit(
                logs, run_uid, phase=phase, mcp_addresses=mcp_addresses, require_exported_spans=True
            )
        return True

    return _poll_otel_evidence(
        core_api,
        namespace,
        run_uid,
        predicate=reconstructable,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        evidence_kind="tool-result inspection span",
    )


def wait_for_otel_audit_logs(
    core_api: CoreV1Api,
    namespace: str,
    run_uid: str,
    *,
    phase: str,
    auth_stdout: str,
    timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> str:
    """Require LLM records and stdout-to-collector auth delivery for the selected run."""
    assert auth_stdout.strip(), "Missing pod log reference for auth delivery"

    def validated(logs: str) -> bool:
        assert_connection_audit(
            logs, run_uid, phase=phase, auth_stdout=auth_stdout, require_exported_spans=True
        )
        return logs_contain_audit_logs_for_run(logs, run_uid, phase=phase)

    return _poll_otel_evidence(
        core_api,
        namespace,
        run_uid,
        predicate=validated,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        evidence_kind=f"audit logs (phase={phase})",
    )


def _poll_otel_evidence(
    core_api: CoreV1Api,
    namespace: str,
    run_uid: str,
    *,
    predicate: Callable[[str], bool],
    timeout_seconds: float,
    poll_interval_seconds: float,
    evidence_kind: str,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_validation_error = ""
    receiver_id = None
    while time.monotonic() < deadline:
        last_logs = fetch_otel_evidence(core_api, namespace)
        exports = otlp_exports(last_logs)
        identities = {export["receiverId"] for export in exports if "receiverId" in export}
        assert len(identities) == 1, "Missing evidence receiver identity"
        assert all(isinstance(item, str) and item for item in identities), (
            "Missing evidence receiver identity"
        )
        current_id = identities.pop()
        assert receiver_id is None or receiver_id == current_id, (
            "Evidence receiver restarted during polling"
        )
        receiver_id = current_id
        try:
            if predicate(last_logs):
                return last_logs
        except AssertionError as exc:
            last_validation_error = str(exc)
        time.sleep(poll_interval_seconds)
    msg = (
        f"OTEL collector missing {evidence_kind} for run_uid={run_uid} "
        f"after {timeout_seconds}s; "
        f"last validation: {last_validation_error or 'no matching evidence'}"
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
