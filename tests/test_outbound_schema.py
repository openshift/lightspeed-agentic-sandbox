"""Independent checks of the shared OLS-4192 version-1 wire contract."""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
from contextlib import nullcontext
from typing import Any

import httpx
import pytest
from opentelemetry import trace

from lightspeed_agentic import connection_audit as audit
from lightspeed_agentic.providers.audit_support import instrument_httpx_client
from tests.helpers import attributes
from tests.helpers import otlp_requests as otlp_requests

pytestmark = pytest.mark.usefixtures("connection_audit_enabled")
RUN = "11111111-1111-4111-8111-111111111111"
OTHER_RUN = "22222222-2222-4222-8222-222222222222"

# Deliberately independent of the producer and the cluster-evidence validator.
FIELDS = {
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
REQUIRED = {
    "schema_version",
    "event",
    "timestamp",
    "service.name",
    "attempt_id",
    "request.kind",
    "network.transport",
}


@pytest.fixture(params=[False, True], ids=["no-context", "unsampled-context"])
def active_context(request):
    span = trace.NonRecordingSpan(
        trace.SpanContext(
            trace_id=int("1234567890abcdef1234567890abcdef", 16),
            span_id=int("1234567890abcdef", 16),
            is_remote=False,
            trace_flags=trace.TraceFlags(0),
        )
        if request.param
        else trace.INVALID_SPAN_CONTEXT
    )
    with trace.use_span(span):
        yield request.param


def rows(capsys) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


@pytest.mark.usefixtures("active_context")
@pytest.mark.parametrize("kind", ["llm", "mcp", "credential"])
def test_scoped_attempts_emit_the_shared_schema(kind, capsys, active_context):
    scope = (
        audit.llm_call(model_key=1, logical_call_id="PRIVATE-flow")
        if kind == "llm"
        else audit.mcp_request()
        if kind == "mcp"
        else audit.auth_request("google")
    )
    with scope:
        attempt = audit.start_attempt(audit.Destination("https", "model.example", 443))
        assert attempt is not None
        attempt.response_headers(200)
    caller_context = trace.get_current_span().get_span_context()
    later_span = trace.NonRecordingSpan(trace.SpanContext(trace_id=42, span_id=43, is_remote=False))
    with trace.use_span(later_span):
        attempt.finish(response_complete=True)
        assert trace.get_current_span() is later_span
    assert trace.get_current_span().get_span_context() == caller_context
    start, end = rows(capsys)
    assert start["event"] == "outbound.request.started"
    assert end["event"] == "outbound.request.finished"
    assert start["attempt_id"] == end["attempt_id"]
    for record in (start, end):
        assert REQUIRED <= record.keys() <= FIELDS
        assert record["service.name"] == "lightspeed-agentic-sandbox"
        assert record["request.kind"] == kind
        assert record["server.address"] == "model.example"
        assert record["server.port"] == 443
        assert record["network.transport"] == "tcp"
        if active_context:
            assert record["trace_id"] == "1234567890abcdef1234567890abcdef"
            assert record["span_id"] == "1234567890abcdef"
        else:
            assert "trace_id" not in record
            assert "span_id" not in record
        assert None not in record.values()
        assert "PRIVATE" not in json.dumps(record)
    assert not {"outcome", "duration_ms", "request.body.size"} & start.keys()
    assert end["outcome"] == "success"
    assert end["http.response.status_code"] == 200
    assert end["duration_ms"] >= 0
    assert "error.type" not in end


@pytest.mark.usefixtures("active_context")
@pytest.mark.parametrize(
    ("status", "error", "complete", "outcome", "category"),
    [
        (403, None, True, "http_error", "http_error"),
        (None, socket.gaierror(), False, "transport_error", "dns_error"),
        (None, httpx.ConnectError("PRIVATE"), False, "transport_error", "connect_error"),
        (None, ssl.SSLCertVerificationError(), False, "transport_error", "tls_error"),
        (None, TimeoutError(), False, "transport_error", "timeout"),
        (200, httpx.RemoteProtocolError("PRIVATE"), False, "transport_error", "protocol_error"),
        (200, None, False, "transport_error", "protocol_error"),
        (None, RuntimeError("PRIVATE"), False, "transport_error", "unknown_error"),
        (None, asyncio.CancelledError(), False, "cancelled", "cancelled"),
    ],
)
def test_shared_outcomes_and_controlled_errors(capsys, status, error, complete, outcome, category):
    with audit.auth_request("google"):
        attempt = audit.start_attempt(audit.Destination("https", "token.example", 443))
        assert attempt is not None
        if status is not None:
            attempt.response_headers(status)
        attempt.finish(response_complete=complete, error=error)
    start, end = rows(capsys)
    assert end["outcome"] == outcome
    assert end["error.type"] == category
    assert end.keys() <= FIELDS
    assert None not in end.values()
    if status is None:
        assert "http.response.status_code" not in end
    assert "PRIVATE" not in json.dumps([start, end])


@pytest.mark.usefixtures("active_context")
@pytest.mark.parametrize("kind", ["llm", "mcp"])
def test_http_method_and_only_mcp_tool_attribution(capsys, kind):
    scope = (
        audit.llm_call(model_key=1, logical_call_id=None)
        if kind == "llm"
        else audit.mcp_request("lookup")
    )
    with (
        scope,
        httpx.Client(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"ok"))
        ) as client,
    ):
        instrument_httpx_client(client)
        assert (
            client.post("https://model.example/PRIVATE?key=PRIVATE", content=b"PRIVATE").text
            == "ok"
        )
    start, end = rows(capsys)
    for record in (start, end):
        assert record["http.request.method"] == "POST"
        assert record.keys() <= FIELDS
        if kind == "llm":
            assert "gen_ai.tool.name" not in record
        else:
            assert record["gen_ai.tool.name"] == "lookup"
        assert "PRIVATE" not in json.dumps(record)
    assert "request.body.size" not in start


def emit_record(capsys, **fields):
    audit.emit(
        {
            "schema_version": 1,
            "event": "outbound.request.started",
            "timestamp": "2026-11-05T10:00:00+00:00",
            "service.name": "lightspeed-agentic-sandbox",
            "attempt_id": "33333333-3333-4333-8333-333333333333",
            "request.kind": "mcp",
            "network.transport": "tcp",
            "trace_id": "1234567890abcdef1234567890abcdef",
            "span_id": "1234567890abcdef",
            **fields,
        }
    )
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("agenticrun.uid", "PRIVATE-invalid-uuid"),
        ("agenticrun.uid", "urn:uuid:" + RUN),
        ("agenticrun.uid", "{" + RUN + "}"),
        ("agenticrun.uid", True),
        ("agenticrun.uid", "1" * 10000),
        ("agenticrun.phase", "PRIVATE-phase"),
        ("agenticrun.phase", "analysis\nPRIVATE"),
        ("agenticrun.phase", "Analysis"),
        ("server.address", "PRIVATE.example/path?token=PRIVATE"),
        ("server.address", "user:PRIVATE@example.com"),
        ("server.address", "PRIVATE\n.example"),
        ("server.address", "a" * 64 + ".example"),
        ("server.address", ".example"),
        ("server.address", "example..com"),
        ("server.address", "-example.com"),
        ("server.address", "example-.com"),
        ("server.address", "exam_ple.com"),
        ("server.address", "999.999.999.999"),
        ("server.address", "fe80::1%PRIVATE"),
        ("server.address", ".".join(["a" * 63] * 4)),
        ("server.port", 0),
        ("server.port", 65536),
        ("server.port", True),
        ("server.port", 443.0),
        ("server.port", "443"),
        ("network.peer.address", "PRIVATE.example"),
        ("network.peer.address", "192.0.2.1."),
        ("network.peer.address", "fe80::1%PRIVATE"),
        ("gen_ai.tool.name", "PRIVATE\nlookup"),
        ("gen_ai.tool.name", "PRIVATE\tlookup"),
        ("gen_ai.tool.name", "PRIVATE\x7flookup"),
        ("gen_ai.tool.name", "PRIVATE\u2028lookup"),
        ("gen_ai.tool.name", "x" * 257),
        ("gen_ai.tool.name", ""),
        ("http.request.method", "POST /PRIVATE"),
        ("http.request.method", "X" * 33),
        ("http.response.status_code", True),
        ("http.response.status_code", 99),
        ("http.response.status_code", 600),
        ("duration_ms", -1),
        ("duration_ms", True),
        ("request.body.size", -1),
        ("request.body.size", True),
    ],
)
def test_invalid_optional_metadata_is_omitted_without_diagnostics(capsys, caplog, field, value):
    record = emit_record(capsys, **{field: value})
    assert field not in record
    assert record["event"] == "outbound.request.started"
    assert "PRIVATE" not in json.dumps(record)
    assert "PRIVATE" not in caplog.text


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("agenticrun.uid", RUN, RUN),
        ("agenticrun.uid", RUN.replace("-", ""), RUN),
        (
            "agenticrun.uid",
            "ABCDEF01-2345-4678-9ABC-DEF012345678",
            "abcdef01-2345-4678-9abc-def012345678",
        ),
        ("gen_ai.conversation.id", OTHER_RUN, OTHER_RUN),
        ("server.address", "Example.COM.", "example.com"),
        ("server.address", "a" * 63 + ".example", "a" * 63 + ".example"),
        (
            "server.address",
            ".".join(["a" * 63] * 3 + ["b" * 61]),
            ".".join(["a" * 63] * 3 + ["b" * 61]),
        ),
        ("server.address", "2001:0DB8:0000:0000:0000:0000:0000:0001", "2001:db8::1"),
        ("server.address", "192.0.2.1", "192.0.2.1"),
        ("network.peer.address", "2001:0db8::1", "2001:db8::1"),
        ("network.peer.address", "192.0.2.1", "192.0.2.1"),
        ("server.port", 1, 1),
        ("server.port", 65535, 65535),
        ("gen_ai.tool.name", "x", "x"),
        ("gen_ai.tool.name", "lookup_é", "lookup_é"),
        ("gen_ai.tool.name", "x" * 256, "x" * 256),
        ("http.request.method", "X" * 32, "X" * 32),
        ("http.response.status_code", 100, 100),
        ("http.response.status_code", 599, 599),
        ("agenticrun.phase", "analysis", "analysis"),
        ("agenticrun.phase", "execution", "execution"),
        ("agenticrun.phase", "verification", "verification"),
        ("agenticrun.phase", "escalation", "escalation"),
        ("duration_ms", 0, 0),
        ("request.body.size", 0, 0),
    ],
)
def test_valid_metadata_is_preserved_or_canonicalized(capsys, field, value, expected):
    record = emit_record(capsys, **{field: value})
    assert record[field] == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attempt_id", "PRIVATE-invalid"),
        ("trace_id", "0" * 32),
        ("trace_id", "PRIVATE" + "a" * 25),
        ("span_id", "0" * 16),
        ("span_id", "g" * 16),
    ],
)
def test_invalid_identifiers_are_never_serialized(capsys, field, value):
    record = emit_record(capsys, **{field: value})
    assert field not in record


def test_invalid_metadata_does_not_change_real_request_or_primary_error(capsys, monkeypatch):
    monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "PRIVATE-invalid")
    monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_STEP", "PRIVATE-phase")
    error = httpx.ConnectError("PRIVATE-primary")

    def network(request):
        assert request.content == b"PRIVATE-body"
        raise error

    with (
        audit.mcp_request("PRIVATE\nlookup"),
        httpx.Client(transport=httpx.MockTransport(network)) as client,
    ):
        instrument_httpx_client(client)
        with pytest.raises(httpx.ConnectError) as raised:
            client.post("https://model.example", content=b"PRIVATE-body")
    assert raised.value is error
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 2
    assert records[0]["attempt_id"] == records[1]["attempt_id"]
    assert records[1]["error.type"] == "connect_error"
    for record in records:
        assert "agenticrun.uid" not in record
        assert "agenticrun.phase" not in record
        assert "gen_ai.tool.name" not in record
        assert "PRIVATE" not in json.dumps(record)


@pytest.mark.parametrize("override", [False, True])
def test_otlp_filter_cannot_reintroduce_unvalidated_identity(otlp_requests, capsys, override):
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest

    from lightspeed_agentic import tracing

    tracing.init_tracer(agenticrun_uid="PRIVATE-invalid", agenticrun_phase="PRIVATE-phase")
    try:
        with (
            audit.invocation_context(RUN, "verification") if override else nullcontext(),
            audit.auth_request("google"),
        ):
            attempt = audit.start_attempt(audit.Destination("https", "token.example", 443))
            assert attempt is not None
            attempt.response_headers(200)
            attempt.finish(response_complete=True)
    finally:
        tracing.shutdown_tracer()
    local = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    wire = []
    while not otlp_requests.logs.empty():
        export = ExportLogsServiceRequest.FromString(otlp_requests.logs.get_nowait())
        wire.extend(
            record
            for resource in export.resource_logs
            for scope in resource.scope_logs
            for record in scope.log_records
            if scope.scope.name == "lightspeed_agentic.connection_audit"
        )
    assert len(local) == len(wire) == 2
    assert [json.loads(record.body.string_value) for record in wire] == local
    for record in wire:
        attrs = attributes(record)
        assert "PRIVATE" not in json.dumps(attrs)
        if override:
            assert attrs["agenticrun.uid"] == RUN
            assert attrs["agenticrun.phase"] == "verification"
        else:
            assert "agenticrun.uid" not in attrs
            assert "agenticrun.phase" not in attrs
    assert "PRIVATE" not in json.dumps(local)
