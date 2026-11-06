"""Offline shared-schema checks for the two existing BDD scenarios."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import pytest

from tests.e2e import otel_verify
from tests.e2e.batch_runner import RunBatchResult
from tests.e2e.run_result import E2ERunResult
from tests.e2e.steps.then import (
    assert_otel_audit_logs_received,
    assert_otel_tool_result_inspection_received,
)
from tests.helpers import log_export
from tests.helpers import otlp_requests as otlp_requests
from tests.test_otel_verify import _debug_logs
from tests.test_otel_verify import _poll_once as _poll_collector_once
from tests.test_otel_verify import _trace_spans as _generic_trace_spans


def _trace_spans():
    return _generic_trace_spans(tool_name="list_namespaces")


def _poll_once(monkeypatch, logs):
    _poll_collector_once(monkeypatch, logs)
    monkeypatch.setattr(otel_verify.time, "monotonic", Mock(side_effect=[0, 0, 1000]))


def _attempt(kind="llm", number=1, *, tool=None):
    body = {
        "schema_version": 1,
        "event": "outbound.request.started",
        "timestamp": "2026-10-29T10:00:00+00:00",
        "service.name": "lightspeed-agentic-sandbox",
        "attempt_id": f"{number:032x}",
        "request.kind": kind,
        "agenticrun.uid": "selected-run",
        "agenticrun.phase": "analysis",
        "trace_id": "0123456789abcdef0123456789abcdef",
        "span_id": "0000000000000001",
        "server.address": "mock-mcp" if kind == "mcp" else "endpoint.example.test",
        "server.port": 443,
        "network.transport": "tcp",
        "http.request.method": "POST",
    }
    if tool is not None:
        body["gen_ai.tool.name"] = tool
    finished = {
        **body,
        "event": "outbound.request.finished",
        "timestamp": "2026-10-29T10:00:00.002000+00:00",
        "outcome": "success",
        "duration_ms": 2.0,
        "http.response.status_code": 200,
        "request.body.size": 12,
    }
    return [
        {
            "body": value,
            "scope": "lightspeed_agentic.connection_audit",
            "attrs": {
                "event": value["event"],
                "agenticrun.uid": "selected-run",
                "agenticrun.phase": "analysis",
            },
            "trace_id": "0123456789abcdef0123456789abcdef",
            "span_id": "0000000000000001",
        }
        for value in (body, finished)
    ]


def _audit_logs(records, *, prefixed=True):
    return log_export(records, resource_attrs=prefixed)


def _config(provider="openai"):
    return Mock(
        namespace="test-namespace",
        lightspeed_provider=provider,
        job_env={
            "LIGHTSPEED_MCP_SERVERS": json.dumps(
                [
                    {"name": "mock-ocp-mcp", "url": "http://mock-mcp/mcp"},
                ]
            ),
        },
    )


def _context(raw_text="INFO batch starting\n"):
    return {
        "run_uid": "selected-run",
        "run_step": "analysis",
        "run_expects_shaping": True,
        "run_result": E2ERunResult(raw_text=raw_text, batch=RunBatchResult()),
    }


def _stdout(records):
    return "\n".join(json.dumps(record["body"]) for record in records)


@pytest.mark.parametrize("prefixed", [False, True])
def test_simple_otel_scenario_accepts_shared_schema(monkeypatch, prefixed):
    _poll_once(
        monkeypatch, _debug_logs(_trace_spans()) + "\n" + _audit_logs(_attempt(), prefixed=prefixed)
    )
    assert_otel_audit_logs_received(_context(), _config(), Mock())


@pytest.mark.parametrize("kind", ["llm", "mcp", "credential"])
@pytest.mark.parametrize("context", ["absent", "zero", "unexported"])
def test_shared_validator_accepts_audit_without_exported_spans(kind, context):
    records = _attempt() + _attempt(kind, 2)
    if context != "unexported":
        for record in records:
            for name, width in (("trace_id", 32), ("span_id", 16)):
                record["body"].pop(name)
                record[name] = "" if context == "absent" else "0" * width
    otel_verify.assert_connection_audit(_audit_logs(records), "selected-run", phase="analysis")


def test_shared_validator_accepts_hyphenated_run_uuid_for_hex_input():
    run_uid = "11111111111141118111111111111111"
    records = _attempt()
    for record in records:
        record["body"]["agenticrun.uid"] = "11111111-1111-4111-8111-111111111111"
        record["attrs"]["agenticrun.uid"] = "11111111-1111-4111-8111-111111111111"

    otel_verify.assert_connection_audit(_audit_logs(records), run_uid, phase="analysis")


def test_audit_log_presence_accepts_hyphenated_run_uuid_for_hex_input():
    run_uid = "11111111111141118111111111111111"
    records = _attempt()
    for record in records:
        record["body"]["agenticrun.uid"] = "11111111-1111-4111-8111-111111111111"
        record["attrs"]["agenticrun.uid"] = "11111111-1111-4111-8111-111111111111"

    assert otel_verify.logs_contain_audit_logs_for_run(
        _audit_logs(records), run_uid, phase="analysis"
    )


@pytest.mark.usefixtures("connection_audit_enabled")
@pytest.mark.parametrize("context_available", [False, True])
def test_real_otlp_logs_validate_without_recorded_spans(otlp_requests, capsys, context_available):
    from opentelemetry import trace
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest

    from lightspeed_agentic import connection_audit as audit
    from lightspeed_agentic import tracing
    from tests.helpers import attributes

    run_uid = "11111111-1111-4111-8111-111111111111"
    span_context = (
        trace.SpanContext(
            trace_id=int("1234567890abcdef1234567890abcdef", 16),
            span_id=int("1234567890abcdef", 16),
            is_remote=False,
            trace_flags=trace.TraceFlags(0),
        )
        if context_available
        else trace.INVALID_SPAN_CONTEXT
    )
    tracing.init_tracer(agenticrun_uid=run_uid, agenticrun_phase="analysis")
    try:
        with (
            trace.use_span(trace.NonRecordingSpan(span_context)),
            audit.llm_call(model_key=1, logical_call_id=None),
        ):
            attempt = audit.start_attempt(audit.Destination("https", "model.example", 443))
            assert attempt is not None
            attempt.response_headers(200)
            attempt.finish(response_complete=True)
    finally:
        tracing.shutdown_tracer()
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 2
    assert all(("trace_id" in record) == context_available for record in records)
    assert all(("span_id" in record) == context_available for record in records)
    assert otlp_requests.empty()
    wire_records = []
    while not otlp_requests.logs.empty():
        export = ExportLogsServiceRequest.FromString(otlp_requests.logs.get_nowait())
        for resource in export.resource_logs:
            for scope in resource.scope_logs:
                for record in scope.log_records:
                    if scope.scope.name == "lightspeed_agentic.connection_audit":
                        wire_records.append(
                            {
                                "body": record.body.string_value,
                                "scope": scope.scope.name,
                                "attrs": attributes(record),
                                "trace_id": record.trace_id.hex(),
                                "span_id": record.span_id.hex(),
                            }
                        )
    assert [json.loads(record["body"]) for record in wire_records] == records
    otel_verify.assert_connection_audit(_audit_logs(wire_records), run_uid, phase="analysis")


@pytest.mark.parametrize("missing", ["trace_id", "span_id"])
def test_shared_validator_rejects_a_lone_context_id(missing):
    records = _attempt()
    for record in records:
        record["body"].pop(missing)
    with pytest.raises(AssertionError, match="Incomplete audit trace context"):
        otel_verify.assert_connection_audit(_audit_logs(records), "selected-run", phase="analysis")


def test_shared_validator_rejects_unreported_native_context():
    records = _attempt()
    for record in records:
        record["body"].pop("trace_id")
        record["body"].pop("span_id")
    with pytest.raises(AssertionError, match="Unreported native OTLP correlation"):
        otel_verify.assert_connection_audit(_audit_logs(records), "selected-run", phase="analysis")


@pytest.mark.parametrize("change", ["appears", "disappears", "changes"])
def test_shared_validator_rejects_context_substitution_at_finish(change):
    records = _attempt()
    if change == "changes":
        records[-1]["body"]["span_id"] = records[-1]["span_id"] = "f" * 16
    else:
        record = records[0 if change == "appears" else 1]
        for name in ("trace_id", "span_id"):
            record["body"].pop(name)
            record[name] = ""
    with pytest.raises(AssertionError, match="Changed attempt identity"):
        otel_verify.assert_connection_audit(_audit_logs(records), "selected-run", phase="analysis")


@pytest.mark.parametrize("field", ["body", "attribute"])
@pytest.mark.parametrize(
    "text", ["first\nsecond", "nested (parentheses)", "literal\\n and \u0085\u2028\u2029"]
)
def test_sdk_values_are_data_not_audit_evidence(monkeypatch, field, text):
    records = _attempt()
    unrelated = deepcopy(records[0])
    unrelated.update(scope="unrelated.sdk", body=text)
    if field == "attribute":
        unrelated["attrs"]["exception.stacktrace"] = text + "\n" + _audit_logs(records)
    logs = _debug_logs(_trace_spans()) + "\n" + _audit_logs([*records, unrelated])
    _poll_once(monkeypatch, logs)
    assert_otel_audit_logs_received(_context(), _config(), Mock())


@pytest.mark.parametrize("field", ["body", "attribute"])
def test_embedded_records_cannot_supply_missing_audit_evidence(monkeypatch, field):
    unrelated = deepcopy(_attempt()[0])
    unrelated.update(scope="unrelated.sdk", body="diagnostic:\n" + _audit_logs(_attempt()))
    if field == "attribute":
        unrelated["attrs"]["diagnostic"] = unrelated.pop("body")
        unrelated["body"] = "ordinary diagnostic"
    _poll_once(monkeypatch, _debug_logs(_trace_spans()) + "\n" + _audit_logs([unrelated]))
    with pytest.raises(AssertionError, match="Missing LLM request attempts"):
        assert_otel_audit_logs_received(_context(), _config(), Mock())


@pytest.mark.parametrize("kind", ["llm", "credential", "mcp"])
@pytest.mark.parametrize(
    "key",
    ["Authorization", "x-api-key", "http.request.body", "exception.stacktrace", "code.arguments"],
)
def test_owned_audit_rejects_unapproved_log_attributes(monkeypatch, kind, key):
    records = _attempt() + _attempt(kind, 2)
    records[-1]["attrs"][key] = "PRIVATE"
    _poll_once(monkeypatch, _debug_logs(_trace_spans()) + "\n" + _audit_logs(records))
    with pytest.raises(AssertionError, match="Unexpected connection-audit attribute"):
        assert_otel_audit_logs_received(_context(_stdout(records)), _config(), Mock())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("code.file.path", {"headers": "PRIVATE"}),
        ("code.file.path", 42),
        ("code.function.name", ["PRIVATE"]),
        ("code.line.number", "42"),
        ("code.line.number", True),
        ("code.line.number", 42.5),
        ("code.line.number", -1),
    ],
)
def test_code_location_metadata_has_bounded_scalar_types(monkeypatch, key, value):
    records = _attempt()
    records[-1]["attrs"][key] = value
    _poll_once(monkeypatch, _debug_logs(_trace_spans()) + "\n" + _audit_logs(records))
    with pytest.raises(AssertionError, match="Invalid code-location attribute"):
        assert_otel_audit_logs_received(_context(), _config(), Mock())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("event", "llm.request.started"),
        ("event", "auth.request.completed"),
        ("service.name", "other-service"),
        ("request.kind", "auth"),
        ("attempt_id", "not-a-uuid"),
        ("network.transport", "HTTP"),
        ("http.request.method", "POST /PRIVATE"),
        ("trace_id", "0" * 32),
        ("span_id", "0" * 16),
        ("server.address", "https://host/path?secret=PRIVATE"),
        ("server.address", "host name"),
        ("server.port", 0),
        ("server.port", 65536),
        ("server.port", True),
        ("network.peer.address", "not-an-IP"),
        ("gen_ai.tool.name", "not-an-MCP-request"),
        ("timestamp", "not-a-date"),
        ("agenticrun.uid", "other-run"),
        ("agenticrun.phase", "other-phase"),
        ("server.address", None),
        ("server.address", {"body": "PRIVATE"}),
        ("logical_call_id", "flow"),
        ("operation", "agent"),
        ("model", "PRIVATE"),
        ("destination_host", "endpoint.example.test"),
        ("response_body_bytes", 12),
        ("failure_category", "tls"),
        ("decision", "denied"),
        ("chunk_index", 0),
        ("chunk_count", 1),
        ("tool_call_id", "id"),
    ],
)
def test_validator_rejects_wrong_or_legacy_v1_fields(monkeypatch, field, value):
    records = _attempt()
    for record in records:
        record["body"][field] = value
    _poll_once(monkeypatch, _debug_logs(_trace_spans()) + "\n" + _audit_logs(records))
    with pytest.raises(AssertionError):
        assert_otel_audit_logs_received(_context(), _config(), Mock())


@pytest.mark.parametrize(
    "defect",
    [
        "no-records",
        "no-start",
        "no-finish",
        "wrong-scope",
        "missing-field",
        "invalid-json",
        "wrong-native-trace",
        "wrong-native-span",
        "wrong-trace",
        "unpaired-attempt",
        "changed-destination",
        "conflicting-duplicate",
        "start-size",
        "null-error",
        "bad-duration",
        "bad-outcome",
        "bad-error",
        "http-error-no-status",
        "boolean-size",
    ],
)
def test_simple_scenario_rejects_broken_audit_evidence(monkeypatch, defect):
    records = _attempt()
    start, end = [record["body"] for record in records]
    if defect == "no-records":
        records = []
    elif defect == "no-start":
        records.pop(0)
    elif defect == "no-finish":
        records.pop()
    elif defect == "wrong-scope":
        for record in records:
            record["scope"] = "unrelated.logger"
    elif defect == "missing-field":
        end.pop("network.transport")
    elif defect == "invalid-json":
        records[-1]["body"] = "not-json"
    elif defect in {"wrong-native-trace", "wrong-native-span"}:
        records[-1]["trace_id" if defect == "wrong-native-trace" else "span_id"] = "f" * (
            32 if defect == "wrong-native-trace" else 16
        )
    elif defect == "wrong-trace":
        for record in records:
            record["body"]["trace_id"] = record["trace_id"] = "f" * 32
    elif defect == "unpaired-attempt":
        end["attempt_id"] = "f" * 32
    elif defect == "changed-destination":
        end["server.address"] = "different.example"
    elif defect == "conflicting-duplicate":
        records.append(deepcopy(records[-1]))
        records[-1]["body"]["request.body.size"] = 999
    elif defect == "start-size":
        start["request.body.size"] = 0
    elif defect == "null-error":
        end["error.type"] = None
    elif defect == "bad-duration":
        end["duration_ms"] = float("inf")
    elif defect == "bad-outcome":
        end["outcome"] = "failure"
    elif defect == "bad-error":
        end.update({"outcome": "transport_error", "error.type": "transport"})
    elif defect == "http-error-no-status":
        end.update({"outcome": "http_error", "error.type": "http_error"})
        end.pop("http.response.status_code")
    else:
        end["request.body.size"] = False
    _poll_once(monkeypatch, _debug_logs(_trace_spans()) + "\n" + _audit_logs(records))
    with pytest.raises(AssertionError):
        assert_otel_audit_logs_received(_context(), _config(), Mock())


@pytest.mark.parametrize("foreign_span", [False, True])
@pytest.mark.parametrize("kind", ["llm", "mcp", "credential"])
def test_audit_requires_the_exact_exported_span_in_the_run_trace(monkeypatch, foreign_span, kind):
    spans = _trace_spans()
    records = _attempt() + _attempt(kind, 2)
    for record in records[-2:]:
        record["body"]["span_id"] = record["span_id"] = "f" * 16
    if foreign_span:
        other = deepcopy(spans[0])
        other.update({"Trace ID": "e" * 32, "ID": "f" * 16})
        other["attrs"]["agenticrun.uid"] = "other-run"
        spans.append(other)
    _poll_once(monkeypatch, _debug_logs(spans) + "\n" + _audit_logs(records))
    with pytest.raises(AssertionError, match="Audit references missing span"):
        assert_otel_audit_logs_received(_context(_stdout(records)), _config(), Mock())


@pytest.mark.parametrize("spans_first", [False, True])
def test_sdk_child_context_without_repeated_run_attributes_is_valid(monkeypatch, spans_first):
    spans = _trace_spans()
    child = deepcopy(spans[1])
    child.update({"ID": "a" * 16, "Name": "native SDK helper", "attrs": {}})
    spans.append(child)
    records = _attempt()
    for record in records:
        record["body"]["span_id"] = record["span_id"] = child["ID"]
    chunks = [_debug_logs(spans), _audit_logs(records)]
    _poll_once(monkeypatch, "\n".join(chunks if spans_first else reversed(chunks)))
    assert_otel_audit_logs_received(_context(), _config(), Mock())


@pytest.mark.parametrize("floating_stdout", [False, True])
def test_integral_numbers_normalize_without_changing_fractional_timings(
    monkeypatch, floating_stdout
):
    original = _attempt() + _attempt("credential", 2)
    records = deepcopy(original)
    for record in records:
        for key, value in record["body"].items():
            if type(value) is int:
                record["body"][key] = float(value)
        record["attrs"]["code.line.number"] = 42.0
    if floating_stdout:
        records, original = original, records
    _poll_once(monkeypatch, _debug_logs(_trace_spans()) + "\n" + _audit_logs(records))
    assert_otel_audit_logs_received(_context(_stdout(original)), _config(), Mock())


@pytest.mark.parametrize("negotiation", [False, True])
def test_credential_delivery_is_selected_by_stdout_not_provider(monkeypatch, negotiation):
    records = _attempt() + (_attempt("credential", 2) if negotiation else [])
    _poll_once(monkeypatch, _debug_logs(_trace_spans()) + "\n" + _audit_logs(records))
    assert_otel_audit_logs_received(_context(_stdout(records)), _config("not-a-selector"), Mock())


@pytest.mark.parametrize(
    "defect",
    [
        "missing-pair",
        "missing-start",
        "missing-finish",
        "extra-pair",
        "missing-retry",
        "status",
        "size",
        "boolean-size",
        "destination",
        "attempt",
    ],
)
def test_credential_collector_evidence_must_match_stdout(monkeypatch, defect):
    emitted = _attempt("credential", 2)
    observed = deepcopy(emitted)
    if defect == "missing-pair":
        observed = []
    elif defect == "missing-start":
        observed.pop(0)
    elif defect == "missing-finish":
        observed.pop()
    elif defect == "extra-pair":
        observed.extend(_attempt("credential", 3))
    elif defect == "missing-retry":
        emitted.extend(_attempt("credential", 3))
    elif defect == "status":
        observed[-1]["body"]["http.response.status_code"] = 201
    elif defect == "size":
        observed[-1]["body"]["request.body.size"] = 13
    elif defect == "boolean-size":
        emitted[-1]["body"]["request.body.size"] = True
    else:
        for record in observed:
            record["body"]["server.address" if defect == "destination" else "attempt_id"] = (
                "different.example" if defect == "destination" else "f" * 32
            )
    _poll_once(monkeypatch, _debug_logs(_trace_spans()) + "\n" + _audit_logs(_attempt() + observed))
    with pytest.raises(AssertionError):
        assert_otel_audit_logs_received(_context(_stdout(emitted)), _config(), Mock())


def test_credential_stdout_matches_hyphenated_uuid_to_hex_run_uid():
    run_uid = "11111111111141118111111111111111"
    canonical_uid = "11111111-1111-4111-8111-111111111111"
    emitted_auth = _attempt("credential", 2)
    observed = _attempt() + emitted_auth
    for record in observed:
        record["body"]["agenticrun.uid"] = canonical_uid
        record["attrs"]["agenticrun.uid"] = canonical_uid

    otel_verify.assert_connection_audit(
        _audit_logs(observed),
        run_uid,
        phase="analysis",
        auth_stdout=_stdout(emitted_auth),
    )


def test_credential_stdout_ignores_other_runs_phases_and_nested_data(monkeypatch):
    auth = _attempt("credential", 2)
    unrelated = _attempt("credential", 3) + _attempt("credential", 4)
    for record in unrelated[:2]:
        record["body"]["agenticrun.uid"] = "other-run"
    for record in unrelated[2:]:
        record["body"]["agenticrun.phase"] = "other-phase"
    raw = "developer log\n{not-json}\n" + json.dumps({"message": _stdout(auth)})
    raw += "\n" + _stdout(auth + unrelated + auth)
    _poll_once(
        monkeypatch,
        _debug_logs(_trace_spans()) + "\n" + _audit_logs(_attempt() + list(reversed(auth)) + auth),
    )
    assert_otel_audit_logs_received(_context(raw), _config(), Mock())


def test_unavailable_pod_logs_cannot_prove_no_credential_http(monkeypatch):
    _poll_once(monkeypatch, _debug_logs(_trace_spans()) + "\n" + _audit_logs(_attempt()))
    with pytest.raises(AssertionError, match="Missing pod log reference"):
        assert_otel_audit_logs_received(_context(""), _config(), Mock())


@pytest.mark.parametrize("missing", [None, "llm", "discovery", "tool", "tool-name", "server"])
def test_inspection_scenario_joins_allowed_mcp_fields_and_native_trace_evidence(
    monkeypatch, missing
):
    records = [] if missing == "llm" else _attempt()
    if missing != "discovery":
        records += _attempt("mcp", 2)
    if missing != "tool":
        records += _attempt(
            "mcp", 3, tool="list_namespaces" if missing != "tool-name" else "unrelated"
        )
    if missing == "server":
        for record in records:
            if record["body"]["request.kind"] == "mcp":
                record["body"]["server.address"] = "unconfigured.example"
    _poll_once(monkeypatch, _debug_logs(_trace_spans()) + "\n" + _audit_logs(records))
    if missing:
        with pytest.raises(AssertionError):
            assert_otel_tool_result_inspection_received(_context(), _config(), Mock())
    else:
        assert_otel_tool_result_inspection_received(_context(), _config(), Mock())


@pytest.mark.parametrize(
    ("outcome", "category", "status"),
    [
        ("http_error", "http_error", 403),
        ("transport_error", "protocol_error", 200),
        ("cancelled", "cancelled", 200),
    ],
)
def test_retries_allow_failures_early_close_and_out_of_order_export(
    monkeypatch, outcome, category, status
):
    records = _attempt() + _attempt(number=2)
    records[1]["body"].update(
        {"outcome": outcome, "error.type": category, "http.response.status_code": status}
    )
    records.reverse()
    records.append(deepcopy(records[0]))
    _poll_once(monkeypatch, _debug_logs(_trace_spans()) + "\n" + _audit_logs(records))
    assert_otel_audit_logs_received(_context(), _config(), Mock())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra_attribute", "omit_parent_span"),
    [
        (None, False),
        ("Authorization", False),
        ("x-api-key", False),
        (None, True),
    ],
)
@pytest.mark.usefixtures("connection_audit_enabled")
async def test_real_otlp_metadata_bodies_and_native_correlation_pass_bdd_validation(
    otlp_requests: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_attribute: str | None,
    omit_parent_span: bool,
) -> None:
    """Keep the emitter, LoggingHandler, gRPC exporter and protobuf decoding real."""
    from opentelemetry import trace
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    from lightspeed_agentic import connection_audit as audit
    from lightspeed_agentic import tracing
    from tests.helpers import attributes

    tracing.init_tracer(
        agenticrun_uid="11111111-1111-4111-8111-111111111111", agenticrun_phase="analysis"
    )

    def inject_attribute(record: Any) -> bool:
        if extra_attribute is not None:
            setattr(record, extra_attribute, "PRIVATE_ATTRIBUTE_MARKER")
        return True

    monkeypatch.setattr(audit._LOGGER, "filters", [*audit._LOGGER.filters, inject_attribute])
    context = trace.SpanContext(
        trace_id=int("0123456789abcdef0123456789abcdef", 16),
        span_id=1,
        is_remote=False,
        trace_flags=trace.TraceFlags(1),
    )
    scopes = [
        audit.llm_call(model_key=0, logical_call_id=None),
        audit.mcp_request(),
        audit.mcp_request("list_namespaces"),
        audit.auth_request("azure"),
    ]
    with (
        trace.use_span(trace.NonRecordingSpan(context)),
        trace.get_tracer("connection-audit-bdd").start_as_current_span("native audit parent"),
    ):
        for scope in scopes:
            with scope:
                attempt = audit.start_attempt(audit.Destination("https", "mock-mcp", 443))
                assert attempt is not None
                attempt.response_headers(200)
                attempt.finish(response_complete=True)
    assert tracing._state.tracer_provider.force_flush(timeout_millis=5000)
    assert tracing._state.logger_provider.force_flush(timeout_millis=5000)
    native_spans = []
    while not otlp_requests.empty():
        request = ExportTraceServiceRequest.FromString(otlp_requests.get_nowait())
        for resource in request.resource_spans:
            for scope in resource.scope_spans:
                for span in scope.spans:
                    native_spans.append(
                        {
                            "Trace ID": span.trace_id.hex(),
                            "ID": span.span_id.hex(),
                            "Parent ID": span.parent_span_id.hex(),
                            "Name": span.name,
                            "attrs": attributes(span),
                        }
                    )
    assert len(native_spans) == 1
    raw = capsys.readouterr().out
    records = []
    while not otlp_requests.logs.empty():
        request = ExportLogsServiceRequest.FromString(otlp_requests.logs.get_nowait())
        for resource in request.resource_logs:
            for scope in resource.scope_logs:
                for record in scope.log_records:
                    if scope.scope.name != "lightspeed_agentic.connection_audit":
                        continue
                    records.append(
                        {
                            "body": record.body.string_value,
                            "scope": scope.scope.name,
                            "attrs": attributes(record),
                            "trace_id": record.trace_id.hex(),
                            "span_id": record.span_id.hex(),
                        }
                    )
    assert len(records) == 8
    assert all(record["span_id"] == native_spans[0]["ID"] for record in records)
    spans = _trace_spans() + ([] if omit_parent_span else native_spans)
    logs = (_debug_logs(spans) + "\n" + _audit_logs(records)).replace(
        "selected-run", "11111111-1111-4111-8111-111111111111"
    )
    assert all(
        {"code.file.path", "code.function.name", "code.line.number"} <= record["attrs"].keys()
        for record in records
    )
    if extra_attribute is not None:
        assert all(
            record["attrs"][extra_attribute] == "PRIVATE_ATTRIBUTE_MARKER" for record in records
        )
        assert "PRIVATE_ATTRIBUTE_MARKER" not in raw
    for check, run_context in (
        (assert_otel_audit_logs_received, _context(raw)),
        (assert_otel_tool_result_inspection_received, _context()),
    ):
        run_context["run_uid"] = "11111111-1111-4111-8111-111111111111"
        with monkeypatch.context() as polling:
            _poll_once(polling, logs)
            if extra_attribute is None and not omit_parent_span:
                check(run_context, _config(), Mock())
            else:
                message = (
                    "Unexpected connection-audit attribute"
                    if extra_attribute is not None
                    else "Audit references missing span"
                )
                with pytest.raises(AssertionError, match=message):
                    check(run_context, _config(), Mock())
