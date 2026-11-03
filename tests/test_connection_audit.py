"""Metadata-only request lifecycle, context isolation and output gates."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from lightspeed_agentic import connection_audit as audit
from tests.helpers import otlp_requests as otlp_requests

pytestmark = pytest.mark.usefixtures("connection_audit_enabled")


def records(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_start_is_immediate_and_completion_is_exactly_once(capsys, monkeypatch):
    clock = iter([10.0, 11.5])
    monkeypatch.setattr(audit, "monotonic", lambda: next(clock))
    with audit.llm_call(model_key=1, logical_call_id="logical"):
        attempt = audit.start_attempt(audit.Destination("https", "model.example", 443))
        assert attempt is not None
        (start,) = records(capsys)
        assert start["event"] == "outbound.request.started"
        attempt.response_headers(200)
        assert records(capsys) == []
        attempt.finish(response_complete=True)
        attempt.finish(response_complete=False)
        (end,) = records(capsys)
    assert end["event"] == "outbound.request.finished"
    assert end["attempt_id"] == start["attempt_id"]
    assert end["outcome"] == "success"
    assert end["duration_ms"] == 1500
    assert end["http.response.status_code"] == 200


@pytest.mark.asyncio
async def test_call_scope_isolation_and_restore(capsys):
    async def run(index):
        with (
            audit.invocation_context(f"11111111-1111-4111-8111-{index:012d}", "analysis"),
            audit.llm_call(model_key=index, logical_call_id=f"call-{index}"),
        ):
            await asyncio.sleep(0)
            attempt = audit.start_attempt(audit.Destination("https", f"model-{index}.example", 443))
            assert attempt is not None
            attempt.finish(response_complete=False, cancelled=True)

    await asyncio.gather(run(1), run(2))
    observed = records(capsys)
    assert len(observed) == 4
    for record in observed:
        assert (
            record["agenticrun.uid"]
            == "11111111-1111-4111-8111-00000000000" + record["server.address"][6]
        )
        assert record["agenticrun.phase"] == "analysis"
    assert audit.start_attempt(audit.Destination("https", "outside.example", 443)) is None


@pytest.mark.asyncio
async def test_mcp_scopes_isolate_tasks_and_restore_llm_metadata(capsys):
    async def request(index):
        with audit.mcp_request(f"lookup{index}"):
            await asyncio.sleep(0)
            attempt = audit.start_attempt(audit.Destination("https", f"mcp-{index}.example", 443))
            assert attempt is not None
        attempt.response_headers(200)
        attempt.finish(response_complete=True)
        attempt.finish(response_complete=False)

    with audit.llm_call(model_key=10, logical_call_id="llm-call"):
        await asyncio.gather(request(1), request(2))
        attempt = audit.start_attempt(audit.Destination("https", "model.example", 443))
        assert attempt is not None
        attempt.response_headers(200)
        attempt.finish(response_complete=True)
    observed = records(capsys)
    mcp = [row for row in observed if row["request.kind"] == "mcp"]
    assert len(mcp) == 4
    assert len({row["attempt_id"] for row in mcp}) == 2
    for row in mcp:
        assert row["gen_ai.tool.name"] == "lookup" + row["server.address"][4]
    assert [row["request.kind"] for row in observed[-2:]] == ["llm", "llm"]
    assert audit.start_attempt(audit.Destination("https", "outside.example", 443)) is None


def test_no_attempt_without_application_scope(capsys):
    assert audit.start_attempt(audit.Destination("https", "auth.example", 443)) is None
    assert records(capsys) == []


def test_emitter_drops_unapproved_fields_and_non_scalar_values(capsys):
    class PrivateObject:
        def __repr__(self):
            pytest.fail("Audit must never repr an arbitrary object")

    audit.emit(
        {
            "schema_version": 1,
            "event": "outbound.request.started",
            "server.address": PrivateObject(),
            "headers": {"Authorization": "PRIVATE"},
            "body": "PRIVATE",
            "duration_ms": float("nan"),
            "agenticrun.uid": None,
        }
    )
    (record,) = records(capsys)
    assert record == {"schema_version": 1, "event": "outbound.request.started"}


def test_stdout_failure_does_not_prevent_other_handlers(monkeypatch):
    import logging
    import sys

    received = []

    class Receiver(logging.Handler):
        def emit(self, record):
            received.append(json.loads(record.getMessage()))

    def unavailable(_text):
        raise OSError("PRIVATE stdout failure")

    handler = Receiver()
    audit.attach_otlp_handler(handler)
    try:
        monkeypatch.setattr(sys.stdout, "write", unavailable)
        with audit.auth_request("google"):
            attempt = audit.start_attempt(audit.Destination("https", "token.example", 443))
            assert attempt is not None
            attempt.response_headers(200)
            attempt.finish(response_complete=True)
        assert len(received) == 2
    finally:
        audit.detach_otlp_handler(handler)


def test_redirect_is_not_an_http_error(capsys):
    with audit.auth_request("google"):
        attempt = audit.start_attempt(audit.Destination("https", "token.example", 443))
        assert attempt is not None
        attempt.response_headers(307)
        attempt.finish(response_complete=True)
    assert records(capsys)[1]["outcome"] == "success"


@pytest.mark.parametrize("cancelled", [False, True])
def test_primary_cancellation_and_timeout_are_not_confused_by_exception_chains(capsys, cancelled):
    import ssl

    error = asyncio.CancelledError() if cancelled else TimeoutError()
    error.__cause__ = ssl.SSLCertVerificationError() if cancelled else asyncio.CancelledError()
    with audit.auth_request("google"):
        attempt = audit.start_attempt(audit.Destination("https", "token.example", 443))
        assert attempt is not None
        attempt.finish(response_complete=False, error=error)
    end = records(capsys)[1]
    assert end["error.type"] == ("cancelled" if cancelled else "timeout")
    assert end["outcome"] == ("cancelled" if cancelled else "transport_error")


def test_logging_failure_preserves_primary_error(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("PRIVATE disk error")

    monkeypatch.setattr(audit, "emit", unavailable)
    with audit.auth_request("google"):
        attempt = audit.start_attempt(audit.Destination("https", "token.example", 443))
        assert attempt is not None
        attempt.finish(response_complete=False, error=RuntimeError("PRIVATE primary"))


@pytest.mark.parametrize("audit_enabled", ["true", "false"])
@pytest.mark.parametrize("capture_content", ["true", "false"])
@pytest.mark.parametrize("kind", ["llm", "mcp", "credential"])
def test_audit_json_and_otlp_match_without_duplicates(
    otlp_requests, capsys, monkeypatch, audit_enabled, capture_content, kind
):
    from opentelemetry import trace
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest

    from lightspeed_agentic import tracing

    monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", audit_enabled)
    monkeypatch.setenv("LIGHTSPEED_CAPTURE_CONTENT", capture_content)
    monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_STEP", "analysis")
    for _ in range(2):
        tracing.init_tracer()
        with trace.get_tracer("audit-test").start_as_current_span("invoke_agent") as span:
            span_id = span.get_span_context().span_id
            scope = (
                audit.llm_call(model_key=10, logical_call_id=None)
                if kind == "llm"
                else audit.mcp_request()
                if kind == "mcp"
                else audit.auth_request("google")
            )
            with scope:
                attempt = audit.start_attempt(audit.Destination("https", "model.example", 443))
                assert attempt is not None
                attempt.response_headers(200)
                attempt.finish(response_complete=True)
        tracing.shutdown_tracer()
        local = [row for row in records(capsys) if row.get("request.kind") == kind]
        wire = []
        while not otlp_requests.logs.empty():
            request = ExportLogsServiceRequest.FromString(otlp_requests.logs.get_nowait())
            wire.extend(
                record
                for resource in request.resource_logs
                for scope in resource.scope_logs
                for record in scope.log_records
                if record.body.string_value.startswith('{"schema_version":1')
            )
        assert len(local) == len(wire) == (2 if audit_enabled == "true" else 0)
        assert [json.loads(record.body.string_value) for record in wire] == local
        if audit_enabled == "true":
            assert all(int.from_bytes(record.span_id, "big") == span_id for record in wire)
            assert all(
                row["agenticrun.uid"] == "11111111-1111-4111-8111-111111111111" for row in local
            )
            assert all(row["agenticrun.phase"] == "analysis" for row in local)
            assert len({row["attempt_id"] for row in local}) == 1


def test_audit_disabled_preserves_otlp_traces(otlp_requests, monkeypatch, capsys):
    from opentelemetry import trace

    from lightspeed_agentic import tracing
    from tests.helpers import exported_spans

    monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "false")
    tracing.init_tracer()
    with trace.get_tracer("audit-test").start_as_current_span("invoke_agent"):
        pass
    tracing.shutdown_tracer()
    assert [span.name for span in exported_spans(otlp_requests)] == ["invoke_agent"]
    assert records(capsys) == []
    assert otlp_requests.logs.empty()


@pytest.mark.asyncio
async def test_auth_scopes_isolate_concurrent_refresh_and_restore_model_context(capsys):
    def complete():
        attempt = audit.start_attempt(audit.Destination("https", "tokens.example", 443))
        assert attempt is not None
        attempt.response_headers(200)
        attempt.finish(response_complete=True)

    async def refresh(provider):
        with audit.auth_request(provider):
            complete()
            await asyncio.sleep(0)
            with audit.auth_request(provider):
                complete()

    with (
        audit.invocation_context("11111111-1111-4111-8111-111111111111", "analysis"),
        audit.llm_call(model_key=73, logical_call_id="llm-parent"),
    ):
        await asyncio.gather(refresh("google"), refresh("azure"), refresh("aws"))
        complete()
    assert audit.start_attempt(audit.Destination("https", "elsewhere.example", 443)) is None
    observed = records(capsys)
    auth = [row for row in observed if row["request.kind"] == "credential"]
    assert len(auth) == 12
    assert len({row["attempt_id"] for row in auth}) == 6
    assert all(row["agenticrun.uid"] == "11111111-1111-4111-8111-111111111111" for row in auth)
    assert observed[-2]["request.kind"] == "llm"
    assert "gen_ai.tool.name" not in observed[-2]
