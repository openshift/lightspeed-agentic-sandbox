"""Tests for OTEL tracing initialization and traceparent parsing."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence

import pytest
from google.protobuf.json_format import ParseDict
from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import (
    Link,
    NonRecordingSpan,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
)

import lightspeed_agentic.tracing as _tracing_mod
from lightspeed_agentic.run_agent import run_agent_query
from lightspeed_agentic.tracing import (
    get_tracer,
    init_tracer,
    otel_runtime_enabled,
    parse_traceparent,
    shutdown_tracer,
)
from lightspeed_agentic.types import ResultEvent, ToolCallEvent, ToolResultEvent

from .conftest import MockProvider

_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


@pytest.fixture(autouse=True)
def _reset_tracer_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shut down providers around each test so Batch exporters do not leak."""
    # Keep shutdown fast when tests point OTLP at an unreachable localhost.
    monkeypatch.setenv("OTEL_BSP_EXPORT_TIMEOUT", "1000")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1")
    shutdown_tracer()
    yield
    shutdown_tracer()


class TestParseTraceparent:
    def test_valid_traceparent(self) -> None:
        header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        trace_id, ctx = parse_traceparent(header)
        assert trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx is not None

    def test_none_header_generates_trace_id(self) -> None:
        trace_id, ctx = parse_traceparent(None)
        assert _TRACE_ID_RE.match(trace_id)
        assert ctx is not None

    def test_empty_header_generates_trace_id(self) -> None:
        trace_id, ctx = parse_traceparent("")
        assert _TRACE_ID_RE.match(trace_id)
        assert ctx is not None

    def test_malformed_header_generates_trace_id(self) -> None:
        trace_id, ctx = parse_traceparent("not-a-traceparent")
        assert _TRACE_ID_RE.match(trace_id)
        assert ctx is not None

    def test_wrong_field_count_generates_trace_id(self) -> None:
        trace_id, ctx = parse_traceparent("00-abc-01")
        assert _TRACE_ID_RE.match(trace_id)
        assert ctx is not None

    def test_all_zero_trace_id_generates_new(self) -> None:
        header = "00-00000000000000000000000000000000-b7ad6b7169203331-01"
        trace_id, ctx = parse_traceparent(header)
        assert trace_id != "00000000000000000000000000000000"
        assert _TRACE_ID_RE.match(trace_id)
        assert ctx is not None

    def test_short_parent_id_generates_new(self) -> None:
        header = "00-0af7651916cd43dd8448eb211c80319c-b7ad-01"
        trace_id, ctx = parse_traceparent(header)
        assert trace_id != "0af7651916cd43dd8448eb211c80319c"
        assert _TRACE_ID_RE.match(trace_id)
        assert ctx is not None

    def test_generated_ids_are_unique(self) -> None:
        id1, _ = parse_traceparent(None)
        id2, _ = parse_traceparent(None)
        assert id1 != id2


class TestOtelRuntimeEnabled:
    def test_false_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LIGHTSPEED_AUDIT_ENABLED", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        assert otel_runtime_enabled() is False

    def test_true_when_audit_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        assert otel_runtime_enabled() is True

    def test_true_when_endpoint_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LIGHTSPEED_AUDIT_ENABLED", raising=False)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        assert otel_runtime_enabled() is True


class TestInitTracer:
    def test_init_without_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        init_tracer()
        tracer = get_tracer()
        assert tracer is not None

    def test_init_with_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        init_tracer()
        tracer = get_tracer()
        assert tracer is not None

    def test_init_with_audit_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        init_tracer()
        tracer = get_tracer()
        assert tracer is not None

    def test_get_tracer_returns_named_tracer(self) -> None:
        tracer = get_tracer()
        assert isinstance(tracer, trace.Tracer)

    def test_shutdown_tracer_flushes_without_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        init_tracer()
        shutdown_tracer()

    def test_double_init_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        init_tracer()
        with pytest.raises(RuntimeError, match="already initialized"):
            init_tracer()

    def test_init_after_shutdown_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        init_tracer()
        shutdown_tracer()
        init_tracer()
        assert get_tracer() is not None

    def test_shared_resource_excludes_agenticrun_attrs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "uid-123")
        init_tracer(agenticrun_phase="execution")
        assert _tracing_mod._state.logger_provider is not None
        assert _tracing_mod._state.tracer_provider is not None
        assert (
            _tracing_mod._state.logger_provider.resource
            is _tracing_mod._state.tracer_provider.resource
        )
        attrs = _tracing_mod._state.logger_provider.resource.attributes
        assert "agenticrun.uid" not in attrs
        assert "agenticrun.phase" not in attrs
        assert attrs["service.name"] == "lightspeed-agentic-sandbox"

    def test_log_filter_does_not_invent_missing_phase(self) -> None:
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "message", (), None)
        stamp = _tracing_mod._AgenticRunFilter(agenticrun_uid="run-uid")

        assert stamp.filter(record) is True
        assert record.__dict__["agenticrun.uid"] == "run-uid"
        assert "agenticrun.phase" not in record.__dict__

    def test_logging_handler_attached_when_endpoint_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from opentelemetry.sdk._logs import LoggingHandler

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        init_tracer()
        assert _tracing_mod._state.logging_handler is not None
        assert isinstance(_tracing_mod._state.logging_handler, LoggingHandler)
        assert _tracing_mod._state.logging_handler in logging.getLogger().handlers

    def test_no_logging_handler_without_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        init_tracer()
        assert _tracing_mod._state.logging_handler is None

    def test_span_events_forwarded_to_logs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from opentelemetry.sdk._logs.export import (
            InMemoryLogRecordExporter,
            SimpleLogRecordProcessor,
        )

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "uid-1")
        init_tracer(agenticrun_phase="execution")

        exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
        assert _tracing_mod._state.logger_provider is not None
        _tracing_mod._state.logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(exporter)
        )

        assert _tracing_mod._state.tracer_provider is not None
        tracer = _tracing_mod._state.tracer_provider.get_tracer("test")
        with tracer.start_as_current_span("chat") as span:
            span.add_event("gen_ai.choice", {"gen_ai.completion": "hello"})

        records = exporter.get_finished_logs()
        assert len(records) >= 1
        matching = [
            r for r in records if (r.log_record.attributes or {}).get("event") == "gen_ai.choice"
        ]
        assert matching
        rec = matching[0].log_record
        attrs = rec.attributes or {}
        # Collector postgresexporter reads these record attrs (not Resource).
        # Stamped via logging extra → LoggingHandler (not direct OTel emit).
        assert attrs.get("agenticrun.uid") == "uid-1"
        assert attrs.get("agenticrun.phase") == "execution"
        assert attrs.get("event") == "gen_ai.choice"
        assert "hello" in str(rec.body)
        assert rec.trace_id != 0

    def test_empty_choice_body_when_no_event_attrs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Capture-off style events (no content attrs) still forward with body {}."""
        from opentelemetry.sdk._logs.export import (
            InMemoryLogRecordExporter,
            SimpleLogRecordProcessor,
        )

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "uid-1")
        init_tracer(agenticrun_phase="execution")

        exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
        assert _tracing_mod._state.logger_provider is not None
        _tracing_mod._state.logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(exporter)
        )

        assert _tracing_mod._state.tracer_provider is not None
        tracer = _tracing_mod._state.tracer_provider.get_tracer("test")
        with tracer.start_as_current_span("chat") as span:
            span.add_event("gen_ai.choice", {})

        matching = [
            r
            for r in exporter.get_finished_logs()
            if (r.log_record.attributes or {}).get("event") == "gen_ai.choice"
        ]
        assert matching
        assert str(matching[0].log_record.body) in ("{}", "")

    def test_exception_span_events_not_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from opentelemetry.sdk._logs.export import (
            InMemoryLogRecordExporter,
            SimpleLogRecordProcessor,
        )

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "uid-1")
        init_tracer(agenticrun_phase="execution")

        exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
        assert _tracing_mod._state.logger_provider is not None
        _tracing_mod._state.logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(exporter)
        )

        assert _tracing_mod._state.tracer_provider is not None
        tracer = _tracing_mod._state.tracer_provider.get_tracer("test")
        with tracer.start_as_current_span("chat") as span:
            span.add_event(
                "exception",
                {
                    "exception.type": "ValueError",
                    "exception.message": "boom",
                    "exception.stacktrace": "traceback...",
                },
            )
            span.add_event("gen_ai.choice", {"gen_ai.completion": "ok"})

        events = [
            (r.log_record.attributes or {}).get("event") for r in exporter.get_finished_logs()
        ]
        assert "exception" not in events
        assert "gen_ai.choice" in events

    def test_warns_when_agenticrun_env_unresolved(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        monkeypatch.delenv("LIGHTSPEED_AGENTICRUN_UID", raising=False)
        monkeypatch.delenv("LIGHTSPEED_AGENTICRUN_STEP", raising=False)
        with caplog.at_level(logging.WARNING, logger="lightspeed_agentic.tracing"):
            init_tracer(agenticrun_phase="execution")
        assert any(
            "cannot resolve env" in r.message and "LIGHTSPEED_AGENTICRUN_UID" in r.message
            for r in caplog.records
        )
        assert not any("LIGHTSPEED_AGENTICRUN_STEP" in r.message for r in caplog.records)

    def test_span_events_not_forwarded_when_audit_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from opentelemetry.sdk._logs.export import (
            InMemoryLogRecordExporter,
            SimpleLogRecordProcessor,
        )

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "false")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "uid-1")
        init_tracer()

        exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
        assert _tracing_mod._state.logger_provider is not None
        _tracing_mod._state.logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(exporter)
        )

        assert _tracing_mod._state.tracer_provider is not None
        tracer = _tracing_mod._state.tracer_provider.get_tracer("test")
        with tracer.start_as_current_span("chat") as span:
            span.add_event("gen_ai.choice", {"gen_ai.completion": "hello"})

        matching = [
            r
            for r in exporter.get_finished_logs()
            if (r.log_record.attributes or {}).get("event") == "gen_ai.choice"
        ]
        assert matching == []


@pytest.mark.asyncio
async def test_json_span_attributes_survive_otlp_protobuf_round_trip(span_exporter) -> None:
    text = "left \ud800 café 🌍 middle \udfff right"
    expected_tool_payload = {"text": text}
    tool_payload = json.dumps(expected_tool_payload, ensure_ascii=True, separators=(",", ":"))

    await run_agent_query(
        MockProvider(
            events=[
                ToolCallEvent(name="lookup", input=tool_payload, call_id="call-1"),
                ToolResultEvent(output=tool_payload, call_id="call-1"),
                ResultEvent(text='{"success": true, "summary": "done"}'),
            ]
        ),
        prompt=text,
        system_prompt=text,
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )

    request = _tracing_mod.encode_spans(span_exporter.get_finished_spans())
    round_tripped_request = ExportTraceServiceRequest()
    round_tripped_request.ParseFromString(request.SerializeToString())

    attributes_by_span = {
        span.name: {attribute.key: attribute.value.string_value for attribute in span.attributes}
        for resource_span in round_tripped_request.resource_spans
        for scope_span in resource_span.scope_spans
        for span in scope_span.spans
    }
    invocation_attributes = attributes_by_span["invoke_agent"]
    tool_attributes = attributes_by_span["execute_tool lookup"]
    invocation_json = invocation_attributes["gen_ai.input.messages"]
    tool_arguments_json = tool_attributes["gen_ai.tool.call.arguments"]
    tool_result_json = tool_attributes["gen_ai.tool.call.result"]

    assert json.loads(invocation_json) == [
        {"role": "user", "parts": [{"type": "text", "content": text}]}
    ]
    assert json.loads(tool_arguments_json) == expected_tool_payload
    assert json.loads(tool_result_json) == expected_tool_payload
    for value in (invocation_json, tool_arguments_json, tool_result_json):
        assert r"\ud800" in value
        assert r"\udfff" in value
        assert "café 🌍" in value


class _RecordingTraceExporter(SpanExporter):
    def __init__(self) -> None:
        self.requests: list[ExportTraceServiceRequest] = []
        self.batches: list[list[ReadableSpan]] = []
        self.force_flush_timeouts: list[int] = []
        self.shutdown_count = 0
        self.force_flush_result = True

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.batches.append(list(spans))
        self.requests.append(_tracing_mod.encode_spans(spans))
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self.force_flush_timeouts.append(timeout_millis)
        return self.force_flush_result

    def shutdown(self) -> None:
        self.shutdown_count += 1


def _span_records(requests: Sequence[ExportTraceServiceRequest]) -> dict:
    return {
        span.name: (
            resource_span.resource,
            resource_span.schema_url,
            scope_span.scope,
            scope_span.schema_url,
            span,
        )
        for request in requests
        for resource_span in request.resource_spans
        for scope_span in resource_span.scope_spans
        for span in scope_span.spans
    }


@pytest.mark.parametrize("protocol", ["http/protobuf", "grpc"])
def test_init_tracer_filters_adk_spans_from_exporters_and_preserves_native_logs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    protocol: str,
) -> None:
    otlp_endpoint = "http://collector.example/v1/traces"
    recorded_exporter = _RecordingTraceExporter()
    selected_protocols: list[str] = []

    def exporter_factory(selected_protocol: str):
        def create(*, endpoint: str) -> _RecordingTraceExporter:
            assert endpoint == otlp_endpoint
            selected_protocols.append(selected_protocol)
            return recorded_exporter

        return create

    monkeypatch.setattr(_tracing_mod, "HttpSpanExporter", exporter_factory("http/protobuf"))
    monkeypatch.setattr(_tracing_mod, "GrpcSpanExporter", exporter_factory("grpc"))
    # Keep log export in-memory while exercising the real span-event bridge.
    monkeypatch.setattr(_tracing_mod, "_configure_log_exporter", lambda *_a, **_k: None)
    monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", otlp_endpoint)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", protocol)
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "8")
    monkeypatch.setenv("OTEL_SPAN_EVENT_COUNT_LIMIT", "1")
    monkeypatch.setenv("OTEL_SPAN_LINK_COUNT_LIMIT", "1")
    init_tracer(agenticrun_uid="test-run", agenticrun_phase="execution")

    provider = _tracing_mod._state.tracer_provider
    logger_provider = _tracing_mod._state.logger_provider
    assert provider is not None
    assert logger_provider is not None
    native_exporter = InMemorySpanExporter()  # type: ignore[no-untyped-call]
    log_exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
    provider.add_span_processor(SimpleSpanProcessor(native_exporter))
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))

    parent_context = SpanContext(
        trace_id=int("4bf92f3577b34da6a3ce929d0e0e4736", 16),
        span_id=int("00f067aa0ba902b7", 16),
        is_remote=True,
        trace_flags=TraceFlags(0x01),
    )
    parent = trace.set_span_in_context(NonRecordingSpan(parent_context))
    link_context = SpanContext(
        trace_id=int("7bba9f33312b3c44c8a6d5e2b0835d9b", 16),
        span_id=int("1d0f4a1e2b3c4d5e", 16),
        is_remote=True,
    )
    link = Link(link_context, {"link.metadata": "preserved"})
    discarded_link = Link(parent_context, {"link.metadata": "dropped"})
    start_time = 1_718_000_000_000_000_000

    tracer = provider.get_tracer("lightspeed_agentic.test", "test-1")
    invocation_span = tracer.start_span(
        "invoke_agent",
        context=parent,
        kind=trace.SpanKind.SERVER,
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gcp.vertex.agent.llm_request": "keep non-ADK attributes",
            "custom.metadata": "unchanged",
        },
        start_time=start_time,
    )
    invocation_context = trace.set_span_in_context(invocation_span)

    adk_schema_url = "https://opentelemetry.io/schemas/1.36.0"
    adk_span = provider.get_tracer("gcp.vertex.agent", "2.11.0", adk_schema_url).start_span(
        "call_llm",
        context=invocation_context,
        kind=trace.SpanKind.CLIENT,
        attributes={
            "overflow.metadata": "dropped by the SDK",
            "gen_ai.output.messages": "native model output",
            "gen_ai.usage.input_tokens": 17,
            "gcp.vertex.agent.llm_request": "raw request",
            "gcp.vertex.agent.llm_response": "raw response",
            "gcp.vertex.agent.tool_call_args": "raw tool arguments",
            "gcp.vertex.agent.tool_response": "raw tool response",
            "gcp.vertex.agent.llm_request_extra": "preserve exact-neighbor metadata",
            "custom.metadata": "preserved",
        },
        links=[discarded_link, link],
        start_time=start_time + 1_000,
    )
    adk_span.add_event("discarded.event", timestamp=start_time + 1_250)
    adk_span.add_event(
        "gen_ai.choice",
        {"gen_ai.completion": "native event payload"},
        timestamp=start_time + 1_500,
    )
    adk_span.set_status(Status(StatusCode.ERROR, "native ADK failure"))
    adk_span.end(end_time=start_time + 10_000)

    vendor_other_span = provider.get_tracer("gcp.vertex.agent.other", "test-1").start_span(
        "vendor.other",
        context=invocation_context,
        kind=trace.SpanKind.INTERNAL,
        attributes={"custom.metadata": "vendor passthrough"},
        start_time=start_time + 11_000,
    )
    vendor_other_span.end(end_time=start_time + 11_500)

    generation_span = tracer.start_span(
        "generate_content gemini-test",
        context=invocation_context,
        kind=trace.SpanKind.CLIENT,
        attributes={
            "gen_ai.operation.name": "generate_content",
            "gen_ai.request.model": "gemini-test",
            "custom.metadata": "canonical generation",
        },
        start_time=start_time + 12_000,
    )
    generation_span.set_status(Status(StatusCode.ERROR, "canonical status preserved"))
    generation_span.end(end_time=start_time + 20_000)
    invocation_span.end(end_time=start_time + 30_000)

    assert provider.force_flush(timeout_millis=5000)
    assert logger_provider.force_flush(timeout_millis=5000)

    stdout_requests = [
        ParseDict(json.loads(line), ExportTraceServiceRequest())
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    stdout_spans = _span_records(stdout_requests)
    otlp_spans = _span_records(recorded_exporter.requests)
    expected_order = ["vendor.other", "generate_content gemini-test", "invoke_agent"]
    assert selected_protocols == [protocol]
    assert list(stdout_spans) == expected_order
    assert [span.name for batch in recorded_exporter.batches for span in batch] == expected_order
    assert stdout_spans == otlp_spans
    assert all(record[2].name != "gcp.vertex.agent" for record in stdout_spans.values())

    native_spans = {span.name: span for span in native_exporter.get_finished_spans()}
    native = native_spans["call_llm"]
    vendor_other = native_spans["vendor.other"]
    generation = native_spans["generate_content gemini-test"]
    invocation = native_spans["invoke_agent"]
    assert native.attributes["gen_ai.output.messages"] == "native model output"
    assert native.attributes["gcp.vertex.agent.llm_request"] == "raw request"
    assert native.events[0].attributes["gen_ai.completion"] == "native event payload"
    assert native.status.status_code == StatusCode.ERROR
    assert (
        native.dropped_attributes,
        native.dropped_events,
        native.dropped_links,
    ) == (1, 1, 1)

    assert generation.parent is not None
    assert generation.parent.span_id == invocation.context.span_id
    assert generation.parent.span_id != native.context.span_id
    invocation_record = _span_records([_tracing_mod.encode_spans([invocation])])["invoke_agent"]
    generation_record = _span_records([_tracing_mod.encode_spans([generation])])[
        "generate_content gemini-test"
    ]
    vendor_record = _span_records([_tracing_mod.encode_spans([vendor_other])])["vendor.other"]
    assert stdout_spans["vendor.other"] == vendor_record
    assert vendor_record[2].name == "gcp.vertex.agent.other"
    assert stdout_spans["invoke_agent"] == invocation_record
    assert stdout_spans["generate_content gemini-test"] == generation_record
    assert generation_record[4].parent_span_id == invocation_record[4].span_id
    _, _, invocation_scope, _, invocation_proto = stdout_spans["invoke_agent"]
    invocation_attributes = {
        attribute.key: attribute.value for attribute in invocation_proto.attributes
    }
    assert invocation_proto.name == "invoke_agent"
    assert invocation_attributes["gen_ai.operation.name"].string_value == "invoke_agent"
    assert invocation_attributes["gcp.vertex.agent.llm_request"].string_value == (
        "keep non-ADK attributes"
    )
    assert invocation_scope.name == "lightspeed_agentic.test"

    mixed_exporter = _RecordingTraceExporter()
    filtered_exporter = _tracing_mod._AdkSpanFilteringExporter(mixed_exporter)
    assert (
        filtered_exporter.export([native, vendor_other, generation, invocation])
        == SpanExportResult.SUCCESS
    )
    assert len(mixed_exporter.batches) == 1
    assert mixed_exporter.batches[0][0] is vendor_other
    assert mixed_exporter.batches[0][1] is generation
    assert mixed_exporter.batches[0][2] is invocation
    assert filtered_exporter.export([native]) == SpanExportResult.SUCCESS
    assert len(mixed_exporter.batches) == 1
    assert len(mixed_exporter.requests) == 1
    mixed_exporter.force_flush_result = False
    assert not filtered_exporter.force_flush(timeout_millis=4321)
    assert mixed_exporter.force_flush_timeouts == [4321]
    filtered_exporter.shutdown()
    assert mixed_exporter.shutdown_count == 1

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1
    record = logs[0].log_record
    assert record.attributes["event"] == "gen_ai.choice"
    assert json.loads(str(record.body)) == {"gen_ai.completion": "native event payload"}
    assert record.trace_id == native.context.trace_id
    assert record.span_id == native.context.span_id
    assert record.attributes["agenticrun.uid"] == "test-run"
    assert record.attributes["agenticrun.phase"] == "execution"
