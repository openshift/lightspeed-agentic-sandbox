"""Tests for audit OTel instrumentation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from lightspeed_agentic.audit import AuditLogger
from lightspeed_agentic.types import (
    ContentBlockStopEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class _InMemorySpanExporter(SpanExporter):
    def __init__(self) -> None:
        self._spans: list[Any] = []

    def export(self, spans: Any) -> SpanExportResult:
        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list[Any]:
        return list(self._spans)

    def shutdown(self) -> None:
        pass


@pytest.fixture
def span_exporter():
    exporter = _InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace.set_tracer_provider(tp)
    yield exporter
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace.set_tracer_provider(TracerProvider())


def _make_logger(**kwargs) -> AuditLogger:
    defaults = {"phase": "analysis", "model": "m", "provider": "p", "enabled": True}
    defaults.update(kwargs)
    return AuditLogger(**defaults)


class TestToolSpanNaming:
    def test_tool_span_uses_execute_tool_name(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls"))
        al.process_event(ToolResultEvent(output="file.txt"))
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "execute_tool bash"

    def test_tool_span_has_gen_ai_attributes(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls -la", call_id="call_1"))
        al.process_event(ToolResultEvent(output="done", call_id="call_1"))
        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["gen_ai.operation.name"] == "execute_tool"
        assert attrs["gen_ai.tool.name"] == "bash"
        assert attrs["gen_ai.tool.call.id"] == "call_1"
        assert attrs["tool.input"] == "ls -la"
        assert attrs["tool.output"] == "done"

    def test_tool_span_kind_internal(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls"))
        al.process_event(ToolResultEvent(output="done"))
        spans = span_exporter.get_finished_spans()
        assert spans[0].kind == trace.SpanKind.INTERNAL


class TestToolSpanLifecycle:
    def test_tool_result_ends_span(self, span_exporter) -> None:  # noqa: ARG002
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls", call_id="c1"))
        assert len(al._tool_spans) == 1
        al.process_event(ToolResultEvent(output="done", call_id="c1"))
        assert len(al._tool_spans) == 0

    def test_parallel_tool_calls_matched_by_id(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls", call_id="c1"))
        al.process_event(ToolCallEvent(name="cat", input="file.txt", call_id="c2"))
        assert len(al._tool_spans) == 2
        al.process_event(ToolResultEvent(output="file.txt", call_id="c1"))
        al.process_event(ToolResultEvent(output="content", call_id="c2"))
        assert len(al._tool_spans) == 0
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2
        by_name = {s.name: dict(s.attributes) for s in spans}
        assert by_name["execute_tool bash"]["tool.output"] == "file.txt"
        assert by_name["execute_tool cat"]["tool.output"] == "content"

    def test_fifo_fallback_when_no_call_id(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls"))
        al.process_event(ToolResultEvent(output="done"))
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert dict(spans[0].attributes)["tool.output"] == "done"

    def test_complete_ends_orphan_tool_spans_with_error(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls", call_id="c1"))
        al.process_event(ToolCallEvent(name="cat", input="f", call_id="c2"))
        al.complete(success=True, input_tokens=0, output_tokens=0)
        assert len(al._tool_spans) == 0
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2
        from opentelemetry.trace import StatusCode

        for s in spans:
            assert s.status.status_code == StatusCode.ERROR


class TestGenAiChoiceEvents:
    def test_text_emits_choice_event_with_content(self, span_exporter) -> None:
        al = _make_logger(capture_content=True)
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat test") as span:
            al.set_parent_context(trace.set_span_in_context(span))
            al.process_event(TextDeltaEvent(text="hello "))
            al.process_event(TextDeltaEvent(text="world"))
            al.process_event(ContentBlockStopEvent())
        spans = span_exporter.get_finished_spans()
        chat_span = next(s for s in spans if s.name == "chat test")
        events = chat_span.events
        assert len(events) == 1
        assert events[0].name == "gen_ai.choice"
        assert events[0].attributes["gen_ai.completion"] == "hello world"

    def test_thinking_emits_choice_event_with_reasoning(self, span_exporter) -> None:
        al = _make_logger(capture_content=True)
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat test") as span:
            al.set_parent_context(trace.set_span_in_context(span))
            al.process_event(ThinkingDeltaEvent(thinking="let me think"))
            al.process_event(ContentBlockStopEvent())
        spans = span_exporter.get_finished_spans()
        chat_span = next(s for s in spans if s.name == "chat test")
        events = chat_span.events
        assert len(events) == 1
        assert events[0].name == "gen_ai.choice"
        assert events[0].attributes["gen_ai.reasoning_content"] == "let me think"

    def test_empty_buffer_no_event(self, span_exporter) -> None:
        al = _make_logger(capture_content=True)
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat test"):
            al.process_event(ContentBlockStopEvent())
        spans = span_exporter.get_finished_spans()
        chat_span = next(s for s in spans if s.name == "chat test")
        assert len(chat_span.events) == 0


class TestComplete:
    def test_sets_usage_attributes_on_span(self, span_exporter) -> None:  # noqa: ARG002
        al = _make_logger()
        span = MagicMock()
        span.is_recording.return_value = True
        al.complete(
            success=True,
            input_tokens=100,
            output_tokens=50,
            reasoning_tokens=10,
            span=span,
        )
        span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 100)
        span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 50)
        span.set_attribute.assert_any_call("gen_ai.usage.reasoning_tokens", 10)

    def test_no_reasoning_attr_when_zero(self, span_exporter) -> None:  # noqa: ARG002
        al = _make_logger()
        span = MagicMock()
        span.is_recording.return_value = True
        al.complete(success=True, input_tokens=10, output_tokens=5, span=span)
        set_calls = {c[0][0] for c in span.set_attribute.call_args_list}
        assert "gen_ai.usage.reasoning_tokens" not in set_calls

    def test_error_status_on_failure(self, span_exporter) -> None:  # noqa: ARG002
        al = _make_logger()
        span = MagicMock()
        span.is_recording.return_value = True
        al.complete(success=False, input_tokens=0, output_tokens=0, span=span)
        span.set_status.assert_called_once()

    def test_no_crash_without_span(self) -> None:
        al = _make_logger()
        al.complete(success=True, input_tokens=0, output_tokens=0)


class TestNoJsonEmission:
    def test_no_stdout_json_from_audit_logger(self, capsys: pytest.CaptureFixture[str]) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls"))
        al.process_event(ToolResultEvent(output="file.txt"))
        al.complete(success=True, input_tokens=100, output_tokens=50)
        out = capsys.readouterr().out
        assert "audit.agent" not in out


class TestContentCapture:
    def test_content_included_when_capture_enabled(self, span_exporter) -> None:
        al = _make_logger(capture_content=True)
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat test") as span:
            al.set_parent_context(trace.set_span_in_context(span))
            al.process_event(TextDeltaEvent(text="hello"))
            al.process_event(ContentBlockStopEvent())
        spans = span_exporter.get_finished_spans()
        chat_span = next(s for s in spans if s.name == "chat test")
        assert chat_span.events[0].attributes["gen_ai.completion"] == "hello"

    def test_content_omitted_when_capture_disabled(self, span_exporter) -> None:
        al = _make_logger(capture_content=False)
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat test") as span:
            al.set_parent_context(trace.set_span_in_context(span))
            al.process_event(TextDeltaEvent(text="hello"))
            al.process_event(ContentBlockStopEvent())
        spans = span_exporter.get_finished_spans()
        chat_span = next(s for s in spans if s.name == "chat test")
        assert len(chat_span.events) == 1
        assert "gen_ai.completion" not in chat_span.events[0].attributes

    def test_thinking_omitted_when_capture_disabled(self, span_exporter) -> None:
        al = _make_logger(capture_content=False)
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat test") as span:
            al.set_parent_context(trace.set_span_in_context(span))
            al.process_event(ThinkingDeltaEvent(thinking="let me think"))
            al.process_event(ContentBlockStopEvent())
        spans = span_exporter.get_finished_spans()
        chat_span = next(s for s in spans if s.name == "chat test")
        assert len(chat_span.events) == 1
        assert "gen_ai.reasoning_content" not in chat_span.events[0].attributes

    def test_tool_io_always_recorded_regardless_of_capture(self, span_exporter) -> None:
        al = _make_logger(capture_content=False)
        al.process_event(ToolCallEvent(name="bash", input="ls -la", call_id="c1"))
        al.process_event(ToolResultEvent(output="done", call_id="c1"))
        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["tool.input"] == "ls -la"
        assert attrs["tool.output"] == "done"


class TestDisabledAudit:
    def test_spans_created_when_disabled(self, span_exporter) -> None:  # noqa: ARG002
        al = _make_logger(enabled=False)
        al.process_event(ToolCallEvent(name="bash", input="ls", call_id="c1"))
        assert len(al._tool_spans) == 1
        al.process_event(ToolResultEvent(output="done", call_id="c1"))
        assert len(al._tool_spans) == 0
