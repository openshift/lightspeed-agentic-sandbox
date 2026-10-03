"""Tests for shared GenAI inference and tool span recording."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, SpanKind, StatusCode

from lightspeed_agentic.audit import AuditLogger, resolve_provider_name


def _make_logger(**kwargs: object) -> AuditLogger:
    defaults: dict[str, object] = {
        "phase": "analysis",
        "model": "requested-model",
        "provider": "openai",
    }
    defaults.update(kwargs)
    return AuditLogger(**defaults)  # type: ignore[arg-type]


def test_inference_span_records_standard_content_and_observations(span_exporter) -> None:
    audit = _make_logger(agenticrun_uid="run-uid")
    tracer = trace.get_tracer("test.audit")
    input_messages = [{"role": "user", "parts": [{"type": "text", "content": "question"}]}]
    output_messages = [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "answer"}],
            "finish_reason": "stop",
        }
    ]

    with tracer.start_as_current_span("invoke_agent lightspeed") as parent:
        parent_span_id = parent.get_span_context().span_id
        audit.set_parent_context(trace.set_span_in_context(parent))
        span = audit.start_inference(
            model="requested-model",
            operation="chat",
            input_messages=input_messages,
            system_instructions=[{"type": "text", "content": "instructions"}],
            tool_definitions=[{"type": "function", "name": "execute"}],
            output_type="json",
        )
        audit.end_inference(
            span,
            output_messages=output_messages,
            response_model="actual-model",
            input_tokens=0,
            output_tokens=2,
            reasoning_tokens=0,
            finish_reasons=["stop"],
        )

    client_span = next(
        s for s in span_exporter.get_finished_spans() if s.name == "chat requested-model"
    )
    attrs = dict(client_span.attributes)
    assert client_span.kind == SpanKind.CLIENT
    assert client_span.parent.span_id == parent_span_id
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "requested-model"
    assert attrs["gen_ai.response.model"] == "actual-model"
    assert attrs["gen_ai.output.type"] == "json"
    assert attrs["gen_ai.usage.input_tokens"] == 0
    assert attrs["gen_ai.usage.output_tokens"] == 2
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 0
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert attrs["agenticrun.uid"] == "run-uid"
    assert attrs["agenticrun.phase"] == "analysis"
    assert json.loads(attrs["gen_ai.input.messages"]) == input_messages
    assert json.loads(attrs["gen_ai.system_instructions"]) == [
        {"type": "text", "content": "instructions"}
    ]
    assert json.loads(attrs["gen_ai.tool.definitions"]) == [{"type": "function", "name": "execute"}]
    assert json.loads(attrs["gen_ai.output.messages"]) == output_messages
    assert client_span.status.status_code == StatusCode.UNSET
    assert client_span.events == ()


def test_metadata_only_inference_omits_unavailable_content(span_exporter) -> None:
    audit = _make_logger()
    span = audit.start_inference(
        model="classifier-model",
        operation="chat",
        input_messages=None,
    )
    audit.end_inference(span)

    attrs = dict(span_exporter.get_finished_spans()[0].attributes)
    assert attrs["gen_ai.request.model"] == "classifier-model"
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.system_instructions" not in attrs
    assert "gen_ai.tool.definitions" not in attrs
    assert "gen_ai.output.messages" not in attrs


def test_tool_spans_record_arguments_and_success_only_results(span_exporter) -> None:
    audit = _make_logger(phase="execution", agenticrun_uid="run-uid")
    tracer = trace.get_tracer("test.audit")

    with tracer.start_as_current_span("invoke_agent lightspeed") as parent:
        parent_span_id = parent.get_span_context().span_id
        audit.set_parent_context(trace.set_span_in_context(parent))
        successful = audit.start_tool(
            name="execute",
            call_id="call-1",
            arguments={"command": "safe"},
        )
        audit.end_tool(successful, result={"stdout": "done"})
        failed = audit.start_tool(name="execute", call_id="call-2", arguments={"command": "fail"})
        audit.end_tool(
            failed,
            result={"stdout": "must-not-be-recorded"},
            error=RuntimeError("private tool output"),
        )

    spans = {
        s.attributes["gen_ai.tool.call.id"]: s
        for s in span_exporter.get_finished_spans()
        if s.name == "execute_tool execute"
    }
    success = spans["call-1"]
    success_attrs = dict(success.attributes)
    assert success.kind == SpanKind.INTERNAL
    assert success.parent.span_id == parent_span_id
    assert success_attrs["gen_ai.operation.name"] == "execute_tool"
    assert success_attrs["gen_ai.tool.name"] == "execute"
    assert success_attrs["gen_ai.tool.type"] == "function"
    assert json.loads(success_attrs["gen_ai.tool.call.arguments"]) == {"command": "safe"}
    assert json.loads(success_attrs["gen_ai.tool.call.result"]) == {"stdout": "done"}
    assert success.status.status_code == StatusCode.UNSET
    assert success_attrs["agenticrun.uid"] == "run-uid"
    assert success_attrs["agenticrun.phase"] == "execution"

    failed_attrs = dict(spans["call-2"].attributes)
    assert spans["call-2"].status.status_code == StatusCode.ERROR
    assert failed_attrs["error.type"] == "RuntimeError"
    assert "gen_ai.tool.call.result" not in failed_attrs
    assert "private tool output" not in str(failed_attrs)


def test_correlation_attributes_use_only_configured_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "environment-uid")
    monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_STEP", "environment-phase")

    assert _make_logger(phase="", agenticrun_uid="").correlation_attributes() == {}
    assert _make_logger(
        phase="",
        agenticrun_uid="run-uid",
    ).correlation_attributes() == {"agenticrun.uid": "run-uid"}
    assert _make_logger(
        phase="execution",
        agenticrun_uid="run-uid",
    ).correlation_attributes() == {
        "agenticrun.uid": "run-uid",
        "agenticrun.phase": "execution",
    }


def test_tool_result_json_escapes_unpaired_surrogate_losslessly(span_exporter) -> None:
    raw = "raw output with an unpaired surrogate: \ud800"
    audit = _make_logger()
    span = audit.start_tool(name="execute", call_id="call-surrogate")
    audit.end_tool(span, result=raw)

    finished = span_exporter.get_finished_spans()[0]
    serialized = dict(finished.attributes)["gen_ai.tool.call.result"]
    assert serialized.isascii()
    assert "\\ud800" in serialized
    assert json.loads(serialized) == raw


def test_close_ends_only_outstanding_children(span_exporter) -> None:
    audit = _make_logger()
    tracer = trace.get_tracer("test.audit")

    with tracer.start_as_current_span("invoke_agent lightspeed") as parent:
        audit.set_parent_context(trace.set_span_in_context(parent))
        completed = audit.start_inference(
            model="completed-model",
            operation="chat",
            input_messages=[],
        )
        audit.end_inference(completed, input_tokens=1, output_tokens=2)
        open_inference = audit.start_inference(
            model="open-model",
            operation="chat",
            input_messages=[],
        )
        open_tool = audit.start_tool(name="execute", arguments={"command": "running"})
        audit.close(TimeoutError())
        audit.close(TimeoutError())

    spans = {
        s.name: s for s in span_exporter.get_finished_spans() if s.name != "invoke_agent lightspeed"
    }
    assert spans["chat completed-model"].status.status_code == StatusCode.UNSET
    assert spans["chat completed-model"].end_time is not None
    assert open_inference.is_recording() is False
    assert open_tool.is_recording() is False
    open_spans = [s for s in spans.values() if s.status.status_code == StatusCode.ERROR]
    assert len(open_spans) == 2
    assert all(s.attributes["error.type"] == "TimeoutError" for s in open_spans)


def test_non_recording_span_does_not_serialize_content() -> None:
    audit = _make_logger()
    audit._tracer = MagicMock()
    span = MagicMock()
    span.is_recording.return_value = False
    audit._tracer.start_span.return_value = span

    audit.start_inference(
        model="requested-model",
        operation="chat",
        input_messages=[{"role": "user", "parts": [object()]}],  # type: ignore[list-item]
    )

    span.set_attribute.assert_not_called()
    audit.close()


def test_noop_spans_keep_parallel_lifecycles_independent() -> None:
    shared_span = NonRecordingSpan(SpanContext(trace_id=0, span_id=0, is_remote=False))
    tracer = MagicMock()
    tracer.start_span.return_value = shared_span
    audit = _make_logger()
    audit._tracer = tracer

    first_inference = audit.start_inference(
        model="first-model",
        operation="chat",
        input_messages=[],
    )
    second_inference = audit.start_inference(
        model="second-model",
        operation="chat",
        input_messages=[],
    )
    first_tool = audit.start_tool(name="first-tool")
    second_tool = audit.start_tool(name="second-tool")
    handles = (first_inference, second_inference, first_tool, second_tool)

    assert len({id(span) for span in handles}) == 4
    assert all(not span.get_span_context().is_valid for span in handles)
    assert len(audit._open_spans) == 4

    audit.end_inference(second_inference, input_tokens=1, output_tokens=2)
    audit.end_tool(first_tool)
    assert len(audit._open_spans) == 2
    assert id(first_inference) in audit._open_spans
    assert id(second_tool) in audit._open_spans

    audit.end_tool(second_tool)
    audit.end_inference(first_inference)
    assert audit._open_spans == {}


def test_resolve_provider_name_uses_actual_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_BEDROCK",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "LIGHTSPEED_PROVIDER",
        "LIGHTSPEED_MODEL_PROVIDER",
        "AZURE_OPENAI_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)

    assert resolve_provider_name("deepagents") == "anthropic"
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    assert resolve_provider_name("deepagents") == "gcp.vertex_ai"
    monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    assert resolve_provider_name("deepagents") == "aws.bedrock"

    assert resolve_provider_name("gemini") == "gcp.gemini"
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    assert resolve_provider_name("gemini") == "gcp.vertex_ai"

    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "openai")
    assert resolve_provider_name("openai") == "openai"
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "azure")
    assert resolve_provider_name("openai") == "azure.ai.openai"
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    assert resolve_provider_name("openai") == "gcp.vertex_ai"
