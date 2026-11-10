"""Tests for run_agent_query and context formatting."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from opentelemetry.trace import SpanKind, StatusCode

from lightspeed_agentic.inspection.errors import ToolResultSafetyInspectionFailed
from lightspeed_agentic.providers.tool_result_middleware import ToolResultInspectionMiddleware
from lightspeed_agentic.run_agent import ContextFormatError, format_context_prefix, run_agent_query
from lightspeed_agentic.types import (
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)

from .conftest import MockProvider


@pytest.mark.asyncio
async def test_run_agent_query_success() -> None:
    """Mock provider completes and returns a successful structured result."""
    result = await run_agent_query(
        MockProvider(),
        prompt="Diagnose the issue",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is True
    assert "mock result" in result.output["summary"]


@pytest.mark.asyncio
async def test_run_agent_query_with_system_prompt() -> None:
    """Custom system_prompt is accepted without changing success semantics."""
    result = await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="Custom persona",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is True


@pytest.mark.parametrize(
    ("audit_enabled", "capture_content"),
    [(False, False), (False, True), (True, False), (True, True)],
)
@pytest.mark.asyncio
async def test_run_agent_query_with_context(
    span_exporter,
    audit_enabled: bool,
    capture_content: bool,
) -> None:
    """Export effective invocation content and preserve the old choice-event gates."""
    prompt = 'Inspect the pod named "café"\nand report ☃.'
    system_prompt = 'System instructions: café "雪"\nsecond line'
    context = {
        "targetNamespaces": ["default", 'café\n"north"'],
        "previousAttempts": [{"attempt": 1, "failureReason": 'previous "failure"\n雪'}],
    }
    output_schema = {"type": "object", "properties": {"success": {"type": "boolean"}}}
    reasoning = 'thinking\n"carefully" 雪'
    completion = 'Searching "pod-a" in café\nnamespace.'
    tool_arguments = {"namespace": 'café\n"north"', "label": "team=ops"}
    tool_result = {"status": "Running", "note": 'retained "as-is"\n雪'}
    terminal_text = (
        '{\n  "success": true,\n  "summary": "done: \\"café\\"\\nsecond line",\n  "stage": 2\n}'
    )
    events = [
        ThinkingDeltaEvent(thinking=reasoning),
        TextDeltaEvent(text=completion),
        ToolCallEvent(
            name="lookup",
            input=json.dumps(tool_arguments, ensure_ascii=False, separators=(",", ":")),
            call_id="call-1",
        ),
        ToolResultEvent(
            output=json.dumps(tool_result, ensure_ascii=False, separators=(",", ":")),
            call_id="call-1",
        ),
        ResultEvent(
            text=terminal_text,
            input_tokens=11,
            output_tokens=7,
            reasoning_tokens=3,
            response_model="observed-model",
        ),
    ]

    class RecordingProvider(MockProvider):
        def __init__(self) -> None:
            super().__init__(events=events)
            self.options: ProviderQueryOptions | None = None

        @property
        def name(self) -> str:
            return "deepagents"

        async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            self.options = options
            async for event in super().query(options):
                yield event

    provider = RecordingProvider()
    result = await run_agent_query(
        provider,
        prompt=prompt,
        system_prompt=system_prompt,
        output_schema=output_schema,
        context=context,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
        audit_enabled=audit_enabled,
        capture_content=capture_content,
        agenticrun_uid="run-uid",
        step="execution",
    )

    effective_prompt = f"{format_context_prefix(context)}\n\n{prompt}"
    assert result.output == {
        "success": True,
        "summary": 'done: "café"\nsecond line',
        "stage": 2,
    }
    assert provider.options is not None
    assert provider.options.prompt == effective_prompt
    assert provider.options.system_prompt == system_prompt
    assert provider.options.output_schema == output_schema

    root_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "invoke_agent"
    )
    attrs = dict(root_span.attributes)
    input_messages = [{"role": "user", "parts": [{"type": "text", "content": effective_prompt}]}]
    system_instructions = [{"type": "text", "content": system_prompt}]
    terminal_messages = [
        {"role": "assistant", "parts": [{"type": "text", "content": terminal_text}]}
    ]
    assert root_span.kind is SpanKind.INTERNAL
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.request.model"] == "test-model"
    assert attrs["agenticrun.uid"] == "run-uid"
    assert attrs["agenticrun.phase"] == "execution"
    assert "gen_ai.provider.name" not in attrs
    assert "gen_ai.response.model" not in attrs
    assert attrs["gen_ai.output.type"] == "json"
    assert json.loads(attrs["gen_ai.input.messages"]) == input_messages
    assert attrs["gen_ai.input.messages"] == json.dumps(
        input_messages, ensure_ascii=False, separators=(",", ":")
    )
    assert json.loads(attrs["gen_ai.system_instructions"]) == system_instructions
    assert attrs["gen_ai.system_instructions"] == json.dumps(
        system_instructions, ensure_ascii=False, separators=(",", ":")
    )
    assert json.loads(attrs["gen_ai.output.messages"]) == terminal_messages
    assert attrs["gen_ai.output.messages"] == json.dumps(
        terminal_messages, ensure_ascii=False, separators=(",", ":")
    )
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 3
    assert "gen_ai.usage.reasoning_tokens" not in attrs

    tool_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "execute_tool lookup"
    )
    tool_attrs = dict(tool_span.attributes)
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == root_span.context.span_id
    assert tool_attrs["gen_ai.tool.call.id"] == "call-1"
    assert json.loads(tool_attrs["gen_ai.tool.call.arguments"]) == tool_arguments
    assert json.loads(tool_attrs["gen_ai.tool.call.result"]) == tool_result

    choice_events = [event for event in root_span.events if event.name == "gen_ai.choice"]
    if not audit_enabled:
        assert choice_events == []
    else:
        assert len(choice_events) == 2
        choice_attrs = [dict(event.attributes or {}) for event in choice_events]
        if capture_content:
            assert {
                key: value for event_attrs in choice_attrs for key, value in event_attrs.items()
            } == {
                "gen_ai.completion": completion,
                "gen_ai.reasoning_content": reasoning,
            }
        else:
            assert choice_attrs == [{}, {}]


@pytest.mark.asyncio
async def test_run_agent_query_with_output_schema() -> None:
    """output_schema is forwarded to the provider query options."""
    schema = {"type": "object", "properties": {"success": {"type": "boolean"}}}
    result = await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=schema,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is True


@pytest.mark.asyncio
async def test_run_agent_query_accepts_traceparent(span_exporter) -> None:
    """W3C traceparent links the invocation span to the remote phase span."""
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    result = await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
        traceparent=traceparent,
    )
    assert result.output["success"] is True

    invocation_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "invoke_agent"
    )
    assert invocation_span.kind is SpanKind.INTERNAL
    assert invocation_span.context.trace_id == int("4bf92f3577b34da6a3ce929d0e0e4736", 16)
    assert invocation_span.parent is not None
    assert invocation_span.parent.span_id == int("00f067aa0ba902b7", 16)
    assert invocation_span.parent.is_remote


@pytest.mark.asyncio
async def test_run_agent_query_stamps_inference_correlation(span_exporter) -> None:
    await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
        agenticrun_uid="run-uid",
        step="execution",
    )

    invocation_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "invoke_agent"
    )
    attrs = dict(invocation_span.attributes)
    assert invocation_span.kind is SpanKind.INTERNAL
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.request.model"] == "test-model"
    assert "gen_ai.provider.name" not in attrs
    assert "gen_ai.output.type" not in attrs
    assert attrs["agenticrun.uid"] == "run-uid"
    assert attrs["agenticrun.phase"] == "execution"


@pytest.mark.asyncio
async def test_run_agent_query_does_not_invent_correlation(span_exporter) -> None:
    await run_agent_query(
        MockProvider(),
        prompt="",
        system_prompt="",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )

    invocation_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "invoke_agent"
    )
    attrs = dict(invocation_span.attributes)
    assert json.loads(attrs["gen_ai.input.messages"]) == [
        {"role": "user", "parts": [{"type": "text", "content": ""}]}
    ]
    assert json.loads(attrs["gen_ai.system_instructions"]) == [{"type": "text", "content": ""}]
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.request.model"] == "test-model"
    assert "gen_ai.provider.name" not in attrs
    assert "gen_ai.output.type" not in attrs
    assert "agenticrun.uid" not in attrs
    assert "agenticrun.phase" not in attrs


@pytest.mark.asyncio
async def test_run_agent_query_re_raises_tool_result_safety_failure(span_exporter) -> None:
    class SafetyFailureProvider(MockProvider):
        async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            raise ToolResultSafetyInspectionFailed()
            yield  # pragma: no cover

    with pytest.raises(ToolResultSafetyInspectionFailed):
        await run_agent_query(
            SafetyFailureProvider(),
            prompt="test",
            system_prompt="You are an AI agent.",
            output_schema=None,
            context=None,
            skills_dir="/workspace",
            model="test-model",
            max_turns=200,
            timeout_seconds=300,
        )

    invocation_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "invoke_agent"
    )
    assert invocation_span.attributes["error.type"] == "ToolResultSafetyInspectionFailed"
    assert invocation_span.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_run_agent_query_records_exception_type(span_exporter) -> None:
    class FailingProvider(MockProvider):
        async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            raise ValueError("provider failure")
            yield  # pragma: no cover

    result = await run_agent_query(
        FailingProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )

    assert result.output == {"success": False, "summary": "Agent error: provider failure"}
    invocation_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "invoke_agent"
    )
    assert invocation_span.attributes["error.type"] == "ValueError"
    assert invocation_span.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_run_agent_query_timeout_flushes_buffered_reasoning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Timeout emits buffered thinking instead of dropping it."""

    class BlockingProvider(MockProvider):
        async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            yield ThinkingDeltaEvent(thinking="BUFFERED_ON_TIMEOUT")
            await asyncio.sleep(30)
            yield ResultEvent(text='{"success":true,"summary":"never"}')  # pragma: no cover

    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    result = await run_agent_query(
        BlockingProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=1,
    )

    assert result.timed_out is True
    assert "[provider:run] thinking: BUFFERED_ON_TIMEOUT" in caplog.messages


@pytest.mark.asyncio
async def test_run_agent_query_exception_flushes_buffered_reasoning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Provider failure emits buffered thinking instead of dropping it."""

    class FailingProvider(MockProvider):
        async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            yield ThinkingDeltaEvent(thinking="BUFFERED_ON_ERROR")
            raise ValueError("provider failure")
            yield  # pragma: no cover

    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    result = await run_agent_query(
        FailingProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )

    assert result.output == {"success": False, "summary": "Agent error: provider failure"}
    assert "[provider:run] thinking: BUFFERED_ON_ERROR" in caplog.messages


@pytest.mark.asyncio
async def test_run_agent_query_cancellation_closes_spans_without_flush(
    span_exporter, caplog: pytest.LogCaptureFixture
) -> None:
    """Cancellation exports failures without flushing buffered legacy events."""
    ready = asyncio.Event()
    resume = asyncio.Event()

    class BlockingProvider(MockProvider):
        async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            yield ToolCallEvent(name="lookup", input='{"query":"safe"}', call_id="call-1")
            yield ThinkingDeltaEvent(thinking="UNFLUSHED_REASONING")
            yield TextDeltaEvent(text="UNFLUSHED_TEXT")
            ready.set()
            await resume.wait()
            yield ResultEvent(text='{"success":true,"summary":"fabricated"}')

    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    task = asyncio.create_task(
        run_agent_query(
            BlockingProvider(),
            prompt="test",
            system_prompt="You are an AI agent.",
            output_schema=None,
            context=None,
            skills_dir="/workspace",
            model="test-model",
            max_turns=200,
            timeout_seconds=300,
            audit_enabled=True,
            capture_content=True,
        )
    )

    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    spans = span_exporter.get_finished_spans()
    root_span = next(span for span in spans if span.name == "invoke_agent")
    root_attrs = dict(root_span.attributes)
    assert root_attrs["error.type"] == "CancelledError"
    assert root_span.status.status_code == StatusCode.ERROR
    assert root_span.status.description == "agent run failed"
    assert "gen_ai.output.messages" not in root_attrs
    assert not [event for event in root_span.events if event.name == "gen_ai.choice"]

    tool_span = next(span for span in spans if span.name == "execute_tool lookup")
    tool_attrs = dict(tool_span.attributes)
    assert tool_attrs["error.type"] == "missing_tool_result"
    assert "gen_ai.tool.call.result" not in tool_attrs
    assert tool_span.status.status_code == StatusCode.ERROR
    assert tool_span.status.description == "tool span not closed by result event"

    assert "UNFLUSHED_REASONING" not in caplog.text
    assert "UNFLUSHED_TEXT" not in caplog.text
    assert "CancelledError" not in caplog.text


@pytest.mark.asyncio
async def test_run_agent_query_deadline_during_inspection_is_safety_failure() -> None:
    blocked = asyncio.Event()

    async def inspect(_name: str, _result_type: str, _content: object, _call_id: str) -> None:
        await blocked.wait()

    class BlockedInspectorProvider(MockProvider):
        async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            middleware = ToolResultInspectionMiddleware(inspect)
            request = SimpleNamespace(
                messages=[ToolMessage(content="result", name="execute", tool_call_id="call-1")]
            )

            async def model_handler(_request: object) -> None:
                return None

            await middleware.awrap_model_call(request, model_handler)
            yield ResultEvent(text='{"success":true}')

    with pytest.raises(ToolResultSafetyInspectionFailed):
        await run_agent_query(
            BlockedInspectorProvider(),
            prompt="test",
            system_prompt="You are an AI agent.",
            output_schema=None,
            context=None,
            skills_dir="/workspace",
            model="test-model",
            max_turns=200,
            timeout_seconds=0.05,
        )


@pytest.mark.asyncio
async def test_run_agent_query_timeout(span_exporter) -> None:
    """Wall-clock timeout yields agent failure with a timed-out summary."""

    class SlowProvider(MockProvider):
        async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            await asyncio.sleep(2)
            async for event in super().query(options):
                yield event

    result = await run_agent_query(
        SlowProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=0.05,
    )
    assert result.output["success"] is False
    assert "timeout" in result.output["summary"].lower()
    assert result.timed_out is True
    invocation_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "invoke_agent"
    )
    assert invocation_span.attributes["error.type"] == "timeout"
    assert invocation_span.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_run_agent_query_empty_response(span_exporter) -> None:
    """Empty ResultEvent text is preserved and treated as agent failure."""
    result = await run_agent_query(
        MockProvider(events=[ResultEvent(text="")]),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is False
    assert result.output["summary"] == "Agent returned empty response"

    invocation_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "invoke_agent"
    )
    attrs = dict(invocation_span.attributes)
    empty_messages = [{"role": "assistant", "parts": [{"type": "text", "content": ""}]}]
    assert attrs["error.type"] == "empty_response"
    assert invocation_span.status.status_code == StatusCode.ERROR
    assert json.loads(attrs["gen_ai.output.messages"]) == empty_messages
    assert attrs["gen_ai.output.messages"] == json.dumps(
        empty_messages, ensure_ascii=False, separators=(",", ":")
    )


@pytest.mark.asyncio
async def test_run_agent_query_absent_result_event_omits_output(span_exporter) -> None:
    result = await run_agent_query(
        MockProvider(events=[TextDeltaEvent(text="partial but not terminal")]),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )

    assert result.output == {"success": False, "summary": "Agent returned empty response"}
    invocation_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "invoke_agent"
    )
    attrs = dict(invocation_span.attributes)
    assert attrs["error.type"] == "empty_response"
    assert invocation_span.status.status_code == StatusCode.ERROR
    assert "gen_ai.output.messages" not in attrs


@pytest.mark.parametrize(
    ("response", "expected_success"),
    [
        ('{"success": false, "summary": "not ready"}', False),
        ('{"success": 0, "summary": "not ready"}', 0),
        ('{"success": null, "summary": "not ready"}', None),
    ],
)
@pytest.mark.asyncio
async def test_run_agent_query_domain_outcomes_do_not_mark_trace_error(
    span_exporter, response: str, expected_success: bool | int | None
) -> None:
    result = await run_agent_query(
        MockProvider(events=[ResultEvent(text=response)]),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )

    assert result.output == {"success": expected_success, "summary": "not ready"}
    assert type(result.output["success"]) is type(expected_success)
    invocation_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "invoke_agent"
    )
    attrs = dict(invocation_span.attributes)
    assert invocation_span.status.status_code == StatusCode.UNSET
    assert "error.type" not in attrs


@pytest.mark.asyncio
async def test_run_agent_query_text_response() -> None:
    """Plain-text ResultEvent becomes summary when no JSON schema is required."""
    result = await run_agent_query(
        MockProvider(events=[ResultEvent(text="plain text answer")]),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is True
    assert result.output["summary"] == "plain text answer"


@pytest.mark.asyncio
async def test_run_agent_query_audit_enabled() -> None:
    """audit_enabled=True runs the audit path without changing agent outcome."""
    result = await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
        audit_enabled=True,
    )
    assert result.output["success"] is True


@pytest.mark.asyncio
async def test_deepagents_logs_redact_tool_payloads_but_audit_keeps_passed_content(
    span_exporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class DeepAgentsProvider(MockProvider):
        @property
        def name(self) -> str:
            return "deepagents"

    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    result = await run_agent_query(
        DeepAgentsProvider(
            events=[
                ToolCallEvent(name="execute", input="SECRET-TOOL-ARGUMENT", call_id="tool-1"),
                ToolResultEvent(output="COMPLETE-PASSED-TOOL-RESULT", call_id="tool-1"),
                ResultEvent(text='{"success":true,"summary":"done"}'),
            ]
        ),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
        audit_enabled=True,
        capture_content=True,
    )

    assert result.output["success"] is True
    assert "SECRET-TOOL-ARGUMENT" not in caplog.text
    assert "COMPLETE-PASSED-TOOL-RESULT" not in caplog.text
    tool_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "execute_tool execute"
    )
    assert json.loads(tool_span.attributes["gen_ai.tool.call.arguments"]) == {
        "content": "SECRET-TOOL-ARGUMENT"
    }
    assert json.loads(tool_span.attributes["gen_ai.tool.call.result"]) == {
        "content": "COMPLETE-PASSED-TOOL-RESULT"
    }


@pytest.mark.asyncio
async def test_run_agent_query_audit_with_tool_events() -> None:
    """Tool call/result events are consumed when audit logging is enabled."""
    events = [
        ToolCallEvent(name="bash", input="ls"),
        ToolResultEvent(output="file.txt"),
        ResultEvent(
            text='{"success": true, "summary": "done"}',
            input_tokens=10,
            output_tokens=5,
        ),
    ]
    result = await run_agent_query(
        MockProvider(events=events),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
        audit_enabled=True,
    )
    assert result.output["success"] is True


def test_format_context_envelope_markers_only() -> None:
    """Rule 12: block starts and ends with fixed marker lines."""
    text = format_context_prefix({})
    assert text == "[context]\n[/context]"


def test_format_context_unknown_keys_ignored() -> None:
    """Keys outside the supported context schema are omitted from the prefix."""
    text = format_context_prefix({"workflowPhase": "diagnose"})
    assert text == "[context]\n[/context]"


def test_format_context_target_namespaces() -> None:
    """Rule 13: comma-separated namespace list."""
    text = format_context_prefix({"targetNamespaces": ["default", "kube-system"]})
    assert "Target namespaces: default, kube-system" in text
    assert text.startswith("[context]")
    assert text.endswith("[/context]")


def test_format_context_target_namespaces_empty_list_omitted() -> None:
    """Empty targetNamespaces list produces no namespace line."""
    text = format_context_prefix({"targetNamespaces": []})
    assert "Target namespaces:" not in text


def test_format_context_attempt_includes_of_max_literal() -> None:
    """Rule 14: attempt line uses literal 'of max' placeholder."""
    text = format_context_prefix({"attempt": 2})
    assert "Attempt: 2 of max" in text


def test_format_context_attempt_zero_included() -> None:
    """Attempt zero is formatted like any other attempt number."""
    text = format_context_prefix({"attempt": 0})
    assert "Attempt: 0 of max" in text


def test_format_context_previous_attempts_with_failure_reason() -> None:
    """Previous attempts list failure reasons when present."""
    text = format_context_prefix(
        {
            "previousAttempts": [
                {"attempt": 1, "failureReason": "timeout"},
                {"attempt": 2},
            ],
        }
    )
    assert "  Attempt 1: timeout" in text
    assert "  Attempt 2" in text
    assert "  Attempt 2:" not in text


def test_format_context_previous_attempts_empty_list_omitted() -> None:
    """Empty previousAttempts list produces no attempts section."""
    text = format_context_prefix({"previousAttempts": []})
    assert "Previous attempts:" not in text


def test_format_context_approved_option_with_actions() -> None:
    """Approved option remediation actions are listed under Actions to execute."""
    text = format_context_prefix(
        {
            "approvedOption": {
                "title": "Restart pod",
                "diagnosis": {"rootCause": "CrashLoopBackOff"},
                "remediationPlan": {
                    "description": "Delete pod to trigger restart",
                    "reversible": True,
                    "actions": [
                        {
                            "type": "mutation",
                            "description": "Delete the crashing pod",
                        },
                    ],
                },
            },
        }
    )
    assert "Title: Restart pod" in text
    assert "  - [mutation] Delete the crashing pod" in text


def test_format_context_approved_option_with_command() -> None:
    """Action commands are included in the formatted remediation plan."""
    text = format_context_prefix(
        {
            "approvedOption": {
                "title": "Patch configmap",
                "diagnosis": {"rootCause": "wrong value"},
                "remediationPlan": {
                    "description": "Patch configmap data",
                    "actions": [
                        {
                            "type": "mutation",
                            "command": 'kubectl patch configmap foo -p \'{"data":{"k":"v"}}\'',
                            "description": "Apply patch",
                        },
                    ],
                },
            },
        }
    )
    assert "kubectl patch configmap foo" in text


def test_format_context_approved_option_without_actions() -> None:
    """Remediation plan without actions omits the Actions to execute section."""
    text = format_context_prefix(
        {
            "approvedOption": {
                "title": "Manual step",
                "diagnosis": {"rootCause": "needs human"},
                "remediationPlan": {"description": "Contact admin"},
            },
        }
    )
    assert "Title: Manual step" in text
    assert "Actions to execute:" not in text


def test_format_context_combined_fields() -> None:
    """All supported context fields appear together inside the envelope."""
    text = format_context_prefix(
        {
            "targetNamespaces": ["openshift-logging"],
            "attempt": 3,
            "previousAttempts": [{"attempt": 2, "failureReason": "denied"}],
            "approvedOption": {
                "title": "Fix RBAC",
                "diagnosis": {"rootCause": "missing role"},
                "remediationPlan": {
                    "description": "Apply RoleBinding",
                    "reversible": False,
                },
            },
        }
    )
    lines = text.splitlines()
    assert lines[0] == "[context]"
    assert lines[-1] == "[/context]"
    assert "Target namespaces: openshift-logging" in text
    assert "Attempt: 3 of max" in text
    assert "  Attempt 2: denied" in text
    assert "Title: Fix RBAC" in text


def test_format_context_approved_option_missing_diagnosis() -> None:
    """Missing approvedOption.diagnosis raises ContextFormatError."""
    with pytest.raises(ContextFormatError, match=r"approvedOption\.diagnosis"):
        format_context_prefix(
            {
                "approvedOption": {
                    "title": "Fix",
                    "remediationPlan": {"description": "plan"},
                },
            }
        )


def test_format_context_approved_option_missing_root_cause() -> None:
    """Missing approvedOption.diagnosis.rootCause raises ContextFormatError."""
    with pytest.raises(ContextFormatError, match=r"approvedOption\.diagnosis\.rootCause"):
        format_context_prefix(
            {
                "approvedOption": {
                    "title": "Fix",
                    "diagnosis": {},
                    "remediationPlan": {"description": "plan"},
                },
            }
        )


def test_format_context_previous_attempts_missing_attempt() -> None:
    """Previous attempt entry without attempt number raises ContextFormatError."""
    with pytest.raises(ContextFormatError, match="previousAttempts\\[0\\] missing attempt"):
        format_context_prefix({"previousAttempts": [{"failureReason": "timeout"}]})


@pytest.mark.asyncio
async def test_run_agent_query_invalid_context_returns_agent_failure() -> None:
    """Invalid context formatting returns agent failure instead of raising."""
    provider = MockProvider(events=[ResultEvent(text='{"success":true,"summary":"ok"}')])

    result = await run_agent_query(
        provider,
        prompt="run",
        system_prompt="sys",
        output_schema=None,
        context={"approvedOption": {"title": "only title"}},
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )

    assert result.output["success"] is False
    assert "Invalid context:" in result.output["summary"]
    assert "approvedOption.diagnosis" in result.output["summary"]


@pytest.mark.asyncio
async def test_run_agent_query_returns_token_counts() -> None:
    """Token counts from ResultEvent are included in the returned dict (OLS-3994)."""
    provider = MockProvider(
        events=[
            ResultEvent(
                text='{"success": true, "summary": "ok"}',
                input_tokens=500,
                output_tokens=200,
            )
        ]
    )
    result = await run_agent_query(
        provider,
        prompt="test",
        system_prompt="sys",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.input_tokens == 500
    assert result.output_tokens == 200


@pytest.mark.asyncio
async def test_run_agent_query_token_counts_zero_on_timeout() -> None:
    """Token counts default to 0 when the agent times out (OLS-3994)."""

    class SlowProvider(MockProvider):
        async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            await asyncio.sleep(10)
            yield ResultEvent(text="late")

    result = await run_agent_query(
        SlowProvider(),
        prompt="test",
        system_prompt="sys",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=1,
    )
    assert result.input_tokens == 0
    assert result.output_tokens == 0


@pytest.mark.asyncio
async def test_run_agent_query_token_counts_on_text_response() -> None:
    """Token counts present on plain text (non-JSON) responses (OLS-3994)."""
    provider = MockProvider(
        events=[ResultEvent(text="plain text", input_tokens=10, output_tokens=5)]
    )
    result = await run_agent_query(
        provider,
        prompt="test",
        system_prompt="sys",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.input_tokens == 10
    assert result.output_tokens == 5
