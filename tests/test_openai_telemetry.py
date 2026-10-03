"""OpenAI model and tool telemetry adapter tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest


class _FakeModel:
    def __init__(
        self,
        *,
        response: Any = None,
        stream_events: list[Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.response = response
        self.stream_events = stream_events or []
        self.error = error

    async def get_response(self, *_args: Any, **_kwargs: Any) -> Any:
        if self.error is not None:
            raise self.error
        return self.response

    def stream_response(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Any]:
        for event in self.stream_events:
            if isinstance(event, BaseException):
                raise event
            yield event


class _OutputSchema:
    def is_plain_text(self) -> bool:
        return False


def _function_tool(callback: Any) -> Any:
    from agents.tool import FunctionTool

    return FunctionTool(
        name="exec_command",
        description="Run a shell command.",
        params_json_schema={
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
        on_invoke_tool=callback,
        strict_json_schema=False,
    )


def _message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="message",
        role="assistant",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


@pytest.mark.asyncio
async def test_model_proxy_records_actual_request_and_response(audit_recorder: Any) -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    audit = audit_recorder
    input_items = [
        {"role": "user", "content": [{"type": "input_text", "text": "run pwd"}]},
        {"type": "function_call_output", "call_id": "call-old", "output": "done"},
    ]

    async def invoke(_context: Any, raw_arguments: str) -> str:
        return raw_arguments

    tool = _function_tool(invoke)
    response = SimpleNamespace(
        output=[
            _message("The command succeeded."),
            SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(type="summary_text", text="Inspecting the result.")],
            ),
            SimpleNamespace(
                type="function_call",
                call_id="call-new",
                name="exec_command",
                arguments='{"cmd":"pwd"}',
            ),
        ],
        usage=SimpleNamespace(
            requests=1,
            input_tokens=12,
            output_tokens=8,
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
    )
    delegate: Any = _FakeModel(response=response)
    proxy = create_model_proxy(delegate, audit, request_model="gpt-4.1-mini", native_responses=True)

    await proxy.get_response(
        "system prompt",
        input_items,
        object(),
        [tool],
        _OutputSchema(),
        [],
        object(),
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )

    _, start = audit.inference_starts[0]
    assert start["operation"] == "chat"
    assert start["model"] == "gpt-4.1-mini"
    assert start["system_instructions"] == [{"type": "text", "content": "system prompt"}]
    assert start["output_type"] == "json"
    assert start["input_messages"] == [
        {"role": "user", "parts": [{"type": "text", "content": "run pwd"}]},
        {
            "role": "tool",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": "call-old",
                    "response": "done",
                }
            ],
        },
    ]
    definition = start["tool_definitions"][0]
    assert definition["type"] == "function"
    assert definition["name"] == "exec_command"
    assert definition["parameters"] == tool.params_json_schema
    assert definition["strict"] is False

    _, ended = audit.inference_ends[0]
    assert ended["output_messages"] == [
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "The command succeeded."},
                {"type": "reasoning", "content": "Inspecting the result."},
                {
                    "type": "tool_call",
                    "id": "call-new",
                    "name": "exec_command",
                    "arguments": {"cmd": "pwd"},
                },
            ],
            "finish_reason": "unknown",
        }
    ]
    assert ended["response_model"] is None
    assert ended["input_tokens"] == 12
    assert ended["output_tokens"] == 8
    assert ended["reasoning_tokens"] == 3
    assert ended["finish_reasons"] == ["unknown"]


@pytest.mark.asyncio
async def test_chat_proxy_uses_normalized_response_without_fabricated_metadata(
    audit_recorder: Any,
) -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    audit = audit_recorder
    response = SimpleNamespace(
        output=[_message("done")],
        usage=SimpleNamespace(
            requests=1,
            input_tokens=4,
            output_tokens=2,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )
    proxy = create_model_proxy(
        _FakeModel(response=response),
        audit,
        request_model="offline-model",
        native_responses=False,
    )

    await proxy.get_response(None, "hello", object(), [], None, [], object())

    _, started = audit.inference_starts[0]
    assert started["operation"] == "chat"
    _, ended = audit.inference_ends[0]
    assert ended["output_messages"] == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "done"}],
            "finish_reason": "unknown",
        }
    ]
    assert ended["response_model"] is None
    assert ended["input_tokens"] == 4
    assert ended["output_tokens"] == 2
    assert ended["reasoning_tokens"] is None
    assert ended["finish_reasons"] == ["unknown"]


@pytest.mark.asyncio
async def test_stream_proxy_records_completed_response_before_yield(audit_recorder: Any) -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    audit = audit_recorder
    delta = SimpleNamespace(type="response.output_text.delta", delta="done")
    response = SimpleNamespace(
        model="gpt-4.1-2025-04-14",
        output=[_message("done")],
        usage=SimpleNamespace(
            input_tokens=4,
            output_tokens=0,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )
    completed = SimpleNamespace(type="response.completed", response=response)
    proxy = create_model_proxy(
        _FakeModel(stream_events=[delta, completed]),
        audit,
        request_model="gpt-4.1",
        native_responses=True,
    )
    stream = proxy.stream_response(None, "hello", object(), [], None, [], object())
    assert await anext(stream) is delta
    assert audit.inference_ends == []
    assert await anext(stream) is completed
    assert len(audit.inference_ends) == 1
    _, ended = audit.inference_ends[0]
    assert ended["response_model"] == "gpt-4.1-2025-04-14"
    assert ended["input_tokens"] == 4
    assert ended["output_tokens"] == 0
    assert ended["reasoning_tokens"] == 0
    assert ended["finish_reasons"] == ["unknown"]
    assert ended["output_messages"] == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "done"}],
            "finish_reason": "unknown",
        }
    ]
    assert "error" not in ended
    await stream.aclose()
    assert len(audit.inference_ends) == 1


@pytest.mark.asyncio
async def test_non_streaming_native_response_preserves_observed_zero_reasoning_tokens(
    span_exporter,
) -> None:
    from agents.usage import Usage
    from openai.types.responses.response_usage import OutputTokensDetails
    from opentelemetry.trace import StatusCode
    from prometheus_client import REGISTRY

    from lightspeed_agentic.audit import AuditLogger
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    model = "openai-nonstream-zero"
    usage = Usage(
        requests=1,
        input_tokens=4,
        output_tokens=2,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )
    assert "reasoning_tokens" in usage.output_tokens_details.model_fields_set
    input_labels = {
        "gen_ai_token_type": "input",
        "gen_ai_request_model": model,
        "gen_ai_provider_name": "openai",
        "gen_ai_operation_name": "chat",
    }
    output_labels = {**input_labels, "gen_ai_token_type": "output"}
    input_count = REGISTRY.get_sample_value("gen_ai_client_token_usage_count", input_labels) or 0
    output_count = REGISTRY.get_sample_value("gen_ai_client_token_usage_count", output_labels) or 0

    audit = AuditLogger(phase="analysis", model=model, provider="openai")
    proxy = create_model_proxy(
        _FakeModel(response=SimpleNamespace(output=[_message("done")], usage=usage)),
        audit,
        request_model=model,
        native_responses=True,
    )
    await proxy.get_response(None, "hello", object(), [], None, [], object())

    span = next(span for span in span_exporter.get_finished_spans() if span.name == f"chat {model}")
    attributes = dict(span.attributes)
    assert attributes["gen_ai.usage.input_tokens"] == 4
    assert attributes["gen_ai.usage.output_tokens"] == 2
    assert attributes["gen_ai.usage.reasoning.output_tokens"] == 0
    assert span.status.status_code == StatusCode.UNSET
    assert (
        REGISTRY.get_sample_value("gen_ai_client_token_usage_count", input_labels)
        == input_count + 1
    )
    assert (
        REGISTRY.get_sample_value("gen_ai_client_token_usage_count", output_labels)
        == output_count + 1
    )


@pytest.mark.asyncio
async def test_non_streaming_native_response_omits_default_usage_and_token_metrics(span_exporter):
    from agents.usage import Usage
    from opentelemetry.trace import StatusCode
    from prometheus_client import REGISTRY

    from lightspeed_agentic.audit import AuditLogger
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    model = "openai-nonstream-missing-usage"
    usage = Usage()
    input_labels = {
        "gen_ai_token_type": "input",
        "gen_ai_request_model": model,
        "gen_ai_provider_name": "openai",
        "gen_ai_operation_name": "chat",
    }
    output_labels = {**input_labels, "gen_ai_token_type": "output"}
    input_count = REGISTRY.get_sample_value("gen_ai_client_token_usage_count", input_labels)
    output_count = REGISTRY.get_sample_value("gen_ai_client_token_usage_count", output_labels)

    audit = AuditLogger(phase="analysis", model=model, provider="openai")
    proxy = create_model_proxy(
        _FakeModel(response=SimpleNamespace(output=[_message("done")], usage=usage)),
        audit,
        request_model=model,
        native_responses=True,
    )
    await proxy.get_response(None, "hello", object(), [], None, [], object())

    span = next(span for span in span_exporter.get_finished_spans() if span.name == f"chat {model}")
    attributes = dict(span.attributes)
    assert "gen_ai.response.model" not in attributes
    assert "gen_ai.output.messages" in attributes
    assert "gen_ai.usage.input_tokens" not in attributes
    assert "gen_ai.usage.output_tokens" not in attributes
    assert "gen_ai.usage.reasoning.output_tokens" not in attributes
    assert span.status.status_code == StatusCode.UNSET
    assert REGISTRY.get_sample_value("gen_ai_client_token_usage_count", input_labels) == input_count
    assert (
        REGISTRY.get_sample_value("gen_ai_client_token_usage_count", output_labels) == output_count
    )


@pytest.mark.asyncio
async def test_model_proxy_records_non_streaming_error_without_response_data(
    audit_recorder: Any,
) -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    audit = audit_recorder
    error = RuntimeError("request failed")
    delegate: Any = _FakeModel(error=error)
    proxy = create_model_proxy(delegate, audit, request_model="gpt-4.1", native_responses=False)

    with pytest.raises(RuntimeError):
        await proxy.get_response(None, "hello", object(), [], None, [], object())
    _, ended = audit.inference_ends[0]
    assert isinstance(ended["error"], RuntimeError)
    assert "output_messages" not in ended
    assert "input_tokens" not in ended


@pytest.mark.asyncio
async def test_stream_proxy_records_terminal_provider_failure_before_yield(span_exporter) -> None:
    from opentelemetry.trace import StatusCode

    from lightspeed_agentic.audit import AuditLogger
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    request_model = "openai-stream-failed"
    audit = AuditLogger(phase="analysis", model=request_model, provider="openai")
    failed = SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(
            model="observed-model",
            output=[_message("partial response")],
            usage=SimpleNamespace(
                input_tokens=4,
                output_tokens=0,
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        ),
    )
    proxy = create_model_proxy(
        _FakeModel(stream_events=[failed]),
        audit,
        request_model=request_model,
        native_responses=True,
    )
    stream = proxy.stream_response(None, "hello", object(), [], None, [], object())

    await stream.__anext__()
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes["error.type"] == "response.failed"
    attributes = dict(spans[0].attributes)
    assert attributes["gen_ai.response.model"] == "observed-model"
    assert attributes["gen_ai.usage.input_tokens"] == 4
    assert attributes["gen_ai.usage.output_tokens"] == 0
    assert attributes["gen_ai.usage.reasoning.output_tokens"] == 0
    assert json.loads(attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "partial response"}],
            "finish_reason": "unknown",
        }
    ]
    await stream.aclose()
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_tool_hooks_record_actual_arguments_result_and_failures_once(
    audit_recorder: Any,
) -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    audit = audit_recorder
    hooks = create_tool_hooks(audit)
    raw_arguments = '{"cmd":"pwd"}'
    context = SimpleNamespace(
        tool_name="exec_command",
        tool_call_id="provider-call-1",
        tool_arguments=raw_arguments,
    )

    async def succeed(_context: Any, _arguments: str) -> dict[str, int]:
        return {"exit_code": 0}

    tool = _function_tool(succeed)
    await hooks.on_tool_start(context, object(), tool)
    result = await tool.on_invoke_tool(context, raw_arguments)
    await hooks.on_tool_end(context, object(), tool, result)

    _, started = audit.tool_starts[0]
    assert started["name"] == "exec_command"
    assert started["call_id"] == "provider-call-1"
    assert started["arguments"] == {"cmd": "pwd"}
    _, ended = audit.tool_ends[0]
    assert ended["result"] == {"exit_code": 0}
    assert ended.get("error") is None

    class _ToolFailureError(Exception):
        pass

    async def fail(_context: Any, _arguments: str) -> str:
        raise _ToolFailureError("sensitive tool failure")

    failing_tool = _function_tool(fail)
    failure_context = SimpleNamespace(
        tool_name="exec_command",
        tool_call_id="provider-call-2",
        tool_arguments=raw_arguments,
    )
    await hooks.on_tool_start(failure_context, object(), failing_tool)
    failing_invoke = failing_tool.on_invoke_tool
    with pytest.raises(_ToolFailureError):
        await failing_invoke(failure_context, raw_arguments)

    await hooks.on_tool_end(failure_context, object(), failing_tool, "not a successful result")
    assert len(audit.tool_ends) == 2
    _, failed = audit.tool_ends[1]
    assert isinstance(failed["error"], _ToolFailureError)
    assert failed["result"] is None
    hooks.close()


@pytest.mark.asyncio
async def test_overlapping_tool_failure_cannot_finish_its_sibling(audit_recorder: Any) -> None:
    from agents import function_tool

    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    audit = audit_recorder
    hooks = create_tool_hooks(audit)
    sibling_started = asyncio.Event()
    release_sibling = asyncio.Event()
    calls = 0

    @function_tool(
        name_override="exec_command",
        failure_error_function=lambda _context, _error: "handled failure",
    )
    async def execute(cmd: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            await sibling_started.wait()
            raise RuntimeError("native execution failed")
        sibling_started.set()
        await release_sibling.wait()
        return f"completed {cmd}"

    raw_arguments = '{"cmd":"pwd"}'
    first = SimpleNamespace(
        tool_name="exec_command", tool_call_id="A", tool_arguments=raw_arguments, run_config=None
    )
    sibling = SimpleNamespace(
        tool_name="exec_command", tool_call_id="B", tool_arguments=raw_arguments, run_config=None
    )
    await hooks.on_tool_start(first, object(), execute)
    await hooks.on_tool_start(sibling, object(), execute)
    first_task = asyncio.create_task(execute.on_invoke_tool(first, raw_arguments))
    sibling_task = asyncio.create_task(execute.on_invoke_tool(sibling, raw_arguments))
    try:
        first_result = await asyncio.wait_for(first_task, timeout=5)
        assert first_result == "handled failure"
        first_span, first_end = audit.tool_ends[0]
        assert first_span is audit.tool_starts[0][0]
        assert isinstance(first_end["error"], RuntimeError)
        assert first_end["result"] is None

        await hooks.on_tool_end(first, object(), execute, first_result)
        assert len(audit.tool_ends) == 1

        release_sibling.set()
        sibling_result = await asyncio.wait_for(sibling_task, timeout=5)
        await hooks.on_tool_end(sibling, object(), execute, sibling_result)
        sibling_span, sibling_end = audit.tool_ends[1]
        assert sibling_span is audit.tool_starts[1][0]
        assert sibling_end["result"] == "completed pwd"
        assert sibling_end["error"] is None
        assert sibling_end["end_time"] >= first_end["end_time"]
    finally:
        release_sibling.set()
        for task in (first_task, sibling_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(first_task, sibling_task, return_exceptions=True)
        hooks.close()


@pytest.mark.asyncio
async def test_duplicate_call_ids_do_not_select_an_arbitrary_pending_span(
    audit_recorder: Any,
) -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    audit = audit_recorder
    hooks = create_tool_hooks(audit)
    tool = SimpleNamespace(name="exec_command")
    contexts = [
        SimpleNamespace(tool_call_id="duplicate", tool_arguments='{"cmd":"pwd"}') for _ in range(2)
    ]
    for context in contexts:
        await hooks.on_tool_start(context, object(), tool)
    try:
        await hooks.on_tool_end(contexts[0], object(), tool, "ambiguous result")
        assert audit.tool_ends == []
    finally:
        hooks.close()


@pytest.mark.asyncio
async def test_registered_skill_load_uses_native_tool_hooks(audit_recorder: Any) -> None:
    from pathlib import Path

    from agents.sandbox.capabilities.skills import (
        LocalDirLazySkillSource,
        Skills,
    )
    from agents.sandbox.entries import LocalDir

    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    class _Skills(Skills):
        async def load_skill(self, skill_name: str) -> dict[str, str]:
            return {"status": "loaded", "skill_name": skill_name}

    skills = _Skills(lazy_from=LocalDirLazySkillSource(source=LocalDir(src=Path.cwd())))
    object.__setattr__(skills, "session", object())
    tool: Any = skills.tools()[0]
    raw_arguments = '{"skill_name":"guide"}'
    context = SimpleNamespace(
        tool_name="load_skill",
        tool_call_id="skill-call",
        tool_arguments=raw_arguments,
    )

    audit = audit_recorder
    hooks = create_tool_hooks(audit)
    await hooks.on_tool_start(context, object(), tool)
    result = await tool.on_invoke_tool(context, raw_arguments)
    await hooks.on_tool_end(context, object(), tool, result)

    _, started = audit.tool_starts[0]
    assert started["name"] == "load_skill"
    assert started["call_id"] == "skill-call"
    assert started["arguments"] == {"skill_name": "guide"}
    _, ended = audit.tool_ends[0]
    assert ended["result"] == {"status": "loaded", "skill_name": "guide"}
    assert ended.get("error") is None
    hooks.close()
