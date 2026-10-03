"""Tests for Gemini's native ADK telemetry callbacks."""

from __future__ import annotations

import json
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry._logs import NoOpLogger
from opentelemetry.trace import StatusCode
from prometheus_client import REGISTRY

import lightspeed_agentic.providers.gemini_telemetry as gemini_telemetry
from lightspeed_agentic.audit import AuditLogger
from lightspeed_agentic.providers.gemini import _trim_tool_response
from lightspeed_agentic.providers.gemini_telemetry import (
    GeminiTelemetry,
    disable_adk_native_telemetry,
)
from lightspeed_agentic.types import MAX_TOOL_RETURN_CHARS, TOOL_RETURN_PREVIEW_CHARS


class _FinishReason(Enum):
    STOP = "STOP"


def test_model_callbacks_record_standard_request_and_terminal_response(audit_recorder: Any) -> None:
    recorder = audit_recorder
    telemetry = GeminiTelemetry(recorder, requested_model="configured-model")
    callback_context = object()
    request = SimpleNamespace(
        model="gemini-requested",
        contents=[
            SimpleNamespace(
                role="user",
                parts=[
                    SimpleNamespace(text="question", thought=False),
                ],
            ),
            SimpleNamespace(
                role="model",
                parts=[
                    SimpleNamespace(text="thinking", thought=True),
                    SimpleNamespace(
                        function_call=SimpleNamespace(
                            id="input-call",
                            name="read_file",
                            args={"path": "/workspace/input"},
                        )
                    ),
                ],
            ),
            SimpleNamespace(
                role="user",
                parts=[
                    SimpleNamespace(
                        function_response=SimpleNamespace(
                            id=None, response={"text": "file content"}
                        )
                    )
                ],
            ),
        ],
        config=SimpleNamespace(
            system_instruction=SimpleNamespace(
                parts=[SimpleNamespace(text="system policy", thought=False)]
            ),
            tools=[
                SimpleNamespace(
                    function_declarations=[
                        SimpleNamespace(
                            name="read_file",
                            description="Read a file",
                            parameters={
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        )
                    ]
                )
            ],
            response_mime_type="application/json",
            response_schema={},
        ),
    )

    telemetry.before_model_callback(callback_context, request)
    span, start = recorder.inference_starts[0]

    assert start["model"] == "gemini-requested"
    assert start["operation"] == "generate_content"
    assert isinstance(start["start_time"], int)
    assert start["system_instructions"] == [{"type": "text", "content": "system policy"}]
    assert start["output_type"] == "json"
    assert start["tool_definitions"] == [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }
    ]
    assert start["input_messages"] == [
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": "question"},
            ],
        },
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "thinking"},
                {
                    "type": "tool_call",
                    "id": "input-call",
                    "name": "read_file",
                    "arguments": {"path": "/workspace/input"},
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "type": "tool_call_response",
                    "response": {"text": "file content"},
                }
            ],
        },
    ]

    telemetry.after_model_callback(
        callback_context,
        SimpleNamespace(
            partial=True,
            model_version="gemini-partial",
            usage_metadata=SimpleNamespace(
                prompt_token_count=0,
                candidates_token_count=1,
                thoughts_token_count=0,
            ),
            finish_reason=None,
            content=SimpleNamespace(
                role="model", parts=[SimpleNamespace(text="partial", thought=False)]
            ),
        ),
    )
    assert recorder.inference_ends == []

    telemetry.after_model_callback(
        callback_context,
        SimpleNamespace(
            partial=False,
            model_version="gemini-actual",
            usage_metadata=SimpleNamespace(
                prompt_token_count=0,
                candidates_token_count=5,
                thoughts_token_count=2,
            ),
            finish_reason=_FinishReason.STOP,
            content=SimpleNamespace(
                role="model",
                parts=[
                    SimpleNamespace(text="final", thought=False),
                    SimpleNamespace(text="reasoning", thought=True),
                    SimpleNamespace(
                        function_call=SimpleNamespace(
                            id="output-call",
                            name="execute_bash",
                            args={"command": "printf ok"},
                        )
                    ),
                ],
            ),
        ),
    )

    ended_span, end = recorder.inference_ends[0]
    assert ended_span is span
    assert end["output_messages"] == [
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "final"},
                {"type": "reasoning", "content": "reasoning"},
                {
                    "type": "tool_call",
                    "id": "output-call",
                    "name": "execute_bash",
                    "arguments": {"command": "printf ok"},
                },
            ],
            "finish_reason": "stop",
        }
    ]
    assert end["response_model"] == "gemini-actual"
    assert end["input_tokens"] == 0
    assert end["output_tokens"] == 7
    assert end["reasoning_tokens"] == 2
    assert end["finish_reasons"] == ["stop"]
    assert isinstance(end["end_time"], int)


def test_terminal_response_uses_buffered_parts_and_unknown_finish_reason(
    audit_recorder: Any,
) -> None:
    recorder = audit_recorder
    telemetry = GeminiTelemetry(recorder, requested_model="requested-model")
    context = object()

    telemetry.before_model_callback(
        context,
        SimpleNamespace(model=None, contents=[], config=SimpleNamespace()),
    )
    telemetry.after_model_callback(
        context,
        SimpleNamespace(
            partial=True,
            content=SimpleNamespace(
                role="model", parts=[SimpleNamespace(text="buffered", thought=False)]
            ),
        ),
    )
    telemetry.after_model_callback(
        context,
        SimpleNamespace(
            partial=False,
            content=None,
            model_version=None,
            usage_metadata=None,
            finish_reason=None,
        ),
    )

    _, end = recorder.inference_ends[0]
    assert end["output_messages"] == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "buffered"}],
            "finish_reason": "unknown",
        }
    ]
    assert end["response_model"] is None
    assert end["input_tokens"] is None
    assert end["output_tokens"] is None
    assert end["reasoning_tokens"] is None
    assert end["finish_reasons"] == ["unknown"]


def test_candidate_only_usage_records_output_without_inventing_reasoning(
    span_exporter: Any,
) -> None:
    model = "gemini-candidate-only"
    labels = {
        "gen_ai_token_type": "output",
        "gen_ai_request_model": model,
        "gen_ai_provider_name": "gcp.gemini",
        "gen_ai_operation_name": "generate_content",
    }
    before = REGISTRY.get_sample_value("gen_ai_client_token_usage_sum", labels) or 0
    audit = AuditLogger(phase="analysis", model=model, provider="gcp.gemini")
    telemetry = GeminiTelemetry(audit, requested_model=model)
    context = object()
    telemetry.before_model_callback(context, SimpleNamespace(model=model, contents=[], config=None))
    telemetry.after_model_callback(
        context,
        SimpleNamespace(
            partial=False,
            usage_metadata=SimpleNamespace(candidates_token_count=5, thoughts_token_count=None),
            content=SimpleNamespace(
                role="model", parts=[SimpleNamespace(text="done", thought=False)]
            ),
        ),
    )
    telemetry.close()

    span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == f"generate_content {model}"
    )
    assert span.attributes["gen_ai.usage.output_tokens"] == 5
    assert "gen_ai.usage.reasoning.output_tokens" not in span.attributes
    assert REGISTRY.get_sample_value("gen_ai_client_token_usage_sum", labels) == before + 5


def test_model_error_records_only_observed_usage_and_partial_text(audit_recorder: Any) -> None:
    recorder = audit_recorder
    telemetry = GeminiTelemetry(recorder, requested_model="requested-model")
    context = object()
    request = SimpleNamespace(model="requested-model", contents=[], config=None)
    error = RuntimeError("provider failed")

    telemetry.before_model_callback(context, request)
    telemetry.after_model_callback(
        context,
        SimpleNamespace(
            partial=True,
            model_version=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=None,
                candidates_token_count=0,
                thoughts_token_count=0,
            ),
            finish_reason=None,
            content=SimpleNamespace(
                role="model", parts=[SimpleNamespace(text="partial", thought=False)]
            ),
        ),
    )
    telemetry.on_model_error_callback(context, request, error)

    _, end = recorder.inference_ends[0]
    assert end["error"] is error
    assert end["output_messages"][0]["parts"] == [{"type": "text", "content": "partial"}]
    assert end["output_messages"][0]["finish_reason"] == "unknown"
    assert end["response_model"] is None
    assert end["input_tokens"] is None
    assert end["output_tokens"] == 0
    assert end["reasoning_tokens"] == 0


def test_tool_callbacks_record_raw_result_before_trim_and_omit_failed_result(
    audit_recorder: Any,
) -> None:
    recorder = audit_recorder
    telemetry = GeminiTelemetry(recorder, requested_model="model")
    tool = SimpleNamespace(name="execute_bash")
    args = {"command": "printf output"}
    context = SimpleNamespace(function_call_id="provider-call-id")
    raw_result = "x" * (MAX_TOOL_RETURN_CHARS + 1)

    telemetry.before_tool_callback(tool, args, context)
    _, start = recorder.tool_starts[0]
    assert start["name"] == "execute_bash"
    assert start["call_id"] == "provider-call-id"
    assert start["arguments"] is args
    assert start["tool_type"] == "function"
    assert isinstance(start["start_time"], int)

    callbacks = [telemetry.after_tool_callback, _trim_tool_response]
    model_result = raw_result
    for callback in callbacks:
        replacement = callback(tool, args, context, raw_result)
        if replacement is not None:
            model_result = replacement
            break

    _, success = recorder.tool_ends[0]
    assert success["result"] is raw_result
    assert "error" not in success
    assert model_result["status"] == "truncated"
    assert len(model_result["preview"]) == TOOL_RETURN_PREVIEW_CHARS

    failed_context = SimpleNamespace(function_call_id="failed-call")
    error = RuntimeError("tool failed")
    telemetry.before_tool_callback(tool, args, failed_context)
    telemetry.on_tool_error_callback(tool, args, failed_context, error)

    _, failure = recorder.tool_ends[1]
    assert failure["error"] is error
    assert "result" not in failure
    assert isinstance(failure["end_time"], int)


def test_native_adk_tracing_suppression_is_module_scoped(monkeypatch: Any) -> None:
    original_tracer = object()
    module_names = (
        "google.adk.telemetry.tracing",
        *gemini_telemetry._ADK_TRACER_ALIAS_MODULES,
    )
    modules = {name: SimpleNamespace(tracer=original_tracer) for name in module_names}
    modules["google.adk.telemetry.tracing"].otel_logger = object()
    monkeypatch.setattr(gemini_telemetry.importlib, "import_module", modules.__getitem__)
    provider = trace.get_tracer_provider()

    disable_adk_native_telemetry()

    assert isinstance(modules["google.adk.telemetry.tracing"].tracer, trace.NoOpTracer)
    assert isinstance(modules["google.adk.telemetry.tracing"].otel_logger, NoOpLogger)
    for name in gemini_telemetry._ADK_TRACER_ALIAS_MODULES:
        assert modules[name].tracer is modules["google.adk.telemetry.tracing"].tracer
    assert trace.get_tracer_provider() is provider


def test_pending_response_ignores_partial_and_wrong_model_events(audit_recorder: Any) -> None:
    recorder = audit_recorder
    telemetry = GeminiTelemetry(recorder, requested_model="requested-model")
    context = SimpleNamespace(agent_name="lightspeed", invocation_id="run-1")
    request = SimpleNamespace(model="requested-model", contents=[], config=SimpleNamespace())
    telemetry.before_model_callback(context, request)
    telemetry.after_model_callback(
        context,
        SimpleNamespace(
            partial=False,
            model_version="observed-model",
            usage_metadata=SimpleNamespace(
                prompt_token_count=1,
                candidates_token_count=2,
                thoughts_token_count=3,
            ),
            finish_reason=_FinishReason.STOP,
            content=SimpleNamespace(
                role="model",
                parts=[
                    SimpleNamespace(
                        function_call=SimpleNamespace(
                            id=None, name="execute_probe", args={"value": "original"}
                        )
                    )
                ],
            ),
        ),
    )
    assert recorder.inference_ends == []

    finalized_content = SimpleNamespace(
        role="model",
        parts=[
            SimpleNamespace(
                function_call=SimpleNamespace(
                    id="adk-finalized-id",
                    name="execute_probe",
                    args={"value": "original"},
                )
            )
        ],
    )
    unrelated_events = (
        SimpleNamespace(
            partial=True,
            author="lightspeed",
            invocation_id="run-1",
            content=finalized_content,
        ),
        SimpleNamespace(
            partial=False,
            author="subagent",
            invocation_id="run-1",
            content=finalized_content,
        ),
        SimpleNamespace(
            partial=False,
            author="lightspeed",
            invocation_id="other-run",
            content=finalized_content,
        ),
        SimpleNamespace(
            partial=False,
            author="lightspeed",
            invocation_id="run-1",
            content=SimpleNamespace(role="user", parts=finalized_content.parts),
        ),
    )
    for event in unrelated_events:
        telemetry.observe_model_event(event)
        assert recorder.inference_ends == []

    telemetry.observe_model_event(
        SimpleNamespace(
            partial=False,
            author="lightspeed",
            invocation_id="run-1",
            content=finalized_content,
        )
    )

    assert len(recorder.inference_ends) == 1
    _, end = recorder.inference_ends[0]
    output = end["output_messages"][0]
    call_part = next(part for part in output["parts"] if part["type"] == "tool_call")
    assert call_part["id"] == "adk-finalized-id"
    assert end["response_model"] == "observed-model"
    assert end["input_tokens"] == 1
    assert end["output_tokens"] == 5
    assert end["reasoning_tokens"] == 3


def test_close_flushes_completed_pending_response_at_native_end_time(
    span_exporter: Any, monkeypatch: Any
) -> None:
    clock = iter((1_000_000_000, 1_000_000_200, 1_000_000_300))
    monkeypatch.setattr(
        gemini_telemetry.time,
        "time_ns",
        lambda: next(clock, 1_000_000_400),
    )
    model_name = "gemini-pending-close-test"
    audit = AuditLogger(phase="analysis", model=model_name, provider="gcp.gemini")
    telemetry = GeminiTelemetry(audit, requested_model=model_name)
    context = SimpleNamespace(agent_name="lightspeed", invocation_id="run-close")
    telemetry.before_model_callback(
        context,
        SimpleNamespace(model=model_name, contents=[], config=SimpleNamespace()),
    )
    telemetry.after_model_callback(
        context,
        SimpleNamespace(
            partial=False,
            model_version="observed-model",
            usage_metadata=SimpleNamespace(
                prompt_token_count=1,
                candidates_token_count=2,
                thoughts_token_count=3,
            ),
            finish_reason=_FinishReason.STOP,
            content=SimpleNamespace(
                role="model",
                parts=[
                    SimpleNamespace(
                        function_call=SimpleNamespace(id=None, name="execute_probe", args={})
                    )
                ],
            ),
        ),
    )

    telemetry.close(RuntimeError("later query failure"))

    span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == f"generate_content {model_name}"
    )
    output = json.loads(span.attributes["gen_ai.output.messages"])[0]
    call_part = next(part for part in output["parts"] if part["type"] == "tool_call")
    assert span.end_time == 1_000_000_200
    assert span.status.status_code == StatusCode.UNSET
    assert "error.type" not in span.attributes
    assert "id" not in call_part
    assert span.attributes["gen_ai.response.model"] == "observed-model"
    assert span.attributes["gen_ai.usage.input_tokens"] == 1
    assert span.attributes["gen_ai.usage.output_tokens"] == 5
    assert span.attributes["gen_ai.usage.reasoning.output_tokens"] == 3


def test_close_marks_active_model_and_tool_operations_as_errors(span_exporter: Any) -> None:
    model_name = "gemini-active-close-test"
    audit = AuditLogger(phase="analysis", model=model_name, provider="gcp.gemini")
    telemetry = GeminiTelemetry(audit, requested_model=model_name)
    error = RuntimeError("query interrupted")
    telemetry.before_model_callback(
        object(),
        SimpleNamespace(model=model_name, contents=[], config=SimpleNamespace()),
    )
    telemetry.close(error)

    tool = SimpleNamespace(name="execute_probe")
    tool_context = SimpleNamespace(function_call_id="tool-call-id")
    telemetry.before_tool_callback(tool, {}, tool_context)
    telemetry.close(error)

    spans = span_exporter.get_finished_spans()
    model_span = next(span for span in spans if span.name == f"generate_content {model_name}")
    tool_span = next(span for span in spans if span.name == "execute_tool execute_probe")
    assert model_span.status.status_code == StatusCode.ERROR
    assert model_span.attributes["error.type"] == "RuntimeError"
    assert tool_span.status.status_code == StatusCode.ERROR
    assert tool_span.attributes["error.type"] == "RuntimeError"
    assert "gen_ai.tool.call.result" not in tool_span.attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_call_id", [None, "provider-call-id"])
async def test_finalized_adk_ids_match_execution_and_effective_request(
    span_exporter: Any, provider_call_id: str | None
) -> None:
    pytest.importorskip("google.adk.models.base_llm")
    from google.adk.agents import Agent, RunConfig
    from google.adk.agents.run_config import StreamingMode
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools.function_tool import FunctionTool
    from google.genai import types

    requests: list[Any] = []
    events: list[Any] = []
    model_name = "gemini-native-join-test"

    class OfflineModel(BaseLlm):
        async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
            del stream
            requests.append(llm_request)
            if len(requests) == 1:
                content = types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id=provider_call_id, name="execute_probe", args={}
                            )
                        )
                    ],
                )
            else:
                content = types.Content(role="model", parts=[types.Part(text="done")])
            yield LlmResponse(content=content, finish_reason=types.FinishReason.STOP)

    def execute_probe() -> str:
        return "native tool result"

    audit = AuditLogger(phase="analysis", model=model_name, provider="gcp.gemini")
    telemetry = GeminiTelemetry(audit, requested_model=model_name)
    agent = Agent(
        name="lightspeed",
        model=OfflineModel(model=model_name),
        instruction="Use execute_probe.",
        tools=[FunctionTool(func=execute_probe)],
        before_model_callback=telemetry.before_model_callback,
        after_model_callback=telemetry.after_model_callback,
        on_model_error_callback=telemetry.on_model_error_callback,
        before_tool_callback=telemetry.before_tool_callback,
        after_tool_callback=telemetry.after_tool_callback,
        on_tool_error_callback=telemetry.on_tool_error_callback,
    )
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name="native-join", user_id="test-user")
    runner = Runner(app_name="native-join", agent=agent, session_service=sessions)
    disable_adk_native_telemetry()
    with trace.get_tracer("native-join").start_as_current_span("invoke_agent lightspeed") as root:
        audit.set_parent_context(trace.set_span_in_context(root))
        try:
            async for event in runner.run_async(
                user_id="test-user",
                session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text="Run the tool.")]),
                run_config=RunConfig(
                    max_llm_calls=2,
                    streaming_mode=StreamingMode.NONE,
                ),
            ):
                telemetry.observe_model_event(event)
                events.append(event)
        finally:
            telemetry.close()

    finalized_call = next(
        event.get_function_calls()[0] for event in events if event.get_function_calls()
    )
    assert finalized_call.id
    if provider_call_id is not None:
        assert finalized_call.id == provider_call_id
    spans = span_exporter.get_finished_spans()
    model_spans = sorted(
        (span for span in spans if span.name == f"generate_content {model_name}"),
        key=lambda span: span.start_time,
    )
    assert len(model_spans) == 2
    tool_span = next(span for span in spans if span.name == "execute_tool execute_probe")
    model_output = json.loads(model_spans[0].attributes["gen_ai.output.messages"])
    request_part = next(part for part in model_output[0]["parts"] if part["type"] == "tool_call")
    assert request_part["id"] == tool_span.attributes["gen_ai.tool.call.id"] == finalized_call.id
    next_input = json.loads(model_spans[1].attributes["gen_ai.input.messages"])
    response_part = next(
        part
        for message in next_input
        for part in message["parts"]
        if part["type"] == "tool_call_response"
    )
    effective_response = next(
        part.function_response
        for content in requests[1].contents
        for part in (content.parts or [])
        if part.function_response is not None
    )
    assert response_part.get("id") == effective_response.id
    if provider_call_id is not None:
        assert response_part["id"] == finalized_call.id
    else:
        assert "id" not in response_part


@pytest.mark.asyncio
async def test_adk_call_limit_does_not_record_rejected_model_request(
    span_exporter: Any, monkeypatch: Any
) -> None:
    pytest.importorskip("google.adk.models.base_llm")
    from google.adk.agents import Agent, RunConfig
    from google.adk.agents.invocation_context import LlmCallsLimitExceededError
    from google.adk.agents.run_config import StreamingMode
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools.function_tool import FunctionTool
    from google.genai import types

    model_name = "gemini-max-llm-calls-test"
    tool_result = "first tool result"
    model_requests: list[Any] = []
    tool_calls: list[bool] = []

    class OfflineModel(BaseLlm):
        async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
            del stream
            model_requests.append(llm_request)
            if len(model_requests) == 1:
                yield LlmResponse(
                    model_version="observed-offline-model",
                    finish_reason=types.FinishReason.STOP,
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(
                                    id="native-call-id",
                                    name="execute_probe",
                                    args={},
                                )
                            )
                        ],
                    ),
                )
            else:
                yield LlmResponse(
                    content=types.Content(
                        role="model", parts=[types.Part(text="unexpected request")]
                    )
                )

    def execute_probe() -> str:
        tool_calls.append(True)
        return tool_result

    audit_logger = AuditLogger(
        phase="analysis",
        model=model_name,
        provider="gcp.gemini",
    )
    inference_starts: list[dict[str, Any]] = []
    start_inference = audit_logger.start_inference

    def capture_inference_start(**kwargs: Any) -> Any:
        inference_starts.append(kwargs)
        return start_inference(**kwargs)

    monkeypatch.setattr(audit_logger, "start_inference", capture_inference_start)
    telemetry = GeminiTelemetry(audit_logger, requested_model=model_name)
    disable_adk_native_telemetry()

    agent = Agent(
        name="lightspeed",
        model=OfflineModel(model=model_name),
        instruction="Use execute_probe.",
        tools=[FunctionTool(func=execute_probe)],
        before_model_callback=telemetry.before_model_callback,
        after_model_callback=telemetry.after_model_callback,
        on_model_error_callback=telemetry.on_model_error_callback,
        before_tool_callback=telemetry.before_tool_callback,
        after_tool_callback=telemetry.after_tool_callback,
        on_tool_error_callback=telemetry.on_tool_error_callback,
    )
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="gemini-call-limit-test",
        user_id="test-user",
    )
    runner = Runner(
        app_name="gemini-call-limit-test",
        agent=agent,
        session_service=session_service,
    )

    duration_labels = {
        "gen_ai_request_model": model_name,
        "gen_ai_provider_name": "gcp.gemini",
        "gen_ai_operation_name": "generate_content",
        "error_type": "",
    }
    error_duration_labels = {
        **duration_labels,
        "error_type": "LlmCallsLimitExceededError",
    }
    duration_count = (
        REGISTRY.get_sample_value("gen_ai_client_operation_duration_seconds_count", duration_labels)
        or 0.0
    )
    error_duration_count = (
        REGISTRY.get_sample_value(
            "gen_ai_client_operation_duration_seconds_count", error_duration_labels
        )
        or 0.0
    )

    with trace.get_tracer("gemini-call-limit-test").start_as_current_span(
        "invoke_agent lightspeed"
    ) as agent_span:
        agent_span_id = agent_span.get_span_context().span_id
        audit_logger.set_parent_context(trace.set_span_in_context(agent_span))
        with pytest.raises(LlmCallsLimitExceededError) as limit_error:
            async for _ in runner.run_async(
                user_id="test-user",
                session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text="Run the tool.")]),
                run_config=RunConfig(
                    max_llm_calls=1,
                    streaming_mode=StreamingMode.NONE,
                ),
            ):
                pass
        telemetry.close(error=limit_error.value)

    inference_spans = [
        span
        for span in span_exporter.get_finished_spans()
        if span.name == f"generate_content {model_name}"
    ]
    assert len(model_requests) == 1
    assert tool_calls == [True]
    assert len(inference_starts) == 1
    assert not any(
        part["type"] == "tool_call_response"
        for message in inference_starts[0]["input_messages"]
        for part in message["parts"]
    )
    assert len(inference_spans) == 1
    assert inference_spans[0].parent.span_id == agent_span_id
    assert (
        REGISTRY.get_sample_value("gen_ai_client_operation_duration_seconds_count", duration_labels)
        == duration_count + 1
    )
    assert (
        REGISTRY.get_sample_value(
            "gen_ai_client_operation_duration_seconds_count", error_duration_labels
        )
        or 0.0
    ) == error_duration_count
