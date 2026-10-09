"""OpenAI generation spans from completed SDK responses."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from lightspeed_agentic.types import ProviderQueryOptions


class _FakeAgent:
    pass


def _options(cwd: Path) -> ProviderQueryOptions:
    return ProviderQueryOptions(
        prompt="user prompt",
        system_prompt="system instructions",
        model="gpt-4.1-mini",
        max_turns=1,
        allowed_tools=[],
        cwd=str(cwd),
    )


def _model_response(
    output: list[Any],
    *,
    response_id: str | None = None,
    request_id: str | None = None,
    requests: int = 1,
    input_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> Any:
    from agents.items import ModelResponse
    from agents.usage import Usage
    from openai.types.responses.response_usage import OutputTokensDetails

    return ModelResponse(
        output=output,
        usage=Usage(
            requests=requests,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
        ),
        response_id=response_id,
        request_id=request_id,
    )


def _text_response(
    text: str,
    *,
    message_id: str,
    response_id: str | None = None,
) -> Any:
    from openai.types.responses.response_output_message import ResponseOutputMessage
    from openai.types.responses.response_output_text import ResponseOutputText

    return _model_response(
        [
            ResponseOutputMessage(
                id=message_id,
                content=[
                    ResponseOutputText(
                        text=text,
                        type="output_text",
                        annotations=[],
                        logprobs=[],
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        response_id=response_id,
    )


def _invocation_span() -> Any:
    return trace.get_tracer("openai-generation-tests").start_span(
        "invoke_agent",
        attributes={"agenticrun.uid": "run-123", "agenticrun.phase": "task"},
    )


def _generation_spans(exporter: Any) -> list[Any]:
    return [
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get("gen_ai.operation.name") == "chat"
    ]


def _generation_span(exporter: Any) -> Any:
    return next(iter(_generation_spans(exporter)))


async def _capture_generation_span(
    response: Any,
    exporter: Any,
    options: ProviderQueryOptions,
    *,
    api_type: str = "responses",
) -> Any:
    from lightspeed_agentic.providers.openai import _create_generation_hooks

    agent = _FakeAgent()
    invocation = _invocation_span()
    with trace.use_span(invocation, end_on_exit=False):
        hooks = _create_generation_hooks(
            agent,
            options,
            otel_context.get_current(),
            api_type=api_type,
        )
        await hooks.on_llm_start(None, agent, None, [])
        await hooks.on_llm_end(None, agent, response)
    invocation.end()
    return _generation_span(exporter)


def _text_delta_event(
    text: str,
    *,
    item_id: str = "msg_1",
    sequence_number: int = 1,
) -> Any:
    from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

    return ResponseTextDeltaEvent(
        content_index=0,
        delta=text,
        item_id=item_id,
        logprobs=[],
        output_index=0,
        sequence_number=sequence_number,
        type="response.output_text.delta",
    )


def _reasoning_delta_event(text: str) -> Any:
    from openai.types.responses.response_reasoning_text_delta_event import (
        ResponseReasoningTextDeltaEvent,
    )

    return ResponseReasoningTextDeltaEvent(
        content_index=0,
        delta=text,
        item_id="rs_1",
        output_index=0,
        sequence_number=1,
        type="response.reasoning_text.delta",
    )


def _apply_patch_tool_events(agent: Any, patch_call: Any, function_call: Any) -> list[Any]:
    from agents.items import ToolCallItem, ToolCallOutputItem
    from agents.stream_events import RunItemStreamEvent

    return [
        RunItemStreamEvent(name="tool_called", item=ToolCallItem(agent, patch_call)),
        RunItemStreamEvent(
            name="tool_output",
            item=ToolCallOutputItem(agent, SimpleNamespace(call_id="call-patch"), {"safe": True}),
        ),
        RunItemStreamEvent(name="tool_called", item=ToolCallItem(agent, function_call)),
        RunItemStreamEvent(
            name="tool_output",
            item=ToolCallOutputItem(agent, SimpleNamespace(call_id="call-dict"), "read result"),
        ),
    ]


class _FakeProviderQueryStream:
    def __init__(
        self,
        agent: Any,
        hooks: Any,
        *,
        response: Any,
        events_before_response: list[Any],
        events_after_response: list[Any],
        other_agent_response: Any | None,
    ) -> None:
        self._agent = agent
        self._hooks = hooks
        self._response = response
        self._events_before_response = events_before_response
        self._events_after_response = events_after_response
        self._other_agent_response = other_agent_response
        self.context_wrapper = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=6,
                output_tokens=7,
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
            model=None,
        )
        self.final_output = "terminal answer"

    async def stream_events(self):
        await self._hooks.on_llm_start(None, self._agent, None, [])
        if self._other_agent_response is not None:
            other_agent = _FakeAgent()
            await self._hooks.on_llm_start(None, other_agent, None, [])
            await self._hooks.on_llm_end(None, other_agent, self._other_agent_response)
        for event in self._events_before_response:
            yield event
        await self._hooks.on_llm_end(None, self._agent, self._response)
        for event in self._events_after_response:
            yield event


def _install_provider_query(
    monkeypatch: pytest.MonkeyPatch,
    cwd: Path,
    *,
    provider_type: str = "openai",
    base_url: str | None = None,
    azure_api_version: str | None = None,
    response: Any | None = None,
    events_before_response: list[Any] | None = None,
    events_after_response: Callable[[Any], list[Any]] | None = None,
    other_agent_response: Any | None = None,
) -> tuple[Any, ProviderQueryOptions]:
    import agents.models.openai_chatcompletions as chatcompletions
    import agents.models.openai_responses as responses
    import agents.sandbox as agents_sandbox
    from agents import Runner

    import lightspeed_agentic.providers.openai as openai_provider

    main_agent = _FakeAgent()
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", provider_type)
    if base_url is None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    if azure_api_version is None:
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    else:
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", azure_api_version)

    monkeypatch.setattr(openai_provider, "_ensure_openai_init", lambda: None)
    monkeypatch.setattr(openai_provider, "_build_manifest", lambda _cwd: None)
    monkeypatch.setattr(agents_sandbox, "SandboxAgent", lambda **_kwargs: main_agent)
    monkeypatch.setattr(responses, "OpenAIResponsesModel", lambda **_kwargs: object())
    monkeypatch.setattr(chatcompletions, "OpenAIChatCompletionsModel", lambda **_kwargs: object())

    provider = openai_provider.OpenAIProvider()
    provider._client = object()
    if provider_type == "azure":
        monkeypatch.setattr(provider, "_build_azure_model", lambda *_args: object())

    model_response = response if response is not None else _model_response([])

    def run_streamed(agent: Any, _prompt: str, *, hooks: Any, **_kwargs: Any) -> Any:
        after_response = events_after_response(agent) if events_after_response else []
        return _FakeProviderQueryStream(
            agent,
            hooks,
            response=model_response,
            events_before_response=events_before_response or [],
            events_after_response=after_response,
            other_agent_response=other_agent_response,
        )

    monkeypatch.setattr(Runner, "run_streamed", staticmethod(run_streamed))
    return provider, _options(cwd)


@pytest.mark.asyncio
async def test_responses_generation_preserves_output_item_and_content_order(
    span_exporter: Any, tmp_path: Path
) -> None:
    from openai.types.responses.response_custom_tool_call import ResponseCustomToolCall
    from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
    from openai.types.responses.response_output_message import ResponseOutputMessage
    from openai.types.responses.response_output_refusal import ResponseOutputRefusal
    from openai.types.responses.response_output_text import ResponseOutputText
    from openai.types.responses.response_reasoning_item import (
        Content,
        ResponseReasoningItem,
        Summary,
    )

    patch_input = '*** Begin Patch\n*** Update File: "雪 file.py"\n+quoted "line"\n*** End Patch'
    response = _model_response(
        [
            ResponseReasoningItem(
                id="rs_1",
                content=[Content(text="think-first", type="reasoning_text")],
                summary=[Summary(text="summary-next", type="summary_text")],
                type="reasoning",
            ),
            ResponseFunctionToolCall(
                arguments='{"pod": "pod-a"}',
                call_id="call-lookup",
                name="lookup",
                type="function_call",
            ),
            ResponseCustomToolCall(
                input=patch_input,
                call_id="call-patch",
                name="apply_patch",
                type="custom_tool_call",
            ),
            ResponseOutputMessage(
                id="msg_1",
                content=[
                    ResponseOutputText(
                        text="say-after-tools",
                        type="output_text",
                        annotations=[],
                        logprobs=[],
                    ),
                    ResponseOutputRefusal(refusal="refusal text", type="refusal"),
                    ResponseOutputText(text="", type="output_text", annotations=[], logprobs=[]),
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ]
    )

    span = await _capture_generation_span(response, span_exporter, _options(tmp_path))

    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "think-first"},
                {"type": "reasoning", "content": "summary-next"},
                {
                    "type": "tool_call",
                    "id": "call-lookup",
                    "name": "lookup",
                    "arguments": {"pod": "pod-a"},
                },
                {
                    "type": "tool_call",
                    "id": "call-patch",
                    "name": "apply_patch",
                    "arguments": patch_input,
                },
                {"type": "text", "content": "say-after-tools"},
                {"type": "refusal", "content": "refusal text"},
                {"type": "text", "content": ""},
            ],
        }
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param("{not-json", id="malformed-json"),
        pytest.param("NaN", id="non-finite-number"),
        pytest.param('{"too_large": 1e999}', id="exponent-overflow"),
    ],
)
@pytest.mark.asyncio
async def test_responses_generation_preserves_raw_non_json_tool_arguments(
    span_exporter: Any, tmp_path: Path, arguments: str
) -> None:
    from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

    response = _model_response(
        [
            ResponseFunctionToolCall(
                arguments=arguments,
                call_id="call-raw",
                name="raw_tool",
                type="function_call",
            )
        ]
    )
    span = await _capture_generation_span(response, span_exporter, _options(tmp_path))

    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "call-raw",
                    "name": "raw_tool",
                    "arguments": arguments,
                }
            ],
        }
    ]


@pytest.mark.asyncio
async def test_generation_hook_keeps_invocation_active_and_records_observed_metadata(
    span_exporter: Any, tmp_path: Path
) -> None:
    from lightspeed_agentic.providers.openai import _create_generation_hooks

    options = _options(tmp_path)
    agent = _FakeAgent()
    invocation = _invocation_span()
    response = _model_response(
        [],
        response_id="resp_actual",
        request_id="transport_request_only",
        input_tokens=0,
        output_tokens=9,
        reasoning_tokens=4,
    )

    with trace.use_span(invocation, end_on_exit=False):
        hooks = _create_generation_hooks(
            agent,
            options,
            otel_context.get_current(),
            api_type="responses",
        )
        await hooks.on_llm_start(
            None,
            agent,
            "system instructions",
            [{"role": "user", "content": "user prompt"}],
        )
        assert (
            trace.get_current_span().get_span_context().span_id
            == invocation.get_span_context().span_id
        )
        await hooks.on_llm_end(None, agent, response)
        assert (
            trace.get_current_span().get_span_context().span_id
            == invocation.get_span_context().span_id
        )
    invocation.end()

    span = _generation_span(span_exporter)
    attributes = span.attributes
    assert span.name == "chat gpt-4.1-mini"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET
    assert span.parent.span_id == invocation.get_span_context().span_id
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["gen_ai.request.model"] == options.model
    assert attributes["gen_ai.provider.name"] == "openai"
    assert attributes["openai.api.type"] == "responses"
    assert attributes["agenticrun.uid"] == "run-123"
    assert attributes["agenticrun.phase"] == "task"
    assert attributes["gen_ai.response.id"] == "resp_actual"
    assert attributes["gen_ai.usage.input_tokens"] == 0
    assert attributes["gen_ai.usage.output_tokens"] == 9
    assert attributes["gen_ai.usage.reasoning.output_tokens"] == 4
    assert "gen_ai.response.model" not in attributes
    assert "gen_ai.response.finish_reasons" not in attributes
    assert "gen_ai.input.messages" not in attributes
    assert "gen_ai.system_instructions" not in attributes
    assert "transport_request_only" not in attributes.values()


@pytest.mark.asyncio
async def test_generation_preserves_zero_usage_and_omits_missing_counts(
    span_exporter: Any, tmp_path: Path
) -> None:
    response = SimpleNamespace(
        output=[],
        usage=SimpleNamespace(
            requests=1,
            output_tokens=0,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )
    span = await _capture_generation_span(response, span_exporter, _options(tmp_path))
    attributes = span.attributes

    assert "gen_ai.usage.input_tokens" not in attributes
    assert attributes["gen_ai.usage.output_tokens"] == 0
    assert "gen_ai.usage.reasoning.output_tokens" not in attributes
    assert "gen_ai.response.id" not in attributes


@pytest.mark.asyncio
async def test_chat_completions_converter_preserves_output_order_and_actual_response_id(
    span_exporter: Any, tmp_path: Path
) -> None:
    from agents.models.chatcmpl_converter import Converter
    from openai.types.chat import (
        ChatCompletionMessage,
        ChatCompletionMessageFunctionToolCall,
    )

    tool_call = ChatCompletionMessageFunctionToolCall(
        id="call-chat-1",
        type="function",
        function={"name": "lookup", "arguments": '{"pod": "pod-b"}'},
    )

    message = ChatCompletionMessage(
        role="assistant",
        content="say-second",
        reasoning_content="think-first",
        tool_calls=[tool_call],
    )
    output_items = Converter.message_to_output_items(
        message,
        provider_data={
            "model": "requested-model-copy",
            "response_id": "chatcmpl_actual",
        },
    )
    response = _model_response(
        output_items,
        response_id=None,
        request_id="transport_request_only",
        input_tokens=13,
        output_tokens=0,
        reasoning_tokens=0,
    )
    options = _options(tmp_path)
    span = await _capture_generation_span(
        response,
        span_exporter,
        options,
        api_type="chat_completions",
    )
    attributes = span.attributes

    assert attributes["openai.api.type"] == "chat_completions"
    assert attributes["gen_ai.response.id"] == "chatcmpl_actual"
    assert attributes["gen_ai.usage.input_tokens"] == 13
    assert attributes["gen_ai.usage.output_tokens"] == 0
    assert "gen_ai.usage.reasoning.output_tokens" not in attributes
    assert "gen_ai.response.model" not in attributes
    assert "gen_ai.response.finish_reasons" not in attributes
    assert "transport_request_only" not in attributes.values()
    assert json.loads(attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "think-first"},
                {"type": "text", "content": "say-second"},
                {
                    "type": "tool_call",
                    "id": "call-chat-1",
                    "name": "lookup",
                    "arguments": {"pod": "pod-b"},
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_generation_omits_usage_when_request_count_is_zero(
    span_exporter: Any, tmp_path: Path
) -> None:
    response = _model_response(
        [],
        requests=0,
        input_tokens=12,
        output_tokens=34,
        reasoning_tokens=56,
    )
    span = await _capture_generation_span(response, span_exporter, _options(tmp_path))
    attributes = span.attributes

    assert "gen_ai.usage.input_tokens" not in attributes
    assert "gen_ai.usage.output_tokens" not in attributes
    assert "gen_ai.usage.reasoning.output_tokens" not in attributes


@pytest.mark.parametrize(
    ("provider_type", "base_url", "api_version", "expected_api_type"),
    [
        pytest.param("openai", None, None, "responses", id="native-openai"),
        pytest.param(
            "openai",
            "https://vllm.example/v1",
            None,
            "chat_completions",
            id="compatible",
        ),
        pytest.param(
            "azure",
            "https://api.openai.com/v1",
            "2024-10-21",
            "chat_completions",
            id="azure-chat-version",
        ),
        pytest.param(
            "azure",
            "https://custom.example/v1",
            "2025-03-01-preview",
            "responses",
            id="azure-responses-version",
        ),
    ],
)
@pytest.mark.asyncio
async def test_query_generation_api_type_matches_existing_client_routing(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider_type: str,
    base_url: str | None,
    api_version: str | None,
    expected_api_type: str,
) -> None:
    provider, options = _install_provider_query(
        monkeypatch,
        tmp_path,
        provider_type=provider_type,
        base_url=base_url,
        azure_api_version=api_version,
    )
    invocation = _invocation_span()
    with trace.use_span(invocation, end_on_exit=False):
        events = [event async for event in provider.query(options)]
        assert (
            trace.get_current_span().get_span_context().span_id
            == invocation.get_span_context().span_id
        )
    invocation.end()

    span = _generation_span(span_exporter)
    assert span.attributes["openai.api.type"] == expected_api_type
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.parent.span_id == invocation.get_span_context().span_id
    assert span.attributes["agenticrun.uid"] == "run-123"
    assert span.attributes["agenticrun.phase"] == "task"
    assert events[-1].type == "result"


@pytest.mark.asyncio
async def test_openai_query_keeps_apply_patch_trace_input_out_of_events_and_logs(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from agents.stream_events import RawResponsesStreamEvent
    from openai.types.responses.response_custom_tool_call import ResponseCustomToolCall
    from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

    from lightspeed_agentic.logging import EventLogger
    from lightspeed_agentic.types import (
        ContentBlockStopEvent,
        ResultEvent,
        TextDeltaEvent,
        ThinkingDeltaEvent,
        ToolCallEvent,
        ToolResultEvent,
        stringify,
    )

    patch_input = "*** Begin Patch\n*** Update File: file.py\n+exact patch input\n*** End Patch"
    custom_call = ResponseCustomToolCall(
        input=patch_input,
        call_id="call-patch",
        name="apply_patch",
        type="custom_tool_call",
    )
    function_call = {
        "type": "function_call",
        "call_id": "call-dict",
        "name": "read_file",
        "arguments": {"path": "dictionary.py"},
    }
    response = _model_response(
        [
            custom_call,
            ResponseFunctionToolCall(
                arguments='{"path": "file.py"}',
                call_id="call-dict",
                name="read_file",
                type="function_call",
            ),
        ],
        response_id="resp-events",
        input_tokens=10,
        output_tokens=4,
    )
    provider, options = _install_provider_query(
        monkeypatch,
        tmp_path,
        response=response,
        events_before_response=[
            RawResponsesStreamEvent(data=_reasoning_delta_event("think-delta")),
            RawResponsesStreamEvent(data=_text_delta_event("text-delta", sequence_number=2)),
        ],
        events_after_response=lambda agent: _apply_patch_tool_events(
            agent, custom_call, function_call
        ),
    )
    events = [event async for event in provider.query(options)]

    tool_calls = [event for event in events if isinstance(event, ToolCallEvent)]
    tool_results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert [event for event in events if isinstance(event, ThinkingDeltaEvent)] == [
        ThinkingDeltaEvent(thinking="think-delta")
    ]
    assert [event for event in events if isinstance(event, TextDeltaEvent)] == [
        TextDeltaEvent(text="text-delta")
    ]
    assert [(event.name, event.input, event.call_id) for event in tool_calls] == [
        ("apply_patch", "", "call-patch"),
        ("read_file", "", "call-dict"),
    ]
    assert [event.trace_input for event in tool_calls] == [
        patch_input,
        stringify({"path": "dictionary.py"}),
    ]
    assert [(event.output, event.call_id) for event in tool_results] == [
        (stringify({"safe": True}), "call-patch"),
        ("read result", "call-dict"),
    ]
    assert isinstance(events[-2], ContentBlockStopEvent)
    assert events[-1] == ResultEvent(
        text="terminal answer",
        input_tokens=6,
        output_tokens=7,
        response_model=options.model,
    )

    generation = _generation_span(span_exporter)
    assert json.loads(generation.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "call-patch",
                    "name": "apply_patch",
                    "arguments": patch_input,
                },
                {
                    "type": "tool_call",
                    "id": "call-dict",
                    "name": "read_file",
                    "arguments": {"path": "file.py"},
                },
            ],
        }
    ]

    with caplog.at_level(logging.INFO, logger="lightspeed_agentic"):
        event_logger = EventLogger("main")
        for event in events:
            event_logger.log(event)
    assert "tool_use: apply_patch()" in caplog.text
    assert "tool_use: read_file()" in caplog.text
    assert patch_input not in caplog.text
    assert 'tool_result: {"safe": true}' in caplog.text
    assert "result: tokens=13" in caplog.text


@pytest.mark.asyncio
async def test_openai_query_records_generation_only_for_main_agent(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    main_response = _text_response("main output", message_id="main_msg")
    other_response = _text_response("nested output", message_id="other_msg")
    provider, options = _install_provider_query(
        monkeypatch,
        tmp_path,
        response=main_response,
        other_agent_response=other_response,
    )

    async for _event in provider.query(options):
        pass

    generations = _generation_spans(span_exporter)
    assert len(generations) == 1
    assert json.loads(generations[0].attributes["gen_ai.output.messages"]) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "main output"}]}
    ]


class _LateCallbackStreamingResult:
    def __init__(
        self,
        agent: Any,
        hooks: Any,
        completed_response: Any,
        late_response: Any,
        delta: Any,
    ) -> None:
        self._agent = agent
        self._hooks = hooks
        self._completed_response = completed_response
        self._late_response = late_response
        self._delta = delta
        self._allow_late_callbacks = asyncio.Event()
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self.background_task: asyncio.Task[Any] | None = None

    async def stream_events(self):
        self.background_task = asyncio.create_task(self._deliver_callbacks())
        yield await self._events.get()

    async def _deliver_callbacks(self) -> None:
        from agents.stream_events import RawResponsesStreamEvent

        await self._hooks.on_llm_start(None, self._agent, None, [])
        await self._hooks.on_llm_end(None, self._agent, self._completed_response)
        await self._hooks.on_llm_start(None, self._agent, None, [])
        await self._events.put(RawResponsesStreamEvent(data=self._delta))
        await self._allow_late_callbacks.wait()

        # SDK work can finish and start another request after the provider closes.
        await self._hooks.on_llm_end(None, self._agent, self._late_response)
        await self._hooks.on_llm_start(None, self._agent, None, [])
        await self._hooks.on_llm_end(None, self._agent, self._late_response)

    async def finish_background(self) -> None:
        self._allow_late_callbacks.set()
        if self.background_task is not None:
            await asyncio.wait_for(self.background_task, timeout=1)


@pytest.mark.asyncio
async def test_cancellation_closes_open_generation_and_ignores_late_runner_callbacks(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import contextlib

    from agents import Runner

    from lightspeed_agentic.types import TextDeltaEvent

    cancellation = asyncio.CancelledError("cancelled while provider was yielding")
    delta = _text_delta_event("observed delta only", item_id="msg_partial")
    completed_response = _text_response(
        "completed before cancellation",
        message_id="msg_before_cancel",
        response_id="resp_before_cancel",
    )
    late_response = _text_response(
        "late background output",
        message_id="msg_late",
        response_id="resp_late",
    )
    provider, options = _install_provider_query(monkeypatch, tmp_path)
    streaming_results: list[_LateCallbackStreamingResult] = []

    def run_streamed(agent: Any, _prompt: str, *, hooks: Any, **_kwargs: Any) -> Any:
        result = _LateCallbackStreamingResult(
            agent, hooks, completed_response, late_response, delta
        )
        streaming_results.append(result)
        return result

    monkeypatch.setattr(Runner, "run_streamed", staticmethod(run_streamed))

    invocation = _invocation_span()
    stream = provider.query(options)
    try:
        with trace.use_span(invocation, end_on_exit=False):
            assert await anext(stream) == TextDeltaEvent(text="observed delta only")
            background_task = streaming_results[0].background_task
            assert background_task is not None
            assert not background_task.done()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await stream.athrow(cancellation)
            assert exc_info.value is cancellation
    finally:
        with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
            await stream.athrow(cancellation)
        invocation.end()
        if streaming_results:
            await streaming_results[0].finish_background()

    generations = _generation_spans(span_exporter)
    assert len(generations) == 2
    completed_span, cancelled_span = generations
    assert completed_span.parent.span_id == invocation.get_span_context().span_id
    assert completed_span.status.status_code == StatusCode.UNSET
    assert json.loads(completed_span.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "completed before cancellation"}],
        }
    ]
    assert cancelled_span.parent.span_id == invocation.get_span_context().span_id
    assert cancelled_span.status.status_code == StatusCode.ERROR
    assert cancelled_span.attributes["error.type"] == "CancelledError"
    assert "gen_ai.output.messages" not in cancelled_span.attributes
