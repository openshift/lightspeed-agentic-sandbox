"""Readable tests for DeepAgents provider-generation spans."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest

from .conftest import make_model_route

_MODEL = "claude-sonnet-4-6"


@pytest.fixture
def generation_callback(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Any, Any]]:
    """Provide the main-generation callback and its correlated invocation parent."""
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind

    from lightspeed_agentic.providers.deepagents import _create_generation_callback

    del span_exporter  # Fixture dependency installs the tracer before creating the parent.

    monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)
    parent = trace.get_tracer("test").start_span(
        "invoke_agent",
        kind=SpanKind.INTERNAL,
        attributes={"agenticrun.uid": "parent-uid", "agenticrun.phase": "analysis"},
    )
    callback = _create_generation_callback(
        make_model_route(Mock(), model=_MODEL),
        trace.set_span_in_context(parent),
        main_name=None,
    )
    try:
        yield callback, parent
    finally:
        parent.end()


def _generation_messages(span: Any) -> list[dict[str, Any]]:
    return json.loads(span.attributes["gen_ai.output.messages"])


def _graph_generation_model() -> Any:
    """Script main task, child reply, main final, then raw structured-output response."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatResult

    class GraphGenerationModel(FakeMessagesListChatModel):
        def bind_tools(self, _tools: Any, **_kwargs: Any) -> GraphGenerationModel:
            return self

        def _generate(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            result = super()._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            response = result.generations[0].message
            if any(call.get("name") == "OutputModel" for call in response.tool_calls):
                return ChatResult(
                    generations=result.generations,
                    llm_output={
                        "model": "shape-observed-model",
                        "id": "shape-response-id",
                        "stop_reason": "end_turn",
                    },
                )
            return result

    # The shared fake model is consumed in this order by the graph:
    # main task -> general-purpose child -> main final answer -> raw OutputModel call.
    return GraphGenerationModel(
        responses=[
            AIMessage(
                content_blocks=[
                    {"type": "reasoning", "reasoning": "think-first"},
                    {"type": "text", "text": "say-second"},
                    {
                        "type": "tool_call",
                        "id": "task-call",
                        "name": "task",
                        "args": {
                            "description": "Inspect the deployment status",
                            "subagent_type": "general-purpose",
                        },
                    },
                ],
                response_metadata={
                    "output_version": "v1",
                    "model": "observed-backend-model",
                    "id": "response-task",
                    "stop_reason": "tool_use",
                },
                usage_metadata={
                    "input_tokens": 0,
                    "output_tokens": 4,
                    "total_tokens": 4,
                    "output_token_details": {"reasoning": 0},
                },
            ),
            AIMessage(
                content_blocks=[{"type": "text", "text": "child report"}],
                response_metadata={"output_version": "v1"},
            ),
            AIMessage(
                content_blocks=[{"type": "text", "text": "main final answer"}],
                response_metadata={"output_version": "v1"},
            ),
            AIMessage(
                content_blocks=[
                    {
                        "type": "tool_call",
                        "id": "shape-call",
                        "name": "OutputModel",
                        "args": {"status": "parsed"},
                    }
                ],
                response_metadata={"output_version": "v1"},
                usage_metadata={
                    "input_tokens": 2,
                    "output_tokens": 0,
                    "total_tokens": 2,
                },
            ),
        ]
    )


def _partial_response_model(
    blocked_after_partial: asyncio.Event | None = None,
) -> Any:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    class PartialResponseModel(FakeMessagesListChatModel):
        async def _astream(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[ChatGenerationChunk]:
            del messages, stop, kwargs
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=[
                        {"type": "thinking", "thinking": "partial-reasoning"},
                        {"type": "text", "text": "partial-text"},
                        {
                            "type": "tool_use",
                            "id": "partial-call",
                            "name": "execute",
                            "input": {},
                        },
                    ],
                    tool_call_chunks=[
                        {
                            "id": "partial-call",
                            "name": "execute",
                            "args": '{"command":',
                            "index": 0,
                        }
                    ],
                    response_metadata={
                        "model_provider": "anthropic",
                        "model_name": "observed-partial-model",
                        "id": "partial-response",
                    },
                    usage_metadata={
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "output_token_details": {"reasoning": 0},
                    },
                )
            )
            if blocked_after_partial is not None:
                blocked_after_partial.set()
                await asyncio.Event().wait()
            raise RuntimeError("partial generation failed")

    return PartialResponseModel(
        responses=[AIMessage(content="unused")],
        output_version="v0",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_id", "telemetry_provider"),
    [
        ("anthropic", "anthropic"),
        ("vertex/anthropic", "gcp.vertex_ai"),
        ("bedrock/anthropic", "aws.bedrock"),
    ],
)
async def test_run_agent_query_traces_main_generations_and_raw_shape_response(
    span_exporter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    route_id: str,
    telemetry_provider: str,
) -> None:
    from opentelemetry.trace import SpanKind

    from lightspeed_agentic.providers import deepagents as mod
    from lightspeed_agentic.run_agent import run_agent_query

    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://ignored-endpoint.example.test")
    model = _graph_generation_model()
    model_route = replace(
        make_model_route(Mock(return_value=model), model="claude-trace-test"),
        route=route_id,
        telemetry_provider=telemetry_provider,
    )

    result = await run_agent_query(
        mod.DeepAgentsProvider(model_route),
        prompt="Summarize the deployment status using a subagent.",
        system_prompt="Follow instructions.",
        output_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        },
        context=None,
        skills_dir=str(tmp_path),
        model="must-not-override-route",
        max_turns=20,
        timeout_seconds=30,
        tool_output_inspection_enabled=False,
        audit_enabled=False,
        capture_content=False,
        agenticrun_uid="run-trace-test",
        step="analysis",
    )

    spans = span_exporter.get_finished_spans()
    invocation = next(span for span in spans if span.name == "invoke_agent")
    generations = [span for span in spans if span.name == "chat claude-trace-test"]
    expected_task_parts = [
        {"type": "reasoning", "content": "think-first"},
        {"type": "text", "content": "say-second"},
        {
            "type": "tool_call",
            "id": "task-call",
            "name": "task",
            "arguments": {
                "description": "Inspect the deployment status",
                "subagent_type": "general-purpose",
            },
        },
    ]
    expected_shape_parts = [
        {
            "type": "tool_call",
            "id": "shape-call",
            "name": "OutputModel",
            "arguments": {"status": "parsed"},
        }
    ]
    # Exact sequence excludes the child's response and keeps the raw shape call last.
    assert [_generation_messages(span)[0]["parts"] for span in generations] == [
        expected_task_parts,
        [{"type": "text", "content": "main final answer"}],
        expected_shape_parts,
    ]
    assert result.output["status"] == "parsed"
    assert all(span.kind is SpanKind.CLIENT for span in generations)
    for span in generations:
        assert span.parent is not None
        assert span.parent.span_id == invocation.context.span_id
        attributes = dict(span.attributes)
        assert attributes["agenticrun.uid"] == "run-trace-test"
        assert attributes["agenticrun.phase"] == "analysis"
        assert attributes["gen_ai.operation.name"] == "chat"
        assert attributes["gen_ai.provider.name"] == telemetry_provider
        assert attributes["gen_ai.request.model"] == "claude-trace-test"

    task_attributes = dict(generations[0].attributes)
    assert task_attributes["gen_ai.response.model"] == "observed-backend-model"
    assert task_attributes["gen_ai.response.id"] == "response-task"
    assert list(task_attributes["gen_ai.response.finish_reasons"]) == ["tool_use"]
    assert task_attributes["gen_ai.usage.input_tokens"] == 0
    assert task_attributes["gen_ai.usage.output_tokens"] == 4
    assert task_attributes["gen_ai.usage.reasoning.output_tokens"] == 0

    shape_attributes = dict(generations[2].attributes)
    assert shape_attributes["gen_ai.response.model"] == "shape-observed-model"
    assert shape_attributes["gen_ai.response.id"] == "shape-response-id"
    assert list(shape_attributes["gen_ai.response.finish_reasons"]) == ["end_turn"]
    terminal = json.loads(invocation.attributes["gen_ai.output.messages"])
    assert json.loads(terminal[0]["parts"][0]["content"]) == {"status": "parsed"}
    assert terminal[0]["parts"] != _generation_messages(generations[2])[0]["parts"]


@pytest.mark.asyncio
async def test_generation_callback_preserves_empty_output_and_omits_missing_metadata(
    span_exporter,
    generation_callback: tuple[Any, Any],
) -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from opentelemetry.trace import SpanKind

    callback, parent = generation_callback
    model = FakeMessagesListChatModel(responses=[AIMessage(content="")])

    await model.ainvoke("main", config={"callbacks": [callback]})

    spans = [span for span in span_exporter.get_finished_spans() if span.name == f"chat {_MODEL}"]
    assert len(spans) == 1
    attributes = dict(spans[0].attributes)
    assert _generation_messages(spans[0]) == [
        {"role": "assistant", "parts": [{"type": "text", "content": ""}]}
    ]
    for attribute in (
        "gen_ai.response.model",
        "gen_ai.response.id",
        "gen_ai.response.finish_reasons",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.reasoning.output_tokens",
    ):
        assert attribute not in attributes
    assert spans[0].kind is SpanKind.CLIENT
    assert spans[0].parent is not None
    assert spans[0].parent.span_id == parent.context.span_id
    assert attributes["agenticrun.uid"] == "parent-uid"
    assert attributes["agenticrun.phase"] == "analysis"


@pytest.mark.asyncio
async def test_generation_callback_falls_back_to_llm_output_metadata(
    span_exporter,
    generation_callback: tuple[Any, Any],
) -> None:
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    callback, _parent = generation_callback
    run_id = uuid4()
    await callback.on_chat_model_start({}, [], run_id=run_id, tags=None, metadata=None)
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content_blocks=[{"type": "text", "text": "fallback response"}],
                        response_metadata={},
                    ),
                    generation_info={},
                )
            ]
        ],
        llm_output={
            "model": "llm-output-model",
            "id": "llm-output-response-id",
            "stop_reason": "end_turn",
        },
    )
    await callback.on_llm_end(response, run_id=run_id)

    span = next(
        span for span in span_exporter.get_finished_spans() if span.name == f"chat {_MODEL}"
    )
    attributes = dict(span.attributes)
    assert attributes["gen_ai.response.model"] == "llm-output-model"
    assert attributes["gen_ai.response.id"] == "llm-output-response-id"
    assert list(attributes["gen_ai.response.finish_reasons"]) == ["end_turn"]
    assert _generation_messages(span) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "fallback response"}]}
    ]


@pytest.mark.parametrize(
    ("metadata", "tags"),
    [
        pytest.param({"lc_agent_name": "general-purpose"}, [], id="named-subagent"),
        pytest.param({"lc_source": "summarization"}, [], id="summarization"),
        pytest.param({}, ["nostream"], id="nostream-classifier"),
    ],
)
@pytest.mark.asyncio
async def test_generation_callback_excludes_subagents_summarization_and_nostream(
    metadata: dict[str, str],
    tags: list[str],
    span_exporter,
    generation_callback: tuple[Any, Any],
) -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    callback, _parent = generation_callback
    model = FakeMessagesListChatModel(responses=[AIMessage(content="excluded response")])

    await model.ainvoke(
        "excluded request",
        config={"callbacks": [callback], "tags": tags, "metadata": metadata},
    )

    assert [
        span for span in span_exporter.get_finished_spans() if span.name == f"chat {_MODEL}"
    ] == []


@pytest.mark.asyncio
async def test_generation_callback_closes_interrupted_span_idempotently(
    span_exporter,
    generation_callback: tuple[Any, Any],
) -> None:
    from opentelemetry.trace import StatusCode

    callback, _parent = generation_callback
    await callback.on_chat_model_start({}, [], run_id=uuid4(), tags=None, metadata=None)
    callback.close_open("generation_interrupted")
    finished_count = len(span_exporter.get_finished_spans())
    callback.close_open("later_error")

    assert len(span_exporter.get_finished_spans()) == finished_count
    interrupted = [
        span
        for span in span_exporter.get_finished_spans()
        if span.name == f"chat {_MODEL}" and span.attributes.get("error.type")
    ]
    assert len(interrupted) == 1
    assert interrupted[0].attributes["error.type"] == "generation_interrupted"
    assert interrupted[0].status.status_code == StatusCode.ERROR
    assert "gen_ai.output.messages" not in interrupted[0].attributes


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        pytest.param("error", "RuntimeError", id="runtime-error"),
        pytest.param("cancel", "CancelledError", id="cancellation"),
    ],
)
@pytest.mark.asyncio
async def test_generation_callback_preserves_partial_response_on_failure_or_cancellation(
    failure: str,
    expected_error: str,
    span_exporter,
    generation_callback: tuple[Any, Any],
) -> None:
    from opentelemetry.trace import StatusCode

    callback, _parent = generation_callback
    reached_cancellation_point = asyncio.Event() if failure == "cancel" else None
    model = _partial_response_model(reached_cancellation_point)

    async def consume_partial_stream() -> None:
        async for _message in model.astream("request", config={"callbacks": [callback]}):
            pass

    task = asyncio.create_task(consume_partial_stream())
    if reached_cancellation_point is not None:
        await asyncio.wait_for(reached_cancellation_point.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(RuntimeError, match="partial generation failed"):
            await task

    generations = [
        span for span in span_exporter.get_finished_spans() if span.name == f"chat {_MODEL}"
    ]
    assert len(generations) == 1
    generation = generations[0]
    attributes = dict(generation.attributes)
    assert _generation_messages(generation) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "partial-reasoning"},
                {"type": "text", "content": "partial-text"},
                {
                    "type": "tool_call_chunk",
                    "id": "partial-call",
                    "name": "execute",
                    "args": '{"command":',
                    "index": 0,
                },
            ],
        }
    ]
    assert attributes["gen_ai.response.model"] == "observed-partial-model"
    assert attributes["gen_ai.response.id"] == "partial-response"
    assert attributes["gen_ai.usage.input_tokens"] == 0
    assert attributes["gen_ai.usage.output_tokens"] == 0
    assert attributes["gen_ai.usage.reasoning.output_tokens"] == 0
    assert attributes["error.type"] == expected_error
    assert generation.status.status_code == StatusCode.ERROR
