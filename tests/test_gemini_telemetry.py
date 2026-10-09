"""Consumer-visible Gemini generation spans from the real ADK Runner."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.trace import SpanKind, StatusCode

from lightspeed_agentic.providers.gemini import GeminiProvider
from lightspeed_agentic.types import (
    MAX_TOOL_RETURN_CHARS,
    TOOL_RETURN_PREVIEW_CHARS,
    ContentBlockStopEvent,
    ProviderQueryOptions,
    ResultEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
)

pytest.importorskip("google.adk")

_MODEL = "gemini-trace-test"


def _options(cwd: Path, *, stream: bool = False) -> ProviderQueryOptions:
    return ProviderQueryOptions(
        prompt="Inspect the requested resource.",
        system_prompt="Follow the user request.",
        model=_MODEL,
        max_turns=5,
        allowed_tools=["Bash"],
        cwd=str(cwd),
        stream=stream,
    )


def _install_scripted_gemini(
    monkeypatch: pytest.MonkeyPatch,
    script: list[list[Any]],
) -> None:
    from google.adk import models
    from google.adk.models import BaseLlm, LlmRequest, LlmResponse

    class ScriptedLlm(BaseLlm):
        responses: list[list[Any]]
        response_index: int = 0

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            _ = llm_request, stream
            responses = self.responses[self.response_index]
            self.response_index += 1
            for response in responses:
                if isinstance(response, BaseException):
                    raise response
                yield response

    def create_model(*, model: str, **_kwargs: Any) -> ScriptedLlm:
        return ScriptedLlm(model=model, responses=script)

    monkeypatch.setattr(models, "Gemini", create_model)


def _sandbox_spans(span_exporter: Any) -> list[Any]:
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.instrumentation_scope is not None
        and span.instrumentation_scope.name == "lightspeed_agentic"
    ]


def _generation_spans(span_exporter: Any) -> list[Any]:
    return sorted(
        (
            span
            for span in _sandbox_spans(span_exporter)
            if span.kind == SpanKind.CLIENT
            and span.attributes.get("gen_ai.operation.name") == "generate_content"
        ),
        key=lambda span: span.start_time,
    )


def _messages(span: Any) -> list[dict[str, Any]]:
    return json.loads(span.attributes["gen_ai.output.messages"])


async def _run_through_agent(provider: GeminiProvider, cwd: Path) -> Any:
    from lightspeed_agentic.run_agent import run_agent_query

    return await run_agent_query(
        provider,
        prompt="Inspect the requested resource.",
        system_prompt="Follow the user request.",
        output_schema=None,
        context=None,
        skills_dir=str(cwd),
        model=_MODEL,
        max_turns=5,
        timeout_seconds=30,
        tool_output_inspection_enabled=False,
        audit_enabled=False,
        capture_content=False,
        agenticrun_uid="run-gemini-trace-test",
        step="analysis",
    )


@pytest.mark.asyncio
async def test_runner_ends_generation_before_local_tool_and_records_trimmed_result(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from google.adk.models import LlmResponse
    from google.genai import types

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    command = "printf '%05000d' 0"
    tool_call = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="execute_bash",
                        args={"command": command},
                    )
                )
            ],
        ),
        model_version="observed-gemini-model",
        finish_reason=types.FinishReason.STOP,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=0,
            candidates_token_count=7,
            thoughts_token_count=0,
        ),
    )
    final = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text='{"success": true, "summary": "finished"}')],
        )
    )
    _install_scripted_gemini(monkeypatch, [[tool_call], [final]])

    result = await _run_through_agent(GeminiProvider(), tmp_path)

    assert result.output["summary"] == "finished"
    spans = _sandbox_spans(span_exporter)
    invocation = next(span for span in spans if span.name == "invoke_agent")
    generations = _generation_spans(span_exporter)
    tool = next(
        span
        for span in spans
        if span.name == "execute_tool execute_bash"
        and span.attributes.get("gen_ai.operation.name") == "execute_tool"
    )
    assert len(generations) == 2
    first, second = generations

    assert first.end_time <= tool.start_time
    assert first.parent is not None
    assert first.parent.span_id == invocation.context.span_id
    assert first.kind == SpanKind.CLIENT
    assert first.attributes["gen_ai.provider.name"] == "gcp.gemini"
    assert first.attributes["gen_ai.request.model"] == _MODEL
    assert first.attributes["agenticrun.uid"] == "run-gemini-trace-test"
    assert first.attributes["agenticrun.phase"] == "analysis"
    call = _messages(first)[0]["parts"][0]
    assert call["type"] == "tool_call"
    assert call["name"] == "execute_bash"
    assert call["arguments"] == {"command": command}
    assert call["id"] == tool.attributes["gen_ai.tool.call.id"]
    assert "gen_ai.response.id" not in first.attributes
    assert first.attributes["gen_ai.response.model"] == "observed-gemini-model"
    assert list(first.attributes["gen_ai.response.finish_reasons"]) == [
        types.FinishReason.STOP.value
    ]
    assert first.attributes["gen_ai.usage.input_tokens"] == 0
    assert first.attributes["gen_ai.usage.output_tokens"] == 7
    assert first.attributes["gen_ai.usage.reasoning.output_tokens"] == 0

    tool_result = json.loads(tool.attributes["gen_ai.tool.call.result"])
    assert tool_result["status"] == "truncated"
    assert len(tool_result["preview"]) == TOOL_RETURN_PREVIEW_CHARS
    assert tool_result["original_size"] > MAX_TOOL_RETURN_CHARS

    assert _messages(second) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": '{"success": true, "summary": "finished"}'}],
        }
    ]
    for attribute in (
        "gen_ai.response.model",
        "gen_ai.response.id",
        "gen_ai.response.finish_reasons",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.reasoning.output_tokens",
    ):
        assert attribute not in second.attributes


@pytest.mark.asyncio
async def test_runner_interrupts_generation_before_next_request_after_non_model_tool_call(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from google.adk.models import LlmResponse
    from google.genai import types

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    tool_call = LlmResponse(
        content=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="execute_bash",
                        args={"command": "true"},
                    )
                )
            ],
        )
    )
    final = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text='{"summary": "continued"}')],
        )
    )
    _install_scripted_gemini(monkeypatch, [[tool_call], [final]])

    result = await _run_through_agent(GeminiProvider(), tmp_path)

    assert result.output["summary"] == "continued"
    generations = _generation_spans(span_exporter)
    assert len(generations) == 2
    first, second = generations
    assert first.attributes["error.type"] == "generation_interrupted"
    assert first.status.status_code == StatusCode.ERROR
    assert "gen_ai.output.messages" not in first.attributes
    assert first.end_time <= second.start_time
    assert _messages(second) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": '{"summary": "continued"}'}],
        }
    ]


@pytest.mark.asyncio
async def test_runner_preserves_hosted_part_order_and_unknown_server_tool_types(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from google.adk.models import LlmResponse
    from google.genai import types

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    final_text = '{"success": true, "summary": "hosted tools observed"}'
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(text=""),
                types.Part(text='Before\nquoted: "café" 雪'),
                types.Part(
                    tool_call=types.ToolCall(
                        id="hosted-call",
                        tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
                        args={"query": 'café\n"pods"'},
                    )
                ),
                types.Part(text="reasoning", thought=True),
                types.Part(
                    tool_response=types.ToolResponse(
                        id="hosted-call",
                        tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
                        response={"results": ["pod-a"]},
                    )
                ),
                types.Part(
                    tool_call=types.ToolCall(
                        id="unknown-call",
                        args={"query": ""},
                    )
                ),
                types.Part(
                    tool_response=types.ToolResponse(
                        id="unknown-call",
                        response={"text": 'line one\n"line two"'},
                    )
                ),
                types.Part(text=final_text),
            ],
        )
    )
    _install_scripted_gemini(monkeypatch, [[response]])

    result = await _run_through_agent(GeminiProvider(), tmp_path)

    assert result.output["summary"] == "hosted tools observed"
    generations = _generation_spans(span_exporter)
    assert len(generations) == 1
    assert _messages(generations[0]) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": ""},
                {"type": "text", "content": 'Before\nquoted: "café" 雪'},
                {
                    "type": "server_tool_call",
                    "id": "hosted-call",
                    "name": "GOOGLE_SEARCH_WEB",
                    "server_tool_call": {
                        "type": "GOOGLE_SEARCH_WEB",
                        "args": {"query": 'café\n"pods"'},
                    },
                },
                {"type": "reasoning", "content": "reasoning"},
                {
                    "type": "server_tool_call_response",
                    "id": "hosted-call",
                    "server_tool_call_response": {
                        "type": "GOOGLE_SEARCH_WEB",
                        "response": {"results": ["pod-a"]},
                    },
                },
                {"type": "tool_call", "id": "unknown-call", "args": {"query": ""}},
                {
                    "type": "tool_response",
                    "id": "unknown-call",
                    "response": {"text": 'line one\n"line two"'},
                },
                {"type": "text", "content": final_text},
            ],
        }
    ]
    assert "gen_ai.response.id" not in generations[0].attributes


@pytest.mark.asyncio
async def test_progressive_sse_uses_sdk_partial_and_final_aggregate_content(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from google.adk.models import LlmResponse
    from google.genai import types
    from opentelemetry import trace

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    partial_thought = LlmResponse(
        content=types.Content(
            role="model", parts=[types.Part(text="partial thought", thought=True)]
        ),
        partial=True,
    )
    partial_text = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text="partial text")]),
        partial=True,
    )
    final_text = 'final "answer" 雪\nsecond line'
    final = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="think-first", thought=True),
                types.Part(text=final_text),
            ],
        ),
        finish_reason=types.FinishReason.STOP,
    )
    _install_scripted_gemini(monkeypatch, [[partial_thought, partial_text, final]])

    with trace.get_tracer("gemini-telemetry-tests").start_as_current_span("test invocation"):
        events = [event async for event in GeminiProvider().query(_options(tmp_path, stream=True))]

    generations = _generation_spans(span_exporter)
    assert len(generations) == 1
    assert _messages(generations[0]) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "think-first"},
                {"type": "text", "content": final_text},
            ],
        }
    ]
    assert events[:2] == [
        ThinkingDeltaEvent(thinking="partial thought"),
        TextDeltaEvent(text="partial text"),
    ]
    assert isinstance(events[-2], ContentBlockStopEvent)
    assert isinstance(events[-1], ResultEvent)


@pytest.mark.asyncio
async def test_generation_provider_label_follows_vertex_selection(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from google.adk.models import LlmResponse
    from google.genai import types

    response = LlmResponse(content=types.Content(role="model", parts=[types.Part(text="done")]))
    expected = []
    for vertex, provider_name in (
        (False, "gcp.gemini"),
        (True, "gcp.vertex_ai"),
    ):
        if vertex:
            monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
        else:
            monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
        _install_scripted_gemini(monkeypatch, [[response]])
        await _run_through_agent(GeminiProvider(), tmp_path)
        expected.append(provider_name)

    assert [
        span.attributes["gen_ai.provider.name"] for span in _generation_spans(span_exporter)
    ] == expected


@pytest.mark.asyncio
async def test_failure_before_output_closes_generation_and_propagates_unchanged(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from opentelemetry import trace

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    failure = RuntimeError("scripted model failure")
    _install_scripted_gemini(monkeypatch, [[failure]])

    with (
        trace.get_tracer("gemini-telemetry-tests").start_as_current_span("test invocation"),
        pytest.raises(RuntimeError) as raised,
    ):
        async for _event in GeminiProvider().query(_options(tmp_path)):
            pass

    assert raised.value is failure
    generation = _generation_spans(span_exporter)[0]
    assert generation.status.status_code == StatusCode.ERROR
    assert generation.attributes["error.type"] == "RuntimeError"
    assert "gen_ai.output.messages" not in generation.attributes
    assert "gen_ai.response.id" not in generation.attributes


@pytest.mark.asyncio
async def test_sdk_error_code_closes_empty_terminal_generation_without_output(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from google.adk.models import LlmResponse
    from opentelemetry import trace

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    _install_scripted_gemini(monkeypatch, [[LlmResponse(error_code="SAFETY")]])

    with trace.get_tracer("gemini-telemetry-tests").start_as_current_span("test invocation"):
        events = [event async for event in GeminiProvider().query(_options(tmp_path))]

    generation = _generation_spans(span_exporter)[0]
    assert events[-1].type == "result"
    assert generation.attributes["error.type"] == "SAFETY"
    assert generation.status.status_code == StatusCode.ERROR
    assert "gen_ai.output.messages" not in generation.attributes
    assert "gen_ai.response.id" not in generation.attributes


@pytest.mark.asyncio
async def test_runner_propagates_explicit_skill_errors_only_to_tool_spans(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from google.adk.models import LlmResponse
    from google.adk.tools.skill_toolset import LoadSkillResourceTool
    from google.genai import types

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    caplog.set_level(logging.INFO, logger="lightspeed_agentic")

    skill_name = "telemetry-skill"
    skill_dir = tmp_path / skill_name
    references_dir = skill_dir / "references"
    references_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: Read telemetry test resources.\n---\n",
        encoding="utf-8",
    )
    literal_content = '{"error":"literal resource data"}'
    (references_dir / "literal.json").write_text(literal_content, encoding="utf-8")

    original_run_async = LoadSkillResourceTool.run_async

    async def return_literal_error(self: Any, *, args: dict[str, Any], tool_context: Any) -> Any:
        if args.get("file_path") == "references/literal-error-key.json":
            return {"error": "literal tool result"}
        return await original_run_async(self, args=args, tool_context=tool_context)

    monkeypatch.setattr(LoadSkillResourceTool, "run_async", return_literal_error)

    calls = [
        ("resource-not-found", "references/missing.md"),
        ("invalid-resource-path", "SKILL.md"),
        ("literal-error-key", "references/literal-error-key.json"),
        ("ordinary-success", "references/literal.json"),
    ]

    def tool_call(call_id: str, file_path: str) -> LlmResponse:
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id=call_id,
                            name="load_skill_resource",
                            args={"skill_name": skill_name, "file_path": file_path},
                        )
                    )
                ],
            )
        )

    final_text = '{"success": true, "summary": "recovered after skill errors"}'
    script = [[tool_call(call_id, file_path)] for call_id, file_path in calls]
    script.append(
        [
            LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=final_text)],
                )
            )
        ]
    )
    _install_scripted_gemini(monkeypatch, script)

    result = await _run_through_agent(GeminiProvider(), tmp_path)

    expected_results = {
        "resource-not-found": {
            "error": "Resource 'references/missing.md' not found in skill 'telemetry-skill'.",
            "error_code": "RESOURCE_NOT_FOUND",
        },
        "invalid-resource-path": {
            "error": "Path must start with 'references/', 'assets/', or 'scripts/'.",
            "error_code": "INVALID_RESOURCE_PATH",
        },
        "literal-error-key": {"error": "literal tool result"},
        "ordinary-success": {
            "skill_name": skill_name,
            "file_path": "references/literal.json",
            "content": literal_content,
        },
    }
    expected_error_types = {
        "resource-not-found": "RESOURCE_NOT_FOUND",
        "invalid-resource-path": "INVALID_RESOURCE_PATH",
    }
    spans = _sandbox_spans(span_exporter)
    invocation = next(span for span in spans if span.name == "invoke_agent")
    tool_spans = {
        span.attributes["gen_ai.tool.call.id"]: span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == "execute_tool"
    }

    assert result.output == {
        "success": True,
        "summary": "recovered after skill errors",
    }
    assert invocation.status.status_code != StatusCode.ERROR
    assert "error.type" not in invocation.attributes
    assert json.loads(invocation.attributes["gen_ai.output.messages"]) == [
        {"role": "assistant", "parts": [{"type": "text", "content": final_text}]}
    ]
    assert all(event.name != "gen_ai.choice" for event in invocation.events)
    assert set(tool_spans) == set(expected_results)

    for call_id, expected_response in expected_results.items():
        tool = tool_spans[call_id]
        assert tool.name == "execute_tool load_skill_resource"
        assert tool.attributes["gen_ai.tool.call.id"] == call_id
        assert json.loads(tool.attributes["gen_ai.tool.call.result"]) == expected_response
        if call_id in expected_error_types:
            assert tool.attributes["error.type"] == expected_error_types[call_id]
            assert tool.status.status_code == StatusCode.ERROR
        else:
            assert "error.type" not in tool.attributes
            assert tool.status.status_code == StatusCode.OK

    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "lightspeed_agentic"
        and record.getMessage().startswith("[provider:run] tool_result: ")
    ] == [
        f"[provider:run] tool_result: {json.dumps(response)}"
        for response in expected_results.values()
    ]


@pytest.mark.asyncio
async def test_cancellation_keeps_sdk_partial_output_and_propagates_unchanged(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from google.adk.models import LlmResponse
    from google.genai import types
    from opentelemetry import trace

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    partial = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="partial thought", thought=True),
                types.Part(text="partial text"),
            ],
        ),
        partial=True,
        model_version="observed-partial-model",
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=0,
            candidates_token_count=0,
            thoughts_token_count=0,
        ),
    )
    _install_scripted_gemini(monkeypatch, [[partial]])
    cancellation = asyncio.CancelledError("cancelled during progressive SSE")
    stream = GeminiProvider().query(_options(tmp_path, stream=True))

    with trace.get_tracer("gemini-telemetry-tests").start_as_current_span("test invocation"):
        assert await anext(stream) == ThinkingDeltaEvent(thinking="partial thought")
        with pytest.raises(asyncio.CancelledError) as raised:
            await stream.athrow(cancellation)

    assert raised.value is cancellation
    generation = _generation_spans(span_exporter)[0]
    assert _messages(generation) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "partial thought"},
                {"type": "text", "content": "partial text"},
            ],
        }
    ]
    assert generation.attributes["gen_ai.response.model"] == "observed-partial-model"
    assert generation.attributes["gen_ai.usage.input_tokens"] == 0
    assert generation.attributes["gen_ai.usage.output_tokens"] == 0
    assert generation.attributes["gen_ai.usage.reasoning.output_tokens"] == 0
    assert generation.attributes["error.type"] == "CancelledError"
    assert generation.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_early_stream_close_marks_generation_interrupted(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from google.adk.models import LlmResponse
    from google.genai import types
    from opentelemetry import trace

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    partial = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text="partial")]),
        partial=True,
    )
    _install_scripted_gemini(monkeypatch, [[partial]])
    stream = GeminiProvider().query(_options(tmp_path, stream=True))

    with trace.get_tracer("gemini-telemetry-tests").start_as_current_span("test invocation"):
        assert await anext(stream) == TextDeltaEvent(text="partial")
        await stream.aclose()

    generation = _generation_spans(span_exporter)[0]
    assert generation.attributes["error.type"] == "generation_interrupted"
    assert generation.status.status_code == StatusCode.ERROR
    assert _messages(generation) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "partial"}]}
    ]
