"""Contracts for invocation-wide token usage collection (OLS-4921).

Task 1 characterizes the existing per-query generation callback and pins the
future collector/snapshot interfaces before implementation. These tests are
expected to fail until the invocation usage collector exists.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import pytest

from .helpers import make_model_route

_MODEL = "claude-sonnet-4-6"


def _usage(
    input_tokens: int,
    output_tokens: int,
    *,
    output_details: dict[str, int] | None = None,
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if output_details is not None:
        usage["output_token_details"] = output_details
    return usage


def _chat_result(
    *,
    usage: dict[str, Any] | None = None,
    model_name: str | None = _MODEL,
    text: str = "answer",
) -> Any:
    """Build the LLMResult shape LangChain passes to ``on_llm_end``."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    response_metadata = {} if model_name is None else {"model_name": model_name}
    message = AIMessage(
        content=text,
        usage_metadata=usage,
        response_metadata=response_metadata,
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def _model_route() -> Any:
    return make_model_route(lambda _params, _config: None, model=_MODEL)


def _collector() -> Any:
    from lightspeed_agentic.usage import InvocationUsage

    return InvocationUsage(route_model=_MODEL)


def test_invocation_collector_sums_input_and_output_across_calls() -> None:
    collector = _collector()

    collector.on_llm_end(_chat_result(usage=_usage(100, 20)))
    collector.on_llm_end(_chat_result(usage=_usage(150, 30)))

    assert collector.input_tokens == 250
    assert collector.output_tokens == 50
    assert collector.usage_call_count == 2
    assert collector.observed_model == _MODEL


def test_invocation_collector_records_partial_usage_from_llm_error() -> None:
    collector = _collector()

    collector.on_llm_error(
        RuntimeError("boom"),
        run_id=__import__("uuid").uuid4(),
        response=_chat_result(usage=_usage(40, 4)),
    )

    assert collector.input_tokens == 40
    assert collector.output_tokens == 4
    assert collector.usage_call_count == 1


def test_invocation_collector_counts_run_once_across_end_and_error() -> None:
    collector = _collector()
    run_id = __import__("uuid").uuid4()
    response = _chat_result(usage=_usage(40, 4))

    collector.on_llm_end(response, run_id=run_id)
    collector.on_llm_error(RuntimeError("boom"), run_id=run_id, response=response)

    assert collector.input_tokens == 40
    assert collector.output_tokens == 4
    assert collector.usage_call_count == 1


def test_invocation_collector_keeps_usage_without_response_model_name() -> None:
    collector = _collector()

    collector.on_llm_end(_chat_result(usage=_usage(80, 10), model_name=None))

    assert collector.input_tokens == 80
    assert collector.output_tokens == 10
    assert collector.usage_call_count == 1
    assert collector.observed_model == _MODEL


@pytest.mark.parametrize(
    ("detail_key", "reasoning"),
    [
        ("reasoning", 7),
        ("flex_reasoning", 7),
        ("priority_reasoning", 7),
    ],
)
def test_invocation_collector_sums_reasoning_detail_suffixes(
    detail_key: str,
    reasoning: int,
) -> None:
    collector = _collector()

    collector.on_llm_end(
        _chat_result(
            usage=_usage(100, 40, output_details={detail_key: reasoning, "audio": 9}),
        )
    )

    assert collector.reasoning_tokens == reasoning
    # Output remains the provider total; reasoning is not subtracted from it.
    assert collector.output_tokens == 40


@pytest.mark.parametrize(
    ("usages", "expected_reasoning", "expected_calls"),
    [
        pytest.param([_usage(50, 5, output_details={"reasoning": 0})], 0, 1, id="zero"),
        pytest.param(
            [_usage(50, 5, output_details={"reasoning": 3}), _usage(60, 6)],
            None,
            2,
            id="partial",
        ),
        pytest.param([_usage(120, 15)], None, 1, id="provider-gap"),
        pytest.param([None], None, 0, id="no-usage"),
    ],
)
def test_invocation_collector_reasoning_coverage(
    usages: list[dict[str, Any] | None],
    expected_reasoning: int | None,
    expected_calls: int,
) -> None:
    collector = _collector()

    for usage in usages:
        collector.on_llm_end(_chat_result(usage=usage))

    assert collector.reasoning_tokens == expected_reasoning
    assert collector.usage_call_count == expected_calls


@contextmanager
def _generation_callback(*, usage: Any = None) -> Any:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind

    from lightspeed_agentic.providers.deepagents import _create_generation_callback

    parent = trace.get_tracer("test").start_span("invoke_agent", kind=SpanKind.INTERNAL)
    try:
        yield _create_generation_callback(
            _model_route(),
            trace.set_span_in_context(parent),
            main_name="main",
            usage=usage,
        )
    finally:
        parent.end()


async def _run_main_call(
    callback: Any,
    *,
    messages: list[Any],
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    model_name: str | None = _MODEL,
) -> None:
    run_id = __import__("uuid").uuid4()
    await callback.on_chat_model_start(
        {},
        [messages],
        run_id=run_id,
        tags=tags,
        metadata=metadata if metadata is not None else {"lc_agent_name": "main"},
    )
    await callback.on_llm_end(
        _chat_result(usage=usage, model_name=model_name),
        run_id=run_id,
    )


@pytest.mark.asyncio
async def test_main_callback_retains_messages_and_pairs_observed_usage() -> None:
    from langchain_core.messages import HumanMessage

    with _generation_callback() as callback:
        await _run_main_call(
            callback,
            messages=[HumanMessage(content="hello")],
            usage=_usage(100, 20, output_details={"reasoning": 5}),
        )

    assert len(callback.main_calls) == 1
    call = callback.main_calls[0]
    assert [message.content for message in call.messages] == ["hello"]
    assert call.input_tokens == 100
    assert call.output_tokens == 20
    assert call.reasoning_tokens == 5
    assert call.model == _MODEL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "tags"),
    [
        ({"lc_agent_name": "main", "lc_source": "summarization"}, None),
        ({"lc_agent_name": "researcher"}, None),
        ({"lc_agent_name": "main"}, ["nostream"]),
        ({"lc_agent_name": "main"}, ["structured_output"]),
        ({"lc_agent_name": "main"}, ["nostream", "structured_output"]),
    ],
    ids=["summarization", "subagent", "classifier", "structured_output", "classifier+shape"],
)
async def test_main_callback_excludes_non_main_calls_from_estimates(
    metadata: dict[str, Any],
    tags: list[str] | None,
) -> None:
    from langchain_core.messages import HumanMessage

    with _generation_callback() as callback:
        await _run_main_call(
            callback,
            messages=[HumanMessage(content="excluded")],
            tags=tags,
            metadata=metadata,
            usage=_usage(500, 50),
        )

    assert callback.main_calls == []


def _usage_logs(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage() for record in caplog.records if "provider:usage" in record.getMessage()
    ]


@pytest.mark.asyncio
async def test_main_call_logs_one_turn_with_observed_usage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from langchain_core.messages import HumanMessage

    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    with _generation_callback() as callback:
        await _run_main_call(
            callback,
            messages=[HumanMessage(content="secret prompt")],
            usage=_usage(100, 20, output_details={"reasoning": 5}),
        )

    logs = _usage_logs(caplog)
    assert len(logs) == 1
    assert "input_tokens=100" in logs[0]
    assert "output_tokens=20" in logs[0]
    assert "reasoning_tokens=5" in logs[0]
    assert f"model={_MODEL}" in logs[0]
    assert "secret prompt" not in logs[0]


@pytest.mark.asyncio
async def test_main_call_logs_not_reported_instead_of_fabricated_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from langchain_core.messages import HumanMessage

    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    with _generation_callback() as callback:
        await _run_main_call(
            callback,
            messages=[HumanMessage(content="hello")],
            usage=None,
        )

    logs = _usage_logs(caplog)
    assert len(logs) == 1
    assert "input_tokens=not_reported" in logs[0]
    assert "output_tokens=not_reported" in logs[0]
    assert "reasoning_tokens=not_reported" in logs[0]


@pytest.mark.asyncio
async def test_main_call_logs_partial_usage_on_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from langchain_core.messages import HumanMessage

    collector = _collector()
    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    with _generation_callback(usage=collector) as callback:
        run_id = __import__("uuid").uuid4()
        await callback.on_chat_model_start(
            {},
            [[HumanMessage(content="hello")]],
            run_id=run_id,
            metadata={"lc_agent_name": "main"},
        )
        await callback.on_llm_error(
            RuntimeError("boom"),
            run_id=run_id,
            response=_chat_result(usage=_usage(40, 4)),
        )

    logs = _usage_logs(caplog)
    assert len(logs) == 1
    assert "input_tokens=40" in logs[0]
    assert "output_tokens=4" in logs[0]
    assert "partial=true" in logs[0]
    assert "error_type=RuntimeError" in logs[0]
    assert "boom" not in logs[0]
    assert collector.input_tokens == 40
    assert collector.output_tokens == 4
    assert collector.usage_call_count == 1


@pytest.mark.asyncio
async def test_excluded_calls_log_nothing(caplog: pytest.LogCaptureFixture) -> None:
    from langchain_core.messages import HumanMessage

    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    with _generation_callback() as callback:
        for metadata, tags in [
            ({"lc_agent_name": "main", "lc_source": "summarization"}, None),
            ({"lc_agent_name": "researcher"}, None),
            ({"lc_agent_name": "main"}, ["nostream"]),
            ({"lc_agent_name": "main"}, ["structured_output"]),
        ]:
            await _run_main_call(
                callback,
                messages=[HumanMessage(content="excluded")],
                tags=tags,
                metadata=metadata,
                usage=_usage(500, 50),
            )

    assert _usage_logs(caplog) == []


@pytest.mark.asyncio
async def test_structured_output_shaping_joins_invocation_totals() -> None:
    """Shaping calls carry the collector and an explicit exclusion tag."""
    from unittest.mock import AsyncMock, Mock

    from langchain_core.messages import AIMessage

    from lightspeed_agentic.providers.deepagents import _shape_structured_output
    from lightspeed_agentic.usage import InvocationUsage

    captured: dict[str, Any] = {}

    async def ainvoke(_messages: Any, config: dict[str, Any]) -> dict[str, Any]:
        captured.update(config)
        for callback in config["callbacks"]:
            outcome = callback.on_llm_end(_chat_result(usage=_usage(30, 6)))
            if outcome is not None:
                await outcome
        return {
            "parsed": {"ok": True},
            "raw": AIMessage(content='{"ok": true}'),
            "parsing_error": None,
        }

    bound = Mock()
    bound.ainvoke = ainvoke
    model = Mock()
    model.with_structured_output.return_value = bound
    generation_callback = Mock()
    generation_callback.on_llm_end = AsyncMock()
    collector = InvocationUsage(route_model=_MODEL)

    parsed = await _shape_structured_output(
        make_model_route(lambda _params, _config: model, model=_MODEL),
        {"type": "object", "properties": {"ok": {"type": "boolean"}}},
        "sys",
        "ask",
        "not JSON",
        generation_callback,
        usage=collector,
    )

    assert parsed == {"ok": True}
    assert "structured_output" in captured["tags"]
    assert collector in captured["callbacks"]
    assert collector.input_tokens == 30
    assert collector.output_tokens == 6
    assert collector.usage_call_count == 1
    generation_callback.on_llm_end.assert_awaited_once()


@pytest.mark.asyncio
async def test_classifier_call_is_counted_through_inherited_callbacks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The real classifier client inherits the run's collector via ambient config."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.runnables.config import var_child_runnable_config

    from lightspeed_agentic.inspection.client import LangChainClassifierClient
    from lightspeed_agentic.inspection.models import ClassifierRequest
    from lightspeed_agentic.usage import InvocationUsage

    decision_text = '{"injectionDetected": false, "category": "none"}'
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content=decision_text, usage_metadata=_usage(70, 9))])
    )
    client = LangChainClassifierClient(model)
    collector = InvocationUsage(route_model=_MODEL)

    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    token = var_child_runnable_config.set({"callbacks": [collector], "metadata": {}, "tags": []})
    try:
        decision = await client.classify(
            ClassifierRequest(
                toolName="execute",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="hello",
            )
        )
    finally:
        var_child_runnable_config.reset(token)

    assert decision.injection_detected is False
    assert collector.usage_call_count == 1
    assert collector.input_tokens == 70
    assert collector.output_tokens == 9
    # The nostream classifier call is never a main-agent turn.
    assert _usage_logs(caplog) == []


def _ai_tool_call(name: str = "read_file", *, args: dict[str, Any] | None = None) -> Any:
    from langchain_core.messages import AIMessage

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args if args is not None else {"file_path": "/skills/demo/SKILL.md"},
                "id": "call-1",
            }
        ],
    )


def _tool_result(content: str) -> Any:
    from langchain_core.messages import ToolMessage

    return ToolMessage(content=content, tool_call_id="call-1", name="read_file")


def _human(content: str) -> Any:
    from langchain_core.messages import HumanMessage

    return HumanMessage(content=content)


async def _run_pairs(
    callback: Any,
    calls: list[tuple[list[Any], dict[str, Any]]],
) -> None:
    from langchain_core.messages import HumanMessage

    for index, (appended, usage) in enumerate(calls):
        messages = [HumanMessage(content="go")]
        for previous_appended, _ in calls[:index]:
            messages.extend(previous_appended)
        messages.extend(appended)
        await _run_main_call(callback, messages=messages, usage=usage)


def _snapshot(
    messages: list[Any],
    *,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int | None,
) -> Any:
    from lightspeed_agentic.usage import MainCallSnapshot

    return MainCallSnapshot(
        messages=tuple(messages),
        model=_MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


@pytest.mark.parametrize(
    ("appended", "reasoning", "current_input", "rewrite", "expected"),
    [
        pytest.param([_ai_tool_call(), _tool_result("tool output")], 0, 180, False, 60, id="tool"),
        pytest.param(
            [_ai_tool_call(), _tool_result("tool output")],
            None,
            180,
            False,
            None,
            id="no-reasoning",
        ),
        pytest.param([_ai_tool_call(), _tool_result("tool output")], 0, 180, True, 0, id="rewrite"),
        pytest.param([_ai_tool_call()], 0, 180, False, 0, id="no-tool-result"),
        pytest.param([_tool_result("orphan")], 0, 180, False, 0, id="orphan-tool"),
        pytest.param(
            [_ai_tool_call(), _tool_result("t"), _tool_result("t2")],
            0,
            180,
            False,
            60,
            id="two-tools",
        ),
        pytest.param(
            [_ai_tool_call(), _tool_result("t"), _human("hi")],
            0,
            180,
            False,
            0,
            id="non-tool-suffix",
        ),
        pytest.param(
            [_ai_tool_call(), _tool_result("tiny")], 0, 120, False, None, id="negative-delta"
        ),
    ],
)
def test_tool_token_estimate(
    appended: list[Any],
    reasoning: int | None,
    current_input: int,
    rewrite: bool,
    expected: int | None,
) -> None:
    from lightspeed_agentic.usage import tool_tokens_from_main_calls

    previous_messages = [_human("go")]
    current_messages = ([_human("rewritten")] if rewrite else previous_messages) + appended
    calls = [
        _snapshot(
            previous_messages,
            input_tokens=100,
            output_tokens=20 if current_input > 120 else 40,
            reasoning_tokens=reasoning,
        ),
        _snapshot(
            current_messages, input_tokens=current_input, output_tokens=10, reasoning_tokens=None
        ),
    ]

    assert tool_tokens_from_main_calls(calls) == expected


def test_peak_input_tokens_track_largest_main_call() -> None:
    from lightspeed_agentic.usage import peak_input_tokens_from_main_calls

    calls = [
        _snapshot([], input_tokens=value, output_tokens=0, reasoning_tokens=None)
        for value in (100, 180, 120)
    ]

    assert peak_input_tokens_from_main_calls(calls) == 180


def test_estimates_are_absent_without_main_calls() -> None:
    with _generation_callback() as callback:
        pass

    assert callback.tool_tokens is None
    assert callback.peak_input_tokens is None


@pytest.mark.asyncio
async def test_estimates_ignore_non_main_calls() -> None:
    from langchain_core.messages import HumanMessage

    with _generation_callback() as callback:
        await _run_main_call(
            callback,
            messages=[HumanMessage(content="go")],
            usage=_usage(100, 20),
        )
        await _run_main_call(
            callback,
            messages=[HumanMessage(content="summarize"), _tool_result("big")],
            metadata={"lc_agent_name": "main", "lc_source": "summarization"},
            usage=_usage(900, 50),
        )
        await _run_main_call(
            callback,
            messages=[HumanMessage(content="classify")],
            tags=["nostream"],
            usage=_usage(700, 50),
        )
        await _run_main_call(
            callback,
            messages=[HumanMessage(content="shape")],
            tags=["structured_output"],
            usage=_usage(800, 50),
        )

    assert callback.peak_input_tokens == 100
    assert callback.tool_tokens == 0


@pytest.mark.asyncio
async def test_skill_reads_are_included_in_tool_tokens_without_a_skill_field() -> None:
    skill_content = "---\nname: demo\n---\n" + "skill body " * 50

    with _generation_callback() as callback:
        await _run_pairs(
            callback,
            [
                ([], _usage(100, 20, output_details={"reasoning": 0})),
                (
                    [
                        _ai_tool_call(args={"file_path": "/skills/demo/SKILL.md"}),
                        _tool_result(skill_content),
                    ],
                    _usage(400, 10),
                ),
            ],
        )

    assert callback.tool_tokens == 280
    assert not hasattr(callback, "skill_tokens")


def _summary(calls: list[Any], collector: Any) -> Any:
    from lightspeed_agentic.usage import summarize_usage

    return summarize_usage(collector, calls, _MODEL)


def test_summarize_usage_reports_collector_totals() -> None:
    from lightspeed_agentic.usage import MainCallSnapshot

    collector = _collector()
    collector.on_llm_end(_chat_result(usage=_usage(100, 20, output_details={"reasoning": 5})))
    calls = [
        MainCallSnapshot(
            messages=(),
            model=_MODEL,
            input_tokens=100,
            output_tokens=20,
            reasoning_tokens=5,
        )
    ]

    summary = _summary(calls, collector)

    assert summary.input_tokens == 100
    assert summary.output_tokens == 20
    assert summary.reasoning_tokens == 5
    assert summary.peak_input_tokens == 100
    assert summary.model == _MODEL


def test_summarize_usage_falls_back_to_the_route_model_without_calls() -> None:
    summary = _summary([], _collector())

    assert summary.model == _MODEL
    assert summary.input_tokens is None
    assert summary.output_tokens is None
    assert summary.reasoning_tokens is None
    assert summary.tool_tokens is None
    assert summary.peak_input_tokens is None


def test_summarize_usage_omits_partial_reasoning() -> None:
    from lightspeed_agentic.usage import MainCallSnapshot

    collector = _collector()
    collector.on_llm_end(_chat_result(usage=_usage(100, 20, output_details={"reasoning": 5})))
    collector.on_llm_end(_chat_result(usage=_usage(30, 5)))
    calls = [
        MainCallSnapshot(
            messages=(), model=_MODEL, input_tokens=100, output_tokens=20, reasoning_tokens=5
        )
    ]

    assert _summary(calls, collector).reasoning_tokens is None
