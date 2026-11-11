"""Native tool evidence is independent of the inspected event projection."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from lightspeed_agentic.providers.deepagents import _create_generation_callback
from lightspeed_agentic.types import ToolResultEvent

from .helpers import make_model_route


@pytest.mark.asyncio
async def test_native_tool_span_waits_for_sanitized_result_handoff(span_exporter: Any) -> None:
    parent = trace.get_tracer("test").start_span("invoke_agent")
    callback = _create_generation_callback(
        make_model_route(Mock()), trace.set_span_in_context(parent), main_name=None
    )
    run_id = uuid4()
    await callback.on_tool_start(
        {"name": "lookup"},
        "{}",
        run_id=run_id,
        tool_call_id="safe-call",
    )
    await callback.on_tool_end("PRIVATE-RAW-OUTPUT", run_id=run_id)

    assert not [s for s in span_exporter.get_finished_spans() if s.name == "execute_tool lookup"]
    callback.record_tool_result(
        "lookup",
        ToolResultEvent(
            output="[REDACTED:email]",
            call_id="safe-call",
            redacted=True,
            redacted_count=1,
            redaction_counts={"email": 1},
        ),
    )
    parent.end()

    span = next(s for s in span_exporter.get_finished_spans() if s.name == "execute_tool lookup")
    assert json.loads(span.attributes["gen_ai.tool.call.result"]) == {"content": "[REDACTED:email]"}
    assert span.attributes["lightspeed.tool_output.redacted"] is True
    assert span.attributes["lightspeed.tool_output.redacted_count"] == 1
    assert "PRIVATE-RAW-OUTPUT" not in str(span.attributes)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "handled_error", "raised_error"])
async def test_native_tool_callback_records_handoffs_and_excludes_raised_errors(
    outcome: str, span_exporter: Any
) -> None:
    from langchain_core.tools import ToolException, tool

    @tool
    def lookup(query: str) -> str:
        """Return a result or report a native execution failure."""
        if outcome != "success":
            raise ToolException("PRIVATE-EXECUTION-ERROR")
        return f"raw result: {query}"

    lookup.handle_tool_error = outcome == "handled_error"
    parent = trace.get_tracer("test").start_span(
        "invoke_agent", attributes={"agenticrun.uid": "native-run"}
    )
    callback = _create_generation_callback(
        make_model_route(Mock()), trace.set_span_in_context(parent), main_name=None
    )
    call = {
        "type": "tool_call",
        "id": "sdk-call-id",
        "name": "lookup",
        "args": {"query": "café"},
    }
    try:
        if outcome == "raised_error":
            with pytest.raises(ToolException):
                await lookup.ainvoke(call, config={"callbacks": [callback]})
        else:
            await lookup.ainvoke(call, config={"callbacks": [callback]})
            callback.record_tool_result(
                "lookup",
                ToolResultEvent(
                    output=(
                        "raw result: café" if outcome == "success" else "[REDACTED:tool_error]"
                    ),
                    call_id="sdk-call-id",
                    error_type="tool_error" if outcome == "handled_error" else None,
                ),
            )
        # Inspection rejection/interruption must not change completed executions.
        callback.close_open("ToolResultSafetyInspectionFailed")
    finally:
        parent.end()

    spans = [s for s in span_exporter.get_finished_spans() if s.name == "execute_tool lookup"]
    assert len(spans) == 1
    span = spans[0]
    assert span.parent.span_id == parent.get_span_context().span_id
    assert span.attributes["agenticrun.uid"] == "native-run"
    assert span.attributes["gen_ai.tool.call.id"] == "sdk-call-id"
    assert json.loads(span.attributes["gen_ai.tool.call.arguments"]) == {"query": "café"}
    if outcome == "success":
        assert json.loads(span.attributes["gen_ai.tool.call.result"]) == {
            "content": "raw result: café"
        }
        assert span.status.status_code == StatusCode.UNSET
        assert "error.type" not in span.attributes
    else:
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == (
            "tool_error" if outcome == "handled_error" else "ToolException"
        )
        if outcome == "handled_error":
            assert json.loads(span.attributes["gen_ai.tool.call.result"]) == {
                "content": "[REDACTED:tool_error]"
            }
        else:
            assert "gen_ai.tool.call.result" not in span.attributes
        assert "PRIVATE-EXECUTION-ERROR" not in str(span.attributes)
        assert not span.events


@pytest.mark.asyncio
async def test_native_command_tool_records_its_result_message(span_exporter: Any) -> None:
    from langchain.agents.middleware.todo import write_todos

    parent = trace.get_tracer("test").start_span("invoke_agent")
    callback = _create_generation_callback(
        make_model_route(Mock()), trace.set_span_in_context(parent), main_name=None
    )
    todos = [{"content": "Check evidence", "status": "in_progress"}]
    result = await write_todos.ainvoke(
        {
            "type": "tool_call",
            "id": "todo-call-id",
            "name": "write_todos",
            "args": {"todos": todos},
        },
        config={"callbacks": [callback]},
    )
    callback.record_tool_result(
        "write_todos",
        ToolResultEvent(
            output="Updated todo list to [{'content': 'Check evidence', 'status': 'in_progress'}]",
            call_id="todo-call-id",
        ),
    )
    parent.end()
    span = next(
        s for s in span_exporter.get_finished_spans() if s.name == "execute_tool write_todos"
    )
    assert json.loads(span.attributes["gen_ai.tool.call.result"]) == {
        "content": "Updated todo list to [{'content': 'Check evidence', 'status': 'in_progress'}]"
    }
    assert span.attributes["gen_ai.tool.call.id"] == "todo-call-id"
    assert span.status.status_code == StatusCode.UNSET
    assert result.update["todos"] == todos


@pytest.mark.asyncio
async def test_native_tool_without_sdk_call_id_does_not_publish_run_uuid(
    span_exporter: Any,
) -> None:
    parent = trace.get_tracer("test").start_span("invoke_agent")
    callback = _create_generation_callback(
        make_model_route(Mock()), trace.set_span_in_context(parent), main_name=None
    )
    run_id = uuid4()
    await callback.on_tool_start({"name": "lookup"}, "", run_id=run_id)
    await callback.on_tool_end("", run_id=run_id)
    callback.record_tool_result("lookup", ToolResultEvent(output="", call_id=""))
    parent.end()

    span = next(s for s in span_exporter.get_finished_spans() if s.name == "execute_tool lookup")
    assert "gen_ai.tool.call.id" not in span.attributes
    assert json.loads(span.attributes["gen_ai.tool.call.arguments"]) == {"content": ""}
    assert json.loads(span.attributes["gen_ai.tool.call.result"]) == {"content": ""}


@pytest.mark.asyncio
async def test_empty_call_id_does_not_match_ambiguous_same_name_tool_spans(
    span_exporter: Any,
) -> None:
    parent = trace.get_tracer("test").start_span("invoke_agent")
    callback = _create_generation_callback(
        make_model_route(Mock()), trace.set_span_in_context(parent), main_name=None
    )
    await callback.on_tool_start({"name": "lookup"}, "{}", run_id=uuid4(), tool_call_id="first")
    await callback.on_tool_start({"name": "lookup"}, "{}", run_id=uuid4(), tool_call_id="second")

    callback.record_tool_result("lookup", ToolResultEvent(output="orphan", call_id=""))
    assert not [s for s in span_exporter.get_finished_spans() if s.name == "execute_tool lookup"]

    callback.close_open("generation_interrupted")
    parent.end()
    spans = [s for s in span_exporter.get_finished_spans() if s.name == "execute_tool lookup"]
    assert len(spans) == 2
    assert all(s.attributes["error.type"] == "missing_tool_result" for s in spans)
    assert all("gen_ai.tool.call.result" not in s.attributes for s in spans)


@pytest.mark.asyncio
async def test_different_call_id_leaves_tool_span_unmatched(span_exporter: Any) -> None:
    parent = trace.get_tracer("test").start_span("invoke_agent")
    callback = _create_generation_callback(
        make_model_route(Mock()), trace.set_span_in_context(parent), main_name=None
    )
    await callback.on_tool_start({"name": "lookup"}, "{}", run_id=uuid4(), tool_call_id="pending")

    callback.record_tool_result("lookup", ToolResultEvent(output="orphan", call_id="other"))
    assert not [s for s in span_exporter.get_finished_spans() if s.name == "execute_tool lookup"]

    callback.close_open("generation_interrupted")
    parent.end()
    span = next(s for s in span_exporter.get_finished_spans() if s.name == "execute_tool lookup")
    assert span.attributes["error.type"] == "missing_tool_result"
    assert "gen_ai.tool.call.result" not in span.attributes


@pytest.mark.asyncio
async def test_handoff_call_id_does_not_match_span_started_without_call_id(
    span_exporter: Any,
) -> None:
    parent = trace.get_tracer("test").start_span("invoke_agent")
    callback = _create_generation_callback(
        make_model_route(Mock()), trace.set_span_in_context(parent), main_name=None
    )
    await callback.on_tool_start({"name": "lookup"}, "{}", run_id=uuid4())

    callback.record_tool_result("lookup", ToolResultEvent(output="orphan", call_id="handoff"))
    assert not [s for s in span_exporter.get_finished_spans() if s.name == "execute_tool lookup"]

    callback.close_open("generation_interrupted")
    parent.end()
    span = next(s for s in span_exporter.get_finished_spans() if s.name == "execute_tool lookup")
    assert "gen_ai.tool.call.id" not in span.attributes
    assert span.attributes["error.type"] == "missing_tool_result"
    assert "gen_ai.tool.call.result" not in span.attributes


@pytest.mark.asyncio
async def test_interruption_closes_only_pending_native_tool_spans(span_exporter: Any) -> None:
    from lightspeed_agentic.metrics import tool_duration

    parent = trace.get_tracer("test").start_span("invoke_agent")
    callback = _create_generation_callback(
        make_model_route(Mock()), trace.set_span_in_context(parent), main_name=None
    )
    run_id = uuid4()
    duration = tool_duration.labels(gen_ai_tool_name="pending_native_lookup")._sum.get()
    await callback.on_tool_start({"name": "pending_native_lookup"}, "{}", run_id=run_id)
    callback.close_open("CancelledError")
    callback.close_open("generation_interrupted")
    parent.end()

    spans = [s for s in span_exporter.get_finished_spans() if s.name.startswith("execute_tool")]
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes["error.type"] == "missing_tool_result"
    assert "gen_ai.tool.call.result" not in spans[0].attributes
    assert tool_duration.labels(gen_ai_tool_name="pending_native_lookup")._sum.get() == duration
