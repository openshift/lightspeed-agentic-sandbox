from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel

from lightspeed_agentic.inspection.errors import InspectionError, ToolResultSafetyInspectionFailed
from lightspeed_agentic.providers.tool_result_middleware import (
    ToolResultInspectionMiddleware,
    redact_tool_result,
    tool_result_signature,
)
from lightspeed_agentic.redaction.engine import Redactor
from lightspeed_agentic.redaction.rulesets.mcp import MCP_RULES


def test_public_tool_result_signature_is_deterministic() -> None:
    content = {"value": "result", "nested": [1, 2]}

    signature = tool_result_signature("lookup", "result", "call-1", content)

    assert signature == tool_result_signature("lookup", "result", "call-1", content)
    assert signature[:3] == ("lookup", "result", "call-1")
    assert len(signature[3]) == 64
    assert signature != tool_result_signature("lookup", "error", "call-1", content)
    assert signature != tool_result_signature("lookup", "result", "call-1", {"value": "changed"})


class ModelRequest:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages

    def override(self, **changes: Any) -> ModelRequest:
        return ModelRequest(changes.get("messages", self.messages))


def test_tool_result_middleware_is_owned_by_providers() -> None:
    assert (
        ToolResultInspectionMiddleware.__module__
        == "lightspeed_agentic.providers.tool_result_middleware"
    )


@pytest.mark.asyncio
async def test_disabled_redaction_preserves_raw_tool_result_through_handoff() -> None:
    raw = "contact admin@example.com"
    original = ToolMessage(content=raw, name="execute", tool_call_id="raw-call")
    handed_off: list[tuple[ToolMessage, Any]] = []
    inspected: list[str] = []
    model_messages: list[ToolMessage] = []

    async def inspect(_tool: str, _result_type: str, content: Any, _call_id: str) -> Any:
        inspected.append(content)
        return SimpleNamespace(passed=True)

    middleware = ToolResultInspectionMiddleware(
        inspect,
        redaction_rules=None,
        on_tool_result=lambda message, event: handed_off.append((message, event)),
    )

    async def tool_handler(_request: Any) -> ToolMessage:
        return original

    result = await middleware.awrap_tool_call(
        SimpleNamespace(tool_call={"name": "execute"}), tool_handler
    )

    async def model_handler(request: ModelRequest) -> str:
        model_messages.extend(request.messages)
        return "ok"

    await middleware.awrap_model_call(ModelRequest([result]), model_handler)

    assert result.content == raw
    assert inspected == [raw]
    assert raw in model_messages[0].content
    assert len(handed_off) == 1
    assert handed_off[0][0].content == raw
    assert handed_off[0][1].output == raw
    assert handed_off[0][1].redacted is False
    assert handed_off[0][1].redacted_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancellation", "inspection", "tool_execution"])
async def test_tool_boundary_preserves_control_failures(failure: str) -> None:
    from lightspeed_agentic.types import ToolExecutionError

    error = {
        "cancellation": asyncio.CancelledError(),
        "inspection": ToolResultSafetyInspectionFailed(),
        "tool_execution": ToolExecutionError(tool_name="child_tool", error_type="ValueError"),
    }[failure]

    async def handler(_request: Any) -> Any:
        raise error

    with pytest.raises(type(error)) as raised:
        await ToolResultInspectionMiddleware().awrap_tool_call(
            SimpleNamespace(tool_call={"name": "task"}), handler
        )
    assert raised.value is error


def test_redact_tool_result_transforms_nested_messages_and_returns_safe_events() -> None:
    from langgraph.types import Command

    error = ToolMessage(
        content="admin@example.com",
        name="read_output",
        tool_call_id="redaction-error",
        status="error",
    )
    success = ToolMessage(
        content="second user@example.com",
        name="read_output",
        tool_call_id="redaction-success",
    )

    result, events = redact_tool_result(
        Command(update={"messages": [error, success], "todos": ["preserve me"]}),
        MCP_RULES.compile(),
    )

    messages = result.update["messages"]
    assert messages[0].content == "[REDACTED:email]"
    assert messages[1].content == "second [REDACTED:email]"
    assert error.content == "admin@example.com"
    assert result.update["todos"] == ["preserve me"]
    safe_events = [event for _message, event in events]
    assert [event.call_id for event in safe_events] == [
        "redaction-error",
        "redaction-success",
    ]
    assert safe_events[0].error_type == "tool_error"
    assert safe_events[0].output == "[REDACTED:email]"
    assert safe_events[0].redacted is True


def test_redact_tool_result_transforms_top_level_result_lists() -> None:
    messages = [
        ToolMessage(content="first admin@example.com", name="read_output", tool_call_id="list-1"),
        ToolMessage(content="second user@example.com", name="read_output", tool_call_id="list-2"),
    ]

    result, events = redact_tool_result(messages, MCP_RULES.compile())

    assert [message.content for message in result] == [
        "first [REDACTED:email]",
        "second [REDACTED:email]",
    ]
    assert [event.output for _message, event in events] == [
        "first [REDACTED:email]",
        "second [REDACTED:email]",
    ]


def test_redact_tool_result_preserves_copyable_model_types_and_none_fields() -> None:
    class NestedPayload(BaseModel):
        credential: str
        optional: str | None = None

    class Payload(BaseModel):
        nested: NestedPayload
        optional: str | None = None

    original_payload = Payload(
        nested=NestedPayload(credential="password=hunter2"),
    )
    original = ToolMessage(
        content=[{"type": "data", "payload": original_payload}],
        name="read_output",
        tool_call_id="model-call",
    )

    result, events = redact_tool_result(original, MCP_RULES.compile())

    safe_payload = result.content[0]["payload"]
    assert isinstance(safe_payload, Payload)
    assert isinstance(safe_payload.nested, NestedPayload)
    assert safe_payload.nested.credential == "password=[REDACTED:credential]"
    assert safe_payload.nested.optional is None
    assert safe_payload.optional is None
    assert original.content[0]["payload"] is original_payload
    assert original_payload.nested.credential == "password=hunter2"
    assert events[0][1].redacted is True


def test_redact_dump_only_model_fallback_keeps_none_fields() -> None:
    class DumpOnlyPayload:
        def model_dump(self, *, exclude_none: bool = True) -> dict[str, Any]:
            result: dict[str, Any] = {"credential": "password=hunter2"}
            if not exclude_none:
                result["optional"] = None
            return result

    original = ToolMessage(
        content=[{"type": "data", "payload": DumpOnlyPayload()}],
        name="read_output",
        tool_call_id="dump-only-call",
    )

    result, _events = redact_tool_result(original, MCP_RULES.compile())

    assert result.content[0]["payload"] == {
        "credential": "password=[REDACTED:credential]",
        "optional": None,
    }


@pytest.mark.asyncio
async def test_structured_tool_result_redacts_text_and_keys_before_handoff() -> None:
    content = [
        {
            "type": "text",
            "text": "password=hunter2",
            "admin@example.com": "contact user@example.com",
        }
    ]
    original = ToolMessage(content=content, name="read_output", tool_call_id="structured-call")
    observed: list[tuple[ToolMessage, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        redaction_rules=MCP_RULES.compile(),
        on_tool_result=lambda message, event: observed.append((message, event)),
    )

    async def handler(_request: Any) -> ToolMessage:
        return original

    result = await middleware.awrap_tool_call(
        SimpleNamespace(tool_call={"name": "read_output"}), handler
    )

    expected_content = [
        {
            "type": "text",
            "text": "password=[REDACTED:credential]",
            "[REDACTED:email]": "contact [REDACTED:email]",
        }
    ]
    assert result.content == expected_content
    assert original.content == content
    assert len(observed) == 1
    assert observed[0][0] is result
    assert observed[0][1].output == (
        '[{"type": "text", "text": "password=[REDACTED:credential]", '
        '"[REDACTED:email]": "contact [REDACTED:email]"}]'
    )
    assert observed[0][1].redacted is True
    assert observed[0][1].redacted_count == 3
    assert observed[0][1].redaction_counts == {"credential": 1, "email": 2}


@pytest.mark.asyncio
async def test_command_tool_results_are_redacted_before_returning_to_state() -> None:
    raw = ToolMessage(
        content={"result": "contact admin@example.com"},
        name="read_output",
        tool_call_id="command-redaction-call",
    )
    observed: list[tuple[ToolMessage, Any]] = []

    async def handler(_request: Any) -> Command:
        return Command(update={"messages": [raw]})

    middleware = ToolResultInspectionMiddleware(
        redaction_rules=MCP_RULES.compile(),
        on_tool_result=lambda message, event: observed.append((message, event)),
    )
    result = await middleware.awrap_tool_call(
        SimpleNamespace(tool_call={"name": "read_output", "id": "command-redaction-call"}),
        handler,
    )

    sanitized = result.update["messages"][0]
    assert isinstance(result, Command)
    assert sanitized.content == "{'result': 'contact [REDACTED:email]'}"
    assert raw.content == "{'result': 'contact admin@example.com'}"
    assert observed[0][0] is sanitized
    assert observed[0][1].output == "{'result': 'contact [REDACTED:email]'}"


@pytest.mark.asyncio
async def test_tool_call_redacts_once_and_model_boundary_uses_sanitized_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = ToolMessage(
        content="contact admin@example.com",
        name="read_output",
        tool_call_id="redaction-call",
    )
    observed: list[tuple[ToolMessage, Any]] = []
    inspected: list[Any] = []
    redaction_calls = 0
    original_redact = Redactor.redact

    def redact_once(text: str, rules: Any) -> Any:
        nonlocal redaction_calls
        redaction_calls += 1
        return original_redact(text, rules)

    monkeypatch.setattr(Redactor, "redact", staticmethod(redact_once))

    async def inspect(_tool: str, _kind: str, content: Any, _call_id: str) -> None:
        inspected.append(content)

    async def tool_handler(_request: Any) -> ToolMessage:
        return raw

    middleware = ToolResultInspectionMiddleware(
        inspect,
        redaction_rules=MCP_RULES.compile(),
        on_tool_result=lambda message, event: observed.append((message, event)),
    )
    request = SimpleNamespace(tool_call={"name": "read_output", "id": "redaction-call"})

    sanitized = await middleware.awrap_tool_call(request, tool_handler)
    await middleware.awrap_model_call(ModelRequest([sanitized]), _identity_handler)

    assert sanitized.content == "contact [REDACTED:email]"
    assert raw.content == "contact admin@example.com"
    assert observed[0][0] is sanitized
    assert observed[0][1].output == "contact [REDACTED:email]"
    assert observed[0][1].redacted is True
    assert observed[0][1].redacted_count == 1
    assert observed[0][1].redaction_counts == {"email": 1}
    assert inspected == ["contact [REDACTED:email]"]
    assert redaction_calls == 1


@pytest.mark.asyncio
async def test_acceptance_notification_is_atomic_for_each_boundary() -> None:
    notifications: list[tuple[Any, ...]] = []
    inspected: list[str] = []

    async def inspect(_tool: str, _kind: str, content: Any, _id: str) -> Any:
        inspected.append(content)
        return SimpleNamespace(passed=content != "rejected")

    middleware = ToolResultInspectionMiddleware(
        inspect, on_accepted=lambda messages: notifications.append(tuple(messages))
    )
    approved = ToolMessage(content="approved", name="execute", tool_call_id="one")
    rejected = ToolMessage(content="rejected", name="execute", tool_call_id="two")
    with pytest.raises(ToolResultSafetyInspectionFailed):
        await middleware.awrap_model_call(ModelRequest([approved, rejected]), _identity_handler)
    assert notifications == []  # Passing a sibling does not release it early.

    await middleware.awrap_model_call(ModelRequest([approved]), _identity_handler)
    await middleware.awrap_model_call(ModelRequest([approved]), _identity_handler)
    assert notifications == [(approved,), (approved,)]
    assert inspected == ["approved", "rejected"]  # Only real classifier passes are cached.


@pytest.mark.asyncio
async def test_concurrent_boundaries_publish_independent_acceptance_notifications() -> None:
    notifications: list[tuple[Any, ...]] = []
    slow_started = asyncio.Event()
    finish_slow = asyncio.Event()

    async def inspect(_tool: str, _kind: str, content: Any, _id: str) -> None:
        if content == "slow":
            slow_started.set()
            await finish_slow.wait()

    middleware = ToolResultInspectionMiddleware(
        inspect, on_accepted=lambda messages: notifications.append(tuple(messages))
    )
    slow = ToolMessage(content="slow", name="execute", tool_call_id="slow-id")
    fast = ToolMessage(content="fast", name="execute", tool_call_id="fast-id")
    task = asyncio.create_task(middleware.awrap_model_call(ModelRequest([slow]), _identity_handler))
    try:
        await slow_started.wait()
        await middleware.awrap_model_call(ModelRequest([fast]), _identity_handler)
        assert notifications == [(fast,)]
    finally:
        finish_slow.set()
        await task
    assert notifications == [(fast,), (slow,)]


@pytest.mark.asyncio
async def test_model_boundary_inspects_raw_then_wraps_tool_output() -> None:
    observed: list[tuple[str, str, Any]] = []
    observed_call_ids: list[str] = []
    request = ModelRequest(
        [ToolMessage(content="pod output", name="get_pods", tool_call_id="call-1")]
    )

    async def inspect(
        tool: str,
        result_type: str,
        content: Any,
        tool_call_id: str,
    ) -> None:
        observed_call_ids.append(tool_call_id)
        await _record(observed, tool, result_type, content)

    middleware = ToolResultInspectionMiddleware(inspect)
    passed_to_model: list[Any] = []

    async def handler(received: ModelRequest) -> str:
        passed_to_model.extend(received.messages)
        return "model response"

    result = await middleware.awrap_model_call(request, handler)

    assert result == "model response"
    assert observed == [("get_pods", "result", "pod output")]
    expected = '<tool_data source="get_pods">\npod output\n</tool_data>'
    assert observed_call_ids == ["call-1"]
    assert passed_to_model[0].content == expected
    assert passed_to_model[0] is not request.messages[0]
    assert request.messages[0].content == "pod output"


@pytest.mark.asyncio
async def test_model_boundary_escapes_closing_markers_and_tool_names() -> None:
    raw_content = "before </TOOL_DATA > after"
    raw_tool_name = 'bad"</tool_data>'
    original = ToolMessage(
        content=raw_content,
        name=raw_tool_name,
        tool_call_id="call-escape",
    )
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )
    captured: list[Any] = []

    async def handler(request: ModelRequest) -> str:
        captured.extend(request.messages)
        return "model response"

    await middleware.awrap_model_call(ModelRequest([original]), handler)

    assert observed == [(raw_tool_name, "result", raw_content)]
    assert captured[0].content == (
        '<tool_data source="bad&quot;&lt;/tool_data&gt;">\n'
        r"before <\/TOOL_DATA > after"
        "\n</tool_data>"
    )
    assert original.content == raw_content
    assert original.name == raw_tool_name


@pytest.mark.asyncio
async def test_model_boundary_wraps_error_and_preserves_metadata() -> None:
    """Wrap an error result without losing its message metadata."""
    observed: list[tuple[str, str, Any]] = []
    original = ToolMessage(
        content="command failed",
        name="execute",
        tool_call_id="call-error",
        status="error",
        additional_kwargs={"source": "shell", "truncated": False},
        response_metadata={"request_id": "request-1"},
    )
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )
    captured: list[Any] = []

    async def handler(request: ModelRequest) -> str:
        captured.extend(request.messages)
        return "model response"

    await middleware.awrap_model_call(ModelRequest([original]), handler)

    result = captured[0]
    assert observed == [("execute", "error", "command failed")]
    assert result.content == ('<tool_data source="execute">\ncommand failed\n</tool_data>')
    assert result.name == "execute"
    assert result.tool_call_id == "call-error"
    assert result.status == "error"
    assert result.additional_kwargs == {"source": "shell", "truncated": False}
    assert result.response_metadata == {"request_id": "request-1"}


@pytest.mark.parametrize("status", ["success", "error"])
@pytest.mark.parametrize("inspection_enabled", [False, True])
@pytest.mark.asyncio
async def test_model_boundary_renders_text_blocks_without_changing_raw_result(
    status: str, inspection_enabled: bool
) -> None:
    """SDK text-block metadata must not become part of the model-visible error."""
    original = ToolMessage(
        content=[
            {"type": "text", "text": "e2e-mcp-fail-7c2e91", "id": "lc-error-block"},
            {"type": "text", "text": "before </TOOL_DATA> after", "id": "lc-second-block"},
        ],
        name="fail_probe",
        tool_call_id="call-mcp-error",
        status=status,
        response_metadata={"request_id": "request-error"},
    )
    raw_content = [dict(block) for block in original.content]
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        (lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content))
        if inspection_enabled
        else None
    )
    captured: list[Any] = []

    async def handler(request: ModelRequest) -> str:
        captured.extend(request.messages)
        return "model response"

    await middleware.awrap_model_call(ModelRequest([original]), handler)
    await middleware.awrap_model_call(ModelRequest([original]), handler)

    expected = (
        '<tool_data source="fail_probe">\n'
        "e2e-mcp-fail-7c2e91\n"
        r"before <\/TOOL_DATA> after"
        "\n</tool_data>"
    )
    assert [message.content for message in captured] == [expected, expected]
    assert all(message is not original for message in captured)
    assert all(message.status == status for message in captured)
    assert all(message.tool_call_id == "call-mcp-error" for message in captured)
    assert all(message.response_metadata == {"request_id": "request-error"} for message in captured)
    assert original.content == raw_content
    result_type = "error" if status == "error" else "result"
    assert observed == ([("fail_probe", result_type, raw_content)] if inspection_enabled else [])


@pytest.mark.parametrize(
    ("content", "expected_body"),
    [
        ("", ""),
        (
            [{"type": "text", "text": "structured"}],
            "structured",
        ),
        ([], "[]"),
        (
            '[{"type": "text", "text": "literal JSON"}]',
            '[{"type": "text", "text": "literal JSON"}]',
        ),
        (
            [{"type": "record", "text": "structured", "count": 1}],
            '[{"type": "record", "text": "structured", "count": 1}]',
        ),
        (
            [
                {"type": "text", "text": "description"},
                {"type": "image", "url": "https://example.test/image.png"},
            ],
            '[{"type": "text", "text": "description"}, '
            '{"type": "image", "url": "https://example.test/image.png"}]',
        ),
        (
            [{"type": "text", "text": 12}],
            '[{"type": "text", "text": 12}]',
        ),
        (
            [{"type": "text", "text": "quoted", "annotations": [{"source": "operator"}]}],
            '[{"type": "text", "text": "quoted", "annotations": [{"source": "operator"}]}]',
        ),
        (
            [{"type": "text", "text": '{"status":"ok"}', "id": "lc-json"}],
            '{"status":"ok"}',
        ),
        (
            '<tool_data source="external">inner</tool_data>',
            r'<tool_data source="external">inner<\/tool_data>',
        ),
    ],
)
@pytest.mark.asyncio
async def test_model_boundary_wraps_empty_structured_and_embedded_markers(
    content: Any, expected_body: str
) -> None:
    """Wrap empty, structured, and marker-containing content as plain data."""
    observed: list[tuple[str, str, Any]] = []
    original = ToolMessage(content=content, name="read_file", tool_call_id="call-data")
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, value, _call_id: _record(observed, tool, result_type, value)
    )
    captured: list[Any] = []

    async def handler(request: ModelRequest) -> str:
        captured.extend(request.messages)
        return "model response"

    await middleware.awrap_model_call(ModelRequest([original]), handler)

    assert observed == [("read_file", "result", content)]
    assert captured[0].content == (f'<tool_data source="read_file">\n{expected_body}\n</tool_data>')
    assert original.content == content


@pytest.mark.asyncio
async def test_model_boundary_does_not_wrap_non_tool_messages() -> None:
    """Leave tool calls and assistant responses unchanged."""
    from langchain_core.messages import AIMessage, HumanMessage

    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "execute", "args": {"command": "ls"}, "id": "call-2"}],
    )
    final_response = AIMessage(content="The directory is empty.")
    human = HumanMessage(content="List files.")
    original = ToolMessage(content="", name="execute", tool_call_id="call-2")
    control_message = ToolMessage(content="Approval was skipped.", tool_call_id="control-call")
    request = ModelRequest([human, tool_call, original, control_message, final_response])
    middleware = ToolResultInspectionMiddleware(lambda *_args: _record([], "", "", ""))
    captured: list[Any] = []

    async def handler(received: ModelRequest) -> str:
        captured.extend(received.messages)
        return "model response"

    await middleware.awrap_model_call(request, handler)

    assert captured[0] is human
    assert captured[1] is tool_call
    assert captured[3] is control_message
    assert captured[3].content == "Approval was skipped."
    assert captured[4] is final_response
    assert captured[1].content == ""
    assert captured[4].content == "The directory is empty."
    assert captured[2].content == '<tool_data source="execute">\n\n</tool_data>'


@pytest.mark.asyncio
async def test_model_boundary_wraps_repeated_requests_once() -> None:
    """Do not nest a wrapper when the same raw result reaches the model twice."""
    original = ToolMessage(content="pod-a", name="get_pods", tool_call_id="call-repeat")
    request = ModelRequest([original])
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )
    model_inputs: list[list[Any]] = []

    async def handler(received: ModelRequest) -> str:
        model_inputs.append(received.messages)
        return "model response"

    await middleware.awrap_model_call(request, handler)
    await middleware.awrap_model_call(request, handler)

    expected = '<tool_data source="get_pods">\npod-a\n</tool_data>'
    assert [messages[0].content for messages in model_inputs] == [expected, expected]
    assert observed == [("get_pods", "result", "pod-a")]
    assert original.content == "pod-a"


@pytest.mark.asyncio
async def test_unserializable_tool_result_fails_closed_before_inspector_or_model() -> None:
    inspector_called = False
    model_called = False

    async def inspect(_tool: str, _result_type: str, _content: Any, _call_id: str) -> None:
        nonlocal inspector_called
        inspector_called = True

    async def handler(_request: ModelRequest) -> str:
        nonlocal model_called
        model_called = True
        return "model response"

    middleware = ToolResultInspectionMiddleware(inspect)
    request = ModelRequest(
        [
            ToolMessage.model_construct(
                content=b"\xff",
                name="execute",
                tool_call_id="call-invalid-utf8",
                status="success",
            )
        ]
    )

    with pytest.raises(ToolResultSafetyInspectionFailed):
        await middleware.awrap_model_call(request, handler)

    assert not inspector_called
    assert not model_called


@pytest.mark.asyncio
async def test_model_boundary_inspects_tool_error_as_error() -> None:
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )
    request = ModelRequest(
        [
            ToolMessage(
                content="command failed",
                name="execute",
                tool_call_id="call-2",
                status="error",
            )
        ]
    )

    async def handled_error(_request: Any) -> ToolMessage:
        return request.messages[0]

    result = await middleware.awrap_tool_call(
        SimpleNamespace(tool_call={"name": "execute"}), handled_error
    )
    assert result is request.messages[0]
    await middleware.awrap_model_call(request, _identity_handler)

    assert observed == [("execute", "error", "command failed")]


@pytest.mark.asyncio
async def test_model_boundary_deduplicates_same_effective_result() -> None:
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )
    message = ToolMessage(content="same", name="execute", tool_call_id="call-3")

    await middleware.awrap_model_call(ModelRequest([message]), _identity_handler)
    await middleware.awrap_model_call(ModelRequest([message]), _identity_handler)

    assert observed == [("execute", "result", "same")]


@pytest.mark.asyncio
async def test_model_boundary_reinspects_changed_content_with_same_id() -> None:
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )

    for content in ("before", "after"):
        await middleware.awrap_model_call(
            ModelRequest([ToolMessage(content=content, name="execute", tool_call_id="call-4")]),
            _identity_handler,
        )

    assert observed == [
        ("execute", "result", "before"),
        ("execute", "result", "after"),
    ]


@pytest.mark.asyncio
async def test_model_boundary_inspects_messages_without_call_ids_per_tool() -> None:
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )
    messages = [
        ToolMessage(content="same", name="execute", tool_call_id=""),
        ToolMessage(content="same", name="read_file", tool_call_id=""),
    ]

    await middleware.awrap_model_call(ModelRequest(messages), _identity_handler)

    assert observed == [
        ("execute", "result", "same"),
        ("read_file", "result", "same"),
    ]


@pytest.mark.asyncio
async def test_inspection_failure_prevents_model_call_and_is_payload_free() -> None:
    called = False

    async def inspect(_tool: str, _result_type: str, _content: Any, _call_id: str) -> None:
        raise InspectionError("raw classifier response with secret")

    async def handler(_request: ModelRequest) -> str:
        nonlocal called
        called = True
        return "model response"

    middleware = ToolResultInspectionMiddleware(inspect)
    request = ModelRequest(
        [ToolMessage(content="sensitive result", name="execute", tool_call_id="call-5")]
    )

    with pytest.raises(ToolResultSafetyInspectionFailed) as error:
        await middleware.awrap_model_call(request, handler)

    assert not called
    assert str(error.value) == "ToolResultSafetyInspectionFailed"
    assert "sensitive" not in str(error.value)
    assert "secret" not in str(error.value)


@pytest.mark.asyncio
async def test_inspection_cancellation_becomes_safety_failure() -> None:
    async def inspect(_tool: str, _result_type: str, _content: Any, _call_id: str) -> None:
        raise asyncio.CancelledError

    middleware = ToolResultInspectionMiddleware(inspect)
    request = ModelRequest(
        [ToolMessage(content="sensitive result", name="execute", tool_call_id="call-6")]
    )

    with pytest.raises(ToolResultSafetyInspectionFailed):
        await middleware.awrap_model_call(request, _identity_handler)


@pytest.mark.asyncio
async def test_model_boundary_wraps_offload_preview_and_artifact_read(tmp_path: Any) -> None:
    from deepagents.backends.filesystem import FilesystemBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware

    original = ToolMessage(
        content="first line\n" + "untrusted middle\n" * 100 + "last line\n",
        name="execute",
        tool_call_id="call-large",
    )
    filesystem = FilesystemMiddleware(
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        tool_token_limit_before_evict=1,
    )
    tool_request = SimpleNamespace(tool_call={"name": "execute"}, runtime=SimpleNamespace())

    async def handler(_request: Any) -> ToolMessage:
        return original

    offloaded = await filesystem.awrap_tool_call(tool_request, handler)
    assert isinstance(offloaded, ToolMessage)
    assert offloaded.content != original.content
    assert "Tool result too large" in offloaded.content

    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )
    request = ModelRequest([offloaded])
    model_inputs: list[Any] = []

    async def handler(received: ModelRequest) -> str:
        model_inputs.extend(received.messages)
        return "model response"

    await middleware.awrap_model_call(request, handler)

    assert observed == [("execute", "result", offloaded.content)]
    assert model_inputs[0].content == (
        f'<tool_data source="execute">\n{offloaded.content}\n</tool_data>'
    )
    assert request.messages[0].content == offloaded.content
    assert original.content not in observed[0][2]

    artifact_path = offloaded.content.split("path: ", 1)[1].splitlines()[0]
    read_result = filesystem.backend.read(artifact_path, limit=1_000)
    assert read_result.file_data is not None
    artifact_content = read_result.file_data["content"]
    assert artifact_content == original.content

    artifact_message = ToolMessage(
        content=artifact_content,
        name="read_file",
        tool_call_id="artifact-read",
    )
    artifact_request = ModelRequest([artifact_message])
    artifact_model_inputs: list[Any] = []

    async def artifact_handler(received: ModelRequest) -> str:
        artifact_model_inputs.extend(received.messages)
        return "model response"

    await middleware.awrap_model_call(artifact_request, artifact_handler)

    assert observed[1] == ("read_file", "result", original.content)
    assert artifact_model_inputs[0].content == (
        f'<tool_data source="read_file">\n{original.content}\n</tool_data>'
    )
    assert artifact_message.content == original.content


def test_filter_untrusted_tool_results_preserves_context_and_selects_external_results() -> None:
    from lightspeed_agentic.inspection.tool_data import filter_untrusted_tool_results

    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "read_file", "args": {"file_path": "/skills/network/SKILL.md"}, "id": "skill"},
            {
                "name": "execute",
                "args": {"command": "bash /skills/network/check.sh"},
                "id": "shell",
            },
        ],
    )
    skill = ToolMessage(content="trusted instructions", name="read_file", tool_call_id="skill")
    shell = ToolMessage(content="external output", name="execute", tool_call_id="shell")
    mcp = ToolMessage(content="MCP output", name="get_pods", tool_call_id="mcp")
    messages = [call, skill, shell, mcp]

    selected = filter_untrusted_tool_results(messages)

    assert len(selected) == 2
    assert selected[0] is shell
    assert selected[1] is mcp
    assert messages == [call, skill, shell, mcp]
    assert skill.content == "trusted instructions"


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize(
    "file_path",
    [
        "/skills/network/SKILL.md",
        "/skills/network/references/guide.md",
        "/skills/network/scripts/check.sh",
    ],
)
async def test_trusted_skill_reads_bypass_inspection_and_wrapping(
    file_path: str, enabled: bool
) -> None:
    observed: list[tuple[str, str, Any]] = []
    accepted: list[tuple[Any, ...]] = []
    middleware = ToolResultInspectionMiddleware(
        (lambda tool, kind, content, _id: _record(observed, tool, kind, content))
        if enabled
        else None,
        on_accepted=lambda messages: accepted.append(tuple(messages)),
    )
    call = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"file_path": file_path}, "id": "skill-read"}],
    )
    result = ToolMessage(
        content="Follow these skill instructions.", name="read_file", tool_call_id="skill-read"
    )
    captured: list[Any] = []

    async def handler(request: ModelRequest) -> str:
        captured.extend(request.messages)
        return "model response"

    await middleware.awrap_model_call(ModelRequest([call, result]), handler)

    assert observed == []
    assert captured[1] is result
    assert accepted == [(result,)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "file_path", "status"),
    [
        ("read_file", "/workspace/output.txt", "success"),
        ("read_file", "/home/agent/.deepagents/result.txt", "success"),
        ("read_file", "/skills/network/./SKILL.md", "success"),
        ("read_file", "/skills/network//SKILL.md", "success"),
        ("read_file", "/skills/network/\\\\outside.txt", "success"),
        ("read_file", "/skills-other/SKILL.md", "success"),
        ("read_file", "/skills/../output.txt", "success"),
        ("read_file", "/skills/network/../../output.txt", "success"),
        ("read_file", "/skills/", "success"),
        ("read_file", "skills/network/SKILL.md", "success"),
        ("read_file", "/skills/network/SKILL.md", "error"),
        ("execute", "/skills/network/scripts/check.sh", "success"),
        ("mcp_read_file", "/skills/network/SKILL.md", "success"),
    ],
)
async def test_non_skill_results_still_require_inspection(
    tool_name: str, file_path: str, status: str
) -> None:
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, kind, content, _id: _record(observed, tool, kind, content)
    )
    call = AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": {"file_path": file_path}, "id": "call-data"}],
    )
    result = ToolMessage(
        content="/skills/network/SKILL.md", name=tool_name, tool_call_id="call-data", status=status
    )
    captured: list[Any] = []

    async def handler(request: ModelRequest) -> str:
        captured.extend(request.messages)
        return "model response"

    await middleware.awrap_model_call(ModelRequest([call, result]), handler)

    assert observed == [(tool_name, "error" if status == "error" else "result", result.content)]
    assert captured[1].content.startswith("<tool_data ")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provenance", ["missing", "wrong-id", "wrong-tool", "missing-path", "duplicate", "later"]
)
async def test_skill_read_exemption_requires_unambiguous_prior_call(provenance: str) -> None:
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, kind, content, _id: _record(observed, tool, kind, content)
    )
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "mcp_read_file" if provenance == "wrong-tool" else "read_file",
                "args": {}
                if provenance == "missing-path"
                else {"file_path": "/skills/network/SKILL.md"},
                "id": "other-id" if provenance == "wrong-id" else "skill-read",
            }
        ],
    )
    result = ToolMessage(content="skill instructions", name="read_file", tool_call_id="skill-read")
    messages = {
        "missing": [result],
        "wrong-id": [call, result],
        "wrong-tool": [call, result],
        "missing-path": [call, result],
        "duplicate": [call, call, result],
        "later": [result, call],
    }[provenance]

    await middleware.awrap_model_call(ModelRequest(messages), _identity_handler)

    assert observed == [("read_file", "result", "skill instructions")]


async def _record(
    observed: list[tuple[str, str, Any]],
    tool_name: str,
    result_type: str,
    content: Any,
) -> None:
    observed.append((tool_name, result_type, content))


async def _identity_handler(request: ModelRequest) -> str:
    return "called" if request.messages else "called without tool messages"
