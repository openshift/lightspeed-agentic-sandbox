from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from lightspeed_agentic.inspection.errors import InspectionError
from lightspeed_agentic.inspection.middleware import (
    ToolResultInspectionMiddleware,
    ToolResultSafetyInspectionFailed,
)


class ModelRequest:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages


@pytest.mark.asyncio
async def test_model_boundary_inspects_new_tool_message_before_handler() -> None:
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
    assert observed_call_ids == ["call-1"]
    assert passed_to_model == request.messages
    assert middleware.is_passed("get_pods", "result", "call-1", "pod output")


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
async def test_rejected_result_does_not_reach_model() -> None:
    model_called = False

    async def inspect(_tool: str, _result_type: str, _content: Any, _call_id: str) -> Any:
        return SimpleNamespace(passed=False)

    async def handler(_request: ModelRequest) -> str:
        nonlocal model_called
        model_called = True
        return "model response"

    middleware = ToolResultInspectionMiddleware(inspect)
    request = ModelRequest(
        [
            ToolMessage(
                content="REJECTED-RESULT-SECRET",
                name="execute",
                tool_call_id="call-rejected",
            )
        ]
    )

    with pytest.raises(ToolResultSafetyInspectionFailed):
        await middleware.awrap_model_call(request, handler)

    assert not model_called


@pytest.mark.asyncio
async def test_model_boundary_inspects_tool_error_as_error() -> None:
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content),
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

    await middleware.awrap_model_call(request, _identity_handler)

    assert observed == [("execute", "error", "command failed")]
    assert middleware.is_passed("execute", "error", "call-2", "command failed")


@pytest.mark.asyncio
async def test_model_boundary_deduplicates_same_effective_result() -> None:
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content),
    )
    message = ToolMessage(content="same", name="execute", tool_call_id="call-3")

    await middleware.awrap_model_call(ModelRequest([message]), _identity_handler)
    await middleware.awrap_model_call(ModelRequest([message]), _identity_handler)

    assert observed == [("execute", "result", "same")]


@pytest.mark.asyncio
async def test_model_boundary_reinspects_changed_content_with_same_id() -> None:
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content),
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
    assert middleware.is_passed("execute", "result", "call-4", "before")
    assert middleware.is_passed("execute", "result", "call-4", "after")


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
    assert middleware.is_passed("execute", "result", "", "same")
    assert middleware.is_passed("read_file", "result", "", "same")


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
    assert not middleware.is_passed("execute", "result", "call-5", "sensitive result")


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
async def test_inspection_sees_filesystem_offload_preview_before_model(tmp_path: Any) -> None:
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
    assert model_inputs[0].content == observed[0][2]
    assert original.content not in observed[0][2]


async def _record(
    observed: list[tuple[str, str, Any]],
    tool_name: str,
    result_type: str,
    content: Any,
) -> None:
    observed.append((tool_name, result_type, content))


async def _identity_handler(request: ModelRequest) -> str:
    return "called" if request.messages else "called without tool messages"
