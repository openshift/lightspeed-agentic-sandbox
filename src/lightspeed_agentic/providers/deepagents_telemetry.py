"""LangChain lifecycle callbacks for DeepAgents GenAI and tool spans."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lightspeed_agentic.providers._telemetry_base import _field, _json_arguments, _mapping

if TYPE_CHECKING:
    from lightspeed_agentic.audit import AuditLogger

_UNSET = object()


def _as_string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _json_text(value: Any) -> str:
    mapped = _mapping(value)
    if mapped is not None:
        value = dict(mapped)
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _block_type(block: Any) -> str:
    return _as_string(_field(block, "type")) or ""


def _part_from_block(block: Any) -> dict[str, Any] | None:
    kind = _block_type(block)
    if kind in {"text", "reasoning"}:
        content = _field(block, "content")
        if content is None:
            content = _field(block, "text")
        if content is None and kind == "reasoning":
            content = _field(block, "reasoning")
        if content is None:
            return None
        return {
            "type": kind,
            "content": content if isinstance(content, str) else _json_text(content),
        }
    if kind in {"tool_call", "tool_use"}:
        name = _as_string(_field(block, "name"))
        if name is None:
            return None
        part: dict[str, Any] = {"type": "tool_call", "name": name}
        call_id = _as_string(_field(block, "id"))
        if call_id:
            part["id"] = call_id
        arguments = _field(block, "arguments", _UNSET)
        if arguments is _UNSET:
            arguments = _field(block, "args", _UNSET)
        if arguments is _UNSET:
            arguments = _field(block, "input", _UNSET)
        if arguments is not _UNSET:
            part["arguments"] = _json_arguments(arguments)
        return part
    if kind == "tool_call_response":
        response = _field(block, "response", _UNSET)
        if response is _UNSET:
            response = _field(block, "result", _UNSET)
        if response is _UNSET:
            return None
        part = {"type": kind, "response": response}
        call_id = _as_string(_field(block, "id"))
        if call_id:
            part["id"] = call_id
        return part
    mapped = _mapping(block)
    return dict(mapped) if mapped is not None and kind else None


def _content_parts(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    if isinstance(content, (list, tuple)):
        parts: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                parts.append({"type": "text", "content": item})
            else:
                part = _part_from_block(item)
                if part is not None:
                    parts.append(part)
        return parts
    mapped = _mapping(content)
    if mapped is not None:
        part = _part_from_block(mapped)
        if part is not None:
            return [part]
        return [{"type": "text", "content": _json_text(content)}]
    return [{"type": "text", "content": str(content)}]


def _message_role(message: Any) -> str:
    role = _as_string(_field(message, "type")) or _as_string(_field(message, "role")) or ""
    return {
        "ai": "assistant",
        "assistant": "assistant",
        "human": "user",
        "user": "user",
        "system": "system",
        "developer": "system",
        "tool": "tool",
        "function": "tool",
    }.get(role, role)


def _message_calls(message: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for call in _field(message, "tool_calls", ()) or ():
        name = _as_string(_field(call, "name"))
        if name is None:
            continue
        call_id = _as_string(_field(call, "id")) or _as_string(_field(call, "tool_call_id")) or ""
        arguments = _field(call, "args", _UNSET)
        if arguments is _UNSET:
            arguments = _field(call, "arguments", _UNSET)
        calls.append(
            {
                "name": name,
                "call_id": call_id,
                "arguments": None if arguments is _UNSET else _json_arguments(arguments),
            }
        )

    blocks = _field(message, "content_blocks")
    if isinstance(blocks, (list, tuple)):
        for block in blocks:
            part = _part_from_block(block)
            if part is None or part.get("type") != "tool_call":
                continue
            name = _as_string(part.get("name"))
            if name is None:
                continue
            call_id = _as_string(part.get("id")) or ""
            arguments = part.get("arguments")
            duplicate = any(
                (call_id and call_id == existing["call_id"])
                or (not call_id and name == existing["name"] and arguments == existing["arguments"])
                for existing in calls
            )
            if not duplicate:
                calls.append({"name": name, "call_id": call_id, "arguments": arguments})
    additional = _mapping(_field(message, "additional_kwargs")) or {}
    for call in additional.get("tool_calls", ()) or ():
        function = _mapping(_field(call, "function")) or {}
        name = _as_string(function.get("name")) or _as_string(_field(call, "name"))
        if name is None:
            continue
        call_id = _as_string(_field(call, "id")) or ""
        arguments = _json_arguments(function.get("arguments", _field(call, "arguments")))
        duplicate = any(
            (call_id and existing["call_id"] == call_id)
            or (not call_id and existing["name"] == name and existing["arguments"] == arguments)
            for existing in calls
        )
        if not duplicate:
            calls.append({"name": name, "call_id": call_id, "arguments": arguments})
    return calls


def _message_parts(message: Any) -> list[dict[str, Any]]:
    if _message_role(message) == "tool":
        part: dict[str, Any] = {
            "type": "tool_call_response",
            "response": _field(message, "content"),
        }
        call_id = _as_string(_field(message, "tool_call_id"))
        if call_id:
            part["id"] = call_id
        return [part]

    blocks = _field(message, "content_blocks")
    if isinstance(blocks, (list, tuple)) and blocks:
        parts: list[dict[str, Any]] = []
        for block in blocks:
            content_part = _part_from_block(block)
            if content_part is not None:
                parts.append(content_part)
    else:
        parts = _content_parts(_field(message, "content"))

    for call in _message_calls(message):
        existing = next(
            (
                part
                for part in parts
                if part.get("type") == "tool_call"
                and (
                    (call["call_id"] and part.get("id") and part.get("id") == call["call_id"])
                    or (
                        (not call["call_id"] or not part.get("id"))
                        and part.get("name") == call["name"]
                        and part.get("arguments") == call["arguments"]
                    )
                )
            ),
            None,
        )
        if existing is not None:
            if call["call_id"] and not existing.get("id"):
                existing["id"] = call["call_id"]
            if call["arguments"] is not None and "arguments" not in existing:
                existing["arguments"] = call["arguments"]
            continue
        part = {"type": "tool_call", "name": call["name"]}
        if call["call_id"]:
            part["id"] = call["call_id"]
        if call["arguments"] is not None:
            part["arguments"] = call["arguments"]
        parts.append(part)
    return parts


def _input_payload(messages: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    if not isinstance(messages, (list, tuple)):
        return [], None
    request_messages = (
        messages[0] if messages and isinstance(messages[0], (list, tuple)) else messages
    )
    input_messages: list[dict[str, Any]] = []
    instructions: list[dict[str, Any]] = []
    has_instructions = False
    for message in request_messages:
        role = _message_role(message)
        parts = _message_parts(message)
        if role == "system":
            has_instructions = True
            instructions.extend(parts)
            continue
        item: dict[str, Any] = {"role": role or "unknown", "parts": parts}
        name = _as_string(_field(message, "name"))
        if name:
            item["name"] = name
        input_messages.append(item)
    return input_messages, instructions if has_instructions else None


def _generation_groups(response: Any) -> list[Any]:
    generations = _field(response, "generations")
    if not isinstance(generations, (list, tuple)):
        return []
    flattened: list[Any] = []
    for group in generations:
        if isinstance(group, (list, tuple)):
            flattened.extend(group)
        else:
            flattened.append(group)
    return flattened


def _finish_reason(generation: Any) -> str:
    message = _field(generation, "message")
    for source in (
        _mapping(_field(generation, "generation_info")) or {},
        _mapping(_field(message, "response_metadata")) or {},
        _mapping(_field(message, "additional_kwargs")) or {},
    ):
        for key in ("finish_reason", "stop_reason", "finishReason", "stopReason"):
            value = source.get(key)
            if value is not None:
                return str(getattr(value, "value", value))
    return "unknown"


def _output_messages(response: Any) -> tuple[list[dict[str, Any]] | None, list[str] | None]:
    generations = _generation_groups(response)
    if not generations:
        return None, None
    messages: list[dict[str, Any]] = []
    finish_reasons: list[str] = []
    for generation in generations:
        message = _field(generation, "message")
        if message is None:
            parts = _content_parts(_field(generation, "text"))
            role = "assistant"
        else:
            parts = _message_parts(message)
            role = _message_role(message) or "assistant"
        reason = _finish_reason(generation)
        messages.append({"role": role, "parts": parts, "finish_reason": reason})
        finish_reasons.append(reason)
    return messages, finish_reasons


def _metadata_sources(response: Any) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    llm_output = _mapping(_field(response, "llm_output")) or {}
    for key in ("token_usage", "usage", "usage_metadata"):
        nested = _mapping(llm_output.get(key))
        if nested is not None:
            sources.append(nested)
    if llm_output:
        sources.append(llm_output)
    for value in (
        _field(response, "token_usage"),
        _field(response, "usage"),
        _field(response, "usage_metadata"),
    ):
        nested = _mapping(value)
        if nested is not None:
            sources.append(nested)
    for generation in _generation_groups(response):
        message = _field(generation, "message")
        for value in (
            _field(message, "usage_metadata"),
            _field(message, "response_metadata"),
            _field(generation, "generation_info"),
        ):
            nested = _mapping(value)
            if nested is not None:
                for key in ("token_usage", "usage", "usage_metadata"):
                    nested_usage = _mapping(nested.get(key))
                    if nested_usage is not None:
                        sources.append(nested_usage)
                sources.append(nested)
    return sources


def _token_count(sources: list[Mapping[str, Any]], keys: tuple[str, ...]) -> int | None:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _reasoning_token_count(sources: list[Mapping[str, Any]]) -> int | None:
    count = _token_count(sources, ("reasoning_tokens", "reasoning_output_tokens"))
    if count is not None:
        return count
    for source in sources:
        for key in ("output_token_details", "completion_tokens_details"):
            details = _mapping(source.get(key)) or {}
            count = _token_count([details], ("reasoning", "reasoning_tokens"))
            if count is not None:
                return count
    return None


def _response_model(response: Any) -> str | None:
    llm_output = _mapping(_field(response, "llm_output")) or {}
    response_metadata = _mapping(_field(response, "response_metadata")) or {}
    sources = [llm_output, response_metadata]
    sources.extend(
        _mapping(_field(_field(generation, "message"), "response_metadata")) or {}
        for generation in _generation_groups(response)
    )
    for source in sources:
        for key in ("model_name", "model", "model_id", "response_model"):
            observed = _as_string(source.get(key))
            if observed:
                return observed
    return None


def _tool_definitions(sources: tuple[Any, ...]) -> list[dict[str, Any]] | None:
    for source in sources:
        mapped = _mapping(source)
        if mapped is None:
            continue
        tools = mapped.get("tools", _UNSET)
        if tools is _UNSET:
            tools = mapped.get("functions", _UNSET)
        if tools is _UNSET or tools is None:
            continue
        if not isinstance(tools, (list, tuple)):
            tools = [tools]
        definitions: list[dict[str, Any]] = []
        for tool in tools:
            raw = _mapping(tool)
            if raw is None:
                continue
            function = _mapping(raw.get("function"))
            if function is not None:
                raw = {**function, "type": raw.get("type", "function")}
            name = _as_string(raw.get("name"))
            if name is None:
                continue
            definition: dict[str, Any] = {
                "type": _as_string(raw.get("type")) or "function",
                "name": name,
            }
            description = raw.get("description")
            if description is not None:
                definition["description"] = description
            parameters = raw.get("parameters", _UNSET)
            if parameters is _UNSET:
                parameters = raw.get("input_schema", _UNSET)
            if parameters is _UNSET:
                parameters = raw.get("schema", _UNSET)
            if parameters is not _UNSET and parameters is not None:
                definition["parameters"] = parameters
            definitions.append(definition)
        return definitions
    return None


def _request_model(sources: tuple[Any, ...], fallback: str) -> str:
    for source in sources:
        mapped = _mapping(source)
        if mapped is None:
            continue
        for key in ("model", "model_name", "model_id"):
            value = _as_string(mapped.get(key))
            if value:
                return value
    return fallback


def _tool_input(inputs: Any, input_str: Any) -> Any:
    if inputs is not None:
        return inputs
    if isinstance(input_str, str) and input_str:
        return _json_arguments(input_str)
    return None


@dataclass
class _Inference:
    span: Any


@dataclass
class _ToolRun:
    span: Any
    call_id: str


def create_callback_handler(
    audit_logger: AuditLogger,
    *,
    model: str,
    output_type: str | None = None,
    capture_content: bool = True,
) -> Any:
    """Create a lazily imported AsyncCallbackHandler for one DeepAgents lifecycle."""
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.messages import ToolMessage

    class DeepAgentsAuditCallbackHandler(AsyncCallbackHandler):
        def __init__(self) -> None:
            super().__init__()
            self._capture_content = capture_content
            self._inference_spans: dict[Any, _Inference] = {}
            self._tool_runs: dict[Any, _ToolRun] = {}

        async def on_chat_model_start(
            self,
            serialized: dict[str, Any],
            messages: list[list[Any]],
            *,
            run_id: Any,
            parent_run_id: Any = None,
            invocation_params: dict[str, Any] | None = None,
            options: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            del parent_run_id
            sources = (invocation_params, options, kwargs, _field(serialized, "kwargs"), serialized)
            request_model = _request_model(sources, model)
            if self._capture_content:
                input_messages, system_instructions = _input_payload(messages)
                tool_definitions = _tool_definitions(sources)
            else:
                input_messages = None
                system_instructions = None
                tool_definitions = None
            span = audit_logger.start_inference(
                model=request_model,
                operation="chat",
                input_messages=input_messages,
                system_instructions=system_instructions,
                tool_definitions=tool_definitions,
                output_type=output_type,
                start_time=time.time_ns(),
            )
            self._inference_spans[run_id] = _Inference(span=span)

        async def on_llm_end(
            self,
            response: Any,
            *,
            run_id: Any,
            parent_run_id: Any = None,
            **kwargs: Any,
        ) -> None:
            del parent_run_id, kwargs
            generations = _generation_groups(response)
            if self._capture_content:
                output_messages, finish_reasons = _output_messages(response)
            else:
                output_messages = None
                finish_reasons = [_finish_reason(generation) for generation in generations] or None
            state = self._inference_spans.pop(run_id, None)
            if state is None:
                return
            sources = _metadata_sources(response)
            audit_logger.end_inference(
                state.span,
                output_messages=output_messages,
                response_model=_response_model(response),
                input_tokens=_token_count(sources, ("input_tokens", "prompt_tokens")),
                output_tokens=_token_count(sources, ("output_tokens", "completion_tokens")),
                reasoning_tokens=_reasoning_token_count(sources),
                finish_reasons=finish_reasons,
                end_time=time.time_ns(),
            )

        async def on_llm_error(
            self,
            error: BaseException,
            *,
            run_id: Any,
            parent_run_id: Any = None,
            **kwargs: Any,
        ) -> None:
            del parent_run_id, kwargs
            state = self._inference_spans.pop(run_id, None)
            if state is not None:
                audit_logger.end_inference(state.span, error=error, end_time=time.time_ns())

        async def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            run_id: Any,
            parent_run_id: Any = None,
            inputs: Any = None,
            **kwargs: Any,
        ) -> None:
            del parent_run_id
            if not self._capture_content:
                return
            name = _as_string(kwargs.get("name")) or _as_string(_field(serialized, "name"))
            if name is None:
                return
            arguments = _tool_input(inputs, input_str)
            call_id = _as_string(kwargs.get("tool_call_id")) or ""
            span = audit_logger.start_tool(
                name=name,
                call_id=call_id,
                arguments=arguments,
                tool_type="function",
                start_time=time.time_ns(),
            )
            self._tool_runs[run_id] = _ToolRun(span=span, call_id=call_id)

        async def on_tool_end(
            self,
            output: Any,
            *,
            run_id: Any,
            parent_run_id: Any = None,
            **kwargs: Any,
        ) -> None:
            del parent_run_id, kwargs
            state = self._tool_runs.pop(run_id, None)
            if state is None:
                return
            if isinstance(output, ToolMessage):
                output_call_id = _as_string(output.tool_call_id)
                if output_call_id:
                    self._set_call_id(state, output_call_id)
                error = "tool_error" if output.status == "error" else None
                result = output.content if error is None else None
            else:
                error = None
                result = output
            audit_logger.end_tool(
                state.span,
                result=result,
                error=error,
                end_time=time.time_ns(),
            )

        async def on_tool_error(
            self,
            error: BaseException,
            *,
            run_id: Any,
            parent_run_id: Any = None,
            **kwargs: Any,
        ) -> None:
            del parent_run_id, kwargs
            state = self._tool_runs.pop(run_id, None)
            if state is not None:
                audit_logger.end_tool(state.span, error=error, end_time=time.time_ns())

        def close(self, error: BaseException | str | None = None) -> None:
            """Close only this handler's still-open spans."""
            end_time = time.time_ns()
            for run_id, state in tuple(self._inference_spans.items()):
                self._inference_spans.pop(run_id, None)
                audit_logger.end_inference(
                    state.span,
                    error=error if error is not None else "operation_cancelled",
                    end_time=end_time,
                )
            for run_id, tool_state in tuple(self._tool_runs.items()):
                self._tool_runs.pop(run_id, None)
                audit_logger.end_tool(
                    tool_state.span,
                    error=error if error is not None else "operation_cancelled",
                    end_time=end_time,
                )

        @staticmethod
        def _set_call_id(state: _ToolRun, call_id: str) -> None:
            if state.call_id or not call_id:
                return
            state.call_id = call_id
            state.span.set_attribute("gen_ai.tool.call.id", call_id)

    return DeepAgentsAuditCallbackHandler()
