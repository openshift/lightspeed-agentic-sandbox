"""Native Google ADK lifecycle callbacks for sandbox-owned telemetry."""

from __future__ import annotations

import base64
import importlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, cast

from lightspeed_agentic.providers._telemetry_base import _field

if TYPE_CHECKING:
    from lightspeed_agentic.audit import AuditLogger


_ADK_TRACER_ALIAS_MODULES = (
    "google.adk.telemetry",
    "google.adk.telemetry.node_tracing",
    "google.adk.runners",
    "google.adk.flows.llm_flows.base_llm_flow",
    "google.adk.flows.llm_flows.functions",
)
_MISSING = object()


class _ADKTracerModule(Protocol):
    tracer: Any


class _ADKTelemetryModule(_ADKTracerModule, Protocol):
    otel_logger: Any


@dataclass
class _InferenceCall:
    span: Any
    partial_parts: list[dict[str, Any]] = field(default_factory=list)
    response_model: str | None = None
    input_tokens: int | None = None
    candidate_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None


@dataclass
class _PendingResponse:
    call: _InferenceCall
    content: Any
    parts: list[dict[str, Any]]
    end_time: int
    agent_name: str | None
    invocation_id: str | None


def _llm_call_limit_reached(callback_context: Any) -> bool:
    """Match ADK 2.5.0's pre-increment rejection condition."""
    invocation_context = _field(callback_context, "_invocation_context")
    run_config = _field(invocation_context, "run_config")
    max_llm_calls = _field(run_config, "max_llm_calls")
    cost_manager = _field(invocation_context, "_invocation_cost_manager")
    call_count = _field(cost_manager, "_number_of_llm_calls")
    return (
        isinstance(max_llm_calls, int)
        and max_llm_calls > 0
        and isinstance(call_count, int)
        and call_count >= max_llm_calls
    )


class GeminiTelemetry:
    """Record actual Gemini requests and ADK tool executions via shared recorder."""

    def __init__(self, audit_logger: AuditLogger, requested_model: str) -> None:
        self._audit_logger = audit_logger
        self._requested_model = requested_model
        self._inference: _InferenceCall | None = None
        self._pending_response: _PendingResponse | None = None
        self._tool_spans: dict[int, Any] = {}

    def before_model_callback(self, callback_context: Any, llm_request: Any) -> None:
        """Start one generate_content span from ADK's actual request boundary."""
        self._finish_pending_response()
        if _llm_call_limit_reached(callback_context):
            return None
        request_config = _field(llm_request, "config")
        request_model = _field(llm_request, "model") or self._requested_model
        span = self._audit_logger.start_inference(
            model=request_model,
            operation="generate_content",
            input_messages=_input_messages(_field(llm_request, "contents")),
            system_instructions=_system_instruction_parts(
                _field(request_config, "system_instruction")
            ),
            tool_definitions=_tool_definitions(_field(request_config, "tools")),
            output_type=_output_type(request_config),
            start_time=time.time_ns(),
        )
        self._inference = _InferenceCall(span=span)

    def after_model_callback(self, callback_context: Any, llm_response: Any) -> None:
        """Accumulate streamed parts and finish at ADK's terminal response."""
        call = self._inference
        if call is None:
            return None

        self._observe_response(call, llm_response)
        partial_parts = _message_parts(_field(llm_response, "content"))
        if _field(llm_response, "partial", False):
            call.partial_parts.extend(partial_parts)
            return None

        output_parts = partial_parts or call.partial_parts
        content = _field(llm_response, "content")
        end_time = time.time_ns()
        self._inference = None
        needs_finalized_event = any(
            part.get("type") == "tool_call" and not part.get("id") for part in output_parts
        )
        if needs_finalized_event:
            self._pending_response = _PendingResponse(
                call=call,
                content=content,
                parts=output_parts,
                end_time=end_time,
                agent_name=_field(callback_context, "agent_name"),
                invocation_id=_field(callback_context, "invocation_id"),
            )
        else:
            self._emit_model_response(call, content, output_parts, end_time)
        return None

    def on_model_error_callback(
        self, callback_context: Any, llm_request: Any, error: BaseException | str
    ) -> None:
        """End a failed request with only response data observed before failure."""
        _ = callback_context, llm_request
        call = self._inference
        if call is None:
            return None

        self._inference = None
        output_messages = None
        finish_reasons = None
        if call.partial_parts:
            finish_reason = call.finish_reason or "unknown"
            output_messages = [_output_message(None, call.partial_parts, finish_reason)]
            finish_reasons = [finish_reason]

        self._audit_logger.end_inference(
            call.span,
            output_messages=output_messages,
            response_model=call.response_model,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            reasoning_tokens=call.reasoning_tokens,
            finish_reasons=finish_reasons,
            error=error,
            end_time=time.time_ns(),
        )
        return None

    def observe_model_event(self, event: Any) -> None:
        """Join pending output to its matching finalized model Event."""
        pending = self._pending_response
        if pending is None or _field(event, "partial", False):
            return
        if not pending.agent_name or not pending.invocation_id:
            return
        if _field(event, "author") != pending.agent_name:
            return
        if _field(event, "invocation_id") != pending.invocation_id:
            return
        content = _field(event, "content")
        if _field(content, "role") != "model":
            return
        self._finish_pending_response(content)

    def close(self, error: BaseException | str | None = None) -> None:
        """Flush completed output and fail only still-active operations."""
        self._finish_pending_response()
        interruption = error if error is not None else "operation_cancelled"
        if self._inference is not None:
            self.on_model_error_callback(None, None, interruption)
        end_time = time.time_ns()
        for key in tuple(self._tool_spans):
            span = self._tool_spans.pop(key)
            self._audit_logger.end_tool(span, error=interruption, end_time=end_time)

    def _emit_model_response(
        self,
        call: _InferenceCall,
        content: Any,
        parts: list[dict[str, Any]],
        end_time: int,
    ) -> None:
        finish_reason = call.finish_reason or "unknown"
        self._audit_logger.end_inference(
            call.span,
            output_messages=[_output_message(content, parts, finish_reason)],
            response_model=call.response_model,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            reasoning_tokens=call.reasoning_tokens,
            finish_reasons=[finish_reason],
            end_time=end_time,
        )

    def _finish_pending_response(self, content: Any | None = None) -> None:
        pending = self._pending_response
        if pending is None:
            return
        self._pending_response = None
        if content is None:
            content = pending.content
        parts = _message_parts(content) or pending.parts
        self._emit_model_response(pending.call, content, parts, pending.end_time)

    def before_tool_callback(self, tool: Any, args: dict[str, Any], tool_context: Any) -> None:
        """Start a span for one actual ADK tool execution."""
        call_id = _field(tool_context, "function_call_id")
        span = self._audit_logger.start_tool(
            name=_field(tool, "name") or "",
            call_id=str(call_id) if call_id is not None else "",
            arguments=args,
            tool_type="function",
            start_time=time.time_ns(),
        )
        self._tool_spans[id(tool_context)] = span
        return None

    def after_tool_callback(
        self,
        tool: Any,
        args: dict[str, Any],
        tool_context: Any,
        tool_response: Any,
    ) -> None:
        """Record the untrimmed execution result before later ADK transforms it."""
        _ = tool, args
        span = self._tool_spans.pop(id(tool_context), None)
        if span is not None:
            self._audit_logger.end_tool(
                span,
                result=tool_response,
                end_time=time.time_ns(),
            )
        return None

    def on_tool_error_callback(
        self,
        tool: Any,
        args: dict[str, Any],
        tool_context: Any,
        error: BaseException,
    ) -> None:
        """End a failed tool span without recording a successful result."""
        _ = tool, args
        span = self._tool_spans.pop(id(tool_context), None)
        if span is not None:
            self._audit_logger.end_tool(
                span,
                error=error,
                end_time=time.time_ns(),
            )
        return None

    @staticmethod
    def _observe_response(call: _InferenceCall, response: Any) -> None:
        model = _field(response, "model_version")
        if model:
            call.response_model = str(model)

        usage = _field(response, "usage_metadata")
        if usage is not None:
            prompt_tokens = _field(usage, "prompt_token_count")
            if prompt_tokens is not None:
                call.input_tokens = prompt_tokens

            candidate_tokens = _field(usage, "candidates_token_count")
            if candidate_tokens is not None:
                call.candidate_tokens = candidate_tokens
            reasoning_tokens = _field(usage, "thoughts_token_count")
            if reasoning_tokens is not None:
                call.reasoning_tokens = reasoning_tokens
            if call.candidate_tokens is not None:
                call.output_tokens = call.candidate_tokens + (call.reasoning_tokens or 0)

        finish_reason = _field(response, "finish_reason")
        if finish_reason is not None:
            call.finish_reason = _finish_reason(finish_reason)


def disable_adk_native_telemetry() -> None:
    """Disable ADK's legacy OTel sources without changing global OTel settings.

    Google ADK 2.5.0 has no per-invocation switch that disables its built-in
    spans. Its tracing module also emits legacy GenAI LogRecords independently
    of those spans, so both module-scoped sources are replaced for this
    one-shot sandbox invocation. The process TracerProvider, LoggerProvider,
    exporters, and capture environment remain untouched.
    """
    from opentelemetry import trace
    from opentelemetry._logs import NoOpLoggerProvider

    adk_tracing = cast(
        _ADKTelemetryModule,
        importlib.import_module("google.adk.telemetry.tracing"),
    )
    adk_tracer = adk_tracing.tracer
    noop_tracer = trace.NoOpTracer()
    adk_tracing.tracer = noop_tracer
    adk_tracing.otel_logger = NoOpLoggerProvider().get_logger("gcp.vertex.agent")

    for module_name in _ADK_TRACER_ALIAS_MODULES:
        module = importlib.import_module(module_name)
        if hasattr(module, "tracer"):
            tracer_module = cast(_ADKTracerModule, module)
            if tracer_module.tracer is adk_tracer:
                tracer_module.tracer = noop_tracer


def _input_messages(contents: Any) -> list[dict[str, Any]]:
    messages = []
    for content in contents or ():
        message = _message(content, default_role="user")
        if message is not None:
            messages.append(message)
    return messages


def _system_instruction_parts(instruction: Any) -> list[dict[str, Any]] | None:
    if instruction is None:
        return None
    if isinstance(instruction, str):
        return [{"type": "text", "content": instruction}]
    return _message_parts(instruction)


def _message(content: Any, *, default_role: str) -> dict[str, Any] | None:
    if content is None:
        return None
    if isinstance(content, str):
        return {
            "role": default_role,
            "parts": [{"type": "text", "content": content}],
        }

    role = _field(content, "role")
    if role is None:
        role = default_role
    elif role == "model":
        role = "assistant"

    message: dict[str, Any] = {
        "role": str(role),
        "parts": _message_parts(content),
    }
    name = _field(content, "name")
    if name is not None:
        message["name"] = name
    return message


def _output_message(
    content: Any, parts: list[dict[str, Any]], finish_reason: str
) -> dict[str, Any]:
    message = _message(content, default_role="assistant")
    if message is None:
        message = {"role": "assistant", "parts": []}
    if parts:
        message["parts"] = parts
    message["finish_reason"] = finish_reason
    return message


def _message_parts(content: Any) -> list[dict[str, Any]]:
    parts = _field(content, "parts")
    if parts is None:
        return []

    normalized: list[dict[str, Any]] = []
    for part in parts:
        text = _field(part, "text", _MISSING)
        if text is not _MISSING and text is not None:
            part_type = "reasoning" if _field(part, "thought", False) else "text"
            normalized.append({"type": part_type, "content": text})

        function_call = _field(part, "function_call")
        if function_call is not None:
            name = _field(function_call, "name")
            if name:
                tool_call: dict[str, Any] = {
                    "type": "tool_call",
                    "name": str(name),
                }
                call_id = _field(function_call, "id")
                if call_id is not None and call_id != "":
                    tool_call["id"] = str(call_id)
                arguments = _field(function_call, "args", _MISSING)
                if arguments is not _MISSING and arguments is not None:
                    tool_call["arguments"] = _json_value(arguments)
                normalized.append(tool_call)

        function_response = _field(part, "function_response")
        if function_response is not None:
            response_part: dict[str, Any] = {"type": "tool_call_response"}
            call_id = _field(function_response, "id")
            if call_id is not None and call_id != "":
                response_part["id"] = str(call_id)
            response = _field(function_response, "response", _MISSING)
            if response is not _MISSING:
                response_part["response"] = _json_value(response)
            normalized.append(response_part)

    return normalized


def _tool_definitions(tools: Any) -> list[dict[str, Any]] | None:
    definitions: list[dict[str, Any]] = []
    for tool in tools or ():
        declarations = _field(tool, "function_declarations")
        if declarations is None:
            declarations = _field(tool, "functionDeclarations")
        if declarations:
            for declaration in declarations:
                definition = _function_definition(declaration)
                if definition is not None:
                    definitions.append(definition)
            continue

        fields = _object_fields(tool)
        if "name" in fields and not any(
            key not in {"name", "description", "parameters", "parameters_json_schema"}
            for key in fields
        ):
            definition = _function_definition(tool)
            if definition is not None:
                definitions.append(definition)
            continue

        for tool_type, config in fields.items():
            if tool_type in {"function_declarations", "functionDeclarations"}:
                continue
            if config is not None:
                definitions.append({"type": tool_type, "name": tool_type})

    return definitions or None


def _function_definition(declaration: Any) -> dict[str, Any] | None:
    name = _field(declaration, "name")
    if not name:
        return None
    definition: dict[str, Any] = {"type": "function", "name": str(name)}
    description = _field(declaration, "description")
    if description is not None:
        definition["description"] = description
    parameters = _field(declaration, "parameters")
    if parameters is None:
        parameters = _field(declaration, "parameters_json_schema")
    if parameters is not None:
        definition["parameters"] = _json_value(parameters)
    return definition


def _output_type(config: Any) -> str | None:
    if config is None:
        return None
    if _field(config, "response_schema") is not None:
        return "json"
    mime_type = _field(config, "response_mime_type")
    if not mime_type:
        return None
    media_type = str(mime_type).split(";", 1)[0].strip().lower()
    if media_type == "application/json":
        return "json"
    if media_type.startswith("text/"):
        return "text"
    modality = media_type.split("/", 1)[0]
    return modality if modality in {"audio", "image", "video"} else None


def _finish_reason(value: Any) -> str:
    reason = getattr(value, "value", None)
    if not isinstance(reason, str):
        reason = getattr(value, "name", None)
    if not isinstance(reason, str):
        reason = str(value)
    return reason.lower() if reason else "unknown"


def _object_fields(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json", exclude_none=True)
        except TypeError:
            dumped = model_dump(exclude_none=True)
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "__dict__"):
        return cast(Mapping[str, Any], vars(value))
    return {}


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json", exclude_none=True)
        except TypeError:
            dumped = model_dump(exclude_none=True)
        return _json_value(dumped)
    return value
