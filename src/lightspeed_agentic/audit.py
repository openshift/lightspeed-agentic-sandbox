"""Audit instrumentation — OTel spans and span events for compliance."""

from __future__ import annotations

import json
import math
import time
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, StatusCode

from lightspeed_agentic.metrics import tool_duration
from lightspeed_agentic.tracing import get_tracer, set_json_span_attribute
from lightspeed_agentic.types import ProviderEvent


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number: {value}")
    return number


def _tool_payload(value: str) -> dict[str, Any]:
    """Normalize payloads to an object; ``content`` wraps nonobject values.

    The envelope is normalization, not a native tool-argument name.
    """
    try:
        parsed = json.loads(value, parse_float=_finite_float, parse_constant=_finite_float)
    except (RecursionError, ValueError):
        parsed = value
    return parsed if isinstance(parsed, dict) else {"content": parsed}


def _set_tool_payload(span: Any, name: str, value: str) -> None:
    payload = _tool_payload(value)
    try:
        set_json_span_attribute(span, name, payload)
    except RecursionError:
        set_json_span_attribute(span, name, {"content": value})


class AuditLogger:
    """Map provider stream events to OTel tool spans and ``gen_ai.choice`` span events."""

    def __init__(
        self,
        *,
        phase: str,
        enabled: bool = True,
        capture_content: bool = False,
        agenticrun_uid: str = "",
        capture_tool_spans: bool = True,
    ) -> None:
        """Configure audit emission for one agent run.

        ``phase`` is the operator-provided AgenticRun phase from
        ``LIGHTSPEED_AGENTICRUN_STEP``. When ``enabled`` is false, buffers are
        cleared without emitting span events.
        """
        self._agenticrun_phase = phase
        self._enabled = enabled
        self._capture_content = capture_content
        self._capture_tool_spans = capture_tool_spans
        self._agenticrun_uid = agenticrun_uid
        self._text_buffer: list[str] = []
        self._thinking_buffer: list[str] = []
        self._tool_spans: dict[str | int, tuple[Any, float]] = {}
        self._next_call_id: int = 0
        self._tracer = get_tracer()
        self._parent_context: Context | None = None

    def set_parent_context(self, ctx: Context) -> None:
        """Set OTel context so tool spans are children of the inference span."""
        self._parent_context = ctx

    def process_event(self, event: ProviderEvent) -> None:
        """Consume one normalized provider event; buffer text or open/close tool spans."""
        match event.type:
            case "text_delta":
                self._text_buffer.append(event.text)
            case "thinking_delta":
                self._thinking_buffer.append(event.thinking)
            case "content_block_stop":
                self._flush_buffers()
            case "tool_call":
                self._flush_buffers()
                if not self._capture_tool_spans:
                    return
                call_key = event.call_id or self._next_call_id
                if not event.call_id:
                    self._next_call_id += 1
                tool_name = event.name or "unknown"
                attrs: dict[str, str] = {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": tool_name,
                    "gen_ai.tool.type": "function",
                }
                if event.call_id:
                    attrs["gen_ai.tool.call.id"] = event.call_id
                if self._agenticrun_uid:
                    attrs["agenticrun.uid"] = self._agenticrun_uid
                if self._agenticrun_phase:
                    attrs["agenticrun.phase"] = self._agenticrun_phase
                span = self._tracer.start_span(
                    f"execute_tool {tool_name}",
                    kind=SpanKind.INTERNAL,
                    context=self._parent_context,
                    attributes=attrs,
                )
                _set_tool_payload(
                    span,
                    "gen_ai.tool.call.arguments",
                    event.trace_input if event.trace_input is not None else event.input,
                )
                self._tool_spans[call_key] = (span, time.monotonic())
            case "tool_result":
                if not self._capture_tool_spans:
                    return
                call_id = event.call_id
                if call_id:
                    entry = self._tool_spans.pop(call_id, None)
                elif len(self._tool_spans) == 1:
                    _, entry = self._tool_spans.popitem()
                else:
                    entry = None
                if entry is not None:
                    tool_span, start = entry
                    tool_name = (
                        tool_span.attributes.get("gen_ai.tool.name", "unknown")
                        if hasattr(tool_span, "attributes")
                        else "unknown"
                    )
                    tool_duration.labels(gen_ai_tool_name=tool_name).observe(
                        time.monotonic() - start
                    )
                    _set_tool_payload(tool_span, "gen_ai.tool.call.result", event.output)
                    if event.error_type is not None:
                        tool_span.set_attribute("error.type", event.error_type)
                        tool_span.set_status(StatusCode.ERROR)
                    else:
                        tool_span.set_status(StatusCode.OK)
                    tool_span.end()
            case "result":
                self._flush_buffers()

    def complete(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        span: Any = None,
    ) -> None:
        """Flush buffers, close open tool spans, and stamp usage on the inference span."""
        self._flush_buffers(span)
        for _span_key, (tool_span, start) in self._tool_spans.items():
            tool_name = (
                tool_span.attributes.get("gen_ai.tool.name", "unknown")
                if hasattr(tool_span, "attributes")
                else "unknown"
            )
            tool_duration.labels(gen_ai_tool_name=tool_name).observe(time.monotonic() - start)
        self.close_pending_tools()
        if span is not None and span.is_recording():
            # Unreported counters are omitted rather than stamped as zero.
            if input_tokens is not None:
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            if output_tokens is not None:
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            if reasoning_tokens is not None:
                span.set_attribute("gen_ai.usage.reasoning.output_tokens", reasoning_tokens)

    def close_pending_tools(self) -> None:
        """Close unresolved tool spans without flushing buffered choice events."""
        for tool_span, _start in self._tool_spans.values():
            tool_span.set_attribute("error.type", "missing_tool_result")
            tool_span.set_status(StatusCode.ERROR, "tool span not closed by result event")
            tool_span.end()
        self._tool_spans.clear()

    def _flush_buffers(self, explicit_span: Any = None) -> None:
        """Emit buffered completion/thinking text as ``gen_ai.choice`` span events."""
        if not self._enabled:
            if self._text_buffer:
                self._text_buffer.clear()
            if self._thinking_buffer:
                self._thinking_buffer.clear()
            return

        span = explicit_span or trace.get_current_span()
        if not span or not span.is_recording():
            self._text_buffer.clear()
            self._thinking_buffer.clear()
            return
        if self._text_buffer:
            text = "".join(self._text_buffer)
            self._text_buffer.clear()
            if text:
                attrs = {"gen_ai.completion": text} if self._capture_content else {}
                span.add_event("gen_ai.choice", attributes=attrs)
        if self._thinking_buffer:
            thinking = "".join(self._thinking_buffer)
            self._thinking_buffer.clear()
            if thinking:
                attrs = {"gen_ai.reasoning_content": thinking} if self._capture_content else {}
                span.add_event("gen_ai.choice", attributes=attrs)
