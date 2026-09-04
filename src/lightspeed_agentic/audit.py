"""Audit instrumentation — OTel spans and span events for compliance."""

from __future__ import annotations

import time
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, StatusCode

from lightspeed_agentic.metrics import tool_duration
from lightspeed_agentic.tracing import get_tracer
from lightspeed_agentic.types import ProviderEvent


class AuditLogger:
    """Map provider stream events to OTel tool spans and ``gen_ai.choice`` span events."""

    def __init__(
        self,
        *,
        phase: str,
        model: str,
        provider: str,
        enabled: bool = True,
        capture_content: bool = False,
        agenticrun_uid: str = "",
    ) -> None:
        """Configure audit emission for one agent run.

        ``phase`` is the workflow step (``analysis``, ``execution``, …) from
        ``result-template.kind``. When ``enabled`` is false, buffers are cleared
        without emitting span events.
        """
        self._phase = phase
        self._model = model
        self._provider = provider
        self._enabled = enabled
        self._capture_content = capture_content
        self._agenticrun_uid = agenticrun_uid
        self._text_buffer: list[str] = []
        self._thinking_buffer: list[str] = []
        self._tool_spans: dict[str, tuple[Any, float]] = {}
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
                call_id = event.call_id or f"_auto_{self._next_call_id}"
                self._next_call_id += 1
                tool_name = event.name or "unknown"
                attrs: dict[str, str] = {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": tool_name,
                    "gen_ai.tool.call.id": call_id,
                    "gen_ai.tool.type": "function",
                }
                if self._agenticrun_uid:
                    attrs["agenticrun.uid"] = self._agenticrun_uid
                if event.input:
                    attrs["tool.input"] = event.input
                span = self._tracer.start_span(
                    f"execute_tool {tool_name}",
                    kind=SpanKind.INTERNAL,
                    context=self._parent_context,
                    attributes=attrs,
                )
                self._tool_spans[call_id] = (span, time.monotonic())
            case "tool_result":
                call_id = event.call_id
                entry = self._tool_spans.pop(call_id, None) if call_id else None
                if entry is None and not call_id and self._tool_spans:
                    entry = self._tool_spans.pop(next(iter(self._tool_spans)))
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
                    if event.output:
                        tool_span.set_attribute("tool.output", event.output)
                    tool_span.set_status(StatusCode.OK)
                    tool_span.end()
            case "result":
                self._flush_buffers()

    def complete(
        self,
        *,
        success: bool,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
        response_model: str = "",
        span: Any = None,
    ) -> None:
        """Flush buffers, close open tool spans, and stamp usage on the inference span."""
        self._flush_buffers(span)
        for _call_id, (tool_span, start) in self._tool_spans.items():
            tool_name = (
                tool_span.attributes.get("gen_ai.tool.name", "unknown")
                if hasattr(tool_span, "attributes")
                else "unknown"
            )
            tool_duration.labels(gen_ai_tool_name=tool_name).observe(time.monotonic() - start)
            tool_span.set_status(StatusCode.ERROR, "tool span not closed by result event")
            tool_span.end()
        self._tool_spans.clear()
        if span is not None and span.is_recording():
            if response_model:
                span.set_attribute("gen_ai.response.model", response_model)
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            if reasoning_tokens:
                span.set_attribute("gen_ai.usage.reasoning_tokens", reasoning_tokens)
            if not success:
                span.set_status(StatusCode.ERROR, "agent run failed")

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
