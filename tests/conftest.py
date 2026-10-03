"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from lightspeed_agentic.types import (
    AgentProvider,
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
)


class AuditRecorder:
    """Collect adapter observations without exporting spans."""

    def __init__(self) -> None:
        self.inference_starts: list[tuple[object, dict[str, Any]]] = []
        self.inference_ends: list[tuple[object, dict[str, Any]]] = []
        self.tool_starts: list[tuple[object, dict[str, Any]]] = []
        self.tool_ends: list[tuple[object, dict[str, Any]]] = []

    def start_inference(self, **attributes: Any) -> object:
        span = object()
        self.inference_starts.append((span, attributes))
        return span

    def end_inference(self, span: object, **attributes: Any) -> None:
        self.inference_ends.append((span, attributes))

    def start_tool(self, **attributes: Any) -> object:
        span = object()
        self.tool_starts.append((span, attributes))
        return span

    def end_tool(self, span: object, **attributes: Any) -> None:
        self.tool_ends.append((span, attributes))


@pytest.fixture
def audit_recorder() -> AuditRecorder:
    return AuditRecorder()


class MockProvider(AgentProvider):
    """Provider that yields a configurable sequence of events."""

    def __init__(self, events: list[ProviderEvent] | None = None) -> None:
        self._events = events or [
            ResultEvent(
                text='{"success": true, "summary": "mock result"}',
                input_tokens=100,
                output_tokens=50,
            ),
        ]
        self.last_options: ProviderQueryOptions | None = None

    @property
    def name(self) -> str:
        return "mock"

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        self.last_options = options
        for event in self._events:
            yield event


class _InMemorySpanExporter(SpanExporter):
    def __init__(self) -> None:
        self._spans: list[Any] = []

    def export(self, spans: Any) -> SpanExportResult:
        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def get_finished_spans(self) -> list[Any]:
        return list(self._spans)


@pytest.fixture
def span_exporter():
    exporter = _InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace.set_tracer_provider(provider)
    yield exporter
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace.set_tracer_provider(TracerProvider())


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()
