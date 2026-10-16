"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from lightspeed_agentic.providers.model_router import (
    ModelCreator,
    ModelParameters,
    ResolvedModelRoute,
)
from lightspeed_agentic.types import (
    AgentProvider,
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
)


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

    @property
    def name(self) -> str:
        return "mock"

    async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
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


def make_model_route(
    create_model: ModelCreator, *, model: str = "claude-sonnet-4-6"
) -> ResolvedModelRoute:
    """Build an explicit route around an offline test model constructor."""
    return ResolvedModelRoute(
        route="anthropic",
        model=model,
        create_model=create_model,
        model_parameters=ModelParameters(model=model),
        telemetry_provider="anthropic",
        structured_output_method="function_calling",
    )


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()
