"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

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


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()
