"""Tests for gen_ai.* Prometheus metrics."""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from lightspeed_agentic.run_agent import run_agent_query
from lightspeed_agentic.types import ResultEvent, ToolCallEvent, ToolResultEvent

from .conftest import MockProvider


def _sample(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


async def _run(provider: MockProvider) -> dict:
    return await run_agent_query(
        provider,
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=300_000,
    )


@pytest.mark.asyncio
async def test_run_records_token_usage() -> None:
    provider = MockProvider(
        events=[
            ResultEvent(
                text='{"success": true, "summary": "ok"}',
                cost_usd=0.01,
                input_tokens=100,
                output_tokens=50,
            ),
        ]
    )
    labels_in = {
        "gen_ai_token_type": "input",
        "gen_ai_request_model": "test-model",
        "gen_ai_provider_name": "mock",
        "gen_ai_operation_name": "chat",
    }
    labels_out = {
        "gen_ai_token_type": "output",
        "gen_ai_request_model": "test-model",
        "gen_ai_provider_name": "mock",
        "gen_ai_operation_name": "chat",
    }
    before_in_count = _sample("gen_ai_client_token_usage_count", labels_in)
    before_out_count = _sample("gen_ai_client_token_usage_count", labels_out)
    before_in_sum = _sample("gen_ai_client_token_usage_sum", labels_in)
    before_out_sum = _sample("gen_ai_client_token_usage_sum", labels_out)

    result = await _run(provider)
    assert result.output["success"] is True

    assert _sample("gen_ai_client_token_usage_count", labels_in) == before_in_count + 1
    assert _sample("gen_ai_client_token_usage_count", labels_out) == before_out_count + 1
    assert _sample("gen_ai_client_token_usage_sum", labels_in) == before_in_sum + 100
    assert _sample("gen_ai_client_token_usage_sum", labels_out) == before_out_sum + 50


@pytest.mark.asyncio
async def test_run_records_operation_duration() -> None:
    labels = {
        "gen_ai_request_model": "test-model",
        "gen_ai_provider_name": "mock",
        "gen_ai_operation_name": "chat",
    }
    before_count = _sample("gen_ai_client_operation_duration_seconds_count", labels)
    before_sum = _sample("gen_ai_client_operation_duration_seconds_sum", labels)

    await _run(MockProvider())

    assert _sample("gen_ai_client_operation_duration_seconds_count", labels) == before_count + 1
    delta = _sample("gen_ai_client_operation_duration_seconds_sum", labels) - before_sum
    assert delta > 0, "operation duration must be positive"


@pytest.mark.asyncio
async def test_run_records_tool_duration() -> None:
    events = [
        ToolCallEvent(name="bash", input="ls"),
        ToolResultEvent(output="file.txt"),
        ResultEvent(
            text='{"success": true, "summary": "done"}',
            cost_usd=0.01,
            input_tokens=10,
            output_tokens=5,
        ),
    ]
    labels = {"gen_ai_tool_name": "bash"}
    before_count = _sample("gen_ai_execute_tool_duration_seconds_count", labels)
    before_sum = _sample("gen_ai_execute_tool_duration_seconds_sum", labels)

    await _run(MockProvider(events=events))

    assert _sample("gen_ai_execute_tool_duration_seconds_count", labels) == before_count + 1
    delta = _sample("gen_ai_execute_tool_duration_seconds_sum", labels) - before_sum
    assert delta > 0, "tool duration must be positive"


@pytest.mark.asyncio
async def test_empty_response_records_metrics() -> None:
    labels = {
        "gen_ai_request_model": "test-model",
        "gen_ai_provider_name": "mock",
        "gen_ai_operation_name": "chat",
    }
    before = _sample("gen_ai_client_operation_duration_seconds_count", labels)

    result = await _run(MockProvider(events=[ResultEvent(text="")]))
    assert result.output["success"] is False

    assert _sample("gen_ai_client_operation_duration_seconds_count", labels) == before + 1


@pytest.mark.asyncio
async def test_zero_tokens_not_recorded() -> None:
    labels_in = {
        "gen_ai_token_type": "input",
        "gen_ai_request_model": "test-model",
        "gen_ai_provider_name": "mock",
        "gen_ai_operation_name": "chat",
    }
    before = _sample("gen_ai_client_token_usage_count", labels_in)

    await _run(MockProvider(events=[ResultEvent(text="")]))

    assert _sample("gen_ai_client_token_usage_count", labels_in) == before
