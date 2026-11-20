"""Tests for event types and AgentProvider interface."""

import pytest

from lightspeed_agentic.mcp import AdmittedMCPProviderServer
from lightspeed_agentic.types import (
    ContentBlockStopEvent,
    ProviderQueryOptions,
    ResultEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    stringify,
)


def test_text_delta_event():
    e = TextDeltaEvent(text="hello")
    assert e.type == "text_delta"
    assert e.text == "hello"


def test_thinking_delta_event():
    e = ThinkingDeltaEvent(thinking="reasoning")
    assert e.type == "thinking_delta"
    assert e.thinking == "reasoning"


def test_content_block_stop_event():
    e = ContentBlockStopEvent()
    assert e.type == "content_block_stop"


def test_tool_call_event():
    e = ToolCallEvent(name="bash", input='{"command": "ls"}')
    assert e.type == "tool_call"
    assert e.name == "bash"


def test_tool_result_event():
    e = ToolResultEvent(output="file1.txt")
    assert e.type == "tool_result"
    assert e.output == "file1.txt"


def test_result_event():
    e = ResultEvent(text="done", input_tokens=100, output_tokens=50)
    assert e.type == "result"
    assert e.input_tokens == 100


def test_events_are_frozen():
    e = TextDeltaEvent(text="hello")
    with pytest.raises(AttributeError):
        e.text = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        ("", ""),
        ("hello", "hello"),
        (0, "0"),
        (False, "false"),
        ([], "[]"),
        ({}, "{}"),
        (1, "1"),
        (True, "true"),
        ({"a": 1}, '{"a": 1}'),
        ([0], "[0]"),
    ],
)
def test_stringify(value, expected):
    assert stringify(value) == expected


def test_query_options_defaults():
    opts = ProviderQueryOptions(
        prompt="test",
        system_prompt="system",
        max_turns=10,
        cwd="/workspace",
    )
    assert opts.output_schema is None
    assert opts.tool_output_inspection_enabled is True


def test_query_options_accepts_admitted_provider_servers():
    server = AdmittedMCPProviderServer(
        name="openshift",
        url="https://mcp.example/mcp",
        allowed_tool_names=("get_pod",),
    )

    opts = ProviderQueryOptions(
        prompt="test",
        system_prompt="system",
        max_turns=10,
        cwd="/workspace",
        mcp_servers=[server],
    )

    assert opts.mcp_servers == [server]


def test_result_event_extended_usage_defaults():
    """Absent extended usage stays unset rather than zero-filled."""
    e = ResultEvent(text="done")
    assert e.reasoning_tokens is None
    assert e.tool_tokens is None
    assert e.peak_input_tokens is None
    assert e.response_model == ""
