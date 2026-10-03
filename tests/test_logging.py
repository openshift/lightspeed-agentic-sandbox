"""Tests that provider diagnostics retain metadata without prompt content."""

from __future__ import annotations

import logging

from lightspeed_agentic.logging import EventLogger
from lightspeed_agentic.types import (
    ContentBlockStopEvent,
    ResultEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)


def test_event_logger_keeps_diagnostics_but_not_payloads(caplog) -> None:
    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    event_logger = EventLogger("analysis")
    events = (
        ThinkingDeltaEvent(thinking="PRIVATE-REASONING"),
        ContentBlockStopEvent(),
        TextDeltaEvent(text="PRIVATE-STREAM-OUTPUT"),
        ToolCallEvent(name="Bash", input="PRIVATE-TOOL-ARGUMENT", call_id="call-1"),
        ToolResultEvent(output="PRIVATE-TOOL-RESULT", call_id="call-1"),
        ResultEvent(
            text="PRIVATE-TERMINAL-OUTPUT",
            input_tokens=3,
            output_tokens=2,
        ),
    )

    for event in events:
        event_logger.log(event)

    assert "[provider:analysis] thinking: chars=" in caplog.text
    assert "[provider:analysis] tool_use: Bash" in caplog.text
    assert "[provider:analysis] tool_result" in caplog.text
    assert "[provider:analysis] result: tokens=5" in caplog.text
    for secret in (
        "PRIVATE-REASONING",
        "PRIVATE-STREAM-OUTPUT",
        "PRIVATE-TOOL-ARGUMENT",
        "PRIVATE-TOOL-RESULT",
        "PRIVATE-TERMINAL-OUTPUT",
    ):
        assert secret not in caplog.text
