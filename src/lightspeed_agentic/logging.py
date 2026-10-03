"""Normalized provider event logging without copying request or response content."""

from __future__ import annotations

import logging

from lightspeed_agentic.types import ProviderEvent

logger = logging.getLogger("lightspeed_agentic")

THINKING_BUF_FLUSH = 50_000


class EventLogger:
    """Log safe event metadata and aggregate token counts from provider events."""

    def __init__(self, phase: str) -> None:
        self._phase = phase
        self._thinking_len = 0

    def _flush_thinking(self) -> None:
        if self._thinking_len:
            logger.info(
                "[provider:%s] thinking: chars=%d",
                self._phase,
                self._thinking_len,
            )
            self._thinking_len = 0

    def log(self, event: ProviderEvent) -> None:
        match event.type:
            case "thinking_delta":
                self._thinking_len += len(event.thinking)
                if self._thinking_len >= THINKING_BUF_FLUSH:
                    self._flush_thinking()
            case "content_block_stop":
                self._flush_thinking()
            case "tool_call":
                self._flush_thinking()
                logger.info("[provider:%s] tool_use: %s", self._phase, event.name)
            case "tool_result":
                logger.info("[provider:%s] tool_result", self._phase)
            case "result":
                self._flush_thinking()
                logger.info(
                    "[provider:%s] result: tokens=%d",
                    self._phase,
                    event.input_tokens + event.output_tokens,
                )
