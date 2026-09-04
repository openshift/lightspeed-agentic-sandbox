"""Normalized provider event logging — maps to lightspeed-agent/src/providers/logging.ts."""

from __future__ import annotations

import logging

from lightspeed_agentic.types import ProviderEvent

logger = logging.getLogger("lightspeed_agentic")

MAX_THINKING_LOG = 2000
MAX_TOOL_INPUT_LOG = 500
MAX_TOOL_OUTPUT_LOG = 1000
MAX_RESULT_LOG = 500
THINKING_BUF_FLUSH = 50_000


class EventLogger:
    """Buffers thinking deltas and logs them as complete blocks."""

    def __init__(self, phase: str) -> None:
        self._phase = phase
        self._thinking_buf: list[str] = []
        self._thinking_len = 0

    def _flush_thinking(self) -> None:
        if self._thinking_buf:
            text = "".join(self._thinking_buf).strip()
            if text:
                logger.info("[provider:%s] thinking: %s", self._phase, text[:MAX_THINKING_LOG])
            self._thinking_buf.clear()
            self._thinking_len = 0

    def log(self, event: ProviderEvent) -> None:
        match event.type:
            case "thinking_delta":
                self._thinking_buf.append(event.thinking)
                self._thinking_len += len(event.thinking)
                if self._thinking_len >= THINKING_BUF_FLUSH:
                    self._flush_thinking()
            case "content_block_stop":
                self._flush_thinking()
            case "tool_call":
                self._flush_thinking()
                logger.info(
                    "[provider:%s] tool_use: %s(%s)",
                    self._phase,
                    event.name,
                    event.input[:MAX_TOOL_INPUT_LOG],
                )
            case "tool_result":
                logger.info(
                    "[provider:%s] tool_result: %s", self._phase, event.output[:MAX_TOOL_OUTPUT_LOG]
                )
            case "result":
                self._flush_thinking()
                logger.info(
                    "[provider:%s] result: tokens=%d",
                    self._phase,
                    event.input_tokens + event.output_tokens,
                )
                if event.text:
                    logger.info(
                        "[provider:%s] output: %s", self._phase, event.text[:MAX_RESULT_LOG]
                    )
