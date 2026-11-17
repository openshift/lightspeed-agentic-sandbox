"""Normalized provider event logging — maps to lightspeed-agent/src/providers/logging.ts."""

from __future__ import annotations

import logging

from lightspeed_agentic.types import ProviderEvent

logger = logging.getLogger("lightspeed_agentic")

MAX_THINKING_LOG = 2000
MAX_RESULT_LOG = 500
THINKING_BUF_FLUSH = MAX_THINKING_LOG


class EventLogger:
    """Buffers thinking deltas and logs them as complete blocks."""

    def __init__(self, phase: str) -> None:
        self._phase = phase
        self._thinking_buf: list[str] = []
        self._thinking_len = 0
        self._continuation = False

    def flush(self) -> None:
        """Emit any buffered thinking (used on timeout/error exits)."""
        self._flush_thinking()

    def _flush_thinking(self, *, end_of_block: bool = True) -> None:
        if self._thinking_buf:
            text = "".join(self._thinking_buf)
            if text:
                marker = " (cont.)" if self._continuation else ""
                logger.info("[provider:%s] thinking%s: %s", self._phase, marker, text)
            self._thinking_buf.clear()
            self._thinking_len = 0
        if end_of_block:
            self._continuation = False

    def log(self, event: ProviderEvent) -> None:
        match event.type:
            case "thinking_delta":
                remaining = event.thinking
                while remaining:
                    if self._thinking_len == THINKING_BUF_FLUSH:
                        # Cap split mid-block: the next flush continues this block.
                        self._flush_thinking(end_of_block=False)
                        self._continuation = True
                    available = THINKING_BUF_FLUSH - self._thinking_len
                    part, remaining = remaining[:available], remaining[available:]
                    self._thinking_buf.append(part)
                    self._thinking_len += len(part)
            case "content_block_stop":
                self._flush_thinking()
            case "tool_call":
                self._flush_thinking()
                logger.info("[provider:%s] tool_use: %s", self._phase, event.name)
            case "tool_result":
                self._flush_thinking()
                logger.info("[provider:%s] tool_result", self._phase)
            case "result":
                self._flush_thinking()
                logger.info(
                    "[provider:%s] result: tokens=%s",
                    self._phase,
                    (event.input_tokens or 0) + (event.output_tokens or 0)
                    if event.input_tokens is not None or event.output_tokens is not None
                    else "not_reported",
                )
                if event.text:
                    logger.info(
                        "[provider:%s] output: %s", self._phase, event.text[:MAX_RESULT_LOG]
                    )
