"""GuardrailsChecker — orchestrates heuristic scan then optional LLM judge."""

from __future__ import annotations

import logging
import time

from lightspeed_agentic.guardrails.heuristics import check_post_execution, check_pre_execution
from lightspeed_agentic.guardrails.types import (
    CheckResult,
    GuardrailContext,
    GuardrailsConfig,
    Verdict,
)

logger = logging.getLogger(__name__)


class GuardrailsChecker:
    def __init__(self, config: GuardrailsConfig) -> None:
        self._config = config

    async def check_tool_request(
        self, command: str, context: GuardrailContext
    ) -> CheckResult:
        start = time.monotonic()
        result = check_pre_execution(command, context)
        result = await self._maybe_escalate_pre(result, command, context)
        self._log_check("pre_execution", command, result, time.monotonic() - start)
        return result

    async def check_tool_output(
        self, output: str, command: str, context: GuardrailContext
    ) -> CheckResult:
        start = time.monotonic()
        result = check_post_execution(output, command, context)
        result = await self._maybe_escalate_post(result, output, command, context)
        self._log_check("post_execution", command, result, time.monotonic() - start)
        return result

    async def _maybe_escalate_pre(
        self, result: CheckResult, command: str, context: GuardrailContext
    ) -> CheckResult:
        if result.verdict != Verdict.SUSPICIOUS:
            return result
        if not self._config.llm_judge_enabled:
            return CheckResult(
                verdict=Verdict.BLOCK,
                reason=f"Suspicious (no judge): {result.reason}",
                layer="heuristic",
            )
        from lightspeed_agentic.guardrails.llm_judge import judge_tool_request

        return await judge_tool_request(command, context, self._config)

    async def _maybe_escalate_post(
        self,
        result: CheckResult,
        output: str,
        command: str,
        context: GuardrailContext,
    ) -> CheckResult:
        if result.verdict != Verdict.SUSPICIOUS:
            return result
        if not self._config.llm_judge_enabled:
            return CheckResult(
                verdict=Verdict.BLOCK,
                reason=f"Suspicious (no judge): {result.reason}",
                layer="heuristic",
            )
        from lightspeed_agentic.guardrails.llm_judge import judge_tool_output

        return await judge_tool_output(output, command, context, self._config)

    def _log_check(
        self, phase: str, command: str, result: CheckResult, elapsed: float
    ) -> None:
        cmd_preview = command[:120] if len(command) > 120 else command
        logger.info(
            "[guardrails] %s verdict=%s layer=%s confidence=%.2f elapsed=%.1fms cmd=%s reason=%s",
            phase,
            result.verdict.value,
            result.layer,
            result.confidence,
            elapsed * 1000,
            cmd_preview,
            result.reason,
        )
