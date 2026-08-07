"""LLM-as-judge utility for troubleshooting scenario e2e validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tests.e2e.runner import run_query

logger = logging.getLogger(__name__)

JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["passed", "reasoning"],
    "additionalProperties": False,
}

_JUDGE_SYSTEM_PROMPT = (
    "You are an evaluator for Kubernetes troubleshooting scenarios. "
    "You will receive a scenario ID, the original troubleshooting request, "
    "a list of expected keywords, and the AI agent's output. "
    "Judge whether the agent correctly identified the root cause and provided "
    "relevant diagnostic information for the described problem. "
    "Respond with a JSON object only (no markdown): "
    "passed (boolean — true if the agent's output correctly diagnoses the problem) "
    "and reasoning (string — a brief explanation of your verdict)."
)


@dataclass(frozen=True)
class JudgeInput:
    scenario_id: str
    request: str
    expected_keywords: list[str]
    agent_output: str


@dataclass(frozen=True)
class JudgeResult:
    passed: bool
    reasoning: str
    error: str | None = None


def evaluate_scenario(server_url: str, judge_input: JudgeInput) -> JudgeResult:
    """Evaluate a troubleshooting scenario output via the sandbox LLM.

    Never raises — returns JudgeResult with error field set on failure.
    """
    query = (
        f"Scenario: {judge_input.scenario_id}\n"
        f"Request: {judge_input.request}\n"
        f"Expected keywords: {', '.join(judge_input.expected_keywords)}\n\n"
        f"Agent output:\n{judge_input.agent_output}"
    )

    try:
        result = run_query(
            server_url,
            query,
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            output_schema=JUDGE_OUTPUT_SCHEMA,
        )
    except Exception as exc:
        return JudgeResult(passed=False, reasoning="", error=f"run_query exception: {exc}")

    if result.error is not None:
        return JudgeResult(passed=False, reasoning="", error=f"transport error: {result.error}")

    if result.status_code != 200:
        return JudgeResult(
            passed=False,
            reasoning="",
            error=f"HTTP {result.status_code}: {result.raw_text[:200]}",
        )

    body = result.body
    if not isinstance(body.get("passed"), bool) or not isinstance(body.get("reasoning"), str):
        return JudgeResult(
            passed=False,
            reasoning="",
            error=f"malformed response: missing passed/reasoning in {body!r}",
        )

    verdict = JudgeResult(passed=body["passed"], reasoning=body["reasoning"])
    logger.info(
        "judge verdict for %s: passed=%s reasoning=%s",
        judge_input.scenario_id,
        verdict.passed,
        verdict.reasoning,
    )
    return verdict
