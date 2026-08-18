from __future__ import annotations

import os

from lightspeed_agentic.guardrails.types import GuardrailsConfig


def load_guardrails_config() -> GuardrailsConfig:
    enabled = os.environ.get("LIGHTSPEED_GUARDRAILS_ENABLED", "").strip().lower() == "true"
    llm_judge = os.environ.get("LIGHTSPEED_GUARDRAILS_LLM_JUDGE", "true").strip().lower() != "false"
    judge_model = os.environ.get("LIGHTSPEED_GUARDRAILS_JUDGE_MODEL", "").strip()
    judge_timeout = os.environ.get("LIGHTSPEED_GUARDRAILS_JUDGE_TIMEOUT", "").strip()

    return GuardrailsConfig(
        enabled=enabled,
        llm_judge_enabled=llm_judge,
        judge_model=judge_model or "claude-haiku-4-5",
        judge_timeout_ms=int(judge_timeout) if judge_timeout else 5000,
    )
