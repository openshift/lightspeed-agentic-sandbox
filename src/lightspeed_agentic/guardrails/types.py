from __future__ import annotations

import dataclasses
import enum


class Verdict(enum.Enum):
    PASS = "pass"  # noqa: S105
    BLOCK = "block"
    SANITIZE = "sanitize"
    SUSPICIOUS = "suspicious"


@dataclasses.dataclass(frozen=True)
class CheckResult:
    verdict: Verdict
    reason: str = ""
    confidence: float = 1.0
    layer: str = "heuristic"
    sanitized_output: str = ""


@dataclasses.dataclass(frozen=True)
class GuardrailsConfig:
    enabled: bool = False
    llm_judge_enabled: bool = True
    judge_model: str = "claude-haiku-4-5"
    judge_timeout_ms: int = 5000


@dataclasses.dataclass(frozen=True)
class GuardrailContext:
    original_query: str = ""
    target_namespaces: list[str] = dataclasses.field(default_factory=list)
