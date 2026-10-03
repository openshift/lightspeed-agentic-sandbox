"""Fail-closed orchestration for complete tool-result inspection."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from opentelemetry.trace import Status, StatusCode

from lightspeed_agentic.inspection.chunking import TokenCodec, chunk_tool_result
from lightspeed_agentic.inspection.errors import (
    CLASSIFIER_FAILURE_MESSAGE,
    ClassifierResponseError,
    InspectionError,
    provider_error_metadata,
)
from lightspeed_agentic.inspection.models import ClassifierDecision, ClassifierRequest

logger = logging.getLogger(__name__)


class ClassifierClient(Protocol):
    """Async boundary for a provider-backed safety classifier."""

    async def classify(
        self,
        request: ClassifierRequest,
        *,
        deadline: float | None = None,
    ) -> ClassifierDecision: ...


@dataclass(frozen=True)
class InspectionResult:
    passed: bool
    category: str = "none"


async def _classify_with_retries(
    client: ClassifierClient,
    request: ClassifierRequest,
    *,
    deadline: float | None,
    monotonic: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
) -> tuple[ClassifierDecision, int]:
    for attempt in range(3):
        provider_status_code: int | None = None
        provider_error_type: str | None = None
        provider_error_reason: str | None = None
        try:
            remaining = None if deadline is None else deadline - monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError
            raw_decision: Any = client.classify(request, deadline=deadline)
            if remaining is not None:
                raw_decision = await asyncio.wait_for(raw_decision, remaining)
            else:
                raw_decision = await raw_decision
            decision = ClassifierDecision.model_validate(raw_decision)
        except Exception as exc:
            (
                provider_status_code,
                provider_error_type,
                provider_error_reason,
            ) = provider_error_metadata(exc)
            response_issue = (
                exc.issue
                if isinstance(exc, ClassifierResponseError)
                else "schema_mismatch"
                if type(exc).__name__ == "ValidationError"
                else None
            )
            failure_type = (
                "timeout"
                if isinstance(exc, (TimeoutError, asyncio.TimeoutError))
                else "invalid_response"
                if isinstance(exc, ValueError) or type(exc).__name__ == "ValidationError"
                else "provider_error"
            )
            if attempt == 2:
                raise InspectionError(
                    CLASSIFIER_FAILURE_MESSAGE,
                    failure_type=failure_type,
                    attempt_count=attempt + 1,
                    provider_status_code=provider_status_code,
                    provider_error_type=provider_error_type,
                    provider_error_reason=provider_error_reason,
                    response_issue=response_issue,
                ) from None
            delay = (0.5, 1.0)[attempt]
            if deadline is not None and monotonic() + delay >= deadline:
                raise InspectionError(
                    CLASSIFIER_FAILURE_MESSAGE,
                    failure_type="timeout",
                    attempt_count=attempt + 1,
                ) from None
            await sleep(delay)
            continue

        return decision, attempt + 1

    raise AssertionError("classifier retry loop did not return or raise")


async def inspect_tool_result(
    client: ClassifierClient,
    *,
    tool_name: str,
    result_type: str,
    value: Any,
    codec: TokenCodec,
    tool_call_id: str | None = None,
    correlation_attributes: Mapping[str, str] | None = None,
    context_window_tokens: int,
    instruction_tokens: int,
    output_tokens: int,
    enabled: bool = True,
    deadline: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    tracer: Any | None = None,
    runtime: str = "deepagents",
    provider: str = "",
    model: str = "",
) -> InspectionResult:
    """Inspect every result chunk in source order, failing closed on errors."""
    if not enabled:
        return InspectionResult(passed=True)
    if result_type not in ("result", "error"):
        raise ValueError("result_type must be 'result' or 'error'")

    chunks = chunk_tool_result(
        value,
        codec=codec,
        context_window_tokens=context_window_tokens,
        instruction_tokens=instruction_tokens,
        output_tokens=output_tokens,
    )
    if tracer is None:
        from lightspeed_agentic.tracing import get_tracer

        tracer = get_tracer()

    for chunk in chunks:
        attributes: dict[str, Any] = {
            "inspection.runtime": runtime,
            "inspection.result_type": result_type,
            "inspection.chunk_count": chunk.chunk_count,
            "inspection.chunk_index": chunk.chunk_index,
            "inspection.enabled": True,
            "tool.name": tool_name,
        }
        for key in ("agenticrun.uid", "agenticrun.phase"):
            correlation_value = (correlation_attributes or {}).get(key)
            if correlation_value:
                attributes[key] = correlation_value
        if provider:
            attributes["llm.provider"] = provider
        if model:
            attributes["llm.model"] = model
        if tool_call_id:
            attributes["gen_ai.tool.call.id"] = tool_call_id
        with tracer.start_as_current_span(
            "tool_result.inspection",
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                request = ClassifierRequest(
                    toolName=tool_name,
                    resultType=cast(Literal["result", "error"], result_type),
                    chunkIndex=chunk.chunk_index,
                    chunkCount=chunk.chunk_count,
                    content=chunk.content,
                )
                decision, attempt_count = await _classify_with_retries(
                    client,
                    request,
                    deadline=deadline,
                    monotonic=monotonic,
                    sleep=sleep,
                )
            except asyncio.CancelledError:
                span.set_attribute("inspection.outcome", "classifier_error")
                span.set_attribute("inspection.failure_type", "cancelled")
                span.set_attribute("error.type", "cancelled")
                span.set_status(Status(StatusCode.ERROR))
                logger.warning(
                    "tool result safety inspection failed",
                    extra={
                        "inspection.outcome": "classifier_error",
                        "inspection.failure_type": "cancelled",
                    },
                )
                raise
            except InspectionError as exc:
                span.set_attribute("inspection.attempt_count", exc.attempt_count or 3)
                span.set_attribute("inspection.outcome", "classifier_error")
                span.set_attribute("inspection.failure_type", exc.failure_type)
                span.set_attribute("error.type", exc.failure_type)
                if exc.response_issue is not None:
                    span.set_attribute("inspection.response_issue", exc.response_issue)
                if exc.provider_status_code is not None:
                    span.set_attribute("inspection.provider_status_code", exc.provider_status_code)
                if exc.provider_error_type is not None:
                    span.set_attribute("inspection.provider_error_type", exc.provider_error_type)
                if exc.provider_error_reason is not None:
                    span.set_attribute(
                        "inspection.provider_error_reason", exc.provider_error_reason
                    )
                span.set_status(Status(StatusCode.ERROR))
                extra: dict[str, object] = {
                    "inspection.outcome": "classifier_error",
                    "inspection.failure_type": exc.failure_type,
                }
                if exc.provider_status_code is not None:
                    extra["inspection.provider_status_code"] = exc.provider_status_code
                if exc.provider_error_type is not None:
                    extra["inspection.provider_error_type"] = exc.provider_error_type
                if exc.provider_error_reason is not None:
                    extra["inspection.provider_error_reason"] = exc.provider_error_reason
                logger.warning("tool result safety inspection failed", extra=extra)
                raise

            span.set_attribute("inspection.attempt_count", attempt_count)
            if decision.injection_detected:
                span.set_attribute("inspection.outcome", "malicious")
                span.set_attribute("inspection.category", decision.category)
                logger.warning(
                    "malicious tool result detected",
                    extra={
                        "inspection.outcome": "malicious",
                        "inspection.category": decision.category,
                    },
                )
                return InspectionResult(passed=False, category=decision.category)
            span.set_attribute("inspection.outcome", "benign")

    return InspectionResult(passed=True)
