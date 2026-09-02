"""Shared agent execution for the batch entrypoint."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from lightspeed_agentic.audit import AuditLogger
from lightspeed_agentic.logging import EventLogger
from lightspeed_agentic.mcp import ResolvedMCPServer
from lightspeed_agentic.metrics import operation_duration, token_usage
from lightspeed_agentic.tools import DEFAULT_ALLOWED_TOOLS
from lightspeed_agentic.tracing import get_tracer, parse_traceparent
from lightspeed_agentic.types import AgentProvider, ProviderQueryOptions

logger = logging.getLogger("lightspeed_agentic")


@dataclass
class AgentResult:
    """Wraps agent output dict with token counts for Result CR publishing."""

    output: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


class ContextFormatError(ValueError):
    """``context`` JSON is present but missing fields required for prefix formatting."""


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContextFormatError(f"Invalid context: {path} must be a JSON object")
    return value


def _require_non_empty_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContextFormatError(f"Invalid context: {path} must be a non-empty string")
    return value


def format_context_prefix(context: dict[str, Any]) -> str:
    """Format context fields as a prefix block prepended to the query text."""
    if not isinstance(context, dict):
        raise ContextFormatError("Invalid context: must be a JSON object")

    lines: list[str] = ["[context]"]

    namespaces = context.get("targetNamespaces")
    if namespaces:
        if not isinstance(namespaces, list):
            raise ContextFormatError("Invalid context: targetNamespaces must be a list")
        lines.append(f"Target namespaces: {', '.join(str(ns) for ns in namespaces)}")

    if (attempt := context.get("attempt")) is not None:
        lines.append(f"Attempt: {attempt} of max")

    prev = context.get("previousAttempts")
    if prev:
        if not isinstance(prev, list):
            raise ContextFormatError("Invalid context: previousAttempts must be a list")
        lines.append("Previous attempts:")
        for i, entry in enumerate(prev):
            if not isinstance(entry, dict):
                raise ContextFormatError(
                    f"Invalid context: previousAttempts[{i}] must be a JSON object"
                )
            attempt_no = entry.get("attempt")
            if attempt_no is None:
                raise ContextFormatError(f"Invalid context: previousAttempts[{i}] missing attempt")
            reason = f": {entry['failureReason']}" if entry.get("failureReason") else ""
            lines.append(f"  Attempt {attempt_no}{reason}")

    opt = context.get("approvedOption")
    if opt is not None:
        opt = _require_mapping(opt, "approvedOption")
        title = _require_non_empty_str(opt.get("title"), "approvedOption.title")
        diagnosis = _require_mapping(opt.get("diagnosis"), "approvedOption.diagnosis")
        root_cause = _require_non_empty_str(
            diagnosis.get("rootCause"),
            "approvedOption.diagnosis.rootCause",
        )
        plan = _require_mapping(opt.get("remediationPlan"), "approvedOption.remediationPlan")
        plan_description = _require_non_empty_str(
            plan.get("description"),
            "approvedOption.remediationPlan.description",
        )
        lines.append("")
        lines.append("=== APPROVED REMEDIATION (execute ONLY these actions) ===")
        lines.append(f"Title: {title}")
        lines.append(f"Diagnosis: {root_cause}")
        lines.append(f"Plan: {plan_description}")
        lines.append(f"Reversible: {plan.get('reversible', 'unknown')}")
        actions = plan.get("actions")
        if actions:
            if not isinstance(actions, list):
                raise ContextFormatError(
                    "Invalid context: approvedOption.remediationPlan.actions must be a list"
                )
            lines.append("Actions to execute:")
            for j, action in enumerate(actions):
                if not isinstance(action, dict):
                    raise ContextFormatError(
                        f"Invalid context: approvedOption.remediationPlan.actions[{j}] "
                        "must be a JSON object"
                    )
                action_type = _require_non_empty_str(
                    action.get("type"),
                    f"approvedOption.remediationPlan.actions[{j}].type",
                )
                action_description = _require_non_empty_str(
                    action.get("description"),
                    f"approvedOption.remediationPlan.actions[{j}].description",
                )
                if cmd := action.get("command"):
                    lines.append(f"  - [{action_type}] {cmd} — {action_description}")
                else:
                    lines.append(f"  - [{action_type}] {action_description}")
        lines.append("=== DO NOT perform any actions beyond what is listed above ===")
        lines.append("")

    lines.append("[/context]")
    return "\n".join(lines)


async def run_agent_query(
    provider: AgentProvider,
    *,
    prompt: str,
    system_prompt: str,
    output_schema: dict[str, Any] | None,
    context: dict[str, Any] | None,
    skills_dir: str,
    model: str,
    max_turns: int,
    timeout_ms: int,
    mcp_servers: list[ResolvedMCPServer] | None = None,
    reasoning_config: dict[str, Any] | None = None,
    audit_enabled: bool = False,
    capture_content: bool = False,
    agenticrun_uid: str = "",
    traceparent: str | None = None,
    step: str = "analysis",
) -> AgentResult:
    """Run the provider agent and return structured output for Result CR publishing.

    When ``traceparent`` is set (W3C value from operator ``TRACEPARENT`` env),
    inference spans are children of the operator phase span. When unset or
    invalid, a new trace ID is generated (graceful degradation).
    """
    if context:
        try:
            prefix = format_context_prefix(context)
        except ContextFormatError as exc:
            return AgentResult(output={"success": False, "summary": str(exc)})
        prompt = f"{prefix}\n\n{prompt}"

    trace_id, trace_ctx = parse_traceparent(traceparent)
    tracer = get_tracer()
    audit_logger = AuditLogger(
        phase=step,
        model=model,
        provider=provider.name,
        enabled=audit_enabled,
        capture_content=capture_content,
        agenticrun_uid=agenticrun_uid,
    )

    logger.info(
        "[agent] Starting query (model=%s, provider=%s, trace_id=%s)",
        model,
        provider.name,
        trace_id,
    )

    start_time = time.monotonic()
    text = ""
    cost = 0.0
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    response_model = ""

    otel_provider_name = {"deepagents": "anthropic", "gemini": "google"}.get(
        provider.name, provider.name
    )
    span_attrs: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": model,
        "gen_ai.provider.name": otel_provider_name,
        "agenticrun.phase": step,
    }
    if agenticrun_uid:
        span_attrs["agenticrun.uid"] = agenticrun_uid
    chat_span = tracer.start_span(
        f"chat {model}",
        kind=SpanKind.CLIENT,
        context=trace_ctx,
        attributes=span_attrs,
    )
    span_ctx = trace.set_span_in_context(chat_span)
    audit_logger.set_parent_context(span_ctx)

    def _record_metrics(*, in_tokens: int, out_tokens: int, elapsed: float) -> None:
        if in_tokens:
            token_usage.labels(
                gen_ai_token_type="input",  # noqa: S106
                gen_ai_request_model=model,
                gen_ai_provider_name=provider.name,
                gen_ai_operation_name="chat",
            ).observe(in_tokens)
        if out_tokens:
            token_usage.labels(
                gen_ai_token_type="output",  # noqa: S106
                gen_ai_request_model=model,
                gen_ai_provider_name=provider.name,
                gen_ai_operation_name="chat",
            ).observe(out_tokens)
        operation_duration.labels(
            gen_ai_request_model=model,
            gen_ai_provider_name=provider.name,
            gen_ai_operation_name="chat",
        ).observe(elapsed)

    try:

        async def run() -> None:
            nonlocal text, cost, input_tokens, output_tokens, reasoning_tokens, response_model
            token = otel_context.attach(span_ctx)
            try:
                result = provider.query(
                    ProviderQueryOptions(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model,
                        max_turns=max_turns,
                        max_budget_usd=5.0,
                        allowed_tools=DEFAULT_ALLOWED_TOOLS,
                        cwd=skills_dir,
                        output_schema=output_schema,
                        mcp_servers=mcp_servers or [],
                        reasoning_config=reasoning_config,
                    )
                )
                event_logger = EventLogger("run")
                async for event in result:
                    event_logger.log(event)
                    audit_logger.process_event(event)
                    if event.type == "result":
                        text = event.text
                        cost = event.cost_usd
                        input_tokens = event.input_tokens
                        output_tokens = event.output_tokens
                        reasoning_tokens = event.reasoning_tokens
                        response_model = event.response_model
                        break
            finally:
                otel_context.detach(token)

        await asyncio.wait_for(run(), timeout=timeout_ms / 1000)

    except TimeoutError:
        audit_logger.complete(
            success=False,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0,
            span=chat_span,
        )
        return AgentResult(
            output={"success": False, "summary": f"Agent timed out after {timeout_ms}ms"},
        )
    except Exception as exc:
        audit_logger.complete(
            success=False,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0,
            span=chat_span,
        )
        logger.exception("[agent] query error")
        return AgentResult(
            output={"success": False, "summary": f"Agent error: {exc}"},
        )
    else:
        if not text:
            audit_logger.complete(
                success=False,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cost_usd=cost,
                response_model=response_model,
                span=chat_span,
            )
            return AgentResult(
                output={"success": False, "summary": "Agent returned empty response"},
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise TypeError("expected dict")
            success = parsed.get("success", True)
        except (json.JSONDecodeError, TypeError):
            parsed = None
            success = True

        audit_logger.complete(
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=cost,
            response_model=response_model,
            span=chat_span,
        )

        if parsed is not None:
            logger.info("[agent] query complete: success=%s, cost=$%.4f", success, cost)
            return AgentResult(
                output={
                    "success": success,
                    "summary": parsed.get("summary", text),
                    **{k: v for k, v in parsed.items() if k not in ("success", "summary")},
                },
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        logger.info("[agent] query complete (text response), cost=$%.4f", cost)
        return AgentResult(
            output={"success": True, "summary": text},
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    finally:
        chat_span.end()
        _record_metrics(
            in_tokens=input_tokens,
            out_tokens=output_tokens,
            elapsed=time.monotonic() - start_time,
        )
