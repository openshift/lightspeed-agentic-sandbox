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
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, StatusCode

from lightspeed_agentic.audit import AuditLogger, resolve_provider_name
from lightspeed_agentic.inspection.errors import ToolResultSafetyInspectionFailed
from lightspeed_agentic.logging import EventLogger
from lightspeed_agentic.mcp import AdmittedMCPProviderServer
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
    timed_out: bool = False


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
    timeout_seconds: int,
    mcp_servers: list[AdmittedMCPProviderServer] | None = None,
    reasoning_config: dict[str, Any] | None = None,
    tool_output_inspection_enabled: bool = True,
    agenticrun_uid: str = "",
    traceparent: str | None = None,
    step: str = "",
) -> AgentResult:
    """Run the provider agent and return structured output for Result CR publishing.

    When a valid ``traceparent`` is supplied from the operator ``TRACEPARENT``
    env, ``invoke_agent`` is its child and model/tool spans are its children.
    Without a valid parent, a new trace ID is generated (graceful degradation).
    """
    if context:
        try:
            prefix = format_context_prefix(context)
        except ContextFormatError as exc:
            return AgentResult(output={"success": False, "summary": str(exc)})
        prompt = f"{prefix}\n\n{prompt}"

    _, traceparent_context = parse_traceparent(traceparent)
    agent_parent_context = traceparent_context if traceparent_context is not None else Context()
    tracer = get_tracer()
    provider_name = resolve_provider_name(provider.name)
    audit_logger = AuditLogger(
        phase=step,
        model=model,
        provider=provider_name,
        agenticrun_uid=agenticrun_uid,
    )

    run_started = time.monotonic()
    text = ""
    input_tokens = 0
    output_tokens = 0
    span_attrs: dict[str, Any] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "lightspeed",
        "gen_ai.provider.name": provider_name,
    }
    if model:
        span_attrs["gen_ai.request.model"] = model
    if output_schema is not None:
        span_attrs["gen_ai.output.type"] = "json"
    if step:
        span_attrs["agenticrun.phase"] = step
    if agenticrun_uid:
        span_attrs["agenticrun.uid"] = agenticrun_uid
    agent_span = tracer.start_span(
        "invoke_agent lightspeed",
        kind=SpanKind.INTERNAL,
        context=agent_parent_context,
        attributes=span_attrs,
    )
    span_context = agent_span.get_span_context()
    trace_id = f"{span_context.trace_id:032x}" if span_context.is_valid else ""
    logger.info(
        "[agent] Starting query (model=%s, provider=%s, trace_id=%s)",
        model,
        provider_name,
        trace_id,
    )
    agent_context = trace.set_span_in_context(agent_span, agent_parent_context)
    audit_logger.set_parent_context(agent_context)
    if agent_span.is_recording():
        agent_span.set_attribute(
            "gen_ai.input.messages",
            json.dumps(
                [{"role": "user", "parts": [{"type": "text", "content": prompt}]}],
                ensure_ascii=False,
            ),
        )
        agent_span.set_attribute(
            "gen_ai.system_instructions",
            json.dumps([{"type": "text", "content": system_prompt}], ensure_ascii=False),
        )

    failure: BaseException | str | None = None

    def _set_agent_error(error: BaseException | str) -> None:
        if agent_span.is_recording():
            error_type = error if isinstance(error, str) else type(error).__name__
            agent_span.set_attribute("error.type", error_type)
            agent_span.set_status(StatusCode.ERROR)

    try:

        async def run() -> None:
            nonlocal text, input_tokens, output_tokens
            token = otel_context.attach(agent_context)
            try:
                result = provider.query(
                    ProviderQueryOptions(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model,
                        max_turns=max_turns,
                        allowed_tools=DEFAULT_ALLOWED_TOOLS,
                        cwd=skills_dir,
                        output_schema=output_schema,
                        mcp_servers=mcp_servers or [],
                        reasoning_config=reasoning_config,
                        tool_output_inspection_enabled=tool_output_inspection_enabled,
                        audit_logger=audit_logger,
                        deadline=time.monotonic() + timeout_seconds,
                    )
                )
                event_logger = EventLogger("run")
                async for event in result:
                    event_logger.log(event)
                    if event.type == "result":
                        text = event.text
                        input_tokens = event.input_tokens
                        output_tokens = event.output_tokens
                        break
            finally:
                otel_context.detach(token)

        await asyncio.wait_for(run(), timeout=timeout_seconds)

    except TimeoutError as exc:
        failure = exc
        _set_agent_error(exc)
        elapsed = time.monotonic() - run_started
        timeout_msg = (
            f"Agent invocation exceeded timeout of {timeout_seconds}s after {elapsed:.1f}s"
        )
        return AgentResult(
            output={
                "success": False,
                "summary": timeout_msg,
            },
            timed_out=True,
        )
    except ToolResultSafetyInspectionFailed as exc:
        failure = exc
        _set_agent_error(exc)
        raise
    except asyncio.CancelledError as exc:
        failure = exc
        _set_agent_error(exc)
        raise
    except Exception as exc:
        failure = exc
        _set_agent_error(exc)
        logger.error("[agent] query error (error_type=%s)", type(exc).__name__)
        return AgentResult(
            output={"success": False, "summary": f"Agent error: {exc}"},
        )
    except BaseException as exc:
        failure = exc
        _set_agent_error(exc)
        raise
    else:
        if not text:
            failure = "empty_response"
            _set_agent_error(failure)
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

        if parsed is not None:
            result = AgentResult(
                output={
                    "success": success,
                    "summary": parsed.get("summary", text),
                    **{k: v for k, v in parsed.items() if k not in ("success", "summary")},
                },
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            logger.info("[agent] query complete: success=%s", bool(success))
        else:
            result = AgentResult(
                output={"success": True, "summary": text},
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            logger.info("[agent] query complete (text response)")

        if agent_span.is_recording():
            agent_span.set_attribute(
                "gen_ai.output.messages",
                json.dumps(
                    [
                        {
                            "role": "assistant",
                            "parts": [
                                {
                                    "type": "text",
                                    "content": json.dumps(result.output, ensure_ascii=False),
                                }
                            ],
                            "finish_reason": "unknown",
                        }
                    ],
                    ensure_ascii=True,
                ),
            )
        return result
    finally:
        try:
            audit_logger.close(failure or "operation_cancelled")
        finally:
            agent_span.end()
