"""Query endpoint — POST /run.

The operator sends {query, systemPrompt, outputSchema, context, timeout_ms}
and the agent runs the LLM and returns {success, summary, ...structured fields}.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Request
from opentelemetry import context as otel_context

from lightspeed_agentic.audit import AuditLogger, derive_phase
from lightspeed_agentic.logging import EventLogger
from lightspeed_agentic.routes.models import RunRequest, RunResponse
from lightspeed_agentic.tools import DEFAULT_ALLOWED_TOOLS
from lightspeed_agentic.tracing import get_tracer, parse_traceparent
from lightspeed_agentic.types import AgentProvider, ProviderQueryOptions

logger = logging.getLogger("lightspeed_agentic")


def _format_context_prefix(context: dict[str, Any]) -> str:
    lines: list[str] = ["[context]"]

    if ns := context.get("targetNamespaces"):
        lines.append(f"Target namespaces: {', '.join(ns)}")
    if (attempt := context.get("attempt")) is not None:
        lines.append(f"Attempt: {attempt} of max")
    if prev := context.get("previousAttempts"):
        lines.append("Previous attempts:")
        for p in prev:
            reason = f": {p['failureReason']}" if p.get("failureReason") else ""
            lines.append(f"  Attempt {p['attempt']}{reason}")
    if opt := context.get("approvedOption"):
        lines.append("")
        lines.append("=== APPROVED REMEDIATION (execute ONLY these actions) ===")
        lines.append(f"Title: {opt['title']}")
        lines.append(f"Diagnosis: {opt['diagnosis']['rootCause']}")
        lines.append(f"Plan: {opt['proposal']['description']}")
        lines.append(
            f"Risk: {opt['proposal']['risk']}, Reversible: {opt['proposal']['reversible']}"
        )
        if actions := opt["proposal"].get("actions"):
            lines.append("Actions to execute:")
            for action in actions:
                lines.append(f"  - [{action['type']}] {action['description']}")
        lines.append("=== DO NOT perform any actions beyond what is listed above ===")
        lines.append("")

    lines.append("[/context]")
    return "\n".join(lines)


def register_query_routes(
    router: APIRouter,
    *,
    provider: AgentProvider,
    skills_dir: str,
    model: str,
    max_turns: int,
    default_timeout_ms: int,
    audit_enabled: bool = False,
) -> None:
    async def run_endpoint(req: RunRequest, request: Request) -> RunResponse:
        timeout = req.timeout_ms if req.timeout_ms is not None else default_timeout_ms
        system_prompt = req.systemPrompt or "You are an AI agent."

        prompt = req.query
        if req.context:
            prefix = _format_context_prefix(req.context)
            prompt = f"{prefix}\n\n{req.query}"

        traceparent = request.headers.get("traceparent")
        trace_id, trace_ctx = parse_traceparent(traceparent)
        tracer = get_tracer()

        phase = derive_phase(req.context)
        audit_logger = AuditLogger(
            trace_id=trace_id,
            phase=phase,
            model=model,
            provider=provider.name,
            log_enabled=audit_enabled,
        )

        logger.info(
            "[agent] Starting query (model=%s, provider=%s, trace_id=%s)",
            model,
            provider.name,
            trace_id,
        )

        try:
            text = ""
            cost = 0.0
            input_tokens = 0
            output_tokens = 0
            event_logger = EventLogger("run")

            async def run() -> None:
                nonlocal text, cost, input_tokens, output_tokens
                with tracer.start_as_current_span(
                    "agent.run",
                    context=trace_ctx,
                    attributes={"model": model, "provider": provider.name},
                ):
                    # Pass the current context to AuditLogger so tool spans
                    # are correctly parented under agent.run
                    audit_logger.set_parent_context(otel_context.get_current())
                    result = provider.query(
                        ProviderQueryOptions(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            model=model,
                            max_turns=max_turns,
                            max_budget_usd=5.0,
                            allowed_tools=DEFAULT_ALLOWED_TOOLS,
                            cwd=skills_dir,
                            output_schema=req.outputSchema,
                        )
                    )
                    async for event in result:
                        event_logger.log(event)
                        audit_logger.process_event(event)
                        if event.type == "result":
                            text = event.text
                            cost = event.cost_usd
                            input_tokens = event.input_tokens
                            output_tokens = event.output_tokens
                            break

            await asyncio.wait_for(run(), timeout=timeout / 1000)

        except TimeoutError:
            audit_logger.complete(
                success=False,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0,
            )
            return RunResponse(success=False, summary=f"Agent timed out after {timeout}ms")
        except Exception as e:
            audit_logger.complete(
                success=False,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0,
            )
            logger.exception("[agent] query error")
            return RunResponse(success=False, summary=f"Agent error: {e}")

        if not text:
            audit_logger.complete(
                success=False,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )
            return RunResponse(success=False, summary="Agent returned empty response")

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
            cost_usd=cost,
        )

        if parsed is not None:
            logger.info("[agent] query complete: success=%s, cost=$%.4f", success, cost)
            return RunResponse(
                success=success,
                summary=parsed.get("summary", text),
                **{k: v for k, v in parsed.items() if k not in ("success", "summary")},
            )

        logger.info("[agent] query complete (text response), cost=$%.4f", cost)
        return RunResponse(success=True, summary=text)

    router.add_api_route("/run", run_endpoint, methods=["POST"], response_model=RunResponse)
