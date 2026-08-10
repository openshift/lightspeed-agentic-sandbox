"""Query endpoint — POST /run.

The operator sends {query, systemPrompt, outputSchema, context, timeout_ms}
and the agent runs the LLM and returns {success, summary, ...structured fields}.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Request
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from lightspeed_agentic.audit import AuditLogger, derive_phase
from lightspeed_agentic.logging import EventLogger
from lightspeed_agentic.mcp import ResolvedMCPServer
from lightspeed_agentic.metrics import operation_duration, token_usage
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
        plan = opt["remediationPlan"]
        lines.append(f"Plan: {plan['description']}")
        lines.append(f"Reversible: {plan.get('reversible', 'unknown')}")
        if actions := plan.get("actions"):
            lines.append("Actions to execute:")
            for action in actions:
                if cmd := action.get("command"):
                    lines.append(f"  - [{action['type']}] {cmd} — {action['description']}")
                else:
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
    capture_content: bool = False,
    mcp_servers: list[ResolvedMCPServer] | None = None,
    reasoning_config: dict[str, Any] | None = None,
) -> None:
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

    async def run_endpoint(req: RunRequest, request: Request) -> RunResponse:
        timeout = req.timeout_ms if req.timeout_ms is not None else default_timeout_ms
        system_prompt = req.systemPrompt or "You are an AI agent."

        prompt = req.query
        if req.context:
            prefix = _format_context_prefix(req.context)
            prompt = f"{prefix}\n\n{req.query}"

        traceparent = request.headers.get("traceparent")
        agenticrun_uid = request.headers.get("x-agenticrun-uid", "")
        trace_id, trace_ctx = parse_traceparent(traceparent)
        tracer = get_tracer()

        phase = derive_phase(req.context)
        audit_logger = AuditLogger(
            phase=phase,
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
            "agenticrun.phase": phase,
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

        response: RunResponse
        try:

            async def run() -> None:
                nonlocal text, cost, input_tokens, output_tokens, reasoning_tokens, response_model
                token = otel_context.attach(span_ctx)
                try:
                    target_ns = req.context.get("targetNamespaces", []) if req.context else []
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
                            mcp_servers=mcp_servers or [],
                            reasoning_config=reasoning_config,
                            target_namespaces=target_ns,
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

            await asyncio.wait_for(run(), timeout=timeout / 1000)

        except TimeoutError:
            audit_logger.complete(
                success=False,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0,
                span=chat_span,
            )
            response = RunResponse(success=False, summary=f"Agent timed out after {timeout}ms")
            return response
        except Exception as e:
            audit_logger.complete(
                success=False,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0,
                span=chat_span,
            )
            logger.exception("[agent] query error")
            response = RunResponse(success=False, summary=f"Agent error: {e}")
            return response
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
                response = RunResponse(success=False, summary="Agent returned empty response")
                return response

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
                response = RunResponse(
                    success=success,
                    summary=parsed.get("summary", text),
                    **{k: v for k, v in parsed.items() if k not in ("success", "summary")},
                )
                return response

            logger.info("[agent] query complete (text response), cost=$%.4f", cost)
            response = RunResponse(success=True, summary=text)
            return response
        finally:
            chat_span.end()
            _record_metrics(
                in_tokens=input_tokens,
                out_tokens=output_tokens,
                elapsed=time.monotonic() - start_time,
            )

    router.add_api_route("/run", run_endpoint, methods=["POST"], response_model=RunResponse)
