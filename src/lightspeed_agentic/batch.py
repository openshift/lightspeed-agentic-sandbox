"""Batch entrypoint — read /input, run agent, publish Result CR, exit.

Reads input ConfigMap files from /input/, runs the LLM provider, creates the
Result CR from result-template, updates status via the Kubernetes API, and
exits 0 on sandbox success (including agent failure).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lightspeed_agentic.config import (
    parse_reasoning_config,
    resolve_router_model,
    resolve_sdk,
    resolve_startup_model,
)
from lightspeed_agentic.factory import create_provider
from lightspeed_agentic.mcp import MCPConfigError, parse_mcp_servers
from lightspeed_agentic.publish_results.publish import (
    PublishError,
    publish_agent_result,
    step_from_result_template,
    write_termination_log,
)
from lightspeed_agentic.readiness import run_readiness_checks
from lightspeed_agentic.run_agent import run_agent_query
from lightspeed_agentic.tracing import init_tracer, otel_runtime_enabled, shutdown_tracer

logger = logging.getLogger(__name__)

INPUT_DIR = "/input"
DEFAULT_SYSTEM_PROMPT = "You are an AI agent."
DEFAULT_TIMEOUT_MS = 300_000
DEFAULT_MAX_TURNS = 200
DEFAULT_SKILLS_DIR = "/app/skills"
TRACEPARENT_ENV = "TRACEPARENT"


class InputReadError(Exception):
    """Required batch input could not be read (spec B6)."""


@dataclass(frozen=True)
class BatchInput:
    """Parsed contents of the operator input ConfigMap mount."""

    query: str
    """Step query text from ``/input/query``."""

    output_schema: dict[str, Any]
    """Structured output JSON schema from ``/input/output-schema``."""

    context: dict[str, Any]
    """Workflow context from ``/input/context``."""

    result_template: dict[str, Any]
    """Pre-filled Result CR template from ``/input/result-template``."""

    system_prompt: str | None = None
    """Optional system instructions from ``/input/system-prompt``."""


def read_batch_inputs(input_dir: str = INPUT_DIR) -> BatchInput:
    """Load required and optional files from the operator input ConfigMap mount."""
    base = Path(input_dir)
    query = _must_read_text(base / "query")
    schema_raw = _must_read_text(base / "output-schema")
    context_raw = _must_read_text(base / "context")
    template_raw = _must_read_text(base / "result-template")
    system_prompt = _read_optional_text(base / "system-prompt")

    try:
        output_schema = json.loads(schema_raw)
        context = json.loads(context_raw)
        result_template = json.loads(template_raw)
    except json.JSONDecodeError as exc:
        raise InputReadError(f"invalid JSON in {input_dir}: {exc}") from exc

    if not isinstance(output_schema, dict):
        raise InputReadError("output-schema must be a JSON object")
    if not isinstance(context, dict):
        raise InputReadError("context must be a JSON object")
    if not isinstance(result_template, dict):
        raise InputReadError("result-template must be a JSON object")

    return BatchInput(
        query=query,
        output_schema=output_schema,
        context=context,
        result_template=result_template,
        system_prompt=system_prompt,
    )


def _must_read_text(path: Path) -> str:
    """Read a required input file; raise InputReadError on failure."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputReadError(f"read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise InputReadError(f"read {path}: invalid UTF-8") from exc


def _read_optional_text(path: Path) -> str | None:
    """Read an optional input file; return None when absent or empty."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise InputReadError(f"read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise InputReadError(f"read {path}: invalid UTF-8") from exc
    return text or None


def main() -> None:
    """Read /input, run the agent, publish the Result CR, and exit.

    On input or publish infrastructure failure, writes to the termination log
    and exits non-zero. Agent failure still publishes a Result CR and exits 0.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info("batch sandbox starting")

    try:
        inputs = read_batch_inputs()
    except InputReadError as exc:
        write_termination_log(str(exc))
        sys.exit(1)
        return

    try:
        step = step_from_result_template(inputs.result_template)
    except ValueError as exc:
        write_termination_log(str(exc))
        sys.exit(1)
        return

    target_ns = _pick_namespace(inputs.context)
    logger.info(
        "step=%s query_len=%d target_ns=%s kind=%s",
        step,
        len(inputs.query),
        target_ns,
        inputs.result_template.get("kind"),
    )

    otel_active = False
    try:
        sdk = resolve_sdk()
        reasoning_config = parse_reasoning_config()
        mcp_servers = parse_mcp_servers()
        readiness_ok, readiness_checks = run_readiness_checks(sdk)
        if not readiness_ok:
            write_termination_log(_format_readiness_failure(readiness_checks))
            sys.exit(1)
            return

        if otel_runtime_enabled():
            init_tracer(agenticrun_phase=step)
            otel_active = True
        provider = create_provider(sdk.name)
        startup_model = resolve_startup_model(sdk.name)
        audit_enabled = os.environ.get("LIGHTSPEED_AUDIT_ENABLED", "").strip().lower() == "true"
        capture_content = _resolve_capture_content(audit_enabled)
        agenticrun_uid = os.environ.get("LIGHTSPEED_AGENTICRUN_UID", "").strip()
        skills_dir = os.environ.get("LIGHTSPEED_SKILLS_DIR", DEFAULT_SKILLS_DIR)
        timeout_ms = _resolve_timeout_ms()
        model = resolve_router_model(provider.name, startup_model)

        logger.info(
            "provider=%s model=%s audit=%s capture_content=%s",
            provider.name,
            model,
            audit_enabled,
            capture_content,
        )

        system_prompt = inputs.system_prompt or DEFAULT_SYSTEM_PROMPT
        traceparent = _resolve_traceparent()
        started_at = datetime.now(UTC)
        agent_result = asyncio.run(
            run_agent_query(
                provider,
                prompt=inputs.query,
                system_prompt=system_prompt,
                output_schema=inputs.output_schema,
                context=inputs.context,
                skills_dir=skills_dir,
                model=model,
                max_turns=DEFAULT_MAX_TURNS,
                timeout_ms=timeout_ms,
                mcp_servers=mcp_servers,
                reasoning_config=reasoning_config,
                audit_enabled=audit_enabled,
                capture_content=capture_content,
                agenticrun_uid=agenticrun_uid,
                traceparent=traceparent,
                step=step,
            )
        )

        completed_at = datetime.now(UTC)
        publish_agent_result(
            inputs.result_template,
            agent_result.output,
            started_at=started_at,
            completed_at=completed_at,
            input_tokens=agent_result.input_tokens,
            output_tokens=agent_result.output_tokens,
        )
        logger.info("status updated — exiting 0")
    except PublishError as exc:
        write_termination_log(str(exc))
        sys.exit(1)
        return
    except MCPConfigError as exc:
        write_termination_log(str(exc))
        sys.exit(1)
        return
    except Exception as exc:
        logger.exception("batch sandbox failed")
        write_termination_log(str(exc))
        sys.exit(1)
        return
    finally:
        if otel_active:
            shutdown_tracer()


def _format_readiness_failure(checks: dict[str, str]) -> str:
    """Human-readable termination-log message for failed readiness checks."""
    detail = "; ".join(f"{name}={status}" for name, status in checks.items())
    return f"readiness failed: {detail}"


def _pick_namespace(context: dict[str, Any]) -> str:
    """Return the first target namespace from context, or ``default``."""
    namespaces = context.get("targetNamespaces")
    if isinstance(namespaces, list) and namespaces and namespaces[0]:
        return str(namespaces[0])
    return "default"


def _resolve_traceparent() -> str | None:
    """Read W3C traceparent from TRACEPARENT env (operator phase span linkage)."""
    raw = os.environ.get(TRACEPARENT_ENV, "").strip()
    return raw or None


def _resolve_capture_content(audit_enabled: bool) -> bool:
    """Resolve whether ``gen_ai.choice`` events include completion/reasoning text.

    Defaults to ``audit_enabled`` when ``LIGHTSPEED_CAPTURE_CONTENT`` is unset.
    Explicit ``true`` / ``false`` overrides the default.
    """
    raw = os.environ.get("LIGHTSPEED_CAPTURE_CONTENT", "").strip().lower()
    if raw == "false":
        return False
    if raw == "true":
        return True
    return audit_enabled


def _resolve_timeout_ms() -> int:
    """Resolve agent timeout from LIGHTSPEED_TIMEOUT_MS or the default."""
    raw = os.environ.get("LIGHTSPEED_TIMEOUT_MS", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_MS
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid LIGHTSPEED_TIMEOUT_MS=%r, using default", raw)
        return DEFAULT_TIMEOUT_MS


if __name__ == "__main__":
    main()
