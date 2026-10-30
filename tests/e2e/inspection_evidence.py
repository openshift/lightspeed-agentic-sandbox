"""Payload-free live safety reports, independent of raw collector/log retention."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from tests.e2e.batch_runner import (
    E2E_DEFAULT_AGENT_MAX_TURNS,
    E2E_DEFAULT_AGENT_TIMEOUT_SECONDS,
    RunBatchResult,
)
from tests.e2e.run_result import E2ERunResult
from tests.e2e.suite_setup import BatchE2EConfig

_PHASE_OUTCOMES: pytest.StashKey[dict[str, str]] = pytest.StashKey()
_EVIDENCE_CONTEXT: pytest.StashKey[tuple[BatchE2EConfig, dict[str, Any]]] = pytest.StashKey()
_REQUIRED_CHECKS = {
    "benign": {"run_success", "schema_valid", "mcp_sentinel", "inspection_span"},
    "malicious": {"safe_termination", "no_result_cr", "inspection_span"},
}


def _endpoint_origin(value: str) -> str | None:
    """Never copy URL userinfo, paths, query parameters or fragments."""
    try:
        url = urlsplit(value)
        if url.scheme not in {"https", "http"} or not url.hostname:
            return None
        host = f"[{url.hostname}]" if ":" in url.hostname else url.hostname
        port = f":{url.port}" if url.port else ""
        return f"{url.scheme}://{host}{port}"
    except ValueError:
        return None


def _auth_mode(provider: str, logs: str) -> str | None:
    """Report Azure's startup decision, not the runner's credential environment."""
    if provider == "azure":
        for line in logs.split("\n"):
            # Credential selection precedes this header and any agent/tool output.
            if line.startswith("INFO lightspeed_agentic.batch: provider="):
                break
            if (
                line == "INFO lightspeed_agentic.config: Azure credential resolution: "
                "Entra ID (service principal)"
            ):
                return "entra_service_principal"
            if line in {
                "INFO lightspeed_agentic.config: Azure credential resolution: API key (from file)",
                "INFO lightspeed_agentic.config: Azure credential resolution: API key (from env)",
            }:
                return "api_key"
        return None
    return {
        "openai": "api_key",
        "anthropic": "api_key",
        "vertex": "google_credentials",
        "bedrock": "aws_sigv4",
    }.get(provider)


def _reasoning_settings(raw: str) -> dict[str, Any] | None:
    """Retain known reasoning settings, never arbitrary nested strings or keys."""
    try:
        value = json.loads(raw or "{}")
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    for key, item in value.items():
        if (
            key == "effort"
            and isinstance(item, str)
            and item in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
        ):
            continue
        if key in {"thinking_budget", "budget_tokens"} and type(item) is int:
            continue
        if key == "include_thoughts" and type(item) is bool:
            continue
        if (
            key == "thinking"
            and isinstance(item, dict)
            and isinstance(item.get("type"), str)
            and item.get("type") in {"enabled", "disabled", "adaptive"}
            and set(item) <= {"type", "budget_tokens"}
            and ("budget_tokens" not in item or type(item["budget_tokens"]) is int)
        ):
            continue
        return None
    return value


def _positive_setting(value: str) -> int | None:
    """Invalid submitted settings remain unverified without copying their raw values."""
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def write_inspection_evidence(
    directory: Path,
    config: BatchE2EConfig,
    context: dict[str, Any],
    scenario: str,
    outcome: str,
) -> None:
    """Write only allowlisted facts; a passing test alone is not complete evidence."""
    run = context.get("run_result")
    batch = run.batch if isinstance(run, E2ERunResult) and run.batch else RunBatchResult()
    expected = context.get("inspection_expected_outcome")
    required = _REQUIRED_CHECKS.get(expected, set())
    checks = {
        key: value is True
        for key, value in context.get("inspection_checks", {}).items()
        if key in set.union(*_REQUIRED_CHECKS.values())
    }
    missing = sorted(required - {key for key, value in checks.items() if value})
    env = {**config.extra_env, **config.job_env, **context.get("job_env_overrides", {})}
    endpoint = env.get("LIGHTSPEED_PROVIDER_URL") or env.get("OPENAI_BASE_URL", "")
    enabled = env.get("LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED") == "true"
    auth_mode = _auth_mode(config.lightspeed_provider, batch.pod_logs)
    image_id = batch.image_id
    reasoning = _reasoning_settings(env.get("LIGHTSPEED_REASONING_CONFIG", ""))
    timeout = _positive_setting(
        env.get("LIGHTSPEED_AGENT_TIMEOUT_SECONDS", str(E2E_DEFAULT_AGENT_TIMEOUT_SECONDS))
    )
    max_turns = _positive_setting(
        env.get("LIGHTSPEED_AGENT_MAX_TURNS", E2E_DEFAULT_AGENT_MAX_TURNS)
    )
    gaps = []
    if not image_id or not re.search(r"sha256:[0-9a-f]{64}$", image_id):
        gaps.append("missing_observed_image_digest")
    if auth_mode is None:
        gaps.append("auth_mode_unverified")
    if reasoning is None:
        gaps.append("reasoning_config_not_retained")
    if timeout is None or max_turns is None:
        gaps.append("invalid_execution_limits")
    report = {
        "report_version": 1,
        "verification_scope": "focused_batch_safety_checks",
        "scenario": scenario,
        "test_outcome": outcome,
        "verified": bool(required) and not missing and outcome == "passed" and enabled and not gaps,
        "expected_outcome": expected,
        "inspection_enabled": enabled,
        "checks": checks,
        "missing_checks": missing,
        "evidence_gaps": gaps,
        "provider_matrix_id": config.provider_name,
        "hosting_provider": config.lightspeed_provider,
        "model": config.model,
        "auth_mode": auth_mode,
        "requested_image": config.sandbox_image,
        "image_id": image_id,
        "endpoint_origin": _endpoint_origin(endpoint),
        "endpoint_source": "runner_configuration_only_not_secret_resolution",
        "api_version": env.get("LIGHTSPEED_PROVIDER_API_VERSION"),
        "namespace": config.namespace,
        "session_id": config.session_id,
        "run_uid": batch.run_uid or None,
        "job_name": batch.job_name or None,
        "result_name": batch.result_name,
        "step": batch.step,
        "container_exit_code": batch.container_exit_code,
        "runner_settings": {
            "agent_timeout_seconds": timeout,
            "agent_max_turns": max_turns,
            "reasoning_config": reasoning,
        },
        "command": [
            "bash",
            "scripts/e2e-containers.sh",
            config.provider_name,
            "-k",
            "tool_result_inspection",
        ],
        "limitations": [
            "Not corpus accuracy or complete provider parity evidence.",
            "Endpoint origin is sanitized runner configuration, not observed routing; "
            "secret-supplied endpoint overrides and exact endpoint paths "
            "require separate configuration evidence.",
            "No model-context or normalized-event reconstruction assertion is made by this report.",
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    # Names come from pytest/Kubernetes, never from tool or classifier payloads.
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{scenario}-{batch.run_uid or config.session_id}")
    (directory / f"inspection-{filename}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[Any]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Require every phase to pass; failures take precedence over skips."""
    del call
    report = yield
    outcomes = item.stash.setdefault(_PHASE_OUTCOMES, {})
    outcomes[report.when] = report.outcome
    if report.when == "teardown" and _EVIDENCE_CONTEXT in item.stash:
        outcome = "unverified"
        if "failed" in outcomes.values():
            outcome = "failed"
        elif "skipped" in outcomes.values():
            outcome = "skipped"
        elif all(outcomes.get(phase) == "passed" for phase in ("setup", "call", "teardown")):
            outcome = "passed"
        config, context = item.stash[_EVIDENCE_CONTEXT]
        enabled = (
            context.get("job_env_overrides", {}).get("LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED")
            == "true"
        )
        directory = os.environ.get("ARTIFACT_DIR", "").strip()
        if enabled and directory:
            write_inspection_evidence(
                Path(directory),
                config,
                context,
                item.name,
                outcome,
            )
    return report


@pytest.fixture(autouse=True)
def inspection_evidence(
    request: pytest.FixtureRequest,
    bdd_context: dict[str, Any],
    batch_e2e_config: BatchE2EConfig,
) -> None:
    """Keep the scenario context until even fixture teardown has finished."""
    request.node.stash[_EVIDENCE_CONTEXT] = (batch_e2e_config, bdd_context)
