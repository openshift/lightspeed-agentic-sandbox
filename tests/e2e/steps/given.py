"""Given steps — service, schemas, prompts."""

from __future__ import annotations

import secrets
from typing import Any

import pytest
from kubernetes.client import ApiException, AppsV1Api, CoreV1Api  # type: ignore[import-untyped]
from pytest_bdd import given

from tests.e2e.credentials import require_credentials
from tests.e2e.otel_verify import assert_otel_deployment_present
from tests.e2e.skills_fixtures import SKILLS_SOURCE, list_skill_dirs
from tests.e2e.suite_setup import BatchE2EConfig, DEFAULT_OTEL_DEPLOYMENT
from schemas_contract import (
    CONTEXT_APPROVED_OPTION_ECHO_SCHEMA,
    CONTEXT_NAMESPACES_ECHO_SCHEMA,
    CONTEXT_PREVIOUS_ATTEMPTS_ECHO_SCHEMA,
    ECHO_TOKEN_SCHEMA,
    FLAT_OUTPUT_SCHEMA,
    MCP_TOOL_OUTPUT_SCHEMA,
    NESTED_OUTPUT_SCHEMA,
    STRICT_CONFLICT_SCHEMA,
)
from analysis_schemas import ANALYSIS_WITH_COMPONENTS_SCHEMA


@given("provider credentials are configured")
def provider_credentials_configured(provider_name: str) -> None:
    require_credentials(provider_name)


@given("the sandbox service is running")
def sandbox_running(batch_e2e_config: BatchE2EConfig) -> None:
    assert batch_e2e_config.namespace


@given("the OTEL collector is available for telemetry verification")
def otel_collector_available(
    batch_e2e_config: BatchE2EConfig,
    k8s_core_client: CoreV1Api,
    k8s_apps_client: AppsV1Api,
) -> None:
    """Require full e2e fixtures including the in-cluster OTEL collector."""
    if not batch_e2e_config.verify_full_fixtures:
        pytest.skip(
            "E2E_BATCH_VERIFY_FIXTURES=0 — OTEL collector verification disabled "
            "(run scripts/e2e-install-fixtures.sh and set E2E_BATCH_VERIFY_FIXTURES=1)"
        )
    if not batch_e2e_config.otel_endpoint:
        pytest.skip("OTEL endpoint not configured for batch e2e")
    try:
        k8s_apps_client.read_namespaced_deployment(
            DEFAULT_OTEL_DEPLOYMENT,
            batch_e2e_config.namespace,
        )
    except ApiException as exc:
        if exc.status == 404:
            pytest.skip(
                f"deployment/{DEFAULT_OTEL_DEPLOYMENT} missing in "
                f"{batch_e2e_config.namespace} (run scripts/e2e-install-fixtures.sh)"
            )
        raise
    assert_otel_deployment_present(k8s_core_client, batch_e2e_config.namespace)


@given("the sandbox service is running with skills")
def sandbox_running_with_skills(
    batch_e2e_config: BatchE2EConfig,
    bdd_context: dict[str, Any],
) -> None:
    assert batch_e2e_config.namespace
    assert SKILLS_SOURCE.is_dir(), f"skills fixtures missing at {SKILLS_SOURCE}"
    assert list_skill_dirs(), f"no skills under {SKILLS_SOURCE}"
    bdd_context["mount_skills"] = True


@given("a simple non-skill query has been prepared")
def prepare_simple_non_skill(bdd_context: dict[str, Any]) -> None:
    bdd_context["query"] = (
        "In one sentence, name any primary color. "
        "Answer with plain text only. Do not call any tools."
    )
    bdd_context["output_schema"] = None


@given("a context with target namespaces and an echo output schema have been prepared")
def prepare_context_namespaces_echo(bdd_context: dict[str, Any]) -> None:
    # Unguessable per-run values: the echo can only match if context was truly
    # injected — real namespaces like "default"/"kube-system" a model could guess.
    nonce = secrets.token_hex(3)
    target_namespaces = [f"ns-{nonce}-alpha", f"ns-{nonce}-bravo"]
    bdd_context["context"] = {"targetNamespaces": target_namespaces}
    bdd_context["expected_namespaces"] = ", ".join(target_namespaces)
    bdd_context["output_schema"] = CONTEXT_NAMESPACES_ECHO_SCHEMA
    bdd_context["query"] = (
        "The user message contains a [context] block with Target namespaces. "
        "Return a single JSON object only (no markdown). "
        "Set success=true, summary='context-echo-ok', and set namespaces to the "
        "comma-separated namespace values from the 'Target namespaces:' line "
        "(values only, not the label)."
    )


@given("a context with previous attempts and an echo output schema have been prepared")
def prepare_context_previous_attempts_echo(bdd_context: dict[str, Any]) -> None:
    # Unguessable per-run failure reason: a generic value like "timeout" a model
    # could produce without any injected context.
    nonce = secrets.token_hex(3)
    first_failure_reason = f"probe-fault-{nonce}"
    bdd_context["context"] = {
        "previousAttempts": [
            {"attempt": 1, "failureReason": first_failure_reason},
            {"attempt": 2},
        ]
    }
    bdd_context["expected_first_failure_reason"] = first_failure_reason
    bdd_context["output_schema"] = CONTEXT_PREVIOUS_ATTEMPTS_ECHO_SCHEMA
    bdd_context["query"] = (
        "The user message contains a [context] block with a Previous attempts section. "
        "Return a single JSON object only (no markdown). "
        "Set success=true, summary='context-echo-ok', and set firstFailureReason to the "
        "failure reason on Attempt 1 (value only, not the label or attempt number)."
    )


_CONTEXT_APPROVED_OPTION = {
    "title": "Restart deployment",
    "diagnosis": {"rootCause": "CrashLoopBackOff"},
    "remediationPlan": {
        "description": "Roll out restart",
        "actions": [
            {
                "command": "kubectl rollout restart deploy/web -n prod",
                "type": "mutation",
                "description": "Restart the deployment",
            },
        ],
        "risk": "low",
        "reversible": True,
    },
}


@given("a context with approved option and an echo output schema have been prepared")
def prepare_context_approved_option_echo(bdd_context: dict[str, Any]) -> None:
    bdd_context["context"] = {"approvedOption": _CONTEXT_APPROVED_OPTION}
    bdd_context["expected_approved_title"] = _CONTEXT_APPROVED_OPTION["title"]
    bdd_context["expected_root_cause"] = _CONTEXT_APPROVED_OPTION["diagnosis"]["rootCause"]
    first_action = _CONTEXT_APPROVED_OPTION["remediationPlan"]["actions"][0]
    bdd_context["expected_first_command"] = first_action["command"]
    bdd_context["output_schema"] = CONTEXT_APPROVED_OPTION_ECHO_SCHEMA
    bdd_context["query"] = (
        "The user message contains a [context] block with an approved remediation section. "
        "Return a single JSON object only (no markdown). "
        "Set success=true, summary='context-echo-ok', approvedTitle to the exact title string, "
        "rootCause to the exact rootCause string, and firstCommand to the exact command string "
        "from the first action in remediationPlan.actions (do not include the action description)."
    )


@given("the echo-token skill query has been prepared")
def prepare_echo_token(bdd_context: dict[str, Any]) -> None:
    bdd_context["system_prompt"] = (
        "You are an agent with shell access. When a skill provides a script, "
        "you must run that script via the shell tool and use its stdout JSON "
        "before producing your final answer. Never invent or placeholder token values."
    )
    bdd_context["query"] = (
        "Use the echo-token skill end-to-end:\n"
        "1. Load the echo-token skill.\n"
        "2. From the skill directory, run: bash scripts/echo-token.sh\n"
        "3. Parse the JSON printed to stdout.\n"
        "4. Reply with a single JSON object only (no markdown): success=true, "
        "summary containing the token verbatim, token equal to the script token field, "
        "and status equal to the script status field.\n"
        "Do not reply until step 2 exits 0 and prints JSON."
    )
    bdd_context["output_schema"] = ECHO_TOKEN_SCHEMA


@given("the find-token analysis query and schema have been prepared")
def prepare_find_token_analysis(bdd_context: dict[str, Any]) -> None:
    bdd_context["system_prompt"] = (
        "You are an assistant. Use your available skills to accomplish tasks. "
        "When a skill prints structured JSON analysis output, use that JSON as the "
        "basis for your response. remediationPlan must include at least one action. "
        "Set actionRequired to the string 'True' and place tokens under "
        "options[].components[].tokens."
    )
    bdd_context["query"] = (
        "Find the hidden token using the 'find-token' skill. "
        "Run the script and return its JSON output as your structured analysis response "
        "(preserve remediationPlan.actions and components with DIAG_/VERIFY_ tokens)."
    )
    bdd_context["output_schema"] = ANALYSIS_WITH_COMPONENTS_SCHEMA
    bdd_context["wait_timeout_seconds"] = 900.0


@given("a flat output schema with required fields has been prepared")
def prepare_flat(bdd_context: dict[str, Any]) -> None:
    bdd_context["output_schema"] = FLAT_OUTPUT_SCHEMA
    bdd_context["query"] = (
        "Respond with a single JSON object only (no markdown). "
        'Fields: success=true, summary="e2e-flat-ok", ticketId="E2E-STRUCT-001".'
    )


@given("a nested output schema has been prepared")
def prepare_nested(bdd_context: dict[str, Any]) -> None:
    bdd_context["output_schema"] = NESTED_OUTPUT_SCHEMA
    bdd_context["query"] = (
        "Respond with a single JSON object only (no markdown). "
        'success=true, summary="e2e-nested-ok", '
        'items=[{"name":"widget","count":1},{"name":"gadget","count":2}].'
    )


@given("no output schema will be sent")
def prepare_no_schema(bdd_context: dict[str, Any]) -> None:
    bdd_context["output_schema"] = None
    bdd_context["query"] = (
        "In one short sentence, name any primary color. "
        "Answer with plain text only. Do not return JSON. Do not call any tools."
    )


@given("an adversarial output schema and prompt have been prepared")
def prepare_adversarial(bdd_context: dict[str, Any]) -> None:
    bdd_context["output_schema"] = STRICT_CONFLICT_SCHEMA
    bdd_context["query"] = (
        "Reply with exactly the single word hello in plain text. "
        "Do not use JSON. Do not use markdown."
    )


# --- MCP scenarios ---


@given("the sandbox service is running with reasoning configured")
def sandbox_running_with_reasoning(batch_e2e_config: BatchE2EConfig) -> None:
    import json as _json

    assert batch_e2e_config.namespace
    raw = batch_e2e_config.job_env.get("LIGHTSPEED_REASONING_CONFIG", "").strip()
    if not raw:
        pytest.skip(
            "LIGHTSPEED_REASONING_CONFIG not set — run via scripts/e2e-containers.sh "
            "or export LIGHTSPEED_REASONING_CONFIG for batch Jobs"
        )
    parsed = _json.loads(raw)
    assert isinstance(parsed, dict), (
        f"LIGHTSPEED_REASONING_CONFIG must be a JSON object, got: {raw!r}"
    )


@given("the sandbox service is running with MCP servers configured")
def sandbox_running_with_mcp(batch_e2e_config: BatchE2EConfig) -> None:
    import json as _json

    assert batch_e2e_config.namespace
    raw = batch_e2e_config.job_env.get("LIGHTSPEED_MCP_SERVERS", "").strip()
    if not raw:
        pytest.skip(
            "LIGHTSPEED_MCP_SERVERS not set — run scripts/e2e-install-fixtures.sh and "
            "scripts/e2e-containers.sh, or export LIGHTSPEED_MCP_SERVERS for batch Jobs"
        )
    parsed = _json.loads(raw)
    assert isinstance(parsed, list), f"LIGHTSPEED_MCP_SERVERS must be a JSON array, got: {raw!r}"
    assert len(parsed) > 0, "LIGHTSPEED_MCP_SERVERS is an empty array"
    assert all("name" in s and "url" in s for s in parsed), (
        f"Each MCP server entry must have 'name' and 'url': {parsed!r}"
    )


@given("an MCP tool invocation query has been prepared")
def prepare_mcp_tool_invocation(bdd_context: dict[str, Any]) -> None:
    bdd_context["output_schema"] = MCP_TOOL_OUTPUT_SCHEMA
    bdd_context["query"] = (
        "Call the 'list_namespaces' tool from the MCP server named 'mock-ocp-mcp'. "
        "Return a single JSON object only (no markdown). "
        "Fields: success=true, summary=<the exact text output from the tool>."
    )


@given("an MCP query targeting a nonexistent tool has been prepared")
def prepare_mcp_nonexistent_tool(bdd_context: dict[str, Any]) -> None:
    bdd_context["output_schema"] = MCP_TOOL_OUTPUT_SCHEMA
    bdd_context["query"] = (
        "You MUST invoke tools only through the MCP server named 'mock-ocp-mcp'. "
        "Do NOT use shell, bash, exec_command, or any local commands. "
        "Call the tool named 'nonexistent_tool_xyz_999' on mock-ocp-mcp. "
        "Do not assume the outcome — make the MCP tool call. "
        "Return a single JSON object only (no markdown). Report the real outcome: "
        "set success to true only if the MCP tool call actually succeeded, false otherwise, "
        "and put a short description of what happened (include mock-ocp-mcp and the tool name) "
        "in summary."
    )
