"""Then steps — batch run and JSON assertions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import jsonschema
from kubernetes.client import CoreV1Api  # type: ignore[import-untyped]
from pytest_bdd import then

from tests.e2e.analysis_schemas import ANALYSIS_WITH_COMPONENTS_SCHEMA
from tests.e2e.analysis_tokens import assert_skill_tokens_in_response
from tests.e2e.otel_verify import wait_for_otel_audit_logs, wait_for_otel_traces
from tests.e2e.run_result import E2ERunResult
from tests.e2e.skills_fixtures import E2E_TOKEN_REL_PATH
from tests.e2e.suite_setup import BatchE2EConfig

# SHA-256 of empty string — models sometimes fabricate this instead of running echo-token.sh
_EMPTY_STRING_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _analysis_status_subset(body: dict[str, Any]) -> dict[str, Any]:
    return {key: body[key] for key in ("actionRequired", "options", "diagnosis") if key in body}


def _require_run_result(bdd_context: dict[str, Any]) -> E2ERunResult:
    result = bdd_context.get("run_result")
    if not isinstance(result, E2ERunResult):
        msg = "run_result missing from BDD context — When step did not store a batch run result"
        raise AssertionError(msg)
    return result


def _assert_batch_job_completes(res: E2ERunResult) -> None:
    assert res.error is None, f"batch harness error: {res.error}"
    assert res.batch is not None, "batch run result missing Job metadata"
    assert res.batch.job_succeeded, (
        f"batch Job did not succeed: {res.batch.termination_message or res.raw_text[:500]}"
    )


@then("the batch job completes")
def assert_batch_job_completes(bdd_context: dict[str, Any]) -> None:
    """Assert the batch Job finished without a harness error."""
    _assert_batch_job_completes(_require_run_result(bdd_context))


@then("the run completes successfully")
def assert_run_completes_successfully(bdd_context: dict[str, Any]) -> None:
    """Assert the batch Job completed and the agent reported success."""
    res = _require_run_result(bdd_context)
    _assert_batch_job_completes(res)
    assert res.batch is not None
    assert res.batch.agent_succeeded, (
        "agent did not succeed: "
        f"{res.body.get('summary') or res.body.get('failureReason') or res.body!r}"
    )


def _require_run_uid(bdd_context: dict[str, Any]) -> str:
    run_uid = bdd_context.get("run_uid", "")
    if not run_uid:
        msg = "run_uid missing from BDD context — batch When step did not store a run result"
        raise AssertionError(msg)
    return str(run_uid)


@then("the OTEL collector received traces for the batch run")
def assert_otel_traces_received(
    bdd_context: dict[str, Any],
    batch_e2e_config: BatchE2EConfig,
    k8s_core_client: CoreV1Api,
) -> None:
    """Assert the e2e OTEL collector debug output includes spans for this batch run."""
    run_uid = _require_run_uid(bdd_context)
    wait_for_otel_traces(k8s_core_client, batch_e2e_config.namespace, run_uid)


@then("the OTEL collector received audit logs with agenticrun attributes")
def assert_otel_audit_logs_received(
    bdd_context: dict[str, Any],
    batch_e2e_config: BatchE2EConfig,
    k8s_core_client: CoreV1Api,
) -> None:
    """Assert bridged audit OTLP logs include agenticrun uid/phase for this batch run."""
    run_uid = _require_run_uid(bdd_context)
    phase = str(bdd_context.get("run_step", "analysis"))
    wait_for_otel_audit_logs(
        k8s_core_client,
        batch_e2e_config.namespace,
        run_uid,
        phase=phase,
    )


@then("the response includes success summary and ticketId fields")
def assert_flat_fields(bdd_context: dict[str, Any]) -> None:
    """Assert structured output includes success, summary, and ticketId."""
    body = bdd_context["response_body"]
    assert "success" in body
    assert "summary" in body
    assert isinstance(body["summary"], str)
    assert body.get("ticketId"), f"missing ticketId in {body!r}"


@then("the response JSON validates against the output schema")
def assert_jsonschema(bdd_context: dict[str, Any]) -> None:
    """Validate response body against the prepared output schema."""
    schema = bdd_context["output_schema"]
    body = bdd_context["response_body"]
    response_token = body.get("token", "")
    if isinstance(response_token, str) and response_token == _EMPTY_STRING_SHA256:
        raise AssertionError(
            "response token looks fabricated (empty-string SHA-256); "
            "run bash scripts/echo-token.sh and use its stdout JSON"
        )
    jsonschema.validate(instance=body, schema=schema)


@then("the response has a non-empty summary")
def assert_nonempty_summary(bdd_context: dict[str, Any]) -> None:
    """Assert summary is a non-empty string."""
    body = bdd_context["response_body"]
    summary = body.get("summary", "")
    assert isinstance(summary, str), f"summary not a string: {body!r}"
    assert summary.strip(), f"summary missing/empty: {body!r}"


@then("success is true")
def assert_success_true(bdd_context: dict[str, Any]) -> None:
    """Assert RunResponse ``success`` is true."""
    body = bdd_context["response_body"]
    assert body.get("success") is True, body


@then("success is false")
def assert_success_false(bdd_context: dict[str, Any]) -> None:
    """Assert RunResponse ``success`` is false."""
    body = bdd_context["response_body"]
    assert body.get("success") is False, body


@then("the response summary indicates a timeout")
def assert_summary_indicates_timeout(bdd_context: dict[str, Any]) -> None:
    """Assert the failure was a timeout, not any other error.

    The endpoint returns ``success=false`` for timeouts, generic errors, and
    empty responses alike; this distinguishes the timeout path via its summary
    (``Agent timed out after {N}ms``).
    """
    body = bdd_context["response_body"]
    summary = body.get("summary", "").lower()
    assert "timed out" in summary or "timeout" in summary, (
        f"summary does not indicate a timeout: {body!r}"
    )


@then("the response summary contains the reasoning answer")
def assert_summary_contains_reasoning_answer(bdd_context: dict[str, Any]) -> None:
    """Assert the model produced the correct answer to 17 * 23 (391)."""
    body = bdd_context["response_body"]
    summary = body.get("summary", "")
    assert re.search(r"\b391\b", summary), f"summary missing correct answer 391: {body!r}"


@then("the response namespaces field matches the prepared context")
def assert_namespaces_match_context(bdd_context: dict[str, Any]) -> None:
    """Assert echoed namespaces match targetNamespaces from prepared context."""
    body = bdd_context["response_body"]
    expected = bdd_context["expected_namespaces"]
    actual = body.get("namespaces", "")

    def _ns_parts(value: str) -> list[str]:
        return [part.strip() for part in value.split(",") if part.strip()]

    assert _ns_parts(actual) == _ns_parts(expected), (
        f"expected namespaces {expected!r}, got {actual!r} in {body!r}"
    )


@then("the response first failure reason matches the prepared context")
def assert_first_failure_reason_matches_context(bdd_context: dict[str, Any]) -> None:
    """Assert echoed firstFailureReason matches previousAttempts from prepared context."""
    body = bdd_context["response_body"]
    expected = bdd_context["expected_first_failure_reason"]
    actual = body.get("firstFailureReason", "")
    assert actual == expected, (
        f"expected firstFailureReason {expected!r}, got {actual!r} in {body!r}"
    )


@then("the response approved option fields match the prepared context")
def assert_approved_option_matches_context(bdd_context: dict[str, Any]) -> None:
    """Assert echoed approvedTitle, rootCause, and firstCommand match context."""
    body = bdd_context["response_body"]
    expected_title = bdd_context["expected_approved_title"]
    expected_root_cause = bdd_context["expected_root_cause"]
    expected_command = bdd_context["expected_first_command"]
    actual_title = body.get("approvedTitle", "")
    actual_root_cause = body.get("rootCause", "")
    actual_command = body.get("firstCommand", "")
    assert actual_title == expected_title, (
        f"expected approvedTitle {expected_title!r}, got {actual_title!r} in {body!r}"
    )
    assert actual_root_cause == expected_root_cause, (
        f"expected rootCause {expected_root_cause!r}, got {actual_root_cause!r} in {body!r}"
    )
    assert actual_command == expected_command, (
        f"expected firstCommand {expected_command!r}, got {actual_command!r} in {body!r}"
    )


@then("the skill script wrote a token file to disk")
def assert_token_file(
    bdd_context: dict[str, Any],
    e2e_output_dir: Path | None,
) -> None:
    """Assert echo-token.sh ran (token recovered from batch pod logs or host output dir)."""
    token = str(bdd_context.get("token_file", "")).strip()
    if not token:
        if bdd_context.get("batch_job_name"):
            msg = (
                "echo-token script output not found in batch pod logs; "
                "the agent must run bash scripts/echo-token.sh from the skill directory"
            )
            raise AssertionError(msg)
        assert e2e_output_dir is not None, "E2E_OUTPUT_DIR not set"
        host_path = e2e_output_dir / E2E_TOKEN_REL_PATH
        assert host_path.exists(), (
            f"token file not found at {host_path}; "
            "the agent must run bash scripts/echo-token.sh from the skill directory"
        )
        token = host_path.read_text(encoding="utf-8").strip()
    assert token, "token file is empty"
    bdd_context["token"] = token


@then("the response contains the generated token")
def assert_token_in_response(bdd_context: dict[str, Any]) -> None:
    """Assert the response body or summary includes the token from disk."""
    body = bdd_context["response_body"]
    token = bdd_context["token"]
    response_token = body.get("token", "")
    summary = body.get("summary", "")
    assert token in response_token or token in summary, (
        f"token {token!r} not found in response token={response_token!r} or summary={summary!r}"
    )


@then("the analysis status validates against the operator components schema")
def assert_analysis_status_jsonschema(bdd_context: dict[str, Any]) -> None:
    body = bdd_context["response_body"]
    analysis = _analysis_status_subset(body)
    jsonschema.validate(instance=analysis, schema=ANALYSIS_WITH_COMPONENTS_SCHEMA)


@then("actionRequired is True")
def assert_action_required_true(bdd_context: dict[str, Any]) -> None:
    body = bdd_context["response_body"]
    assert body.get("actionRequired") == "True", body


@then("the response contains DIAG and VERIFY tokens in component tokens")
def assert_diag_verify_component_tokens(bdd_context: dict[str, Any], provider_name: str) -> None:
    assert_skill_tokens_in_response(bdd_context["response_body"], provider_name)


@then("the first analysis option has remediation and component audit fields")
def assert_first_option_analysis_fields(bdd_context: dict[str, Any]) -> None:
    body = bdd_context["response_body"]
    analysis = _analysis_status_subset(body)
    option = analysis["options"][0]
    assert option["remediationPlan"]["reversible"] in (
        "Reversible",
        "Irreversible",
        "Partial",
    )
    assert len(option["remediationPlan"]["actions"]) > 0
    assert len(option["components"]) > 0

    comp = option["components"][0]
    assert comp["tokens"]["primary"]["valid"] is True
    assert comp["tokens"]["secondary"]["valid"] is True
    assert comp["audit"]["outcome"] in ("pass", "fail", "partial")
    assert len(comp["audit"]["findings"]) > 0
    for finding in comp["audit"]["findings"]:
        assert finding["severity"] in ("info", "warning", "critical")


@then("the run completes successfully and the envelope has success and summary")
def assert_run_envelope(bdd_context: dict[str, Any]) -> None:
    """Assert successful batch run and response envelope fields without schema validation."""
    res = _require_run_result(bdd_context)
    _assert_batch_job_completes(res)
    body = bdd_context["response_body"]
    assert "success" in body, body
    assert "summary" in body, body
    assert isinstance(body["summary"], str), body


# --- MCP assertions ---

# Unguessable marker returned only by the mock's list_namespaces tool — proves the
# agent actually invoked the tool. Keep in sync with mock_mcp_server.MOCK_NAMESPACES.
_MOCK_SENTINEL_NAMESPACE = "e2e-sentinel-ns-7f3a9"


@then("the response summary contains the sentinel namespace from the tool")
def assert_summary_contains_namespace_output(bdd_context: dict[str, Any]) -> None:
    """Assert the summary contains the unguessable sentinel namespace.

    A model cannot fabricate the sentinel, so its presence proves the agent
    actually invoked the mock MCP ``list_namespaces`` tool and used its output.
    """
    body = bdd_context["response_body"]
    summary = body.get("summary", "").lower()
    assert _MOCK_SENTINEL_NAMESPACE in summary, (
        f"summary does not contain sentinel namespace {_MOCK_SENTINEL_NAMESPACE!r} "
        f"(tool was likely not invoked): {summary!r}"
    )


_SHELL_FAILURE_MARKERS = (
    "exit code 127",
    "command not found",
    "local environment",
)


@then("the response summary indicates an MCP tool failure")
def assert_summary_indicates_mcp_tool_failure(bdd_context: dict[str, Any]) -> None:
    """Assert failure came from mock-ocp-mcp, not a local shell workaround."""
    body = bdd_context["response_body"]
    summary = str(body.get("summary", ""))
    lowered = summary.lower()
    assert "mock-ocp-mcp" in lowered, (
        f"summary does not reference mock-ocp-mcp (MCP path likely not used): {body!r}"
    )
    assert "nonexistent_tool_xyz_999" in lowered, (
        f"summary does not mention the requested MCP tool name: {body!r}"
    )
    for marker in _SHELL_FAILURE_MARKERS:
        assert marker not in lowered, (
            f"summary looks like a local shell failure ({marker!r}), not MCP: {body!r}"
        )
