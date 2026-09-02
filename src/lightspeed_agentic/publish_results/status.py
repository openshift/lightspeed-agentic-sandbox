"""Build Result CR status dicts from schema-driven agent output.

Agent field shapes come from /input/output-schema — this module does not
validate or reshape them. It copies allowed keys per Result kind, wraps
lifecycle conditions, and sets failureReason when the agent or sandbox
reports failure.

For AnalysisResult, ``_sanitize_analysis_options`` truncates string fields
that exceed CRD maxLength limits and removes per-option diagnosis dicts
whose required fields are empty, preventing CRD validation failures on
publish (OLS-3864).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

CONDITION_STARTED = "Started"
CONDITION_COMPLETED = "Completed"
REASON_STEP_STARTED = "StepStarted"
REASON_SUCCEEDED = "Succeeded"
REASON_FAILED = "Failed"

ACTION_REQUIRED_TRUE = "True"
ACTION_REQUIRED_FALSE = "False"

# CRD maxLength limits for AnalysisResult option fields.  Must stay in
# sync with the operator constants in sandbox_agent.go.
_MAX_LEN_OPTION_TITLE = 256
_MAX_LEN_OPTION_SUMMARY = 1024
_MAX_LEN_DIAGNOSIS_SUMMARY = 8192
_MAX_LEN_DIAGNOSIS_ROOT_CAUSE = 1024
_MAX_LEN_PLAN_DESCRIPTION = 8192
_MAX_LEN_ACTION_COMMAND = 4096
_MAX_LEN_ACTION_TYPE = 256
_MAX_LEN_ACTION_DESCRIPTION = 4096
_MAX_LEN_ROLLBACK_DESCRIPTION = 4096
_MAX_LEN_ROLLBACK_COMMAND = 4096
_MAX_LEN_VERIFICATION_DESCRIPTION = 4096
_MAX_LEN_VERIFICATION_STEP_NAME = 253
_MAX_LEN_VERIFICATION_STEP_COMMAND = 4096
_MAX_LEN_VERIFICATION_STEP_EXPECTED = 1024
_MAX_LEN_VERIFICATION_STEP_TYPE = 256
_MAX_LEN_RBAC_JUSTIFICATION = 1024

# Result kind → agent output keys copied into status (schema-owned shapes).
_STATUS_FIELDS_BY_KIND: dict[str, tuple[str, ...]] = {
    "AnalysisResult": ("options", "actionRequired", "diagnosis"),
    "ExecutionResult": ("actionsTaken",),
    "VerificationResult": ("checks", "summary"),
    "EscalationResult": ("summary", "content"),
}


def format_condition_time(when: datetime) -> str:
    """Format a timestamp for Kubernetes condition lastTransitionTime (RFC3339)."""
    if when.tzinfo is None:
        return when.isoformat() + "Z"
    return when.isoformat().replace("+00:00", "Z")


def action_required_value(value: bool) -> str:
    """Convert agent boolean actionRequired to CRD ActionRequiredValue."""
    return ACTION_REQUIRED_TRUE if value else ACTION_REQUIRED_FALSE


def build_conditions(
    *,
    started_at: datetime,
    completed_at: datetime,
    succeeded: bool,
) -> list[dict[str, Any]]:
    """Build Started + Completed conditions for a Result CR status."""
    completed_reason = REASON_SUCCEEDED if succeeded else REASON_FAILED
    return [
        {
            "type": CONDITION_STARTED,
            "status": "True",
            "reason": REASON_STEP_STARTED,
            "message": "Step started",
            "lastTransitionTime": format_condition_time(started_at),
        },
        {
            "type": CONDITION_COMPLETED,
            "status": "True",
            "reason": completed_reason,
            "message": "Step completed" if succeeded else "Step failed",
            "lastTransitionTime": format_condition_time(completed_at),
        },
    ]


def _infer_failure_reason(
    agent_output: dict[str, Any],
    failure_reason: str | None,
) -> str | None:
    """Resolve failureReason from explicit input or agent success=false."""
    if failure_reason is not None:
        return failure_reason
    if agent_output.get("success") is False:
        summary = agent_output.get("summary")
        if isinstance(summary, str) and summary:
            return summary
        return "Agent reported failure"
    return None


def _agent_succeeded(agent_output: dict[str, Any], failure_reason: str | None) -> bool:
    """Whether the agent run should be recorded as succeeded on the Result CR."""
    if failure_reason is not None:
        return False
    success = agent_output.get("success")
    if success is False:
        return False
    if success is True:
        return True
    # Analysis schema has no top-level success — absence means agent completed.
    return True


def _strip_empty_values(obj: Any) -> None:
    """Remove empty strings, lists, and dicts in-place so CRD constraints hold.

    Avoids violating minLength/minItems/minProperties on published status.
    LLMs often produce empty placeholders (e.g. ``"rootCause": ""``,
    ``"clusterScoped": []``) that are valid JSON but violate Kubernetes
    CRD validation rules.
    """
    if not isinstance(obj, dict):
        return
    for key in list(obj):
        val = obj[key]
        if isinstance(val, dict):
            _strip_empty_values(val)
            if not val:
                del obj[key]
        elif isinstance(val, list):
            for item in val:
                _strip_empty_values(item)
            if not val:
                del obj[key]
        elif isinstance(val, str) and val == "":
            del obj[key]


def _truncate(value: str, max_len: int) -> str:
    """Truncate a string to *max_len* characters."""
    if len(value) <= max_len:
        return value
    return value[:max_len]


def _sanitize_analysis_options(
    options: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Sanitize analysis options so they pass CRD validation.

    Returns ``(options, errors)`` where *errors* lists structural problems
    (pairing violations, incomplete diagnosis) that must cause the step to
    fail.  Oversized-string truncation is silent salvage; structural
    incompleteness is not.
    """
    errors: list[str] = []
    options[:] = [o for o in options if isinstance(o, dict)]
    for idx, opt in enumerate(options):
        # --- truncate top-level option fields ---
        if isinstance(opt.get("title"), str):
            opt["title"] = _truncate(opt["title"], _MAX_LEN_OPTION_TITLE)
        if isinstance(opt.get("summary"), str):
            opt["summary"] = _truncate(opt["summary"], _MAX_LEN_OPTION_SUMMARY)

        # --- diagnosis / remediationPlan pairing ---
        diag = opt.get("diagnosis")
        has_plan = "remediationPlan" in opt
        diag_valid = False
        if isinstance(diag, dict):
            has_summary = isinstance(diag.get("summary"), str) and diag["summary"]
            has_root = isinstance(diag.get("rootCause"), str) and diag["rootCause"]
            if has_summary and has_root:
                diag["summary"] = _truncate(diag["summary"], _MAX_LEN_DIAGNOSIS_SUMMARY)
                diag["rootCause"] = _truncate(diag["rootCause"], _MAX_LEN_DIAGNOSIS_ROOT_CAUSE)
                diag_valid = True
            else:
                errors.append(f"option {idx}: incomplete diagnosis")
                opt.pop("diagnosis", None)
                opt.pop("remediationPlan", None)
        elif has_plan:
            errors.append(f"option {idx}: remediationPlan without diagnosis")
            opt.pop("diagnosis", None)
            opt.pop("remediationPlan", None)

        if diag_valid and not has_plan:
            errors.append(f"option {idx}: diagnosis without remediationPlan")
            opt.pop("diagnosis", None)

        # --- remediation plan ---
        plan = opt.get("remediationPlan")
        if isinstance(plan, dict):
            if isinstance(plan.get("description"), str):
                plan["description"] = _truncate(plan["description"], _MAX_LEN_PLAN_DESCRIPTION)
            for action in plan.get("actions") or []:
                if not isinstance(action, dict):
                    continue
                if isinstance(action.get("command"), str):
                    action["command"] = _truncate(action["command"], _MAX_LEN_ACTION_COMMAND)
                if isinstance(action.get("type"), str):
                    action["type"] = _truncate(action["type"], _MAX_LEN_ACTION_TYPE)
                if isinstance(action.get("description"), str):
                    max_len = _MAX_LEN_ACTION_DESCRIPTION
                    action["description"] = _truncate(action["description"], max_len)
            rollback = plan.get("rollbackPlan")
            if isinstance(rollback, dict):
                if isinstance(rollback.get("description"), str):
                    max_len = _MAX_LEN_ROLLBACK_DESCRIPTION
                    rollback["description"] = _truncate(rollback["description"], max_len)
                if isinstance(rollback.get("command"), str):
                    rollback["command"] = _truncate(rollback["command"], _MAX_LEN_ROLLBACK_COMMAND)

        # --- verification ---
        verif = opt.get("verification")
        if isinstance(verif, dict):
            if isinstance(verif.get("description"), str):
                max_len = _MAX_LEN_VERIFICATION_DESCRIPTION
                verif["description"] = _truncate(verif["description"], max_len)
            for step in verif.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                if isinstance(step.get("name"), str):
                    step["name"] = _truncate(step["name"], _MAX_LEN_VERIFICATION_STEP_NAME)
                if isinstance(step.get("command"), str):
                    step["command"] = _truncate(step["command"], _MAX_LEN_VERIFICATION_STEP_COMMAND)
                if isinstance(step.get("expected"), str):
                    max_len = _MAX_LEN_VERIFICATION_STEP_EXPECTED
                    step["expected"] = _truncate(step["expected"], max_len)
                if isinstance(step.get("type"), str):
                    step["type"] = _truncate(step["type"], _MAX_LEN_VERIFICATION_STEP_TYPE)

        # --- rbac ---
        rbac = opt.get("rbac")
        if isinstance(rbac, dict):
            for scope in ("namespaceScoped", "clusterScoped"):
                for rule in rbac.get(scope) or []:
                    if not isinstance(rule, dict):
                        continue
                    if isinstance(rule.get("justification"), str):
                        max_len = _MAX_LEN_RBAC_JUSTIFICATION
                        rule["justification"] = _truncate(rule["justification"], max_len)

    return options, errors


def build_status(
    kind: str,
    agent_output: dict[str, Any],
    *,
    failure_reason: str | None = None,
    started_at: datetime,
    completed_at: datetime,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict[str, Any]:
    """Assemble a Result CR status dict from agent output and lifecycle metadata.

    Parameters
    ----------
    kind:
        Result CR kind from result-template (e.g. AnalysisResult).
    agent_output:
        Parsed structured agent JSON (shape from /input/output-schema).
    failure_reason:
        Sandbox-level agent failure message (infra success, agent failed).
    started_at, completed_at:
        Wall-clock bounds for Started / Completed conditions.
    """
    fields = _STATUS_FIELDS_BY_KIND.get(kind)
    if fields is None:
        msg = f"unsupported Result kind: {kind!r}"
        raise ValueError(msg)

    resolved_failure = _infer_failure_reason(agent_output, failure_reason)
    succeeded = _agent_succeeded(agent_output, resolved_failure)

    status: dict[str, Any] = {}
    if resolved_failure is not None:
        status["failureReason"] = resolved_failure

    for key in fields:
        if key not in agent_output:
            continue
        value = agent_output[key]
        if key == "actionRequired" and isinstance(value, bool):
            value = action_required_value(value)
        status[key] = value

    if kind == "AnalysisResult" and isinstance(status.get("options"), list):
        status["options"], sanitize_errors = _sanitize_analysis_options(status["options"])
        if sanitize_errors:
            reason = "; ".join(sanitize_errors)
            logger.warning("analysis option pairing violations: %s", reason)
            status["failureReason"] = reason
            succeeded = False

    _strip_empty_values(status)

    status["tokenUsage"] = {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
    }
    status["conditions"] = build_conditions(
        started_at=started_at,
        completed_at=completed_at,
        succeeded=succeeded,
    )
    return status
