"""Safe inspection error types."""

from __future__ import annotations

from typing import Any, Literal

CLASSIFIER_FAILURE_MESSAGE = (
    "Lightspeed stopped the operation because a tool result failed the safety inspection."
)


class ToolResultSafetyInspectionFailed(RuntimeError):  # noqa: N818
    """Stable termination marker for a rejected or uninspectable tool result."""

    def __init__(self) -> None:
        super().__init__("ToolResultSafetyInspectionFailed")


_PROVIDER_ERROR_TYPES = frozenset(
    {
        "invalid_request_error",
        "authentication_error",
        "permission_error",
        "not_found_error",
        "request_too_large",
        "rate_limit_error",
        "api_error",
        "overloaded_error",
        "server_error",
        "timeout_error",
    }
)
_PROVIDER_ERROR_CODES = frozenset(
    {
        "content_filter",
        "unsupported_parameter",
        "unsupported_value",
        "context_length_exceeded",
        "invalid_prompt",
        "invalid_request_error",
        "invalid_value",
    }
)


def provider_error_metadata(
    exc: BaseException,
) -> tuple[int | None, str | None, str | None]:
    """Extract bounded HTTP metadata without retaining provider response content."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int) or not 100 <= status_code <= 599:
        status_code = None

    body: Any = getattr(exc, "body", None)
    # OpenAI's native SDK unwraps {"error": ...}; other clients retain the envelope.
    error = body.get("error", body) if isinstance(body, dict) else {}
    if not isinstance(error, dict):
        error = {}
    error_type = error.get("type", getattr(exc, "type", None))
    if not isinstance(error_type, str) or error_type not in _PROVIDER_ERROR_TYPES:
        error_type = None

    code = error.get("code")
    if code is None:
        code = getattr(exc, "code", None)
    # Provider messages can contain tool content. Do not inspect or retain them;
    # unknown codes/types must not become arbitrary diagnostic strings either.
    reason = code if isinstance(code, str) and code in _PROVIDER_ERROR_CODES else "unknown"
    inner = error.get("innererror")
    if status_code == 400 and reason == "content_filter" and isinstance(inner, dict):
        filters = inner.get("content_filter_result")
        jailbreak = filters.get("jailbreak") if isinstance(filters, dict) else None
        if (
            inner.get("code") == "ResponsibleAIPolicyViolation"
            and isinstance(jailbreak, dict)
            and jailbreak.get("detected") is True
            and jailbreak.get("filtered") is True
        ):
            reason = "prompt_attack_filtered"
    if status_code is None and body is None and code is None and error_type is None:
        return None, None, None
    return status_code, error_type, reason


class ClassifierResponseError(ValueError):
    """Invalid classifier reply with a fixed, payload-free reason."""

    def __init__(
        self,
        issue: Literal[
            "missing_text",
            "non_json",
            "refusal",
            "invalid_envelope",
            "schema_mismatch",
            "output_limit",
            "incomplete_response",
            "provider_response_error",
        ],
    ) -> None:
        super().__init__("classifier response was not a recognized decision")
        self.issue = issue


class InspectionError(RuntimeError):
    """A classifier failure that carries no untrusted content."""

    def __init__(
        self,
        message: str = CLASSIFIER_FAILURE_MESSAGE,
        *,
        failure_type: str = "provider_error",
        attempt_count: int = 0,
        provider_status_code: int | None = None,
        provider_error_type: str | None = None,
        provider_error_reason: str | None = None,
        response_issue: str | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.attempt_count = attempt_count
        self.provider_status_code = provider_status_code
        self.provider_error_type = provider_error_type
        self.provider_error_reason = provider_error_reason
        self.response_issue = response_issue
