"""Tests for run_agent_query and context formatting."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from lightspeed_agentic.run_agent import ContextFormatError, format_context_prefix, run_agent_query
from lightspeed_agentic.types import (
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
    ToolCallEvent,
    ToolResultEvent,
)

from .conftest import MockProvider


@pytest.mark.asyncio
async def test_run_agent_query_success() -> None:
    """Mock provider completes and returns a successful structured result."""
    result = await run_agent_query(
        MockProvider(),
        prompt="Diagnose the issue",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=300_000,
    )
    assert result.output["success"] is True
    assert "mock result" in result.output["summary"]


@pytest.mark.asyncio
async def test_run_agent_query_with_system_prompt() -> None:
    """Custom system_prompt is accepted without changing success semantics."""
    result = await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="Custom persona",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=300_000,
    )
    assert result.output["success"] is True


@pytest.mark.asyncio
async def test_run_agent_query_with_context() -> None:
    """Workflow context dict is formatted and passed through to the provider."""
    result = await run_agent_query(
        MockProvider(),
        prompt="fix it",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context={
            "targetNamespaces": ["default"],
            "previousAttempts": [{"attempt": 1, "failureReason": "timeout"}],
        },
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=300_000,
    )
    assert result.output["success"] is True


@pytest.mark.asyncio
async def test_run_agent_query_with_output_schema() -> None:
    """output_schema is forwarded to the provider query options."""
    schema = {"type": "object", "properties": {"success": {"type": "boolean"}}}
    result = await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=schema,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=300_000,
    )
    assert result.output["success"] is True


@pytest.mark.asyncio
async def test_run_agent_query_accepts_traceparent() -> None:
    """W3C traceparent (operator TRACEPARENT env) links inference span to phase trace."""
    result = await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=300_000,
        traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    )
    assert result.output["success"] is True


@pytest.mark.asyncio
async def test_run_agent_query_timeout() -> None:
    """Wall-clock timeout yields agent failure with a timed-out summary."""

    class SlowProvider(MockProvider):
        async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            await asyncio.sleep(0.05)
            async for event in super().query(options):
                yield event

    result = await run_agent_query(
        SlowProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=1,
    )
    assert result.output["success"] is False
    assert "timed out" in result.output["summary"]


@pytest.mark.asyncio
async def test_run_agent_query_empty_response() -> None:
    """Empty ResultEvent text is treated as agent failure."""
    result = await run_agent_query(
        MockProvider(events=[ResultEvent(text="")]),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=300_000,
    )
    assert result.output["success"] is False
    assert result.output["summary"] == "Agent returned empty response"


@pytest.mark.asyncio
async def test_run_agent_query_text_response() -> None:
    """Plain-text ResultEvent becomes summary when no JSON schema is required."""
    result = await run_agent_query(
        MockProvider(events=[ResultEvent(text="plain text answer")]),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=300_000,
    )
    assert result.output["success"] is True
    assert result.output["summary"] == "plain text answer"


@pytest.mark.asyncio
async def test_run_agent_query_audit_enabled() -> None:
    """audit_enabled=True runs the audit path without changing agent outcome."""
    result = await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=300_000,
        audit_enabled=True,
    )
    assert result.output["success"] is True


@pytest.mark.asyncio
async def test_run_agent_query_audit_with_tool_events() -> None:
    """Tool call/result events are consumed when audit logging is enabled."""
    events = [
        ToolCallEvent(name="bash", input="ls"),
        ToolResultEvent(output="file.txt"),
        ResultEvent(
            text='{"success": true, "summary": "done"}',
            input_tokens=10,
            output_tokens=5,
        ),
    ]
    result = await run_agent_query(
        MockProvider(events=events),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=300_000,
        audit_enabled=True,
    )
    assert result.output["success"] is True


def test_format_context_envelope_markers_only() -> None:
    """Rule 12: block starts and ends with fixed marker lines."""
    text = format_context_prefix({})
    assert text == "[context]\n[/context]"


def test_format_context_unknown_keys_ignored() -> None:
    """Keys outside the supported context schema are omitted from the prefix."""
    text = format_context_prefix({"workflowPhase": "diagnose"})
    assert text == "[context]\n[/context]"


def test_format_context_target_namespaces() -> None:
    """Rule 13: comma-separated namespace list."""
    text = format_context_prefix({"targetNamespaces": ["default", "kube-system"]})
    assert "Target namespaces: default, kube-system" in text
    assert text.startswith("[context]")
    assert text.endswith("[/context]")


def test_format_context_target_namespaces_empty_list_omitted() -> None:
    """Empty targetNamespaces list produces no namespace line."""
    text = format_context_prefix({"targetNamespaces": []})
    assert "Target namespaces:" not in text


def test_format_context_attempt_includes_of_max_literal() -> None:
    """Rule 14: attempt line uses literal 'of max' placeholder."""
    text = format_context_prefix({"attempt": 2})
    assert "Attempt: 2 of max" in text


def test_format_context_attempt_zero_included() -> None:
    """Attempt zero is formatted like any other attempt number."""
    text = format_context_prefix({"attempt": 0})
    assert "Attempt: 0 of max" in text


def test_format_context_previous_attempts_with_failure_reason() -> None:
    """Previous attempts list failure reasons when present."""
    text = format_context_prefix(
        {
            "previousAttempts": [
                {"attempt": 1, "failureReason": "timeout"},
                {"attempt": 2},
            ],
        }
    )
    assert "  Attempt 1: timeout" in text
    assert "  Attempt 2" in text
    assert "  Attempt 2:" not in text


def test_format_context_previous_attempts_empty_list_omitted() -> None:
    """Empty previousAttempts list produces no attempts section."""
    text = format_context_prefix({"previousAttempts": []})
    assert "Previous attempts:" not in text


def test_format_context_approved_option_with_actions() -> None:
    """Approved option remediation actions are listed under Actions to execute."""
    text = format_context_prefix(
        {
            "approvedOption": {
                "title": "Restart pod",
                "diagnosis": {"rootCause": "CrashLoopBackOff"},
                "remediationPlan": {
                    "description": "Delete pod to trigger restart",
                    "reversible": True,
                    "actions": [
                        {
                            "type": "mutation",
                            "description": "Delete the crashing pod",
                        },
                    ],
                },
            },
        }
    )
    assert "Title: Restart pod" in text
    assert "  - [mutation] Delete the crashing pod" in text


def test_format_context_approved_option_with_command() -> None:
    """Action commands are included in the formatted remediation plan."""
    text = format_context_prefix(
        {
            "approvedOption": {
                "title": "Patch configmap",
                "diagnosis": {"rootCause": "wrong value"},
                "remediationPlan": {
                    "description": "Patch configmap data",
                    "actions": [
                        {
                            "type": "mutation",
                            "command": 'kubectl patch configmap foo -p \'{"data":{"k":"v"}}\'',
                            "description": "Apply patch",
                        },
                    ],
                },
            },
        }
    )
    assert "kubectl patch configmap foo" in text


def test_format_context_approved_option_without_actions() -> None:
    """Remediation plan without actions omits the Actions to execute section."""
    text = format_context_prefix(
        {
            "approvedOption": {
                "title": "Manual step",
                "diagnosis": {"rootCause": "needs human"},
                "remediationPlan": {"description": "Contact admin"},
            },
        }
    )
    assert "Title: Manual step" in text
    assert "Actions to execute:" not in text


def test_format_context_combined_fields() -> None:
    """All supported context fields appear together inside the envelope."""
    text = format_context_prefix(
        {
            "targetNamespaces": ["openshift-logging"],
            "attempt": 3,
            "previousAttempts": [{"attempt": 2, "failureReason": "denied"}],
            "approvedOption": {
                "title": "Fix RBAC",
                "diagnosis": {"rootCause": "missing role"},
                "remediationPlan": {
                    "description": "Apply RoleBinding",
                    "reversible": False,
                },
            },
        }
    )
    lines = text.splitlines()
    assert lines[0] == "[context]"
    assert lines[-1] == "[/context]"
    assert "Target namespaces: openshift-logging" in text
    assert "Attempt: 3 of max" in text
    assert "  Attempt 2: denied" in text
    assert "Title: Fix RBAC" in text


def test_format_context_approved_option_missing_diagnosis() -> None:
    """Missing approvedOption.diagnosis raises ContextFormatError."""
    with pytest.raises(ContextFormatError, match=r"approvedOption\.diagnosis"):
        format_context_prefix(
            {
                "approvedOption": {
                    "title": "Fix",
                    "remediationPlan": {"description": "plan"},
                },
            }
        )


def test_format_context_approved_option_missing_root_cause() -> None:
    """Missing approvedOption.diagnosis.rootCause raises ContextFormatError."""
    with pytest.raises(ContextFormatError, match=r"approvedOption\.diagnosis\.rootCause"):
        format_context_prefix(
            {
                "approvedOption": {
                    "title": "Fix",
                    "diagnosis": {},
                    "remediationPlan": {"description": "plan"},
                },
            }
        )


def test_format_context_previous_attempts_missing_attempt() -> None:
    """Previous attempt entry without attempt number raises ContextFormatError."""
    with pytest.raises(ContextFormatError, match="previousAttempts\\[0\\] missing attempt"):
        format_context_prefix({"previousAttempts": [{"failureReason": "timeout"}]})


@pytest.mark.asyncio
async def test_run_agent_query_invalid_context_returns_agent_failure() -> None:
    """Invalid context formatting returns agent failure instead of raising."""
    provider = MockProvider(events=[ResultEvent(text='{"success":true,"summary":"ok"}')])

    result = await run_agent_query(
        provider,
        prompt="run",
        system_prompt="sys",
        output_schema=None,
        context={"approvedOption": {"title": "only title"}},
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=300_000,
    )

    assert result.output["success"] is False
    assert "Invalid context:" in result.output["summary"]
    assert "approvedOption.diagnosis" in result.output["summary"]


@pytest.mark.asyncio
async def test_run_agent_query_returns_token_counts() -> None:
    """Token counts from ResultEvent are included in the returned dict (OLS-3994)."""
    provider = MockProvider(
        events=[
            ResultEvent(
                text='{"success": true, "summary": "ok"}',
                input_tokens=500,
                output_tokens=200,
            )
        ]
    )
    result = await run_agent_query(
        provider,
        prompt="test",
        system_prompt="sys",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=300_000,
    )
    assert result.input_tokens == 500
    assert result.output_tokens == 200


@pytest.mark.asyncio
async def test_run_agent_query_token_counts_zero_on_timeout() -> None:
    """Token counts default to 0 when the agent times out (OLS-3994)."""

    class SlowProvider(MockProvider):
        async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            await asyncio.sleep(10)
            yield ResultEvent(text="late")

    result = await run_agent_query(
        SlowProvider(),
        prompt="test",
        system_prompt="sys",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=1,
    )
    assert result.input_tokens == 0
    assert result.output_tokens == 0


@pytest.mark.asyncio
async def test_run_agent_query_token_counts_on_text_response() -> None:
    """Token counts present on plain text (non-JSON) responses (OLS-3994)."""
    provider = MockProvider(
        events=[ResultEvent(text="plain text", input_tokens=10, output_tokens=5)]
    )
    result = await run_agent_query(
        provider,
        prompt="test",
        system_prompt="sys",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_ms=300_000,
    )
    assert result.input_tokens == 10
    assert result.output_tokens == 5
