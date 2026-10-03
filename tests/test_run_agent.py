"""Tests for run_agent_query and context formatting."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from lightspeed_agentic.inspection.errors import ToolResultSafetyInspectionFailed
from lightspeed_agentic.inspection.middleware import ToolResultInspectionMiddleware
from lightspeed_agentic.run_agent import ContextFormatError, format_context_prefix, run_agent_query
from lightspeed_agentic.types import ProviderEvent, ProviderQueryOptions, ResultEvent

from .conftest import MockProvider


@pytest.mark.asyncio
async def test_run_agent_query_shapes_structured_output() -> None:
    provider = MockProvider(
        events=[
            ResultEvent(
                text='{"success": true, "summary": "fixed", "actionRequired": false, "options": []}'
            )
        ]
    )
    result = await run_agent_query(
        provider,
        prompt="Diagnose the issue",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output == {
        "success": True,
        "summary": "fixed",
        "actionRequired": False,
        "options": [],
    }


@pytest.mark.asyncio
async def test_run_agent_query_passes_system_prompt() -> None:
    provider = MockProvider()
    result = await run_agent_query(
        provider,
        prompt="test",
        system_prompt="Custom persona",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is True
    assert provider.last_options is not None
    assert provider.last_options.system_prompt == "Custom persona"


@pytest.mark.asyncio
async def test_run_agent_query_with_context() -> None:
    provider = MockProvider()
    result = await run_agent_query(
        provider,
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
        timeout_seconds=300,
    )
    assert result.output["success"] is True
    assert provider.last_options is not None
    assert "Target namespaces: default" in provider.last_options.prompt
    assert "Attempt 1: timeout" in provider.last_options.prompt
    assert provider.last_options.prompt.endswith("fix it")


@pytest.mark.asyncio
async def test_run_agent_query_with_output_schema() -> None:
    schema = {"type": "object", "properties": {"success": {"type": "boolean"}}}
    provider = MockProvider()
    result = await run_agent_query(
        provider,
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=schema,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is True
    assert provider.last_options is not None
    assert provider.last_options.output_schema == schema


@pytest.mark.asyncio
async def test_run_agent_query_accepts_traceparent(span_exporter) -> None:
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    result = await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
        traceparent=traceparent,
    )

    assert result.output["success"] is True
    agent_span = next(
        s for s in span_exporter.get_finished_spans() if s.name == "invoke_agent lightspeed"
    )
    operator_trace_id = int("4bf92f3577b34da6a3ce929d0e0e4736", 16)
    assert agent_span.parent is not None
    assert agent_span.context.trace_id == operator_trace_id
    assert agent_span.parent.trace_id == operator_trace_id
    assert agent_span.parent.span_id == int("00f067aa0ba902b7", 16)
    assert agent_span.kind == SpanKind.INTERNAL
    assert agent_span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert agent_span.attributes["gen_ai.agent.name"] == "lightspeed"


@pytest.mark.asyncio
async def test_run_agent_query_stamps_agent_correlation(span_exporter) -> None:
    await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
        agenticrun_uid="run-uid",
        step="execution",
    )

    agent_span = next(
        s for s in span_exporter.get_finished_spans() if s.name == "invoke_agent lightspeed"
    )
    attrs = dict(agent_span.attributes)
    assert attrs["agenticrun.uid"] == "run-uid"
    assert attrs["agenticrun.phase"] == "execution"


@pytest.mark.asyncio
async def test_run_agent_query_does_not_invent_correlation(span_exporter) -> None:
    await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )

    agent_span = next(
        s for s in span_exporter.get_finished_spans() if s.name == "invoke_agent lightspeed"
    )
    attrs = dict(agent_span.attributes)
    assert "agenticrun.uid" not in attrs
    assert "agenticrun.phase" not in attrs


@pytest.mark.parametrize("traceparent", [None, "not-a-traceparent"])
@pytest.mark.asyncio
async def test_agent_and_native_spans_use_standard_shapes_and_sibling_parentage(
    span_exporter,
    caplog: pytest.LogCaptureFixture,
    traceparent: str | None,
) -> None:
    context = {
        "targetNamespaces": ["default"],
        "previousAttempts": [{"attempt": 1, "failureReason": "timeout"}],
    }
    effective_prompt = (
        "[context]\n"
        "Target namespaces: default\n"
        "Previous attempts:\n"
        "  Attempt 1: timeout\n"
        "[/context]\n\n"
        "effective user input"
    )

    caplog.set_level(logging.INFO, logger="lightspeed_agentic")

    class HookedProvider(MockProvider):
        async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            audit_logger = options.audit_logger
            assert audit_logger is not None
            inference = audit_logger.start_inference(
                model=options.model,
                operation="chat",
                input_messages=[
                    {"role": "user", "parts": [{"type": "text", "content": options.prompt}]}
                ],
                system_instructions=[{"type": "text", "content": options.system_prompt}],
                tool_definitions=[{"type": "function", "name": "execute"}],
            )
            audit_logger.end_inference(
                inference,
                output_messages=[
                    {
                        "role": "assistant",
                        "parts": [{"type": "text", "content": "model answer"}],
                        "finish_reason": "stop",
                    }
                ],
                input_tokens=3,
                output_tokens=4,
                finish_reasons=["stop"],
            )
            tool = audit_logger.start_tool(
                name="execute",
                call_id="call-1",
                arguments={"command": "true"},
            )
            audit_logger.end_tool(tool, result={"exit_code": 0})
            yield ResultEvent(text='{"success": true, "summary": "done"}')

    tracer = trace.get_tracer("test.run_agent.ambient")
    with tracer.start_as_current_span("ambient-local-span") as ambient_span:
        assert ambient_span.is_recording()
        result = await run_agent_query(
            HookedProvider(),
            prompt="effective user input",
            system_prompt="separate instructions",
            output_schema={"type": "object"},
            context=context,
            skills_dir="/workspace",
            model="requested-model",
            max_turns=200,
            timeout_seconds=300,
            agenticrun_uid="run-uid",
            step="analysis",
            traceparent=traceparent,
        )

    assert result.output["success"] is True
    spans = span_exporter.get_finished_spans()
    agent_span = next(s for s in spans if s.name == "invoke_agent lightspeed")
    ambient_export = next(s for s in spans if s.name == "ambient-local-span")
    assert ambient_export.context.trace_id != agent_span.context.trace_id
    assert agent_span.parent is None
    assert f"trace_id={agent_span.context.trace_id:032x}" in caplog.text
    inference = next(s for s in spans if s.name == "chat requested-model")
    tool = next(s for s in spans if s.name == "execute_tool execute")
    agent_attrs = dict(agent_span.attributes)
    assert agent_attrs["gen_ai.request.model"] == "requested-model"
    assert agent_attrs["gen_ai.output.type"] == "json"
    assert json.loads(agent_attrs["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "parts": [{"type": "text", "content": effective_prompt}],
        }
    ]
    assert json.loads(inference.attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "parts": [{"type": "text", "content": effective_prompt}],
        }
    ]
    assert json.loads(agent_attrs["gen_ai.system_instructions"]) == [
        {"type": "text", "content": "separate instructions"}
    ]
    output_messages = json.loads(agent_attrs["gen_ai.output.messages"])
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"][0]["type"] == "text"
    assert json.loads(output_messages[0]["parts"][0]["content"]) == result.output
    assert output_messages[0]["finish_reason"] == "unknown"
    assert inference.parent.span_id == agent_span.context.span_id
    assert tool.parent.span_id == agent_span.context.span_id
    assert inference.context.trace_id == agent_span.context.trace_id
    assert inference.parent.trace_id == agent_span.context.trace_id
    assert tool.context.trace_id == agent_span.context.trace_id
    assert tool.parent.trace_id == agent_span.context.trace_id
    assert inference.kind == SpanKind.CLIENT
    assert tool.kind == SpanKind.INTERNAL
    assert agent_span.status.status_code == StatusCode.UNSET
    assert inference.status.status_code == StatusCode.UNSET
    assert tool.status.status_code == StatusCode.UNSET
    assert all(not span.events for span in spans)
    assert all(
        span.attributes["agenticrun.uid"] == "run-uid" for span in (agent_span, inference, tool)
    )
    assert all(
        span.attributes["agenticrun.phase"] == "analysis" for span in (agent_span, inference, tool)
    )


@pytest.mark.asyncio
async def test_run_agent_query_re_raises_tool_result_safety_failure(span_exporter) -> None:
    class SafetyFailureProvider(MockProvider):
        async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            assert options.audit_logger is not None
            options.audit_logger.start_tool(
                name="execute",
                call_id="rejected-call",
                arguments={"command": "secret"},
            )
            raise ToolResultSafetyInspectionFailed()
            yield  # pragma: no cover

    with pytest.raises(ToolResultSafetyInspectionFailed):
        await run_agent_query(
            SafetyFailureProvider(),
            prompt="test",
            system_prompt="You are an AI agent.",
            output_schema=None,
            context=None,
            skills_dir="/workspace",
            model="test-model",
            max_turns=200,
            timeout_seconds=300,
        )

    spans = span_exporter.get_finished_spans()
    agent_span = next(s for s in spans if s.name == "invoke_agent lightspeed")
    tool_span = next(s for s in spans if s.name == "execute_tool execute")
    assert agent_span.status.status_code == StatusCode.ERROR
    assert agent_span.attributes["error.type"] == "ToolResultSafetyInspectionFailed"
    assert "gen_ai.output.messages" not in agent_span.attributes
    assert tool_span.status.status_code == StatusCode.ERROR
    assert "gen_ai.tool.call.result" not in tool_span.attributes


@pytest.mark.asyncio
async def test_run_agent_query_deadline_during_inspection_is_safety_failure() -> None:
    blocked = asyncio.Event()

    async def inspect(_name: str, _result_type: str, _content: object, _call_id: str) -> None:
        await blocked.wait()

    class BlockedInspectorProvider(MockProvider):
        async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            middleware = ToolResultInspectionMiddleware(inspect)
            request = SimpleNamespace(
                messages=[ToolMessage(content="result", name="execute", tool_call_id="call-1")]
            )

            async def model_handler(_request: object) -> None:
                return None

            await middleware.awrap_model_call(request, model_handler)
            yield ResultEvent(text='{"success":true}')

    with pytest.raises(ToolResultSafetyInspectionFailed):
        await run_agent_query(
            BlockedInspectorProvider(),
            prompt="test",
            system_prompt="You are an AI agent.",
            output_schema=None,
            context=None,
            skills_dir="/workspace",
            model="test-model",
            max_turns=200,
            timeout_seconds=0.05,
        )


@pytest.mark.asyncio
async def test_run_agent_query_timeout_closes_open_children(span_exporter) -> None:
    class HangingProvider(MockProvider):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            assert options.audit_logger is not None
            options.audit_logger.start_tool(
                name="execute",
                call_id="running-call",
                arguments={"command": "running"},
            )
            self.started.set()
            await asyncio.Event().wait()
            yield ResultEvent(text="too late")

    provider = HangingProvider()
    result = await run_agent_query(
        provider,
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=0.05,
    )

    assert provider.started.is_set()
    assert result.output["success"] is False
    assert result.timed_out is True
    spans = span_exporter.get_finished_spans()
    agent_span = next(s for s in spans if s.name == "invoke_agent lightspeed")
    tool_span = next(s for s in spans if s.name == "execute_tool execute")
    assert agent_span.status.status_code == StatusCode.ERROR
    assert agent_span.attributes["error.type"] == "TimeoutError"
    assert "gen_ai.output.messages" not in agent_span.attributes
    assert tool_span.status.status_code == StatusCode.ERROR
    assert tool_span.attributes["error.type"] == "TimeoutError"
    assert "gen_ai.tool.call.result" not in tool_span.attributes


@pytest.mark.asyncio
async def test_run_agent_query_cancellation_closes_open_children(span_exporter) -> None:
    class HangingProvider(MockProvider):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            assert options.audit_logger is not None
            options.audit_logger.start_inference(
                model=options.model,
                operation="chat",
                input_messages=None,
            )
            self.started.set()
            await asyncio.Event().wait()
            yield ResultEvent(text="too late")

    provider = HangingProvider()
    task = asyncio.create_task(
        run_agent_query(
            provider,
            prompt="test",
            system_prompt="You are an AI agent.",
            output_schema=None,
            context=None,
            skills_dir="/workspace",
            model="test-model",
            max_turns=200,
            timeout_seconds=300,
        )
    )
    await provider.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    spans = span_exporter.get_finished_spans()
    agent_span = next(s for s in spans if s.name == "invoke_agent lightspeed")
    inference = next(s for s in spans if s.name == "chat test-model")
    assert agent_span.status.status_code == StatusCode.ERROR
    assert agent_span.attributes["error.type"] == "CancelledError"
    assert "gen_ai.output.messages" not in agent_span.attributes
    assert inference.status.status_code == StatusCode.ERROR
    assert inference.attributes["error.type"] == "CancelledError"


@pytest.mark.asyncio
async def test_run_agent_query_exception_closes_open_children_without_logging_details(
    span_exporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class RaisingProvider(MockProvider):
        async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            assert options.audit_logger is not None
            options.audit_logger.start_inference(
                model=options.model,
                operation="chat",
                input_messages=None,
            )
            raise RuntimeError("secret provider detail")
            yield  # pragma: no cover

    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    result = await run_agent_query(
        RaisingProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )

    assert result.output["success"] is False
    assert "secret provider detail" not in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    spans = span_exporter.get_finished_spans()
    agent_span = next(s for s in spans if s.name == "invoke_agent lightspeed")
    inference = next(s for s in spans if s.name == "chat test-model")
    assert agent_span.status.status_code == StatusCode.ERROR
    assert agent_span.attributes["error.type"] == "RuntimeError"
    assert "gen_ai.output.messages" not in agent_span.attributes
    assert inference.status.status_code == StatusCode.ERROR
    assert inference.attributes["error.type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_run_agent_query_empty_response_is_an_error_without_output_content(
    span_exporter,
) -> None:
    result = await run_agent_query(
        MockProvider(events=[ResultEvent(text="")]),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is False
    assert result.output["summary"] == "Agent returned empty response"
    agent_span = next(
        s for s in span_exporter.get_finished_spans() if s.name == "invoke_agent lightspeed"
    )
    assert agent_span.status.status_code == StatusCode.ERROR
    assert agent_span.attributes["error.type"] == "empty_response"
    assert "gen_ai.output.messages" not in agent_span.attributes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected_output"),
    [
        (
            "The deployment is healthy.",
            {"success": True, "summary": "The deployment is healthy."},
        ),
        (
            '{ "ticketId": "CASE-1" }',
            {
                "success": True,
                "summary": '{ "ticketId": "CASE-1" }',
                "ticketId": "CASE-1",
            },
        ),
        (
            '{ "success": false, "summary": "Could not repair", "ticketId": "CASE-2" }',
            {
                "success": False,
                "summary": "Could not repair",
                "ticketId": "CASE-2",
            },
        ),
    ],
)
async def test_terminal_trace_matches_returned_shaped_result(
    span_exporter, text: str, expected_output: dict[str, object]
) -> None:
    result = await run_agent_query(
        MockProvider(events=[ResultEvent(text=text)]),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output == expected_output

    agent_span = next(
        s for s in span_exporter.get_finished_spans() if s.name == "invoke_agent lightspeed"
    )
    message = json.loads(agent_span.attributes["gen_ai.output.messages"])[0]
    assert message["role"] == "assistant"
    assert message["parts"][0]["type"] == "text"
    assert json.loads(message["parts"][0]["content"]) == result.output
    assert message["finish_reason"] == "unknown"
    assert agent_span.status.status_code == StatusCode.UNSET
    assert "error.type" not in agent_span.attributes


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
        timeout_seconds=300,
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
        timeout_seconds=300,
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
        timeout_seconds=1,
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
        timeout_seconds=300,
    )
    assert result.input_tokens == 10
    assert result.output_tokens == 5
