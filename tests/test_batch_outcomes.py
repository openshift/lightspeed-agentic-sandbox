"""Offline completed outcomes through batch, DeepAgents, and Result publication."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager, suppress
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest


@contextmanager
def _scripted_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    kind: str,
    schema: dict[str, Any],
    create_model: Any,
    provider: str = "anthropic",
    max_turns: int = 20,
    inspection_enabled: bool = False,
) -> Iterator[tuple[Any, dict[str, Any], list[tuple[str, dict[str, Any]]], Path]]:
    """Replace only model construction and external I/O around the real runtime."""
    from lightspeed_agentic import batch
    from lightspeed_agentic.providers.models import anthropic, gemini, openai
    from lightspeed_agentic.publish_results import publish as publisher

    template = {
        "apiVersion": "agentic.openshift.io/v1alpha1",
        "kind": kind,
        "metadata": {"name": "completed-run", "namespace": "default"},
        "spec": {"agenticRunName": "run"},
    }
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name, content in {
        "query": "Complete the requested work and report its outcome.",
        "output-schema": json.dumps(schema),
        "context": "{}",
        "result-template": json.dumps(template),
    }.items():
        (input_dir / name).write_text(content)
    termination_path = tmp_path / "termination.log"
    monkeypatch.setattr(
        batch, "read_batch_inputs", partial(batch.read_batch_inputs, str(input_dir))
    )
    monkeypatch.setattr(
        batch,
        "write_termination_log",
        partial(publisher.write_termination_log, path=str(termination_path)),
    )
    factories = {
        "anthropic": (anthropic, "create_direct_model"),
        "vertex/anthropic": (anthropic, "create_vertex_model"),
        "vertex/google": (gemini, "create_vertex_gemini_model"),
        "bedrock/anthropic": (anthropic, "create_bedrock_model"),
        "openai": (openai, "create_native_openai_model"),
        "openai-compatible": (openai, "create_openai_compatible_model"),
        "azure": (openai, "create_azure_model"),
    }

    def reject_wrong_route(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Batch constructed a model through the wrong route")

    for route_id, (module, factory_name) in factories.items():
        monkeypatch.setattr(
            module, factory_name, create_model if route_id == provider else reject_wrong_route
        )
    route_env: dict[str, str] = {}
    hosting = provider.split("/")[0]
    if hosting == "vertex":
        credentials_dir = tmp_path / "credentials"
        credentials_dir.mkdir()
        (credentials_dir / "GOOGLE_APPLICATION_CREDENTIALS").write_text("{}")
        route_env = {
            "LIGHTSPEED_MODEL_PROVIDER": provider.split("/")[1],
            "LIGHTSPEED_PROVIDER_PROJECT": "offline-project",
            "LIGHTSPEED_PROVIDER_REGION": "us-central1",
            "LIGHTSPEED_LLM_CREDENTIALS_PATH": str(credentials_dir),
        }
    elif hosting == "bedrock":
        route_env = {
            "LIGHTSPEED_PROVIDER_REGION": "us-east-1",
            "AWS_ACCESS_KEY_ID": "offline-access-key",
            "AWS_SECRET_ACCESS_KEY": "offline-secret-key",
        }
    elif hosting == "openai-compatible":
        hosting = "openai"
        route_env = {"LIGHTSPEED_PROVIDER_URL": "https://vllm.example.test/v1"}
    elif hosting == "azure":
        route_env = {
            "LIGHTSPEED_PROVIDER_URL": "https://azure.example.test",
            "LIGHTSPEED_PROVIDER_API_VERSION": "2024-08-01-preview",
            "AZURE_OPENAI_API_KEY": "offline-key",
            "LIGHTSPEED_LLM_CREDENTIALS_PATH": str(tmp_path / "credentials"),
        }
    calls: list[tuple[str, dict[str, Any]]] = []

    class RecordingApi:
        def create_namespaced_custom_object(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(("create", deepcopy(kwargs)))
            return {"metadata": {"resourceVersion": "1"}}

        def replace_namespaced_custom_object_status(self, **kwargs: Any) -> None:
            calls.append(("status", deepcopy(kwargs)))

    monkeypatch.setattr(publisher, "_default_api", RecordingApi)
    with patch.dict(
        os.environ,
        {
            "HOME": str(tmp_path),
            "LIGHTSPEED_PROVIDER": hosting,
            "LIGHTSPEED_MODEL": "offline-model",
            "ANTHROPIC_API_KEY": "offline-key",
            "OPENAI_API_KEY": "offline-key",
            "LIGHTSPEED_SKILLS_DIR": str(tmp_path),
            "LIGHTSPEED_AGENT_TIMEOUT_SECONDS": "30",
            "LIGHTSPEED_AGENT_MAX_TURNS": str(max_turns),
            "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED": str(inspection_enabled).lower(),
            **route_env,
        },
        clear=True,
    ):
        yield batch.main, template, calls, termination_path


def _run_scripted_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    with _scripted_batch(tmp_path, monkeypatch, **kwargs) as (
        main,
        template,
        calls,
        termination_path,
    ):
        assert main() is None  # Agent outcomes use the normal process exit-0 path.

    assert not termination_path.exists()
    assert [operation for operation, _ in calls] == ["create", "status"]
    assert calls[0][1]["body"] == template
    assert calls[1][1]["name"] == template["metadata"]["name"]
    assert all(kwargs["namespace"] == "default" for _, kwargs in calls)
    return template, calls


@pytest.mark.parametrize(
    "provider",
    [
        "anthropic",
        "vertex/anthropic",
        "vertex/google",
        "bedrock/anthropic",
        "openai",
        "openai-compatible",
        "azure",
    ],
)
@pytest.mark.parametrize("outcome", ["success", "domain-failure", "empty-terminal"])
def test_migrated_routes_preserve_batch_outcomes_events_limits_and_cleanup(
    provider: str,
    outcome: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    from langgraph.graph.state import CompiledStateGraph

    from lightspeed_agentic.logging import EventLogger
    from lightspeed_agentic.types import ResultEvent

    summary = "Work completed" if outcome == "success" else "Work could not be completed"
    output = {"success": outcome == "success", "summary": summary, "content": "Offline findings"}
    schema = {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "summary": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["success", "summary", "content"],
    }
    terminal_events: list[ResultEvent] = []
    recursion_limits: list[int] = []
    clients: list[Any] = []
    original_log = EventLogger.log
    original_astream = CompiledStateGraph.astream

    def observe_event(self: Any, event: Any) -> None:
        if isinstance(event, ResultEvent):
            terminal_events.append(event)
        original_log(self, event)

    async def observe_astream(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        recursion_limits.append(kwargs["config"]["recursion_limit"])
        async for event in original_astream(self, *args, **kwargs):
            yield event

    class TrackedClient:
        def __init__(self) -> None:
            self.loop = asyncio.get_running_loop()
            self.close_count = 0

        async def aclose(self) -> None:
            assert asyncio.get_running_loop() is self.loop
            self.close_count += 1

    class ScriptedModel(FakeMessagesListChatModel):
        openai_api_version: str = "2024-08-01-preview"

        def bind_tools(self, _tools: Any, **_kwargs: Any) -> Any:
            return self

        def with_structured_output(self, _schema: Any, **_kwargs: Any) -> Any:
            async def shape(messages: Any) -> dict[str, Any]:
                raw = await self.ainvoke(messages)
                return {"raw": raw, "parsed": json.loads(raw.content), "parsing_error": None}

            return RunnableLambda(shape)

    def create_model(parameters: Any, reasoning_config: Any = None) -> Any:
        assert parameters.model == "offline-model"
        assert reasoning_config is None
        usage = (
            {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
            if not clients
            else {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9}
        )
        model = ScriptedModel(
            responses=[
                AIMessage(
                    content="" if outcome == "empty-terminal" else json.dumps(output),
                    usage_metadata=usage,
                )
            ]
        )
        client = TrackedClient()
        clients.append(client)
        model.__dict__["_async_client"] = client
        return model

    monkeypatch.setattr(EventLogger, "log", observe_event)
    monkeypatch.setattr(CompiledStateGraph, "astream", observe_astream)
    _, calls = _run_scripted_batch(
        tmp_path,
        monkeypatch,
        kind="EscalationResult",
        schema={} if outcome == "empty-terminal" else schema,
        create_model=create_model,
        provider=provider,
        max_turns=71,
    )
    assert recursion_limits == [71]
    assert len(clients) == 1
    assert all(client.close_count == 1 for client in clients)
    assert len(terminal_events) == 1
    terminal = terminal_events[0]
    assert terminal.response_model == ""  # No observed response-model metadata in the script.
    expected_usage = {"inputTokens": 3, "outputTokens": 2}
    assert (terminal.input_tokens, terminal.output_tokens) == tuple(expected_usage.values())
    if outcome == "empty-terminal":
        assert terminal.text == ""
    else:
        assert json.loads(terminal.text) == output
    status = calls[1][1]["body"]["status"]
    conditions = status.pop("conditions")
    assert [(c["type"], c["status"], c["reason"]) for c in conditions] == [
        ("Started", "True", "StepStarted"),
        ("Completed", "True", "Succeeded" if outcome == "success" else "Failed"),
    ]
    expected: dict[str, Any] = {
        "summary": summary,
        "content": "Offline findings",
        "tokenUsage": expected_usage,
    }
    if outcome == "empty-terminal":
        expected = {
            "summary": "Agent returned empty response",
            "failureReason": "Agent returned empty response",
            "tokenUsage": expected_usage,
        }
    elif outcome == "domain-failure":
        expected["failureReason"] = summary
    assert status == expected


@pytest.mark.parametrize("max_turns", [37, 113])
def test_batch_max_turns_reaches_real_graph_recursion_limit(
    max_turns: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch dropped/defaulted limits anywhere between batch config and graph execution."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langgraph.graph.state import CompiledStateGraph

    recursion_limits: list[int] = []
    original_astream = CompiledStateGraph.astream

    async def observe_astream(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        recursion_limits.append(kwargs["config"]["recursion_limit"])
        async for event in original_astream(self, *args, **kwargs):
            yield event

    monkeypatch.setattr(CompiledStateGraph, "astream", observe_astream)

    class ScriptedModel(FakeMessagesListChatModel):
        def bind_tools(self, _tools: Any, **_kwargs: Any) -> Any:
            return self

    def create_model(_parameters: Any, reasoning_config: Any = None) -> Any:
        assert reasoning_config is None
        return ScriptedModel(
            responses=[AIMessage(content='{"success": true, "summary": "Work completed"}')]
        )

    _, calls = _run_scripted_batch(
        tmp_path,
        monkeypatch,
        kind="ExecutionResult",
        schema={},
        create_model=create_model,
        max_turns=max_turns,
    )
    # LangGraph recursion steps are not equivalent to the retired SDK's turn counts.
    assert recursion_limits == [max_turns]
    status = calls[1][1]["body"]["status"]
    assert "failureReason" not in status
    assert status["conditions"][-1]["reason"] == "Succeeded"


@pytest.mark.parametrize("success", [True, False], ids=["success", "domain-failure"])
@pytest.mark.parametrize("report_usage", [True, False], ids=["usage", "no-usage"])
@pytest.mark.parametrize(
    ("kind", "fields"),
    [
        ("AnalysisResult", {"actionRequired": False}),
        (
            "ExecutionResult",
            {
                "actionsTaken": [
                    {"type": "patch", "description": "Update deployment", "outcome": "Succeeded"}
                ]
            },
        ),
        (
            "VerificationResult",
            {"checks": [{"name": "ready", "result": "Passed", "source": "cmd", "value": "ok"}]},
        ),
        ("EscalationResult", {"content": "Operator intervention required"}),
    ],
)
def test_completed_batch_publishes_exact_outcome(
    kind: str,
    fields: dict[str, Any],
    success: bool,
    report_usage: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    summary = "Work completed" if success else "Unable to complete the requested work"
    output = {"success": success, "summary": summary, **fields}
    schema = {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "summary": {"type": "string"},
            **{
                name: {
                    "type": "boolean"
                    if isinstance(value, bool)
                    else "array"
                    if isinstance(value, list)
                    else "string",
                    **(
                        {
                            "items": {
                                "type": "object",
                                "properties": {key: {"type": "string"} for key in value[0]},
                                "required": list(value[0]),
                            }
                        }
                        if isinstance(value, list)
                        else {}
                    ),
                }
                for name, value in fields.items()
            },
        },
        "required": list(output),
    }
    model_calls: list[str] = []

    class ScriptedModel(FakeMessagesListChatModel):
        def bind_tools(self, _tools: Any, **_kwargs: Any) -> Any:
            return self

        def with_structured_output(self, _schema: Any, **_kwargs: Any) -> Any:
            async def shape(messages: Any) -> dict[str, Any]:
                raw = await self.ainvoke(messages)
                return {"raw": raw, "parsed": json.loads(raw.content), "parsing_error": None}

            return RunnableLambda(shape)

    def create_model(_parameters: Any, reasoning_config: Any = None) -> Any:
        assert reasoning_config is None
        role = "main" if not model_calls else "shaping"
        model_calls.append(role)
        usage = (
            {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
            if role == "main"
            else {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9}
        )
        return ScriptedModel(
            responses=[
                AIMessage(
                    content=json.dumps(output), usage_metadata=usage if report_usage else None
                )
            ]
        )

    _, calls = _run_scripted_batch(
        tmp_path, monkeypatch, kind=kind, schema=schema, create_model=create_model
    )
    assert model_calls == ["main"]
    status = calls[1][1]["body"]["status"]
    conditions = status["conditions"]
    assert len(conditions) == 2
    times = [condition["lastTransitionTime"] for condition in conditions]
    assert datetime.fromisoformat(times[0]) <= datetime.fromisoformat(times[1])
    expected = {
        **fields,
        "tokenUsage": {
            "inputTokens": 3 if report_usage else 0,
            "outputTokens": 2 if report_usage else 0,
        },
        "conditions": [
            {
                "type": "Started",
                "status": "True",
                "reason": "StepStarted",
                "message": "Step started",
                "lastTransitionTime": times[0],
            },
            {
                "type": "Completed",
                "status": "True",
                "reason": "Succeeded" if success else "Failed",
                "message": "Step completed" if success else "Step failed",
                "lastTransitionTime": times[1],
            },
        ],
    }
    if kind == "AnalysisResult":
        expected["actionRequired"] = "False"
    if kind in ("VerificationResult", "EscalationResult"):
        expected["summary"] = summary
    if not success:
        expected["failureReason"] = summary
    assert status == expected


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing-field", "Agent error: Structured output does not match configured schema"),
        ("wrong-type", "Agent error: Structured output does not match configured schema"),
        ("non-object", "Agent error: Structured output does not match configured schema"),
        ("empty-output", "Agent error: Structured output does not match configured schema"),
        ("missing-parsed", "Agent error: Structured output response is missing parsed output"),
        ("parsing-error", "Agent error: Structured output parsing failed"),
        ("main-error", "Agent error: Scripted provider failure"),
        ("shaping-error", "Agent error: Scripted provider failure"),
    ],
)
def test_invalid_output_and_provider_errors_publish_failed_result(
    failure: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    schema = {
        "type": "object",
        "properties": {"success": {"type": "boolean"}, "summary": {"type": "string"}},
        "required": ["success", "summary"],
        "additionalProperties": False,
    }
    model_calls: list[str] = []
    model_requests: list[str] = []

    class ScriptedModel(FakeMessagesListChatModel):
        def bind_tools(self, _tools: Any, **_kwargs: Any) -> Any:
            return self

        def _generate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> Any:
            model_requests.append("main")
            if failure == "main-error":
                raise RuntimeError("Scripted provider failure")
            return super()._generate(messages, stop, run_manager, **kwargs)

        def with_structured_output(self, _schema: Any, **_kwargs: Any) -> Any:
            async def shape(_messages: Any) -> dict[str, Any]:
                model_requests.append("shaping")
                if failure == "shaping-error":
                    raise RuntimeError("Scripted provider failure")
                parsed: Any = {
                    "missing-field": {"success": True},
                    "wrong-type": {"success": "true", "summary": "INVALID-OUTPUT"},
                    "non-object": ["INVALID-OUTPUT"],
                    "empty-output": None,
                }.get(failure, {"success": True, "summary": "INVALID-OUTPUT"})
                envelope = {
                    "raw": AIMessage(content="" if failure == "empty-output" else "INVALID-OUTPUT"),
                    "parsed": parsed,
                    "parsing_error": ValueError("INVALID-OUTPUT")
                    if failure == "parsing-error"
                    else None,
                }
                if failure == "missing-parsed":
                    envelope.pop("parsed")
                return envelope

            return RunnableLambda(shape)

    def create_model(_parameters: Any, reasoning_config: Any = None) -> Any:
        assert reasoning_config is None
        model_calls.append("main" if not model_calls else "shaping")
        return ScriptedModel(
            responses=[
                AIMessage(
                    content="" if failure == "empty-output" else "Agent draft",
                    usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                )
            ]
        )

    _, calls = _run_scripted_batch(
        tmp_path,
        monkeypatch,
        kind="ExecutionResult",
        schema=schema,
        create_model=create_model,
        provider="openai",
    )
    expected_model_calls = ["main"] if failure == "main-error" else ["main", "shaping"]
    expected_model_requests = {
        "main-error": ["main"],
        "shaping-error": ["main", "shaping"],
    }.get(failure, ["main", "shaping", "shaping"])
    assert model_calls == expected_model_calls
    assert model_requests == expected_model_requests
    status = calls[1][1]["body"]["status"]
    assert set(status) == {"failureReason", "tokenUsage", "conditions"}
    assert status["failureReason"] == message
    # No terminal ResultEvent was emitted, so partial main-call usage is not fabricated as a total.
    assert status["tokenUsage"] == {"inputTokens": 0, "outputTokens": 0}
    conditions = status["conditions"]
    assert [
        (condition["type"], condition["status"], condition["reason"]) for condition in conditions
    ] == [
        ("Started", "True", "StepStarted"),
        ("Completed", "True", "Failed"),
    ]
    assert "INVALID-OUTPUT" not in json.dumps(calls)


@pytest.mark.parametrize(
    ("stage", "interruption"),
    [
        ("main", "timeout"),
        ("shaping", "timeout"),
        ("inspection", "timeout"),
        ("main", "cancellation"),
        ("shaping", "cancellation"),
        ("inspection", "cancellation"),
        ("inspection", "inspection-deadline"),
        ("tool", "timeout"),
        ("tool", "cancellation"),
    ],
)
def test_batch_interruption_preserves_outcome_and_closes_clients(
    stage: str,
    interruption: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    from lightspeed_agentic.logging import EventLogger
    from lightspeed_agentic.providers.deepagents import DeepAgentsProvider

    entered = asyncio.Event()
    events: list[str] = []
    clients: dict[str, Any] = {}
    original_wait_for = asyncio.wait_for
    original_run = asyncio.run
    original_aclose = DeepAgentsProvider.aclose
    original_log = EventLogger.log
    clock_now = 1000.0
    provider_deadlines: list[float | None] = []
    inspection_deadlines: list[float | None] = []
    classifier_budgets: list[float] = []
    sentinel_file = tmp_path / "findings.txt"
    sentinel_file.write_text("PRIVATE-INTERRUPTED-TOOL-OUTPUT")

    def run_with_watchdog(awaitable: Any, **kwargs: Any) -> Any:
        async def guarded() -> Any:
            # Independent of the production wait_for call and its test interception.
            watchdog = asyncio.timeout(5)
            try:
                async with watchdog:
                    return await awaitable
            finally:
                # Still fail if cancellation was translated to another exception or result.
                if watchdog.expired():
                    pytest.fail("Independent batch watchdog expired after 5s")

        return original_run(guarded(), **kwargs)

    async def block(role: str) -> Any:
        events.append(f"{role}:blocked")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            events.append(f"{role}:unwound")

    if stage == "tool":
        import deepagents
        from langchain_core.tools import StructuredTool

        async def blocked_lookup() -> str:
            return await block("tool")

        tool = StructuredTool.from_function(
            name="blocked_lookup", description="Read external findings.", coroutine=blocked_lookup
        )
        original_create_agent = deepagents.create_deep_agent

        def with_blocking_tool(**kwargs: Any) -> Any:
            kwargs["tools"] = [*kwargs.get("tools", []), tool]
            return original_create_agent(**kwargs)

        monkeypatch.setattr(deepagents, "create_deep_agent", with_blocking_tool)

    async def interrupt_when_entered(awaitable: Any, timeout: Any) -> Any:  # noqa: ASYNC109 - matches asyncio.wait_for's signature
        nonlocal clock_now
        # Leave SDK/internal waits alone; target only the runner or classifier coroutine.
        target_inspection = interruption == "inspection-deadline"
        target_name = (
            "LangChainClassifierClient.classify"
            if target_inspection
            else "run_agent_query.<locals>.run"
        )
        code = getattr(awaitable, "cr_code", None)
        if getattr(code, "co_qualname", None) != target_name:
            return await original_wait_for(awaitable, timeout)
        assert 0 < timeout <= 30
        if target_inspection:
            classifier_budgets.append(timeout)
        else:
            assert timeout == 30
        task = asyncio.create_task(awaitable)
        try:
            # Synchronize on actual pending work; the outer test watchdog bounds this wait.
            await entered.wait()
            events.append(interruption)
            if interruption in ("timeout", "inspection-deadline"):
                # Use real wait_for cancellation/timeout conversion once the chosen call is pending.
                if target_inspection:
                    clock_now = 1031.0
                return await original_wait_for(task, 0)
            asyncio.current_task().cancel()
            return await task
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    def observe_event(self: Any, event: Any) -> None:
        events.append(f"event:{event.type}")
        original_log(self, event)

    async def observe_provider_close(self: Any) -> None:
        await original_aclose(self)
        events.append("provider:closed")

    class TrackedClient:
        def __init__(self, role: str) -> None:
            self.role = role
            self.loop = asyncio.get_running_loop()
            self.close_count = 0

        async def aclose(self) -> None:
            assert asyncio.get_running_loop() is self.loop
            self.close_count += 1
            events.append(f"{self.role}:closed")

    class ScriptedModel(FakeMessagesListChatModel):
        role: str

        def bind_tools(self, _tools: Any, **_kwargs: Any) -> Any:
            return self

        async def _agenerate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> Any:
            nonlocal clock_now
            events.append(f"{self.role}:called")
            if self.role == stage:
                return await block(self.role)
            if interruption == "inspection-deadline" and self.role == "main":
                clock_now = 1002.0  # Two seconds of agent work before inspecting its tool result.
            return self._generate(messages, stop, run_manager, **kwargs)

        def with_structured_output(self, _schema: Any, **_kwargs: Any) -> Any:
            async def shape(_messages: Any) -> Any:
                return await block(self.role)

            return RunnableLambda(shape)

    def create_model(_parameters: Any, reasoning_config: Any = None) -> Any:
        assert reasoning_config is None
        role = "main" if not clients else "inspection" if stage == "inspection" else "shaping"
        client = TrackedClient(role)
        clients[role] = client
        response = (
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "type": "tool_call",
                        "id": "interrupted-file-call",
                        "name": "read_file" if stage == "inspection" else "blocked_lookup",
                        "args": {"file_path": str(sentinel_file)} if stage == "inspection" else {},
                    }
                ],
            )
            if stage in ("inspection", "tool") and role == "main"
            else AIMessage(content="Agent draft")
        )
        model = ScriptedModel(role=role, responses=[response])
        model.__dict__["_async_client"] = client
        return model

    if interruption == "inspection-deadline":
        from lightspeed_agentic import run_agent
        from lightspeed_agentic.inspection.inspector import inspect_tool_result

        # Control runner/inspection budget arithmetic without changing asyncio's real clock.
        def controlled_clock() -> float:
            return clock_now

        original_query = DeepAgentsProvider.query

        def observe_query(self: Any, options: Any) -> AsyncIterator[Any]:
            provider_deadlines.append(options.deadline)
            return original_query(self, options)

        def observe_inspection(*args: Any, **kwargs: Any) -> Any:
            inspection_deadlines.append(kwargs.get("deadline"))
            return inspect_tool_result(*args, **kwargs, monotonic=controlled_clock)

        monkeypatch.setattr(run_agent, "time", SimpleNamespace(monotonic=controlled_clock))
        monkeypatch.setattr(DeepAgentsProvider, "query", observe_query)
        monkeypatch.setattr(
            "lightspeed_agentic.inspection.inspector.inspect_tool_result", observe_inspection
        )

    monkeypatch.setattr(asyncio, "run", run_with_watchdog)
    monkeypatch.setattr(asyncio, "wait_for", interrupt_when_entered)
    monkeypatch.setattr(DeepAgentsProvider, "aclose", observe_provider_close)
    monkeypatch.setattr(EventLogger, "log", observe_event)
    with _scripted_batch(
        tmp_path,
        monkeypatch,
        kind="ExecutionResult",
        schema={"type": "object", "properties": {"summary": {"type": "string"}}},
        create_model=create_model,
        inspection_enabled=stage == "inspection",
    ) as (main, template, calls, termination_path):
        if interruption == "inspection-deadline":
            # The classifier's own expired budget is fail-closed, not an AgentTimeout Result.
            with pytest.raises(SystemExit) as exited:
                main()
            assert exited.value.code == 1
            assert termination_path.read_text() == "ToolResultSafetyInspectionFailed"
            assert calls == []
        elif interruption == "cancellation":
            with pytest.raises(asyncio.CancelledError):
                main()
            assert not termination_path.exists()
            assert calls == []
        else:
            assert main() is None
            assert not termination_path.exists()
            assert [operation for operation, _ in calls] == ["create", "status"]
            assert calls[0][1]["body"] == template
            status = calls[1][1]["body"]["status"]
            assert set(status) == {"conditions", "failureReason", "tokenUsage"}
            assert status["failureReason"].startswith(
                "Agent invocation exceeded timeout of 30s after "
            )
            assert status["tokenUsage"] == {"inputTokens": 0, "outputTokens": 0}
            assert [(c["type"], c["status"], c["reason"]) for c in status["conditions"]] == [
                ("Started", "True", "StepStarted"),
                ("Completed", "True", "AgentTimeout"),
            ]

    if interruption == "inspection-deadline":
        assert provider_deadlines == [1030.0]
        assert inspection_deadlines == [1030.0]
        assert classifier_budgets == [28.0]
    assert entered.is_set()
    assert events.count(interruption) == 1
    assert events.count(f"{stage}:blocked") == 1
    assert events.count(f"{stage}:unwound") == 1
    assert events.count("main:called") == 1  # No subsequent agent call after interruption.
    assert "event:result" not in events
    assert "event:tool_result" not in events
    assert set(clients) == ({"main"} if stage in ("main", "tool") else {"main", stage})
    assert all(client.close_count == 1 for client in clients.values())
    assert events[-1] == "provider:closed"
    assert "PRIVATE-INTERRUPTED-TOOL-OUTPUT" not in json.dumps(calls)
