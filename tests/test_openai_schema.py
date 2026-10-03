"""Tests for OpenAI provider configuration and manifest building."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_adds_additional_properties_false() -> None:
    from lightspeed_agentic.providers.openai import _make_strict  # type: ignore[import-untyped]

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = _make_strict(schema)
    assert result["additionalProperties"] is False


def test_sets_required_to_all_keys() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }
    result = _make_strict(schema)
    assert sorted(result["required"]) == ["age", "name"]


def test_adds_required_when_missing() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    result = _make_strict(schema)
    assert result["required"] == ["x"]


def test_recurses_into_nested_objects() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {
        "type": "object",
        "properties": {"inner": {"type": "object", "properties": {"val": {"type": "string"}}}},
    }
    result = _make_strict(schema)
    inner = result["properties"]["inner"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["val"]


def test_recurses_into_array_items() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {
        "type": "object",
        "properties": {
            "items_list": {
                "type": "array",
                "items": {"type": "object", "properties": {"id": {"type": "integer"}}},
            }
        },
    }
    result = _make_strict(schema)
    items_obj = result["properties"]["items_list"]["items"]
    assert items_obj["additionalProperties"] is False
    assert items_obj["required"] == ["id"]


def test_recurses_into_anyof() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {
        "anyOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"type": "string"},
        ]
    }
    result = _make_strict(schema)
    assert result["anyOf"][0]["additionalProperties"] is False
    assert result["anyOf"][0]["required"] == ["a"]
    assert result["anyOf"][1] == {"type": "string"}


def test_converts_oneof_to_anyof() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {"oneOf": [{"type": "object", "properties": {"b": {"type": "integer"}}}]}
    result = _make_strict(schema)
    assert "oneOf" not in result
    assert result["anyOf"][0]["additionalProperties"] is False


def test_oneof_preserves_existing_anyof() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {
        "anyOf": [{"type": "object", "properties": {"x": {"type": "string"}}}],
        "oneOf": [{"type": "object", "properties": {"y": {"type": "integer"}}}],
    }
    result = _make_strict(schema)
    assert "oneOf" not in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["additionalProperties"] is False
    assert result["anyOf"][1]["additionalProperties"] is False


def test_recurses_into_allof() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    result = _make_strict({"allOf": [{"type": "object", "properties": {"c": {"type": "boolean"}}}]})
    assert result["allOf"][0]["additionalProperties"] is False


def test_recurses_into_not() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    result = _make_strict({"not": {"type": "object", "properties": {"d": {"type": "string"}}}})
    assert result["not"]["additionalProperties"] is False


def test_recurses_into_defs() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    result = _make_strict(
        {"$defs": {"thing": {"type": "object", "properties": {"e": {"type": "string"}}}}}
    )
    assert result["$defs"]["thing"]["additionalProperties"] is False


def test_does_not_modify_original() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    _make_strict(schema)
    assert "additionalProperties" not in schema


def test_non_object_passthrough() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    assert _make_strict({"type": "string"}) == {"type": "string"}


def test_non_dict_passthrough() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema: Any = "not a dict"
    assert _make_strict(schema) == "not a dict"


def test_native_openai_schema_adds_strict_requirements() -> None:
    from lightspeed_agentic.providers.openai import _RawJsonSchema

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    wrapper = _RawJsonSchema(schema, is_native=True)
    assert wrapper.json_schema()["additionalProperties"] is False
    assert wrapper.is_strict_json_schema() is True
    assert "additionalProperties" not in schema


def test_custom_endpoint_keeps_schema_non_strict() -> None:
    from lightspeed_agentic.providers.openai import _RawJsonSchema

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    wrapper = _RawJsonSchema(schema, is_native=False)
    assert wrapper.is_strict_json_schema() is False
    assert "additionalProperties" not in wrapper.json_schema()


def test_openai_init_disables_sdk_tracing_without_verbose_content_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightspeed_agentic.providers import openai as provider

    monkeypatch.setattr(provider, "_openai_initialized", False)
    with (
        patch("agents.tracing.set_tracing_disabled") as set_tracing_disabled,
        patch("agents.enable_verbose_stdout_logging") as verbose_logging,
        patch.object(provider, "_patch_exec_command_args"),
    ):
        provider._ensure_openai_init()

    set_tracing_disabled.assert_called_once_with(True)
    verbose_logging.assert_not_called()


def test_build_manifest_parent_of_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifest root should be cwd's parent so exec_command reaches the full workspace."""
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    from lightspeed_agentic.providers.openai import _build_manifest

    manifest = _build_manifest(str(Path("/app/skills").parent))
    assert manifest.root == "/app"


def test_build_manifest_without_e2e_output_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    from lightspeed_agentic.providers.openai import _build_manifest

    manifest = _build_manifest("/app/skills")
    assert manifest.root == "/app/skills"
    assert manifest.extra_path_grants == ()


def test_build_manifest_grants_e2e_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("E2E_OUTPUT_DIR", str(tmp_path))
    from lightspeed_agentic.providers.openai import _build_manifest

    manifest = _build_manifest("/app/skills")
    assert len(manifest.extra_path_grants) == 1
    grant = manifest.extra_path_grants[0]
    assert grant.path == str(tmp_path.resolve())
    assert grant.read_only is False


def test_build_manifest_skips_e2e_output_dir_outside_temp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("E2E_OUTPUT_DIR", "/etc")
    from lightspeed_agentic.providers.openai import _build_manifest

    manifest = _build_manifest("/app/skills")
    assert manifest.extra_path_grants == ()


@pytest.mark.asyncio
async def test_openai_model_uses_shared_tls_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_context = object()
    from lightspeed_agentic.providers.openai import OpenAIProvider

    monkeypatch.setattr(OpenAIProvider, "_client", None)
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    import lightspeed_agentic.tls as tls  # type: ignore[import-untyped]

    monkeypatch.setattr(tls, "get_ssl_context", lambda: shared_context)

    with patch("openai.DefaultAsyncHttpxClient") as http_client:
        http_client.return_value = MagicMock()
        await _run_openai_provider(str(tmp_path))

    http_client.assert_called_once_with(verify=shared_context)


async def _empty_stream() -> AsyncIterator[None]:
    return
    yield


def _run_openai_provider(cwd: str) -> Any:
    """Run OpenAIProvider.query() with mocked SDK internals.

    Returns (events, mock_sandbox_agent_cls, mock_runner) so callers can inspect
    the emitted events, SandboxAgent kwargs, and run configuration.
    """
    from lightspeed_agentic.providers.openai import OpenAIProvider
    from lightspeed_agentic.types import ProviderQueryOptions  # type: ignore[import-untyped]

    mock_result = MagicMock()
    mock_result.stream_events = _empty_stream
    mock_result.final_output = ""
    mock_result.context_wrapper.usage.input_tokens = 0
    mock_result.context_wrapper.usage.output_tokens = 0

    async def _collect() -> tuple[list[Any], MagicMock, MagicMock]:
        with (
            patch("agents.Runner.run_streamed", return_value=mock_result) as mock_runner,
            patch("agents.sandbox.SandboxAgent", return_value=MagicMock()) as mock_cls,
            patch("agents.models.openai_responses.OpenAIResponsesModel"),
            patch("openai.AsyncOpenAI"),
        ):
            provider = OpenAIProvider()
            options = ProviderQueryOptions(
                prompt="test",
                system_prompt="you are a test agent",
                model="gpt-4.1-mini",
                max_turns=1,
                allowed_tools=[],
                cwd=cwd,
            )
            events = [e async for e in provider.query(options)]
            return events, mock_cls, mock_runner

    return _collect()


@pytest.mark.asyncio
async def test_tool_output_trimmer_uses_shared_limits(tmp_path: Path) -> None:
    _, _, mock_runner = await _run_openai_provider(str(tmp_path))

    from agents.extensions import ToolOutputTrimmer

    from lightspeed_agentic.types import (
        MAX_TOOL_RETURN_CHARS,
        TOOL_RETURN_PREVIEW_CHARS,
    )

    run_config = mock_runner.call_args.kwargs["run_config"]
    trimmer = run_config.call_model_input_filter

    assert isinstance(trimmer, ToolOutputTrimmer)
    assert trimmer.max_output_chars == MAX_TOOL_RETURN_CHARS
    assert trimmer.preview_chars == TOOL_RETURN_PREVIEW_CHARS


@pytest.mark.asyncio
async def test_mcp_servers_are_passed_to_sandbox_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MCP servers must use SandboxAgent's mcp_servers argument, not capabilities."""
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    from lightspeed_agentic.mcp import (  # type: ignore[import-untyped]
        AdmittedMCPProviderServer,
    )

    mcp_server = object()
    manager = MagicMock(active_servers=[mcp_server])
    manager.__aenter__ = AsyncMock(return_value=manager)
    manager.__aexit__ = AsyncMock(return_value=None)
    admitted_server = AdmittedMCPProviderServer(
        name="test",
        url="http://mcp.test/mcp",
        allowed_tool_names=("get_pod",),
    )

    from lightspeed_agentic.providers.openai import OpenAIProvider
    from lightspeed_agentic.types import ProviderQueryOptions

    mock_result = MagicMock()
    mock_result.stream_events = _empty_stream
    mock_result.final_output = ""
    mock_result.context_wrapper.usage.input_tokens = 0
    mock_result.context_wrapper.usage.output_tokens = 0

    async def collect() -> MagicMock:
        with (
            patch("agents.sandbox.SandboxAgent", return_value=MagicMock()) as mock_cls,
            patch("agents.Runner.run_streamed", return_value=mock_result),
            patch("agents.models.openai_responses.OpenAIResponsesModel"),
            patch("openai.AsyncOpenAI"),
            patch("agents.mcp.MCPServerManager", return_value=manager),
            patch(
                "lightspeed_agentic.mcp.to_openai_mcp_servers",
                return_value=[admitted_server],
            ),
        ):
            options = ProviderQueryOptions(
                prompt="test",
                system_prompt="you are a test agent",
                model="gpt-4.1-mini",
                max_turns=1,
                allowed_tools=[],
                cwd=str(tmp_path),
                mcp_servers=[admitted_server],
            )
            provider = OpenAIProvider()
            [event async for event in provider.query(options)]
            return mock_cls

    mock_cls = await collect()
    assert mock_cls.call_args.kwargs["mcp_servers"] == [mcp_server]
    assert mcp_server not in mock_cls.call_args.kwargs["capabilities"]


@pytest.mark.asyncio
async def test_admitted_mcp_conversion_failure_fails_query(tmp_path: Path) -> None:
    from lightspeed_agentic.mcp import AdmittedMCPProviderServer
    from lightspeed_agentic.providers.openai import OpenAIProvider
    from lightspeed_agentic.types import ProviderQueryOptions

    options = ProviderQueryOptions(
        prompt="test",
        system_prompt="system",
        model="gpt-4.1-mini",
        max_turns=1,
        allowed_tools=[],
        cwd=str(tmp_path),
        mcp_servers=[
            AdmittedMCPProviderServer(
                name="openshift",
                url="https://mcp.example/mcp",
                allowed_tool_names=("get_pod",),
            )
        ],
    )

    provider = OpenAIProvider()
    with (
        patch("agents.models.openai_responses.OpenAIResponsesModel"),
        patch("lightspeed_agentic.mcp.to_openai_mcp_servers", return_value=[]),
        pytest.raises(RuntimeError, match="conversion produced no servers"),
    ):
        [event async for event in provider.query(options)]


@pytest.mark.asyncio
async def test_missing_admitted_mcp_active_server_fails_query(tmp_path: Path) -> None:
    from lightspeed_agentic.mcp import AdmittedMCPProviderServer
    from lightspeed_agentic.providers.openai import OpenAIProvider
    from lightspeed_agentic.types import ProviderQueryOptions

    options = ProviderQueryOptions(
        prompt="test",
        system_prompt="system",
        model="gpt-4.1-mini",
        max_turns=1,
        allowed_tools=[],
        cwd=str(tmp_path),
        mcp_servers=[
            AdmittedMCPProviderServer(
                name="openshift",
                url="https://mcp.example/mcp",
                allowed_tool_names=("get_pod",),
            )
        ],
    )
    converted_server = object()
    manager = MagicMock(active_servers=[])
    manager.__aenter__ = AsyncMock(return_value=manager)
    manager.__aexit__ = AsyncMock(return_value=None)

    provider = OpenAIProvider()
    with (
        patch("agents.models.openai_responses.OpenAIResponsesModel"),
        patch("lightspeed_agentic.mcp.to_openai_mcp_servers", return_value=[converted_server]),
        patch("agents.mcp.MCPServerManager", return_value=manager),
        pytest.raises(RuntimeError, match="initialized 0 of 1 admitted servers"),
    ):
        [event async for event in provider.query(options)]

    manager.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_admitted_mcp_manager_initialization_failure_propagates(tmp_path: Path) -> None:
    from lightspeed_agentic.mcp import AdmittedMCPProviderServer
    from lightspeed_agentic.providers.openai import OpenAIProvider
    from lightspeed_agentic.types import ProviderQueryOptions

    options = ProviderQueryOptions(
        prompt="test",
        system_prompt="system",
        model="gpt-4.1-mini",
        max_turns=1,
        allowed_tools=[],
        cwd=str(tmp_path),
        mcp_servers=[
            AdmittedMCPProviderServer(
                name="openshift",
                url="https://mcp.example/mcp",
                allowed_tool_names=("get_pod",),
            )
        ],
    )
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(side_effect=ConnectionError("unavailable"))
    manager.__aexit__ = AsyncMock(return_value=None)

    provider = OpenAIProvider()
    with (
        patch("agents.models.openai_responses.OpenAIResponsesModel"),
        patch("lightspeed_agentic.mcp.to_openai_mcp_servers", return_value=[object()]),
        patch("agents.mcp.MCPServerManager", return_value=manager),
        pytest.raises(ConnectionError, match="unavailable"),
    ):
        [event async for event in provider.query(options)]

    manager.__aexit__.assert_not_awaited()


@pytest.mark.asyncio
async def test_skills_registered_when_skill_md_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Skills capability must be registered when a subdirectory contains SKILL.md."""
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    (tmp_path / "my-skill").mkdir()
    (tmp_path / "my-skill" / "SKILL.md").write_text("# skill")

    _, mock_cls, _ = await _run_openai_provider(str(tmp_path))

    capabilities = mock_cls.call_args.kwargs["capabilities"]

    from agents.sandbox.capabilities import Skills

    skills_caps = [c for c in capabilities if isinstance(c, Skills)]
    assert len(skills_caps) == 1
    assert skills_caps[0].skills_path == "skills/.agents"


@pytest.mark.asyncio
async def test_skills_capability_omitted_when_no_skill_md(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Skills capability must not be registered when no SKILL.md exists under cwd."""
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    _, mock_cls, _ = await _run_openai_provider(str(tmp_path))

    capabilities = mock_cls.call_args.kwargs["capabilities"]

    from agents.sandbox.capabilities import Skills

    skills_caps = [c for c in capabilities if isinstance(c, Skills)]
    assert len(skills_caps) == 0


class TestExecCommandShellCoercion:
    """OLS-3257: model sends shell:bool instead of shell:string."""

    @pytest.fixture(autouse=True)
    def _init_openai(self) -> None:
        from lightspeed_agentic.providers.openai import _ensure_openai_init

        _ensure_openai_init()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("shell_input", "expected"),
        [
            (True, None),
            (False, None),
            ("/bin/bash", "/bin/bash"),
            (None, None),
        ],
        ids=["bool-true", "bool-false", "string-path", "null"],
    )
    async def test_shell_coercion(self, shell_input: Any, expected: Any) -> None:
        from agents.sandbox.capabilities.tools.shell_tool import ExecCommandTool

        raw_input = json.dumps({"cmd": "echo hello", "shell": shell_input})
        captured: list[Any] = []
        original_run = ExecCommandTool.run

        async def mock_run(self: object, args: Any) -> str:  # noqa: ARG001
            captured.append(args.shell)
            return "ok"

        type.__setattr__(ExecCommandTool, "run", mock_run)
        try:
            tool = ExecCommandTool.__new__(ExecCommandTool)
            object.__setattr__(tool, "args_model", ExecCommandTool.args_model)
            await tool._invoke(None, raw_input)
            assert captured == [expected]
        finally:
            type.__setattr__(ExecCommandTool, "run", original_run)


@pytest.mark.asyncio
async def test_overlapping_custom_openai_runs_isolate_function_tool_hooks(
    span_exporter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from contextvars import ContextVar
    from types import SimpleNamespace

    from agents.tool import function_tool
    from opentelemetry.trace import StatusCode

    import lightspeed_agentic.function_tools as function_tools
    from lightspeed_agentic.audit import AuditLogger
    from lightspeed_agentic.providers.openai import OpenAIProvider
    from lightspeed_agentic.types import ProviderQueryOptions

    monkeypatch.setenv("OPENAI_BASE_URL", "http://offline.test/v1")
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    started = {name: asyncio.Event() for name in ("failure", "success")}
    release = {name: asyncio.Event() for name in ("failure", "success")}
    tool_outputs: dict[str, Any] = {}
    current_run: ContextVar[str] = ContextVar("current_openai_run")

    async def controlled_read(query: str) -> dict[str, str]:
        """Wait for both provider queries before returning their distinct outcomes."""
        run = current_run.get()
        if run in started:
            started[run].set()
            await release[run].wait()
        if run == "failure":
            raise ValueError("private controlled tool failure")
        return {"run": run, "query": query}

    async def format_failure(context: Any, _error: Exception) -> str:
        return f"handled:{context.tool_call_id}"

    shared_tool = function_tool(
        name_override="read_file",
        description_override="Read the controlled test value.",
        failure_error_function=format_failure,
        strict_mode=False,
    )(controlled_read)
    monkeypatch.setattr(function_tools, "read_file", shared_tool)

    class _RunnerResult:
        def __init__(self, agent: Any, hooks: Any, query: str) -> None:
            self.agent = agent
            self.hooks = hooks
            self.query = query
            self.final_output = "completed"
            self.context_wrapper = SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=0,
                    output_tokens=0,
                    output_tokens_details=None,
                )
            )

        async def stream_events(self) -> AsyncIterator[Any]:
            tool = next(tool for tool in self.agent.tools if tool.name == "read_file")
            raw_arguments = '{"query":"shared-input"}'
            context = SimpleNamespace(
                tool_name=tool.name,
                tool_call_id=f"call-{self.query}",
                tool_arguments=raw_arguments,
                run_config=None,
            )
            await self.hooks.on_tool_start(context, self.agent, tool)
            invoke = tool.on_invoke_tool
            token = current_run.set(self.query)
            try:
                tool_outputs[self.query] = await invoke(context, raw_arguments)
            finally:
                current_run.reset(token)
            await self.hooks.on_tool_end(context, self.agent, tool, tool_outputs[self.query])
            if False:
                yield None

    def run_streamed(agent: Any, prompt: str, **kwargs: Any) -> _RunnerResult:
        return _RunnerResult(agent, kwargs["hooks"], prompt)

    async def collect(query: str) -> None:
        audit = AuditLogger(
            phase="analysis",
            model="overlapping-openai-test",
            provider="openai",
            agenticrun_uid=f"run-{query}",
        )
        options = ProviderQueryOptions(
            prompt=query,
            system_prompt="You are an offline test agent.",
            model="overlapping-openai-test",
            max_turns=1,
            allowed_tools=[],
            cwd=str(tmp_path),
            audit_logger=audit,
        )
        async for _event in OpenAIProvider().query(options):
            pass

    with (
        patch(
            "agents.sandbox.SandboxAgent", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)
        ),
        patch("agents.Runner.run_streamed", side_effect=run_streamed),
        patch("agents.models.openai_chatcompletions.OpenAIChatCompletionsModel"),
        patch("openai.AsyncOpenAI"),
    ):
        failure_query = asyncio.create_task(collect("failure"))
        success_query = asyncio.create_task(collect("success"))
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started.values())),
            timeout=5,
        )
        release["failure"].set()
        try:
            await failure_query
        finally:
            release["success"].set()
            await success_query

    assert tool_outputs["failure"] == "handled:call-failure"
    assert tool_outputs["success"] == {"run": "success", "query": "shared-input"}
    tool_spans = [
        span for span in span_exporter.get_finished_spans() if span.name == "execute_tool read_file"
    ]
    assert len(tool_spans) == 2
    spans_by_call_id = {span.attributes["gen_ai.tool.call.id"]: span for span in tool_spans}
    failure_span = spans_by_call_id["call-failure"]
    failure_attributes = dict(failure_span.attributes)
    assert failure_span.status.status_code == StatusCode.ERROR
    assert failure_attributes["error.type"] == "ValueError"
    assert "gen_ai.tool.call.result" not in failure_attributes
    assert "private controlled tool failure" not in str(failure_attributes)

    success_span = spans_by_call_id["call-success"]
    success_attributes = dict(success_span.attributes)
    assert success_span.status.status_code == StatusCode.UNSET
    assert json.loads(success_attributes["gen_ai.tool.call.result"]) == {
        "run": "success",
        "query": "shared-input",
    }

    future_context = SimpleNamespace(
        tool_name="read_file",
        tool_call_id="call-future",
        tool_arguments='{"query":"future"}',
        run_config=None,
    )
    token = current_run.set("future")
    try:
        future_result = await shared_tool.on_invoke_tool(
            future_context,
            '{"query":"future"}',
        )
    finally:
        current_run.reset(token)
    assert future_result == {"run": "future", "query": "future"}
    assert (
        len(
            [
                span
                for span in span_exporter.get_finished_spans()
                if span.name == "execute_tool read_file"
            ]
        )
        == 2
    )
