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

    from lightspeed_agentic.mcp import ResolvedMCPServer  # type: ignore[import-untyped]

    mcp_server = object()
    manager = MagicMock(active_servers=[mcp_server])
    manager.__aenter__ = AsyncMock(return_value=manager)
    manager.__aexit__ = AsyncMock(return_value=None)
    resolved_server = ResolvedMCPServer(name="test", url="http://mcp.test/mcp")

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
                return_value=[resolved_server],
            ),
        ):
            options = ProviderQueryOptions(
                prompt="test",
                system_prompt="you are a test agent",
                model="gpt-4.1-mini",
                max_turns=1,
                allowed_tools=[],
                cwd=str(tmp_path),
                mcp_servers=[resolved_server],
            )
            provider = OpenAIProvider()
            [event async for event in provider.query(options)]
            return mock_cls

    mock_cls = await collect()
    assert mock_cls.call_args.kwargs["mcp_servers"] == [mcp_server]
    assert mcp_server not in mock_cls.call_args.kwargs["capabilities"]


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
