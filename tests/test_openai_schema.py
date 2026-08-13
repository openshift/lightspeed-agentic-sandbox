"""Tests for OpenAI provider strict schema transform and input sanitization."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lightspeed_agentic.providers.openai import _make_strict, _RawJsonSchema


def test_adds_additional_properties_false():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = _make_strict(schema)
    assert result["additionalProperties"] is False


def test_sets_required_to_all_keys():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
    }
    result = _make_strict(schema)
    assert sorted(result["required"]) == ["age", "name"]


def test_adds_required_when_missing():
    schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }
    result = _make_strict(schema)
    assert result["required"] == ["x"]


def test_recurses_into_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "inner": {
                "type": "object",
                "properties": {
                    "val": {"type": "string"},
                },
            },
        },
    }
    result = _make_strict(schema)
    inner = result["properties"]["inner"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["val"]


def test_recurses_into_array_items():
    schema = {
        "type": "object",
        "properties": {
            "items_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                    },
                },
            },
        },
    }
    result = _make_strict(schema)
    items_obj = result["properties"]["items_list"]["items"]
    assert items_obj["additionalProperties"] is False
    assert items_obj["required"] == ["id"]


def test_recurses_into_anyof():
    schema = {
        "anyOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"type": "string"},
        ],
    }
    result = _make_strict(schema)
    assert result["anyOf"][0]["additionalProperties"] is False
    assert result["anyOf"][0]["required"] == ["a"]
    assert result["anyOf"][1] == {"type": "string"}


def test_converts_oneof_to_anyof():
    schema = {
        "oneOf": [
            {"type": "object", "properties": {"b": {"type": "integer"}}},
        ],
    }
    result = _make_strict(schema)
    assert "oneOf" not in result
    assert result["anyOf"][0]["additionalProperties"] is False


def test_oneof_preserves_existing_anyof():
    schema = {
        "anyOf": [
            {"type": "object", "properties": {"x": {"type": "string"}}},
        ],
        "oneOf": [
            {"type": "object", "properties": {"y": {"type": "integer"}}},
        ],
    }
    result = _make_strict(schema)
    assert "oneOf" not in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["additionalProperties"] is False
    assert result["anyOf"][1]["additionalProperties"] is False


def test_recurses_into_allof():
    schema = {
        "allOf": [
            {"type": "object", "properties": {"c": {"type": "boolean"}}},
        ],
    }
    result = _make_strict(schema)
    assert result["allOf"][0]["additionalProperties"] is False


def test_recurses_into_not():
    schema = {
        "not": {"type": "object", "properties": {"d": {"type": "string"}}},
    }
    result = _make_strict(schema)
    assert result["not"]["additionalProperties"] is False


def test_recurses_into_defs():
    schema = {
        "$defs": {
            "thing": {"type": "object", "properties": {"e": {"type": "string"}}},
        },
    }
    result = _make_strict(schema)
    assert result["$defs"]["thing"]["additionalProperties"] is False


def test_does_not_modify_original():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    }
    _make_strict(schema)
    assert "additionalProperties" not in schema


def test_non_object_passthrough():
    schema = {"type": "string"}
    result = _make_strict(schema)
    assert result == {"type": "string"}


def test_non_dict_passthrough():
    assert _make_strict("not a dict") == "not a dict"


@patch.dict("os.environ", {}, clear=True)
def test_strict_enabled_no_base_url():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    wrapper = _RawJsonSchema(schema)
    assert wrapper.is_strict_json_schema() is True
    assert wrapper.json_schema()["additionalProperties"] is False


@patch.dict("os.environ", {"OPENAI_BASE_URL": "https://api.openai.com/v1"})
def test_strict_enabled_for_explicit_openai_url():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    wrapper = _RawJsonSchema(schema)
    assert wrapper.is_strict_json_schema() is True
    assert wrapper.json_schema()["additionalProperties"] is False


@patch.dict("os.environ", {"OPENAI_BASE_URL": "http://vllm:8000/v1"})
def test_strict_disabled_for_custom_endpoint():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    wrapper = _RawJsonSchema(schema)
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


async def _empty_stream():
    return
    yield


def _run_openai_provider(cwd: str):
    """Run OpenAIProvider.query() with mocked SDK internals.

    Returns (events, mock_sandbox_agent_cls) so callers can inspect both the
    emitted events and the kwargs passed to SandboxAgent.
    """
    from lightspeed_agentic.providers.openai import OpenAIProvider
    from lightspeed_agentic.types import ProviderQueryOptions

    mock_result = MagicMock()
    mock_result.stream_events = _empty_stream
    mock_result.final_output = ""
    mock_result.context_wrapper.usage.input_tokens = 0
    mock_result.context_wrapper.usage.output_tokens = 0

    async def _collect():
        with (
            patch("agents.sandbox.SandboxAgent", return_value=MagicMock()) as mock_cls,
            patch("agents.Runner.run_streamed", return_value=mock_result),
            patch("agents.models.openai_responses.OpenAIResponsesModel"),
            patch("openai.AsyncOpenAI"),
        ):
            provider = OpenAIProvider()
            options = ProviderQueryOptions(
                prompt="test",
                system_prompt="you are a test agent",
                model="gpt-4.1-mini",
                max_turns=1,
                max_budget_usd=0.0,
                allowed_tools=[],
                cwd=cwd,
            )
            events = [e async for e in provider.query(options)]
            return events, mock_cls

    return _collect()


@pytest.mark.asyncio
async def test_skills_path_set_to_skills_agents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Skills capability must use skills_path='skills/.agents', not the SDK default."""
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    (tmp_path / "skills" / ".agents").mkdir(parents=True)

    _, mock_cls = await _run_openai_provider(str(tmp_path))

    capabilities = mock_cls.call_args.kwargs["capabilities"]

    from agents.sandbox.capabilities import Skills

    skills_caps = [c for c in capabilities if isinstance(c, Skills)]
    assert len(skills_caps) == 1
    assert skills_caps[0].skills_path == "skills/.agents"


@pytest.mark.asyncio
async def test_skills_capability_omitted_when_dir_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Skills capability must not be registered when skills/.agents doesn't exist."""
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert not (tmp_path / "skills" / ".agents").exists()

    _, mock_cls = await _run_openai_provider(str(tmp_path))

    events, _ = await _run_openai_provider(str(tmp_path))
    assert any(isinstance(e, ContentBlockStopEvent) for e in events)
    assert any(isinstance(e, ResultEvent) for e in events)
    
    capabilities = mock_cls.call_args.kwargs["capabilities"]

    from agents.sandbox.capabilities import Skills

    skills_caps = [c for c in capabilities if isinstance(c, Skills)]
    assert len(skills_caps) == 0

class TestExecCommandShellCoercion:
    """OLS-3257: model sends shell:bool instead of shell:string."""

    @pytest.fixture(autouse=True)
    def _init_openai(self):
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
    async def test_shell_coercion(self, shell_input, expected):
        from agents.sandbox.capabilities.tools.shell_tool import ExecCommandTool

        raw_input = json.dumps({"cmd": "echo hello", "shell": shell_input})
        captured: list = []
        original_run = ExecCommandTool.run

        async def mock_run(_self, args):
            captured.append(args.shell)
            return "ok"

        ExecCommandTool.run = mock_run
        try:
            tool = ExecCommandTool.__new__(ExecCommandTool)
            tool.args_model = ExecCommandTool.args_model
            await tool._invoke(None, raw_input)
            assert captured == [expected]
        finally:
            ExecCommandTool.run = original_run

