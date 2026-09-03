"""Tests for DeepAgents provider."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightspeed_agentic.mcp import ResolvedMCPHeader, ResolvedMCPServer
from lightspeed_agentic.providers.deepagents import TOOL_INPUT_MAX_CHARS, TOOL_OUTPUT_MAX_CHARS
from lightspeed_agentic.types import (
    ContentBlockStopEvent,
    ProviderQueryOptions,
    ResultEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)

_TEST_WORKSPACE = "/workspace"


def _base_options(**overrides: Any) -> ProviderQueryOptions:
    defaults = {
        "prompt": "hello",
        "system_prompt": "you are helpful",
        "model": "claude-sonnet-4-6",
        "max_turns": 10,
        "max_budget_usd": 1.0,
        "allowed_tools": ["Bash", "Read"],
        "cwd": _TEST_WORKSPACE,
    }
    defaults.update(overrides)
    return ProviderQueryOptions(**defaults)


def _mock_deepagents_modules(
    mock_create: MagicMock,
    mock_backend: MagicMock,
    *,
    mcp_client_cls: MagicMock | None = None,
) -> dict[str, Any]:
    mock_tool_strategy = MagicMock(side_effect=lambda schema, **_kw: schema)
    mock_provider_strategy = MagicMock(side_effect=lambda schema, **_kw: schema)
    mock_structured_output = MagicMock(
        ToolStrategy=mock_tool_strategy,
        ProviderStrategy=mock_provider_strategy,
    )
    mock_agents = MagicMock(structured_output=mock_structured_output)
    mock_langchain = MagicMock(agents=mock_agents)

    modules: dict[str, Any] = {
        "deepagents": MagicMock(create_deep_agent=mock_create),
        "deepagents.backends": MagicMock(LocalShellBackend=MagicMock(return_value=mock_backend)),
        "langchain": mock_langchain,
        "langchain.agents": mock_agents,
        "langchain.agents.structured_output": mock_structured_output,
        "langchain_anthropic": MagicMock(),
        "langchain_core": MagicMock(),
        "langchain_core.messages": MagicMock(),
    }
    if mcp_client_cls is not None:
        modules["langchain_mcp_adapters"] = MagicMock()
        modules["langchain_mcp_adapters.client"] = MagicMock(MultiServerMCPClient=mcp_client_cls)
    return modules


def _resolve_model_patch() -> Any:
    return patch(
        "lightspeed_agentic.providers.deepagents._resolve_model",
        return_value=MagicMock(),
    )


async def _collect_events(
    provider: Any,
    options: ProviderQueryOptions,
) -> list[Any]:
    events = []
    async for event in provider.query(options):
        events.append(event)
    return events


@contextmanager
def _deepagents_provider(
    mock_create: MagicMock,
    mock_backend: MagicMock,
    *,
    mcp_client_cls: MagicMock | None = None,
) -> Iterator[Any]:
    import importlib

    import lightspeed_agentic.providers.deepagents as mod

    with (
        patch.dict(
            sys.modules,
            _mock_deepagents_modules(
                mock_create,
                mock_backend,
                mcp_client_cls=mcp_client_cls,
            ),
        ),
        _resolve_model_patch(),
    ):
        importlib.reload(mod)
        yield mod.DeepAgentsProvider()


class TestResolveModel:
    """Test _resolve_model() returns correct ChatModel class per env."""

    def test_direct_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        mock_chat_anthropic = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatAnthropic = mock_chat_anthropic

        with patch.dict(sys.modules, {"langchain_anthropic": mock_module}):
            from lightspeed_agentic.providers.deepagents import _resolve_model

            _resolve_model("claude-sonnet-4-6", reasoning_config=None)

        mock_chat_anthropic.assert_called_once()
        call_kwargs = mock_chat_anthropic.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert "thinking" not in call_kwargs

    def test_direct_anthropic_with_thinking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        mock_chat_anthropic = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatAnthropic = mock_chat_anthropic

        with patch.dict(sys.modules, {"langchain_anthropic": mock_module}):
            from lightspeed_agentic.providers.deepagents import _resolve_model

            _resolve_model(
                "claude-opus-4-8",
                reasoning_config={"thinking": {"type": "adaptive"}},
            )

        call_kwargs = mock_chat_anthropic.call_args[1]
        assert call_kwargs["thinking"] == {"type": "adaptive"}

    def test_vertex_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
        monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "my-project")
        monkeypatch.setenv("CLOUD_ML_REGION", "us-east5")
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        mock_vertex = MagicMock()
        mock_garden_module = MagicMock()
        mock_garden_module.ChatAnthropicVertex = mock_vertex

        with patch.dict(
            sys.modules,
            {
                "langchain_google_vertexai": MagicMock(),
                "langchain_google_vertexai.model_garden": mock_garden_module,
            },
        ):
            from lightspeed_agentic.providers.deepagents import _resolve_model

            _resolve_model("claude-sonnet-4-6", reasoning_config=None)

        mock_vertex.assert_called_once()
        call_kwargs = mock_vertex.call_args[1]
        assert call_kwargs["model_name"] == "claude-sonnet-4-6"
        assert call_kwargs["project"] == "my-project"
        assert call_kwargs["location"] == "us-east5"

    def test_bedrock_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        mock_bedrock = MagicMock()
        mock_aws_module = MagicMock()
        mock_aws_module.ChatAnthropicBedrock = mock_bedrock

        with patch.dict(sys.modules, {"langchain_aws": mock_aws_module}):
            from lightspeed_agentic.providers.deepagents import _resolve_model

            _resolve_model("claude-sonnet-4-6", reasoning_config=None)

        mock_bedrock.assert_called_once()
        call_kwargs = mock_bedrock.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert call_kwargs["region_name"] == "us-east-1"


class TestJsonSchemaToPydantic:
    """Test _json_schema_to_pydantic() conversion."""

    def test_simple_object_schema(self) -> None:
        from lightspeed_agentic.providers.deepagents import _json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name"],
        }
        model = _json_schema_to_pydantic(schema)
        instance = model(name="test", count=5)
        assert instance.name == "test"
        assert instance.count == 5

    def test_enum_field(self) -> None:
        from lightspeed_agentic.providers.deepagents import _json_schema_to_pydantic

        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ok", "error"]},
            },
            "required": ["status"],
        }
        model = _json_schema_to_pydantic(schema)
        instance = model(status="ok")
        assert instance.status == "ok"

    def test_missing_properties_raises(self) -> None:
        from lightspeed_agentic.providers.deepagents import _json_schema_to_pydantic

        with pytest.raises(ValueError, match="missing 'properties'"):
            _json_schema_to_pydantic({"type": "object"})


class TestEventMapping:
    """Test query() event mapping from deepagents stream to ProviderEvent."""

    @pytest.mark.asyncio
    async def test_text_and_result_events(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Text message yields TextDeltaEvent; stream end yields ResultEvent."""
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        mock_ai_message = MagicMock()
        mock_ai_message.type = "ai"
        mock_ai_message.content = "Hello world"
        mock_ai_message.tool_calls = []
        mock_ai_message.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

        content_block = MagicMock()
        content_block.type = "text"
        content_block.text = "Hello world"
        mock_ai_message.content_blocks = [content_block]

        async def mock_astream(
            *_args: Any, **_kwargs: Any
        ) -> AsyncIterator[tuple[Any, dict[str, Any]]]:
            yield (mock_ai_message, {"langgraph_node": "agent"})

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream
        mock_create = MagicMock(return_value=mock_agent)
        with _deepagents_provider(mock_create, MagicMock()) as provider:
            events = await _collect_events(provider, _base_options())

        assert any(isinstance(e, TextDeltaEvent) for e in events)
        result_events = [e for e in events if isinstance(e, ResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].text == "Hello world"
        assert result_events[0].input_tokens == 10
        assert result_events[0].output_tokens == 5
        assert result_events[0].cost_usd == 0

    @pytest.mark.asyncio
    async def test_text_accumulates_across_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Incremental AIMessage chunks accumulate into the final ResultEvent text."""
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        def make_chunk(text: str) -> MagicMock:
            msg = MagicMock()
            msg.type = "ai"
            msg.content = text
            msg.tool_calls = []
            msg.usage_metadata = None
            block = MagicMock()
            block.type = "text"
            block.text = text
            msg.content_blocks = [block]
            return msg

        async def mock_astream(
            *_args: Any, **_kwargs: Any
        ) -> AsyncIterator[tuple[Any, dict[str, Any]]]:
            yield (make_chunk("Hello "), {"langgraph_node": "agent"})
            yield (make_chunk("world"), {"langgraph_node": "agent"})

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream
        with _deepagents_provider(MagicMock(return_value=mock_agent), MagicMock()) as provider:
            events = await _collect_events(provider, _base_options())
        result_events = [e for e in events if isinstance(e, ResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].text == "Hello world"

    @pytest.mark.asyncio
    async def test_plain_content_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Messages without content_blocks fall back to plain msg.content."""
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        mock_ai_message = MagicMock()
        mock_ai_message.type = "ai"
        mock_ai_message.content = "Plain fallback text"
        mock_ai_message.tool_calls = []
        mock_ai_message.usage_metadata = {"input_tokens": 3, "output_tokens": 2}
        mock_ai_message.content_blocks = []

        async def mock_astream(
            *_args: Any, **_kwargs: Any
        ) -> AsyncIterator[tuple[Any, dict[str, Any]]]:
            yield (mock_ai_message, {"langgraph_node": "agent"})

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream
        with _deepagents_provider(MagicMock(return_value=mock_agent), MagicMock()) as provider:
            events = await _collect_events(provider, _base_options())
        text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
        result_events = [e for e in events if isinstance(e, ResultEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "Plain fallback text"
        assert result_events[0].text == "Plain fallback text"

    @pytest.mark.asyncio
    async def test_thinking_events(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reasoning content blocks yield ThinkingDeltaEvent + ContentBlockStopEvent."""
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        mock_ai_message = MagicMock()
        mock_ai_message.type = "ai"
        mock_ai_message.content = "Final answer"
        mock_ai_message.tool_calls = []
        mock_ai_message.usage_metadata = {"input_tokens": 50, "output_tokens": 20}

        thinking_block = MagicMock()
        thinking_block.type = "reasoning"
        thinking_block.reasoning = "Let me think about this..."

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Final answer"

        mock_ai_message.content_blocks = [thinking_block, text_block]

        async def mock_astream(
            *_args: Any, **_kwargs: Any
        ) -> AsyncIterator[tuple[Any, dict[str, Any]]]:
            yield (mock_ai_message, {"langgraph_node": "agent"})

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream
        with _deepagents_provider(MagicMock(return_value=mock_agent), MagicMock()) as provider:
            events = await _collect_events(
                provider,
                _base_options(reasoning_config={"thinking": {"type": "adaptive"}}),
            )

        thinking_events = [e for e in events if isinstance(e, ThinkingDeltaEvent)]
        stop_events = [e for e in events if isinstance(e, ContentBlockStopEvent)]
        assert len(thinking_events) >= 1
        assert thinking_events[0].thinking == "Let me think about this..."
        assert len(stop_events) >= 1

    @pytest.mark.asyncio
    async def test_tool_call_and_result_events(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tool calls yield ToolCallEvent; ToolMessages yield ToolResultEvent."""
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        mock_ai_tool_msg = MagicMock()
        mock_ai_tool_msg.type = "ai"
        mock_ai_tool_msg.content = ""
        mock_ai_tool_msg.tool_calls = [
            {"name": "execute", "args": {"command": "ls -la"}, "id": "tc_1"}
        ]
        mock_ai_tool_msg.usage_metadata = {"input_tokens": 20, "output_tokens": 10}
        mock_ai_tool_msg.content_blocks = []

        mock_tool_result = MagicMock()
        mock_tool_result.type = "tool"
        mock_tool_result.content = "file1.py\nfile2.py"
        mock_tool_result.tool_call_id = "tc_1"

        mock_ai_final = MagicMock()
        mock_ai_final.type = "ai"
        mock_ai_final.content = "I found 2 files."
        mock_ai_final.tool_calls = []
        mock_ai_final.usage_metadata = {"input_tokens": 30, "output_tokens": 15}
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I found 2 files."
        mock_ai_final.content_blocks = [text_block]

        async def mock_astream(
            *_args: Any, **_kwargs: Any
        ) -> AsyncIterator[tuple[Any, dict[str, Any]]]:
            yield (mock_ai_tool_msg, {"langgraph_node": "agent"})
            yield (mock_tool_result, {"langgraph_node": "tools"})
            yield (mock_ai_final, {"langgraph_node": "agent"})

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream
        with _deepagents_provider(MagicMock(return_value=mock_agent), MagicMock()) as provider:
            events = await _collect_events(provider, _base_options())

        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "execute"
        assert len(tool_results) == 1
        assert "file1.py" in tool_results[0].output

    @pytest.mark.asyncio
    async def test_tool_io_truncation_at_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tool call input and tool result output are truncated at max char limits."""
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        long_arg = "x" * (TOOL_INPUT_MAX_CHARS + 50)
        long_output = "y" * (TOOL_OUTPUT_MAX_CHARS + 50)

        mock_ai_tool_msg = MagicMock()
        mock_ai_tool_msg.type = "ai"
        mock_ai_tool_msg.content = ""
        mock_ai_tool_msg.tool_calls = [
            {"name": "execute", "args": {"command": long_arg}, "id": "tc_long"}
        ]
        mock_ai_tool_msg.usage_metadata = None
        mock_ai_tool_msg.content_blocks = []

        mock_tool_result = MagicMock()
        mock_tool_result.type = "tool"
        mock_tool_result.content = long_output
        mock_tool_result.tool_call_id = "tc_long"

        async def mock_astream(
            *_args: Any, **_kwargs: Any
        ) -> AsyncIterator[tuple[Any, dict[str, Any]]]:
            yield (mock_ai_tool_msg, {"langgraph_node": "agent"})
            yield (mock_tool_result, {"langgraph_node": "tools"})

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream
        with _deepagents_provider(MagicMock(return_value=mock_agent), MagicMock()) as provider:
            events = await _collect_events(provider, _base_options())

        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_calls[0].input) == TOOL_INPUT_MAX_CHARS
        assert len(tool_results[0].output) == TOOL_OUTPUT_MAX_CHARS

    @pytest.mark.asyncio
    async def test_structured_output_two_phase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When output_schema is set, agent runs then shape pass produces Result JSON."""
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        mock_ai_message = MagicMock()
        mock_ai_message.type = "ai"
        mock_ai_message.content = "agent answer"
        mock_ai_message.tool_calls = []
        mock_ai_message.usage_metadata = {"input_tokens": 3, "output_tokens": 4}
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "agent answer"
        mock_ai_message.content_blocks = [text_block]

        async def mock_astream(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
            yield (mock_ai_message, {"langgraph_node": "agent"})

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream
        mock_create = MagicMock(return_value=mock_agent)

        mock_raw = MagicMock(usage_metadata={"input_tokens": 5, "output_tokens": 6})
        mock_structured_runnable = MagicMock()
        mock_structured_runnable.ainvoke = AsyncMock(
            return_value={"parsed": {"status": "ok"}, "raw": mock_raw}
        )
        mock_format_model = MagicMock()
        mock_format_model.with_structured_output = MagicMock(return_value=mock_structured_runnable)
        mock_agent_model = MagicMock()

        def resolve_model_side_effect(
            _model: str, reasoning_config: dict[str, Any] | None = None
        ) -> MagicMock:
            if reasoning_config and reasoning_config.get("thinking"):
                return mock_agent_model
            return mock_format_model

        output_schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }

        with patch.dict(sys.modules, _mock_deepagents_modules(mock_create, MagicMock())):
            import importlib

            import lightspeed_agentic.providers.deepagents as mod

            importlib.reload(mod)
            with patch.object(mod, "_resolve_model", side_effect=resolve_model_side_effect):
                provider = mod.DeepAgentsProvider()
                events = await _collect_events(
                    provider,
                    _base_options(output_schema=output_schema),
                )

        assert "response_format" not in mock_create.call_args[1]
        mock_format_model.with_structured_output.assert_called_once()
        result_events = [e for e in events if isinstance(e, ResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].text == '{"status": "ok"}'
        assert result_events[0].input_tokens == 8
        assert result_events[0].output_tokens == 10

    @pytest.mark.asyncio
    async def test_two_phase_structured_output_with_thinking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Thinking + schema: no response_format on agent; shape via with_structured_output."""
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        mock_ai_message = MagicMock()
        mock_ai_message.type = "ai"
        mock_ai_message.content = "agent answer"
        mock_ai_message.tool_calls = []
        mock_ai_message.usage_metadata = {"input_tokens": 3, "output_tokens": 4}
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "agent answer"
        mock_ai_message.content_blocks = [text_block]

        async def mock_astream(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
            yield (mock_ai_message, {"langgraph_node": "agent"})

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream
        mock_create = MagicMock(return_value=mock_agent)

        mock_raw = MagicMock(usage_metadata={"input_tokens": 5, "output_tokens": 6})
        mock_structured_runnable = MagicMock()
        mock_structured_runnable.ainvoke = AsyncMock(
            return_value={"parsed": {"status": "ok"}, "raw": mock_raw}
        )
        mock_format_model = MagicMock()
        mock_format_model.with_structured_output = MagicMock(return_value=mock_structured_runnable)
        mock_agent_model = MagicMock()

        def resolve_model_side_effect(
            _model: str, reasoning_config: dict[str, Any] | None = None
        ) -> MagicMock:
            if reasoning_config and reasoning_config.get("thinking"):
                return mock_agent_model
            return mock_format_model

        output_schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }

        with patch.dict(sys.modules, _mock_deepagents_modules(mock_create, MagicMock())):
            import importlib

            import lightspeed_agentic.providers.deepagents as mod

            importlib.reload(mod)
            with patch.object(mod, "_resolve_model", side_effect=resolve_model_side_effect):
                provider = mod.DeepAgentsProvider()
                events = await _collect_events(
                    provider,
                    _base_options(
                        output_schema=output_schema,
                        reasoning_config={"thinking": {"type": "enabled", "budget_tokens": 1024}},
                    ),
                )

        assert "response_format" not in mock_create.call_args[1]
        mock_format_model.with_structured_output.assert_called_once()
        call_kwargs = mock_format_model.with_structured_output.call_args[1]
        assert call_kwargs["method"] == "json_schema"
        assert call_kwargs["include_raw"] is True

        result_events = [e for e in events if isinstance(e, ResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].text == '{"status": "ok"}'
        assert result_events[0].input_tokens == 8
        assert result_events[0].output_tokens == 10

    def test_structured_output_method_json_schema_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)
        from lightspeed_agentic.providers.deepagents import _structured_output_method

        assert _structured_output_method() == "json_schema"

    def test_structured_output_method_function_calling_on_bedrock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
        from lightspeed_agentic.providers.deepagents import _structured_output_method

        assert _structured_output_method() == "function_calling"

    def test_conflicting_vertex_and_bedrock_flags_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
        monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
        from lightspeed_agentic.providers.deepagents import (
            _anthropic_backend,
            _structured_output_method,
        )

        with pytest.raises(ValueError, match="cannot both be set"):
            _anthropic_backend()
        with pytest.raises(ValueError, match="cannot both be set"):
            _structured_output_method()

    @pytest.mark.asyncio
    async def test_shape_pass_uses_function_calling_on_bedrock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")

        mock_ai_message = MagicMock()
        mock_ai_message.type = "ai"
        mock_ai_message.content = "agent answer"
        mock_ai_message.tool_calls = []
        mock_ai_message.usage_metadata = {"input_tokens": 1, "output_tokens": 2}
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "agent answer"
        mock_ai_message.content_blocks = [text_block]

        async def mock_astream(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
            yield (mock_ai_message, {"langgraph_node": "agent"})

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream
        mock_create = MagicMock(return_value=mock_agent)

        mock_raw = MagicMock(usage_metadata={"input_tokens": 3, "output_tokens": 4})
        mock_structured_runnable = MagicMock()
        mock_structured_runnable.ainvoke = AsyncMock(
            return_value={"parsed": {"status": "ok"}, "raw": mock_raw}
        )
        mock_format_model = MagicMock()
        mock_format_model.with_structured_output = MagicMock(return_value=mock_structured_runnable)
        mock_agent_model = MagicMock()

        def resolve_model_side_effect(
            _model: str, reasoning_config: dict[str, Any] | None = None
        ) -> MagicMock:
            if reasoning_config and reasoning_config.get("thinking"):
                return mock_agent_model
            return mock_format_model

        output_schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }

        with patch.dict(sys.modules, _mock_deepagents_modules(mock_create, MagicMock())):
            import importlib

            import lightspeed_agentic.providers.deepagents as mod

            importlib.reload(mod)
            with patch.object(mod, "_resolve_model", side_effect=resolve_model_side_effect):
                provider = mod.DeepAgentsProvider()
                await _collect_events(provider, _base_options(output_schema=output_schema))

        call_kwargs = mock_format_model.with_structured_output.call_args[1]
        assert call_kwargs["method"] == "function_calling"
        assert call_kwargs["include_raw"] is True

    @pytest.mark.asyncio
    async def test_recursion_limit_passed_to_astream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """max_turns is forwarded to astream config as recursion_limit."""
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        mock_ai_message = MagicMock()
        mock_ai_message.type = "ai"
        mock_ai_message.content = "done"
        mock_ai_message.tool_calls = []
        mock_ai_message.usage_metadata = None
        mock_ai_message.content_blocks = []

        captured_config: dict[str, Any] = {}

        async def mock_astream(
            *_args: Any, **kwargs: Any
        ) -> AsyncIterator[tuple[Any, dict[str, Any]]]:
            captured_config.update(kwargs.get("config", {}))
            yield (mock_ai_message, {"langgraph_node": "agent"})

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream
        with _deepagents_provider(MagicMock(return_value=mock_agent), MagicMock()) as provider:
            await _collect_events(provider, _base_options(max_turns=25))
        assert captured_config["recursion_limit"] == 25

    @pytest.mark.asyncio
    async def test_mcp_tools_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MCP servers are passed to MultiServerMCPClient and tools merged into agent."""
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        mock_ai_message = MagicMock()
        mock_ai_message.type = "ai"
        mock_ai_message.content = "done"
        mock_ai_message.tool_calls = []
        mock_ai_message.usage_metadata = {"input_tokens": 1, "output_tokens": 1}
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "done"
        mock_ai_message.content_blocks = [text_block]

        async def mock_astream(
            *_args: Any, **_kwargs: Any
        ) -> AsyncIterator[tuple[Any, dict[str, Any]]]:
            yield (mock_ai_message, {"langgraph_node": "agent"})

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream
        mock_create = MagicMock(return_value=mock_agent)
        mock_mcp_tool = MagicMock()
        mock_mcp_client = MagicMock()
        mock_mcp_client.get_tools = AsyncMock(return_value=[mock_mcp_tool])
        mock_mcp_client_cls = MagicMock(return_value=mock_mcp_client)

        mcp_server = ResolvedMCPServer(
            name="test-server",
            url="http://mcp.example.com",
            timeout=30,
            headers=[ResolvedMCPHeader(name="Authorization", value="Bearer token")],
        )

        with _deepagents_provider(
            mock_create,
            MagicMock(),
            mcp_client_cls=mock_mcp_client_cls,
        ) as provider:
            await _collect_events(provider, _base_options(mcp_servers=[mcp_server]))

        mock_mcp_client_cls.assert_called_once()
        server_config = mock_mcp_client_cls.call_args[0][0]
        assert "test-server" in server_config
        assert server_config["test-server"]["url"] == "http://mcp.example.com"
        assert server_config["test-server"]["headers"]["Authorization"] == "Bearer token"
        mock_mcp_client.get_tools.assert_awaited_once()
        mock_create.assert_called_once()
        create_kwargs = mock_create.call_args[1]
        assert create_kwargs["tools"] == [mock_mcp_tool]


class TestSkillsGating:
    """Test that skills= is only passed when SKILL.md files exist under cwd."""

    @pytest.mark.asyncio
    async def test_skills_passed_when_skill_md_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """skills= should be set when a SKILL.md exists under cwd."""
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        (tmp_path / "my-skill" / "SKILL.md").parent.mkdir(parents=True)
        (tmp_path / "my-skill" / "SKILL.md").write_text("# skill")

        mock_ai = MagicMock()
        mock_ai.type = "ai"
        mock_ai.content = "ok"
        mock_ai.tool_calls = []
        mock_ai.usage_metadata = None
        mock_ai.content_blocks = []

        async def mock_astream(
            *_args: Any, **_kwargs: Any
        ) -> AsyncIterator[tuple[Any, dict[str, Any]]]:
            yield (mock_ai, {"langgraph_node": "agent"})

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream
        mock_create = MagicMock(return_value=mock_agent)
        with _deepagents_provider(mock_create, MagicMock()) as provider:
            await _collect_events(provider, _base_options(cwd=str(tmp_path)))

        create_kwargs = mock_create.call_args[1]
        assert create_kwargs["skills"] == [str(tmp_path)]

    @pytest.mark.asyncio
    async def test_skills_omitted_when_no_skill_md(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """skills= must not be passed when no SKILL.md exists under cwd."""
        monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        assert not (tmp_path / "SKILL.md").exists()

        mock_ai = MagicMock()
        mock_ai.type = "ai"
        mock_ai.content = "ok"
        mock_ai.tool_calls = []
        mock_ai.usage_metadata = None
        mock_ai.content_blocks = []

        async def mock_astream(
            *_args: Any, **_kwargs: Any
        ) -> AsyncIterator[tuple[Any, dict[str, Any]]]:
            yield (mock_ai, {"langgraph_node": "agent"})

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream
        mock_create = MagicMock(return_value=mock_agent)
        with _deepagents_provider(mock_create, MagicMock()) as provider:
            await _collect_events(provider, _base_options(cwd=str(tmp_path)))

        create_kwargs = mock_create.call_args[1]
        assert "skills" not in create_kwargs
