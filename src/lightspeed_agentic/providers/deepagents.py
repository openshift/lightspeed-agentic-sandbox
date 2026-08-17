"""DeepAgents provider — wraps langchain-ai/deepagents for Anthropic model support.

Uses create_deep_agent() with LocalShellBackend for shell + filesystem access,
native skills loading, and v3 event streaming for event mapping.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

from lightspeed_agentic.types import (
    AgentProvider,
    ContentBlockStopEvent,
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    stringify,
)

logger = logging.getLogger(__name__)

TOOL_INPUT_MAX_CHARS = 10_000
TOOL_OUTPUT_MAX_CHARS = 10_000

_JSON_SCHEMA_TYPE_MAP: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _has_skills(cwd: str) -> bool:
    """Return True when at least one subdirectory of *cwd* contains a SKILL.md."""
    try:
        for entry in os.listdir(cwd):
            child = os.path.join(cwd, entry)
            if os.path.isdir(child) and os.path.isfile(os.path.join(child, "SKILL.md")):
                return True
    except OSError:
        pass
    return False


def _resolve_model(model: str, reasoning_config: dict[str, Any] | None = None) -> Any:
    """Build a LangChain chat model instance based on env vars set by config.py."""
    thinking = reasoning_config.get("thinking") if reasoning_config else None

    if os.environ.get("CLAUDE_CODE_USE_VERTEX") == "1":
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex

        kwargs: dict[str, Any] = {
            "model_name": model,
            "project": os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", ""),
            "location": os.environ.get("CLOUD_ML_REGION", "us-east5"),
        }
        if thinking:
            kwargs["thinking"] = thinking
        return ChatAnthropicVertex(**kwargs)

    if os.environ.get("CLAUDE_CODE_USE_BEDROCK") == "1":
        from langchain_aws import ChatAnthropicBedrock

        kwargs = {
            "model": model,
            "region_name": os.environ.get("AWS_REGION", "us-east-1"),
        }
        if thinking:
            kwargs["thinking"] = thinking
        return ChatAnthropicBedrock(**kwargs)

    from langchain_anthropic import ChatAnthropic

    kwargs = {"model": model}
    if thinking:
        kwargs["thinking"] = thinking
    return ChatAnthropic(**kwargs)


def _json_schema_to_pydantic(schema: dict[str, Any], name: str = "OutputModel") -> Any:
    """Convert a JSON schema dict to a dynamic Pydantic model."""
    import pydantic

    if "properties" not in schema:
        raise ValueError(f"Schema {name!r} missing 'properties'")

    props = schema["properties"]
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}

    for field_name, field_schema in props.items():
        field_type = _resolve_field_type(field_schema, field_name)
        if field_name in required:
            fields[field_name] = (field_type, ...)
        else:
            fields[field_name] = (field_type | None, None)

    return pydantic.create_model(name, **fields)


def _resolve_field_type(schema: dict[str, Any], name: str) -> Any:
    from typing import Literal

    json_type = schema.get("type", "string")

    if json_type == "object":
        return _json_schema_to_pydantic(schema, name.title().replace("_", ""))

    if json_type == "array":
        if "items" not in schema:
            raise ValueError(f"Array field {name!r} missing 'items'")
        item_type = _resolve_field_type(schema["items"], f"{name}_item")
        return list[item_type]  # type: ignore[valid-type]

    if "enum" in schema:
        return Literal[tuple(schema["enum"])]

    return _JSON_SCHEMA_TYPE_MAP.get(json_type, str)


def _process_ai_message(
    msg: Any,
) -> tuple[list[ProviderEvent], str, int, int]:
    """Map one AIMessage chunk to provider events and token deltas."""
    events: list[ProviderEvent] = []
    text_delta = ""
    input_tokens = 0
    output_tokens = 0

    for tc in msg.tool_calls or []:
        events.append(
            ToolCallEvent(
                name=tc.get("name", ""),
                input=json.dumps(tc.get("args", {}))[:TOOL_INPUT_MAX_CHARS],
                call_id=tc.get("id", ""),
            )
        )

    for block in getattr(msg, "content_blocks", []):
        btype = block["type"] if isinstance(block, dict) else getattr(block, "type", "")
        if btype == "reasoning":
            reasoning = (
                block.get("reasoning", "")
                if isinstance(block, dict)
                else getattr(block, "reasoning", "")
            )
            events.append(ThinkingDeltaEvent(thinking=reasoning))
            events.append(ContentBlockStopEvent())
        elif btype == "text":
            text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
            if text:
                events.append(TextDeltaEvent(text=text))
                text_delta += text

    if not getattr(msg, "content_blocks", None):
        content = msg.content if isinstance(msg.content, str) else stringify(msg.content)
        if content and not msg.tool_calls:
            events.append(TextDeltaEvent(text=content))
            text_delta += content

    usage = getattr(msg, "usage_metadata", None)
    if usage:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

    return events, text_delta, input_tokens, output_tokens


class DeepAgentsProvider(AgentProvider):
    @property
    def name(self) -> str:
        return "deepagents"

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        from deepagents import create_deep_agent
        from deepagents.backends import LocalShellBackend

        logger.debug(
            "Starting deepagents query model=%s cwd=%s max_turns=%s",
            options.model,
            options.cwd,
            options.max_turns,
        )

        chat_model = _resolve_model(options.model, options.reasoning_config)
        backend = LocalShellBackend(root_dir=options.cwd, inherit_env=True)

        agent_kwargs: dict[str, Any] = {
            "model": chat_model,
            "backend": backend,
            "system_prompt": options.system_prompt,
        }

        if _has_skills(options.cwd):
            agent_kwargs["skills"] = [options.cwd]

        if options.output_schema:
            from langchain.agents.structured_output import ToolStrategy

            schema = (
                _json_schema_to_pydantic(options.output_schema)
                if isinstance(options.output_schema, dict)
                else options.output_schema
            )
            agent_kwargs["response_format"] = ToolStrategy(schema=schema)

        mcp_tools: list[Any] = []
        if options.mcp_servers:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(
                {
                    server.name: {  # type: ignore[misc]
                        "transport": "http",
                        "url": server.url,
                        "headers": {h.name: h.value for h in server.headers},
                        "timeout": server.timeout,
                    }
                    for server in options.mcp_servers
                }
            )
            mcp_tools = await client.get_tools()

        if mcp_tools:
            agent_kwargs["tools"] = mcp_tools

        # allowed_tools is not forwarded: deepagents' LocalShellBackend exposes a broader
        # built-in tool set than DEFAULT_ALLOWED_TOOLS. Filtering is a follow-up.
        agent = create_deep_agent(**agent_kwargs)

        thread_id = f"ls-{uuid.uuid4().hex[:12]}"
        stream_config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": options.max_turns,
        }
        result_text = ""
        total_input_tokens = 0
        total_output_tokens = 0
        final_state: dict[str, Any] | None = None

        stream_modes: str | list[str] = (
            ["messages", "values"] if options.output_schema else "messages"
        )
        input_state = {"messages": [{"role": "user", "content": options.prompt}]}

        async for item in agent.astream(  # type: ignore[call-overload]
            input_state,
            config=stream_config,
            stream_mode=stream_modes,
        ):
            if options.output_schema:
                mode, chunk = item
                if mode == "values":
                    final_state = chunk
                    continue
                msg, _stream_metadata = chunk
            else:
                msg, _stream_metadata = item

            if msg.type in ("ai", "AIMessageChunk"):
                events, text_delta, in_tok, out_tok = _process_ai_message(msg)
                for event in events:
                    yield event
                result_text += text_delta
                total_input_tokens += in_tok
                total_output_tokens += out_tok

            elif msg.type in ("tool", "ToolMessageChunk"):
                yield ToolResultEvent(
                    output=stringify(msg.content)[:TOOL_OUTPUT_MAX_CHARS],
                    call_id=getattr(msg, "tool_call_id", ""),
                )

        if options.output_schema and final_state:
            structured = final_state.get("structured_response")
            if structured is not None:
                result_text = stringify(structured)

        yield ResultEvent(
            text=result_text,
            cost_usd=0,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )
