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

from lightspeed_agentic.skills import has_skills
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


class GuardedBackend:
    """Wraps a LocalShellBackend with pre- and post-execution guardrail checks.

    Delegates all BackendProtocol/SandboxBackendProtocol methods so the SDK's
    class-level attribute checks and ``isinstance`` gates work correctly.
    """

    def __init__(
        self,
        real_backend: Any,
        checker: Any,
        context: Any,
    ) -> None:
        self._backend = real_backend
        self._checker = checker
        self._context = context

    @property
    def id(self) -> str:
        return self._backend.id

    def execute(self, command: str, **kwargs: Any) -> Any:
        raise NotImplementedError("Use aexecute for async guardrail checks")

    async def aexecute(self, command: str, **kwargs: Any) -> Any:
        pre = await self._checker.check_tool_request(command, self._context)
        from lightspeed_agentic.guardrails.types import Verdict

        if pre.verdict == Verdict.BLOCK:
            from deepagents.backends.protocol import ExecuteResponse

            return ExecuteResponse(
                output=f"[GUARDRAIL] Command blocked: {pre.reason}",
                exit_code=1,
            )

        result = self._backend.execute(command, **kwargs)
        output = result.output if hasattr(result, "output") else str(result)

        post = await self._checker.check_tool_output(output, command, self._context)
        if post.verdict == Verdict.BLOCK:
            from deepagents.backends.protocol import ExecuteResponse

            return ExecuteResponse(
                output=f"[GUARDRAIL] Tool output blocked: {post.reason}",
                exit_code=1,
            )
        if post.verdict == Verdict.SANITIZE:
            if hasattr(result, "output"):
                result.output = post.sanitized_output
                return result
            return post.sanitized_output

        return result

    def read(self, *a: Any, **kw: Any) -> Any:
        return self._backend.read(*a, **kw)

    async def aread(self, *a: Any, **kw: Any) -> Any:
        return await self._backend.aread(*a, **kw)

    def write(self, *a: Any, **kw: Any) -> Any:
        return self._backend.write(*a, **kw)

    async def awrite(self, *a: Any, **kw: Any) -> Any:
        return await self._backend.awrite(*a, **kw)

    def edit(self, *a: Any, **kw: Any) -> Any:
        return self._backend.edit(*a, **kw)

    async def aedit(self, *a: Any, **kw: Any) -> Any:
        return await self._backend.aedit(*a, **kw)

    def delete(self, *a: Any, **kw: Any) -> Any:
        return self._backend.delete(*a, **kw)

    async def adelete(self, *a: Any, **kw: Any) -> Any:
        return await self._backend.adelete(*a, **kw)

    def ls(self, *a: Any, **kw: Any) -> Any:
        return self._backend.ls(*a, **kw)

    async def als(self, *a: Any, **kw: Any) -> Any:
        return await self._backend.als(*a, **kw)

    def glob(self, *a: Any, **kw: Any) -> Any:
        return self._backend.glob(*a, **kw)

    async def aglob(self, *a: Any, **kw: Any) -> Any:
        return await self._backend.aglob(*a, **kw)

    def grep(self, *a: Any, **kw: Any) -> Any:
        return self._backend.grep(*a, **kw)

    async def agrep(self, *a: Any, **kw: Any) -> Any:
        return await self._backend.agrep(*a, **kw)

    def upload_files(self, *a: Any, **kw: Any) -> Any:
        return self._backend.upload_files(*a, **kw)

    async def aupload_files(self, *a: Any, **kw: Any) -> Any:
        return await self._backend.aupload_files(*a, **kw)

    def download_files(self, *a: Any, **kw: Any) -> Any:
        return self._backend.download_files(*a, **kw)

    async def adownload_files(self, *a: Any, **kw: Any) -> Any:
        return await self._backend.adownload_files(*a, **kw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


class DeepAgentsProvider(AgentProvider):
    def __init__(self, guardrails_config: Any | None = None) -> None:
        self._guardrails_config = guardrails_config

    @property
    def name(self) -> str:
        return "deepagents"

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        from deepagents import create_deep_agent
        from deepagents.backends import LocalShellBackend
        from deepagents.backends.protocol import SandboxBackendProtocol

        SandboxBackendProtocol.register(GuardedBackend)

        logger.debug(
            "Starting deepagents query model=%s cwd=%s max_turns=%s",
            options.model,
            options.cwd,
            options.max_turns,
        )

        chat_model = _resolve_model(options.model, options.reasoning_config)
        real_backend = LocalShellBackend(root_dir=options.cwd, inherit_env=True)

        backend: Any = real_backend
        if self._guardrails_config and self._guardrails_config.enabled:
            from lightspeed_agentic.guardrails.checker import GuardrailsChecker
            from lightspeed_agentic.guardrails.types import GuardrailContext

            checker = GuardrailsChecker(self._guardrails_config)
            context = GuardrailContext(
                original_query=options.prompt,
                target_namespaces=options.target_namespaces,
            )
            backend = GuardedBackend(real_backend, checker, context)
            logger.info("[guardrails] GuardedBackend active for this query")

        agent_kwargs: dict[str, Any] = {
            "model": chat_model,
            "backend": backend,
            "system_prompt": options.system_prompt,
        }

        if has_skills(options.cwd):
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
