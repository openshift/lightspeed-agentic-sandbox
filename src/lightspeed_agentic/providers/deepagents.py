"""DeepAgents provider — wraps langchain-ai/deepagents for Anthropic model support.

Uses create_deep_agent() with LocalShellBackend for shell + filesystem access,
native skills loading, and v3 event streaming for event mapping.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlparse

if TYPE_CHECKING:
    from lightspeed_agentic.audit import AuditLogger

from lightspeed_agentic.skills import has_skills
from lightspeed_agentic.types import (
    MAX_TOOL_RETURN_CHARS,
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

# Provider SDK imports (deepagents, langchain-*, MCP) stay inside functions:
# - _resolve_model loads only the active backend branch (Vertex / Bedrock / direct).
# - query() / shape / MCP load their SDKs on first use, not at module import.
# That keeps optional-extra isolation and avoids importing unused backends; it does
# not skip work on the hot path once a run is underway.

logger = logging.getLogger(__name__)

_JSON_SCHEMA_TYPE_MAP: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

_NATIVE_ANTHROPIC_HOSTS = {"api.anthropic.com"}


def _anthropic_backend() -> Literal["vertex", "bedrock", "direct"]:
    """Resolve Anthropic backend from env; reject conflicting Vertex/Bedrock flags."""
    use_vertex = os.environ.get("CLAUDE_CODE_USE_VERTEX") == "1"
    use_bedrock = os.environ.get("CLAUDE_CODE_USE_BEDROCK") == "1"
    if use_vertex and use_bedrock:
        raise ValueError("CLAUDE_CODE_USE_VERTEX and CLAUDE_CODE_USE_BEDROCK cannot both be set")
    if use_vertex:
        return "vertex"
    if use_bedrock:
        return "bedrock"
    return "direct"


def _resolve_model(model: str, reasoning_config: dict[str, Any] | None = None) -> Any:
    """Build a LangChain chat model instance based on env vars set by config.py."""
    from functools import cached_property

    thinking = reasoning_config.get("thinking") if reasoning_config else None
    backend = _anthropic_backend()

    if backend == "vertex":
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex

        kwargs: dict[str, Any] = {
            "model_name": model,
            "project": os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", ""),
            "location": os.environ.get("CLOUD_ML_REGION", "us-east5"),
        }
        if thinking:
            kwargs["thinking"] = thinking
        from lightspeed_agentic.tls import create_async_http_client, create_http_client

        kwargs["http_client"] = create_http_client()
        kwargs["async_http_client"] = create_async_http_client()
        return ChatAnthropicVertex(**kwargs)

    if backend == "bedrock":
        # langchain_aws uses these Anthropic Bedrock clients internally, but does not
        # expose a stable injection point for a custom HTTPX client. Keep this import
        # aligned with the installed anthropic SDK version.
        from anthropic.lib.bedrock._client import AnthropicBedrock, AsyncAnthropicBedrock
        from langchain_aws import ChatAnthropicBedrock

        from lightspeed_agentic.tls import create_async_http_client, create_http_client

        class TLSChatAnthropicBedrock(ChatAnthropicBedrock):
            @cached_property
            def _client(self) -> Any:
                return AnthropicBedrock(**self._client_params, http_client=create_http_client())

            @cached_property
            def _async_client(self) -> Any:
                return AsyncAnthropicBedrock(
                    **self._client_params,
                    http_client=create_async_http_client(),
                )

        kwargs = {
            "model": model,
            "region_name": os.environ.get("AWS_REGION", "us-east-1"),
        }
        if thinking:
            kwargs["thinking"] = thinking
        if not isinstance(ChatAnthropicBedrock, type):
            return ChatAnthropicBedrock(**kwargs)
        return TLSChatAnthropicBedrock(**kwargs)

    from anthropic import Anthropic, AsyncAnthropic
    from langchain_anthropic import ChatAnthropic

    from lightspeed_agentic.tls import create_async_http_client, create_http_client

    class TLSChatAnthropic(ChatAnthropic):
        @cached_property
        def _client(self) -> Any:
            return Anthropic(**self._client_params, http_client=create_http_client())

        @cached_property
        def _async_client(self) -> Any:
            return AsyncAnthropic(**self._client_params, http_client=create_async_http_client())

    kwargs = {"model": model}
    if thinking:
        kwargs["thinking"] = thinking

    # Support bearer token auth for vLLM and other Anthropic-compatible endpoints
    default_headers = {}
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if auth_token:
        default_headers["Authorization"] = f"Bearer {auth_token}"
    if default_headers:
        kwargs["default_headers"] = default_headers

    if not isinstance(ChatAnthropic, type):
        return ChatAnthropic(**kwargs)
    return TLSChatAnthropic(**kwargs)


async def _close_model_clients(model: Any) -> None:
    """Close already-created sync and async clients without triggering lazy creation."""
    clients: list[Any] = []
    model_state = getattr(model, "__dict__", {})
    for name in ("_async_client", "async_client", "_client", "client"):
        if name in model_state and model_state[name] not in clients:
            clients.append(model_state[name])

    for client in clients:
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result


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


def _usage_from_message(msg: Any) -> tuple[int, int]:
    usage = getattr(msg, "usage_metadata", None)
    if not usage:
        return 0, 0
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def _is_custom_anthropic_endpoint() -> bool:
    """Return whether Anthropic base URL points to a custom endpoint."""
    raw_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    if not raw_url:
        return False

    hostname = urlparse(raw_url).hostname
    return not hostname or hostname.lower().rstrip(".") not in _NATIVE_ANTHROPIC_HOSTS


def _structured_output_method() -> str:
    """Select structured-output binding compatible with active Anthropic endpoint."""
    backend = _anthropic_backend()
    if backend == "bedrock":
        return "function_calling"
    if backend == "direct" and not _is_custom_anthropic_endpoint():
        return "function_calling"
    return "json_schema"


async def _shape_structured_output(
    model: str,
    schema: Any,
    system_prompt: str,
    prompt: str,
    agent_text: str,
    audit_logger: AuditLogger | None = None,
) -> tuple[Any, int, int]:
    """Shape pass: tool-free structured binding on a model without thinking."""
    from langchain_core.messages import HumanMessage, SystemMessage

    format_model = _resolve_model(model, reasoning_config=None)
    structured = format_model.with_structured_output(
        schema,
        method=_structured_output_method(),
        include_raw=True,
    )
    shape_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Original user request:\n{prompt}\n\n"
                f"Agent run output:\n{agent_text}\n\n"
                "Produce the structured response matching the required schema."
            )
        ),
    ]
    callback_handler = None
    if audit_logger is not None:
        from lightspeed_agentic.providers.deepagents_telemetry import create_callback_handler

        callback_handler = create_callback_handler(
            audit_logger,
            model=model,
            output_type="json",
        )
    request_error: BaseException | None = None
    try:
        if callback_handler is None:
            result = await structured.ainvoke(shape_messages)
        else:
            result = await structured.ainvoke(
                shape_messages,
                config={"callbacks": [callback_handler], "tags": ["nostream"]},
            )
    except BaseException as exc:
        request_error = exc
        raise
    finally:
        if callback_handler is not None:
            callback_handler.close(error=request_error)
        await _close_model_clients(format_model)
    if isinstance(result, dict) and "parsed" in result:
        parsed = result["parsed"]
        in_tok, out_tok = _usage_from_message(result.get("raw"))
        return parsed, in_tok, out_tok
    return result, 0, 0


def _tool_call_events(msg: Any) -> list[ProviderEvent]:
    """Map complete parsed tool calls to provider events."""
    return [
        ToolCallEvent(
            name=tc.get("name", ""),
            input=json.dumps(tc.get("args", {})),
            call_id=tc.get("id", ""),
        )
        for tc in msg.tool_calls or []
    ]


def _process_ai_message(
    msg: Any,
    *,
    include_tool_calls: bool = True,
) -> tuple[list[ProviderEvent], str, int, int]:
    """Map one AIMessage chunk to provider events and token deltas."""
    events: list[ProviderEvent] = _tool_call_events(msg) if include_tool_calls else []
    text_delta = ""
    input_tokens = 0
    output_tokens = 0

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

        classifier_model: Any | None = None
        inspection_middleware: Any | None = None

        logger.debug(
            "Starting deepagents query model=%s cwd=%s max_turns=%s",
            options.model,
            options.cwd,
            options.max_turns,
        )

        chat_model = _resolve_model(options.model, options.reasoning_config)
        audit_callbacks = None
        if options.audit_logger is not None:
            from lightspeed_agentic.providers.deepagents_telemetry import create_callback_handler

            audit_callbacks = create_callback_handler(
                options.audit_logger,
                model=options.model,
            )
        backend = LocalShellBackend(
            root_dir=options.cwd,
            inherit_env=True,
            max_output_bytes=MAX_TOOL_RETURN_CHARS,
        )

        agent_kwargs: dict[str, Any] = {
            "model": chat_model,
            "backend": backend,
            "system_prompt": options.system_prompt,
        }

        if options.tool_output_inspection_enabled:
            from lightspeed_agentic.inspection.errors import ToolResultSafetyInspectionFailed

            try:
                from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

                from lightspeed_agentic.inspection.chunking import Utf8ByteCodec
                from lightspeed_agentic.inspection.client import LangChainClassifierClient
                from lightspeed_agentic.inspection.inspector import (
                    inspect_tool_result as run_inspection,
                )
                from lightspeed_agentic.inspection.middleware import ToolResultInspectionMiddleware

                classifier_model = _resolve_model(options.model, reasoning_config=None)
                classifier_client = LangChainClassifierClient(
                    classifier_model,
                    audit_logger=options.audit_logger,
                    requested_model=options.model,
                )
                model_profile = getattr(classifier_model, "profile", None) or {}
                context_window_tokens = (
                    model_profile.get("max_input_tokens")
                    or model_profile.get("max_context_size")
                    or 100_000
                )
                inspection_correlation = (
                    options.audit_logger.correlation_attributes()
                    if options.audit_logger is not None
                    else {}
                )

                async def inspect_tool_result_callback(
                    tool_name: str,
                    result_type: str,
                    value: Any,
                    tool_call_id: str,
                ) -> Any:
                    return await run_inspection(
                        classifier_client,
                        tool_name=tool_name,
                        result_type=result_type,
                        value=value,
                        codec=Utf8ByteCodec(),
                        tool_call_id=tool_call_id or None,
                        context_window_tokens=context_window_tokens,
                        instruction_tokens=512,
                        output_tokens=128,
                        deadline=options.deadline,
                        provider="anthropic",
                        model=options.model,
                        correlation_attributes=inspection_correlation,
                    )

                inspection_middleware = ToolResultInspectionMiddleware(inspect_tool_result_callback)
                agent_kwargs["middleware"] = [inspection_middleware]
                agent_kwargs["subagents"] = [
                    {
                        **GENERAL_PURPOSE_SUBAGENT,
                        "middleware": [inspection_middleware],
                    }
                ]
            except Exception as exc:
                if classifier_model is not None:
                    await _close_model_clients(classifier_model)
                raise ToolResultSafetyInspectionFailed() from exc

        if has_skills(options.cwd):
            agent_kwargs["skills"] = [options.cwd]

        schema_model: Any | None = None
        if options.output_schema:
            schema_model = (
                _json_schema_to_pydantic(options.output_schema)
                if isinstance(options.output_schema, dict)
                else options.output_schema
            )

        mcp_tools: list[Any] = []
        if options.mcp_servers:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            from lightspeed_agentic.tls import create_async_http_client

            client = MultiServerMCPClient(
                {
                    server.name: {  # type: ignore[misc]
                        "transport": "http",
                        "url": server.url,
                        "headers": {h.name: h.value for h in server.headers},
                        "timeout": server.timeout,
                        "httpx_client_factory": create_async_http_client,
                    }
                    for server in options.mcp_servers
                }
            )
            for server in options.mcp_servers:
                allowed_tool_names = set(server.allowed_tool_names)
                server_tools = await client.get_tools(server_name=server.name)
                mcp_tools.extend(tool for tool in server_tools if tool.name in allowed_tool_names)

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
        if audit_callbacks is not None:
            stream_config["callbacks"] = [audit_callbacks]
        result_text = ""
        pending_tool_results: list[tuple[str, str, str, Any, ToolResultEvent]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        pending_tool_call_chunk: Any | None = None
        input_state = {"messages": [{"role": "user", "content": options.prompt}]}

        def flush_pending_tool_calls() -> list[ProviderEvent]:
            nonlocal pending_tool_call_chunk
            if pending_tool_call_chunk is None:
                return []
            events = _tool_call_events(pending_tool_call_chunk)
            pending_tool_call_chunk = None
            return events

        stream_error: BaseException | None = None
        try:
            async for msg, _stream_metadata in cast(Any, agent).astream(
                input_state,
                config=stream_config,
                stream_mode="messages",
            ):
                if msg.type in ("ai", "AIMessageChunk"):
                    if inspection_middleware is not None:
                        for (
                            tool_name,
                            result_type,
                            call_id,
                            content,
                            pending_event,
                        ) in pending_tool_results:
                            if not inspection_middleware.is_passed(
                                tool_name,
                                result_type,
                                call_id,
                                content,
                            ):
                                raise ToolResultSafetyInspectionFailed()
                            yield pending_event
                        pending_tool_results.clear()

                    include_tool_calls = msg.type != "AIMessageChunk"
                    if msg.type == "AIMessageChunk":
                        is_last_chunk = getattr(msg, "chunk_position", None) == "last"
                        tool_call_chunks = getattr(msg, "tool_call_chunks", []) or []
                        if tool_call_chunks or (
                            pending_tool_call_chunk is not None and is_last_chunk
                        ):
                            current_tool_call_chunk = type(msg)(
                                content="",
                                tool_call_chunks=tool_call_chunks,
                                chunk_position="last" if is_last_chunk else None,
                            )
                            pending_tool_call_chunk = (
                                current_tool_call_chunk
                                if pending_tool_call_chunk is None
                                else pending_tool_call_chunk + current_tool_call_chunk
                            )
                        if is_last_chunk:
                            for event in flush_pending_tool_calls():
                                yield event
                    elif pending_tool_call_chunk is not None:
                        if getattr(msg, "tool_calls", None):
                            pending_tool_call_chunk = None
                        else:
                            for event in flush_pending_tool_calls():
                                yield event

                    events, text_delta, in_tok, out_tok = _process_ai_message(
                        msg,
                        include_tool_calls=include_tool_calls,
                    )
                    for provider_event in events:
                        yield provider_event
                    result_text += text_delta
                    total_input_tokens += in_tok
                    total_output_tokens += out_tok

                elif msg.type in ("tool", "ToolMessageChunk"):
                    for event in flush_pending_tool_calls():
                        yield event
                    tool_name = getattr(msg, "name", "") or ""
                    result_type = (
                        "error" if getattr(msg, "status", "success") == "error" else "result"
                    )
                    call_id = getattr(msg, "tool_call_id", "") or ""
                    tool_result_event = ToolResultEvent(
                        output=stringify(msg.content),
                        call_id=call_id,
                    )
                    if inspection_middleware is None:
                        yield tool_result_event
                    else:
                        pending_tool_results.append(
                            (tool_name, result_type, call_id, msg.content, tool_result_event)
                        )
            for event in flush_pending_tool_calls():
                yield event
        except BaseException as exc:
            stream_error = exc
            raise
        finally:
            if audit_callbacks is not None:
                audit_callbacks.close(error=stream_error)
            await _close_model_clients(chat_model)
            if classifier_model is not None:
                await _close_model_clients(classifier_model)

        if schema_model is not None:
            structured, in_tok, out_tok = await _shape_structured_output(
                options.model,
                schema_model,
                options.system_prompt,
                options.prompt,
                result_text,
                audit_logger=options.audit_logger,
            )
            result_text = stringify(structured)
            total_input_tokens += in_tok
            total_output_tokens += out_tok

        yield ResultEvent(
            text=result_text,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )
