"""DeepAgents provider — wraps langchain-ai/deepagents for a resolved chat model.

Uses create_deep_agent() with LocalShellBackend for shell + filesystem access,
native skills loading, and v3 event streaming for event mapping.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast
from uuid import UUID

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.trace import Span, StatusCode

from lightspeed_agentic.skills import has_skills
from lightspeed_agentic.tracing import set_json_span_attribute, start_generation_span
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

if TYPE_CHECKING:
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute

# Harness SDK imports (deepagents, langchain-*, MCP) stay inside functions.
# Model construction and hosting configuration belong to the supplied model route.

logger = logging.getLogger(__name__)


_JSON_SCHEMA_TYPE_MAP: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


async def _close_model_clients(model: Any) -> None:
    """Close initialized SDK clients and constructor-owned resources without lazy creation."""
    clients: list[Any] = []
    model_state = getattr(model, "__dict__", {})
    for name in (
        "root_async_client",
        "_async_client",
        "async_client",
        "root_client",
        "_client",
        "client",
    ):
        if name in model_state and model_state[name] not in clients:
            clients.append(model_state[name])

    try:
        for client in clients:
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
    finally:
        for resource in model_state.get("_lightspeed_owned_resources", ()):
            await resource.aclose()


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
        description = field_schema.get("description")
        if field_name in required:
            fields[field_name] = (field_type, pydantic.Field(..., description=description))
        else:
            fields[field_name] = (field_type | None, pydantic.Field(None, description=description))

    return pydantic.create_model(
        name, __config__=pydantic.ConfigDict(regex_engine="python-re"), **fields
    )


def _resolve_field_type(schema: dict[str, Any], name: str) -> Any:
    json_type = schema.get("type", "string")

    if json_type == "object":
        return _json_schema_to_pydantic(schema, name.title().replace("_", ""))

    if json_type == "array":
        if "items" not in schema:
            raise ValueError(f"Array field {name!r} missing 'items'")
        item_type = _resolve_field_type(schema["items"], f"{name}_item")
        return list[item_type]  # type: ignore[valid-type]

    field_type: Any = (
        Literal[tuple(schema["enum"])]
        if "enum" in schema
        else _JSON_SCHEMA_TYPE_MAP.get(json_type, str)
    )
    if json_type == "string" and any(
        key in schema for key in ("minLength", "maxLength", "pattern")
    ):
        from pydantic import Field, StringConstraints

        return Annotated[
            field_type,
            StringConstraints(
                min_length=schema.get("minLength"),
                max_length=schema.get("maxLength"),
                pattern=schema.get("pattern"),
            ),
            # Pydantic can omit Literal constraints from the generated JSON Schema.
            Field(
                json_schema_extra={
                    key: schema[key]
                    for key in ("minLength", "maxLength", "pattern")
                    if key in schema
                }
            ),
        ]
    return field_type


def _usage_from_message(msg: Any) -> tuple[int, int]:
    usage = getattr(msg, "usage_metadata", None)
    if not usage:
        return 0, 0
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def _content_block_value(block: Any, name: str) -> Any:
    if isinstance(block, Mapping):
        return block.get(name)
    return getattr(block, name, None)


def _generation_parts(message: Any) -> list[dict[str, Any]]:
    """Map observed chat content blocks to ordered GenAI output parts."""
    content_blocks = getattr(message, "content_blocks", None)
    parts: list[dict[str, Any]] = []
    for block in content_blocks or []:
        block_type = _content_block_value(block, "type")
        if block_type == "text":
            text = _content_block_value(block, "text")
            if text is not None:
                parts.append({"type": "text", "content": text})
        elif block_type == "reasoning":
            reasoning = _content_block_value(block, "reasoning")
            if reasoning is not None:
                parts.append({"type": "reasoning", "content": reasoning})
        elif block_type == "tool_call":
            part: dict[str, Any] = {"type": "tool_call"}
            for source, target in (("id", "id"), ("name", "name"), ("args", "arguments")):
                value = _content_block_value(block, source)
                if value is not None and (target != "id" or value != ""):
                    part[target] = value
            parts.append(part)
        elif block_type == "tool_call_chunk":
            part = {"type": "tool_call_chunk"}
            for key in ("id", "name", "args", "index"):
                value = _content_block_value(block, key)
                if value is not None:
                    part[key] = value
            parts.append(part)

    content = getattr(message, "content", None)
    if not content_blocks and isinstance(content, str):
        parts.append({"type": "text", "content": content})
    return parts


def _generation_metadata_value(
    generations: list[tuple[Any, Any]],
    *keys: str,
    llm_output: Any = None,
) -> Any:
    for generation, message in generations:
        for metadata in (
            getattr(message, "response_metadata", None),
            getattr(generation, "generation_info", None),
        ):
            if isinstance(metadata, Mapping):
                for key in keys:
                    value = metadata.get(key)
                    if value is not None:
                        return value
    if isinstance(llm_output, Mapping):
        for key in keys:
            value = llm_output.get(key)
            if value is not None:
                return value
    return None


def _is_error_metadata_only(message: Any) -> bool:
    metadata = getattr(message, "response_metadata", None)
    return (
        isinstance(metadata, Mapping)
        and any(key in metadata for key in ("body", "headers", "status_code", "request_id"))
        and getattr(message, "content", None) == ""
        and not getattr(message, "content_blocks", None)
        and not getattr(message, "tool_calls", None)
    )


def _record_generation_result(
    span: Span,
    response: Any,
    *,
    partial: bool = False,
) -> None:
    """Record only output, metadata, and usage exposed by this SDK result."""
    generations: list[tuple[Any, Any]] = []
    llm_output = getattr(response, "llm_output", None)
    output_messages: list[dict[str, Any]] = []
    for prompt_index, choices in enumerate(getattr(response, "generations", None) or []):
        if partial and prompt_index > 0:
            break
        for generation in choices:
            message = getattr(generation, "message", None)
            if message is not None and not (partial and _is_error_metadata_only(message)):
                generations.append((generation, message))
                output_messages.append({"role": "assistant", "parts": _generation_parts(message)})

    if output_messages:
        set_json_span_attribute(span, "gen_ai.output.messages", output_messages)

    for attribute, keys in (
        ("gen_ai.response.model", ("model", "model_name")),
        ("gen_ai.response.id", ("id", "response_id")),
    ):
        value = _generation_metadata_value(
            generations,
            *keys,
            llm_output=llm_output,
        )
        if value is not None:
            span.set_attribute(attribute, value)

    finish_reasons: list[str] = []
    for generation, message in generations:
        reason = _generation_metadata_value([(generation, message)], "stop_reason", "finish_reason")
        if reason is not None:
            finish_reasons.append(str(reason))
    if not finish_reasons:
        reason = _generation_metadata_value(
            [],
            "stop_reason",
            "finish_reason",
            llm_output=llm_output,
        )
        if reason is not None:
            finish_reasons.append(str(reason))
    if finish_reasons:
        span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)

    usage_attributes = (
        ("input_tokens", "gen_ai.usage.input_tokens"),
        ("output_tokens", "gen_ai.usage.output_tokens"),
    )
    for key, attribute in usage_attributes:
        for _generation, message in generations:
            usage = getattr(message, "usage_metadata", None)
            if isinstance(usage, Mapping) and (value := usage.get(key)) is not None:
                span.set_attribute(attribute, value)
                break

    for _generation, message in generations:
        usage = getattr(message, "usage_metadata", None)
        details = usage.get("output_token_details") if isinstance(usage, Mapping) else None
        if isinstance(details, Mapping) and (value := details.get("reasoning")) is not None:
            span.set_attribute("gen_ai.usage.reasoning.output_tokens", value)
            break


def _create_generation_callback(
    model_route: ResolvedModelRoute,
    parent_context: Context,
    *,
    main_name: str | None,
) -> Any:
    """Create per-query tracing for the main DeepAgents model generations."""
    from langchain_core.callbacks import AsyncCallbackHandler

    class GenerationCallback(AsyncCallbackHandler):
        def __init__(self) -> None:
            super().__init__()
            self._spans: dict[UUID, Span] = {}
            self._parent_context = parent_context
            self._model = model_route.model
            self._provider = model_route.telemetry_provider
            self._main_name = main_name

        async def on_chat_model_start(
            self,
            serialized: dict[str, Any],
            messages: list[list[Any]],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            del serialized, messages, parent_run_id, kwargs
            tags = tags or []
            metadata = metadata or {}
            if (
                metadata.get("lc_agent_name") != self._main_name
                or "nostream" in tags
                or metadata.get("lc_source") == "summarization"
            ):
                return

            self._spans[run_id] = start_generation_span(
                "chat",
                self._model,
                self._provider,
                parent_context=self._parent_context,
            )

        async def on_llm_end(
            self,
            response: Any,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            del kwargs
            span = self._spans.pop(run_id, None)
            if span is None:
                return
            try:
                _record_generation_result(span, response)
            finally:
                span.end()

        async def on_llm_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            from langchain_core.outputs import LLMResult

            span = self._spans.pop(run_id, None)
            if span is None:
                return
            response = kwargs.get("response")
            try:
                if isinstance(response, LLMResult):
                    _record_generation_result(span, response, partial=True)
            finally:
                try:
                    span.set_attribute("error.type", type(error).__name__)
                    span.set_status(StatusCode.ERROR)
                finally:
                    span.end()

        def close_open(self, error_type: str) -> None:
            while self._spans:
                _run_id, span = self._spans.popitem()
                span.set_attribute("error.type", error_type)
                span.set_status(StatusCode.ERROR)
                span.end()

    return GenerationCallback()


async def _shape_structured_output(
    model_route: ResolvedModelRoute,
    schema: Any,
    system_prompt: str,
    prompt: str,
    agent_text: str,
    generation_callback: Any,
    *,
    prepared_schema: Any | None = None,
) -> tuple[Any, int, int]:
    """Shape pass: tool-free structured binding on a model without thinking."""
    from langchain_core.messages import HumanMessage, SystemMessage

    format_model = model_route.create_model(model_route.model_parameters, None)
    try:
        validation_schema = schema
        if prepared_schema is not None:
            schema = prepared_schema
        elif model_route.translate_output_schema is not None:
            schema = model_route.translate_output_schema(schema)
        if model_route.bind_structured_output is None:
            structured = format_model.with_structured_output(
                schema,
                method=model_route.structured_output_method,
                include_raw=True,
            )
        else:
            structured = model_route.bind_structured_output(format_model, schema)
        shape_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Original user request:\n{prompt}\n\n"
                    f"Agent run output:\n{agent_text}\n\n"
                    "Call the structured response tool with every required field. "
                    "Use native JSON types: booleans for boolean fields, arrays for array fields, "
                    "and objects for object fields. "
                    "Never serialize an array or object as a string. "
                    "Base the field values only on the agent run output."
                )
            ),
        ]
        result = await structured.ainvoke(
            shape_messages,
            config={"callbacks": [generation_callback]},
        )
    except BaseException as exc:
        generation_callback.close_open(type(exc).__name__)
        raise
    finally:
        generation_callback.close_open("generation_interrupted")
        await _close_model_clients(format_model)
    if not isinstance(result, dict) or "parsed" not in result:
        raise ValueError("Structured output response is missing parsed output")
    if result.get("parsing_error") is not None:
        raise ValueError("Structured output parsing failed")
    parsed = result["parsed"]
    if isinstance(validation_schema, dict):
        from jsonschema import ValidationError, validate  # type: ignore[import-untyped]
        from referencing import Registry

        try:
            # Local definitions/references work; an empty registry never fetches URLs.
            validate(parsed, validation_schema, registry=Registry())
        except ValidationError:
            raise ValueError("Structured output does not match configured schema") from None
    elif parsed is None:
        raise ValueError("Structured output response is missing parsed output")
    in_tok, out_tok = _usage_from_message(result.get("raw"))
    return parsed, in_tok, out_tok


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

    if not getattr(msg, "content_blocks", None) and msg.content:
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
    def __init__(self, model_route: ResolvedModelRoute) -> None:
        self._model_route = model_route

    @property
    def name(self) -> str:
        return "deepagents"

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        parent_context = otel_context.get_current()

        from deepagents import create_deep_agent
        from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend

        from lightspeed_agentic.inspection.errors import ToolResultSafetyInspectionFailed

        try:
            from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

            from lightspeed_agentic.inspection.middleware import (
                TOOL_DATA_TRUST_INSTRUCTION,
                ToolResultInspectionMiddleware,
            )
            from lightspeed_agentic.inspection.summarization import (
                create_tool_data_summarization_middleware,
            )
        except Exception as exc:
            raise ToolResultSafetyInspectionFailed() from exc

        classifier_model: Any | None = None
        inspector_callback: Any | None = None

        logger.debug(
            "Starting deepagents query route=%s model=%s cwd=%s max_turns=%s",
            self._model_route.route,
            self._model_route.model,
            options.cwd,
            options.max_turns,
        )

        # Reject known schema/capability errors before constructing clients or
        # letting the graph execute tools. Keep the original schema for validation.
        schema_model: Any | None = None
        prepared_schema: Any | None = None
        if options.output_schema:
            schema_model = (
                _json_schema_to_pydantic(options.output_schema)
                if isinstance(options.output_schema, dict)
                and self._model_route.structured_output_schema_mode == "pydantic"
                else options.output_schema
            )
            prepared_schema = schema_model
            if self._model_route.translate_output_schema is not None:
                prepared_schema = self._model_route.translate_output_schema(schema_model)

        chat_model = self._model_route.create_model(
            self._model_route.model_parameters, options.reasoning_config
        )
        generation_callback: Any | None = None
        try:
            local_backend = LocalShellBackend(
                root_dir=options.cwd,
                virtual_mode=False,
                inherit_env=True,
                max_output_bytes=MAX_TOOL_RETURN_CHARS,
            )
            skills_root = str(local_backend.cwd)
            # SDK-native skills layout: the skills root is a virtual CompositeBackend
            # route, so file tools advertise /skills/<name>/... while the SDK's
            # FilesystemMiddleware injects the virtual-to-host path mapping that the
            # shell needs (deepagents #3050/#3762 pattern). The default backend keeps
            # physical paths for general work; artifacts stay in the agent home.
            skills_backend = FilesystemBackend(root_dir=skills_root, virtual_mode=True)
            backend = CompositeBackend(
                default=local_backend,
                routes={"/skills/": skills_backend},
                artifacts_root=os.path.join(os.environ.get("HOME", "/home/agent"), ".deepagents"),
            )

            instruction = TOOL_DATA_TRUST_INSTRUCTION.strip()
            subagent_spec = {**GENERAL_PURPOSE_SUBAGENT}
            subagent_system_prompt = subagent_spec.get("system_prompt") or ""
            subagent_spec["system_prompt"] = (
                f"{subagent_system_prompt}\n{instruction}"
                if subagent_system_prompt
                else instruction
            )
            agent_kwargs: dict[str, Any] = {
                "model": chat_model,
                "backend": backend,
                "system_prompt": f"{options.system_prompt}\n{instruction}",
            }

            if options.tool_output_inspection_enabled:
                try:
                    from lightspeed_agentic.inspection.chunking import Utf8ByteCodec
                    from lightspeed_agentic.inspection.client import (
                        CLASSIFIER_OUTPUT_TOKENS,
                        LangChainClassifierClient,
                    )
                    from lightspeed_agentic.inspection.inspector import (
                        inspect_tool_result as run_inspection,
                    )

                    classifier_model = self._model_route.create_model(
                        replace(
                            self._model_route.model_parameters,
                            max_output_tokens=CLASSIFIER_OUTPUT_TOKENS,
                        ),
                        None,
                    )
                    # Inherited graph callbacks otherwise auto-stream nested calls;
                    # nostream hides graph events, not SDK transport streaming.
                    classifier_model.disable_streaming = True
                    classifier_client = LangChainClassifierClient(classifier_model)
                    model_profile = getattr(classifier_model, "profile", None) or {}
                    context_window_tokens = (
                        model_profile.get("max_input_tokens")
                        or model_profile.get("max_context_size")
                        or 100_000
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
                            output_tokens=CLASSIFIER_OUTPUT_TOKENS,
                            deadline=options.deadline,
                            provider=self._model_route.telemetry_provider,
                            model=self._model_route.model,
                        )

                    inspector_callback = inspect_tool_result_callback
                except Exception as exc:
                    raise ToolResultSafetyInspectionFailed() from exc

            inspection_middleware = ToolResultInspectionMiddleware(inspector_callback)
            summarization_middleware = create_tool_data_summarization_middleware(
                chat_model, backend
            )
            agent_kwargs["middleware"] = [inspection_middleware, summarization_middleware]
            subagent_spec["middleware"] = [inspection_middleware, summarization_middleware]
            agent_kwargs["subagents"] = [subagent_spec]
            generation_callback = _create_generation_callback(
                self._model_route,
                parent_context,
                main_name=agent_kwargs.get("name"),
            )

            if has_skills(skills_root):
                agent_kwargs["skills"] = ["/skills/"]

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
                    mcp_tools.extend(
                        tool for tool in server_tools if tool.name in allowed_tool_names
                    )

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
            stream_config["callbacks"] = [generation_callback]
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

            async for msg, _stream_metadata in cast(Any, agent).astream(
                input_state,
                config=stream_config,
                stream_mode="messages",
            ):
                if msg.type in ("ai", "AIMessageChunk"):
                    if options.tool_output_inspection_enabled:
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
                        error_type=("tool_error" if result_type == "error" else None),
                    )
                    if not options.tool_output_inspection_enabled:
                        if not inspection_middleware.is_passed(
                            tool_name,
                            result_type,
                            call_id,
                            msg.content,
                        ):
                            raise ToolResultSafetyInspectionFailed()
                        yield tool_result_event
                    else:
                        pending_tool_results.append(
                            (tool_name, result_type, call_id, msg.content, tool_result_event)
                        )
            for event in flush_pending_tool_calls():
                yield event
        except BaseException as exc:
            if generation_callback is not None and not isinstance(exc, GeneratorExit):
                generation_callback.close_open(type(exc).__name__)
            raise
        finally:
            if generation_callback is not None:
                generation_callback.close_open("generation_interrupted")
            await _close_model_clients(chat_model)
            if classifier_model is not None:
                await _close_model_clients(classifier_model)

        if schema_model is not None:
            structured, in_tok, out_tok = await _shape_structured_output(
                self._model_route,
                schema_model,
                options.system_prompt,
                options.prompt,
                result_text,
                generation_callback,
                prepared_schema=prepared_schema,
            )
            result_text = stringify(structured)
            total_input_tokens += in_tok
            total_output_tokens += out_tok

        yield ResultEvent(
            text=result_text,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )
