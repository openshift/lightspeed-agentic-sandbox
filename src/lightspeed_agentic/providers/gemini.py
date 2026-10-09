"""Gemini provider — wraps google-adk.

Uses native ExecuteBashTool for shell execution and SkillToolset for
skill discovery. The SDK handles tool registration and command execution.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shlex
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

from lightspeed_agentic.types import (
    MAX_TOOL_RETURN_CHARS,
    TOOL_RETURN_PREVIEW_CHARS,
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


def _trim_tool_response(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
    tool_response: Any,
) -> Any:
    """Replace oversized Gemini tool results with a bounded preview."""
    _ = tool, args, tool_context
    serialized = stringify(tool_response)
    if len(serialized) <= MAX_TOOL_RETURN_CHARS:
        return None

    return {
        "status": "truncated",
        "preview": serialized[:TOOL_RETURN_PREVIEW_CHARS],
        "original_size": len(serialized),
        "message": "Tool output was truncated; request a narrower result if needed.",
    }


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _raw_sdk_fields(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if model_dump is not None:
        return cast(dict[str, Any], model_dump(mode="json", exclude_unset=True))
    return {}


def _generation_output_part(part: Any) -> dict[str, Any] | None:
    text = _field(part, "text")
    if text is not None:
        return {
            "type": "reasoning" if _field(part, "thought") else "text",
            "content": text,
        }

    function_call = _field(part, "function_call")
    if function_call is not None:
        output: dict[str, Any] = {"type": "tool_call"}
        for source, target in (
            ("id", "id"),
            ("name", "name"),
            ("args", "arguments"),
        ):
            value = _field(function_call, source)
            if value is not None:
                output[target] = value
        return output

    if _field(part, "function_response") is not None:
        return None

    tool_call = _field(part, "tool_call")
    if tool_call is not None:
        tool_type = _field(tool_call, "tool_type")
        if tool_type is None:
            return {**_raw_sdk_fields(tool_call), "type": "tool_call"}
        tool_type_name = getattr(tool_type, "value", tool_type)
        server_tool_call: dict[str, Any] = {"type": tool_type_name}
        args = _field(tool_call, "args")
        if args is not None:
            server_tool_call["args"] = args
        output = {
            "type": "server_tool_call",
            "name": tool_type_name,
            "server_tool_call": server_tool_call,
        }
        call_id = _field(tool_call, "id")
        if call_id is not None:
            output["id"] = call_id
        return output

    tool_response = _field(part, "tool_response")
    if tool_response is not None:
        tool_type = _field(tool_response, "tool_type")
        if tool_type is None:
            return {**_raw_sdk_fields(tool_response), "type": "tool_response"}
        tool_type_name = getattr(tool_type, "value", tool_type)
        server_tool_response: dict[str, Any] = {"type": tool_type_name}
        response = _field(tool_response, "response")
        if response is not None:
            server_tool_response["response"] = response
        output = {
            "type": "server_tool_call_response",
            "server_tool_call_response": server_tool_response,
        }
        call_id = _field(tool_response, "id")
        if call_id is not None:
            output["id"] = call_id
        return output

    return None


def _load_skills_toolset(skills_dir: str) -> Any:
    try:
        from google.adk.code_executors.unsafe_local_code_executor import (
            UnsafeLocalCodeExecutor,
        )
        from google.adk.skills import list_skills_in_dir, load_skill_from_dir
        from google.adk.tools.skill_toolset import SkillToolset

        target = pathlib.Path(skills_dir)
        skill_entries = list_skills_in_dir(target)
        skills = [
            load_skill_from_dir(target / skill_id)
            for skill_id in skill_entries
            if (target / skill_id).is_dir()
        ]
        if skills:
            return SkillToolset(
                skills=skills,
                code_executor=UnsafeLocalCodeExecutor(),
            )
    except Exception as e:
        logger.debug("Failed to load skills toolset from %s: %s", skills_dir, e)
    return None


class GeminiProvider(AgentProvider):
    def __init__(self) -> None:
        self._cached_skills: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "gemini"

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        from opentelemetry import context as otel_context
        from opentelemetry.trace import Status, StatusCode

        from lightspeed_agentic.tracing import (
            set_json_span_attribute,
            start_generation_span,
        )

        parent_context = otel_context.get_current()

        from google.adk.agents import Agent, RunConfig
        from google.adk.agents.run_config import StreamingMode  # type: ignore[attr-defined]
        from google.adk.models import Gemini
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.adk.tools import (  # type: ignore[attr-defined]
            exit_loop,
            google_search,
            url_context,
        )
        from google.adk.tools.bash_tool import ExecuteBashTool
        from google.adk.tools.tool_confirmation import ToolConfirmation
        from google.genai import types

        from lightspeed_agentic.tls import get_ssl_context

        workspace = pathlib.Path(options.cwd)

        bash = ExecuteBashTool(workspace=workspace)
        _orig_run = bash.run_async

        async def _auto_confirm_run(*, args: Any, tool_context: Any) -> Any:
            tool_context.tool_confirmation = ToolConfirmation(confirmed=True)
            # ExecuteBashTool uses subprocess_exec (no shell), so wrap through
            # bash -c to support shell builtins, PATH lookups, and pipes.
            if "command" in args:
                args = {**args, "command": f"bash -c {shlex.quote(args['command'])}"}
            return await _orig_run(args=args, tool_context=tool_context)

        bash.run_async = _auto_confirm_run  # type: ignore[method-assign]

        # TODO: investigate more ADK built-in tools:
        # load_artifacts, load_memory, computer_use, file_search, mcp_servers
        is_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE"
        tools: list[Any] = [bash]
        # Vertex AI rejects mixing search tools (google_search, url_context)
        # with non-search tools like bash in the same request.
        if not is_vertex:
            tools.extend([google_search, url_context])

        if options.cwd not in self._cached_skills:
            self._cached_skills[options.cwd] = _load_skills_toolset(options.cwd)
        skill_toolset = self._cached_skills[options.cwd]
        if skill_toolset is not None:
            tools.append(skill_toolset)

        mcp_toolsets: list[Any] = []
        if options.mcp_servers:
            from lightspeed_agentic.mcp import to_gemini_mcp_toolsets

            mcp_toolsets = to_gemini_mcp_toolsets(options.mcp_servers)
            try:
                for toolset in mcp_toolsets:
                    tools.extend(await toolset.get_tools())
            except BaseException:
                for toolset in mcp_toolsets:
                    await toolset.close()
                raise

        if not options.output_schema:
            tools.append(exit_loop)

        tool_config_kwargs: dict[str, Any] = {}
        if not is_vertex:
            tool_config_kwargs["include_server_side_tool_invocations"] = True

        gen_content_kwargs: dict[str, Any] = {
            "tool_config": types.ToolConfig(**tool_config_kwargs),
        }

        if options.reasoning_config:
            gen_content_kwargs["thinking_config"] = types.ThinkingConfig(**options.reasoning_config)

        gemini_model = Gemini(
            model=options.model,
            client_kwargs={
                "http_options": types.HttpOptions(
                    async_client_args={"verify": get_ssl_context()},
                ),
            },
        )
        gen_ai_provider = "gcp.vertex_ai" if is_vertex else "gcp.gemini"
        generation_span: Any = None
        generation_end_time: int | None = None

        def before_model_callback(callback_context: Any, llm_request: Any) -> None:
            nonlocal generation_span, generation_end_time
            _ = callback_context, llm_request
            close_open_generation("generation_interrupted")
            generation_span = start_generation_span(
                "generate_content",
                options.model,
                gen_ai_provider,
                parent_context=parent_context,
            )
            generation_end_time = None
            return None

        def after_model_callback(callback_context: Any, llm_response: Any) -> None:
            nonlocal generation_end_time
            _ = callback_context
            if not llm_response.partial:
                generation_end_time = time.time_ns()
            return None

        def close_open_generation(error_type: str) -> None:
            nonlocal generation_span, generation_end_time
            if generation_span is None:
                return
            generation_span.set_attribute("error.type", error_type)
            generation_span.set_status(Status(StatusCode.ERROR, error_type))
            generation_span.end()
            generation_span = None
            generation_end_time = None

        agent_kwargs: dict[str, Any] = {
            "name": "lightspeed",
            "model": gemini_model,
            "instruction": options.system_prompt,
            "tools": tools,
            "before_model_callback": before_model_callback,
            "after_model_callback": after_model_callback,
            "after_tool_callback": _trim_tool_response,
            "generate_content_config": types.GenerateContentConfig(**gen_content_kwargs),
        }

        agent = Agent(**agent_kwargs)

        if options.output_schema:
            # Bypass ADK's output_schema (routes through broken SetModelResponseTool)
            # and use Gemini's native response_schema directly.
            gen_cfg = agent.generate_content_config
            if gen_cfg is not None:
                gen_cfg.response_mime_type = "application/json"
                gen_cfg.response_schema = options.output_schema

        session_service = InMemorySessionService()
        runner = Runner(
            app_name="lightspeed",
            agent=agent,
            session_service=session_service,
        )

        try:
            user_id = f"agent-{int(time.time())}"
        except (OSError, OverflowError, ValueError):
            user_id = "agent"
        session = await session_service.create_session(app_name="lightspeed", user_id=user_id)

        streaming_mode = StreamingMode.SSE if options.stream else StreamingMode.NONE
        run_config = RunConfig(
            streaming_mode=streaming_mode,
            max_llm_calls=options.max_turns,
        )

        result_text = ""
        total_input_tokens = 0
        total_output_tokens = 0

        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=options.prompt)],
                ),
                run_config=run_config,
            ):
                if event.author == agent.name and generation_span is not None:
                    content = event.content
                    empty_terminal = (
                        (content is None or (not content.parts and content.role in (None, "model")))
                        and not event.partial
                        and generation_end_time is not None
                        and event.is_final_response()
                    )
                    is_model_event = content is not None and content.role == "model"
                    if is_model_event or empty_terminal:
                        if is_model_event and content is not None:
                            parts = [
                                output_part
                                for source_part in content.parts or []
                                if (output_part := _generation_output_part(source_part)) is not None
                            ]
                            set_json_span_attribute(
                                generation_span,
                                "gen_ai.output.messages",
                                [{"role": "assistant", "parts": parts}],
                            )

                        model_version = event.model_version
                        if model_version is not None:
                            generation_span.set_attribute("gen_ai.response.model", model_version)

                        finish_reason = event.finish_reason
                        if finish_reason is not None:
                            generation_span.set_attribute(
                                "gen_ai.response.finish_reasons",
                                [
                                    str(
                                        getattr(
                                            finish_reason,
                                            "value",
                                            finish_reason,
                                        )
                                    )
                                ],
                            )

                        usage = event.usage_metadata
                        for source, target in (
                            ("prompt_token_count", "gen_ai.usage.input_tokens"),
                            ("candidates_token_count", "gen_ai.usage.output_tokens"),
                            (
                                "thoughts_token_count",
                                "gen_ai.usage.reasoning.output_tokens",
                            ),
                        ):
                            value = _field(usage, source)
                            if value is not None:
                                generation_span.set_attribute(target, value)

                        error_code = event.error_code
                        if error_code is not None:
                            error_type = str(getattr(error_code, "value", error_code))
                            generation_span.set_attribute("error.type", error_type)
                            generation_span.set_status(Status(StatusCode.ERROR, error_type))

                        # ADK can queue partial events after the final callback ran.
                        # Consume its timestamp only with the finalized response.
                        if generation_end_time is not None and not event.partial:
                            generation_span.end(end_time=generation_end_time)
                            generation_span = None
                            generation_end_time = None
                if not event.content or not event.content.parts:
                    continue

                is_partial = getattr(event, "partial", False)

                for part in event.content.parts:
                    if (
                        hasattr(part, "thought")
                        and part.thought
                        and hasattr(part, "text")
                        and part.text
                    ):
                        yield ThinkingDeltaEvent(thinking=part.text)
                        continue

                    if hasattr(part, "text") and part.text:
                        if options.stream and is_partial:
                            yield TextDeltaEvent(text=part.text)
                        if not is_partial and not event.get_function_calls():
                            result_text = part.text

                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        yield ToolCallEvent(
                            name=fc.name or "",
                            input=json.dumps(dict(fc.args) if fc.args else {}),
                            call_id=getattr(fc, "id", "") or "",
                        )

                    if hasattr(part, "function_response") and part.function_response:
                        fr = part.function_response
                        response = fr.response
                        tool_error_type: str | None = None
                        if isinstance(response, Mapping) and response.get("error"):
                            error_code = response.get("error_code")
                            if isinstance(error_code, str) and error_code:
                                tool_error_type = error_code
                        yield ToolResultEvent(
                            output=stringify(response),
                            call_id=getattr(fr, "id", "") or "",
                            error_type=tool_error_type,
                        )

                usage = getattr(event, "usage_metadata", None)
                if usage:
                    total_input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    total_output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        except BaseException as exc:
            if not isinstance(exc, GeneratorExit):
                close_open_generation(type(exc).__name__)
            raise
        finally:
            if generation_span is not None:
                close_open_generation("generation_interrupted")
            for toolset in mcp_toolsets:
                await toolset.close()

        yield ContentBlockStopEvent()

        yield ResultEvent(
            text=result_text,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            response_model="",
        )
