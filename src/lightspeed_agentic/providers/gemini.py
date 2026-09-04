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
                code_executor=UnsafeLocalCodeExecutor(),  # type: ignore[no-untyped-call]
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
        from google.adk.agents import Agent, RunConfig
        from google.adk.agents.run_config import StreamingMode
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

        if options.mcp_servers:
            from lightspeed_agentic.mcp import to_gemini_mcp_toolsets

            tools.extend(to_gemini_mcp_toolsets(options.mcp_servers))

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

        agent_kwargs: dict[str, Any] = {
            "name": "lightspeed",
            "model": options.model,
            "instruction": options.system_prompt,
            "tools": tools,
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

        session_service = InMemorySessionService()  # type: ignore[no-untyped-call]
        runner = Runner(
            app_name="lightspeed",
            agent=agent,
            session_service=session_service,
        )

        user_id = f"agent-{int(time.time())}"
        session = await session_service.create_session(app_name="lightspeed", user_id=user_id)

        streaming_mode = StreamingMode.SSE if options.stream else StreamingMode.NONE
        run_config = RunConfig(
            streaming_mode=streaming_mode,
            max_llm_calls=options.max_turns,
        )

        result_text = ""
        total_input_tokens = 0
        total_output_tokens = 0

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=options.prompt)],
            ),
            run_config=run_config,
        ):
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
                    yield ToolResultEvent(
                        output=stringify(fr.response),
                        call_id=getattr(fr, "id", "") or "",
                    )

            usage = getattr(event, "usage_metadata", None)
            if usage:
                total_input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                total_output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        yield ContentBlockStopEvent()

        yield ResultEvent(
            text=result_text,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            response_model="",
        )
