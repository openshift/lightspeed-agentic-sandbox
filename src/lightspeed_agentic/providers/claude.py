"""Claude provider — wraps claude-agent-sdk.

Maps to lightspeed-agent/src/providers/claude.ts.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from lightspeed_agentic.types import (
    TOOL_INPUT_MAX_CHARS,
    TOOL_OUTPUT_MAX_CHARS,
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

_skills_link_cache: dict[str, str] = {}


class ClaudeProvider(AgentProvider):
    @property
    def name(self) -> str:
        return "claude"

    @staticmethod
    def _ensure_skills_link(cwd: str) -> str:
        """Set up .claude/skills/ with symlinks to each skill dir in cwd.

        Claude Code discovers skills from .claude/skills/ under its cwd.
        Creates the structure in cwd if writable (evals), otherwise falls
        back to /tmp (cluster with read-only volume mounts).
        """
        if cwd in _skills_link_cache:
            return _skills_link_cache[cwd]

        cwd_path = Path(cwd)
        if (cwd_path / ".claude" / "skills").exists():
            _skills_link_cache[cwd] = cwd
            return cwd

        fallback = Path(tempfile.gettempdir()) / "claude-workspace"
        for base in [cwd_path, fallback]:
            skills_dir = base / ".claude" / "skills"
            try:
                skills_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            for entry in cwd_path.iterdir():
                if entry.name.startswith(".") or not entry.is_dir():
                    continue
                with contextlib.suppress(FileExistsError):
                    (skills_dir / entry.name).symlink_to(entry)
            result = str(base)
            _skills_link_cache[cwd] = result
            return result

        _skills_link_cache[cwd] = cwd
        return cwd

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            StreamEvent,
            query,
        )

        effective_cwd = self._ensure_skills_link(options.cwd)

        output_format: dict[str, object] | None = None
        if options.output_schema:
            output_format = {
                "type": "json_schema",
                "schema": options.output_schema,
            }

        mcp_servers: dict[str, object] | None = None
        allowed_tools = list(options.allowed_tools)
        if options.mcp_servers:
            mcp_servers = {
                server.name: {
                    "type": "http",
                    "url": server.url,
                    **({"headers": server.headers} if server.headers else {}),
                }
                for server in options.mcp_servers
            }
            for server in options.mcp_servers:
                allowed_tools.append(f"mcp__{server.name}__*")

        sdk_options = ClaudeAgentOptions(
            model=options.model,
            max_turns=options.max_turns,
            max_budget_usd=options.max_budget_usd,
            system_prompt=options.system_prompt,
            allowed_tools=allowed_tools,
            permission_mode="bypassPermissions",
            cwd=effective_cwd,
            skills="all",
            include_partial_messages=True,
            output_format=output_format,
            mcp_servers=mcp_servers,
        )

        _tool_name = ""
        _tool_input_parts: list[str] = []
        _tool_input_len = 0

        async for msg in query(prompt=options.prompt, options=sdk_options):
            if isinstance(msg, StreamEvent):
                event = msg.event
                etype = event.get("type")
                if etype == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "tool_use":
                        _tool_name = block.get("name", "")
                        _tool_input_parts.clear()
                        _tool_input_len = 0
                elif etype == "content_block_delta":
                    delta = event.get("delta", {})
                    dtype = delta.get("type")
                    if dtype == "text_delta" and delta.get("text"):
                        yield TextDeltaEvent(text=delta["text"])
                    elif dtype == "thinking_delta" and delta.get("thinking"):
                        yield ThinkingDeltaEvent(thinking=delta["thinking"])
                    elif (
                        dtype == "input_json_delta"
                        and delta.get("partial_json")
                        and _tool_input_len < TOOL_INPUT_MAX_CHARS
                    ):
                        _tool_input_parts.append(delta["partial_json"])
                        _tool_input_len += len(delta["partial_json"])
                elif etype == "content_block_stop":
                    if _tool_name:
                        yield ToolCallEvent(
                            name=_tool_name,
                            input="".join(_tool_input_parts)[:TOOL_INPUT_MAX_CHARS],
                        )
                        _tool_name = ""
                        _tool_input_parts.clear()
                        _tool_input_len = 0
                    yield ContentBlockStopEvent()
                continue

            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if getattr(block, "type", None) == "tool_use":
                        yield ToolCallEvent(
                            name=getattr(block, "name", ""),
                            input=json.dumps(getattr(block, "input", {}))[:TOOL_INPUT_MAX_CHARS],
                        )

            if getattr(msg, "type", None) == "tool":
                for block in getattr(msg, "content", []):
                    if getattr(block, "type", None) == "tool_result":
                        yield ToolResultEvent(
                            output=stringify(getattr(block, "content", ""))[:TOOL_OUTPUT_MAX_CHARS],
                        )

            if isinstance(msg, ResultMessage):
                structured = getattr(msg, "structured_output", None)
                text = (
                    stringify(structured)
                    if structured is not None
                    else (getattr(msg, "result", None) or "")
                )

                usage = getattr(msg, "usage", None)
                yield ResultEvent(
                    text=text,
                    cost_usd=getattr(msg, "total_cost_usd", 0) or 0,
                    input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                    output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                )
