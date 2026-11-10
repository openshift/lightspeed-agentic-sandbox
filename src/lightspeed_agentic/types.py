from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from lightspeed_agentic.mcp import AdmittedMCPProviderServer

MAX_TOOL_RETURN_CHARS = 32_000


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(exclude_none=True))
    return json.dumps(value) if value is not None else ""


@dataclass(frozen=True)
class TextDeltaEvent:
    type: Literal["text_delta"] = field(default="text_delta", init=False)
    text: str = ""


@dataclass(frozen=True)
class ThinkingDeltaEvent:
    type: Literal["thinking_delta"] = field(default="thinking_delta", init=False)
    thinking: str = ""


@dataclass(frozen=True)
class ContentBlockStopEvent:
    type: Literal["content_block_stop"] = field(default="content_block_stop", init=False)


@dataclass(frozen=True)
class ToolCallEvent:
    type: Literal["tool_call"] = field(default="tool_call", init=False)
    name: str = ""
    input: str = ""
    call_id: str = ""
    trace_input: str | None = None


@dataclass(frozen=True)
class ToolResultEvent:
    type: Literal["tool_result"] = field(default="tool_result", init=False)
    output: str = ""
    call_id: str = ""
    error_type: str | None = None
    redacted: bool = False
    redacted_count: int = 0
    redaction_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ResultEvent:
    type: Literal["result"] = field(default="result", init=False)
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    response_model: str = ""


ProviderEvent = (
    TextDeltaEvent
    | ThinkingDeltaEvent
    | ContentBlockStopEvent
    | ToolCallEvent
    | ToolResultEvent
    | ResultEvent
)


@dataclass
class ProviderQueryOptions:
    prompt: str
    system_prompt: str
    max_turns: int
    cwd: str
    output_schema: dict[str, Any] | None = None
    mcp_servers: list[AdmittedMCPProviderServer] = field(default_factory=list)
    reasoning_config: dict[str, Any] | None = None
    tool_output_inspection_enabled: bool = True
    deadline: float | None = None


class MCPInitializationError(RuntimeError):
    """Payload-free operational failure while materializing admitted MCP tools."""

    def __init__(self, *, server_names: tuple[str, ...], error_type: str) -> None:
        super().__init__("MCP initialization failed.")
        self.server_names = server_names
        self.error_type = error_type


class MCPToolNameCollisionError(RuntimeError):
    """Payload-free rejection of an ambiguous admitted MCP tool namespace."""

    def __init__(self, *, tool_name: str, server_names: tuple[str, str]) -> None:
        super().__init__("MCP tool name collision.")
        self.tool_name = tool_name
        self.server_names = server_names


class ToolExecutionError(RuntimeError):
    """Payload-free operational failure raised at the SDK tool-handler boundary."""

    def __init__(self, *, tool_name: str, error_type: str) -> None:
        super().__init__("tool execution failed.")
        self.tool_name = tool_name
        self.error_type = error_type


class AgentProvider(ABC):
    # Native capture must not be duplicated by the normalized-event audit fallback.
    captures_native_tool_spans = False

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]: ...

    async def aclose(self) -> None:
        """Release provider-owned async resources, if any."""
        return None
