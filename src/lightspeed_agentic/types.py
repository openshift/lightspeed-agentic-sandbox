from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from lightspeed_agentic.config import McpServerConfig

TOOL_INPUT_MAX_CHARS = 300
TOOL_OUTPUT_MAX_CHARS = 500
DEFAULT_MODEL = "claude-opus-4-6"


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(exclude_none=True))
    return json.dumps(value) if value else ""


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


@dataclass(frozen=True)
class ToolResultEvent:
    type: Literal["tool_result"] = field(default="tool_result", init=False)
    output: str = ""


@dataclass(frozen=True)
class ResultEvent:
    type: Literal["result"] = field(default="result", init=False)
    text: str = ""
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


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
    model: str
    max_turns: int
    max_budget_usd: float
    allowed_tools: list[str]
    cwd: str
    output_schema: dict[str, Any] | None = None
    stream: bool = False
    mcp_servers: tuple[McpServerConfig, ...] = ()


class AgentProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]: ...
