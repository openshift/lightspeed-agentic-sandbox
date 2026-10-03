"""DeepAgents middleware that gates model-visible tool results."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from lightspeed_agentic.inspection.chunking import serialize_tool_result
from lightspeed_agentic.inspection.errors import ToolResultSafetyInspectionFailed


class ToolResultInspector(Protocol):
    def __call__(
        self,
        tool_name: str,
        result_type: str,
        value: Any,
        tool_call_id: str,
    ) -> Awaitable[Any]: ...


class ToolResultInspectionMiddleware(AgentMiddleware[Any, Any, Any]):
    """Inspect tool output at the model boundary, after result transformations."""

    def __init__(self, inspector: ToolResultInspector) -> None:
        self._inspector = inspector
        self._passed_signatures: set[tuple[str, str, str, str]] = set()

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        for message in request.messages:
            if not isinstance(message, ToolMessage):
                continue

            tool_name = message.name or ""
            result_type = "error" if message.status == "error" else "result"
            try:
                signature = self._signature(
                    tool_name,
                    result_type,
                    message.tool_call_id or "",
                    message.content,
                )
            except (TypeError, ValueError):
                raise ToolResultSafetyInspectionFailed() from None
            if signature in self._passed_signatures:
                continue

            try:
                outcome = await self._inspector(
                    tool_name,
                    result_type,
                    message.content,
                    message.tool_call_id or "",
                )
            except asyncio.CancelledError:
                raise ToolResultSafetyInspectionFailed() from None
            except Exception:
                raise ToolResultSafetyInspectionFailed() from None

            if getattr(outcome, "passed", True) is False:
                raise ToolResultSafetyInspectionFailed()
            self._passed_signatures.add(signature)
        return await handler(request)

    def is_passed(
        self,
        tool_name: str,
        result_type: str,
        tool_call_id: str,
        content: Any,
    ) -> bool:
        """Return whether this exact model-visible result passed inspection."""
        try:
            signature = self._signature(tool_name, result_type, tool_call_id, content)
        except (TypeError, ValueError):
            return False
        return signature in self._passed_signatures

    @staticmethod
    def _signature(
        tool_name: str,
        result_type: str,
        tool_call_id: str,
        content: Any,
    ) -> tuple[str, str, str, str]:
        serialized = serialize_tool_result(content)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return tool_name, result_type, tool_call_id, digest
