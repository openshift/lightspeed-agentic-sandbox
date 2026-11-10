"""DeepAgents middleware that gates model-visible tool results."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any, Protocol

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from lightspeed_agentic.inspection.chunking import serialize_tool_result
from lightspeed_agentic.inspection.errors import ToolResultSafetyInspectionFailed
from lightspeed_agentic.inspection.tool_data import (
    _wrap_tool_output,
    filter_untrusted_tool_results,
)
from lightspeed_agentic.redaction.engine import CompiledRuleSet, Redactor
from lightspeed_agentic.types import ToolExecutionError, ToolResultEvent, stringify

logger = logging.getLogger("lightspeed_agentic")


def _stringify_tool_output(value: Any) -> str:
    """Serialize nested model values for the redaction event without changing tool data."""
    if isinstance(value, str):
        return value

    def normalize(item: Any) -> Any:
        if hasattr(item, "model_dump"):
            return normalize(item.model_dump(exclude_none=False))
        if isinstance(item, dict):
            return {normalize(key): normalize(child) for key, child in item.items()}
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, tuple):
            return tuple(normalize(child) for child in item)
        return item

    return stringify(normalize(value))


def tool_result_signature(
    tool_name: str,
    result_type: str,
    tool_call_id: str,
    content: Any,
) -> tuple[str, str, str, str]:
    """Return the stable identity used to correlate a tool result across the provider."""
    serialized = serialize_tool_result(content)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return tool_name, result_type, tool_call_id, digest


def redact_tool_result(
    value: Any, rules: CompiledRuleSet | None
) -> tuple[Any, list[tuple[ToolMessage, ToolResultEvent]]]:
    """Sanitize ToolMessages in a tool return and collect their safe event projections."""
    from langgraph.types import Command

    events: list[tuple[ToolMessage, ToolResultEvent]] = []

    def redact_structure(value: Any) -> tuple[Any, dict[str, int]]:
        counts: dict[str, int] = {}

        def visit(item: Any) -> tuple[Any, bool]:
            if isinstance(item, str):
                if rules is None:
                    return item, False
                result = Redactor.redact(item, rules)
                for name, count in result.counts.items():
                    counts[name] = counts.get(name, 0) + count
                return result.text, result.redacted
            if isinstance(item, dict):
                sanitized: dict[Any, Any] = {}
                changed = False
                for key, child in item.items():
                    safe_key, key_changed = visit(key) if isinstance(key, str) else (key, False)
                    safe_child, child_changed = visit(child)
                    if safe_key in sanitized:
                        raise ValueError("redaction produced duplicate mapping keys")
                    sanitized[safe_key] = safe_child
                    changed = changed or key_changed or child_changed
                return (sanitized, True) if changed else (item, False)
            if isinstance(item, list):
                sanitized_items = [visit(child) for child in item]
                if any(changed for _, changed in sanitized_items):
                    return [child for child, _ in sanitized_items], True
                return item, False
            if isinstance(item, tuple):
                sanitized_items = [visit(child) for child in item]
                if any(changed for _, changed in sanitized_items):
                    return tuple(child for child, _ in sanitized_items), True
                return item, False
            if hasattr(item, "model_dump"):
                model_fields = getattr(type(item), "model_fields", None)
                model_copy = getattr(item, "model_copy", None)
                model_extra = getattr(item, "model_extra", None) or {}
                computed_fields = getattr(type(item), "model_computed_fields", None)
                if isinstance(model_fields, dict) and callable(model_copy) and not computed_fields:
                    model_keys = (*model_fields, *model_extra)
                    keys_need_normalization = rules is not None and any(
                        isinstance(key, str) and Redactor.redact(key, rules).redacted
                        for key in model_keys
                    )
                    if not keys_need_normalization:
                        updates: dict[str, Any] = {}
                        for name in model_fields:
                            safe_value, changed = visit(getattr(item, name))
                            if changed:
                                updates[name] = safe_value
                        for name, value in model_extra.items():
                            safe_value, changed = visit(value)
                            if changed:
                                updates[name] = safe_value
                        if updates:
                            return model_copy(update=updates), True
                        return item, False

                sanitized, changed = visit(item.model_dump(exclude_none=False))
                return (sanitized, True) if changed else (item, False)
            return item, False

        sanitized, _ = visit(value)
        return sanitized, counts

    def transform(item: Any) -> tuple[Any, bool]:
        if isinstance(item, ToolMessage):
            try:
                content, counts = redact_structure(item.content)
                output = _stringify_tool_output(content)
            except Exception:
                raise ToolResultSafetyInspectionFailed() from None

            redacted_count = sum(counts.values())
            message = item.model_copy(update={"content": content}) if redacted_count else item
            events.append(
                (
                    message,
                    ToolResultEvent(
                        output=output,
                        call_id=item.tool_call_id or "",
                        error_type="tool_error" if item.status == "error" else None,
                        redacted=redacted_count > 0,
                        redacted_count=redacted_count,
                        redaction_counts=counts,
                    ),
                )
            )
            return message, message is not item

        if isinstance(item, Command):
            update, changed = transform(item.update)
            return (replace(item, update=update) if changed else item), changed
        if isinstance(item, dict):
            dict_items = {key: transform(child) for key, child in item.items()}
            return (
                {key: child for key, (child, _) in dict_items.items()},
                any(changed for _, changed in dict_items.values()),
            )
        if isinstance(item, list):
            list_items = [transform(child) for child in item]
            return [child for child, _ in list_items], any(changed for _, changed in list_items)
        if isinstance(item, tuple):
            tuple_items = [transform(child) for child in item]
            return tuple(child for child, _ in tuple_items), any(
                changed for _, changed in tuple_items
            )
        return item, False

    sanitized, _ = transform(value)
    return sanitized, events


class ToolResultInspector(Protocol):
    def __call__(
        self,
        tool_name: str,
        result_type: str,
        value: Any,
        tool_call_id: str,
    ) -> Awaitable[Any]: ...


class ToolResultInspectionMiddleware(AgentMiddleware[Any, Any, Any]):
    """Redact tool returns and inspect results before model calls."""

    def __init__(
        self,
        inspector: ToolResultInspector | None = None,
        *,
        on_accepted: Callable[[Sequence[ToolMessage]], None] | None = None,
        redaction_rules: CompiledRuleSet | None = None,
        on_tool_result: Callable[[ToolMessage, ToolResultEvent], None] | None = None,
    ) -> None:
        self._inspector = inspector
        self._on_accepted = on_accepted
        self._redaction_rules = redaction_rules
        self._on_tool_result = on_tool_result
        self._classified_signatures: set[tuple[str, str, str, str]] = set()

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        """Redact each effective tool result before it enters graph state."""
        try:
            result = await handler(request)
        except (ToolResultSafetyInspectionFailed, ToolExecutionError):
            raise
        except Exception as exc:
            raise ToolExecutionError(
                tool_name=request.tool_call["name"], error_type=type(exc).__name__
            ) from None

        result, redacted_results = redact_tool_result(result, self._redaction_rules)
        try:
            for message, event in redacted_results:
                if event.redacted:
                    logger.info(
                        "tool_output_redaction tool=%s redacted_count=%d rule_counts=%s",
                        message.name or "unknown",
                        event.redacted_count,
                        sorted(event.redaction_counts.items()),
                    )
                if self._on_tool_result is not None:
                    self._on_tool_result(message, event)
        except Exception:
            raise ToolResultSafetyInspectionFailed() from None
        return result

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        untrusted_results = {
            id(message) for message in filter_untrusted_tool_results(request.messages)
        }
        for message in request.messages:
            if not isinstance(message, ToolMessage) or self._inspector is None:
                continue

            tool_name = message.name or ""
            result_type = "error" if message.status == "error" else "result"
            try:
                signature = tool_result_signature(
                    tool_name,
                    result_type,
                    message.tool_call_id or "",
                    message.content,
                )
            except (TypeError, ValueError):
                raise ToolResultSafetyInspectionFailed() from None
            if id(message) not in untrusted_results:
                continue
            if signature in self._classified_signatures:
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
            self._classified_signatures.add(signature)

        model_messages: list[Any] = []
        wrapped_by_identity: dict[int, ToolMessage] = {}
        for message in request.messages:
            if (
                not isinstance(message, ToolMessage)
                or not message.name
                or id(message) not in untrusted_results
            ):
                model_messages.append(message)
                continue

            identity = id(message)
            model_message = wrapped_by_identity.get(identity)
            if model_message is None:
                model_message = message.model_copy(
                    update={"content": _wrap_tool_output(message.content, message.name)}
                )
                wrapped_by_identity[identity] = model_message
            model_messages.append(model_message)

        if self._on_accepted is not None:
            self._on_accepted(
                tuple(message for message in request.messages if isinstance(message, ToolMessage))
            )
        return await handler(request.override(messages=model_messages))
