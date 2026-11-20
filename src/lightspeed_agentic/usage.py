"""Invocation-wide token usage accounting for sandbox model calls.

One :class:`InvocationUsage` collector is created per query and attached to every
callback config the query runs, so the totals survive a timeout, cancellation or
provider exception. Reasoning is complete-or-unset: it is only reported when every
call that reported usage also reported a reasoning count.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

REASONING_DETAIL_SUFFIX = "reasoning"


@dataclass(frozen=True, slots=True)
class MainCallSnapshot:
    """One main-agent model call, retained for tool/peak derivation."""

    messages: tuple[Any, ...]
    model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None


def message_from_response(response: Any) -> Any:
    """Return the first generation's message from an ``LLMResult``."""
    generations = getattr(response, "generations", None)
    if not isinstance(generations, list) or not generations:
        return None
    first = generations[0]
    if isinstance(first, list):
        if not first:
            return None
        first = first[0]
    return getattr(first, "message", None)


def model_name_from_message(message: Any, fallback: str) -> str:
    """Observed response model, falling back to the configured route model."""
    metadata = getattr(message, "response_metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("model_name", "model"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    return fallback


def reasoning_tokens_from_usage(usage: Mapping[str, Any]) -> int | None:
    """Sum reasoning ``output_token_details`` values; ``None`` when not reported."""
    details = usage.get("output_token_details")
    if not isinstance(details, Mapping):
        return None
    total: int | None = None
    for key, value in details.items():
        if (
            isinstance(key, str)
            and key.endswith(REASONING_DETAIL_SUFFIX)
            and isinstance(value, int)
        ):
            total = (total or 0) + value
    return total


@dataclass(frozen=True, slots=True)
class QueryUsage:
    """Usage summary published for one query.

    Every counter is ``None`` when it was never observed: a counter that no call
    reported is omitted from the Result CR instead of being published as ``0``.
    """

    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    tool_tokens: int | None
    peak_input_tokens: int | None
    model: str


def summarize_usage(
    usage: InvocationUsage,
    calls: Sequence[MainCallSnapshot],
    route_model: str,
) -> QueryUsage:
    """Collector totals plus main-agent estimates and the model to publish.

    A query whose calls never reported usage yields no counters at all: the
    caller publishes a Result CR without ``tokenUsage`` rather than zeros.
    """
    reported = usage.usage_call_count > 0
    return QueryUsage(
        input_tokens=usage.input_tokens if reported else None,
        output_tokens=usage.output_tokens if reported else None,
        reasoning_tokens=usage.reasoning_tokens,
        tool_tokens=tool_tokens_from_main_calls(calls),
        peak_input_tokens=peak_input_tokens_from_main_calls(calls),
        model=_published_model(calls, usage, route_model),
    )


def _published_model(
    calls: Sequence[MainCallSnapshot],
    usage: InvocationUsage,
    route_model: str,
) -> str:
    """Prefer the main-agent response model; fall back to any observed model.

    One invocation runs one model, so differing observed strings are provider
    identity variants of that model rather than a mixed-model run; no warning.
    """
    if calls:
        return calls[-1].model
    return usage.observed_model or route_model


def tool_tokens_from_main_calls(calls: Sequence[MainCallSnapshot]) -> int | None:
    """Estimated input tokens contributed by tool results to main-agent calls.

    Skill content is read through tools, so skill tokens are included here; there
    is deliberately no separate skill-token estimate. ``None`` when there was no
    main-agent call to derive the estimate from.

    The estimate is a lower bound: a pair whose history was not extended in place
    (compaction or summarization rewrote it) cannot be attributed, so its growth
    is not counted and the omission is logged. The whole estimate is unset when
    an attributable pair lacks reasoning usage or produces a negative delta,
    because either case makes subtraction unreliable.
    """
    if not calls:
        return None
    total = 0
    for previous, current in pairwise(calls):
        attributable, delta = _appended_tool_delta(previous, current)
        if not attributable:
            if len(current.messages) > len(previous.messages):
                logger.debug(
                    "tool token estimate skipped a call pair: history was rewritten "
                    "rather than extended (previous=%d current=%d messages)",
                    len(previous.messages),
                    len(current.messages),
                )
            continue
        if delta is None:
            return None
        total += delta
    return total


def peak_input_tokens_from_main_calls(calls: Sequence[MainCallSnapshot]) -> int | None:
    """Largest single main-agent call input, or ``None`` when there were no calls."""
    return max((call.input_tokens for call in calls), default=None)


def _appended_tool_delta(
    previous: MainCallSnapshot, current: MainCallSnapshot
) -> tuple[bool, int | None]:
    """Whether a pair is attributable and its tool-input estimate when reliable."""
    prefix = len(previous.messages)
    if len(current.messages) <= prefix:
        return False, None
    if tuple(current.messages[:prefix]) != previous.messages:
        return False, None
    suffix = current.messages[prefix:]
    if len(suffix) < 2 or getattr(suffix[0], "type", None) != "ai":
        return False, None
    if any(getattr(message, "type", None) != "tool" for message in suffix[1:]):
        return False, None
    if previous.reasoning_tokens is None:
        return True, None
    growth = (
        current.input_tokens
        - previous.input_tokens
        - (previous.output_tokens - previous.reasoning_tokens)
    )
    if growth < 0:
        return True, None
    return True, min(growth, current.input_tokens)


class InvocationUsage(BaseCallbackHandler):
    """Per-invocation usage totals across every model call made by the query."""

    def __init__(self, *, route_model: str) -> None:
        self._route_model = route_model
        self._lock = threading.Lock()
        self._recorded_run_ids: set[Any] = set()
        self._input_tokens = 0
        self._output_tokens = 0
        self._usage_calls = 0
        self._reasoning_total = 0
        self._reasoning_complete = True
        self._observed_model: str | None = None
        self.main_calls: list[MainCallSnapshot] = []

    def record_main_call(self, messages: tuple[Any, ...], response: Any) -> None:
        """Retain one main-agent call for tool and peak-input derivation."""
        message = message_from_response(response)
        usage = getattr(message, "usage_metadata", None)
        if not isinstance(usage, Mapping):
            return
        snapshot = MainCallSnapshot(
            messages=messages,
            model=model_name_from_message(message, self._route_model),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            reasoning_tokens=reasoning_tokens_from_usage(usage),
        )
        with self._lock:
            self.main_calls.append(snapshot)

    @property
    def input_tokens(self) -> int:
        with self._lock:
            return self._input_tokens

    @property
    def output_tokens(self) -> int:
        with self._lock:
            return self._output_tokens

    @property
    def reasoning_tokens(self) -> int | None:
        """Reasoning sum, ``None`` when no call or any call omitted it."""
        with self._lock:
            if self._usage_calls == 0:
                return None
            return self._reasoning_total if self._reasoning_complete else None

    @property
    def usage_call_count(self) -> int:
        with self._lock:
            return self._usage_calls

    @property
    def observed_model(self) -> str | None:
        with self._lock:
            return self._observed_model

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Record one completed call; calls without usage are ignored."""
        self._record_response(response, run_id=kwargs.get("run_id"))

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Record usage returned with a failed or partial model response."""
        del error
        self._record_response(kwargs.get("response"), run_id=kwargs.get("run_id"))

    def _record_response(self, response: Any, *, run_id: Any) -> None:
        """Record response usage once per callback run ID."""
        message = message_from_response(response)
        usage = getattr(message, "usage_metadata", None)
        if not isinstance(usage, Mapping):
            return
        model = model_name_from_message(message, self._route_model)
        reasoning = reasoning_tokens_from_usage(usage)
        with self._lock:
            if run_id is not None and run_id in self._recorded_run_ids:
                return
            if run_id is not None:
                self._recorded_run_ids.add(run_id)
            self._input_tokens += int(usage.get("input_tokens", 0) or 0)
            self._output_tokens += int(usage.get("output_tokens", 0) or 0)
            self._usage_calls += 1
            self._observed_model = model
            if reasoning is None:
                self._reasoning_complete = False
            else:
                self._reasoning_total += reasoning
