"""SDK-independent value access and argument decoding for telemetry adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

_MISSING = object()


def _mapping(value: Any, **dump_options: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(**dump_options)
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    attribute = getattr(value, name, _MISSING)
    if attribute is not _MISSING:
        return attribute
    mapped = _mapping(value)
    return mapped.get(name, default) if mapped is not None else default


def _json_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
