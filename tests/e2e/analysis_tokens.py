"""Helpers for operator analysis token placement in batch run bodies."""

from __future__ import annotations

from typing import Any


def component_token_values(raw: dict[str, Any]) -> list[str]:
    """Collect primary/secondary token values from options[].components[]."""
    values: list[str] = []
    options = raw.get("options")
    if not isinstance(options, list):
        return values
    for option in options:
        if not isinstance(option, dict):
            continue
        components = option.get("components")
        if not isinstance(components, list):
            continue
        for component in components:
            if not isinstance(component, dict):
                continue
            tokens = component.get("tokens")
            if not isinstance(tokens, dict):
                continue
            for key in ("primary", "secondary"):
                entry = tokens.get(key)
                if isinstance(entry, dict) and isinstance(entry.get("value"), str):
                    values.append(entry["value"])
    return values


def assert_skill_tokens_in_response(
    raw: dict[str, Any],
    provider_name: str,
    *,
    prefixes: tuple[str, ...] = ("DIAG_", "VERIFY_"),
) -> None:
    """Assert find-token tokens appear under options[].components[] (proves script ran)."""
    token_values = component_token_values(raw)
    assert token_values, (
        f"{provider_name}: no token values under options[].components[] "
        f"(find-token.sh output may be missing or misplaced)"
    )
    for prefix in prefixes:
        assert any(value.startswith(prefix) for value in token_values), (
            f"{provider_name}: expected token prefix {prefix!r} in "
            f"options[].components[].tokens; got {token_values!r}"
        )
