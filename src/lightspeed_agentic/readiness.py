"""Readiness checks for provider credentials."""

from __future__ import annotations

import os
from pathlib import Path

from lightspeed_agentic.config import ResolvedModelFamily, ResolvedSDK


def read_mounted_secret(path: Path) -> str | None:
    """Return stripped non-empty UTF-8 file contents.

    Returns ``None`` when the path is not a readable regular file, content is
    empty or whitespace-only, or decoding fails. Shared with MCP header resolution.
    """
    try:
        if not path.is_file():
            return None
        content = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return content or None


def read_first_mounted_secret_in_dir(directory: Path) -> str | None:
    """Return contents of the first regular file in ``directory`` (sorted by name)."""
    try:
        if not directory.is_dir():
            return None
        files = sorted((f for f in directory.iterdir() if f.is_file()), key=lambda f: f.name)
    except OSError:
        return None
    if not files:
        return None
    return read_mounted_secret(files[0])


def check_provider_env(
    expected_envs: tuple[str, ...],
    credential_file_envs: tuple[str, ...] = (),
) -> str:
    """R1: required credential env var(s) present; file paths exist and non-empty."""
    missing = [var for var in expected_envs if not os.environ.get(var, "").strip()]
    if missing:
        return f"error: missing {', '.join(missing)}"

    for var in credential_file_envs:
        path = os.environ.get(var, "").strip()
        if not path:
            continue
        if not os.path.isfile(path):
            return f"error: {var} file not found: {path}"
        content = read_mounted_secret(Path(path))
        if content is None:
            return f"error: {var} file is empty or unreadable: {path}"

    return "ok"


def run_readiness_checks(sdk: ResolvedSDK | ResolvedModelFamily) -> tuple[bool, dict[str, str]]:
    """Return overall readiness and per-check status strings."""
    status = check_provider_env(sdk.expected_envs, sdk.credential_file_envs)
    checks = {"provider_env": status}
    return status == "ok", checks
