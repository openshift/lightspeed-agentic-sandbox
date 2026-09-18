"""Tests for Gemini provider skill loading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from lightspeed_agentic.providers.gemini import (  # type: ignore[import-untyped]
    _load_skills_toolset,
    _trim_tool_response,
)
from lightspeed_agentic.types import (  # type: ignore[import-untyped]
    MAX_TOOL_RETURN_CHARS,
    TOOL_RETURN_PREVIEW_CHARS,
)


def test_trim_tool_response_keeps_bounded_result() -> None:
    assert _trim_tool_response(None, {}, None, "small result") is None


def test_trim_tool_response_returns_preview_for_oversized_result() -> None:
    result = _trim_tool_response(None, {}, None, "x" * (MAX_TOOL_RETURN_CHARS + 1))

    assert result["status"] == "truncated"
    assert len(result["preview"]) == TOOL_RETURN_PREVIEW_CHARS
    assert result["original_size"] == MAX_TOOL_RETURN_CHARS + 1


def test_load_skills_toolset_passes_code_executor(tmp_path: Path) -> None:
    (tmp_path / "echo-token").mkdir()
    mock_skill = MagicMock()
    mock_executor = MagicMock()
    mock_toolset = MagicMock()

    with (
        patch(
            "google.adk.skills.list_skills_in_dir",
            return_value=["echo-token"],
        ),
        patch(
            "google.adk.skills.load_skill_from_dir",
            return_value=mock_skill,
        ),
        patch(
            "google.adk.code_executors.unsafe_local_code_executor.UnsafeLocalCodeExecutor",
            return_value=mock_executor,
        ),
        patch(
            "google.adk.tools.skill_toolset.SkillToolset",
            return_value=mock_toolset,
        ) as mock_skill_toolset,
    ):
        result = _load_skills_toolset(str(tmp_path))

    mock_skill_toolset.assert_called_once_with(
        skills=[mock_skill],
        code_executor=mock_executor,
    )
    assert result is mock_toolset
