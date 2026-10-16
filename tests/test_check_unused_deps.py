"""Regression tests for the unused-dependency audit exclusion patterns.

Deptry 0.24 prepends ``^`` to the combined exclusion regex, so ``(^|/)`` can
never match past a parent-directory prefix. The exclusion patterns in
``pyproject.toml`` must keep imports inside nested ``venv`` directories and
nested ``setup.py`` files from masking an otherwise unused dependency.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

DEPTRY_ARGS = ["tool", "run", "deptry==0.24.0", "."]


def deptry_section() -> str:
    """Extract the [tool.deptry] config verbatim from the real pyproject."""
    text = (ROOT / "pyproject.toml").read_text()
    start = text.index("[tool.deptry]\n")
    end = text.index("\n[tool.", start + 1)
    return text[start:end]


def make_repo(tmp_path: Path, masked: str | None) -> Path:
    """Create a repository whose only 'requests' import is in a masked path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "audit-probe"\nversion = "0.0.0"\n'
        'dependencies = ["requests"]\n\n' + deptry_section() + "\n"
    )
    (repo / "src").mkdir()
    (repo / "src/tool.py").write_text("# production code without imports\n")
    if masked:
        path = repo / masked
        path.parent.mkdir(parents=True)
        path.write_text("import requests\n")
    return repo


def run_deptry(repo: Path) -> subprocess.CompletedProcess[str]:
    """Run the pinned deptry command exactly as the Makefile does."""
    return subprocess.run(  # noqa: S603 - controlled pinned audit command
        ["uv", *DEPTRY_ARGS],  # noqa: S607 - resolved from PATH like the Makefile
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
class TestNestedExclusions:
    """Nested environment paths must not mask unused dependencies."""

    @pytest.mark.parametrize("masked", ["nested/venv/probe.py", "nested/setup.py"])
    def test_nested_masked_import_reports_dep002(self, tmp_path: Path, masked: str) -> None:
        """Report DEP002 when the only import lives in a masked path."""
        result = run_deptry(make_repo(tmp_path, masked))
        assert result.returncode != 0, result.stdout + result.stderr
        assert "DEP002" in result.stdout + result.stderr

    def test_used_dependency_still_passes(self, tmp_path: Path) -> None:
        """Keep the audit green when the dependency is genuinely used."""
        repo = make_repo(tmp_path, None)
        (repo / "src/tool.py").write_text("import requests\n")
        result = run_deptry(repo)
        assert result.returncode == 0, result.stdout + result.stderr

    def test_stdlib_backport_dependency_does_not_report_dep005(self, tmp_path: Path) -> None:
        """Keep the audit DEP002-only; DEP005 stays disabled."""
        repo = make_repo(tmp_path, None)
        (repo / "pyproject.toml").write_text(
            (repo / "pyproject.toml")
            .read_text()
            .replace('dependencies = ["requests"]', 'dependencies = ["typing"]')
        )
        (repo / "src/tool.py").write_text("import typing\n")
        result = run_deptry(repo)
        assert result.returncode == 0, result.stdout + result.stderr
