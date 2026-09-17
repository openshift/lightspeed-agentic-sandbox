"""Tests for gen-build-deps exact-pin conflict handling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen-build-deps.py"
_spec = importlib.util.spec_from_file_location("gen_build_deps", _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
_gen_build_deps = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _gen_build_deps
_spec.loader.exec_module(_gen_build_deps)


def test_build_replace_pins_single_replacement() -> None:
    pins = {"hatchling": {"1.26.3", "1.32.0"}}
    resolved = {"hatchling": "1.32.0"}
    assert _gen_build_deps._build_replace_pins(pins, resolved) == {"hatchling": ["1.26.3"]}


def test_build_replace_pins_no_replacement_when_matches_resolution() -> None:
    pins = {"hatchling": {"1.32.0"}}
    resolved = {"hatchling": "1.32.0"}
    assert _gen_build_deps._build_replace_pins(pins, resolved) == {}


def test_build_replace_pins_multiple_versions() -> None:
    pins = {"hatchling": {"1.26.3", "1.27.0"}}
    resolved = {"hatchling": "1.32.0"}
    result = _gen_build_deps._build_replace_pins(pins, resolved)
    assert result == {"hatchling": ["1.26.3", "1.27.0"]}
