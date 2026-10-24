"""Optional provider SDKs stay out of base batch and provider imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_gemini_route_is_lazy_and_legacy_providers_are_absent() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = r"""
import builtins

blocked = ("deepagents", "langchain", "langgraph", "langchain_google_genai", "agents")
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if any(name == package or name.startswith(package + ".") for package in blocked):
        raise AssertionError(f"optional module import attempted: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import lightspeed_agentic.batch
from lightspeed_agentic.factory import create_provider

for name in ("openai", "gemini"):
    try:
        create_provider(name)
    except ValueError as exc:
        assert "Unknown provider" in str(exc)
    else:
        raise AssertionError(f"legacy provider should not be registered: {name}")

from pathlib import Path
from lightspeed_agentic.config import ResolvedModelFamily
from lightspeed_agentic.config import (
    VertexCredentialSource,
    VertexProviderParameters,
)
from lightspeed_agentic.providers.model_router import resolve_model_route
family = ResolvedModelFamily(
    "google",
    ("GOOGLE_APPLICATION_CREDENTIALS",),
    hosting="vertex",
    model="gemini-2.5-flash",
    provider=VertexProviderParameters(project="project", region="us-central1"),
    credentials=VertexCredentialSource(Path("/unused/credentials.json")),
)
sdk, route = resolve_model_route(model_family=family)
assert sdk is None
assert route is not None and route.route == "vertex/google"
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(  # noqa: S603 - controlled inline import-isolation test script
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
