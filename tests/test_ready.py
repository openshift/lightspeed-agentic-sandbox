"""Tests for GET /ready readiness endpoint."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from lightspeed_agentic.config import McpServerConfig, ResolvedSDK
from lightspeed_agentic.health import (
    check_mcp_endpoints,
    check_provider_endpoint,
    check_provider_env,
    probe_provider_endpoint,
    register_ready_route,
    run_readiness_checks,
)

_ANTHROPIC_DIRECT = ResolvedSDK(
    "claude",
    ("ANTHROPIC_API_KEY",),
    "https://api.anthropic.com/",
)

_VERTEX_ANTHROPIC = ResolvedSDK(
    "claude",
    ("GOOGLE_APPLICATION_CREDENTIALS",),
    "https://us-east5-aiplatform.googleapis.com/",
)

_VERTEX_GOOGLE = ResolvedSDK(
    "gemini",
    ("GOOGLE_APPLICATION_CREDENTIALS",),
    "https://us-central1-aiplatform.googleapis.com/",
)

_OPENAI_DIRECT = ResolvedSDK(
    "openai",
    ("OPENAI_API_KEY",),
    "https://api.openai.com/",
)

_BEDROCK = ResolvedSDK(
    "claude",
    ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
    "https://bedrock-runtime.us-east-1.amazonaws.com/",
)


# --- R1: credential env checks ---


def test_check_provider_env_anthropic_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert check_provider_env(_ANTHROPIC_DIRECT.expected_envs) == "ok"


def test_check_provider_env_anthropic_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "error: missing" in check_provider_env(_ANTHROPIC_DIRECT.expected_envs)


def test_check_provider_env_anthropic_wrong_cred_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct Anthropic with only GOOGLE_APPLICATION_CREDENTIALS must fail."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/some/path")
    assert "error: missing" in check_provider_env(_ANTHROPIC_DIRECT.expected_envs)


def test_check_provider_env_vertex_anthropic_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "/var/run/secrets/llm-credentials/GOOGLE_APPLICATION_CREDENTIALS",
    )
    assert check_provider_env(_VERTEX_ANTHROPIC.expected_envs) == "ok"


def test_check_provider_env_vertex_anthropic_wrong_cred_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vertex/Anthropic with only ANTHROPIC_API_KEY must fail."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert "error: missing" in check_provider_env(_VERTEX_ANTHROPIC.expected_envs)


def test_check_provider_env_vertex_google_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "/var/run/secrets/llm-credentials/GOOGLE_APPLICATION_CREDENTIALS",
    )
    assert check_provider_env(_VERTEX_GOOGLE.expected_envs) == "ok"


def test_check_provider_env_vertex_google_wrong_cred_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vertex/Google with only GOOGLE_API_KEY must fail."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert "error: missing" in check_provider_env(_VERTEX_GOOGLE.expected_envs)


def test_check_provider_env_openai_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert check_provider_env(_OPENAI_DIRECT.expected_envs) == "ok"


def test_check_provider_env_openai_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "error: missing" in check_provider_env(_OPENAI_DIRECT.expected_envs)


def test_check_provider_env_bedrock_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA...")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    assert check_provider_env(_BEDROCK.expected_envs) == "ok"


def test_check_provider_env_bedrock_partial_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bedrock with only one of the two required AWS vars must fail."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA...")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    assert "error: missing" in check_provider_env(_BEDROCK.expected_envs)


# --- R2: endpoint reachability ---


def test_probe_provider_endpoint_http_error_is_ok() -> None:
    with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("", 403, "", {}, None)):
        assert probe_provider_endpoint("https://api.anthropic.com/") == "ok"


def test_probe_provider_endpoint_connection_error() -> None:
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("connection refused")):
        result = probe_provider_endpoint("https://api.anthropic.com/")
    assert result.startswith("error: ")


def test_check_provider_endpoint_uses_probe_url() -> None:
    custom = ResolvedSDK("openai", ("OPENAI_API_KEY",), "https://custom.example/v1")
    with patch(
        "lightspeed_agentic.health.probe_provider_endpoint",
        return_value="ok",
    ) as mock_probe:
        assert check_provider_endpoint(custom.probe_url) == "ok"
    mock_probe.assert_called_once_with("https://custom.example/v1")


def test_check_provider_endpoint_vertex_url() -> None:
    with patch(
        "lightspeed_agentic.health.probe_provider_endpoint",
        return_value="ok",
    ) as mock_probe:
        assert check_provider_endpoint(_VERTEX_ANTHROPIC.probe_url) == "ok"
    mock_probe.assert_called_once_with("https://us-east5-aiplatform.googleapis.com/")


def test_check_provider_endpoint_empty_url() -> None:
    empty = ResolvedSDK("openai", ("OPENAI_API_KEY",), "")
    assert check_provider_endpoint(empty.probe_url) == "error: empty probe URL"


# --- Full readiness route ---


@pytest.mark.asyncio
async def test_ready_route_all_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    app = FastAPI()
    register_ready_route(app, sdk=_ANTHROPIC_DIRECT)
    with patch(
        "lightspeed_agentic.health.probe_provider_endpoint",
        return_value="ok",
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_ready_route_provider_env_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = FastAPI()
    register_ready_route(app, sdk=_ANTHROPIC_DIRECT)
    with patch(
        "lightspeed_agentic.health.probe_provider_endpoint",
        return_value="ok",
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "error"
    assert body["checks"]["provider_env"].startswith("error: ")
    assert body["checks"]["provider_endpoint"] == "ok"


def test_run_readiness_checks_all_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with patch("lightspeed_agentic.health.probe_provider_endpoint", return_value="ok"):
        ok, checks = run_readiness_checks(_OPENAI_DIRECT)
    assert ok is True
    assert checks == {"provider_env": "ok", "provider_endpoint": "ok"}


# --- R3: MCP endpoint reachability ---

_MCP_SERVERS = (
    McpServerConfig(name="jira", url="https://mcp-jira.example.com/sse"),
    McpServerConfig(name="github", url="https://mcp-github.example.com/"),
)


def test_check_mcp_endpoints_all_ok() -> None:
    with patch("lightspeed_agentic.health.probe_provider_endpoint", return_value="ok"):
        results = check_mcp_endpoints(_MCP_SERVERS)
    assert results == {"mcp_jira": "ok", "mcp_github": "ok"}


def test_check_mcp_endpoints_one_failing() -> None:
    def _side_effect(url: str, *_: object) -> str:
        if "jira" in url:
            return "error: connection refused"
        return "ok"

    with patch("lightspeed_agentic.health.probe_provider_endpoint", side_effect=_side_effect):
        results = check_mcp_endpoints(_MCP_SERVERS)
    assert results["mcp_jira"].startswith("error: ")
    assert results["mcp_github"] == "ok"


def test_check_mcp_endpoints_empty() -> None:
    assert check_mcp_endpoints(()) == {}


def test_run_readiness_checks_with_mcp_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    sdk_with_mcp = ResolvedSDK(
        "openai",
        ("OPENAI_API_KEY",),
        "https://api.openai.com/",
        mcp_servers=_MCP_SERVERS,
    )
    with patch("lightspeed_agentic.health.probe_provider_endpoint", return_value="ok"):
        ok, checks = run_readiness_checks(sdk_with_mcp)
    assert ok is True
    assert "mcp_jira" in checks
    assert "mcp_github" in checks


def test_run_readiness_checks_with_mcp_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    sdk_with_mcp = ResolvedSDK(
        "openai",
        ("OPENAI_API_KEY",),
        "https://api.openai.com/",
        mcp_servers=_MCP_SERVERS,
    )

    def _side_effect(url: str, *_: object) -> str:
        if "jira" in url:
            return "error: timeout"
        return "ok"

    with patch("lightspeed_agentic.health.probe_provider_endpoint", side_effect=_side_effect):
        ok, checks = run_readiness_checks(sdk_with_mcp)
    assert ok is False
    assert checks["mcp_jira"].startswith("error: ")


@pytest.mark.asyncio
async def test_ready_route_with_mcp_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    sdk_with_mcp = ResolvedSDK(
        "openai",
        ("OPENAI_API_KEY",),
        "https://api.openai.com/",
        mcp_servers=(_MCP_SERVERS[0],),
    )

    def _fail_mcp(url: str, *_: object) -> str:
        if "jira" in url:
            return "error: connection refused"
        return "ok"

    app = FastAPI()
    register_ready_route(app, sdk=sdk_with_mcp)
    with patch("lightspeed_agentic.health.probe_provider_endpoint", side_effect=_fail_mcp):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "error"
    assert body["checks"]["mcp_jira"].startswith("error: ")
