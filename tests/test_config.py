"""Tests for LIGHTSPEED_* → SDK env var mapping."""

from __future__ import annotations

import json
import os

import pytest

from lightspeed_agentic.config import resolve_mcp_servers, resolve_sdk


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = os.environ.copy()
    _clean_env(monkeypatch)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all LIGHTSPEED_* and SDK-specific vars to isolate tests."""
    for var in [
        "LIGHTSPEED_PROVIDER",
        "LIGHTSPEED_MODEL",
        "LIGHTSPEED_MODEL_PROVIDER",
        "LIGHTSPEED_PROVIDER_URL",
        "LIGHTSPEED_PROVIDER_PROJECT",
        "LIGHTSPEED_PROVIDER_REGION",
        "LIGHTSPEED_PROVIDER_API_VERSION",
        "LIGHTSPEED_LLM_CREDENTIALS_PATH",
        "ANTHROPIC_MODEL",
        "GEMINI_MODEL",
        "OPENAI_MODEL",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_BEDROCK",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "CLOUD_ML_REGION",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AWS_REGION",
        "LIGHTSPEED_MCP_SERVERS",
    ]:
        monkeypatch.delenv(var, raising=False)


def test_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "anthropic")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "claude-sonnet-4-20250514")

    sdk = resolve_sdk()

    assert sdk.name == "claude"
    assert sdk.probe_url == "https://api.anthropic.com/"
    assert os.environ["ANTHROPIC_MODEL"] == "claude-sonnet-4-20250514"


def test_anthropic_with_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "anthropic")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "claude-opus-4-6")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://proxy.example.com")

    resolve_sdk()

    assert os.environ["ANTHROPIC_BASE_URL"] == "https://proxy.example.com"


def test_vertex_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_PROJECT", "my-project")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_REGION", "us-east5")

    sdk = resolve_sdk()

    assert sdk.name == "claude"
    assert os.environ["ANTHROPIC_MODEL"] == "claude-sonnet-4-20250514"
    assert os.environ["CLAUDE_CODE_USE_VERTEX"] == "1"
    assert os.environ["ANTHROPIC_VERTEX_PROJECT_ID"] == "my-project"
    assert os.environ["CLOUD_ML_REGION"] == "us-east5"
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == (
        "/var/run/secrets/llm-credentials/GOOGLE_APPLICATION_CREDENTIALS"
    )
    assert sdk.probe_url == "https://us-east5-aiplatform.googleapis.com/"


def test_vertex_google(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", "google")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_PROJECT", "my-project")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_REGION", "us-central1")

    sdk = resolve_sdk()

    assert sdk.name == "gemini"
    assert os.environ["GEMINI_MODEL"] == "gemini-2.5-flash"
    assert os.environ["GOOGLE_GENAI_USE_VERTEXAI"] == "true"
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "my-project"
    assert os.environ["GOOGLE_CLOUD_LOCATION"] == "us-central1"
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == (
        "/var/run/secrets/llm-credentials/GOOGLE_APPLICATION_CREDENTIALS"
    )


def test_vertex_llm_credentials_path_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    creds_dir = "/var/run/secrets/e2e-llm-credentials"
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", "google")
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", creds_dir)

    resolve_sdk()

    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == (
        f"{creds_dir}/GOOGLE_APPLICATION_CREDENTIALS"
    )


def test_vertex_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "gpt-4.1")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://vertex-openai.example.com")

    sdk = resolve_sdk()

    assert sdk.name == "openai"
    assert os.environ["OPENAI_MODEL"] == "gpt-4.1"
    assert os.environ["OPENAI_BASE_URL"] == "https://vertex-openai.example.com"
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == (
        "/var/run/secrets/llm-credentials/GOOGLE_APPLICATION_CREDENTIALS"
    )


def test_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "openai")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "gpt-4.1")

    sdk = resolve_sdk()

    assert sdk.name == "openai"
    assert os.environ["OPENAI_MODEL"] == "gpt-4.1"


def test_openai_with_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "openai")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "gpt-4.1")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://custom.openai.com/v1")

    resolve_sdk()

    assert os.environ["OPENAI_BASE_URL"] == "https://custom.openai.com/v1"


def test_azure(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "azure")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "gpt-4.1")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://my-resource.openai.azure.com")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_API_VERSION", "2024-08-01-preview")

    sdk = resolve_sdk()

    assert sdk.name == "openai"
    assert os.environ["OPENAI_MODEL"] == "gpt-4.1"
    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://my-resource.openai.azure.com"
    assert os.environ["AZURE_OPENAI_API_VERSION"] == "2024-08-01-preview"


def test_bedrock(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "bedrock")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_REGION", "us-east-1")

    sdk = resolve_sdk()

    assert sdk.name == "claude"
    assert os.environ["ANTHROPIC_MODEL"] == "claude-sonnet-4-20250514"
    assert os.environ["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert os.environ["AWS_REGION"] == "us-east-1"


def test_default_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)

    sdk = resolve_sdk()

    assert sdk.name == "claude"


def test_default_model_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "anthropic")

    resolve_sdk()

    assert "ANTHROPIC_MODEL" not in os.environ


def test_vertex_missing_model_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "some-model")

    with pytest.raises(ValueError, match="LIGHTSPEED_MODEL_PROVIDER"):
        resolve_sdk()


def test_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "watsonx")

    with pytest.raises(ValueError, match="Unknown provider"):
        resolve_sdk()


# --- MCP server resolution ---


def test_mcp_servers_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    assert resolve_mcp_servers() == ()


def test_mcp_servers_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_MCP_SERVERS", "  ")
    assert resolve_mcp_servers() == ()


def test_mcp_servers_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv(
        "LIGHTSPEED_MCP_SERVERS",
        '[{"url": "https://mcp.example.com/sse", "name": "jira"},'
        ' {"url": "https://mcp2.example.com/", "name": "github", "headers": {"x-api-key": "k"}}]',
    )
    servers = resolve_mcp_servers()
    assert len(servers) == 2
    assert servers[0].name == "jira"
    assert servers[0].url == "https://mcp.example.com/sse"
    assert servers[0].headers == {}
    assert servers[1].name == "github"
    assert servers[1].headers == {"x-api-key": "k"}


def test_mcp_servers_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_MCP_SERVERS", "not json")
    with pytest.raises(json.JSONDecodeError):
        resolve_mcp_servers()


def test_mcp_servers_not_array(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_MCP_SERVERS", '{"url": "x", "name": "y"}')
    with pytest.raises(ValueError, match="must be a JSON array"):
        resolve_mcp_servers()


def test_mcp_servers_missing_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_MCP_SERVERS", '[{"url": "https://x.com"}]')
    with pytest.raises(ValueError, match="missing required 'name'"):
        resolve_mcp_servers()


def test_mcp_servers_missing_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_MCP_SERVERS", '[{"name": "test"}]')
    with pytest.raises(ValueError, match="missing required 'url'"):
        resolve_mcp_servers()


def test_resolve_sdk_populates_mcp_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "anthropic")
    monkeypatch.setenv(
        "LIGHTSPEED_MCP_SERVERS",
        '[{"url": "https://mcp.example.com", "name": "test"}]',
    )
    sdk = resolve_sdk()
    assert len(sdk.mcp_servers) == 1
    assert sdk.mcp_servers[0].name == "test"


def test_resolve_sdk_no_mcp_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "anthropic")
    sdk = resolve_sdk()
    assert sdk.mcp_servers == ()
