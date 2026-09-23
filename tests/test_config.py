"""Tests for LIGHTSPEED_* → SDK env var mapping."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lightspeed_agentic.config import (
    parse_agent_timeout,
    parse_max_turns,
    parse_reasoning_config,
    resolve_sdk,
)


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
        "AZURE_OPENAI_API_KEY",
        "AWS_REGION",
    ]:
        monkeypatch.delenv(var, raising=False)


def test_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "anthropic")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "claude-sonnet-4-20250514")

    sdk = resolve_sdk()

    assert sdk.name == "deepagents"
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

    assert sdk.name == "deepagents"
    assert os.environ["ANTHROPIC_MODEL"] == "claude-sonnet-4-20250514"
    assert os.environ["CLAUDE_CODE_USE_VERTEX"] == "1"
    assert os.environ["ANTHROPIC_VERTEX_PROJECT_ID"] == "my-project"
    assert os.environ["CLOUD_ML_REGION"] == "us-east5"
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == (
        "/var/run/secrets/llm-credentials/GOOGLE_APPLICATION_CREDENTIALS"
    )


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
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

    sdk = resolve_sdk()

    assert sdk.name == "openai"
    assert sdk.azure_auth_mode == "api_key"
    assert os.environ["OPENAI_MODEL"] == "gpt-4.1"
    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://my-resource.openai.azure.com"
    assert os.environ["AZURE_OPENAI_API_VERSION"] == "2024-08-01-preview"


def test_bedrock(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "bedrock")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_REGION", "us-east-1")

    sdk = resolve_sdk()

    assert sdk.name == "deepagents"
    assert os.environ["ANTHROPIC_MODEL"] == "claude-sonnet-4-20250514"
    assert os.environ["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert os.environ["AWS_REGION"] == "us-east-1"


def test_default_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)

    sdk = resolve_sdk()

    assert sdk.name == "deepagents"


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


# --- parse_reasoning_config tests ---


def test_reasoning_config_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIGHTSPEED_REASONING_CONFIG", raising=False)
    assert parse_reasoning_config() is None


def test_reasoning_config_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_REASONING_CONFIG", "")
    assert parse_reasoning_config() is None


def test_reasoning_config_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_REASONING_CONFIG", "   ")
    assert parse_reasoning_config() is None


def test_reasoning_config_valid_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "LIGHTSPEED_REASONING_CONFIG",
        '{"effort": "high", "thinking": {"type": "enabled"}}',
    )
    result = parse_reasoning_config()
    assert result == {"effort": "high", "thinking": {"type": "enabled"}}


def test_reasoning_config_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_REASONING_CONFIG", "{not valid json")
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_reasoning_config()


def test_reasoning_config_array_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_REASONING_CONFIG", '["not", "an", "object"]')
    with pytest.raises(ValueError, match="must be a JSON object, got list"):
        parse_reasoning_config()


def test_reasoning_config_string_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_REASONING_CONFIG", '"just a string"')
    with pytest.raises(ValueError, match="must be a JSON object, got str"):
        parse_reasoning_config()


def test_reasoning_config_number_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_REASONING_CONFIG", "42")
    with pytest.raises(ValueError, match="must be a JSON object, got int"):
        parse_reasoning_config()


# --- parse_agent_timeout tests ---


def test_agent_timeout_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIGHTSPEED_AGENT_TIMEOUT_SECONDS", raising=False)
    with pytest.raises(
        ValueError, match="LIGHTSPEED_AGENT_TIMEOUT_SECONDS is required but not set"
    ):
        parse_agent_timeout()


def test_agent_timeout_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_TIMEOUT_SECONDS", "")
    with pytest.raises(
        ValueError, match="LIGHTSPEED_AGENT_TIMEOUT_SECONDS is required but not set"
    ):
        parse_agent_timeout()


def test_agent_timeout_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_TIMEOUT_SECONDS", "   ")
    with pytest.raises(
        ValueError, match="LIGHTSPEED_AGENT_TIMEOUT_SECONDS is required but not set"
    ):
        parse_agent_timeout()


def test_agent_timeout_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_TIMEOUT_SECONDS", "not_a_number")
    with pytest.raises(ValueError, match="must be a positive integer"):
        parse_agent_timeout()


def test_agent_timeout_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        parse_agent_timeout()


def test_agent_timeout_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_TIMEOUT_SECONDS", "-10")
    with pytest.raises(ValueError, match="must be positive"):
        parse_agent_timeout()


def test_agent_timeout_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_TIMEOUT_SECONDS", "300")
    result = parse_agent_timeout()
    assert result == 300


def test_agent_timeout_large_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_TIMEOUT_SECONDS", "3600")
    result = parse_agent_timeout()
    assert result == 3600


# --- parse_max_turns tests ---


def test_max_turns_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIGHTSPEED_AGENT_MAX_TURNS", raising=False)
    with pytest.raises(ValueError, match="LIGHTSPEED_AGENT_MAX_TURNS is required but not set"):
        parse_max_turns()


def test_max_turns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_MAX_TURNS", "")
    with pytest.raises(ValueError, match="LIGHTSPEED_AGENT_MAX_TURNS is required but not set"):
        parse_max_turns()


def test_max_turns_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_MAX_TURNS", "   ")
    with pytest.raises(ValueError, match="LIGHTSPEED_AGENT_MAX_TURNS is required but not set"):
        parse_max_turns()


def test_max_turns_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_MAX_TURNS", "not_a_number")
    with pytest.raises(ValueError, match="must be an integer"):
        parse_max_turns()


def test_max_turns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_MAX_TURNS", "0")
    with pytest.raises(ValueError, match="must be between 1 and 500"):
        parse_max_turns()


def test_max_turns_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_MAX_TURNS", "-5")
    with pytest.raises(ValueError, match="must be between 1 and 500"):
        parse_max_turns()


def test_max_turns_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_MAX_TURNS", "501")
    with pytest.raises(ValueError, match="must be between 1 and 500"):
        parse_max_turns()


def test_max_turns_valid_min(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_MAX_TURNS", "1")
    result = parse_max_turns()
    assert result == 1


def test_max_turns_valid_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_MAX_TURNS", "500")
    result = parse_max_turns()
    assert result == 500


def test_max_turns_valid_middle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_MAX_TURNS", "10")
    result = parse_max_turns()
    assert result == 10


# ── Azure Entra ID credential resolution ──


def _setup_azure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the common LIGHTSPEED_* vars for Azure tests."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "azure")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "gpt-4.1")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://my-resource.openai.azure.com")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_API_VERSION", "2025-03-01-preview")


def _write_sp_files(
    tmp_path: Path,
    *,
    client_id: str = "cid",
    tenant_id: str = "tid",
    client_secret: str = "csec",  # noqa: S107
) -> None:
    """Write Entra ID service-principal credential files."""
    (tmp_path / "client_id").write_text(client_id, encoding="utf-8")
    (tmp_path / "tenant_id").write_text(tenant_id, encoding="utf-8")
    (tmp_path / "client_secret").write_text(client_secret, encoding="utf-8")


def test_azure_entra_id_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """All three SP files present → Entra ID mode, no AZURE_OPENAI_API_KEY."""
    _setup_azure_env(monkeypatch)
    _write_sp_files(tmp_path)
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))

    sdk = resolve_sdk()

    assert sdk.name == "openai"
    assert sdk.azure_auth_mode == "entra_id"
    assert sdk.azure_credentials == {
        "client_id": "cid",
        "tenant_id": "tid",
        "client_secret": "csec",
    }
    # MUST NOT set API key in Entra mode
    assert "AZURE_OPENAI_API_KEY" not in os.environ


def test_azure_api_key_mode_from_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """apitoken file present, no SP files → API-key mode."""
    _setup_azure_env(monkeypatch)
    (tmp_path / "apitoken").write_text("my-azure-key", encoding="utf-8")
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))

    sdk = resolve_sdk()

    assert sdk.name == "openai"
    assert sdk.azure_auth_mode == "api_key"
    assert sdk.azure_credentials is None
    assert os.environ["AZURE_OPENAI_API_KEY"] == "my-azure-key"


def test_azure_api_version_is_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Azure provider requires an explicit API version from the operator."""
    _setup_azure_env(monkeypatch)
    monkeypatch.delenv("LIGHTSPEED_PROVIDER_API_VERSION", raising=False)
    (tmp_path / "apitoken").write_text("my-azure-key", encoding="utf-8")
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))

    with pytest.raises(ValueError, match="LIGHTSPEED_PROVIDER_API_VERSION is required"):
        resolve_sdk()

    assert "AZURE_OPENAI_API_VERSION" not in os.environ


def test_azure_api_key_mode_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """AZURE_OPENAI_API_KEY env var present, no SP files → API-key mode."""
    _setup_azure_env(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))

    sdk = resolve_sdk()

    assert sdk.name == "openai"
    assert sdk.azure_auth_mode == "api_key"
    assert sdk.azure_credentials is None


def test_azure_entra_id_takes_precedence_over_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When both SP files and apitoken exist, Entra ID wins."""
    _setup_azure_env(monkeypatch)
    _write_sp_files(tmp_path)
    (tmp_path / "apitoken").write_text("also-a-key", encoding="utf-8")
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))

    sdk = resolve_sdk()

    assert sdk.azure_auth_mode == "entra_id"
    assert "AZURE_OPENAI_API_KEY" not in os.environ


def test_azure_incomplete_sp_files_falls_back_to_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only two of three SP files → falls back to API key if available."""
    _setup_azure_env(monkeypatch)
    (tmp_path / "client_id").write_text("cid", encoding="utf-8")
    (tmp_path / "tenant_id").write_text("tid", encoding="utf-8")
    # client_secret missing
    (tmp_path / "apitoken").write_text("fallback-key", encoding="utf-8")
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))

    sdk = resolve_sdk()

    assert sdk.azure_auth_mode == "api_key"


def test_azure_empty_sp_file_not_entra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SP files present but one is empty → not Entra mode."""
    _setup_azure_env(monkeypatch)
    (tmp_path / "client_id").write_text("cid", encoding="utf-8")
    (tmp_path / "tenant_id").write_text("", encoding="utf-8")  # empty
    (tmp_path / "client_secret").write_text("csec", encoding="utf-8")
    (tmp_path / "apitoken").write_text("key", encoding="utf-8")
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))

    sdk = resolve_sdk()

    assert sdk.azure_auth_mode == "api_key"


def test_azure_no_credentials_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No SP files and no API key → ValueError."""
    _setup_azure_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))

    with pytest.raises(ValueError, match=r"Azure .* credentials"):
        resolve_sdk()
