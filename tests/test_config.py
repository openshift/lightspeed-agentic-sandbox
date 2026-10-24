"""Tests for provider configuration resolution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import lightspeed_agentic.config as config
from lightspeed_agentic.config import (
    parse_agent_timeout,
    parse_max_turns,
    parse_reasoning_config,
    resolve_model_family,
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
        "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED",
        "LIGHTSPEED_MODEL",
        "LIGHTSPEED_MODEL_PROVIDER",
        "LIGHTSPEED_PROVIDER_URL",
        "LIGHTSPEED_PROVIDER_PROJECT",
        "LIGHTSPEED_PROVIDER_REGION",
        "LIGHTSPEED_PROVIDER_API_VERSION",
        "LIGHTSPEED_LLM_CREDENTIALS_PATH",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
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
        "OPENAI_API_KEY",
        "ANTHROPIC_BASE_URL",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_API_KEY",
        "AWS_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "role_arn",
    ]:
        monkeypatch.delenv(var, raising=False)


def test_provider_parameter_types_are_defined_in_config() -> None:
    from lightspeed_agentic.config import (
        AnthropicCredentialSource,
        BedrockCredentialSource,
        BedrockProviderParameters,
        ModelParameters,
        VertexCredentialSource,
        VertexProviderParameters,
    )

    types = (
        AnthropicCredentialSource,
        BedrockCredentialSource,
        BedrockProviderParameters,
        ModelParameters,
        VertexCredentialSource,
        VertexProviderParameters,
    )
    assert all(provider_type.__module__ == config.__name__ for provider_type in types)


def test_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    from lightspeed_agentic.config import AnthropicCredentialSource

    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "anthropic")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "resolved-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "resolved-auth-token")

    model_family = resolve_model_family()

    assert model_family.name == "anthropic"
    assert model_family.model == "claude-sonnet-4-20250514"
    assert model_family.credentials == AnthropicCredentialSource(
        api_key="resolved-api-key",
        auth_token="resolved-auth-token",  # noqa: S106 - Test credential value.
    )
    assert "resolved-api-key" not in repr(model_family)
    assert "resolved-auth-token" not in repr(model_family)


def test_anthropic_with_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "anthropic")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "claude-opus-4-6")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://proxy.example.com")

    model_family = resolve_model_family()

    assert model_family.base_url == "https://proxy.example.com"


def test_vertex_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    from lightspeed_agentic.config import (
        VertexCredentialSource,
        VertexProviderParameters,
    )

    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_PROJECT", "my-project")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_REGION", "us-east5")

    model_family = resolve_model_family()

    assert model_family.name == "anthropic"
    assert model_family.model == "claude-sonnet-4-20250514"
    assert model_family.provider == VertexProviderParameters(
        project="my-project", region="us-east5"
    )
    assert model_family.credentials == VertexCredentialSource(
        credentials_file=Path("/var/run/secrets/llm-credentials/GOOGLE_APPLICATION_CREDENTIALS")
    )


def test_vertex_anthropic_defaults_region_to_global(monkeypatch: pytest.MonkeyPatch) -> None:
    from lightspeed_agentic.config import VertexProviderParameters

    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "claude-sonnet-4-20250514")

    model_family = resolve_model_family()

    assert model_family.provider == VertexProviderParameters(project=None, region="global")


def test_vertex_google(monkeypatch: pytest.MonkeyPatch) -> None:
    from lightspeed_agentic.config import (
        VertexCredentialSource,
        VertexProviderParameters,
    )

    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", "google")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_PROJECT", "my-project")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_REGION", "us-central1")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://vertex-proxy.example.test/custom")

    model_family = resolve_model_family()

    assert model_family.name == "google"
    assert model_family.model == "gemini-2.5-flash"
    assert model_family.base_url == "https://vertex-proxy.example.test/custom"
    assert model_family.provider == VertexProviderParameters(
        project="my-project", region="us-central1"
    )
    assert model_family.credentials == VertexCredentialSource(
        credentials_file=Path("/var/run/secrets/llm-credentials/GOOGLE_APPLICATION_CREDENTIALS")
    )


def test_vertex_google_uses_canonical_provider_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightspeed_agentic.config import VertexProviderParameters

    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", "google")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_REGION", "us-east5")
    monkeypatch.setenv("CLOUD_ML_REGION", "ignored-anthropic-region")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "ignored-google-region")

    model_family = resolve_model_family()

    assert model_family.provider == VertexProviderParameters(project=None, region="us-east5")


def test_vertex_google_defaults_region_to_global(monkeypatch: pytest.MonkeyPatch) -> None:
    from lightspeed_agentic.config import VertexProviderParameters

    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", "google")

    model_family = resolve_model_family()

    assert model_family.provider == VertexProviderParameters(project=None, region="global")


def test_vertex_llm_credentials_path_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from lightspeed_agentic.config import VertexCredentialSource

    _clean_env(monkeypatch)
    creds_dir = "/var/run/secrets/e2e-llm-credentials"
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", "google")
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", creds_dir)

    model_family = resolve_model_family()

    assert model_family.credentials == VertexCredentialSource(
        credentials_file=Path(creds_dir) / "GOOGLE_APPLICATION_CREDENTIALS"
    )


def test_vertex_openai_resolves_legacy_settings_without_mutating_sdk_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightspeed_agentic.config import VertexCredentialSource

    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "gpt-4.1")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://vertex-openai.example.com")

    model_family = resolve_model_family()

    assert model_family.name == "openai"
    assert model_family.model == "gpt-4.1"
    assert model_family.base_url == "https://vertex-openai.example.com"
    assert model_family.credentials == VertexCredentialSource(
        Path("/var/run/secrets/llm-credentials/GOOGLE_APPLICATION_CREDENTIALS")
    )
    assert "OPENAI_MODEL" not in os.environ
    assert "OPENAI_BASE_URL" not in os.environ
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


def test_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "openai")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "gpt-4.1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    model_family = resolve_model_family()

    assert model_family.name == "openai"
    assert model_family.api_key == "openai-secret"
    assert "openai-secret" not in repr(model_family)
    assert "OPENAI_MODEL" not in os.environ


def test_openai_with_url_resolves_legacy_endpoint_without_mutating_sdk_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "openai")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "gpt-4.1")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://custom.openai.com/v1")

    model_family = resolve_model_family()

    assert model_family.base_url == "https://custom.openai.com/v1"
    assert "OPENAI_BASE_URL" not in os.environ


def test_azure_resolves_legacy_settings_without_mutating_sdk_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "azure")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "gpt-4.1")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://my-resource.openai.azure.com")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_API_VERSION", "2024-08-01-preview")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

    model_family = resolve_model_family()

    assert model_family.name == "openai"
    assert model_family.model == "gpt-4.1"
    assert model_family.base_url == "https://my-resource.openai.azure.com"
    assert model_family.azure_api_version == "2024-08-01-preview"
    assert model_family.provider == config.AzureProviderParameters(api_version="2024-08-01-preview")
    assert "OPENAI_MODEL" not in os.environ
    assert "AZURE_OPENAI_ENDPOINT" not in os.environ
    assert "AZURE_OPENAI_API_VERSION" not in os.environ


def test_bedrock(monkeypatch: pytest.MonkeyPatch) -> None:
    from lightspeed_agentic.config import (
        BedrockCredentialSource,
        BedrockProviderParameters,
    )

    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "bedrock")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_REGION", "us-east-1")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")

    model_family = resolve_model_family()

    assert model_family.name == "anthropic"
    assert model_family.model == "claude-sonnet-4-20250514"
    assert model_family.provider == BedrockProviderParameters(region="us-east-1")
    assert model_family.credentials == BedrockCredentialSource(
        access_key_id="test-access-key",
        secret_access_key="test-secret-key",  # noqa: S106 - test credential
    )


def test_bedrock_region_falls_back_to_aws_region(monkeypatch: pytest.MonkeyPatch) -> None:
    from lightspeed_agentic.config import BedrockProviderParameters

    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "bedrock")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")

    model_family = resolve_model_family()

    assert model_family.provider == BedrockProviderParameters(region="eu-west-1")


def test_bedrock_resolves_credentials_from_files_with_environment_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lightspeed_agentic.config import BedrockCredentialSource

    _clean_env(monkeypatch)
    credentials_dir = tmp_path / "llm-credentials"
    credentials_dir.mkdir()
    (credentials_dir / "aws_access_key_id").write_text("file-access\n", encoding="utf-8")
    (credentials_dir / "aws_secret_access_key").write_text("file-secret\n", encoding="utf-8")
    (credentials_dir / "aws_session_token").write_text("file-session\n", encoding="utf-8")
    (credentials_dir / "role_arn").write_text(
        "arn:aws:iam::123456789012:role/file-role\n", encoding="utf-8"
    )
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "bedrock")
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(credentials_dir))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "environment-access")

    model_family = resolve_model_family()

    assert model_family.credentials == BedrockCredentialSource(
        access_key_id="environment-access",
        secret_access_key="file-secret",  # noqa: S106 - test credential
        session_token="file-session",  # noqa: S106 - test credential
        role_arn="arn:aws:iam::123456789012:role/file-role",
    )
    assert "environment-access" not in repr(model_family)
    assert "file-secret" not in repr(model_family)


@pytest.mark.parametrize(
    ("provider", "model_provider"),
    [("anthropic", None), ("bedrock", None), ("vertex", "anthropic"), ("vertex", "google")],
)
def test_migrated_provider_resolution_does_not_mutate_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    model_provider: str | None,
) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", provider)
    monkeypatch.setenv("LIGHTSPEED_MODEL", "configured-model")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://provider.example.test")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_PROJECT", "configured-project")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_REGION", "us-central1")
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))
    if model_provider:
        monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", model_provider)

    env_before = dict(os.environ)

    resolve_model_family()

    assert dict(os.environ) == env_before


def test_vertex_google_uses_only_the_canonical_region_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightspeed_agentic.config import VertexProviderParameters

    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", "google")
    monkeypatch.setenv("CLOUD_ML_REGION", "legacy-anthropic-region")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "legacy-google-region")

    model_family = resolve_model_family()

    assert model_family.provider == VertexProviderParameters(project=None, region="global")
    assert os.environ["CLOUD_ML_REGION"] == "legacy-anthropic-region"
    assert os.environ["GOOGLE_CLOUD_LOCATION"] == "legacy-google-region"


def test_anthropic_and_gemini_ignore_legacy_model_and_endpoint_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "legacy-model")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://legacy.example.test")

    anthropic_family = resolve_model_family()

    assert anthropic_family.model is None
    assert anthropic_family.base_url is None

    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", "google")
    monkeypatch.setenv("GEMINI_MODEL", "legacy-gemini-model")

    gemini_family = resolve_model_family()

    assert gemini_family.model is None


def test_default_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)

    model_family = resolve_model_family()

    assert isinstance(model_family, config.ResolvedModelFamily)
    assert model_family.name == "anthropic"


def test_default_model_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "anthropic")

    model_family = resolve_model_family()

    assert model_family.model is None


def test_vertex_missing_model_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "some-model")

    with pytest.raises(ValueError, match="LIGHTSPEED_MODEL_PROVIDER"):
        resolve_model_family()


def test_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "watsonx")

    with pytest.raises(ValueError, match="Unknown provider"):
        resolve_model_family()


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


def test_tool_output_inspection_defaults_to_enabled_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED", raising=False)

    assert config.parse_tool_output_inspection_enabled() is True


def test_tool_output_inspection_defaults_to_enabled_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED", "  ")

    assert config.parse_tool_output_inspection_enabled() is True


@pytest.mark.parametrize("value", ["true", "TRUE", "TrUe"])
def test_tool_output_inspection_accepts_true_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED", value)

    assert config.parse_tool_output_inspection_enabled() is True


@pytest.mark.parametrize("value", ["false", "FALSE", "FaLsE"])
def test_tool_output_inspection_accepts_false_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED", value)

    assert config.parse_tool_output_inspection_enabled() is False


def test_tool_output_inspection_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED", "yes")

    with pytest.raises(ValueError, match="LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED"):
        config.parse_tool_output_inspection_enabled()


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

    model_family = resolve_model_family()

    assert model_family.name == "openai"
    assert model_family.expected_envs == ()
    assert model_family.azure_credentials == {
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

    model_family = resolve_model_family()

    assert model_family.name == "openai"
    assert model_family.azure_credentials is None
    assert model_family.azure_api_version == "2025-03-01-preview"
    assert model_family.azure_api_key == "my-azure-key"
    assert "my-azure-key" not in repr(model_family)
    assert "AZURE_OPENAI_API_KEY" not in os.environ
    assert "AZURE_OPENAI_API_VERSION" not in os.environ


def test_azure_defaults_to_structured_output_api_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Omitted Azure API version uses the structured-output-capable default."""
    _setup_azure_env(monkeypatch)
    monkeypatch.delenv("LIGHTSPEED_PROVIDER_API_VERSION", raising=False)
    (tmp_path / "apitoken").write_text("my-azure-key", encoding="utf-8")
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))

    model_family = resolve_model_family()

    assert model_family.name == "openai"
    assert model_family.azure_api_version == "2024-08-01-preview"
    assert model_family.provider == config.AzureProviderParameters(api_version="2024-08-01-preview")
    assert "AZURE_OPENAI_API_VERSION" not in os.environ


def test_azure_api_key_mode_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """AZURE_OPENAI_API_KEY env var present, no SP files → API-key mode."""
    _setup_azure_env(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))

    model_family = resolve_model_family()

    assert model_family.name == "openai"
    assert model_family.azure_credentials is None
    assert model_family.azure_api_key == "env-key"
    assert "env-key" not in repr(model_family)
    assert os.environ["AZURE_OPENAI_API_KEY"] == "env-key"


def test_azure_entra_id_takes_precedence_over_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Complete SP files win over both API-key sources without rewriting env."""
    _setup_azure_env(monkeypatch)
    _write_sp_files(
        tmp_path,
        client_id=" cid\n",
        tenant_id=" tid\n",
        client_secret=" csec\n",  # noqa: S106 - Test credential.
    )
    (tmp_path / "apitoken").write_text("also-a-key", encoding="utf-8")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "ambient-azure-key")
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))
    env_before = dict(os.environ)

    model_family = resolve_model_family()

    assert model_family.credentials == config.AzureEntraCredentialSource(
        client_id="cid",
        tenant_id="tid",
        client_secret="csec",  # noqa: S106
    )
    assert model_family.azure_credentials == {
        "client_id": "cid",
        "tenant_id": "tid",
        "client_secret": "csec",
    }
    assert model_family.azure_api_key is None
    assert model_family.azure_auth_mode == "entra_id"
    assert model_family.expected_envs == ()
    assert dict(os.environ) == env_before


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

    model_family = resolve_model_family()

    assert model_family.azure_credentials is None
    assert model_family.azure_api_key == "fallback-key"
    assert "AZURE_OPENAI_API_KEY" not in os.environ


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

    model_family = resolve_model_family()

    assert model_family.azure_credentials is None
    assert model_family.azure_api_key == "key"
    assert "AZURE_OPENAI_API_KEY" not in os.environ


def test_azure_no_credentials_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No SP files and no API key → ValueError."""
    _setup_azure_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))

    with pytest.raises(ValueError, match=r"Azure .* credentials"):
        resolve_model_family()


@pytest.mark.parametrize("auth_source", ["entra", "file", "env"])
def test_azure_resolution_populates_typed_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, auth_source: str
) -> None:
    _setup_azure_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))
    if auth_source == "entra":
        _write_sp_files(tmp_path)
    elif auth_source == "file":
        (tmp_path / "apitoken").write_text(" file-key\n", encoding="utf-8")
    else:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", " env-key\n")
    env_before = dict(os.environ)

    family = resolve_model_family()

    assert family.provider is not None
    assert family.provider == config.AzureProviderParameters(api_version="2025-03-01-preview")
    assert family.hosting == "azure"
    assert family.expected_envs == ()  # Readiness uses the resolved credential source.
    assert family.model == "gpt-4.1"
    assert family.base_url == "https://my-resource.openai.azure.com"
    if auth_source == "entra":
        assert family.credentials == config.AzureEntraCredentialSource(
            client_id="cid",
            tenant_id="tid",
            client_secret="csec",  # noqa: S106
        )
    else:
        assert family.credentials == config.AzureApiKeyCredentialSource(
            api_key="file-key" if auth_source == "file" else "env-key"
        )
    assert dict(os.environ) == env_before


@pytest.mark.parametrize(
    ("file_content", "expected_key"),
    [(" file-key\n", "file-key"), (" \n\t", "env-key"), (None, "env-key")],
    ids=["file-before-env", "blank-file-falls-back", "missing-file-falls-back"],
)
def test_azure_typed_api_key_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    file_content: str | None,
    expected_key: str,
) -> None:
    _setup_azure_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", " env-key\n")
    if file_content is not None:
        (tmp_path / "apitoken").write_text(file_content, encoding="utf-8")
    env_before = dict(os.environ)

    family = resolve_model_family()

    assert family.credentials is not None
    assert family.credentials == config.AzureApiKeyCredentialSource(api_key=expected_key)
    assert expected_key not in repr(family.credentials)
    assert family.azure_api_key == expected_key
    assert family.azure_credentials is None
    assert family.azure_auth_mode == "api_key"
    assert dict(os.environ) == env_before


@pytest.mark.parametrize("unusable_file", ["client_id", "tenant_id", "client_secret"])
@pytest.mark.parametrize("file_state", ["missing", "blank"])
@pytest.mark.parametrize("fallback", ["file", "env", None], ids=["file-key", "env-key", "no-key"])
def test_azure_incomplete_credentials_keep_fallback_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    unusable_file: str,
    file_state: str,
    fallback: str | None,
) -> None:
    _setup_azure_env(monkeypatch)
    _write_sp_files(
        tmp_path,
        client_id="fallback-client-private",
        tenant_id="fallback-tenant-private",
        client_secret="fallback-secret-private",  # noqa: S106 - Test credential.
    )
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))
    if file_state == "missing":
        (tmp_path / unusable_file).unlink()
    else:
        (tmp_path / unusable_file).write_text(" \n\t", encoding="utf-8")
    if fallback == "file":
        (tmp_path / "apitoken").write_text("fallback-key", encoding="utf-8")
    elif fallback == "env":
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fallback-key")
    else:
        # Credentials for another provider must not rescue an unusable Azure set.
        monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    env_before = dict(os.environ)

    if fallback is None:
        with pytest.raises(ValueError, match="Azure OpenAI credentials not found") as error:
            resolve_model_family()
        assert "client_id, tenant_id, client_secret" in str(error.value)
        assert "apitoken" in str(error.value)
        assert "AZURE_OPENAI_API_KEY" in str(error.value)
        for secret in (
            "unrelated-openai-key",
            "fallback-client-private",
            "fallback-tenant-private",
            "fallback-secret-private",
        ):
            assert secret not in str(error.value)
    else:
        family = resolve_model_family()
        assert family.credentials is not None
        assert family.credentials == config.AzureApiKeyCredentialSource(api_key="fallback-key")
        assert family.azure_credentials is None
        assert family.azure_auth_mode == "api_key"
    assert dict(os.environ) == env_before


@pytest.mark.parametrize("legacy_sdk", [False, True], ids=["family", "legacy-handoff"])
def test_azure_service_principal_values_are_hidden_from_configuration_repr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, legacy_sdk: bool
) -> None:
    _setup_azure_env(monkeypatch)
    _write_sp_files(
        tmp_path,
        client_id="private-client-value",
        tenant_id="private-tenant-value",
        client_secret="private-secret-value",  # noqa: S106
    )
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))
    family = resolve_model_family()
    value = (
        config.ResolvedSDK(
            name="openai",
            expected_envs=(),
            azure_auth_mode=family.azure_auth_mode,
            azure_credentials=family.azure_credentials,
        )
        if legacy_sdk
        else family
    )

    for secret in ("private-client-value", "private-tenant-value", "private-secret-value"):
        assert secret not in repr(value)
        assert secret not in repr(family.credentials)


@pytest.mark.parametrize("entra", [False, True], ids=["api-key", "entra"])
def test_azure_typed_configuration_retains_the_resolved_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entra: bool
) -> None:
    _setup_azure_env(monkeypatch)
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(tmp_path))
    if entra:
        _write_sp_files(tmp_path)
    else:
        (tmp_path / "apitoken").write_text("original-key", encoding="utf-8")
    family = resolve_model_family()

    _write_sp_files(
        tmp_path,
        client_id="new-client",
        tenant_id="new-tenant",
        client_secret="new-secret",  # noqa: S106 - Test credential.
    )
    (tmp_path / "apitoken").write_text("new-key", encoding="utf-8")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "new-env-key")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_API_VERSION", "2024-08-01-preview")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://different.example.test")

    assert family.provider is not None
    assert family.provider == config.AzureProviderParameters(api_version="2025-03-01-preview")
    assert family.base_url == "https://my-resource.openai.azure.com"
    if entra:
        assert family.credentials == config.AzureEntraCredentialSource(
            client_id="cid",
            tenant_id="tid",
            client_secret="csec",  # noqa: S106
        )
    else:
        assert family.credentials == config.AzureApiKeyCredentialSource(api_key="original-key")
