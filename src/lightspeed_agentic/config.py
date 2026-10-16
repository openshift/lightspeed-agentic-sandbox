"""Configuration mapping: LIGHTSPEED_* generic vars → SDK-specific env vars.

The operator sets generic LIGHTSPEED_* env vars on the sandbox pod.
This module maps them to the SDK-specific env vars that each provider
SDK reads internally. Called once at startup before provider construction.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

LLM_CREDENTIALS_PATH = "/var/run/secrets/llm-credentials"
DEFAULT_AZURE_API_VERSION = "2024-08-01-preview"


def _llm_credentials_path() -> str:
    """Operator mount path; override for local e2e host mode when /var/run is not writable."""
    override = os.environ.get("LIGHTSPEED_LLM_CREDENTIALS_PATH", "").strip()
    return override or LLM_CREDENTIALS_PATH


@dataclasses.dataclass(frozen=True)
class ResolvedModelFamily:
    """Resolved model family with credential and readiness configuration."""

    name: Literal["anthropic", "google", "openai"]
    expected_envs: tuple[str, ...]
    credential_file_envs: tuple[str, ...] = ()
    azure_auth_mode: str | None = None
    azure_credentials: dict[str, str] | None = None


@dataclasses.dataclass(frozen=True)
class ResolvedSDK:
    """Legacy adapter and readiness configuration retained during migration."""

    name: str  # "deepagents", "gemini", "openai"
    expected_envs: tuple[str, ...]  # credential env vars expected from envFrom
    credential_file_envs: tuple[str, ...] = ()  # env vars whose value is a credentials file path
    azure_auth_mode: str | None = None  # "entra_id" or "api_key"; None for non-Azure
    azure_credentials: dict[str, str] | None = None  # Entra SP values for adapter


def _setenv(key: str, value: str) -> None:
    os.environ[key] = value


def _setenv_if_value(key: str, value: str | None) -> None:
    if value:
        _setenv(key, value)


def _resolve_anthropic(model: str | None, url: str | None) -> ResolvedModelFamily:
    _setenv_if_value("ANTHROPIC_MODEL", model)
    _setenv_if_value("ANTHROPIC_BASE_URL", url)
    return ResolvedModelFamily(
        "anthropic",
        ("ANTHROPIC_API_KEY",),
        # ANTHROPIC_AUTH_TOKEN is optional, used for vLLM bearer token auth
    )


def _resolve_vertex(
    model_provider: str | None,
    model: str | None,
    url: str | None,
    project: str | None,
    region: str | None,
) -> ResolvedModelFamily:
    if not model_provider:
        raise ValueError("LIGHTSPEED_MODEL_PROVIDER is required when LIGHTSPEED_PROVIDER=vertex")

    match model_provider:
        case "anthropic":
            _setenv_if_value("ANTHROPIC_MODEL", model)
            _setenv("CLAUDE_CODE_USE_VERTEX", "1")
            _setenv_if_value("ANTHROPIC_VERTEX_PROJECT_ID", project)
            _setenv_if_value("CLOUD_ML_REGION", region)
            _setenv(
                "GOOGLE_APPLICATION_CREDENTIALS",
                f"{_llm_credentials_path()}/GOOGLE_APPLICATION_CREDENTIALS",
            )
            _setenv_if_value("ANTHROPIC_BASE_URL", url)
            return ResolvedModelFamily(
                "anthropic",
                ("GOOGLE_APPLICATION_CREDENTIALS",),
                ("GOOGLE_APPLICATION_CREDENTIALS",),
            )
        case "google":
            _setenv_if_value("GEMINI_MODEL", model)
            _setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
            _setenv_if_value("GOOGLE_CLOUD_PROJECT", project)
            _setenv_if_value("GOOGLE_CLOUD_LOCATION", region)
            _setenv(
                "GOOGLE_APPLICATION_CREDENTIALS",
                f"{_llm_credentials_path()}/GOOGLE_APPLICATION_CREDENTIALS",
            )
            return ResolvedModelFamily(
                "google",
                ("GOOGLE_APPLICATION_CREDENTIALS",),
                ("GOOGLE_APPLICATION_CREDENTIALS",),
            )
        case "openai":
            _setenv_if_value("OPENAI_MODEL", model)
            _setenv_if_value("OPENAI_BASE_URL", url)
            _setenv(
                "GOOGLE_APPLICATION_CREDENTIALS",
                f"{_llm_credentials_path()}/GOOGLE_APPLICATION_CREDENTIALS",
            )
            return ResolvedModelFamily(
                "openai",
                ("GOOGLE_APPLICATION_CREDENTIALS",),
                ("GOOGLE_APPLICATION_CREDENTIALS",),
            )
        case _:
            raise ValueError(
                f"Unknown LIGHTSPEED_MODEL_PROVIDER: {model_provider!r}. "
                "Supported: anthropic, google, openai"
            )


def _resolve_openai(model: str | None, url: str | None) -> ResolvedModelFamily:
    _setenv_if_value("OPENAI_MODEL", model)
    _setenv_if_value("OPENAI_BASE_URL", url)
    return ResolvedModelFamily(
        "openai",
        ("OPENAI_API_KEY",),
    )


def _read_credential_file(path: str) -> str | None:
    """Read a single credential file, return stripped content or None."""
    try:
        p = Path(path)
        if not p.is_file():
            return None
        content = p.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return content or None


def _parse_azure_api_version_date(api_version: str | None) -> tuple[int, int, int] | None:
    if not api_version:
        return None
    try:
        version_date = api_version.split("-preview", 1)[0]
        year, month, day = (int(part) for part in version_date.split("-"))
    except ValueError:
        return None
    return year, month, day


def azure_api_version_supports_structured_outputs(api_version: str | None) -> bool:
    """Return whether an Azure OpenAI API version supports structured outputs."""
    version = _parse_azure_api_version_date(api_version)
    return version is not None and version >= (2024, 8, 1)


def azure_api_version_supports_responses(api_version: str | None) -> bool:
    """Return whether an Azure OpenAI API version supports Responses API."""
    version = _parse_azure_api_version_date(api_version)
    return version is not None and version >= (2025, 3, 1)


def _resolve_azure(
    model: str | None,
    url: str | None,
    api_version: str | None,
) -> ResolvedModelFamily:
    _setenv_if_value("OPENAI_MODEL", model)
    _setenv_if_value("AZURE_OPENAI_ENDPOINT", url)
    _setenv("AZURE_OPENAI_API_VERSION", api_version or DEFAULT_AZURE_API_VERSION)

    creds_path = _llm_credentials_path()
    client_id = _read_credential_file(f"{creds_path}/client_id")
    tenant_id = _read_credential_file(f"{creds_path}/tenant_id")
    client_secret = _read_credential_file(f"{creds_path}/client_secret")

    if client_id and tenant_id and client_secret:
        # Entra ID mode — do NOT set AZURE_OPENAI_API_KEY
        logger.info("Azure credential resolution: Entra ID (service principal)")
        return ResolvedModelFamily(
            "openai",
            (),  # no env-based credentials needed
            azure_auth_mode="entra_id",
            azure_credentials={
                "client_id": client_id,
                "tenant_id": tenant_id,
                "client_secret": client_secret,
            },
        )

    # API-key mode — check file first, then env
    api_key_file = _read_credential_file(f"{creds_path}/apitoken")
    if api_key_file:
        _setenv("AZURE_OPENAI_API_KEY", api_key_file)
        logger.info("Azure credential resolution: API key (from file)")
        return ResolvedModelFamily(
            "openai",
            ("AZURE_OPENAI_API_KEY",),
            azure_auth_mode="api_key",
        )

    api_key_env = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    if api_key_env:
        logger.info("Azure credential resolution: API key (from env)")
        return ResolvedModelFamily(
            "openai",
            ("AZURE_OPENAI_API_KEY",),
            azure_auth_mode="api_key",
        )

    raise ValueError(
        "Azure OpenAI credentials not found. Provide either "
        "all three Entra ID files (client_id, tenant_id, client_secret) "
        f"at {creds_path}/, or an API key via apitoken file or "
        "AZURE_OPENAI_API_KEY env var."
    )


def _resolve_bedrock(
    model: str | None,
    url: str | None,
    region: str | None,
) -> ResolvedModelFamily:
    _setenv_if_value("ANTHROPIC_MODEL", model)
    _setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    _setenv_if_value("AWS_REGION", region)
    _setenv_if_value("ANTHROPIC_BASE_URL", url)
    return ResolvedModelFamily(
        "anthropic",
        ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
    )


def resolve_model_family() -> ResolvedModelFamily:
    """Read LIGHTSPEED_* vars, map SDK env vars, and resolve model family and credentials."""
    provider = os.environ.get("LIGHTSPEED_PROVIDER", "").strip().lower() or "anthropic"
    model = os.environ.get("LIGHTSPEED_MODEL", "").strip() or None
    model_provider = os.environ.get("LIGHTSPEED_MODEL_PROVIDER", "").strip().lower() or None
    url = os.environ.get("LIGHTSPEED_PROVIDER_URL", "").strip() or None
    project = os.environ.get("LIGHTSPEED_PROVIDER_PROJECT", "").strip() or None
    region = os.environ.get("LIGHTSPEED_PROVIDER_REGION", "").strip() or None
    api_version = os.environ.get("LIGHTSPEED_PROVIDER_API_VERSION", "").strip() or None

    match provider:
        case "anthropic":
            model_family = _resolve_anthropic(model, url)
        case "vertex":
            model_family = _resolve_vertex(model_provider, model, url, project, region)
        case "openai":
            model_family = _resolve_openai(model, url)
        case "azure":
            model_family = _resolve_azure(model, url, api_version)
        case "bedrock":
            model_family = _resolve_bedrock(model, url, region)
        case _:
            raise ValueError(
                f"Unknown provider: {provider!r}. "
                "Supported: anthropic, vertex, openai, azure, bedrock"
            )

    logger.info("Resolved LIGHTSPEED_PROVIDER=%s → model family=%s", provider, model_family.name)
    return model_family


def parse_tool_output_inspection_enabled() -> bool:
    """Parse the DeepAgents tool-result inspection toggle.

    Inspection is enabled by default. Non-empty values must be an explicit
    case-insensitive Boolean value.
    """
    raw = os.environ.get("LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED", "").strip()
    if not raw:
        return True
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    raise ValueError("LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED must be true or false")


def parse_reasoning_config() -> dict[str, Any] | None:
    """Parse LIGHTSPEED_REASONING_CONFIG env var at startup.

    Returns None when absent/empty. Raises ValueError on malformed JSON
    or non-object types (per configuration.md rule 9a).
    """
    raw = os.environ.get("LIGHTSPEED_REASONING_CONFIG", "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LIGHTSPEED_REASONING_CONFIG contains invalid JSON: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError(
            f"LIGHTSPEED_REASONING_CONFIG must be a JSON object, got {type(parsed).__name__}"
        )

    return parsed


_MODEL_ENV_VARS = {
    "deepagents": "ANTHROPIC_MODEL",
    "gemini": "GEMINI_MODEL",
    "openai": "OPENAI_MODEL",
}


def _model_env_var(provider_name: str) -> str:
    """Return the SDK model env var for a resolved provider name."""
    env_var = _MODEL_ENV_VARS.get(provider_name)
    if env_var is None:
        supported = ", ".join(sorted(_MODEL_ENV_VARS))
        raise ValueError(f"Unknown SDK provider: {provider_name!r}. Supported: {supported}")
    return env_var


def resolve_startup_model(provider_name: str) -> str | None:
    """Return startup model hint for logging; None when only defaults apply."""
    lightspeed_model = os.environ.get("LIGHTSPEED_MODEL", "").strip()
    if lightspeed_model:
        return lightspeed_model
    env_var = _model_env_var(provider_name)
    sdk_model = os.environ.get(env_var, "").strip()
    return sdk_model or None


def resolve_router_model(provider_name: str, model: str | None = None) -> str:
    """Resolve model per configuration.md rule 5."""
    from lightspeed_agentic.types import DEFAULT_MODEL

    if model:
        return model
    lightspeed_model = os.environ.get("LIGHTSPEED_MODEL", "").strip()
    if lightspeed_model:
        return lightspeed_model
    env_var = _model_env_var(provider_name)
    sdk_model = os.environ.get(env_var, "").strip()
    if sdk_model:
        return sdk_model
    return DEFAULT_MODEL


def parse_agent_timeout() -> int:
    """Parse LIGHTSPEED_AGENT_TIMEOUT_SECONDS env var at startup.

    Returns the timeout in seconds (int). Requires positive integer.
    Raises ValueError on missing, zero, negative, or malformed input.
    """
    raw = os.environ.get("LIGHTSPEED_AGENT_TIMEOUT_SECONDS", "").strip()
    if not raw:
        raise ValueError("LIGHTSPEED_AGENT_TIMEOUT_SECONDS is required but not set")

    try:
        timeout_seconds = int(raw)
    except ValueError as e:
        raise ValueError(
            f"LIGHTSPEED_AGENT_TIMEOUT_SECONDS must be a positive integer, got {raw!r}"
        ) from e

    if timeout_seconds <= 0:
        raise ValueError(
            f"LIGHTSPEED_AGENT_TIMEOUT_SECONDS must be positive, got {timeout_seconds}"
        )

    return timeout_seconds


def parse_max_turns() -> int:
    """Parse LIGHTSPEED_AGENT_MAX_TURNS env var at startup.

    Returns the max turns count (int). Requires integer between 1 and 500 inclusive.
    Raises ValueError on missing, out-of-range, or malformed input.
    """
    raw = os.environ.get("LIGHTSPEED_AGENT_MAX_TURNS", "").strip()
    if not raw:
        raise ValueError("LIGHTSPEED_AGENT_MAX_TURNS is required but not set")

    try:
        max_turns = int(raw)
    except ValueError as e:
        raise ValueError(f"LIGHTSPEED_AGENT_MAX_TURNS must be an integer, got {raw!r}") from e

    if max_turns < 1 or max_turns > 500:
        raise ValueError(f"LIGHTSPEED_AGENT_MAX_TURNS must be between 1 and 500, got {max_turns}")

    return max_turns
