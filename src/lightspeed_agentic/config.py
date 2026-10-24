"""Resolve generic provider settings and credential references for startup.

The operator sets LIGHTSPEED_* values on the sandbox pod. Migrated Anthropic,
Vertex, and Bedrock routes keep these values in typed configuration rather than
mutating process-global SDK settings. Legacy OpenAI/Azure adapters still depend
on SDK-specific environment settings.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

LLM_CREDENTIALS_PATH = "/var/run/secrets/llm-credentials"
DEFAULT_AZURE_API_VERSION = "2024-08-01-preview"
DEFAULT_VERTEX_REGION = "global"


@dataclasses.dataclass(frozen=True)
class AnthropicCredentialSource:
    """Resolved direct-Anthropic credentials, hidden from configuration reprs."""

    api_key: str = dataclasses.field(default="", repr=False)
    auth_token: str | None = dataclasses.field(default=None, repr=False)


@dataclasses.dataclass(frozen=True)
class AzureApiKeyCredentialSource:
    """Resolved Azure API key, hidden from configuration reprs."""

    api_key: str = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True)
class AzureEntraCredentialSource:
    """Resolved Azure service-principal values; SDK credentials are created later."""

    tenant_id: str = dataclasses.field(repr=False)
    client_id: str = dataclasses.field(repr=False)
    client_secret: str = dataclasses.field(repr=False)


type AzureCredentialSource = AzureApiKeyCredentialSource | AzureEntraCredentialSource


@dataclasses.dataclass(frozen=True)
class VertexCredentialSource:
    """Mounted ADC file reference; credential contents load during model construction."""

    credentials_file: Path


@dataclasses.dataclass(frozen=True)
class BedrockCredentialSource:
    """Resolved Bedrock credentials, hidden from configuration reprs."""

    access_key_id: str = dataclasses.field(default="", repr=False)
    secret_access_key: str = dataclasses.field(default="", repr=False)
    session_token: str | None = dataclasses.field(default=None, repr=False)
    role_arn: str | None = dataclasses.field(default=None, repr=False)


type CredentialSource = (
    AnthropicCredentialSource
    | AzureCredentialSource
    | VertexCredentialSource
    | BedrockCredentialSource
)


@dataclasses.dataclass(frozen=True)
class VertexProviderParameters:
    """Project and region settings shared by Vertex model families."""

    project: str | None
    region: str


@dataclasses.dataclass(frozen=True)
class BedrockProviderParameters:
    """Resolved Bedrock region."""

    region: str


@dataclasses.dataclass(frozen=True)
class AzureProviderParameters:
    """Resolved Azure API version; endpoint and deployment use the shared model fields."""

    api_version: str


type ProviderParameters = (
    VertexProviderParameters | BedrockProviderParameters | AzureProviderParameters
)


@dataclasses.dataclass(frozen=True)
class ModelParameters:
    """Resolved model settings and lazy credentials for one provider constructor."""

    model: str
    base_url: str | None = None
    provider: ProviderParameters | None = None
    credentials: CredentialSource | None = dataclasses.field(default=None, repr=False)
    api_key: str | None = dataclasses.field(default=None, repr=False)
    max_output_tokens: int | None = None


def _llm_credentials_path() -> str:
    """Operator mount path; override for local e2e host mode when /var/run is not writable."""
    override = os.environ.get("LIGHTSPEED_LLM_CREDENTIALS_PATH", "").strip()
    return override or LLM_CREDENTIALS_PATH


@dataclasses.dataclass(frozen=True)
class ResolvedModelFamily:
    """Resolved model family with hosting, connection, and readiness configuration."""

    name: Literal["anthropic", "google", "openai"]
    expected_envs: tuple[str, ...]
    credential_file_envs: tuple[str, ...] = ()
    azure_auth_mode: str | None = None
    azure_credentials: dict[str, str] | None = dataclasses.field(default=None, repr=False)
    azure_api_version: str | None = dataclasses.field(kw_only=True, default=None)
    azure_api_key: str | None = dataclasses.field(kw_only=True, default=None, repr=False)
    hosting: Literal["anthropic", "vertex", "bedrock", "openai", "azure"] = dataclasses.field(
        kw_only=True
    )
    model: str | None = dataclasses.field(kw_only=True, default=None)
    base_url: str | None = None
    provider: ProviderParameters | None = None
    credentials: CredentialSource | None = dataclasses.field(default=None, repr=False)
    api_key: str | None = dataclasses.field(default=None, repr=False)


@dataclasses.dataclass(frozen=True)
class ResolvedSDK:
    """Legacy adapter and readiness configuration retained during migration."""

    name: str  # "openai"
    expected_envs: tuple[str, ...]  # credential env vars expected from envFrom
    credential_file_envs: tuple[str, ...] = ()  # env vars whose value is a credentials file path
    azure_auth_mode: str | None = None  # "entra_id" or "api_key"; None for non-Azure
    azure_credentials: dict[str, str] | None = dataclasses.field(default=None, repr=False)


def _resolved_value(value: str | None, env_name: str) -> str | None:
    """Return the explicit value or the corresponding legacy SDK environment value."""
    return value or os.environ.get(env_name, "").strip() or None


def _is_native_openai_endpoint(base_url: str | None) -> bool:
    """Native OpenAI uses HTTPS to the public API host or the SDK's default endpoint."""
    if not base_url:
        return True
    try:
        parsed_url = urlparse(base_url)
        hostname = (parsed_url.hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if hostname != "api.openai.com":
        return False
    if parsed_url.scheme.lower() != "https":
        raise ValueError("Native OpenAI endpoint must use HTTPS")
    return True


def _resolve_vertex_region(region: str | None) -> str:
    """Resolve the canonical Vertex region, defaulting to the global endpoint."""
    return region or DEFAULT_VERTEX_REGION


def _resolve_anthropic(model: str | None, url: str | None) -> ResolvedModelFamily:
    return ResolvedModelFamily(
        "anthropic",
        ("ANTHROPIC_API_KEY",),
        hosting="anthropic",
        model=model,
        base_url=url,
        credentials=AnthropicCredentialSource(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip(),
            auth_token=os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip() or None,
        ),
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
            resolved_region = _resolve_vertex_region(region)
            return ResolvedModelFamily(
                "anthropic",
                (),
                hosting="vertex",
                model=model,
                base_url=url,
                provider=VertexProviderParameters(project=project, region=resolved_region),
                credentials=VertexCredentialSource(
                    credentials_file=Path(_llm_credentials_path())
                    / "GOOGLE_APPLICATION_CREDENTIALS"
                ),
            )
        case "google":
            resolved_region = _resolve_vertex_region(region)
            return ResolvedModelFamily(
                "google",
                (),
                hosting="vertex",
                model=model,
                base_url=url,
                provider=VertexProviderParameters(project=project, region=resolved_region),
                credentials=VertexCredentialSource(
                    credentials_file=Path(_llm_credentials_path())
                    / "GOOGLE_APPLICATION_CREDENTIALS"
                ),
            )
        case "openai":
            return ResolvedModelFamily(
                "openai",
                ("GOOGLE_APPLICATION_CREDENTIALS",),
                ("GOOGLE_APPLICATION_CREDENTIALS",),
                hosting="vertex",
                model=_resolved_value(model, "OPENAI_MODEL"),
                base_url=url or os.environ.get("OPENAI_BASE_URL", "").strip() or None,
                credentials=VertexCredentialSource(
                    credentials_file=Path(_llm_credentials_path())
                    / "GOOGLE_APPLICATION_CREDENTIALS"
                ),
            )
        case _:
            raise ValueError(
                f"Unknown LIGHTSPEED_MODEL_PROVIDER: {model_provider!r}. "
                "Supported: anthropic, google, openai"
            )


def _resolve_openai(model: str | None, url: str | None) -> ResolvedModelFamily:
    base_url = url or os.environ.get("OPENAI_BASE_URL", "").strip() or None
    api_key = os.environ.get("OPENAI_API_KEY", "").strip() or None
    if _is_native_openai_endpoint(base_url):
        return ResolvedModelFamily(
            "openai",
            ("OPENAI_API_KEY",),
            hosting="openai",
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    return ResolvedModelFamily(
        "openai",
        ("OPENAI_API_KEY",),
        hosting="openai",
        model=_resolved_value(model, "OPENAI_MODEL"),
        base_url=base_url,
        api_key=api_key,
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
    resolved_model = _resolved_value(model, "OPENAI_MODEL")
    resolved_base_url = url or os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip() or None
    resolved_api_version = api_version or DEFAULT_AZURE_API_VERSION
    provider = AzureProviderParameters(api_version=resolved_api_version)

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
            azure_api_version=resolved_api_version,
            hosting="azure",
            model=resolved_model,
            base_url=resolved_base_url,
            azure_credentials={
                "client_id": client_id,
                "tenant_id": tenant_id,
                "client_secret": client_secret,
            },
            provider=provider,
            credentials=AzureEntraCredentialSource(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            ),
        )

    # API-key mode — check file first, then env
    api_key_file = _read_credential_file(f"{creds_path}/apitoken")
    if api_key_file:
        logger.info("Azure credential resolution: API key (from file)")
        return ResolvedModelFamily(
            "openai",
            (),
            azure_auth_mode="api_key",
            azure_api_version=resolved_api_version,
            azure_api_key=api_key_file,
            provider=provider,
            credentials=AzureApiKeyCredentialSource(api_key=api_key_file),
            hosting="azure",
            model=resolved_model,
            base_url=resolved_base_url,
        )

    api_key_env = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    if api_key_env:
        logger.info("Azure credential resolution: API key (from env)")
        return ResolvedModelFamily(
            "openai",
            (),
            azure_auth_mode="api_key",
            azure_api_version=resolved_api_version,
            azure_api_key=api_key_env,
            provider=provider,
            credentials=AzureApiKeyCredentialSource(api_key=api_key_env),
            hosting="azure",
            model=resolved_model,
            base_url=resolved_base_url,
        )

    raise ValueError(
        "Azure OpenAI credentials not found. Provide either "
        "all three Entra ID files (client_id, tenant_id, client_secret) "
        f"at {creds_path}/, or an API key via apitoken file or "
        "AZURE_OPENAI_API_KEY env var."
    )


def _resolve_bedrock_credential(env_name: str, file_name: str) -> str | None:
    """Resolve one Bedrock credential from its environment or mounted file."""
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    return _read_credential_file(str(Path(_llm_credentials_path()) / file_name))


def _resolve_bedrock(
    model: str | None,
    url: str | None,
    region: str | None,
) -> ResolvedModelFamily:
    return ResolvedModelFamily(
        "anthropic",
        (),
        hosting="bedrock",
        model=model,
        base_url=url,
        provider=BedrockProviderParameters(
            region=region or os.environ.get("AWS_REGION", "").strip() or "us-east-1"
        ),
        credentials=BedrockCredentialSource(
            access_key_id=_resolve_bedrock_credential("AWS_ACCESS_KEY_ID", "aws_access_key_id")
            or "",
            secret_access_key=_resolve_bedrock_credential(
                "AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"
            )
            or "",
            session_token=_resolve_bedrock_credential("AWS_SESSION_TOKEN", "aws_session_token"),
            role_arn=_resolve_bedrock_credential("role_arn", "role_arn"),
        ),
    )


def resolve_model_family() -> ResolvedModelFamily:
    """Read LIGHTSPEED_* vars and resolve typed provider and credential settings."""
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
