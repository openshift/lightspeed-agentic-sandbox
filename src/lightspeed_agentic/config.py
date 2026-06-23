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
from typing import Any

logger = logging.getLogger(__name__)

LLM_CREDENTIALS_PATH = "/var/run/secrets/llm-credentials"
_SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"  # noqa: S105


def _llm_credentials_path() -> str:
    """Operator mount path; override for local e2e host mode when /var/run is not writable."""
    override = os.environ.get("LIGHTSPEED_LLM_CREDENTIALS_PATH", "").strip()
    return override or LLM_CREDENTIALS_PATH


@dataclasses.dataclass(frozen=True)
class McpServerConfig:
    """A single MCP server endpoint configured by the operator."""

    name: str
    url: str
    headers: dict[str, str] = dataclasses.field(default_factory=dict)


def _resolve_header_source(source: str) -> str:
    """Resolve a dynamic header source to its value."""
    if source == "ServiceAccountToken":
        try:
            with open(_SA_TOKEN_PATH) as f:
                return f.read().strip()
        except FileNotFoundError as err:
            raise ValueError(
                f"Header source 'ServiceAccountToken' requires {_SA_TOKEN_PATH}"
            ) from err
    raise ValueError(f"Unknown header source: {source!r}")


def _resolve_headers(raw: Any, index: int) -> dict[str, str]:
    """Accept headers as a dict or as the operator's list-of-objects format.

    Dict format (static):  {"header-name": "value"}
    List format (dynamic): [{"name": "header-name", "source": "ServiceAccountToken"}]
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        resolved: dict[str, str] = {}
        for j, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(
                    f"LIGHTSPEED_MCP_SERVERS[{index}].headers[{j}] must be an object"
                )
            hdr_name = item.get("name")
            if not hdr_name or not isinstance(hdr_name, str):
                raise ValueError(
                    f"LIGHTSPEED_MCP_SERVERS[{index}].headers[{j}] missing 'name'"
                )
            if "value" in item:
                resolved[hdr_name] = str(item["value"])
            elif "source" in item:
                resolved[hdr_name] = _resolve_header_source(item["source"])
            else:
                raise ValueError(
                    f"LIGHTSPEED_MCP_SERVERS[{index}].headers[{j}] needs 'value' or 'source'"
                )
        return resolved
    raise ValueError(f"LIGHTSPEED_MCP_SERVERS[{index}] 'headers' must be an object or array")


def resolve_mcp_servers() -> tuple[McpServerConfig, ...]:
    """Parse LIGHTSPEED_MCP_SERVERS env var into validated configs."""
    raw = os.environ.get("LIGHTSPEED_MCP_SERVERS", "").strip()
    if not raw:
        return ()

    entries: list[Any] = json.loads(raw)
    if not isinstance(entries, list):
        raise ValueError("LIGHTSPEED_MCP_SERVERS must be a JSON array")

    servers: list[McpServerConfig] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"LIGHTSPEED_MCP_SERVERS[{i}] must be an object")
        name = entry.get("name")
        url = entry.get("url")
        if not name or not isinstance(name, str):
            raise ValueError(f"LIGHTSPEED_MCP_SERVERS[{i}] missing required 'name' string")
        if not url or not isinstance(url, str):
            raise ValueError(f"LIGHTSPEED_MCP_SERVERS[{i}] missing required 'url' string")
        headers = _resolve_headers(entry.get("headers", {}), i)
        servers.append(McpServerConfig(name=name, url=url, headers=headers))

    return tuple(servers)


_DEFAULT_VERTEX_REGION = "us-east5"
_DEFAULT_BEDROCK_REGION = "us-east-1"


@dataclasses.dataclass(frozen=True)
class ResolvedSDK:
    """Result of resolving LIGHTSPEED_* env vars to an SDK backend."""

    name: str  # "claude", "gemini", "openai"
    expected_envs: tuple[str, ...]  # credential env vars expected from envFrom
    probe_url: str  # R2 reachability probe base URL
    mcp_servers: tuple[McpServerConfig, ...] = ()


def _setenv(key: str, value: str) -> None:
    os.environ[key] = value


def _setenv_if_value(key: str, value: str | None) -> None:
    if value:
        _setenv(key, value)


def _vertex_probe_url(region: str | None) -> str:
    r = region or _DEFAULT_VERTEX_REGION
    return f"https://{r}-aiplatform.googleapis.com/"


def _bedrock_probe_url(region: str | None) -> str:
    r = region or _DEFAULT_BEDROCK_REGION
    return f"https://bedrock-runtime.{r}.amazonaws.com/"


def _resolve_anthropic(model: str | None, url: str | None) -> ResolvedSDK:
    _setenv_if_value("ANTHROPIC_MODEL", model)
    _setenv_if_value("ANTHROPIC_BASE_URL", url)
    probe = url or "https://api.anthropic.com/"
    return ResolvedSDK(
        "claude",
        ("ANTHROPIC_API_KEY",),
        probe,
    )


def _resolve_vertex(
    model_provider: str | None,
    model: str | None,
    url: str | None,
    project: str | None,
    region: str | None,
) -> ResolvedSDK:
    if not model_provider:
        raise ValueError("LIGHTSPEED_MODEL_PROVIDER is required when LIGHTSPEED_PROVIDER=vertex")

    vertex_probe = _vertex_probe_url(region)

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
            return ResolvedSDK(
                "claude",
                ("GOOGLE_APPLICATION_CREDENTIALS",),
                url or vertex_probe,
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
            return ResolvedSDK(
                "gemini",
                ("GOOGLE_APPLICATION_CREDENTIALS",),
                url or vertex_probe,
            )
        case "openai":
            _setenv_if_value("OPENAI_MODEL", model)
            _setenv_if_value("OPENAI_BASE_URL", url)
            _setenv(
                "GOOGLE_APPLICATION_CREDENTIALS",
                f"{_llm_credentials_path()}/GOOGLE_APPLICATION_CREDENTIALS",
            )
            return ResolvedSDK(
                "openai",
                ("GOOGLE_APPLICATION_CREDENTIALS",),
                url or vertex_probe,
            )
        case _:
            raise ValueError(
                f"Unknown LIGHTSPEED_MODEL_PROVIDER: {model_provider!r}. "
                "Supported: anthropic, google, openai"
            )


def _resolve_openai(model: str | None, url: str | None) -> ResolvedSDK:
    _setenv_if_value("OPENAI_MODEL", model)
    _setenv_if_value("OPENAI_BASE_URL", url)
    return ResolvedSDK(
        "openai",
        ("OPENAI_API_KEY",),
        url or "https://api.openai.com/",
    )


def _resolve_azure(
    model: str | None,
    url: str | None,
    api_version: str | None,
) -> ResolvedSDK:
    _setenv_if_value("OPENAI_MODEL", model)
    _setenv_if_value("AZURE_OPENAI_ENDPOINT", url)
    _setenv_if_value("AZURE_OPENAI_API_VERSION", api_version)
    return ResolvedSDK(
        "openai",
        ("AZURE_OPENAI_API_KEY",),
        url or "",
    )


def _resolve_bedrock(
    model: str | None,
    url: str | None,
    region: str | None,
) -> ResolvedSDK:
    _setenv_if_value("ANTHROPIC_MODEL", model)
    _setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    _setenv_if_value("AWS_REGION", region)
    _setenv_if_value("ANTHROPIC_BASE_URL", url)
    return ResolvedSDK(
        "claude",
        ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        url or _bedrock_probe_url(region),
    )


def resolve_sdk() -> ResolvedSDK:
    """Read LIGHTSPEED_* env vars, set SDK-specific env vars, return resolved SDK."""
    provider = os.environ.get("LIGHTSPEED_PROVIDER", "").strip().lower() or "anthropic"
    model = os.environ.get("LIGHTSPEED_MODEL", "").strip() or None
    model_provider = os.environ.get("LIGHTSPEED_MODEL_PROVIDER", "").strip().lower() or None
    url = os.environ.get("LIGHTSPEED_PROVIDER_URL", "").strip() or None
    project = os.environ.get("LIGHTSPEED_PROVIDER_PROJECT", "").strip() or None
    region = os.environ.get("LIGHTSPEED_PROVIDER_REGION", "").strip() or None
    api_version = os.environ.get("LIGHTSPEED_PROVIDER_API_VERSION", "").strip() or None

    match provider:
        case "anthropic":
            sdk = _resolve_anthropic(model, url)
        case "vertex":
            sdk = _resolve_vertex(model_provider, model, url, project, region)
        case "openai":
            sdk = _resolve_openai(model, url)
        case "azure":
            sdk = _resolve_azure(model, url, api_version)
        case "bedrock":
            sdk = _resolve_bedrock(model, url, region)
        case _:
            raise ValueError(
                f"Unknown provider: {provider!r}. "
                "Supported: anthropic, vertex, openai, azure, bedrock"
            )

    mcp_servers = resolve_mcp_servers()
    if mcp_servers:
        sdk = dataclasses.replace(sdk, mcp_servers=mcp_servers)
        logger.info(
            "Resolved LIGHTSPEED_PROVIDER=%s → SDK=%s (MCP servers: %s)",
            provider,
            sdk.name,
            ", ".join(s.name for s in mcp_servers),
        )
    else:
        logger.info("Resolved LIGHTSPEED_PROVIDER=%s → SDK=%s", provider, sdk.name)
    return sdk
