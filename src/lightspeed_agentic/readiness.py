"""Readiness checks for provider credentials."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from lightspeed_agentic.config import (
    AnthropicCredentialSource,
    AzureApiKeyCredentialSource,
    AzureEntraCredentialSource,
    AzureProviderParameters,
    BedrockCredentialSource,
    BedrockProviderParameters,
    ResolvedModelFamily,
    ResolvedSDK,
    VertexCredentialSource,
    VertexProviderParameters,
)

if TYPE_CHECKING:
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute


def read_mounted_secret(path: Path) -> str | None:
    """Return stripped non-empty UTF-8 file contents.

    Returns ``None`` when the path is not a readable regular file, content is
    empty or whitespace-only, or decoding fails. Shared with MCP header resolution.
    """
    try:
        if not path.is_file():
            return None
        content = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return content or None


def read_first_mounted_secret_in_dir(directory: Path) -> str | None:
    """Return contents of the first regular file in ``directory`` (sorted by name)."""
    try:
        if not directory.is_dir():
            return None
        files = sorted((f for f in directory.iterdir() if f.is_file()), key=lambda f: f.name)
    except OSError:
        return None
    if not files:
        return None
    return read_mounted_secret(files[0])


def check_provider_env(
    expected_envs: tuple[str, ...],
    credential_file_envs: tuple[str, ...] = (),
) -> str:
    """R1: required credential env var(s) present; file paths exist and non-empty."""
    missing = [var for var in expected_envs if not os.environ.get(var, "").strip()]
    if missing:
        return f"error: missing {', '.join(missing)}"

    for var in credential_file_envs:
        path = os.environ.get(var, "").strip()
        if not path:
            continue
        if not os.path.isfile(path):
            return f"error: {var} file not found: {path}"
        content = read_mounted_secret(Path(path))
        if content is None:
            return f"error: {var} file is empty or unreadable: {path}"

    return "ok"


def _check_bedrock_credentials(source: object) -> str:
    """Require a complete static credential pair for Bedrock, with optional role ARN."""
    if not isinstance(source, BedrockCredentialSource):
        return "error: missing Bedrock credential source"
    missing = []
    if not source.access_key_id:
        missing.append("aws_access_key_id")
    if not source.secret_access_key:
        missing.append("aws_secret_access_key")
    if missing:
        return f"error: missing {', '.join(missing)}"
    return "ok"


def _check_azure_configuration(family: ResolvedModelFamily) -> str:
    """Validate resolved Azure startup settings without constructing SDK clients."""
    if not family.base_url or urlsplit(family.base_url).scheme != "https":
        return "error: Azure OpenAI requires a resolved HTTPS endpoint"
    if (
        not isinstance(family.provider, AzureProviderParameters)
        or not family.provider.api_version.strip()
    ):
        return "error: Azure OpenAI requires a resolved API version"
    if not isinstance(
        family.credentials, (AzureApiKeyCredentialSource, AzureEntraCredentialSource)
    ):
        return "error: Azure OpenAI requires resolved Azure credentials"
    if (
        isinstance(family.credentials, AzureApiKeyCredentialSource)
        and not family.credentials.api_key.strip()
    ):
        return "error: Azure OpenAI requires a nonempty resolved API key"
    return "ok"


def _check_model_route(route: ResolvedModelRoute) -> str:
    """Validate required connection settings for a resolved model route."""
    parameters = route.model_parameters
    match route.route:
        case "openai" | "openai-compatible":
            if not parameters.api_key or not parameters.api_key.strip():
                return f"error: missing API key for {route.route}"
            if route.route == "openai-compatible" and (
                not parameters.base_url or not parameters.base_url.strip()
            ):
                return "error: missing base URL for openai-compatible"
            return "ok"
        case "anthropic":
            if not isinstance(parameters.credentials, AnthropicCredentialSource):
                return "error: missing credential source for anthropic"
            if not parameters.credentials.api_key:
                return "error: missing API key for anthropic"
            return "ok"
        case "vertex/anthropic" | "vertex/google":
            if not isinstance(parameters.provider, VertexProviderParameters):
                return f"error: missing provider parameters for {route.route}"
            required = {
                "project": parameters.provider.project,
                "region": parameters.provider.region,
            }
        case "bedrock/anthropic":
            if not isinstance(parameters.provider, BedrockProviderParameters):
                return f"error: missing provider parameters for {route.route}"
            required = {"region": parameters.provider.region}
        case _:
            return "ok"

    missing = [name for name, value in required.items() if not value]
    if missing:
        return f"error: missing {', '.join(missing)} for {route.route}"
    if route.route in {"vertex/anthropic", "vertex/google"} and not isinstance(
        parameters.credentials, VertexCredentialSource
    ):
        return f"error: missing credential source for {route.route}"
    return "ok"


def run_readiness_checks(
    sdk: ResolvedSDK | ResolvedModelFamily,
    model_route: ResolvedModelRoute | None = None,
) -> tuple[bool, dict[str, str]]:
    """Return overall readiness and per-check status strings."""
    checks = {"provider_env": check_provider_env(sdk.expected_envs, sdk.credential_file_envs)}
    if isinstance(sdk, ResolvedModelFamily) and sdk.hosting == "azure":
        checks["provider_configuration"] = _check_azure_configuration(sdk)
    elif isinstance(sdk, ResolvedModelFamily) and isinstance(
        sdk.credentials, VertexCredentialSource
    ):
        credential_path = sdk.credentials.credentials_file
        if not credential_path.is_file() or read_mounted_secret(credential_path) is None:
            checks["provider_credentials"] = (
                f"error: credential file missing or empty: {credential_path}"
            )
        else:
            checks["provider_credentials"] = "ok"
    elif isinstance(sdk, ResolvedModelFamily) and sdk.hosting == "bedrock":
        checks["provider_credentials"] = _check_bedrock_credentials(sdk.credentials)
    if model_route is not None:
        checks["model_route"] = _check_model_route(model_route)
    return all(status == "ok" for status in checks.values()), checks
