"""Offline contracts for incremental model routing to DeepAgents."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from lightspeed_agentic.config import ResolvedSDK, resolve_router_model
from lightspeed_agentic.providers.models import anthropic
from lightspeed_agentic.readiness import check_provider_env, run_readiness_checks
from lightspeed_agentic.types import DEFAULT_MODEL

if TYPE_CHECKING:
    from lightspeed_agentic.providers.model_router import ModelCreator


_ROUTES = [
    ("anthropic", None, "ANTHROPIC_MODEL", "deepagents"),
    ("vertex", "anthropic", "ANTHROPIC_MODEL", "deepagents"),
    ("bedrock", None, "ANTHROPIC_MODEL", "deepagents"),
    ("vertex", "google", "GEMINI_MODEL", "gemini"),
    ("vertex", "openai", "OPENAI_MODEL", "openai"),
    ("openai", None, "OPENAI_MODEL", "openai"),
    ("azure", None, "OPENAI_MODEL", "openai"),
]


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path) -> Iterator[None]:
    with patch.dict(os.environ, {"LIGHTSPEED_LLM_CREDENTIALS_PATH": str(tmp_path)}, clear=True):
        yield


def test_model_route_has_typed_construction_inputs_without_legacy_sdk() -> None:
    from lightspeed_agentic.providers.model_router import ModelParameters, ResolvedModelRoute

    route = ResolvedModelRoute(
        route="vertex/anthropic",
        model="test-model",
        create_model=anthropic.create_vertex_model,
        model_parameters=ModelParameters(
            model="test-model", project="test-project", region="us-east5"
        ),
        telemetry_provider="gcp.vertex_ai",
        structured_output_method="json_schema",
    )

    assert route.route == "vertex/anthropic"
    assert route.telemetry_provider == "gcp.vertex_ai"
    assert route.model_parameters.project == "test-project"


@pytest.mark.parametrize(("provider", "family", "legacy_env", "adapter"), _ROUTES)
@pytest.mark.parametrize("source", ["explicit", "canonical", "legacy"])
def test_route_preserves_legacy_dispatch_and_model_precedence(
    provider: str, family: str | None, legacy_env: str, adapter: str, source: str
) -> None:
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute, resolve_model_route

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": provider,
            "ANTHROPIC_MODEL": "anthropic-legacy",
            "GEMINI_MODEL": "google-legacy",
            "OPENAI_MODEL": "openai-legacy",
            "AZURE_OPENAI_API_KEY": "test-key",
        }
    )
    if family:
        os.environ["LIGHTSPEED_MODEL_PROVIDER"] = family
    os.environ[legacy_env] = "route-legacy"
    if source != "legacy":
        os.environ["LIGHTSPEED_MODEL"] = "canonical-model"
    explicit_model = "explicit-model" if source == "explicit" else None
    expected_model = {
        "explicit": "explicit-model",
        "canonical": "canonical-model",
        "legacy": "route-legacy",
    }[source]

    sdk, route = resolve_model_route(explicit_model)

    if adapter == "deepagents":
        assert sdk is None
        assert isinstance(route, ResolvedModelRoute)
        assert route.model == expected_model
        assert route.model_parameters.model == expected_model
        assert route.route == ("anthropic" if provider == "anthropic" else f"{provider}/anthropic")
    else:
        assert route is None
        assert isinstance(sdk, ResolvedSDK)
        assert sdk.name == adapter
        # Unmigrated callers retain the existing model-resolution path.
        assert resolve_router_model(sdk.name, explicit_model) == expected_model
    assert os.environ[legacy_env] == ("route-legacy" if source == "legacy" else "canonical-model")


@pytest.mark.parametrize(
    ("provider", "family", "expected_name", "expected_envs", "file_envs"),
    [
        ("anthropic", None, "anthropic", ("ANTHROPIC_API_KEY",), ()),
        (
            "vertex",
            "anthropic",
            "anthropic",
            ("GOOGLE_APPLICATION_CREDENTIALS",),
            ("GOOGLE_APPLICATION_CREDENTIALS",),
        ),
        ("bedrock", None, "anthropic", ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"), ()),
        (
            "vertex",
            "google",
            "google",
            ("GOOGLE_APPLICATION_CREDENTIALS",),
            ("GOOGLE_APPLICATION_CREDENTIALS",),
        ),
        (
            "vertex",
            "openai",
            "openai",
            ("GOOGLE_APPLICATION_CREDENTIALS",),
            ("GOOGLE_APPLICATION_CREDENTIALS",),
        ),
        ("openai", None, "openai", ("OPENAI_API_KEY",), ()),
        ("azure", None, "openai", ("AZURE_OPENAI_API_KEY",), ()),
    ],
)
def test_model_family_resolution_retains_readiness_configuration(
    provider: str,
    family: str | None,
    expected_name: str,
    expected_envs: tuple[str, ...],
    file_envs: tuple[str, ...],
) -> None:
    from lightspeed_agentic.config import ResolvedModelFamily, resolve_model_family

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": f" {provider.upper()} ",
            "LIGHTSPEED_MODEL": "configured-model",
            "AZURE_OPENAI_API_KEY": "test-key",
        }
    )
    if family:
        os.environ["LIGHTSPEED_MODEL_PROVIDER"] = f" {family.upper()} "

    result = resolve_model_family()

    assert isinstance(result, ResolvedModelFamily)
    assert result.name == expected_name
    assert result.expected_envs == expected_envs
    assert result.credential_file_envs == file_envs
    assert result.azure_auth_mode == ("api_key" if provider == "azure" else None)


@pytest.mark.parametrize("provider", ["anthropic", "vertex", "bedrock"])
def test_anthropic_routes_preserve_default_model(provider: str) -> None:
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute, resolve_model_route

    os.environ["LIGHTSPEED_PROVIDER"] = provider
    if provider == "vertex":
        os.environ["LIGHTSPEED_MODEL_PROVIDER"] = "anthropic"

    sdk, route = resolve_model_route()

    assert sdk is None
    assert isinstance(route, ResolvedModelRoute)
    assert route.model == DEFAULT_MODEL
    assert route.model_parameters.model == DEFAULT_MODEL


@pytest.mark.parametrize(
    ("provider", "family", "adapter", "model_env"),
    [
        ("vertex", "google", "gemini", "GEMINI_MODEL"),
        ("vertex", "openai", "openai", "OPENAI_MODEL"),
        ("openai", None, "openai", "OPENAI_MODEL"),
        ("azure", None, "openai", "OPENAI_MODEL"),
    ],
)
def test_unmigrated_routes_return_sdk_without_applying_anthropic_model_resolution(
    provider: str, family: str | None, adapter: str, model_env: str
) -> None:
    from lightspeed_agentic.providers.model_router import resolve_model_route

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": provider,
            "ANTHROPIC_MODEL": "must-not-be-copied",
            "AZURE_OPENAI_API_KEY": "test-key",
        }
    )
    if family:
        os.environ["LIGHTSPEED_MODEL_PROVIDER"] = family

    sdk, route = resolve_model_route()

    assert route is None
    assert isinstance(sdk, ResolvedSDK)
    assert sdk.name == adapter
    assert model_env not in os.environ


@pytest.mark.parametrize(
    ("provider", "constructor", "telemetry_provider", "method", "project", "region", "auth_token"),
    [
        (
            "anthropic",
            anthropic.create_direct_model,
            "anthropic",
            "json_schema",
            None,
            None,
            "test-bearer",
        ),
        (
            "vertex",
            anthropic.create_vertex_model,
            "gcp.vertex_ai",
            "json_schema",
            "operator-project",
            "us-east5",
            None,
        ),
        (
            "bedrock",
            anthropic.create_bedrock_model,
            "aws.bedrock",
            "function_calling",
            None,
            "us-east5",
            None,
        ),
    ],
)
def test_anthropic_route_selects_constructor_and_typed_parameters(
    provider: str,
    constructor: ModelCreator,
    telemetry_provider: str,
    method: str,
    project: str | None,
    region: str | None,
    auth_token: str | None,
) -> None:
    from lightspeed_agentic.providers.model_router import (
        ModelParameters,
        ResolvedModelRoute,
        resolve_model_route,
    )

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": provider,
            "LIGHTSPEED_MODEL_PROVIDER": "anthropic",
            "LIGHTSPEED_MODEL": "exact-model-or-profile-id",
            "LIGHTSPEED_PROVIDER_URL": "https://operator.example.test",
            "LIGHTSPEED_PROVIDER_PROJECT": "operator-project",
            "LIGHTSPEED_PROVIDER_REGION": "us-east5",
            "ANTHROPIC_AUTH_TOKEN": "test-bearer",
            # The operator route, not these stale flags, selects the constructor.
            "CLAUDE_CODE_USE_VERTEX": "1",
            "CLAUDE_CODE_USE_BEDROCK": "1",
        }
    )

    sdk, route = resolve_model_route()

    assert sdk is None
    assert isinstance(route, ResolvedModelRoute)
    assert route.create_model is constructor
    assert route.model_parameters == ModelParameters(
        model="exact-model-or-profile-id",
        base_url="https://operator.example.test",
        project=project,
        region=region,
        auth_token=auth_token,
    )
    assert route.telemetry_provider == telemetry_provider
    assert route.structured_output_method == method
    assert "test-bearer" not in repr(route)
    assert "test-bearer" not in repr(route.model_parameters)


@pytest.mark.parametrize(
    ("base_url", "method"),
    [
        (None, "function_calling"),
        ("https://api.anthropic.com/v1", "function_calling"),
        ("https://API.ANTHROPIC.COM./v1", "function_calling"),
        ("https://custom.example.test", "json_schema"),
    ],
)
def test_direct_route_selects_endpoint_specific_shaping(base_url: str | None, method: str) -> None:
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute, resolve_model_route

    if base_url:
        os.environ["ANTHROPIC_BASE_URL"] = base_url

    sdk, route = resolve_model_route()

    assert sdk is None
    assert isinstance(route, ResolvedModelRoute)
    assert route.create_model is anthropic.create_direct_model
    assert route.model_parameters.base_url == base_url
    assert route.structured_output_method == method


@pytest.mark.parametrize(
    ("provider", "project", "region"),
    [("vertex", "", "us-east5"), ("bedrock", None, "us-east-1")],
)
def test_cloud_anthropic_routes_preserve_existing_defaults(
    provider: str, project: str | None, region: str
) -> None:
    from lightspeed_agentic.providers.model_router import (
        ModelParameters,
        ResolvedModelRoute,
        resolve_model_route,
    )

    os.environ["LIGHTSPEED_PROVIDER"] = provider
    if provider == "vertex":
        os.environ["LIGHTSPEED_MODEL_PROVIDER"] = "anthropic"

    sdk, route = resolve_model_route()

    assert sdk is None
    assert isinstance(route, ResolvedModelRoute)
    assert route.model_parameters == ModelParameters(
        model=DEFAULT_MODEL, project=project, region=region
    )


def test_vertex_route_preserves_legacy_settings_and_file_readiness(tmp_path: Path) -> None:
    from lightspeed_agentic.providers.model_router import (
        ModelParameters,
        ResolvedModelRoute,
        resolve_model_route,
    )

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "vertex",
            "LIGHTSPEED_MODEL_PROVIDER": "anthropic",
            "ANTHROPIC_VERTEX_PROJECT_ID": "legacy-project",
            "CLOUD_ML_REGION": "global",
            "ANTHROPIC_BASE_URL": "https://legacy.example.test",
        }
    )
    sdk, route = resolve_model_route()

    assert sdk is None
    assert isinstance(route, ResolvedModelRoute)
    assert route.model_parameters == ModelParameters(
        model=DEFAULT_MODEL,
        project="legacy-project",
        region="global",
        base_url="https://legacy.example.test",
    )
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(
        tmp_path / "GOOGLE_APPLICATION_CREDENTIALS"
    )
    credential_envs = ("GOOGLE_APPLICATION_CREDENTIALS",)
    assert check_provider_env(credential_envs, credential_envs) != "ok"
    (tmp_path / "GOOGLE_APPLICATION_CREDENTIALS").write_text("test-credentials")
    assert check_provider_env(credential_envs, credential_envs) == "ok"


@pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
def test_router_preserves_credential_readiness(provider: str) -> None:
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute, resolve_model_route

    os.environ["LIGHTSPEED_PROVIDER"] = provider
    sdk, route = resolve_model_route()

    assert sdk is None
    assert isinstance(route, ResolvedModelRoute)
    credential_envs = (
        ("ANTHROPIC_API_KEY",)
        if provider == "anthropic"
        else ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    )
    assert check_provider_env(credential_envs) != "ok"
    os.environ.update(
        {
            "ANTHROPIC_API_KEY": "test-key",
            "AWS_ACCESS_KEY_ID": "test-access",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
        }
    )
    assert check_provider_env(credential_envs) == "ok"


@pytest.mark.parametrize("auth_mode", ["api_key", "entra_id"])
def test_azure_returns_existing_sdk_configuration_and_credentials(
    tmp_path: Path, auth_mode: str
) -> None:
    from lightspeed_agentic.providers.model_router import resolve_model_route

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "azure",
            "LIGHTSPEED_MODEL": "azure-deployment",
            "LIGHTSPEED_PROVIDER_URL": "https://azure.example.test",
            "LIGHTSPEED_PROVIDER_API_VERSION": "2025-03-01-preview",
        }
    )
    if auth_mode == "entra_id":
        for name in ("client_id", "tenant_id", "client_secret"):
            (tmp_path / name).write_text(f"test-{name}")
    else:
        (tmp_path / "apitoken").write_text("test-key")

    sdk, route = resolve_model_route()

    assert route is None
    assert isinstance(sdk, ResolvedSDK)
    assert sdk.name == "openai"
    assert sdk.azure_auth_mode == auth_mode
    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://azure.example.test"
    assert os.environ["AZURE_OPENAI_API_VERSION"] == "2025-03-01-preview"
    if auth_mode == "entra_id":
        assert sdk.azure_credentials == {
            "client_id": "test-client_id",
            "tenant_id": "test-tenant_id",
            "client_secret": "test-client_secret",
        }
    else:
        assert os.environ["AZURE_OPENAI_API_KEY"] == "test-key"
    assert run_readiness_checks(sdk)[0]


@pytest.mark.parametrize(
    ("provider", "family", "message"),
    [
        ("unknown", None, "Unknown provider"),
        ("vertex", None, "LIGHTSPEED_MODEL_PROVIDER is required"),
        ("vertex", "unknown", "Unknown LIGHTSPEED_MODEL_PROVIDER"),
    ],
)
def test_invalid_routes_fail_even_with_an_explicit_model(
    provider: str, family: str | None, message: str
) -> None:
    from lightspeed_agentic.providers.model_router import resolve_model_route

    os.environ["LIGHTSPEED_PROVIDER"] = provider
    if family:
        os.environ["LIGHTSPEED_MODEL_PROVIDER"] = family
    with pytest.raises(ValueError, match=message):
        resolve_model_route("explicit-model")


def test_router_does_not_import_optional_sdks_or_construct_models() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sys\n"
            "before = set(sys.modules)\n"
            "from lightspeed_agentic.providers.model_router import resolve_model_route\n"
            "os.environ['LIGHTSPEED_MODEL'] = 'test-model'\n"
            "for provider in ('anthropic', 'vertex', 'bedrock', 'openai'):\n"
            "    os.environ['LIGHTSPEED_PROVIDER'] = provider\n"
            "    os.environ['LIGHTSPEED_MODEL_PROVIDER'] = 'anthropic'\n"
            "    resolve_model_route()\n"
            "assert not any(name.split('.')[0] in "
            "{'anthropic', 'langchain_anthropic', 'langchain_core', 'langchain_aws', "
            "'langchain_google_vertexai', 'google', 'agents'} "
            "for name in set(sys.modules) - before)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
