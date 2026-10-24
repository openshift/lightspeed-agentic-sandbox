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

from lightspeed_agentic.providers.models import anthropic
from lightspeed_agentic.readiness import check_provider_env, run_readiness_checks

if TYPE_CHECKING:
    from lightspeed_agentic.providers.model_router import ModelCreator


_ROUTES = [
    ("anthropic", None, "ANTHROPIC_MODEL"),
    ("vertex", "anthropic", "ANTHROPIC_MODEL"),
    ("bedrock", None, "ANTHROPIC_MODEL"),
    ("vertex", "google", "GEMINI_MODEL"),
    ("openai", None, "OPENAI_MODEL"),
    ("azure", None, "OPENAI_MODEL"),
]


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path) -> Iterator[None]:
    with patch.dict(os.environ, {"LIGHTSPEED_LLM_CREDENTIALS_PATH": str(tmp_path)}, clear=True):
        yield


def test_model_route_has_typed_construction_inputs_without_legacy_sdk() -> None:
    from lightspeed_agentic.config import VertexProviderParameters
    from lightspeed_agentic.providers.model_router import ModelParameters, ResolvedModelRoute

    route = ResolvedModelRoute(
        route="vertex/anthropic",
        model="test-model",
        create_model=anthropic.create_vertex_model,
        model_parameters=ModelParameters(
            model="test-model",
            provider=VertexProviderParameters(project="test-project", region="us-east5"),
        ),
        telemetry_provider="gcp.vertex_ai",
        structured_output_method="json_schema",
    )

    assert route.route == "vertex/anthropic"
    assert route.telemetry_provider == "gcp.vertex_ai"
    assert route.model_parameters.provider.project == "test-project"


@pytest.mark.parametrize(("provider", "family", "legacy_env"), _ROUTES)
@pytest.mark.parametrize("source", ["explicit", "canonical", "legacy"])
def test_route_preserves_model_precedence(
    provider: str, family: str | None, legacy_env: str, source: str
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

    if source == "legacy" and provider != "azure":
        # Migrated routes use only the generic model setting, not SDK model env vars.
        with pytest.raises(ValueError, match="No model configured"):
            resolve_model_route(explicit_model)
        return

    sdk, route = resolve_model_route(explicit_model)

    assert sdk is None
    assert isinstance(route, ResolvedModelRoute)
    assert route.model == expected_model
    assert route.model_parameters.model == expected_model
    expected_route = (
        "anthropic"
        if provider == "anthropic"
        else "openai"
        if provider == "openai"
        else "azure"
        if provider == "azure"
        else f"{provider}/{family or 'anthropic'}"
    )
    assert route.route == expected_route
    assert os.environ[legacy_env] == "route-legacy"


@pytest.mark.parametrize(
    ("provider", "family", "expected_name", "expected_envs", "file_envs"),
    [
        ("anthropic", None, "anthropic", ("ANTHROPIC_API_KEY",), ()),
        (
            "vertex",
            "anthropic",
            "anthropic",
            (),
            (),
        ),
        ("bedrock", None, "anthropic", (), ()),
        (
            "vertex",
            "google",
            "google",
            (),
            (),
        ),
        (
            "vertex",
            "openai",
            "openai",
            ("GOOGLE_APPLICATION_CREDENTIALS",),
            ("GOOGLE_APPLICATION_CREDENTIALS",),
        ),
        ("openai", None, "openai", ("OPENAI_API_KEY",), ()),
        ("azure", None, "openai", (), ()),
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
    assert result.model == "configured-model"
    expected_hosting = {
        "anthropic": "anthropic",
        "vertex": "vertex",
        "bedrock": "bedrock",
        "openai": "openai",
        "azure": "azure",
    }[provider]
    assert result.hosting == expected_hosting
    assert result.expected_envs == expected_envs
    assert result.credential_file_envs == file_envs
    assert result.azure_auth_mode == ("api_key" if provider == "azure" else None)


@pytest.mark.parametrize(
    ("provider", "family"),
    [
        ("anthropic", None),
        ("vertex", "anthropic"),
        ("bedrock", None),
        ("vertex", "google"),
        ("vertex", "openai"),
        ("openai", None),
        ("azure", None),
    ],
)
def test_model_routes_require_a_configured_model(provider: str, family: str | None) -> None:
    from lightspeed_agentic.providers.model_router import resolve_model_route

    os.environ["LIGHTSPEED_PROVIDER"] = provider
    if family:
        os.environ["LIGHTSPEED_MODEL_PROVIDER"] = family
    if provider == "azure":
        os.environ["AZURE_OPENAI_API_KEY"] = "test-key"

    with pytest.raises(ValueError, match="No model configured"):
        resolve_model_route()


@pytest.mark.parametrize("source", ["explicit", "canonical", "legacy"])
@pytest.mark.parametrize("ambient_settings", [False, True])
def test_vertex_openai_route_is_unimplemented_without_mutating_environment(
    source: str, ambient_settings: bool
) -> None:
    from lightspeed_agentic.providers.model_router import resolve_model_route

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "vertex",
            "LIGHTSPEED_MODEL_PROVIDER": "openai",
            "LIGHTSPEED_PROVIDER_URL": "https://vertex-openai.example.test/v1",
        }
    )
    if ambient_settings:
        os.environ.update(
            {
                "OPENAI_MODEL": "ambient-model",
                "OPENAI_BASE_URL": "https://ambient.example.test/v1",
                "GOOGLE_APPLICATION_CREDENTIALS": "/ambient/credentials.json",
            }
        )
    if source == "canonical":
        os.environ["LIGHTSPEED_MODEL"] = "configured-model"
    elif source == "legacy":
        os.environ["OPENAI_MODEL"] = "legacy-model"
    environment = dict(os.environ)

    with pytest.raises(NotImplementedError, match=r"Vertex/OpenAI.*OLS-4472"):
        resolve_model_route("explicit-model" if source == "explicit" else None)

    assert dict(os.environ) == environment


@pytest.mark.parametrize(
    ("provider_url", "legacy_url", "canonical_model", "override", "expected_url", "expected_model"),
    [
        (
            "https://vllm.example.test/serving/v1",
            "https://api.openai.com/v1",
            "org/Gemma-4-served:stable",
            None,
            "https://vllm.example.test/serving/v1",
            "org/Gemma-4-served:stable",
        ),
        (None, "http://localhost:8000/v1", None, None, "http://localhost:8000/v1", "legacy-model"),
        (
            "https://vllm.example.test/v1",
            "https://ignored.example.test/v1",
            "configured-model",
            "explicit/Model:tag",
            "https://vllm.example.test/v1",
            "explicit/Model:tag",
        ),
    ],
)
def test_compatible_openai_route_preserves_resolved_settings_without_env_mutation(
    provider_url: str | None,
    legacy_url: str,
    canonical_model: str | None,
    override: str | None,
    expected_url: str,
    expected_model: str,
) -> None:
    from lightspeed_agentic.config import resolve_model_family
    from lightspeed_agentic.providers.model_router import ModelParameters, resolve_model_route
    from lightspeed_agentic.providers.models.openai import (
        create_openai_compatible_model,
        openai_compatible_structured_output_schema,
    )

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "openai",
            "OPENAI_MODEL": "legacy-model",
            "OPENAI_BASE_URL": legacy_url,
            "OPENAI_API_KEY": "EMPTY",
        }
    )
    if provider_url is not None:
        os.environ["LIGHTSPEED_PROVIDER_URL"] = provider_url
    if canonical_model is not None:
        os.environ["LIGHTSPEED_MODEL"] = canonical_model
    family = resolve_model_family()
    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "azure",
            "LIGHTSPEED_MODEL": "changed-model",
            "LIGHTSPEED_PROVIDER_URL": "https://api.openai.com/v1",
            "OPENAI_BASE_URL": "https://changed.example.test/v1",
            "OPENAI_MODEL": "changed-legacy-model",
            "OPENAI_API_KEY": "ambient-key-must-not-win",
        }
    )
    before = dict(os.environ)

    sdk, route = resolve_model_route(override, model_family=family)

    assert sdk is None
    assert route is not None
    assert route.route == "openai-compatible"
    assert route.model == expected_model
    assert route.create_model is create_openai_compatible_model
    assert route.model_parameters == ModelParameters(
        model=expected_model, base_url=expected_url, api_key="EMPTY"
    )
    assert route.telemetry_provider == "openai"
    assert route.structured_output_method == "json_schema"
    assert route.structured_output_schema_mode == "raw_json_schema"
    assert route.translate_output_schema is openai_compatible_structured_output_schema
    assert "EMPTY" not in repr(route)
    assert run_readiness_checks(family, route) == (
        True,
        {"provider_env": "ok", "model_route": "ok"},
    )
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    "base_url",
    [None, "https://api.openai.com/v1", "https://API.OPENAI.COM/v1/"],
)
def test_native_openai_routes_to_deepagents(base_url: str | None) -> None:
    from lightspeed_agentic.providers.model_router import (
        ModelParameters,
        ResolvedModelRoute,
        resolve_model_route,
    )
    from lightspeed_agentic.providers.models.openai import (
        create_native_openai_model,
        native_openai_structured_output_schema,
    )

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "openai",
            "LIGHTSPEED_MODEL": "gpt-4.1-mini",
            "OPENAI_API_KEY": "resolved-openai-key",
        }
    )
    if base_url is not None:
        os.environ["LIGHTSPEED_PROVIDER_URL"] = base_url

    sdk, route = resolve_model_route()

    assert sdk is None
    assert isinstance(route, ResolvedModelRoute)
    assert route.route == "openai"
    assert route.model == "gpt-4.1-mini"
    assert route.create_model is create_native_openai_model
    assert route.model_parameters == ModelParameters(
        model="gpt-4.1-mini",
        base_url=base_url,
        api_key="resolved-openai-key",
    )
    assert route.structured_output_schema_mode == "raw_json_schema"
    assert route.translate_output_schema is native_openai_structured_output_schema
    assert "OPENAI_MODEL" not in os.environ
    assert "OPENAI_BASE_URL" not in os.environ


def test_native_openai_rejects_http_endpoint_before_routing() -> None:
    from lightspeed_agentic.providers.model_router import resolve_model_route

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "openai",
            "LIGHTSPEED_MODEL": "gpt-4.1-mini",
            "LIGHTSPEED_PROVIDER_URL": "http://api.openai.com/v1",
            "OPENAI_API_KEY": "resolved-openai-key",
        }
    )

    with pytest.raises(ValueError, match="must use HTTPS"):
        resolve_model_route()

    assert "OPENAI_BASE_URL" not in os.environ


@pytest.mark.parametrize(
    ("provider", "constructor", "telemetry_provider", "method", "project", "region"),
    [
        (
            "anthropic",
            anthropic.create_direct_model,
            "anthropic",
            "json_schema",
            None,
            None,
        ),
        (
            "vertex",
            anthropic.create_vertex_model,
            "gcp.vertex_ai",
            "json_schema",
            "operator-project",
            "us-east5",
        ),
        (
            "bedrock",
            anthropic.create_bedrock_model,
            "aws.bedrock",
            "function_calling",
            None,
            "us-east5",
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
    tmp_path: Path,
) -> None:
    from lightspeed_agentic.config import (
        AnthropicCredentialSource,
        BedrockCredentialSource,
        BedrockProviderParameters,
        VertexCredentialSource,
        VertexProviderParameters,
    )
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute, resolve_model_route

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": provider,
            "LIGHTSPEED_MODEL_PROVIDER": "anthropic",
            "LIGHTSPEED_MODEL": "exact-model-or-profile-id",
            "LIGHTSPEED_PROVIDER_URL": "https://operator.example.test",
            "LIGHTSPEED_PROVIDER_PROJECT": "operator-project",
            "LIGHTSPEED_PROVIDER_REGION": "us-east5",
            "ANTHROPIC_API_KEY": "test-api-key",
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
    parameters = route.model_parameters
    assert parameters.model == "exact-model-or-profile-id"
    assert parameters.base_url == "https://operator.example.test"
    if provider == "anthropic":
        assert parameters.provider is None
        assert parameters.credentials == AnthropicCredentialSource(
            api_key="test-api-key",
            auth_token="test-bearer",  # noqa: S106 - Test credential value.
        )
    elif provider == "vertex":
        assert parameters.provider == VertexProviderParameters(project=project, region=region)
        assert parameters.credentials == VertexCredentialSource(
            tmp_path / "GOOGLE_APPLICATION_CREDENTIALS"
        )
    else:
        assert parameters.provider == BedrockProviderParameters(region=region)
        assert isinstance(parameters.credentials, BedrockCredentialSource)
    assert route.telemetry_provider == telemetry_provider
    assert route.structured_output_method == method
    assert "test-bearer" not in repr(route)
    assert "test-bearer" not in repr(parameters)


def test_vertex_google_route_selects_gemini_constructor_and_typed_parameters(
    tmp_path: Path,
) -> None:
    from lightspeed_agentic.config import (
        VertexCredentialSource,
        VertexProviderParameters,
    )
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute, resolve_model_route
    from lightspeed_agentic.providers.models.gemini import create_vertex_gemini_model

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "vertex",
            "LIGHTSPEED_MODEL_PROVIDER": "google",
            "LIGHTSPEED_MODEL": "gemini-2.5-flash",
            "LIGHTSPEED_PROVIDER_PROJECT": "test-project",
            "LIGHTSPEED_PROVIDER_REGION": "us-central1",
            "LIGHTSPEED_PROVIDER_URL": "https://vertex-proxy.example.test/custom",
        }
    )

    sdk, route = resolve_model_route()

    assert sdk is None
    assert isinstance(route, ResolvedModelRoute)
    assert route.route == "vertex/google"
    assert route.model == "gemini-2.5-flash"
    assert route.create_model is create_vertex_gemini_model
    assert route.model_parameters.model == "gemini-2.5-flash"
    assert route.model_parameters.base_url == "https://vertex-proxy.example.test/custom"
    assert route.model_parameters.provider == VertexProviderParameters(
        project="test-project", region="us-central1"
    )
    assert route.model_parameters.credentials == VertexCredentialSource(
        tmp_path / "GOOGLE_APPLICATION_CREDENTIALS"
    )
    assert route.telemetry_provider == "gcp.vertex_ai"
    assert route.structured_output_method == "json_schema"
    assert route.structured_output_schema_mode == "raw_json_schema"


def test_route_uses_resolved_anthropic_hosting_and_settings_not_current_environment(
    tmp_path: Path,
) -> None:
    from lightspeed_agentic.config import (
        VertexCredentialSource,
        VertexProviderParameters,
        resolve_model_family,
    )
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute, resolve_model_route

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "vertex",
            "LIGHTSPEED_MODEL_PROVIDER": "anthropic",
            "LIGHTSPEED_MODEL": "snapshot-model",
            "LIGHTSPEED_PROVIDER_URL": "https://resolved-vertex.example.test",
            "LIGHTSPEED_PROVIDER_PROJECT": "resolved-project",
            "LIGHTSPEED_PROVIDER_REGION": "us-west1",
        }
    )
    model_family = resolve_model_family()

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "bedrock",
            "ANTHROPIC_MODEL": "changed-model",
            "ANTHROPIC_BASE_URL": "https://changed.example.test",
            "ANTHROPIC_VERTEX_PROJECT_ID": "changed-project",
            "CLOUD_ML_REGION": "europe-west2",
            "AWS_REGION": "eu-west-1",
        }
    )

    sdk, route = resolve_model_route(model_family=model_family)

    assert sdk is None
    assert isinstance(route, ResolvedModelRoute)
    assert route.create_model is anthropic.create_vertex_model
    assert route.model_parameters.model == "snapshot-model"
    assert route.model_parameters.base_url == "https://resolved-vertex.example.test"
    assert route.model_parameters.provider == VertexProviderParameters(
        project="resolved-project", region="us-west1"
    )
    assert route.model_parameters.credentials == VertexCredentialSource(
        tmp_path / "GOOGLE_APPLICATION_CREDENTIALS"
    )


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

    os.environ["LIGHTSPEED_MODEL"] = "configured-model"
    if base_url:
        os.environ["LIGHTSPEED_PROVIDER_URL"] = base_url

    sdk, route = resolve_model_route()

    assert sdk is None
    assert isinstance(route, ResolvedModelRoute)
    assert route.create_model is anthropic.create_direct_model
    assert route.model_parameters.base_url == base_url
    assert route.structured_output_method == method


@pytest.mark.parametrize(
    ("provider", "project", "region"),
    [("vertex", None, "global"), ("bedrock", None, "us-east-1")],
)
def test_cloud_anthropic_routes_preserve_region_defaults(
    provider: str, project: str | None, region: str
) -> None:
    from lightspeed_agentic.config import (
        BedrockProviderParameters,
        VertexProviderParameters,
    )
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute, resolve_model_route

    os.environ["LIGHTSPEED_PROVIDER"] = provider
    if provider == "vertex":
        os.environ["LIGHTSPEED_MODEL_PROVIDER"] = "anthropic"
    os.environ["LIGHTSPEED_MODEL"] = "configured-model"

    sdk, route = resolve_model_route()

    assert sdk is None
    assert isinstance(route, ResolvedModelRoute)
    expected_provider = (
        VertexProviderParameters(project=project, region=region)
        if provider == "vertex"
        else BedrockProviderParameters(region=region)
    )
    assert route.model_parameters.provider == expected_provider


def test_vertex_route_uses_typed_settings_and_file_readiness(tmp_path: Path) -> None:
    from lightspeed_agentic.config import (
        VertexCredentialSource,
        VertexProviderParameters,
        resolve_model_family,
    )
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute, resolve_model_route

    credential_file = tmp_path / "GOOGLE_APPLICATION_CREDENTIALS"
    credential_file.write_text("test-credentials", encoding="utf-8")
    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "vertex",
            "LIGHTSPEED_MODEL_PROVIDER": "anthropic",
            "LIGHTSPEED_MODEL": "configured-model",
            "LIGHTSPEED_PROVIDER_PROJECT": "configured-project",
            "LIGHTSPEED_PROVIDER_REGION": "us-east5",
            "LIGHTSPEED_PROVIDER_URL": "https://vertex.example.test",
            "LIGHTSPEED_LLM_CREDENTIALS_PATH": str(tmp_path),
            "ANTHROPIC_MODEL": "ignored-legacy-model",
            "ANTHROPIC_VERTEX_PROJECT_ID": "ignored-legacy-project",
            "CLOUD_ML_REGION": "ignored-legacy-region",
            "ANTHROPIC_BASE_URL": "https://ignored.example.test",
        }
    )
    env_before = dict(os.environ)
    family = resolve_model_family()
    sdk, route = resolve_model_route(model_family=family)

    assert sdk is None
    assert isinstance(route, ResolvedModelRoute)
    assert route.model_parameters.model == "configured-model"
    assert route.model_parameters.provider == VertexProviderParameters(
        project="configured-project", region="us-east5"
    )
    assert route.model_parameters.base_url == "https://vertex.example.test"
    assert route.model_parameters.credentials == VertexCredentialSource(credential_file)
    assert dict(os.environ) == env_before
    ready, checks = run_readiness_checks(family, route)
    assert ready is True
    assert checks == {
        "provider_env": "ok",
        "provider_credentials": "ok",
        "model_route": "ok",
    }


def test_model_parameters_carry_resolved_anthropic_credentials_without_repr_leaks() -> None:
    from lightspeed_agentic.config import AnthropicCredentialSource
    from lightspeed_agentic.providers.model_router import resolve_model_route

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "anthropic",
            "LIGHTSPEED_MODEL": "claude-sonnet-4-6",
            "ANTHROPIC_API_KEY": "do-not-copy-this-api-key",
            "ANTHROPIC_AUTH_TOKEN": "do-not-copy-this-token",
        }
    )

    _sdk, route = resolve_model_route()

    assert route is not None
    assert route.model_parameters.credentials == AnthropicCredentialSource(
        api_key="do-not-copy-this-api-key",
        auth_token="do-not-copy-this-token",  # noqa: S106 - Test credential value.
    )
    assert "do-not-copy-this-api-key" not in repr(route)
    assert "do-not-copy-this-token" not in repr(route)
    assert route.model_parameters.provider is None


def test_vertex_route_carries_provider_settings_and_unparsed_credential_file(
    tmp_path: Path,
) -> None:
    from lightspeed_agentic.config import (
        VertexCredentialSource,
        VertexProviderParameters,
    )
    from lightspeed_agentic.providers.model_router import resolve_model_route

    credential_file = tmp_path / "GOOGLE_APPLICATION_CREDENTIALS"
    credential_file.write_text("not parsed during routing", encoding="utf-8")
    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "vertex",
            "LIGHTSPEED_MODEL_PROVIDER": "google",
            "LIGHTSPEED_MODEL": "gemini-2.5-flash",
            "LIGHTSPEED_PROVIDER_PROJECT": "test-project",
            "LIGHTSPEED_PROVIDER_REGION": "us-central1",
            "LIGHTSPEED_LLM_CREDENTIALS_PATH": str(tmp_path),
        }
    )

    _sdk, route = resolve_model_route()

    assert route is not None
    assert route.model_parameters.provider == VertexProviderParameters(
        project="test-project", region="us-central1"
    )
    assert route.model_parameters.credentials == VertexCredentialSource(
        credentials_file=credential_file
    )


def test_bedrock_route_carries_bedrock_provider_and_credential_settings() -> None:
    from lightspeed_agentic.config import (
        BedrockCredentialSource,
        BedrockProviderParameters,
    )
    from lightspeed_agentic.providers.model_router import resolve_model_route

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "bedrock",
            "LIGHTSPEED_MODEL": "claude-sonnet-4-6",
            "LIGHTSPEED_PROVIDER_REGION": "us-east-1",
        }
    )

    _sdk, route = resolve_model_route()

    assert route is not None
    assert route.model_parameters.provider == BedrockProviderParameters(region="us-east-1")
    assert isinstance(route.model_parameters.credentials, BedrockCredentialSource)


@pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
def test_router_preserves_credential_readiness(provider: str) -> None:
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute, resolve_model_route

    os.environ["LIGHTSPEED_PROVIDER"] = provider
    os.environ["LIGHTSPEED_MODEL"] = "configured-model"
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
@pytest.mark.parametrize(
    ("configured_api_version", "expected_api_version"),
    [("2025-03-01-preview", "2025-03-01-preview"), (None, "2024-08-01-preview")],
)
def test_azure_route_preserves_resolved_auth_version_and_readiness(
    tmp_path: Path,
    auth_mode: str,
    configured_api_version: str | None,
    expected_api_version: str,
) -> None:
    from lightspeed_agentic.config import (
        AzureApiKeyCredentialSource,
        AzureEntraCredentialSource,
        AzureProviderParameters,
        resolve_model_family,
    )
    from lightspeed_agentic.providers.model_router import resolve_model_route
    from lightspeed_agentic.providers.models.openai import bind_azure_output, create_azure_model

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "azure",
            "LIGHTSPEED_MODEL": "azure-deployment",
            "LIGHTSPEED_PROVIDER_URL": "https://azure.example.test",
        }
    )
    if configured_api_version:
        os.environ["LIGHTSPEED_PROVIDER_API_VERSION"] = configured_api_version
    if auth_mode == "entra_id":
        for name in ("client_id", "tenant_id", "client_secret"):
            (tmp_path / name).write_text(f"test-{name}")
    else:
        (tmp_path / "apitoken").write_text("test-key")

    family = resolve_model_family()
    environment = dict(os.environ)
    sdk, route = resolve_model_route(model_family=family)

    assert sdk is None
    assert route is not None
    assert route.route == "azure"
    assert route.create_model is create_azure_model
    assert route.bind_structured_output is bind_azure_output
    assert route.model == "azure-deployment"
    assert route.model_parameters.model == "azure-deployment"
    assert route.model_parameters.base_url == "https://azure.example.test"
    assert route.model_parameters.provider == AzureProviderParameters(expected_api_version)
    assert route.telemetry_provider == "openai"
    assert route.structured_output_method == "json_schema"
    assert route.structured_output_schema_mode == "raw_json_schema"
    if auth_mode == "entra_id":
        assert route.model_parameters.credentials == AzureEntraCredentialSource(
            "test-tenant_id", "test-client_id", "test-client_secret"
        )
    else:
        assert route.model_parameters.credentials == AzureApiKeyCredentialSource("test-key")
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
    }
    prepared = route.translate_output_schema(schema)
    assert prepared["strict"] is (expected_api_version == "2025-03-01-preview")
    assert prepared["schema"]["properties"]["status"] == {"type": "string"}
    assert run_readiness_checks(family, route) == (
        True,
        {"provider_env": "ok", "provider_configuration": "ok", "model_route": "ok"},
    )
    assert dict(os.environ) == environment


@pytest.mark.parametrize("auth_mode", ["api-key", "entra"])
def test_azure_route_uses_snapshot_despite_later_environment_changes(
    tmp_path: Path, auth_mode: str
) -> None:
    from lightspeed_agentic.config import resolve_model_family
    from lightspeed_agentic.providers.model_router import resolve_model_route

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "azure",
            "LIGHTSPEED_MODEL": "resolved-model",
            "LIGHTSPEED_PROVIDER_URL": "https://resolved.example.test/proxy",
        }
    )
    if auth_mode == "api-key":
        (tmp_path / "apitoken").write_text("resolved-key")
    else:
        for name in ("client_id", "tenant_id", "client_secret"):
            (tmp_path / name).write_text(f"resolved-{name}")
    family = resolve_model_family()
    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "anthropic",
            "LIGHTSPEED_MODEL": "wrong-model",
            "LIGHTSPEED_PROVIDER_URL": "https://wrong.example.test",
            "AZURE_OPENAI_API_KEY": "wrong-key",
            "AZURE_OPENAI_API_VERSION": "wrong-version",
        }
    )
    environment = dict(os.environ)

    sdk, route = resolve_model_route("override-deployment", model_family=family)

    assert sdk is None
    assert route.route == "azure"
    assert route.model == route.model_parameters.model == "override-deployment"
    assert route.model_parameters.base_url == "https://resolved.example.test/proxy"
    assert route.model_parameters.credentials is family.credentials
    assert route.model_parameters.provider is family.provider
    assert run_readiness_checks(family, route)[0] is True
    assert dict(os.environ) == environment


@pytest.mark.parametrize("field", ["provider", "credentials"])
def test_azure_router_leaves_invalid_configuration_to_readiness(tmp_path: Path, field: str) -> None:
    from dataclasses import replace

    from lightspeed_agentic.config import resolve_model_family
    from lightspeed_agentic.providers.model_router import resolve_model_route

    os.environ.update(
        {
            "LIGHTSPEED_PROVIDER": "azure",
            "LIGHTSPEED_MODEL": "deployment",
            "LIGHTSPEED_PROVIDER_URL": "https://resolved.example.test",
        }
    )
    (tmp_path / "apitoken").write_text("resolved-key")
    family = replace(resolve_model_family(), **{field: None})

    sdk, route = resolve_model_route(model_family=family)

    assert sdk is None
    assert route.route == "azure"
    ready, checks = run_readiness_checks(family, route)
    assert ready is False
    assert checks["provider_configuration"] == (
        "error: Azure OpenAI requires a resolved API version"
        if field == "provider"
        else "error: Azure OpenAI requires resolved Azure credentials"
    )


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
            "os.environ['AZURE_OPENAI_API_KEY'] = 'offline-key'\n"
            "for provider in ('anthropic', 'vertex', 'bedrock', 'openai', 'azure'):\n"
            "    os.environ['LIGHTSPEED_PROVIDER'] = provider\n"
            "    os.environ['LIGHTSPEED_MODEL_PROVIDER'] = 'anthropic'\n"
            "    resolve_model_route()\n"
            "os.environ['LIGHTSPEED_PROVIDER'] = 'openai'\n"
            "os.environ['LIGHTSPEED_PROVIDER_URL'] = 'https://vllm.example.test/v1'\n"
            "sdk, route = resolve_model_route()\n"
            "assert sdk is None and route.route == 'openai-compatible'\n"
            "assert not any(name.split('.')[0] in "
            "{'anthropic', 'langchain_anthropic', 'langchain_core', 'langchain_aws', "
            "'langchain_google_vertexai', 'langchain_openai', "
            "'google', 'agents', 'azure', 'openai'} "
            "for name in set(sys.modules) - before)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
