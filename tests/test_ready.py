"""Tests for provider readiness checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from lightspeed_agentic.config import ResolvedModelFamily, ResolvedSDK
from lightspeed_agentic.readiness import (
    check_provider_env,
    read_first_mounted_secret_in_dir,
    read_mounted_secret,
    run_readiness_checks,
)

_ANTHROPIC_DIRECT = ResolvedSDK(
    "deepagents",
    ("ANTHROPIC_API_KEY",),
)

_VERTEX_ANTHROPIC = ResolvedSDK(
    "deepagents",
    ("GOOGLE_APPLICATION_CREDENTIALS",),
    ("GOOGLE_APPLICATION_CREDENTIALS",),
)

_VERTEX_GOOGLE = ResolvedSDK(
    "gemini",
    ("GOOGLE_APPLICATION_CREDENTIALS",),
    ("GOOGLE_APPLICATION_CREDENTIALS",),
)

_OPENAI_DIRECT = ResolvedSDK(
    "openai",
    ("OPENAI_API_KEY",),
)

_BEDROCK = ResolvedSDK(
    "deepagents",
    ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
)


def test_check_provider_env_anthropic_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert check_provider_env(_ANTHROPIC_DIRECT.expected_envs) == "ok"


def test_check_provider_env_anthropic_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "error: missing" in check_provider_env(_ANTHROPIC_DIRECT.expected_envs)


def test_check_provider_env_anthropic_wrong_cred_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/some/path")
    assert "error: missing" in check_provider_env(_ANTHROPIC_DIRECT.expected_envs)


def test_check_provider_env_vertex_anthropic_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cred = tmp_path / "gac.json"
    cred.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred))
    assert (
        check_provider_env(_VERTEX_ANTHROPIC.expected_envs, _VERTEX_ANTHROPIC.credential_file_envs)
        == "ok"
    )


def test_check_provider_env_vertex_anthropic_wrong_cred_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert "error: missing" in check_provider_env(
        _VERTEX_ANTHROPIC.expected_envs, _VERTEX_ANTHROPIC.credential_file_envs
    )


def test_check_provider_env_vertex_google_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cred = tmp_path / "gac.json"
    cred.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred))
    assert (
        check_provider_env(_VERTEX_GOOGLE.expected_envs, _VERTEX_GOOGLE.credential_file_envs)
        == "ok"
    )


def test_check_provider_env_vertex_google_wrong_cred_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert "error: missing" in check_provider_env(
        _VERTEX_GOOGLE.expected_envs, _VERTEX_GOOGLE.credential_file_envs
    )


def test_check_provider_env_credential_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/gac.json")
    result = check_provider_env(
        _VERTEX_ANTHROPIC.expected_envs, _VERTEX_ANTHROPIC.credential_file_envs
    )
    assert "file not found" in result


def test_check_provider_env_credential_file_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cred = tmp_path / "gac.json"
    cred.write_text("", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred))
    result = check_provider_env(
        _VERTEX_ANTHROPIC.expected_envs, _VERTEX_ANTHROPIC.credential_file_envs
    )
    assert "file is empty or unreadable" in result


def test_check_provider_env_credential_file_whitespace_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cred = tmp_path / "gac.json"
    cred.write_text("   \n", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred))
    result = check_provider_env(
        _VERTEX_ANTHROPIC.expected_envs, _VERTEX_ANTHROPIC.credential_file_envs
    )
    assert "file is empty or unreadable" in result


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


def test_check_provider_env_bedrock_partial_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA...")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    assert "error: missing" in check_provider_env(_BEDROCK.expected_envs)


def test_run_readiness_checks_all_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    ok, checks = run_readiness_checks(_OPENAI_DIRECT)
    assert ok is True
    assert checks == {"provider_env": "ok"}


def test_run_readiness_checks_provider_env_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ok, checks = run_readiness_checks(_ANTHROPIC_DIRECT)
    assert ok is False
    assert checks["provider_env"].startswith("error: ")


@pytest.mark.parametrize(
    ("api_key", "expected_ready"),
    [(None, False), ("", False), ("   \t\n", False), ("resolved-key", True), ("EMPTY", True)],
)
@pytest.mark.parametrize(
    ("route_id", "base_url"),
    [("openai", None), ("openai-compatible", "https://vllm.example.test/v1")],
)
def test_openai_readiness_validates_resolved_api_key(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str | None,
    expected_ready: bool,
    route_id: str,
    base_url: str | None,
) -> None:
    from lightspeed_agentic.providers.model_router import resolve_model_route

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-must-not-replace-resolved-key")
    family = ResolvedModelFamily(
        "openai",
        ("OPENAI_API_KEY",),
        hosting="openai",
        model="test-model",
        api_key=api_key,
        base_url=base_url,
    )
    _sdk, route = resolve_model_route(model_family=family)

    assert route is not None
    ready, checks = run_readiness_checks(family, route)

    assert route.route == route_id
    assert ready is expected_ready
    expected_status = "ok" if expected_ready else f"error: missing API key for {route_id}"
    assert checks == {"provider_env": "ok", "model_route": expected_status}


@pytest.mark.parametrize("base_url", [None, "", "   "])
def test_compatible_openai_readiness_requires_resolved_base_url(base_url: str | None) -> None:
    from dataclasses import replace

    from lightspeed_agentic.providers.model_router import resolve_model_route

    family = ResolvedModelFamily(
        "openai",
        (),
        hosting="openai",
        model="test-model",
        base_url="https://vllm.example.test/v1",
        api_key="EMPTY",
    )
    _sdk, route = resolve_model_route(model_family=family)
    assert route is not None
    route = replace(route, model_parameters=replace(route.model_parameters, base_url=base_url))

    ready, checks = run_readiness_checks(family, route)

    assert ready is False
    assert checks == {
        "provider_env": "ok",
        "model_route": "error: missing base URL for openai-compatible",
    }


@pytest.mark.parametrize(
    ("missing_source", "expected_route_status"),
    [
        (True, "error: missing credential source for anthropic"),
        (False, "error: missing API key for anthropic"),
    ],
)
def test_direct_anthropic_readiness_requires_resolved_api_key(
    monkeypatch: pytest.MonkeyPatch,
    missing_source: bool,
    expected_route_status: str,
) -> None:
    from lightspeed_agentic.config import AnthropicCredentialSource
    from lightspeed_agentic.providers.model_router import resolve_model_route

    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    source = None if missing_source else AnthropicCredentialSource()
    family = ResolvedModelFamily(
        "anthropic",
        ("ANTHROPIC_API_KEY",),
        hosting="anthropic",
        model="test-model",
        credentials=source,
    )
    _sdk, route = resolve_model_route(model_family=family)

    assert route is not None
    ready, checks = run_readiness_checks(family, route)

    assert ready is False
    assert checks == {"provider_env": "ok", "model_route": expected_route_status}


@pytest.mark.parametrize(
    ("project", "region", "expected_route_status"),
    [
        ("", "us-east5", "error: missing project for vertex/anthropic"),
        ("test-project", "", "error: missing region for vertex/anthropic"),
    ],
)
def test_run_readiness_checks_rejects_vertex_route_without_parameters(
    project: str, region: str, expected_route_status: str
) -> None:
    from lightspeed_agentic.config import VertexProviderParameters
    from lightspeed_agentic.providers.model_router import ModelParameters, ResolvedModelRoute
    from lightspeed_agentic.providers.models.anthropic import create_vertex_model

    family = ResolvedModelFamily("anthropic", (), hosting="vertex")
    route = ResolvedModelRoute(
        route="vertex/anthropic",
        model="test-model",
        create_model=create_vertex_model,
        model_parameters=ModelParameters(
            model="test-model",
            provider=VertexProviderParameters(project=project, region=region),
        ),
        telemetry_provider="gcp.vertex_ai",
        structured_output_method="json_schema",
    )

    ready, checks = run_readiness_checks(family, route)

    assert ready is False
    assert checks == {"provider_env": "ok", "model_route": expected_route_status}


@pytest.mark.parametrize(
    ("project", "region", "expected_route_status"),
    [
        (None, "us-central1", "error: missing project for vertex/google"),
        ("test-project", None, "error: missing region for vertex/google"),
        ("", "", "error: missing project, region for vertex/google"),
        ("test-project", "us-central1", "ok"),
    ],
)
def test_run_readiness_checks_validates_vertex_gemini_route(
    project: str | None,
    region: str | None,
    expected_route_status: str,
) -> None:
    from lightspeed_agentic.config import VertexCredentialSource, VertexProviderParameters
    from lightspeed_agentic.providers.model_router import ModelParameters, ResolvedModelRoute
    from lightspeed_agentic.providers.models.gemini import create_vertex_gemini_model

    family = ResolvedModelFamily("google", (), hosting="vertex")
    route = ResolvedModelRoute(
        route="vertex/google",
        model="gemini-2.5-flash",
        create_model=create_vertex_gemini_model,
        model_parameters=ModelParameters(
            model="gemini-2.5-flash",
            provider=VertexProviderParameters(project=project, region=region or ""),
            credentials=VertexCredentialSource(Path("/unused/vertex-credentials.json")),
        ),
        telemetry_provider="gcp.vertex_ai",
        structured_output_method="json_schema",
    )

    ready, checks = run_readiness_checks(family, route)

    assert ready is (expected_route_status == "ok")
    assert checks == {"provider_env": "ok", "model_route": expected_route_status}


def test_run_readiness_checks_rejects_vertex_route_without_provider_parameters() -> None:
    from lightspeed_agentic.providers.model_router import ModelParameters, ResolvedModelRoute
    from lightspeed_agentic.providers.models.gemini import create_vertex_gemini_model

    family = ResolvedModelFamily("google", (), hosting="vertex")
    route = ResolvedModelRoute(
        route="vertex/google",
        model="gemini-2.5-flash",
        create_model=create_vertex_gemini_model,
        model_parameters=ModelParameters(model="gemini-2.5-flash"),
        telemetry_provider="gcp.vertex_ai",
        structured_output_method="json_schema",
    )

    ready, checks = run_readiness_checks(family, route)

    assert ready is False
    assert checks == {
        "provider_env": "ok",
        "model_route": "error: missing provider parameters for vertex/google",
    }


def test_run_readiness_checks_rejects_vertex_route_without_credential_source() -> None:
    from lightspeed_agentic.config import VertexProviderParameters
    from lightspeed_agentic.providers.model_router import ModelParameters, ResolvedModelRoute
    from lightspeed_agentic.providers.models.gemini import create_vertex_gemini_model

    family = ResolvedModelFamily("google", (), hosting="vertex")
    route = ResolvedModelRoute(
        route="vertex/google",
        model="gemini-2.5-flash",
        create_model=create_vertex_gemini_model,
        model_parameters=ModelParameters(
            model="gemini-2.5-flash",
            provider=VertexProviderParameters(project="test-project", region="us-central1"),
        ),
        telemetry_provider="gcp.vertex_ai",
        structured_output_method="json_schema",
    )

    ready, checks = run_readiness_checks(family, route)

    assert ready is False
    assert checks == {
        "provider_env": "ok",
        "model_route": "error: missing credential source for vertex/google",
    }


def test_run_readiness_checks_rejects_bedrock_route_without_provider_parameters() -> None:
    from lightspeed_agentic.providers.model_router import ModelParameters, ResolvedModelRoute
    from lightspeed_agentic.providers.models.anthropic import create_bedrock_model

    family = ResolvedModelFamily("anthropic", (), hosting="bedrock")
    route = ResolvedModelRoute(
        route="bedrock/anthropic",
        model="test-model",
        create_model=create_bedrock_model,
        model_parameters=ModelParameters(model="test-model"),
        telemetry_provider="aws.bedrock",
        structured_output_method="function_calling",
    )

    ready, checks = run_readiness_checks(family, route)

    assert ready is False
    assert checks == {
        "provider_env": "ok",
        "provider_credentials": "error: missing Bedrock credential source",
        "model_route": "error: missing provider parameters for bedrock/anthropic",
    }


def test_run_readiness_checks_rejects_bedrock_route_without_region() -> None:
    from lightspeed_agentic.config import BedrockProviderParameters
    from lightspeed_agentic.providers.model_router import ModelParameters, ResolvedModelRoute
    from lightspeed_agentic.providers.models.anthropic import create_bedrock_model

    family = ResolvedModelFamily("anthropic", (), hosting="bedrock")
    route = ResolvedModelRoute(
        route="bedrock/anthropic",
        model="test-model",
        create_model=create_bedrock_model,
        model_parameters=ModelParameters(
            model="test-model", provider=BedrockProviderParameters(region="")
        ),
        telemetry_provider="aws.bedrock",
        structured_output_method="function_calling",
    )

    ready, checks = run_readiness_checks(family, route)

    assert ready is False
    assert checks == {
        "provider_env": "ok",
        "provider_credentials": "error: missing Bedrock credential source",
        "model_route": "error: missing region for bedrock/anthropic",
    }


def test_run_readiness_checks_accepts_default_bedrock_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightspeed_agentic.config import resolve_model_family
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute, resolve_model_route

    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "bedrock")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "test-model")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    family = resolve_model_family()
    _, route = resolve_model_route(model_family=family)

    assert isinstance(route, ResolvedModelRoute)
    ready, checks = run_readiness_checks(family, route)

    assert ready is True
    assert checks == {
        "provider_env": "ok",
        "provider_credentials": "ok",
        "model_route": "ok",
    }


def test_bedrock_readiness_accepts_credentials_from_mounted_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lightspeed_agentic.config import resolve_model_family
    from lightspeed_agentic.providers.model_router import resolve_model_route

    credentials_dir = tmp_path / "llm-credentials"
    credentials_dir.mkdir()
    (credentials_dir / "aws_access_key_id").write_text("file-access", encoding="utf-8")
    (credentials_dir / "aws_secret_access_key").write_text("file-secret", encoding="utf-8")
    (credentials_dir / "role_arn").write_text(
        "arn:aws:iam::123456789012:role/test", encoding="utf-8"
    )
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "bedrock")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "test-model")
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(credentials_dir))
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    family = resolve_model_family()
    _, route = resolve_model_route(model_family=family)
    ready, checks = run_readiness_checks(family, route)

    assert ready is True
    assert checks == {
        "provider_env": "ok",
        "provider_credentials": "ok",
        "model_route": "ok",
    }


def test_bedrock_readiness_rejects_partial_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightspeed_agentic.config import resolve_model_family
    from lightspeed_agentic.providers.model_router import resolve_model_route

    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "bedrock")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "test-model")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    family = resolve_model_family()
    _, route = resolve_model_route(model_family=family)
    ready, checks = run_readiness_checks(family, route)

    assert ready is False
    assert checks == {
        "provider_env": "ok",
        "provider_credentials": "error: missing aws_secret_access_key",
        "model_route": "ok",
    }


def test_read_mounted_secret_returns_content(tmp_path: Path) -> None:
    secret = tmp_path / "key"
    secret.write_text("  secret-value  \n", encoding="utf-8")
    assert read_mounted_secret(secret) == "secret-value"


def test_read_mounted_secret_missing_returns_none(tmp_path: Path) -> None:
    assert read_mounted_secret(tmp_path / "missing") is None


def test_read_mounted_secret_empty_returns_none(tmp_path: Path) -> None:
    secret = tmp_path / "empty"
    secret.write_text("", encoding="utf-8")
    assert read_mounted_secret(secret) is None


def test_read_first_mounted_secret_in_dir_sorted(tmp_path: Path) -> None:
    secret_dir = tmp_path / "mount"
    secret_dir.mkdir()
    (secret_dir / "b").write_text("second", encoding="utf-8")
    (secret_dir / "a").write_text("first", encoding="utf-8")
    assert read_first_mounted_secret_in_dir(secret_dir) == "first"


# ── Azure readiness ──


@pytest.mark.parametrize(
    ("change", "expected_status"),
    [
        ({"base_url": None}, "error: Azure OpenAI requires a resolved HTTPS endpoint"),
        ({"base_url": ""}, "error: Azure OpenAI requires a resolved HTTPS endpoint"),
        (
            {"base_url": "http://azure.example.test"},
            "error: Azure OpenAI requires a resolved HTTPS endpoint",
        ),
        ({"provider": None}, "error: Azure OpenAI requires a resolved API version"),
        ({"provider_kind": "vertex"}, "error: Azure OpenAI requires a resolved API version"),
        ({"api_version": ""}, "error: Azure OpenAI requires a resolved API version"),
        ({"api_version": "   "}, "error: Azure OpenAI requires a resolved API version"),
        ({"credentials": None}, "error: Azure OpenAI requires resolved Azure credentials"),
        ({"credentials_kind": "vertex"}, "error: Azure OpenAI requires resolved Azure credentials"),
        ({"api_key": ""}, "error: Azure OpenAI requires a nonempty resolved API key"),
        ({"api_key": "  \t\n"}, "error: Azure OpenAI requires a nonempty resolved API key"),
    ],
)
def test_azure_readiness_rejects_unresolved_configuration_without_ambient_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: dict[str, object], expected_status: str
) -> None:
    from dataclasses import replace

    from lightspeed_agentic.config import (
        AzureApiKeyCredentialSource,
        AzureProviderParameters,
        VertexCredentialSource,
        VertexProviderParameters,
    )

    # Ambient values must not repair the immutable resolved snapshot.
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://ambient.example.test")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "AMBIENT-SECRET")
    monkeypatch.setenv("OPENAI_API_KEY", "UNRELATED-SECRET")
    family = ResolvedModelFamily(
        "openai",
        (),
        hosting="azure",
        model="operator-deployment",
        base_url="https://resolved.example.test",
        provider=AzureProviderParameters("2024-08-01-preview"),
        credentials=AzureApiKeyCredentialSource("RESOLVED-SECRET"),
    )
    if "credentials_kind" in change:
        credential_file = tmp_path / "credentials.json"
        credential_file.write_text("{}", encoding="utf-8")
        family = replace(family, credentials=VertexCredentialSource(credential_file))
    elif "provider_kind" in change:
        family = replace(
            family, provider=VertexProviderParameters(project="project", region="region")
        )
    elif "api_version" in change:
        family = replace(family, provider=AzureProviderParameters(change["api_version"]))
    elif "api_key" in change:
        family = replace(family, credentials=AzureApiKeyCredentialSource(change["api_key"]))
    else:
        family = replace(family, **change)

    ready, checks = run_readiness_checks(family)

    assert ready is False
    assert checks == {"provider_env": "ok", "provider_configuration": expected_status}
    assert "SECRET" not in repr(checks)


@pytest.mark.parametrize("auth_mode", ["api-key", "entra"])
def test_azure_readiness_accepts_resolved_configuration_without_constructing_clients(
    monkeypatch: pytest.MonkeyPatch, auth_mode: str
) -> None:
    from lightspeed_agentic.config import (
        AzureApiKeyCredentialSource,
        AzureEntraCredentialSource,
        AzureProviderParameters,
    )

    def no_client_or_token(*_args: object, **_kwargs: object) -> None:
        pytest.fail("readiness must not construct clients or acquire tokens")

    monkeypatch.setattr(
        "lightspeed_agentic.providers.models.openai.create_azure_model", no_client_or_token
    )
    monkeypatch.setattr(
        "lightspeed_agentic.providers.cloud_credentials.create_azure_credentials",
        no_client_or_token,
    )
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    source = (
        AzureApiKeyCredentialSource("resolved-key")
        if auth_mode == "api-key"
        else AzureEntraCredentialSource("tenant", "client", "secret")
    )
    family = ResolvedModelFamily(
        "openai",
        (),
        hosting="azure",
        model="operator-deployment",
        base_url="https://resolved.example.test",
        provider=AzureProviderParameters("2024-08-01-preview"),
        credentials=source,
    )

    ready, checks = run_readiness_checks(family)

    assert ready is True
    assert checks == {"provider_env": "ok", "provider_configuration": "ok"}


_AZURE_ENTRA = ResolvedSDK(
    "openai",
    (),  # no env vars needed — credentials are file-based
    azure_auth_mode="entra_id",
    azure_credentials={
        "client_id": "cid",
        "tenant_id": "tid",
        "client_secret": "csec",
    },
)

_AZURE_API_KEY = ResolvedSDK(
    "openai",
    ("AZURE_OPENAI_API_KEY",),
    azure_auth_mode="api_key",
)


def test_run_readiness_checks_azure_entra_ok() -> None:
    """Entra ID mode with empty expected_envs passes readiness."""
    ok, checks = run_readiness_checks(_AZURE_ENTRA)
    assert ok is True
    assert checks["provider_env"] == "ok"


def test_run_readiness_checks_azure_api_key_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """API key mode passes when AZURE_OPENAI_API_KEY is set."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    ok, checks = run_readiness_checks(_AZURE_API_KEY)
    assert ok is True
    assert checks["provider_env"] == "ok"


def test_run_readiness_checks_azure_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """API key mode fails when AZURE_OPENAI_API_KEY is missing."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    ok, checks = run_readiness_checks(_AZURE_API_KEY)
    assert ok is False
    assert "error: missing" in checks["provider_env"]
