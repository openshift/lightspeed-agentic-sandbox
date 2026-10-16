"""Offline tests for explicit Anthropic model construction."""

from __future__ import annotations

import asyncio
import ssl
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from lightspeed_agentic import tls


@pytest.fixture(autouse=True)
def _credential_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    # Explicit constructors must not select a backend from these flags.
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")


@pytest.fixture
def http_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[httpx.Client, httpx.AsyncClient]]:
    context = ssl.create_default_context()
    monkeypatch.setattr(tls, "get_ssl_context", lambda: context)
    sync_client = tls.create_http_client()
    async_client = tls.create_async_http_client()
    monkeypatch.setattr(tls, "create_http_client", lambda: sync_client)
    monkeypatch.setattr(tls, "create_async_http_client", lambda: async_client)
    try:
        yield sync_client, async_client
    finally:
        sync_client.close()
        asyncio.run(async_client.aclose())


@pytest.mark.parametrize(
    ("reasoning_config", "expected_thinking"),
    [
        (None, None),
        ({}, None),
        ({"thinking": {"type": "adaptive"}}, {"type": "adaptive"}),
        (
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            {"type": "enabled", "budget_tokens": 1024},
        ),
    ],
)
def test_direct_model_preserves_endpoint_bearer_auth_thinking_and_tls(
    http_clients: tuple[httpx.Client, httpx.AsyncClient],
    reasoning_config: dict[str, Any] | None,
    expected_thinking: dict[str, Any] | None,
) -> None:
    from langchain_anthropic import ChatAnthropic

    from lightspeed_agentic.providers.model_router import ModelParameters
    from lightspeed_agentic.providers.models.anthropic import create_direct_model

    model = create_direct_model(
        ModelParameters(
            model="claude-sonnet-4-6",
            base_url="https://anthropic.example.test",
            auth_token="test-bearer-token",  # noqa: S106 — dummy credential
        ),
        reasoning_config,
    )

    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-4-6"
    assert model.thinking == expected_thinking
    assert model._client.base_url == "https://anthropic.example.test"
    assert model._async_client.base_url == "https://anthropic.example.test"
    assert model._client.default_headers["Authorization"] == "Bearer test-bearer-token"
    assert model._async_client.default_headers["Authorization"] == "Bearer test-bearer-token"
    assert model._client._client is http_clients[0]
    assert model._async_client._client is http_clients[1]


def test_direct_model_delegates_api_key_credentials_to_sdk(
    http_clients: tuple[httpx.Client, httpx.AsyncClient],
) -> None:
    from lightspeed_agentic.providers.model_router import ModelParameters
    from lightspeed_agentic.providers.models.anthropic import create_direct_model

    model = create_direct_model(ModelParameters(model="claude-sonnet-4-6"))

    assert model._client.api_key == "test-api-key"
    assert model._async_client.api_key == "test-api-key"
    assert "Authorization" not in model._client.default_headers
    assert model._client._client is http_clients[0]


@pytest.mark.parametrize(
    "reasoning_config", [None, {"thinking": {"type": "enabled", "budget_tokens": 1024}}]
)
def test_bedrock_model_preserves_inference_profile_region_endpoint_and_tls(
    http_clients: tuple[httpx.Client, httpx.AsyncClient],
    reasoning_config: dict[str, Any] | None,
) -> None:
    from langchain_aws import ChatAnthropicBedrock

    from lightspeed_agentic.providers.model_router import ModelParameters
    from lightspeed_agentic.providers.models.anthropic import create_bedrock_model

    model = create_bedrock_model(
        ModelParameters(
            model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
            region="us-west-2",
            base_url="https://bedrock.example.test",
        ),
        reasoning_config,
    )

    assert isinstance(model, ChatAnthropicBedrock)
    assert model.model == "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert model.region_name == "us-west-2"
    assert model.thinking == (reasoning_config["thinking"] if reasoning_config else None)
    assert model._client.base_url == "https://bedrock.example.test"
    assert model._async_client.base_url == "https://bedrock.example.test"
    assert model._client.aws_region == "us-west-2"
    assert model._async_client.aws_region == "us-west-2"
    assert model._client._client is http_clients[0]
    assert model._async_client._client is http_clients[1]


@pytest.mark.parametrize("reasoning_config", [None, {"thinking": {"type": "adaptive"}}])
def test_vertex_model_preserves_project_region_endpoint_thinking_and_tls(
    http_clients: tuple[httpx.Client, httpx.AsyncClient],
    reasoning_config: dict[str, Any] | None,
) -> None:
    from langchain_google_vertexai.model_garden import ChatAnthropicVertex

    from lightspeed_agentic.providers.model_router import ModelParameters
    from lightspeed_agentic.providers.models.anthropic import create_vertex_model

    model = create_vertex_model(
        ModelParameters(
            model="claude-sonnet-4-6@20260217",
            project="test-project",
            region="us-east5",
            base_url="https://vertex.example.test",
        ),
        reasoning_config,
    )

    assert isinstance(model, ChatAnthropicVertex)
    assert model.model_name == "claude-sonnet-4-6@20260217"
    assert model.project == "test-project"
    assert model.location == "us-east5"
    assert model.model_kwargs.get("thinking") == (
        reasoning_config["thinking"] if reasoning_config else None
    )
    assert model.client.base_url == "https://vertex.example.test/v1/"
    assert model.async_client.base_url == "https://vertex.example.test/v1/"
    assert model.client._client is http_clients[0]
    assert model.async_client._client is http_clients[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "expected_root"),
    [
        (None, "https://us-east5-aiplatform.googleapis.com/v1"),
        ("https://vertex.example.test", "https://vertex.example.test/v1"),
        ("https://vertex.example.test/", "https://vertex.example.test/v1"),
        ("https://vertex.example.test/v1", "https://vertex.example.test/v1"),
        ("https://vertex.example.test/v1/", "https://vertex.example.test/v1"),
        ("https://vertex.example.test/gateway/v1", "https://vertex.example.test/gateway/v1"),
        ("https://vertex.example.test/gateway", "https://vertex.example.test/gateway"),
    ],
)
async def test_vertex_requests_preserve_api_version_and_explicit_paths(
    monkeypatch: pytest.MonkeyPatch, endpoint: str | None, expected_root: str
) -> None:
    from lightspeed_agentic.providers.model_router import ModelParameters
    from lightspeed_agentic.providers.models.anthropic import create_vertex_model

    monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
    monkeypatch.delenv("ANTHROPIC_VERTEX_BASE_URL", raising=False)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6@20260217",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as sync_client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            monkeypatch.setattr(tls, "create_http_client", lambda: sync_client)
            monkeypatch.setattr(tls, "create_async_http_client", lambda: async_client)
            model = create_vertex_model(
                ModelParameters(
                    model="claude-sonnet-4-6@20260217",
                    project="test-project",
                    region="us-east5",
                    base_url=endpoint,
                )
            )
            model.client.access_token = "test-token"  # noqa: S105 — offline SDK credential
            model.async_client.access_token = "test-token"  # noqa: S105
            await asyncio.to_thread(model.invoke, "hello")
            await model.ainvoke("hello")

    expected_url = (
        f"{expected_root}/projects/test-project/locations/us-east5/"
        "publishers/anthropic/models/claude-sonnet-4-6@20260217:rawPredict"
    )
    assert [str(request.url) for request in requests] == [expected_url, expected_url]


@pytest.mark.parametrize(
    ("hosting", "project", "region", "message"),
    [
        ("bedrock", None, None, "Bedrock requires a region"),
        ("vertex", None, "us-east5", "Vertex requires a project"),
        ("vertex", "test-project", None, "Vertex requires a region"),
    ],
)
def test_cloud_models_reject_missing_required_parameters(
    hosting: str, project: str | None, region: str | None, message: str
) -> None:
    from lightspeed_agentic.providers.model_router import ModelParameters
    from lightspeed_agentic.providers.models.anthropic import (
        create_bedrock_model,
        create_vertex_model,
    )

    constructor = create_bedrock_model if hosting == "bedrock" else create_vertex_model
    parameters = ModelParameters(model="test-model", project=project, region=region)

    with pytest.raises(ValueError, match=message):
        constructor(parameters)


def test_module_import_does_not_load_optional_provider_sdks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import lightspeed_agentic.providers.models.anthropic; "
            "assert not any(name.split('.')[0] in "
            "{'anthropic', 'langchain_anthropic', 'langchain_aws', "
            "'langchain_google_vertexai'} for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
