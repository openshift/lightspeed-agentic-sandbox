"""Offline tests for explicit Anthropic model construction."""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from lightspeed_agentic import tls
from lightspeed_agentic.inspection.client import CLASSIFIER_OUTPUT_TOKENS


@pytest.fixture(autouse=True)
def _credential_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    # Explicit constructors must not select a backend from these flags.
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")


@pytest.fixture(autouse=True)
def _stub_vertex_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.auth.credentials import AnonymousCredentials

    from lightspeed_agentic.providers import cloud_credentials

    monkeypatch.setattr(
        cloud_credentials, "create_vertex_credentials", lambda _source: AnonymousCredentials()
    )


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
    monkeypatch: pytest.MonkeyPatch,
    reasoning_config: dict[str, Any] | None,
    expected_thinking: dict[str, Any] | None,
) -> None:
    from langchain_anthropic import ChatAnthropic

    from lightspeed_agentic.config import AnthropicCredentialSource
    from lightspeed_agentic.providers.model_router import ModelParameters
    from lightspeed_agentic.providers.models.anthropic import create_direct_model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "environment-bearer-token")
    model = create_direct_model(
        ModelParameters(
            model="claude-sonnet-4-6",
            base_url="https://anthropic.example.test",
            credentials=AnthropicCredentialSource(
                api_key="resolved-api-key",
                auth_token="resolved-bearer-token",  # noqa: S106 - Test bearer credential.
            ),
        ),
        reasoning_config,
    )

    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-4-6"
    assert model.anthropic_api_key.get_secret_value() == "resolved-api-key"
    assert model.thinking == expected_thinking
    assert model._client.base_url == "https://anthropic.example.test"
    assert model._async_client.base_url == "https://anthropic.example.test"
    assert model._client.default_headers["Authorization"] == "Bearer resolved-bearer-token"
    assert model._async_client.default_headers["Authorization"] == "Bearer resolved-bearer-token"
    assert model._client._client is http_clients[0]
    assert model._async_client._client is http_clients[1]


def test_direct_model_uses_resolved_credentials_not_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightspeed_agentic.config import AnthropicCredentialSource, ModelParameters
    from lightspeed_agentic.providers.models.anthropic import create_direct_model

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_GATEWAY", "true")
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")

    model = create_direct_model(
        ModelParameters(
            model="claude-sonnet-4-6",
            credentials=AnthropicCredentialSource(api_key="resolved-api-key"),
        )
    )

    assert model.anthropic_api_key.get_secret_value() == "resolved-api-key"


def test_direct_model_delegates_api_key_credentials_to_sdk(
    http_clients: tuple[httpx.Client, httpx.AsyncClient],
) -> None:
    from lightspeed_agentic.config import AnthropicCredentialSource
    from lightspeed_agentic.providers.model_router import ModelParameters
    from lightspeed_agentic.providers.models.anthropic import create_direct_model

    model = create_direct_model(
        ModelParameters(
            model="claude-sonnet-4-6",
            credentials=AnthropicCredentialSource(api_key="test-api-key"),
        )
    )

    assert model._client.api_key == "test-api-key"
    assert model._async_client.api_key == "test-api-key"
    assert "Authorization" not in model._client.default_headers
    assert model._client._client is http_clients[0]


@pytest.mark.parametrize(
    "reasoning_config", [None, {"thinking": {"type": "enabled", "budget_tokens": 1024}}]
)
def test_bedrock_model_preserves_inference_profile_region_endpoint_and_tls(
    http_clients: tuple[httpx.Client, httpx.AsyncClient],
    monkeypatch: pytest.MonkeyPatch,
    reasoning_config: dict[str, Any] | None,
) -> None:
    from langchain_aws import ChatAnthropicBedrock

    from lightspeed_agentic.config import BedrockCredentialSource, BedrockProviderParameters
    from lightspeed_agentic.providers.model_router import ModelParameters
    from lightspeed_agentic.providers.models.anthropic import create_bedrock_model

    monkeypatch.setenv("AWS_SESSION_TOKEN", "stale-session-token")
    model = create_bedrock_model(
        ModelParameters(
            model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
            provider=BedrockProviderParameters(region="us-west-2"),
            credentials=BedrockCredentialSource(
                access_key_id="parameter-access-key",
                secret_access_key="parameter-secret-key",  # noqa: S106 - test credential
            ),
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
    assert model._client.aws_access_key == "parameter-access-key"
    assert model._client.aws_secret_key == "parameter-secret-key"  # noqa: S105
    assert model._client.aws_session_token is None
    assert os.environ["AWS_ACCESS_KEY_ID"] == "parameter-access-key"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "parameter-secret-key"  # noqa: S105
    assert "AWS_SESSION_TOKEN" not in os.environ
    assert model._client._client is http_clients[0]
    assert model._async_client._client is http_clients[1]


def test_bedrock_role_model_signs_with_refreshable_role_credentials_without_env_mutation(
    http_clients: tuple[httpx.Client, httpx.AsyncClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from botocore.credentials import Credentials

    from lightspeed_agentic.config import BedrockCredentialSource, BedrockProviderParameters
    from lightspeed_agentic.providers import cloud_credentials
    from lightspeed_agentic.providers.model_router import ModelParameters
    from lightspeed_agentic.providers.models.anthropic import create_bedrock_model

    role_credentials = Credentials(
        access_key="assumed-access-key",
        secret_key="assumed-secret-key",  # noqa: S106 - test credential
        token="assumed-session-token",  # noqa: S106 - test credential
    )
    monkeypatch.setattr(
        cloud_credentials,
        "create_bedrock_credentials",
        lambda _source, _region: role_credentials,
    )
    original_aws_env = {
        key: os.environ.get(key)
        for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
    }
    model = create_bedrock_model(
        ModelParameters(
            model="claude-sonnet-4-6",
            provider=BedrockProviderParameters(region="us-west-2"),
            credentials=BedrockCredentialSource(
                access_key_id="source-access-key",
                secret_access_key="source-secret-key",  # noqa: S106 - test credential
                role_arn="arn:aws:iam::123456789012:role/test",
            ),
        )
    )

    assert {key: os.environ.get(key) for key in original_aws_env} == original_aws_env
    request = httpx.Request(
        "POST",
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/test/invoke",
        headers={"connection": "keep-alive"},
        json={"prompt": "offline"},
    )
    model._client._prepare_request(request)
    assert "Credential=assumed-access-key/" in request.headers["Authorization"]
    assert request.headers["X-Amz-Security-Token"] == "assumed-session-token"

    async def prepare_async_request() -> httpx.Request:
        async_request = httpx.Request(
            "POST",
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/test/invoke",
            headers={"connection": "keep-alive"},
            json={"prompt": "offline"},
        )
        await model._async_client._prepare_request(async_request)
        return async_request

    async_request = asyncio.run(prepare_async_request())
    assert "Credential=assumed-access-key/" in async_request.headers["Authorization"]
    assert async_request.headers["X-Amz-Security-Token"] == "assumed-session-token"
    assert model._client._client is http_clients[0]
    assert model._async_client._client is http_clients[1]


@pytest.mark.parametrize("reasoning_config", [None, {"thinking": {"type": "adaptive"}}])
def test_vertex_model_preserves_project_region_endpoint_thinking_and_tls(
    http_clients: tuple[httpx.Client, httpx.AsyncClient],
    reasoning_config: dict[str, Any] | None,
) -> None:
    from langchain_google_vertexai.model_garden import ChatAnthropicVertex

    from lightspeed_agentic.config import (
        VertexCredentialSource,
        VertexProviderParameters,
    )
    from lightspeed_agentic.providers.model_router import ModelParameters
    from lightspeed_agentic.providers.models.anthropic import create_vertex_model

    model = create_vertex_model(
        ModelParameters(
            model="claude-sonnet-4-6@20260217",
            provider=VertexProviderParameters(project="test-project", region="us-east5"),
            credentials=VertexCredentialSource(Path("/unused/credentials.json")),
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


def test_vertex_model_uses_impersonated_credentials(
    http_clients: tuple[httpx.Client, httpx.AsyncClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.auth.credentials import AnonymousCredentials

    from lightspeed_agentic.config import (
        VertexCredentialSource,
        VertexProviderParameters,
    )
    from lightspeed_agentic.providers import cloud_credentials
    from lightspeed_agentic.providers.model_router import ModelParameters
    from lightspeed_agentic.providers.models.anthropic import create_vertex_model

    impersonated = AnonymousCredentials()
    monkeypatch.setattr(
        cloud_credentials, "create_vertex_credentials", lambda _source: impersonated
    )

    model = create_vertex_model(
        ModelParameters(
            model="claude-sonnet-4-6@20260217",
            provider=VertexProviderParameters(project="test-project", region="us-east5"),
            credentials=VertexCredentialSource(Path("/unused/credentials.json")),
        )
    )

    assert model.credentials is impersonated
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
    from lightspeed_agentic.config import (
        VertexCredentialSource,
        VertexProviderParameters,
    )
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
                    provider=VertexProviderParameters(project="test-project", region="us-east5"),
                    credentials=VertexCredentialSource(Path("/unused/credentials.json")),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("hosting", ["direct", "bedrock", "vertex"])
@pytest.mark.parametrize("stop_reason", ["tool_use", "max_tokens"])
@pytest.mark.parametrize(
    "max_output_tokens", [None, CLASSIFIER_OUTPUT_TOKENS], ids=["sdk-default", "classifier-limit"]
)
async def test_classifier_output_budget_reaches_anthropic_http_request(
    monkeypatch: pytest.MonkeyPatch,
    hosting: str,
    max_output_tokens: int | None,
    stop_reason: str,
) -> None:
    from lightspeed_agentic.config import (
        AnthropicCredentialSource,
        BedrockCredentialSource,
        BedrockProviderParameters,
        ModelParameters,
        VertexCredentialSource,
        VertexProviderParameters,
    )
    from lightspeed_agentic.inspection.client import LangChainClassifierClient
    from lightspeed_agentic.inspection.errors import ClassifierResponseError
    from lightspeed_agentic.inspection.models import ClassifierRequest
    from lightspeed_agentic.providers.models import anthropic as anthropic_models

    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "msg_classifier",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_decision",
                        "name": "ClassifierDecision",
                        "input": {"injectionDetected": False, "category": "none"},
                    }
                ],
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 13},
            },
        )

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as sync_client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            monkeypatch.setattr(tls, "create_http_client", lambda: sync_client)
            monkeypatch.setattr(tls, "create_async_http_client", lambda: async_client)
            if hosting == "direct":
                model = anthropic_models.create_direct_model(
                    ModelParameters(
                        model="claude-sonnet-4-6",
                        credentials=AnthropicCredentialSource(api_key="test-api-key"),
                        max_output_tokens=max_output_tokens,
                    )
                )
            elif hosting == "bedrock":
                model = anthropic_models.create_bedrock_model(
                    ModelParameters(
                        model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
                        provider=BedrockProviderParameters(region="us-west-2"),
                        credentials=BedrockCredentialSource(
                            access_key_id="test-access-key",
                            secret_access_key="test-secret-key",  # noqa: S106 - test credential
                        ),
                        max_output_tokens=max_output_tokens,
                    )
                )
            else:
                model = anthropic_models.create_vertex_model(
                    ModelParameters(
                        model="claude-sonnet-4-6@20260217",
                        provider=VertexProviderParameters(
                            project="test-project", region="us-east5"
                        ),
                        credentials=VertexCredentialSource(Path("/unused/credentials.json")),
                        max_output_tokens=max_output_tokens,
                    )
                )
                model.async_client.access_token = "test-token"  # noqa: S105 - offline SDK credential
            call = LangChainClassifierClient(model).classify(
                ClassifierRequest(
                    toolName="probe",
                    resultType="result",
                    chunkIndex=0,
                    chunkCount=1,
                    content="safe",
                )
            )
            if stop_reason == "max_tokens":
                with pytest.raises(ClassifierResponseError) as error:
                    await call
                assert error.value.issue == "output_limit"
            else:
                assert (await call).injection_detected is False
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    if max_output_tokens is None:
        assert body["max_tokens"] > 1024  # Keep the SDK's model-profile default.
    else:
        assert body["max_tokens"] == 4096
    assert not {"max_output_tokens", "max_completion_tokens"} & body.keys()


@pytest.mark.parametrize(
    ("decision_kind", "expected_outcome", "response_issue", "attempts"),
    [
        ("benign", "benign", None, 1),
        ("malicious", "malicious", None, 1),
        ("cutoff", "classifier_error", "output_limit", 3),
        ("ambiguous", "classifier_error", "invalid_envelope", 3),
    ],
)
@pytest.mark.asyncio
async def test_anthropic_classifier_uses_nonstreaming_transport_inside_agent_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    span_exporter: Any,
    decision_kind: str,
    expected_outcome: str,
    response_issue: str | None,
    attempts: int,
) -> None:
    from lightspeed_agentic.config import AnthropicCredentialSource, ModelParameters
    from lightspeed_agentic.inspection.errors import ToolResultSafetyInspectionFailed
    from lightspeed_agentic.providers.deepagents import DeepAgentsProvider
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute
    from lightspeed_agentic.providers.models.anthropic import create_direct_model
    from lightspeed_agentic.types import ProviderQueryOptions, ResultEvent, ToolResultEvent

    monkeypatch.setenv("HOME", str(tmp_path))
    main: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []

    def reply(content: list[dict[str, Any]], stop: str, stream: bool) -> httpx.Response:
        message = {
            "id": "msg_offline",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 13},
        }
        if not stream:
            return httpx.Response(200, json=message)
        events = [
            {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}}
        ]
        for index, block in enumerate(content):
            tool = block["type"] == "tool_use"
            events.extend(
                [
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {**block, "input": {}} if tool else {**block, "text": ""},
                    },
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(block["input"]),
                        }
                        if tool
                        else {"type": "text_delta", "text": block["text"]},
                    },
                    {"type": "content_block_stop", "index": index},
                ]
            )
        events.extend(
            [
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop, "stop_sequence": None},
                    "usage": message["usage"],
                },
                {"type": "message_stop"},
            ]
        )
        sse = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        classifier = {tool["name"] for tool in body.get("tools", [])} == {"ClassifierDecision"}
        if classifier:
            classifications.append(body)
            content = [
                {
                    "type": "tool_use",
                    "id": "call_decision",
                    "name": "ClassifierDecision",
                    "input": {
                        "injectionDetected": decision_kind == "malicious",
                        "category": "unknown" if decision_kind == "malicious" else "none",
                    },
                }
            ]
            if decision_kind == "ambiguous":
                content.append(
                    {
                        **content[0],
                        "id": "call_other",
                        "input": {"injectionDetected": True, "category": "unknown"},
                    }
                )
            stop = "max_tokens" if decision_kind == "cutoff" else "tool_use"
        else:
            main.append(body)
            if len(main) == 1:
                content = [
                    {
                        "type": "tool_use",
                        "id": "call_execute",
                        "name": "execute",
                        "input": {"command": "printf 'offline-tool-result'"},
                    }
                ]
                stop = "tool_use"
            else:
                assert len(main) == 2
                content = [{"type": "text", "text": "offline-answer"}]
                stop = "end_turn"
        return reply(content, stop, bool(body.get("stream")))

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as sync_client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            monkeypatch.setattr(tls, "create_http_client", lambda: sync_client)
            monkeypatch.setattr(tls, "create_async_http_client", lambda: async_client)
            route = ResolvedModelRoute(
                route="anthropic",
                model="claude-haiku-4-5",
                create_model=create_direct_model,
                model_parameters=ModelParameters(
                    model="claude-haiku-4-5",
                    credentials=AnthropicCredentialSource(api_key="test-api-key"),
                ),
                telemetry_provider="anthropic",
                structured_output_method="function_calling",
            )
            provider = DeepAgentsProvider(route)
            options = ProviderQueryOptions(
                prompt="Run the shell command and finish.",
                system_prompt="Follow the request.",
                model=route.model,
                cwd=str(tmp_path),
                allowed_tools=["Bash"],
                max_turns=5,
                tool_output_inspection_enabled=True,
            )
            events: list[Any] = []

            async def collect() -> None:
                async for event in provider.query(options):
                    events.append(event)

            if decision_kind == "benign":
                await collect()
                assert isinstance(events[-1], ResultEvent)
                assert events[-1].text == "offline-answer"
                assert any(isinstance(event, ToolResultEvent) for event in events)
                assert len(main) == 2
            else:
                with pytest.raises(ToolResultSafetyInspectionFailed):
                    await collect()
                assert not any(
                    isinstance(event, (ToolResultEvent, ResultEvent)) for event in events
                )
                assert len(main) == 1

    assert len(classifications) == attempts
    assert all(not body.get("stream") for body in classifications)
    assert all(body["max_tokens"] == 4096 and "thinking" not in body for body in classifications)
    assert all(body.get("stream") is True for body in main)
    inspections = [
        span for span in span_exporter.get_finished_spans() if span.name == "tool_result.inspection"
    ]
    assert len(inspections) == 1
    assert inspections[0].attributes["inspection.outcome"] == expected_outcome
    assert inspections[0].attributes["inspection.attempt_count"] == attempts
    if response_issue:
        assert inspections[0].attributes["inspection.response_issue"] == response_issue


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
