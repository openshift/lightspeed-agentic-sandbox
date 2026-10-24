"""Offline tests for Vertex Gemini model construction."""

from __future__ import annotations

import json
import ssl
from pathlib import Path
from typing import Any

import httpx
import pytest
from google.auth.credentials import AnonymousCredentials
from langchain_google_genai import ChatGoogleGenerativeAI

from lightspeed_agentic import tls
from lightspeed_agentic.config import (
    VertexCredentialSource,
    VertexProviderParameters,
)
from lightspeed_agentic.inspection.client import CLASSIFIER_OUTPUT_TOKENS
from lightspeed_agentic.providers import cloud_credentials
from lightspeed_agentic.providers.model_router import ModelParameters
from lightspeed_agentic.providers.models.gemini import create_vertex_gemini_model


def test_vertex_gemini_model_uses_impersonated_credentials_and_tls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = AnonymousCredentials()
    ssl_context = ssl.create_default_context()
    reasoning_config: dict[str, Any] = {"thinking_budget": 1024, "include_thoughts": True}
    monkeypatch.setattr(cloud_credentials, "create_vertex_credentials", lambda _source: credentials)
    monkeypatch.setattr(tls, "get_ssl_context", lambda: ssl_context)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    model = create_vertex_gemini_model(
        ModelParameters(
            model="gemini-2.5-flash",
            base_url="https://vertex-proxy.example.test/custom",
            provider=VertexProviderParameters(project="test-project", region="us-central1"),
            credentials=VertexCredentialSource(Path("/unused/credentials.json")),
        ),
        reasoning_config,
    )

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model == "gemini-2.5-flash"
    assert model.vertexai is True
    assert model.project == "test-project"
    assert model.location == "us-central1"
    assert model.base_url == "https://vertex-proxy.example.test/custom"
    assert model.credentials is credentials
    assert model.thinking_config == reasoning_config
    http_options = model.client._api_client._http_options
    assert http_options.base_url == "https://vertex-proxy.example.test/custom"
    assert http_options.client_args == {"verify": ssl_context}
    assert http_options.async_client_args == {"verify": ssl_context}


def test_vertex_gemini_model_leaves_adc_to_sdk_without_impersonation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cloud_credentials, "create_vertex_credentials", lambda _source: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    model = create_vertex_gemini_model(
        ModelParameters(
            model="gemini-2.5-flash",
            provider=VertexProviderParameters(project="test-project", region="us-central1"),
            credentials=VertexCredentialSource(Path("/unused/credentials.json")),
        )
    )

    assert model.vertexai is True
    assert model.credentials is None
    assert model.thinking_config is None


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["STOP", "MAX_TOKENS"])
@pytest.mark.parametrize(
    "max_output_tokens", [None, CLASSIFIER_OUTPUT_TOKENS], ids=["sdk-default", "classifier-limit"]
)
async def test_classifier_output_budget_reaches_gemini_http_request(
    monkeypatch: pytest.MonkeyPatch, max_output_tokens: int | None, finish_reason: str
) -> None:
    from lightspeed_agentic.inspection.client import LangChainClassifierClient
    from lightspeed_agentic.inspection.errors import ClassifierResponseError
    from lightspeed_agentic.inspection.models import ClassifierRequest

    credentials = AnonymousCredentials()
    credentials.token = "test-token"  # noqa: S105 - offline SDK credential
    monkeypatch.setattr(cloud_credentials, "create_vertex_credentials", lambda _source: credentials)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "ClassifierDecision",
                                        "args": {"injectionDetected": False, "category": "none"},
                                    }
                                }
                            ],
                        },
                        "finishReason": finish_reason,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 13,
                    "totalTokenCount": 23,
                },
                "modelVersion": "gemini-2.5-flash",
                "responseId": "gemini-classifier",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as async_client:
        model = create_vertex_gemini_model(
            ModelParameters(
                model="gemini-2.5-flash",
                provider=VertexProviderParameters(project="test-project", region="us-central1"),
                credentials=VertexCredentialSource(Path("/unused/credentials.json")),
                max_output_tokens=max_output_tokens,
            )
        )
        # Replace only the HTTP transport, retaining SDK auth and serialization.
        api_client = model.client._api_client
        await api_client._async_httpx_client.aclose()
        api_client._async_httpx_client = async_client
        api_client._http_options.httpx_async_client = async_client
        try:
            call = LangChainClassifierClient(model).classify(
                ClassifierRequest(
                    toolName="probe",
                    resultType="result",
                    chunkIndex=0,
                    chunkCount=1,
                    content="safe",
                )
            )
            if finish_reason == "MAX_TOKENS":
                with pytest.raises(ClassifierResponseError) as error:
                    await call
                assert error.value.issue == "output_limit"
            else:
                assert (await call).injection_detected is False
        finally:
            model.client.close()
            await model.client.aio.aclose()

    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body.get("generationConfig", {}).get("maxOutputTokens") == (
        None if max_output_tokens is None else 4096
    )
    assert not {"max_tokens", "max_completion_tokens", "max_output_tokens"} & body.keys()
