"""Native Azure SDK model requests, offline for both auth modes and APIs."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace
from functools import partial
from typing import Any

import httpx
import pytest

from lightspeed_agentic import tls
from lightspeed_agentic.config import (
    AzureApiKeyCredentialSource,
    AzureEntraCredentialSource,
    AzureProviderParameters,
    ModelParameters,
)
from lightspeed_agentic.inspection.client import CLASSIFIER_OUTPUT_TOKENS
from lightspeed_agentic.providers.deepagents import _close_model_clients
from tests.test_azure_credentials import azure_identity_http as azure_identity_http
from tests.test_openai_models import (
    _openai_response,
    _openai_response_stream,
)


@pytest.fixture
def azure_http(monkeypatch: pytest.MonkeyPatch, azure_identity_http: Any) -> Iterator[Any]:
    state = azure_identity_http
    state.model_requests = []
    state.model_status = 200
    state.model_handler = None
    state.text = '{"status":"ok"}'
    state.classifier_function = True
    state.sync_clients = []
    state.async_clients = []
    state.responses.extend(
        [
            (200, {"access_token": "entra-token-one", "expires_in": 3600, "token_type": "Bearer"}),
            (200, {"access_token": "entra-token-two", "expires_in": 3600, "token_type": "Bearer"}),
            (
                200,
                {"access_token": "entra-token-three", "expires_in": 3600, "token_type": "Bearer"},
            ),
        ]
    )
    for key in (
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "OPENAI_API_VERSION",
        "AZURE_OPENAI_API_VERSION",
    ):
        monkeypatch.setenv(key, "https://ambient-must-not-win.invalid")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "ambient-token-must-not-win")

    def capture(request: httpx.Request) -> httpx.Response:
        if state.model_handler is not None:
            return state.model_handler(request)
        state.model_requests.append(request)
        if state.model_status != 200:
            return httpx.Response(
                state.model_status,
                json={
                    "error": {
                        "message": "offline rejection",
                        "type": "authentication_error",
                        "code": "invalid_api_key",
                    }
                },
            )
        body = json.loads(request.content)
        classifier_call = state.classifier_function and any(
            tool.get("function", tool).get("name") == "ClassifierDecision"
            for tool in body.get("tools", [])
        )
        if request.url.path.endswith("/responses"):
            item = {
                "id": "msg_azure",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": state.text, "annotations": []}],
            }
            if classifier_call:
                item = {
                    "id": "fc_azure",
                    "type": "function_call",
                    "call_id": "call_azure-classifier",
                    "name": "ClassifierDecision",
                    "arguments": state.text,
                    "status": "completed",
                }
            if body.get("stream"):
                return _openai_response_stream(
                    "resp_azure-stream", item, input_tokens=3, output_tokens=2
                )
            response = _openai_response("resp_azure-capture", item, input_tokens=3, output_tokens=2)
            response["model"] = "observed-azure-model"
            return httpx.Response(200, json=response)
        if body.get("stream"):
            chunks = [
                {
                    "id": "chatcmpl_azure-stream",
                    "object": "chat.completion.chunk",
                    "created": 1_700_000_000,
                    "model": "observed-azure-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": state.text},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl_azure-stream",
                    "object": "chat.completion.chunk",
                    "created": 1_700_000_000,
                    "model": "observed-azure-model",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
                {
                    "id": "chatcmpl_azure-stream",
                    "object": "chat.completion.chunk",
                    "created": 1_700_000_000,
                    "model": "observed-azure-model",
                    "choices": [],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            ]
            sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=sse.encode()
            )
        message = {"role": "assistant", "content": state.text}
        if classifier_call:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_azure-classifier",
                        "type": "function",
                        "function": {"name": "ClassifierDecision", "arguments": state.text},
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_azure-capture",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": "observed-azure-model",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if classifier_call else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    def sync_client(**kwargs: Any) -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(capture), **kwargs)
        state.sync_clients.append(client)
        return client

    def async_client(**kwargs: Any) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(capture), **kwargs)
        state.async_clients.append(client)
        return client

    monkeypatch.setattr(tls, "create_http_client", sync_client)
    monkeypatch.setattr(tls, "create_async_http_client", async_client)
    yield state
    for client in state.sync_clients:
        client.close()

    async def close_remaining() -> None:
        for client in state.async_clients:
            await client.aclose()
        for session in state.sessions:
            await session.close()

    asyncio.run(close_remaining())


def _parameters(
    api_version: str, auth_mode: str = "api-key", model: str = "operator-deployment"
) -> ModelParameters:
    source = (
        AzureApiKeyCredentialSource("resolved-azure-key")
        if auth_mode == "api-key"
        else AzureEntraCredentialSource("resolved-tenant", "resolved-client", "resolved-secret")
    )
    return ModelParameters(
        model=model,
        base_url="https://resolved-resource.example.test/proxy",
        provider=AzureProviderParameters(api_version),
        credentials=source,
    )


@pytest.mark.parametrize("auth_mode", ["api-key", "entra"])
@pytest.mark.parametrize(
    ("version", "responses"),
    [
        ("2024-07-01-preview", False),
        ("2024-08-01-preview", False),
        ("2025-02-28-preview", False),
        ("2025-03-01-preview", True),
        ("2025-03-01", True),
    ],
)
@pytest.mark.asyncio
async def test_azure_native_client_preserves_route_and_selected_auth(
    azure_http: Any, version: str, responses: bool, auth_mode: str
) -> None:
    from langchain_openai import AzureChatOpenAI
    from openai import AsyncAzureOpenAI

    from lightspeed_agentic.providers.models.openai import create_azure_model

    environment = dict(os.environ)
    model = create_azure_model(_parameters(version, auth_mode))
    assert isinstance(model, AzureChatOpenAI)
    assert isinstance(model.root_async_client, AsyncAzureOpenAI)
    assert model.use_responses_api is responses
    assert model.model_name == "operator-deployment"
    assert model.deployment_name == "operator-deployment"
    assert azure_http.requests == []
    try:
        answer = await model.ainvoke("use the resolved route")
        assert answer.text == '{"status":"ok"}'
        assert answer.response_metadata["model_name"] == "observed-azure-model"
        assert answer.response_metadata["id"] == (
            "resp_azure-capture" if responses else "chatcmpl_azure-capture"
        )
        assert answer.usage_metadata["input_tokens"] == 3
        assert answer.usage_metadata["output_tokens"] == 2
    finally:
        await _close_model_clients(model)
    assert len(azure_http.model_requests) == 1
    request = azure_http.model_requests[0]
    expected_path = (
        "/proxy/openai/responses"
        if responses
        else "/proxy/openai/deployments/operator-deployment/chat/completions"
    )
    assert request.url.host == "resolved-resource.example.test"
    assert request.url.path == expected_path
    assert request.url.params["api-version"] == version
    assert json.loads(request.content)["model"] == "operator-deployment"
    if auth_mode == "api-key":
        assert request.headers["api-key"] == "resolved-azure-key"
        assert "authorization" not in request.headers
        assert azure_http.requests == []
    else:
        assert request.headers["authorization"] == "Bearer entra-token-one"
        assert "api-key" not in request.headers
        assert len(azure_http.requests) == 1
        assert all(session.closed for session in azure_http.sessions)
    assert all(client.is_closed for client in [*azure_http.sync_clients, *azure_http.async_clients])
    assert dict(os.environ) == environment
    assert "_lightspeed_owned_resources" not in model.model_dump()


@pytest.mark.parametrize("version", ["2024-08-01-preview", "2025-03-01-preview"])
@pytest.mark.asyncio
async def test_azure_native_client_uses_refreshed_entra_token(
    azure_http: Any, version: str
) -> None:
    from lightspeed_agentic.providers.models.openai import create_azure_model

    model = create_azure_model(_parameters(version, "entra"))
    try:
        await model.ainvoke("first")
        azure_http.clock += 3601
        await model.ainvoke("second")
        assert [request.headers["authorization"] for request in azure_http.model_requests] == [
            "Bearer entra-token-one",
            "Bearer entra-token-two",
        ]
        assert len(azure_http.requests) == 2
    finally:
        await _close_model_clients(model)
    assert all(session.closed for session in azure_http.sessions)


@pytest.mark.parametrize("auth_mode", ["api-key", "entra"])
@pytest.mark.parametrize("version", ["2024-08-01-preview", "2025-03-01-preview"])
@pytest.mark.asyncio
async def test_azure_reasoning_and_stream_usage_preserve_observed_metadata(
    azure_http: Any, auth_mode: str, version: str
) -> None:
    from lightspeed_agentic.providers.models.openai import create_azure_model

    config = {"effort": "low", "verbosity": "low"}
    model = create_azure_model(_parameters(version, auth_mode), config)
    try:
        chunks = [chunk async for chunk in model.astream("stream this")]
        combined = sum(chunks[1:], chunks[0])
        assert combined.text == '{"status":"ok"}'
        assert combined.usage_metadata["input_tokens"] == 3
        assert combined.usage_metadata["output_tokens"] == 2
        assert combined.response_metadata["id"] == (
            "resp_azure-stream" if version == "2025-03-01-preview" else "chatcmpl_azure-stream"
        )
        body = json.loads(azure_http.model_requests[0].content)
        if version == "2025-03-01-preview":
            assert body["reasoning"] == {"effort": "low"}
            assert body["text"]["verbosity"] == "low"
        else:
            assert body["reasoning_effort"] == "low"
            assert body["verbosity"] == "low"
            assert body["stream_options"] == {"include_usage": True}
    finally:
        await _close_model_clients(model)
    assert config == {"effort": "low", "verbosity": "low"}


@pytest.mark.parametrize("auth_mode", ["api-key", "entra"])
@pytest.mark.parametrize(
    ("version", "strict"), [("2024-08-01-preview", False), ("2025-03-01-preview", True)]
)
@pytest.mark.asyncio
async def test_azure_shaping_preserves_version_schema_contract(
    azure_http: Any, auth_mode: str, version: str, strict: bool
) -> None:
    from opentelemetry.context import Context

    from lightspeed_agentic.providers.deepagents import (
        _create_generation_callback,
        _shape_structured_output,
    )
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute
    from lightspeed_agentic.providers.models.openai import (
        azure_structured_output_schema,
        bind_azure_output,
        create_azure_model,
    )

    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "description": "Overall status"},
            "variant": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        },
        "required": ["status"],
    }
    original = deepcopy(schema)
    azure_http.text = '{"status":"ok","variant":"example"}'
    route = ResolvedModelRoute(
        route="azure-test-only",
        model="gpt-4",
        create_model=create_azure_model,
        model_parameters=_parameters(version, auth_mode, model="gpt-4"),
        telemetry_provider="openai",
        structured_output_method="json_schema",
        structured_output_schema_mode="raw_json_schema",
        translate_output_schema=partial(azure_structured_output_schema, api_version=version),
        bind_structured_output=bind_azure_output,
    )
    result, input_tokens, output_tokens = await _shape_structured_output(
        route,
        schema,
        "Return status",
        "query",
        "agent summary",
        _create_generation_callback(route, Context(), main_name=None),
    )
    assert result == {"status": "ok", "variant": "example"}
    assert (input_tokens, output_tokens) == (3, 2)
    assert schema == original
    body = json.loads(azure_http.model_requests[0].content)
    assert "tools" not in body
    assert "reasoning" not in body
    assert "reasoning_effort" not in body
    output_format = body["text"]["format"] if strict else body["response_format"]["json_schema"]
    assert output_format["strict"] is strict
    if strict:
        assert output_format["schema"]["required"] == ["status", "variant"]
        assert output_format["schema"]["properties"]["variant"]["anyOf"] == [
            {"type": "string"},
            {"type": "integer"},
        ]
    else:
        assert output_format["schema"] == original
    assert all(client.is_closed for client in [*azure_http.sync_clients, *azure_http.async_clients])
    assert all(session.closed for session in azure_http.sessions)


@pytest.mark.parametrize("auth_mode", ["api-key", "entra"])
@pytest.mark.parametrize("version", ["2024-08-01-preview", "2025-03-01-preview"])
@pytest.mark.parametrize("stage", ["translation", "route-binding", "sdk-binding", "messages"])
@pytest.mark.parametrize("cancelled", [False, True], ids=["error", "setup-cancellation"])
@pytest.mark.asyncio
async def test_azure_formatter_setup_failure_closes_native_resources(
    azure_http: Any,
    monkeypatch: pytest.MonkeyPatch,
    auth_mode: str,
    version: str,
    stage: str,
    cancelled: bool,
) -> None:
    from langchain_core.messages import HumanMessage
    from opentelemetry.context import Context

    from lightspeed_agentic.providers.deepagents import (
        _create_generation_callback,
        _shape_structured_output,
    )
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute
    from lightspeed_agentic.providers.models.openai import (
        azure_structured_output_schema,
        bind_azure_output,
        create_azure_model,
    )

    parameters = _parameters(version, auth_mode)
    model = create_azure_model(parameters)
    failure = (
        asyncio.CancelledError() if cancelled else RuntimeError("offline formatter setup failure")
    )
    identity_closes = 0
    auth = model.__dict__["_lightspeed_owned_resources"][0]
    if auth.credential is not None:
        native_close = auth.credential.close

        async def close_identity() -> None:
            nonlocal identity_closes
            await native_close()
            identity_closes += 1

        monkeypatch.setattr(auth.credential, "close", close_identity)

    def create_model(settings: ModelParameters, reasoning: Any) -> Any:
        assert settings is parameters
        assert reasoning is None
        return model

    def reject_setup(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    class RejectHumanMessage(HumanMessage):
        def __init__(self, **_kwargs: Any) -> None:
            raise failure

    if stage == "sdk-binding":
        monkeypatch.setattr(type(model), "with_structured_output", reject_setup)
    elif stage == "messages":
        monkeypatch.setattr("langchain_core.messages.HumanMessage", RejectHumanMessage)

    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
    }
    route = ResolvedModelRoute(
        route="azure",
        model=parameters.model,
        create_model=create_model,
        model_parameters=parameters,
        telemetry_provider="openai",
        structured_output_method="json_schema",
        structured_output_schema_mode="raw_json_schema",
        translate_output_schema=(
            reject_setup
            if stage == "translation"
            else partial(azure_structured_output_schema, api_version=version)
        ),
        bind_structured_output=(
            reject_setup
            if stage == "route-binding"
            else None
            if stage == "sdk-binding"
            else bind_azure_output
        ),
    )
    # Binding/message failures use the same prepared-schema path as routed queries.
    # Translation failures exercise direct helper calls without that preflight.
    prepared = (
        None
        if stage == "translation"
        else azure_structured_output_schema(schema, api_version=version)
    )
    with pytest.raises(type(failure)) as error:
        await _shape_structured_output(
            route,
            schema,
            "system",
            "prompt",
            "summary",
            _create_generation_callback(route, Context(), main_name=None),
            prepared_schema=prepared,
        )
    assert error.value is failure
    assert azure_http.async_clients
    assert all(client.is_closed for client in [*azure_http.sync_clients, *azure_http.async_clients])
    assert identity_closes == (1 if auth_mode == "entra" else 0)
    assert not azure_http.model_requests
    assert not azure_http.requests
    assert not azure_http.sessions


@pytest.mark.parametrize("version", ["2024-07-01-preview", "invalid-version"])
def test_azure_shaping_rejects_unsupported_api_version(version: str) -> None:
    from lightspeed_agentic.providers.models.openai import azure_structured_output_schema

    with pytest.raises(ValueError, match="structured output requires API version 2024-08-01"):
        azure_structured_output_schema({"type": "object"}, api_version=version)


@pytest.mark.parametrize("auth_mode", ["api-key", "entra"])
@pytest.mark.parametrize("version", ["2024-08-01-preview", "2025-03-01-preview"])
@pytest.mark.asyncio
async def test_azure_classifier_budget_does_not_leak_into_main_model(
    azure_http: Any, auth_mode: str, version: str
) -> None:
    from lightspeed_agentic.inspection.client import LangChainClassifierClient
    from lightspeed_agentic.inspection.models import ClassifierRequest
    from lightspeed_agentic.providers.models.openai import create_azure_model

    azure_http.text = '{"injectionDetected":false,"category":"none"}'
    parameters = _parameters(version, auth_mode)
    main = create_azure_model(parameters)
    classifier = create_azure_model(replace(parameters, max_output_tokens=CLASSIFIER_OUTPUT_TOKENS))
    try:
        await main.ainvoke("main")
        decision = await LangChainClassifierClient(classifier).classify(
            ClassifierRequest(
                toolName="probe", resultType="result", chunkIndex=0, chunkCount=1, content="safe"
            )
        )
        assert decision.injection_detected is False
        main_body, classifier_body = [
            json.loads(request.content) for request in azure_http.model_requests
        ]
        assert not {"max_tokens", "max_completion_tokens", "max_output_tokens"} & main_body.keys()
        token_field = (
            "max_output_tokens" if version == "2025-03-01-preview" else "max_completion_tokens"
        )
        assert classifier_body[token_field] == 4096
        assert "tools" not in main_body
        tool = classifier_body["tools"][0]
        function = tool.get("function", tool)
        assert function["name"] == "ClassifierDecision"
        assert "reasoning" not in classifier_body
        assert "reasoning_effort" not in classifier_body
        assert parameters.max_output_tokens is None
    finally:
        await _close_model_clients(main)
        await _close_model_clients(classifier)
    assert all(client.is_closed for client in [*azure_http.sync_clients, *azure_http.async_clients])
    assert all(session.closed for session in azure_http.sessions)


@pytest.mark.parametrize("auth_mode", ["api-key", "entra"])
@pytest.mark.parametrize("version", ["2024-08-01-preview", "2025-03-01-preview"])
@pytest.mark.parametrize("function_call", [False, True], ids=["json-text", "function-call"])
@pytest.mark.asyncio
async def test_azure_classifier_rejects_complete_decision_in_truncated_response(
    azure_http: Any,
    monkeypatch: pytest.MonkeyPatch,
    auth_mode: str,
    version: str,
    function_call: bool,
) -> None:
    from lightspeed_agentic.inspection.client import LangChainClassifierClient
    from lightspeed_agentic.inspection.errors import ClassifierResponseError
    from lightspeed_agentic.inspection.models import ClassifierRequest
    from lightspeed_agentic.providers.models.openai import create_azure_model

    azure_http.text = '{"injectionDetected":false,"category":"none"}'
    azure_http.classifier_function = function_call
    model = create_azure_model(
        replace(_parameters(version, auth_mode), max_output_tokens=CLASSIFIER_OUTPUT_TOKENS)
    )
    transport = model.root_async_client._client._transport
    native_request = transport.handle_async_request

    async def truncated(request: httpx.Request) -> httpx.Response:
        response = await native_request(request)
        payload = response.json()
        if version == "2025-03-01-preview":
            payload["status"] = "incomplete"
            payload["incomplete_details"] = {"reason": "max_output_tokens"}
        else:
            payload["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=payload, request=request)

    monkeypatch.setattr(transport, "handle_async_request", truncated)
    try:
        with pytest.raises(ClassifierResponseError) as error:
            await LangChainClassifierClient(model).classify(
                ClassifierRequest(
                    toolName="probe",
                    resultType="result",
                    chunkIndex=0,
                    chunkCount=1,
                    content="safe",
                )
            )
        assert error.value.issue == "output_limit"
        body = json.loads(azure_http.model_requests[0].content)
        assert (
            body[
                "max_output_tokens" if version == "2025-03-01-preview" else "max_completion_tokens"
            ]
            == 4096
        )
    finally:
        await _close_model_clients(model)
    assert len(azure_http.model_requests) == 1
    assert all(client.is_closed for client in [*azure_http.sync_clients, *azure_http.async_clients])
    assert all(session.closed for session in azure_http.sessions)


@pytest.mark.parametrize("auth_mode", ["api-key", "entra"])
@pytest.mark.parametrize("version", ["2024-08-01-preview", "2025-03-01-preview"])
@pytest.mark.parametrize("ambiguity", ["conflicting-decisions", "duplicate-keys"])
@pytest.mark.parametrize("recover", [False, True], ids=["exhausted", "recovered"])
@pytest.mark.asyncio
async def test_azure_classifier_retries_ambiguous_native_decisions(
    azure_http: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    auth_mode: str,
    version: str,
    ambiguity: str,
    recover: bool,
) -> None:
    from lightspeed_agentic.inspection.chunking import Utf8ByteCodec
    from lightspeed_agentic.inspection.client import LangChainClassifierClient
    from lightspeed_agentic.inspection.errors import InspectionError
    from lightspeed_agentic.inspection.inspector import inspect_tool_result
    from lightspeed_agentic.providers.models.openai import create_azure_model

    azure_http.text = '{"injectionDetected":false,"category":"none"}'
    model = create_azure_model(
        replace(_parameters(version, auth_mode), max_output_tokens=CLASSIFIER_OUTPUT_TOKENS)
    )
    transport = model.root_async_client._client._transport
    native_request = transport.handle_async_request
    attempts = 0

    async def ambiguous(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        response = await native_request(request)
        if recover and attempts > 1:
            return response
        payload = response.json()
        responses = version == "2025-03-01-preview"
        calls = payload["output"] if responses else payload["choices"][0]["message"]["tool_calls"]
        if ambiguity == "conflicting-decisions":
            extra = deepcopy(calls[0])
            extra["id"] = "second-decision"
            if responses:
                extra["call_id"] = "second-decision"
            function = extra if responses else extra["function"]
            function["arguments"] = '{"injectionDetected":true,"category":"instruction_override"}'
            calls.append(extra)
        else:
            function = calls[0] if responses else calls[0]["function"]
            function["arguments"] = (
                '{"injectionDetected":true,"category":"instruction_override",'
                '"injectionDetected":false,"category":"none"}'
            )
        return httpx.Response(200, json=payload, request=request)

    async def no_delay(_seconds: float) -> None:
        pass

    monkeypatch.setattr(transport, "handle_async_request", ambiguous)
    try:
        inspection = inspect_tool_result(
            LangChainClassifierClient(model),
            tool_name="probe",
            result_type="result",
            value="TOOL-SECRET",
            codec=Utf8ByteCodec(),
            context_window_tokens=100_000,
            instruction_tokens=512,
            output_tokens=CLASSIFIER_OUTPUT_TOKENS,
            sleep=no_delay,
        )
        if recover:
            assert (await inspection).passed is True
            assert attempts == 2
        else:
            with pytest.raises(InspectionError) as error:
                await inspection
            assert error.value.attempt_count == 3
            assert error.value.response_issue == (
                "invalid_envelope" if ambiguity == "conflicting-decisions" else "non_json"
            )
            assert attempts == 3
        assert "response rejected" in caplog.text
        assert all(
            secret not in caplog.text
            for secret in (
                "TOOL-SECRET",
                "injectionDetected",
                "resolved-azure-key",
                "resolved-secret",
            )
        )
    finally:
        await _close_model_clients(model)
    assert len(azure_http.model_requests) == attempts
    assert all(client.is_closed for client in [*azure_http.sync_clients, *azure_http.async_clients])
    assert all(session.closed for session in azure_http.sessions)


@pytest.mark.parametrize("auth_mode", ["api-key", "entra"])
@pytest.mark.asyncio
async def test_azure_resource_rejection_retains_failure_and_cleanup(
    azure_http: Any, auth_mode: str
) -> None:
    from openai import AuthenticationError

    from lightspeed_agentic.providers.models.openai import create_azure_model

    azure_http.model_status = 401
    model = create_azure_model(_parameters("2024-08-01-preview", auth_mode))
    try:
        with pytest.raises(AuthenticationError):
            await model.ainvoke("do not recover authentication failure")
    finally:
        await _close_model_clients(model)
    assert len(azure_http.model_requests) == 1
    assert all(client.is_closed for client in [*azure_http.sync_clients, *azure_http.async_clients])
    assert all(session.closed for session in azure_http.sessions)


@pytest.mark.asyncio
async def test_azure_identity_rejection_never_sends_model_request(azure_http: Any) -> None:
    from azure.core.exceptions import ClientAuthenticationError

    from lightspeed_agentic.providers.models.openai import create_azure_model

    azure_http.responses[:] = [
        (400, {"error": "invalid_client", "error_description": "offline rejection"})
    ]
    model = create_azure_model(_parameters("2025-03-01-preview", "entra"))
    try:
        with pytest.raises(ClientAuthenticationError):
            await model.ainvoke("do not fall back")
    finally:
        await _close_model_clients(model)
    assert azure_http.model_requests == []
    assert len(azure_http.requests) == 1
    assert all(client.is_closed for client in [*azure_http.sync_clients, *azure_http.async_clients])
    assert all(session.closed for session in azure_http.sessions)


@pytest.mark.parametrize("version", ["2024-08-01-preview", "2025-03-01-preview"])
@pytest.mark.parametrize(
    "text",
    [
        '{"injectionDetected":false,"injectionDetected":true,"category":"none"}',
        '{"injectionDetected":"false","category":"none"}',
        '{"injectionDetected":false,"category":"none","extra":true}',
    ],
)
@pytest.mark.asyncio
async def test_azure_classifier_json_text_remains_strict(
    azure_http: Any, version: str, text: str
) -> None:
    from lightspeed_agentic.inspection.client import LangChainClassifierClient
    from lightspeed_agentic.inspection.errors import ClassifierResponseError
    from lightspeed_agentic.inspection.models import ClassifierRequest
    from lightspeed_agentic.providers.models.openai import create_azure_model

    azure_http.classifier_function = False
    azure_http.text = text
    model = create_azure_model(
        replace(_parameters(version), max_output_tokens=CLASSIFIER_OUTPUT_TOKENS)
    )
    try:
        with pytest.raises(ClassifierResponseError):
            await LangChainClassifierClient(model).classify(
                ClassifierRequest(
                    toolName="probe",
                    resultType="result",
                    chunkIndex=0,
                    chunkCount=1,
                    content="safe",
                )
            )
    finally:
        await _close_model_clients(model)
    assert len(azure_http.model_requests) == 1
    assert all(client.is_closed for client in [*azure_http.sync_clients, *azure_http.async_clients])


@pytest.mark.asyncio
async def test_azure_identity_owner_closes_even_if_sdk_close_raises(
    azure_http: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lightspeed_agentic.providers.models.openai import create_azure_model

    model = create_azure_model(_parameters("2024-08-01-preview", "entra"))
    await model.ainvoke("initialize the native identity transport")
    native_close = model.root_async_client.close

    async def fail_close() -> None:
        await native_close()
        raise RuntimeError("offline SDK close failure")

    monkeypatch.setattr(model.root_async_client, "close", fail_close)
    with pytest.raises(RuntimeError, match="offline SDK close failure"):
        await _close_model_clients(model)
    assert len(azure_http.sessions) == 1
    assert azure_http.sessions[0].closed


@pytest.mark.asyncio
async def test_azure_model_clients_retain_shared_tls_context(
    azure_identity_http: Any,
) -> None:
    from lightspeed_agentic.providers.models.openai import create_azure_model

    model = create_azure_model(_parameters("2024-08-01-preview"))
    try:
        assert (
            model.root_client._client._transport._pool._ssl_context
            is azure_identity_http.ssl_context
        )
        assert (
            model.root_async_client._client._transport._pool._ssl_context
            is azure_identity_http.ssl_context
        )
    finally:
        await _close_model_clients(model)
