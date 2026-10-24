"""Offline OpenAI model, request-contract, and DeepAgents integration tests."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import pytest

from lightspeed_agentic import tls
from lightspeed_agentic.inspection.client import CLASSIFIER_OUTPUT_TOKENS
from lightspeed_agentic.providers.model_router import ModelParameters


@pytest.fixture
def http_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[httpx.Client, httpx.AsyncClient]]:
    sync_client = httpx.Client()
    async_client = httpx.AsyncClient()
    monkeypatch.setattr(tls, "create_http_client", lambda: sync_client)
    monkeypatch.setattr(tls, "create_async_http_client", lambda: async_client)
    try:
        yield sync_client, async_client
    finally:
        sync_client.close()
        asyncio.run(async_client.aclose())


@pytest.fixture
def captured_openai_requests(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Iterator[tuple[list[httpx.Request], httpx.Client, httpx.AsyncClient]]:
    requests: list[httpx.Request] = []
    response_text = json.dumps(getattr(request, "param", {"status": "ok"}))

    def capture(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_capture-test",
                    "object": "chat.completion",
                    "created": 1_700_000_000,
                    "model": "served-custom-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": response_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "resp_capture-test",
                "object": "response",
                "created_at": 1_700_000_000,
                "model": "gpt-4.1-mini",
                "output": [
                    {
                        "id": "msg_capture-test",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": response_text,
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "status": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            },
        )

    transport = httpx.MockTransport(capture)
    sync_client = httpx.Client(transport=transport)
    async_client = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(tls, "create_http_client", lambda: sync_client)
    monkeypatch.setattr(tls, "create_async_http_client", lambda: async_client)
    try:
        yield requests, sync_client, async_client
    finally:
        sync_client.close()
        asyncio.run(async_client.aclose())


@pytest.mark.parametrize(
    ("config", "responses_kwargs", "chat_completions_kwargs"),
    [
        (None, {}, {}),
        ({}, {}, {}),
        ({"verbosity": "low"}, {"verbosity": "low"}, {"verbosity": "low"}),
        (
            {"effort": "high"},
            {"reasoning": {"effort": "high"}},
            {"reasoning_effort": "high"},
        ),
        (
            {
                "effort": "high",
                "verbosity": "low",
                "summary": "auto",
                "custom": {"mode": "future"},
            },
            {
                "verbosity": "low",
                "reasoning": {
                    "effort": "high",
                    "summary": "auto",
                    "custom": {"mode": "future"},
                },
            },
            {
                "verbosity": "low",
                "reasoning_effort": "high",
                "reasoning": {"summary": "auto", "custom": {"mode": "future"}},
            },
        ),
        (
            {"effort": None, "verbosity": None, "summary": None},
            {"verbosity": None, "reasoning": {"effort": None, "summary": None}},
            {"verbosity": None, "reasoning_effort": None, "reasoning": {"summary": None}},
        ),
    ],
)
def test_openai_reasoning_mappers_preserve_settings_without_mutation(
    config: dict[str, Any] | None,
    responses_kwargs: dict[str, Any],
    chat_completions_kwargs: dict[str, Any],
) -> None:
    from copy import deepcopy

    from lightspeed_agentic.providers.models.openai import (
        _chat_completions_reasoning_kwargs,
        _responses_reasoning_kwargs,
    )

    original = deepcopy(config)
    assert _responses_reasoning_kwargs(config) == responses_kwargs
    assert config == original
    assert _chat_completions_reasoning_kwargs(config) == chat_completions_kwargs
    assert config == original


def test_openai_model_uses_typed_responses_settings_credentials_and_tls(
    http_clients: tuple[httpx.Client, httpx.AsyncClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_openai import ChatOpenAI

    from lightspeed_agentic.providers.models.openai import create_native_openai_model

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-must-not-win")
    parameters = ModelParameters(
        model="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        api_key="resolved-openai-key",
    )

    model = create_native_openai_model(
        parameters,
        {"effort": "high", "summary": "auto", "verbosity": "low"},
    )

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4.1-mini"
    assert model.use_responses_api is True
    assert model.output_version == "responses/v1"
    assert model.reasoning == {"effort": "high", "summary": "auto"}
    assert model.verbosity == "low"
    assert model.root_client.api_key == "resolved-openai-key"
    assert str(model.root_client.base_url) == "https://api.openai.com/v1/"
    assert model.root_client._client is http_clients[0]
    assert model.root_async_client.api_key == "resolved-openai-key"
    assert str(model.root_async_client.base_url) == "https://api.openai.com/v1/"
    assert model.root_async_client._client is http_clients[1]
    assert "resolved-openai-key" not in repr(parameters)


def test_openai_model_omits_reasoning_settings_when_unconfigured(
    http_clients: tuple[httpx.Client, httpx.AsyncClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightspeed_agentic.providers.models.openai import create_native_openai_model

    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient-endpoint.invalid/v1")
    model = create_native_openai_model(
        ModelParameters(model="gpt-4.1-mini", api_key="resolved-openai-key")
    )

    assert model.reasoning is None
    assert model.verbosity is None
    assert model.use_responses_api is True
    assert str(model.root_client.base_url) == "https://api.openai.com/v1/"
    assert model.root_client._client is http_clients[0]


@pytest.mark.asyncio
async def test_openai_model_sends_responses_request_with_configured_settings(
    captured_openai_requests: tuple[list[httpx.Request], httpx.Client, httpx.AsyncClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightspeed_agentic.providers.models.openai import create_native_openai_model

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-must-not-win")
    model = create_native_openai_model(
        ModelParameters(
            model="gpt-4.1-mini",
            base_url="https://api.openai.com/v1",
            api_key="resolved-openai-key",
        ),
        {"effort": "high", "summary": "auto", "verbosity": "low"},
    )

    await model.ainvoke("capture this request")

    requests, _sync_client, async_client = captured_openai_requests
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == "Bearer resolved-openai-key"
    assert async_client is model.root_async_client._client
    body = json.loads(request.content)
    assert body["model"] == "gpt-4.1-mini"
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["text"]["verbosity"] == "low"
    assert "capture this request" in json.dumps(body["input"])


@pytest.mark.asyncio
async def test_openai_model_omits_reasoning_from_request_when_unconfigured(
    captured_openai_requests: tuple[list[httpx.Request], httpx.Client, httpx.AsyncClient],
) -> None:
    from lightspeed_agentic.providers.models.openai import create_native_openai_model

    model = create_native_openai_model(
        ModelParameters(model="gpt-4.1-mini", api_key="resolved-openai-key")
    )
    await model.ainvoke("use model defaults")

    requests, _sync_client, _async_client = captured_openai_requests
    body = json.loads(requests[0].content)
    assert "reasoning" not in body
    assert "verbosity" not in body.get("text", {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "captured_openai_requests",
    [
        {
            "status": "ok",
            "details": {"count": 1},
            "variant": "example",
            "reference": {"enabled": True},
        }
    ],
    indirect=True,
)
async def test_native_openai_structured_output_sends_strict_schema_in_responses_request(
    captured_openai_requests: tuple[list[httpx.Request], httpx.Client, httpx.AsyncClient],
) -> None:
    from opentelemetry.context import Context

    from lightspeed_agentic.providers.deepagents import (
        _create_generation_callback,
        _shape_structured_output,
    )
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute
    from lightspeed_agentic.providers.models.openai import (
        create_native_openai_model,
        native_openai_structured_output_schema,
    )

    route = ResolvedModelRoute(
        route="openai",
        model="gpt-4.1-mini",
        create_model=create_native_openai_model,
        model_parameters=ModelParameters(
            model="gpt-4.1-mini",
            base_url="https://api.openai.com/v1",
            api_key="resolved-openai-key",
        ),
        telemetry_provider="openai",
        structured_output_method="json_schema",
        structured_output_schema_mode="raw_json_schema",
        translate_output_schema=native_openai_structured_output_schema,
    )
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "description": "Overall status"},
            "details": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
            "variant": {
                "oneOf": [{"type": "string"}, {"type": "integer"}],
                "description": "A string or integer variant",
            },
            "reference": {"$ref": "#/$defs/Reference"},
        },
        "required": ["status"],
        "$defs": {
            "Reference": {
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
            }
        },
    }

    result, _input_tokens, _output_tokens = await _shape_structured_output(
        route,
        schema,
        "Return a status.",
        "How did it go?",
        "It went well.",
        _create_generation_callback(route, Context(), main_name=None),
    )

    requests, _sync_client, _async_client = captured_openai_requests
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["model"] == "gpt-4.1-mini"
    assert "reasoning" not in body
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert body["text"]["format"]["schema"]["additionalProperties"] is False
    strict_schema = body["text"]["format"]["schema"]
    jsonschema.validate(result, strict_schema)
    assert strict_schema["required"] == ["status", "details", "variant", "reference"]
    assert strict_schema["properties"]["details"]["required"] == ["count"]
    assert strict_schema["properties"]["details"]["additionalProperties"] is False
    assert strict_schema["properties"]["variant"]["anyOf"] == [
        {"type": "string"},
        {"type": "integer"},
    ]
    assert "oneOf" not in strict_schema["properties"]["variant"]
    assert strict_schema["properties"]["reference"] == {"$ref": "#/$defs/Reference"}
    assert strict_schema["$defs"]["Reference"]["required"] == ["enabled"]
    assert strict_schema["properties"]["status"]["description"] == "Overall status"
    assert result == {
        "status": "ok",
        "details": {"count": 1},
        "variant": "example",
        "reference": {"enabled": True},
    }
    assert "oneOf" in schema["properties"]["variant"]
    assert "anyOf" not in schema["properties"]["variant"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model", ["org/served-model", "gpt-3.5-turbo", "gpt-4", "gpt-4-served-locally"]
)
@pytest.mark.parametrize(
    ("schema", "captured_openai_requests"),
    [
        (
            {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Overall status"},
                    "optional_note": {"type": "string"},
                },
                "required": ["status"],
            },
            {"status": "ok"},
        ),
        (
            {
                "type": "object",
                "description": "A result with optional nested fields",
                "properties": {
                    "status": {"type": "string"},
                    "details": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "optional_note": {"type": "string"},
                        },
                        "required": ["count"],
                        "additionalProperties": False,
                    },
                    "variant": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                    "reference": {"$ref": "#/$defs/Reference"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"],
                        },
                    },
                },
                "required": ["status"],
                "$defs": {
                    "Reference": {
                        "type": "object",
                        "properties": {"enabled": {"type": "boolean"}},
                        "required": ["enabled"],
                        "additionalProperties": True,
                    }
                },
                "allOf": [{"properties": {"status": {"minLength": 1}}}],
                "additionalProperties": True,
            },
            {
                "status": "ok",
                "details": {"count": 1},
                "variant": "example",
                "reference": {"enabled": True},
                "items": [{"value": "first"}],
                "extra": "allowed",
            },
        ),
    ],
    indirect=["captured_openai_requests"],
)
async def test_compatible_openai_shaping_preserves_schema_in_chat_completions_request(
    model: str,
    schema: dict[str, Any],
    captured_openai_requests: tuple[list[httpx.Request], httpx.Client, httpx.AsyncClient],
) -> None:
    from copy import deepcopy

    from opentelemetry.context import Context

    from lightspeed_agentic.config import ResolvedModelFamily
    from lightspeed_agentic.providers.deepagents import (
        _create_generation_callback,
        _shape_structured_output,
    )
    from lightspeed_agentic.providers.model_router import resolve_model_route

    original = deepcopy(schema)
    sdk, route = resolve_model_route(
        model,
        model_family=ResolvedModelFamily(
            "openai",
            (),
            hosting="openai",
            model="org/served-model",
            base_url="https://vllm.example.test/v1",
            api_key="EMPTY",
        ),
    )
    assert sdk is None
    assert route is not None

    result, input_tokens, output_tokens = await _shape_structured_output(
        route,
        schema,
        "Return a status.",
        "How did it go?",
        "It went well.",
        _create_generation_callback(route, Context(), main_name=None),
    )

    requests, sync_client, async_client = captured_openai_requests
    assert len(requests) == 1
    assert str(requests[0].url) == "https://vllm.example.test/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer EMPTY"
    body = json.loads(requests[0].content)
    assert body["model"] == model
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "structured_output", "schema": original, "strict": False},
    }
    assert "tools" not in body
    assert "reasoning" not in body
    assert "reasoning_effort" not in body
    assert "verbosity" not in body
    assert "It went well." in json.dumps(body["messages"])
    assert schema == original
    assert result["status"] == "ok"
    jsonschema.validate(result, original)
    assert (input_tokens, output_tokens) == (3, 2)
    assert sync_client.is_closed
    assert async_client.is_closed


@pytest.mark.asyncio
async def test_compatible_shaping_binds_schema_on_unmodified_sdk_model(
    captured_openai_requests: tuple[list[httpx.Request], httpx.Client, httpx.AsyncClient],
) -> None:
    from dataclasses import replace

    from langchain_openai import ChatOpenAI
    from opentelemetry.context import Context

    from lightspeed_agentic.config import ResolvedModelFamily
    from lightspeed_agentic.providers.deepagents import (
        _create_generation_callback,
        _shape_structured_output,
    )
    from lightspeed_agentic.providers.model_router import resolve_model_route

    sdk, route = resolve_model_route(
        model_family=ResolvedModelFamily(
            "openai",
            (),
            hosting="openai",
            model="gpt-4-served-locally",
            base_url="https://vllm.example.test/v1",
            api_key="EMPTY",
        )
    )
    assert sdk is None
    assert route is not None
    requests, sync_client, async_client = captured_openai_requests
    model = ChatOpenAI(
        model=route.model,
        base_url=route.model_parameters.base_url,
        api_key=route.model_parameters.api_key,
        use_responses_api=False,
        http_client=sync_client,
        http_async_client=async_client,
    )
    route = replace(route, create_model=lambda _parameters, _reasoning: model)
    result, input_tokens, output_tokens = await _shape_structured_output(
        route,
        {"type": "object", "properties": {"status": {"type": "string"}}},
        "Return the status.",
        "How did it go?",
        "It went well.",
        _create_generation_callback(route, Context(), main_name=None),
    )
    assert result == {"status": "ok"}
    assert (input_tokens, output_tokens) == (3, 2)
    assert len(requests) == 1
    assert sync_client.is_closed
    assert async_client.is_closed


def _openai_response(
    response_id: str,
    item: dict[str, Any],
    *,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1_700_000_000,
        "model": "gpt-4.1-mini",
        "output": [item],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _openai_response_stream(
    response_id: str,
    item: dict[str, Any],
    *,
    input_tokens: int,
    output_tokens: int,
    reasoning: bool = False,
) -> httpx.Response:
    """Encode scripted native Responses events, including real streaming tool arguments."""
    response = _openai_response(
        response_id, item, input_tokens=input_tokens, output_tokens=output_tokens
    )
    events = [
        {
            "type": "response.created",
            "response": {**response, "output": [], "status": "in_progress", "usage": None},
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**item, "arguments": "", "status": "in_progress"}
            if item["type"] == "function_call"
            else {**item, "content": [], "status": "in_progress"},
        },
    ]
    if item["type"] == "function_call":
        events.append(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"],
                "output_index": 0,
                "delta": item["arguments"],
            }
        )
    else:
        events.extend(
            [
                {
                    "type": "response.content_part.added",
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "part": {**item["content"][0], "text": ""},
                },
                {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "delta": item["content"][0]["text"],
                    "logprobs": [],
                },
            ]
        )
    events.extend(
        [
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": response},
        ]
    )
    if reasoning:
        reasoning_item = {
            "id": f"rs_{response_id}",
            "type": "reasoning",
            "summary": [],
            "status": "completed",
        }
        response["output"].insert(0, reasoning_item)
        for event in events:
            if "output_index" in event:
                event["output_index"] += 1
        # A completed stream must not be confused with the initial item snapshot
        # that LangChain retains in its aggregated reasoning content block.
        events[1:1] = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**reasoning_item, "status": "in_progress"},
            },
            {"type": "response.output_item.done", "output_index": 0, "item": reasoning_item},
        ]
    content = "".join(
        f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': index})}\n\n"
        for index, event in enumerate(events)
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)


@pytest.mark.asyncio
async def test_native_openai_deepagent_reads_file_and_shapes_structured_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    span_exporter: Any,
) -> None:
    from lightspeed_agentic.config import ResolvedModelFamily
    from lightspeed_agentic.providers.deepagents import DeepAgentsProvider
    from lightspeed_agentic.providers.model_router import resolve_model_route
    from lightspeed_agentic.types import (
        ProviderQueryOptions,
        ResultEvent,
        ToolCallEvent,
        ToolResultEvent,
    )

    sentinel_file = tmp_path / "sentinel.txt"
    sentinel_file.write_text("native-openai-file-sentinel\n")
    schema = {
        "type": "object",
        "properties": {"sentinel": {"type": "string", "description": "Contents of the file"}},
        "required": ["sentinel"],
    }
    requests: list[dict[str, Any]] = []

    def message(text: str) -> dict[str, Any]:
        return {
            "id": "msg_native-integration",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer resolved-openai-key"
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == "gpt-4.1-mini"
        if len(requests) == 1:
            assert body["stream"] is True
            assert "read_file" in {tool["name"] for tool in body["tools"]}
            assert "format" not in body.get("text", {})
            return _openai_response_stream(
                "resp_read-file",
                {
                    "id": "fc_read-file",
                    "type": "function_call",
                    "name": "read_file",
                    "call_id": "call_read-file",
                    "arguments": json.dumps({"file_path": str(sentinel_file)}),
                    "status": "completed",
                },
                input_tokens=10,
                output_tokens=2,
            )
        if len(requests) == 2:
            tool_output = next(
                item for item in body["input"] if item.get("type") == "function_call_output"
            )
            assert tool_output["call_id"] == "call_read-file"
            assert "native-openai-file-sentinel" in tool_output["output"]
            assert '<tool_data source="read_file">' in tool_output["output"]
            return _openai_response_stream(
                "resp_agent-answer",
                message("The file contains native-openai-file-sentinel."),
                input_tokens=20,
                output_tokens=4,
            )
        assert len(requests) == 3, "Unexpected extra model call"
        assert not body.get("tools")
        assert "native-openai-file-sentinel" in json.dumps(body["input"])
        assert body["text"]["format"] == {
            "type": "json_schema",
            "name": "structured_output",
            "strict": True,
            "schema": {**schema, "additionalProperties": False},
        }
        return httpx.Response(
            200,
            json=_openai_response(
                "resp_structured-answer",
                message('{"sentinel": "native-openai-file-sentinel"}'),
                input_tokens=5,
                output_tokens=3,
            ),
        )

    transport = httpx.MockTransport(respond)
    sync_clients: list[httpx.Client] = []
    async_clients: list[httpx.AsyncClient] = []

    def create_http_client() -> httpx.Client:
        client = httpx.Client(transport=transport)
        sync_clients.append(client)
        return client

    def create_async_http_client() -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=transport)
        async_clients.append(client)
        return client

    monkeypatch.setattr(tls, "create_http_client", create_http_client)
    monkeypatch.setattr(tls, "create_async_http_client", create_async_http_client)
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-must-not-win")
    monkeypatch.setenv("HOME", str(tmp_path))
    sdk, route = resolve_model_route(
        model_family=ResolvedModelFamily(
            name="openai",
            hosting="openai",
            model="gpt-4.1-mini",
            api_key="resolved-openai-key",
            expected_envs=("OPENAI_API_KEY",),
        )
    )
    assert sdk is None
    assert route is not None
    provider = DeepAgentsProvider(route)
    try:
        events = [
            event
            async for event in provider.query(
                ProviderQueryOptions(
                    prompt=f"Read {sentinel_file} and report its contents.",
                    system_prompt="Read the requested file and report its contents unchanged.",
                    model="gpt-4.1-mini",
                    cwd=str(tmp_path),
                    max_turns=10,
                    allowed_tools=["Read"],
                    output_schema=schema,
                    tool_output_inspection_enabled=False,
                )
            )
        ]
    finally:
        for sync_client in sync_clients:
            sync_client.close()
        for async_client in async_clients:
            await async_client.aclose()

    assert len(requests) == 3
    calls = [event for event in events if isinstance(event, ToolCallEvent)]
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].call_id == "call_read-file"
    assert json.loads(calls[0].input) == {"file_path": str(sentinel_file)}
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].call_id == "call_read-file"
    assert results[0].error_type is None
    assert "native-openai-file-sentinel" in results[0].output
    assert isinstance(events[-1], ResultEvent)
    assert json.loads(events[-1].text) == {"sentinel": "native-openai-file-sentinel"}
    # Both streamed generations and the shaping call contribute exactly once.
    assert (events[-1].input_tokens, events[-1].output_tokens) == (35, 9)

    generations = [
        span for span in span_exporter.get_finished_spans() if span.name == "chat gpt-4.1-mini"
    ]
    assert [
        (
            span.attributes["gen_ai.response.id"],
            span.attributes["gen_ai.usage.input_tokens"],
            span.attributes["gen_ai.usage.output_tokens"],
        )
        for span in generations
    ] == [
        ("resp_read-file", 10, 2),
        ("resp_agent-answer", 20, 4),
        ("resp_structured-answer", 5, 3),
    ]
    for span in generations:
        assert span.attributes["gen_ai.provider.name"] == "openai"
        assert span.attributes["gen_ai.response.model"] == "gpt-4.1-mini"


@pytest.mark.parametrize("native", [False, True], ids=["compatible-chat", "native-responses"])
@pytest.mark.parametrize(
    "reply_kind", ["benign", "malicious", "conflicting-decisions", "duplicate-keys"]
)
@pytest.mark.asyncio
async def test_openai_classifier_validates_original_native_decisions(
    captured_openai_requests: Any,
    monkeypatch: pytest.MonkeyPatch,
    native: bool,
    reply_kind: str,
) -> None:
    from lightspeed_agentic.inspection.client import LangChainClassifierClient
    from lightspeed_agentic.inspection.errors import ClassifierResponseError
    from lightspeed_agentic.inspection.models import ClassifierRequest
    from lightspeed_agentic.providers.deepagents import _close_model_clients
    from lightspeed_agentic.providers.models.openai import (
        create_native_openai_model,
        create_openai_compatible_model,
    )

    parameters = ModelParameters(
        model="gpt-4.1-mini" if native else "served-model",
        api_key="resolved-key",
        base_url="https://api.openai.com/v1" if native else "https://custom.example/v1",
    )
    model = (create_native_openai_model if native else create_openai_compatible_model)(parameters)
    output_version = model.output_version
    transport = model.root_async_client._client._transport
    native_request = transport.handle_async_request

    async def reply(request: httpx.Request) -> httpx.Response:
        await native_request(request)
        detected = reply_kind == "malicious"
        arguments = json.dumps(
            {"injectionDetected": detected, "category": "unknown" if detected else "none"}
        )
        if reply_kind == "duplicate-keys":
            arguments = (
                '{"injectionDetected":true,"category":"unknown",'
                '"injectionDetected":false,"category":"none"}'
            )
        calls = [
            {
                "type": "function_call",
                "name": "ClassifierDecision",
                "arguments": arguments,
                "id": "fc_one",
                "call_id": "call_one",
                "status": "completed",
            }
        ]
        if reply_kind == "conflicting-decisions":
            calls.append(
                {
                    "type": "function_call",
                    "name": "ClassifierDecision",
                    "arguments": '{"injectionDetected":true,"category":"unknown"}',
                    "id": "fc_two",
                    "call_id": "call_two",
                    "status": "completed",
                }
            )
        if native:
            payload = _openai_response("resp_classifier", calls[0], input_tokens=3, output_tokens=2)
            payload["output"] = calls
            return httpx.Response(200, json=payload, request=request)
        return _openai_chat_completion(
            "chatcmpl_classifier",
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call["call_id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                    for call in calls
                ],
            },
            stream=False,
        )

    monkeypatch.setattr(transport, "handle_async_request", reply)
    try:
        client = LangChainClassifierClient(model)
        request = ClassifierRequest(
            toolName="probe", resultType="result", chunkIndex=0, chunkCount=1, content="data"
        )
        if reply_kind in ("conflicting-decisions", "duplicate-keys"):
            with pytest.raises(ClassifierResponseError):
                await client.classify(request)
        else:
            decision = await client.classify(request)
            assert (decision.injection_detected, decision.category) == (
                reply_kind == "malicious",
                "unknown" if reply_kind == "malicious" else "none",
            )
        assert model.output_version == output_version
    finally:
        await _close_model_clients(model)
    requests, sync_client, async_client = captured_openai_requests
    assert len(requests) == 1
    assert sync_client.is_closed
    assert async_client.is_closed


def _openai_chat_completion(
    response_id: str,
    message: dict[str, Any],
    *,
    stream: bool = False,
    input_tokens: int = 10,
    output_tokens: int = 2,
    reasoning_field: str = "reasoning_content",
) -> httpx.Response:
    """Script Chat Completions HTTP, including fragmented streaming tool arguments."""
    metadata = {
        "id": response_id,
        "created": 1_700_000_000,
        "model": "served-model-revision-7",
    }
    usage = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
    if not stream:
        return httpx.Response(
            200,
            json={
                **metadata,
                "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": usage,
            },
        )

    deltas: list[dict[str, Any]] = [{"role": "assistant", "content": ""}]
    for index, call in enumerate(message.get("tool_calls", [])):
        arguments = call["function"]["arguments"]
        split = len(arguments) // 2
        deltas.extend(
            [
                {
                    "tool_calls": [
                        {
                            "index": index,
                            **call,
                            "function": {
                                **call["function"],
                                "arguments": arguments[:split],
                            },
                        }
                    ]
                },
                {"tool_calls": [{"index": index, "function": {"arguments": arguments[split:]}}]},
            ]
        )
    if message.get("content"):
        deltas.append({"content": message["content"]})
    reasoning = message.get(reasoning_field)
    if reasoning:
        split = len(reasoning) // 2
        deltas[0][reasoning_field] = None
        deltas.insert(1, {"content": None, reasoning_field: reasoning[:split]})
        if len(deltas) == 2:
            deltas.append({})
        deltas[2][reasoning_field] = reasoning[split:]
    chunks = [
        {
            **metadata,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        for delta in deltas
    ]
    chunks.extend(
        [
            {
                **metadata,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {reasoning_field: ""} if reasoning else {},
                        "finish_reason": finish_reason,
                    }
                ],
            },
            {**metadata, "object": "chat.completion.chunk", "choices": [], "usage": usage},
        ]
    )
    content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)


@pytest.mark.asyncio
@pytest.mark.parametrize("classifier_format", ["function_call", "json_text"])
@pytest.mark.parametrize("reasoning_field", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize(
    ("structured_output", "expected_usage", "expected_shape_count", "served_model"),
    [(False, (60, 9), 0, "org/served-model"), (True, (65, 10), 1, "gpt-4-served-locally")],
)
async def test_compatible_openai_deepagent_runs_skill_and_admitted_mcp_with_inspection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    span_exporter: Any,
    structured_output: bool,
    expected_usage: tuple[int, int],
    expected_shape_count: int,
    served_model: str,
    reasoning_field: str,
    classifier_format: str,
) -> None:
    from lightspeed_agentic.config import resolve_model_family
    from lightspeed_agentic.factory import create_provider
    from lightspeed_agentic.mcp import AdmittedMCPProviderServer, ResolvedMCPHeader
    from lightspeed_agentic.providers.model_router import resolve_model_route
    from lightspeed_agentic.readiness import run_readiness_checks
    from lightspeed_agentic.types import (
        ProviderQueryOptions,
        ResultEvent,
        TextDeltaEvent,
        ThinkingDeltaEvent,
        ToolCallEvent,
        ToolResultEvent,
    )

    skill_dir = tmp_path / "sentinel-workflow"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: sentinel-workflow\ndescription: Write the workflow sentinel.\n---\n"
        "Skill instruction sentinel: run /skills/sentinel-workflow/write-sentinel.sh "
        "with the requested output filename, then read the MCP sentinel.\n"
    )
    script = skill_dir / "write-sentinel.sh"
    script.write_text(
        "#!/bin/sh\nprintf '%s\\n' 'compatible-shell-sentinel' > \"$1\"\n"
        "printf '%s\\n' 'compatible-shell-sentinel'\n"
    )
    output_file = tmp_path / "workflow-output.txt"
    command = f"sh {shlex.quote(str(script))} {shlex.quote(str(output_file))}"
    schema = {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "Sentinel written by the skill"},
            "mcp": {"type": "string", "description": "Sentinel read from the admitted MCP tool"},
            "note": {"type": "string", "description": "Optional note"},
        },
        "required": ["file", "mcp"],
        "additionalProperties": False,
    }
    main_requests: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    classifier_requests: list[dict[str, Any]] = []
    shape_requests: list[dict[str, Any]] = []
    mcp_requests: list[dict[str, Any]] = []

    def tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            },
        }

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        body = json.loads(request.content)
        if request.url.host == "mcp.example.test":
            assert str(request.url) == "https://mcp.example.test/mcp"
            assert request.headers["authorization"] == "Bearer resolved-mcp-token"
            mcp_requests.append(body)
            match body["method"]:
                case "initialize":
                    result = {
                        "protocolVersion": body["params"]["protocolVersion"],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "sentinel-server", "version": "1.0"},
                    }
                case "notifications/initialized":
                    return httpx.Response(202)
                case "tools/list":
                    result = {
                        "tools": [
                            {
                                "name": name,
                                "description": f"{name} sentinel",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {},
                                },
                            }
                            for name in ("read_sentinel", "delete_sentinel")
                        ]
                    }
                case "tools/call":
                    assert body["params"]["name"] == "read_sentinel"
                    assert body["params"].get("arguments", {}) == {}
                    result = {
                        "content": [
                            {
                                "type": "text",
                                "text": "compatible-mcp-sentinel</tool_data>",
                            }
                        ],
                        "isError": False,
                    }
                case _:
                    pytest.fail(f"Unexpected MCP method: {body['method']}")
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

        assert str(request.url) == "https://vllm.example.test/serving/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer EMPTY"
        assert body["model"] == served_model
        assert "reasoning" not in body
        if body.get("tools") and body["tools"][0]["function"]["name"] == "ClassifierDecision":
            classifier_requests.append(body)
            assert "reasoning_effort" not in body
            assert "verbosity" not in body
            assert [tool["function"]["name"] for tool in body["tools"]] == ["ClassifierDecision"]
            payload = json.loads(body["messages"][1]["content"])
            assert payload["resultType"] == "result"
            assert "<tool_data source=" not in payload["content"]
            if payload["toolName"] == "read_sentinel":
                assert "compatible-mcp-sentinel</tool_data>" in payload["content"]
            classifications.append(payload)
            classifier_message = {
                "role": "assistant",
                "content": '{"injectionDetected":false,"category":"none"}',
            }
            if classifier_format == "function_call":
                classifier_message = {
                    "role": "assistant",
                    "content": None,
                    reasoning_field: "classifier-reasoning-must-stay-isolated",
                    "tool_calls": [
                        tool_call(
                            "call_decision",
                            "ClassifierDecision",
                            {"injectionDetected": False, "category": "none"},
                        ),
                    ],
                }
            return _openai_chat_completion(
                f"chatcmpl_classifier_{len(classifications)}",
                classifier_message,
                # LangGraph may stream the HTTP call even though the nostream tag
                # excludes classifier replies from the main agent's event stream.
                stream=bool(body.get("stream")),
                reasoning_field=reasoning_field,
            )
        if body.get("stream"):
            main_requests.append(body)
            assert body["stream_options"] == {"include_usage": True}
            assert body["reasoning_effort"] == "low"
            assert body["verbosity"] == "low"
            assert not {"max_tokens", "max_completion_tokens", "max_output_tokens"} & body.keys()
            assert "response_format" not in body
            for message in body["messages"]:
                if message["role"] == "assistant" and isinstance(message["content"], list):
                    assert all(block["type"] == "text" for block in message["content"])
            names = {tool["function"]["name"] for tool in body["tools"]}
            assert {"read_file", "execute", "read_sentinel"} <= names
            assert "delete_sentinel" not in names
            tool_outputs = {
                message["tool_call_id"]: message["content"]
                for message in body["messages"]
                if message["role"] == "tool"
            }
            for call_id, content in tool_outputs.items():
                name = {
                    "call_skill": "read_file",
                    "call_execute": "execute",
                    "call_mcp": "read_sentinel",
                }[call_id]
                assert content.startswith(f'<tool_data source="{name}">\n')
                assert content.endswith("\n</tool_data>")
                assert content.count("</tool_data>") == 1
            if len(main_requests) == 1:
                assert "/skills/sentinel-workflow/SKILL.md" in json.dumps(body["messages"])
                assert not tool_outputs
                calls = [
                    tool_call(
                        "call_skill",
                        "read_file",
                        {
                            "file_path": "/skills/sentinel-workflow/SKILL.md",
                        },
                    )
                ]
            elif len(main_requests) == 2:
                assert "Skill instruction sentinel" in tool_outputs["call_skill"]
                assert [item["toolName"] for item in classifications] == ["read_file"]
                calls = [
                    tool_call("call_execute", "execute", {"command": command}),
                    tool_call("call_mcp", "read_sentinel", {}),
                ]
            else:
                assert len(main_requests) == 3, "Unexpected extra agent call"
                assert "compatible-shell-sentinel" in tool_outputs["call_execute"]
                assert "compatible-mcp-sentinel<\\/tool_data>" in tool_outputs["call_mcp"]
                assert {item["toolName"] for item in classifications} == {
                    "read_file",
                    "execute",
                    "read_sentinel",
                }
                assert output_file.read_text() == "compatible-shell-sentinel\n"
                return _openai_chat_completion(
                    "chatcmpl_answer",
                    {
                        "role": "assistant",
                        "content": "The skill wrote compatible-shell-sentinel; "
                        "the admitted MCP tool returned compatible-mcp-sentinel.",
                        reasoning_field: "main-reasoning-3-part-a;main-reasoning-3-part-b;",
                    },
                    stream=True,
                    reasoning_field=reasoning_field,
                    input_tokens=30,
                    output_tokens=4,
                )
            return _openai_chat_completion(
                f"chatcmpl_tools_{len(main_requests)}",
                {
                    "role": "assistant",
                    "content": None,
                    reasoning_field: f"main-reasoning-{len(main_requests)}-part-a;"
                    f"main-reasoning-{len(main_requests)}-part-b;",
                    "tool_calls": calls,
                },
                stream=True,
                reasoning_field=reasoning_field,
                input_tokens=10 if len(main_requests) == 1 else 20,
                output_tokens=2 if len(main_requests) == 1 else 3,
            )

        assert "reasoning_effort" not in body
        assert "verbosity" not in body
        assert not {"max_tokens", "max_completion_tokens", "max_output_tokens"} & body.keys()
        assert not body.get("tools")
        shape_requests.append(body)
        assert len(main_requests) == 3
        assert body["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": schema,
                "strict": False,
            },
        }
        shape_input = body["messages"][1]["content"]
        assert (
            "Agent run output:\n"
            "The skill wrote compatible-shell-sentinel; "
            "the admitted MCP tool returned compatible-mcp-sentinel.\n\n"
        ) in shape_input
        assert "call_decision" not in shape_input
        assert "tool_data" not in shape_input
        assert "main-reasoning-" not in shape_input
        assert "classifier-reasoning-" not in shape_input
        return _openai_chat_completion(
            "chatcmpl_shape",
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "file": "compatible-shell-sentinel",
                        "mcp": "compatible-mcp-sentinel",
                    }
                ),
            },
            input_tokens=5,
            output_tokens=1,
        )

    transport = httpx.MockTransport(respond)
    sync_clients: list[httpx.Client] = []
    async_clients: list[httpx.AsyncClient] = []

    def create_http_client() -> httpx.Client:
        client = httpx.Client(transport=transport)
        sync_clients.append(client)
        return client

    def create_async_http_client(**kwargs: Any) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=transport, **kwargs)
        async_clients.append(client)
        return client

    monkeypatch.setattr(tls, "create_http_client", create_http_client)
    monkeypatch.setattr(tls, "create_async_http_client", create_async_http_client)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "openai")
    monkeypatch.setenv("LIGHTSPEED_MODEL", served_model)
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://vllm.example.test/serving/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    family = resolve_model_family()
    sdk, route = resolve_model_route(model_family=family)
    assert sdk is None
    assert route is not None
    assert run_readiness_checks(family, route) == (
        True,
        {"provider_env": "ok", "model_route": "ok"},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-must-not-win")
    provider = create_provider("deepagents", model_route=route)
    try:
        events = [
            event
            async for event in provider.query(
                ProviderQueryOptions(
                    prompt=(
                        "Use sentinel-workflow to write workflow-output.txt, "
                        "then read the MCP sentinel."
                    ),
                    system_prompt="Follow the skill and report both sentinels unchanged.",
                    model="options-model-must-not-win",
                    cwd=str(tmp_path),
                    max_turns=10,
                    allowed_tools=["Read", "Bash"],
                    reasoning_config={"effort": "low", "verbosity": "low"},
                    output_schema=schema if structured_output else None,
                    tool_output_inspection_enabled=True,
                    mcp_servers=[
                        AdmittedMCPProviderServer(
                            name="sentinel-server",
                            url="https://mcp.example.test/mcp",
                            headers=(
                                ResolvedMCPHeader("Authorization", "Bearer resolved-mcp-token"),
                            ),
                            allowed_tool_names=("read_sentinel",),
                        )
                    ],
                )
            )
        ]
        assert all(client.is_closed for client in sync_clients + async_clients)
    finally:
        for sync_client in sync_clients:
            sync_client.close()
        for async_client in async_clients:
            await async_client.aclose()

    assert (len(main_requests), len(classifications), len(shape_requests)) == (
        3,
        3,
        expected_shape_count,
    )
    assert [
        body.get("max_completion_tokens", body.get("max_tokens")) for body in classifier_requests
    ] == [4096, 4096, 4096]
    assert [item["params"]["name"] for item in mcp_requests if item["method"] == "tools/call"] == [
        "read_sentinel"
    ]
    calls = [event for event in events if isinstance(event, ToolCallEvent)]
    assert [(call.name, call.call_id) for call in calls] == [
        ("read_file", "call_skill"),
        ("execute", "call_execute"),
        ("read_sentinel", "call_mcp"),
    ]
    assert json.loads(calls[0].input) == {"file_path": "/skills/sentinel-workflow/SKILL.md"}
    assert json.loads(calls[1].input) == {"command": command}
    tool_results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(tool_results) == 3
    results = {event.call_id: event for event in tool_results}
    assert set(results) == {"call_skill", "call_execute", "call_mcp"}
    assert all(result.error_type is None for result in results.values())
    assert "Skill instruction sentinel" in results["call_skill"].output
    assert "compatible-shell-sentinel" in results["call_execute"].output
    assert "compatible-mcp-sentinel</tool_data>" in results["call_mcp"].output
    assert output_file.read_text() == "compatible-shell-sentinel\n"
    expected_answer = (
        "The skill wrote compatible-shell-sentinel; "
        "the admitted MCP tool returned compatible-mcp-sentinel."
    )
    assert [
        (event.type, event.thinking if isinstance(event, ThinkingDeltaEvent) else event.text)
        for event in events
        if isinstance(event, (ThinkingDeltaEvent, TextDeltaEvent))
    ] == [
        ("thinking_delta", "main-reasoning-1-part-a;"),
        ("thinking_delta", "main-reasoning-1-part-b;"),
        ("thinking_delta", "main-reasoning-2-part-a;"),
        ("thinking_delta", "main-reasoning-2-part-b;"),
        ("thinking_delta", "main-reasoning-3-part-a;"),
        ("thinking_delta", "main-reasoning-3-part-b;"),
        ("text_delta", expected_answer),
    ]
    assert isinstance(events[-1], ResultEvent)
    if structured_output:
        assert json.loads(events[-1].text) == {
            "file": "compatible-shell-sentinel",
            "mcp": "compatible-mcp-sentinel",
        }
        jsonschema.validate(json.loads(events[-1].text), schema)
    else:
        assert events[-1].text == expected_answer
    assert os.environ["OPENAI_API_KEY"] == "ambient-key-must-not-win"
    assert os.environ["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
    assert (events[-1].input_tokens, events[-1].output_tokens) == expected_usage
    generations = [
        span for span in span_exporter.get_finished_spans() if span.name == f"chat {served_model}"
    ]
    expected_generations = [
        ("chatcmpl_tools_1", 10, 2, ["tool_calls"]),
        ("chatcmpl_tools_2", 20, 3, ["tool_calls"]),
        ("chatcmpl_answer", 30, 4, ["stop"]),
    ]
    if structured_output:
        expected_generations.append(("chatcmpl_shape", 5, 1, ["stop"]))
    assert [
        (
            span.attributes["gen_ai.response.id"],
            span.attributes["gen_ai.usage.input_tokens"],
            span.attributes["gen_ai.usage.output_tokens"],
            list(span.attributes["gen_ai.response.finish_reasons"]),
        )
        for span in generations
    ] == expected_generations
    for span in generations:
        assert span.attributes["gen_ai.provider.name"] == "openai"
        assert span.attributes["gen_ai.request.model"] == served_model
        assert span.attributes["gen_ai.response.model"] == "served-model-revision-7"


@pytest.mark.parametrize(
    ("case", "shape_content"),
    [
        ("invalid-json", "not-json"),
        ("unclosed-json", '{"status":"ok"'),
        ("truncated-extra-field", '{"status":"ok","forbidden":'),
        ("empty", ""),
        ("missing-content", None),
        ("null", "null"),
        ("missing-field", "{}"),
        ("wrong-type", '{"status": 7}'),
        ("wrong-enum", '{"status": "other"}'),
        ("extra-field", '{"status": "ok", "extra": true}'),
        ("non-object", "[]"),
        ("agent-http-error", None),
        ("shape-http-error", None),
        ("mcp-initialize-error", None),
        ("graph-setup-error", None),
    ],
)
@pytest.mark.asyncio
async def test_compatible_openai_failures_return_failed_result_and_close_clients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    span_exporter: Any,
    case: str,
    shape_content: str | None,
) -> None:
    from opentelemetry.trace import StatusCode

    from lightspeed_agentic.config import ResolvedModelFamily
    from lightspeed_agentic.factory import create_provider
    from lightspeed_agentic.mcp import AdmittedMCPProviderServer, ResolvedMCPHeader
    from lightspeed_agentic.providers.model_router import resolve_model_route
    from lightspeed_agentic.run_agent import run_agent_query

    setup_failure = case in {"mcp-initialize-error", "graph-setup-error"}
    requests: list[dict[str, Any]] = []
    mcp_requests: list[dict[str, Any]] = []
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["ok"]}},
        "required": ["status"],
        "additionalProperties": False,
    }

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.host == "mcp.example.test":
            assert case == "mcp-initialize-error"
            assert str(request.url) == "https://mcp.example.test/mcp"
            assert request.headers["authorization"] == "Bearer resolved-mcp-token"
            body = json.loads(request.content)
            assert body["method"] == "initialize"
            mcp_requests.append(body)
            return httpx.Response(401, json={"error": "Unauthorized"})
        assert str(request.url) == "https://vllm.example.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer EMPTY"
        body = json.loads(request.content)
        requests.append(body)
        assert len(requests) <= 2, "Unexpected retry or additional model call"
        if case == "agent-http-error" or (case == "shape-http-error" and len(requests) == 2):
            return httpx.Response(
                401,
                headers={"x-request-id": "req_unauthorized"},
                json={"error": {"message": "Invalid credentials", "type": "authentication_error"}},
            )
        if len(requests) == 1:
            assert body["stream"] is True
            assert "response_format" not in body
            return _openai_chat_completion(
                "chatcmpl_failure_agent",
                {"role": "assistant", "content": "The status is ok."},
                stream=True,
            )
        assert not body.get("tools")
        assert "reasoning_effort" not in body
        assert body["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": schema,
                "strict": False,
            },
        }
        return _openai_chat_completion(
            "chatcmpl_invalid_shape", {"role": "assistant", "content": shape_content}
        )

    transport = httpx.MockTransport(respond)
    sync_clients: list[httpx.Client] = []
    async_clients: list[httpx.AsyncClient] = []

    def create_http_client() -> httpx.Client:
        client = httpx.Client(transport=transport)
        sync_clients.append(client)
        return client

    def create_async_http_client(**kwargs: Any) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=transport, **kwargs)
        async_clients.append(client)
        return client

    monkeypatch.setattr(tls, "create_http_client", create_http_client)
    monkeypatch.setattr(tls, "create_async_http_client", create_async_http_client)
    monkeypatch.setenv("HOME", str(tmp_path))
    if case == "graph-setup-error":

        def fail_graph(**_kwargs: Any) -> Any:
            raise RuntimeError("Graph construction failed")

        monkeypatch.setattr("deepagents.create_deep_agent", fail_graph)
    sdk, route = resolve_model_route(
        model_family=ResolvedModelFamily(
            "openai",
            (),
            hosting="openai",
            model="org/served-model",
            base_url="https://vllm.example.test/v1",
            api_key="EMPTY",
        )
    )
    assert sdk is None
    assert route is not None
    provider = create_provider("deepagents", model_route=route)
    try:
        result = await run_agent_query(
            provider,
            prompt="Report the status.",
            system_prompt="Report the status unchanged.",
            output_schema=schema,
            context=None,
            skills_dir=str(tmp_path),
            model=route.model,
            max_turns=10,
            timeout_seconds=30,
            tool_output_inspection_enabled=setup_failure,
            mcp_servers=[
                AdmittedMCPProviderServer(
                    name="unavailable-server",
                    url="https://mcp.example.test/mcp",
                    headers=(ResolvedMCPHeader("Authorization", "Bearer resolved-mcp-token"),),
                    allowed_tool_names=("read_sentinel",),
                )
            ]
            if case == "mcp-initialize-error"
            else None,
        )
        expected_clients = {
            "agent-http-error": (1, 1),
            "mcp-initialize-error": (2, 3),
        }.get(case, (2, 2))
        assert (len(sync_clients), len(async_clients)) == expected_clients
        assert all(client.is_closed for client in sync_clients + async_clients)
    finally:
        for client in sync_clients:
            client.close()
        for async_client in async_clients:
            await async_client.aclose()

    assert result.output["success"] is False
    if shape_content:
        assert shape_content not in result.output["summary"]
    spans = span_exporter.get_finished_spans()
    invocation = next(span for span in spans if span.name.startswith("invoke_agent"))
    assert invocation.status.status_code == StatusCode.ERROR
    assert "gen_ai.output.messages" not in invocation.attributes
    expected_error = {
        "agent-http-error": "OpenAIAuthenticationError",
        "shape-http-error": "OpenAIAuthenticationError",
        "mcp-initialize-error": "ExceptionGroup",
        "graph-setup-error": "RuntimeError",
    }.get(case, "ValueError")
    assert invocation.attributes["error.type"] == expected_error
    generations = [span for span in spans if span.name == "chat org/served-model"]
    expected_generations = 0 if setup_failure else 1 if case == "agent-http-error" else 2
    assert len(generations) == expected_generations
    assert len(requests) == len(generations)
    assert all(span.parent.span_id == invocation.context.span_id for span in generations)
    assert [body["method"] for body in mcp_requests] == (
        ["initialize"] if case == "mcp-initialize-error" else []
    )
    if not setup_failure:
        if case.endswith("http-error"):
            assert generations[-1].status.status_code == StatusCode.ERROR
            assert generations[-1].attributes["error.type"] == "OpenAIAuthenticationError"
            assert "gen_ai.response.id" not in generations[-1].attributes
        else:
            assert generations[-1].attributes["gen_ai.response.id"] == "chatcmpl_invalid_shape"
            assert generations[-1].attributes["gen_ai.response.model"] == "served-model-revision-7"


@pytest.mark.parametrize(
    ("reasoning_config", "api_key"),
    [
        (None, "resolved-compatible-key"),
        ({}, "EMPTY"),
        ({"effort": "high", "verbosity": "low"}, "resolved-compatible-key"),
    ],
)
@pytest.mark.asyncio
async def test_compatible_openai_model_uses_explicit_chat_completions_settings(
    captured_openai_requests: tuple[list[httpx.Request], httpx.Client, httpx.AsyncClient],
    monkeypatch: pytest.MonkeyPatch,
    reasoning_config: dict[str, Any] | None,
    api_key: str,
) -> None:
    from lightspeed_agentic.config import resolve_model_family
    from lightspeed_agentic.providers.model_router import resolve_model_route

    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "openai")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "org/Gemma-4-served:stable")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://vllm.example.test/serving/openai/v1")
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    family = resolve_model_family()
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-must-not-win")
    sdk, route = resolve_model_route(model_family=family)
    assert sdk is None
    assert route is not None
    model = route.create_model(route.model_parameters, reasoning_config)
    assert model.use_responses_api is False
    assert model.stream_usage is True
    assert model.root_client._client is captured_openai_requests[1]
    assert model.root_async_client._client is captured_openai_requests[2]

    result = await model.ainvoke("Use the configured custom endpoint.")

    requests, _sync_client, _async_client = captured_openai_requests
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://vllm.example.test/serving/openai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {api_key}"
    body = json.loads(request.content)
    assert body["model"] == "org/Gemma-4-served:stable"
    assert body["messages"] == [{"role": "user", "content": "Use the configured custom endpoint."}]
    assert "reasoning" not in body
    if reasoning_config:
        assert body["reasoning_effort"] == "high"
        assert body["verbosity"] == "low"
        assert reasoning_config == {"effort": "high", "verbosity": "low"}
    else:
        assert "reasoning_effort" not in body
        assert "verbosity" not in body
    assert result.text == '{"status": "ok"}'
    assert result.response_metadata["id"] == "chatcmpl_capture-test"
    assert os.environ["OPENAI_API_KEY"] == "ambient-key-must-not-win"
    assert os.environ["OPENAI_BASE_URL"] == "https://api.openai.com/v1"


@pytest.mark.asyncio
async def test_compatible_openai_model_does_not_silently_drop_unsupported_reasoning(
    captured_openai_requests: tuple[list[httpx.Request], httpx.Client, httpx.AsyncClient],
) -> None:
    from lightspeed_agentic.providers.models.openai import create_openai_compatible_model

    model = create_openai_compatible_model(
        ModelParameters(
            model="served-model",
            base_url="https://vllm.example.test/v1",
            api_key="resolved-compatible-key",
        ),
        {"effort": "high", "summary": "auto"},
    )

    assert model.reasoning == {"summary": "auto"}
    with pytest.raises(TypeError, match="reasoning"):
        await model.ainvoke("Unsupported settings must fail explicitly.")

    assert captured_openai_requests[0] == []


@pytest.mark.usefixtures("http_clients")
@pytest.mark.parametrize(
    ("reasoning_fields", "expected_reasoning"),
    [
        ({"reasoning": "gemma-reasoning"}, ["gemma-reasoning"]),
        ({"reasoning_content": "legacy-reasoning"}, ["legacy-reasoning"]),
        ({"reasoning": None}, []),
        ({"reasoning": ""}, []),
        ({"reasoning": {"effort": "low"}}, []),
        (
            {"reasoning_content": "same-reasoning", "reasoning": "same-reasoning"},
            ["same-reasoning"],
        ),
        (
            {"reasoning_content": "legacy-reasoning", "reasoning": "other-reasoning"},
            ["legacy-reasoning"],
        ),
        (
            {"reasoning_content": None, "reasoning": "gemma-reasoning"},
            ["gemma-reasoning"],
        ),
        (
            {"reasoning_content": "", "reasoning": "gemma-reasoning"},
            ["gemma-reasoning"],
        ),
        (
            {"reasoning_content": {"effort": "low"}, "reasoning": "gemma-reasoning"},
            ["gemma-reasoning"],
        ),
    ],
)
def test_compatible_stream_normalizes_reasoning_aliases_once_without_polluting_text(
    reasoning_fields: dict[str, Any], expected_reasoning: list[str]
) -> None:
    from langchain_core.messages import AIMessageChunk

    from lightspeed_agentic.providers.models.openai import create_openai_compatible_model

    model = create_openai_compatible_model(
        ModelParameters(
            model="redhataigemma-4-31b-it-fp8-blo",
            base_url="https://vllm.example.test/v1",
            api_key="EMPTY",
        )
    )
    generation = model._convert_chunk_to_generation_chunk(
        {
            "id": "chatcmpl_gemma_reasoning",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": "redhataigemma-4-31b-it-fp8-blo",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "258", **reasoning_fields},
                    "finish_reason": "stop",
                }
            ],
        },
        AIMessageChunk,
        None,
    )

    assert generation is not None
    assert [
        block["reasoning"]
        for block in generation.message.content_blocks
        if block["type"] == "reasoning"
    ] == expected_reasoning
    assert generation.message.text == "258"
    assert generation.message.response_metadata["id"] == "chatcmpl_gemma_reasoning"
    assert generation.generation_info["finish_reason"] == "stop"


@pytest.mark.parametrize("base_url", [None, "", "   "])
def test_compatible_openai_model_requires_resolved_base_url(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str | None,
) -> None:
    from lightspeed_agentic.providers.models.openai import create_openai_compatible_model

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    with pytest.raises(ValueError, match="resolved base URL"):
        create_openai_compatible_model(
            ModelParameters(model="served-model", base_url=base_url, api_key="EMPTY")
        )


@pytest.mark.usefixtures("http_clients")
@pytest.mark.parametrize("compatible", [False, True])
@pytest.mark.parametrize("api_key", [None, ""])
def test_openai_model_does_not_fall_back_to_ambient_api_key(
    monkeypatch: pytest.MonkeyPatch,
    compatible: bool,
    api_key: str | None,
) -> None:
    from openai import OpenAIError

    from lightspeed_agentic.providers.models import openai as openai_models

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-must-not-be-used")
    create_model = (
        openai_models.create_openai_compatible_model
        if compatible
        else openai_models.create_native_openai_model
    )

    with pytest.raises(OpenAIError, match="Missing credentials"):
        create_model(
            ModelParameters(
                model="configured-model",
                base_url="https://vllm.example.test/v1",
                api_key=api_key,
            )
        )


@pytest.mark.parametrize("compatible", [False, True])
@pytest.mark.asyncio
async def test_openai_model_clients_are_closed_by_deepagents_cleanup(
    http_clients: tuple[httpx.Client, httpx.AsyncClient],
    compatible: bool,
) -> None:
    from lightspeed_agentic.providers.deepagents import _close_model_clients
    from lightspeed_agentic.providers.models import openai as openai_models

    create_model = (
        openai_models.create_openai_compatible_model
        if compatible
        else openai_models.create_native_openai_model
    )
    model = create_model(
        ModelParameters(
            model="configured-model", base_url="https://vllm.example.test/v1", api_key="EMPTY"
        )
    )

    await _close_model_clients(model)

    assert http_clients[0].is_closed
    assert http_clients[1].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("compatible", [False, True], ids=["native", "compatible"])
@pytest.mark.parametrize(
    "max_output_tokens", [None, CLASSIFIER_OUTPUT_TOKENS], ids=["sdk-default", "classifier-limit"]
)
@pytest.mark.parametrize(
    "captured_openai_requests",
    [{"injectionDetected": False, "category": "none"}],
    indirect=True,
)
async def test_classifier_output_budget_reaches_openai_http_request(
    captured_openai_requests: tuple[list[httpx.Request], httpx.Client, httpx.AsyncClient],
    compatible: bool,
    max_output_tokens: int | None,
) -> None:
    from lightspeed_agentic.inspection.client import LangChainClassifierClient
    from lightspeed_agentic.inspection.models import ClassifierRequest
    from lightspeed_agentic.providers.models import openai as openai_models

    create_model = (
        openai_models.create_openai_compatible_model
        if compatible
        else openai_models.create_native_openai_model
    )
    model = create_model(
        ModelParameters(
            model="configured-model",
            base_url="https://configured.example.test/v1",
            api_key="resolved-key",
            max_output_tokens=max_output_tokens,
        )
    )
    decision = await LangChainClassifierClient(model).classify(
        ClassifierRequest(
            toolName="probe", resultType="result", chunkIndex=0, chunkCount=1, content="safe"
        )
    )

    assert decision.injection_detected is False
    requests, _sync_client, _async_client = captured_openai_requests
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    token_field = "max_completion_tokens" if compatible else "max_output_tokens"
    assert body.get(token_field) == (None if max_output_tokens is None else 4096)
    assert (
        not ({"max_tokens", "max_completion_tokens", "max_output_tokens"} - {token_field})
        & body.keys()
    )


def test_openai_model_module_import_does_not_load_optional_sdk() -> None:
    project_root = os.path.dirname(os.path.dirname(__file__))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import lightspeed_agentic.providers.models.openai; "
            "assert 'langchain_openai' not in sys.modules; "
            "assert 'azure.identity' not in sys.modules; "
            "assert 'openai' not in sys.modules",
        ],
        cwd=project_root,
        env={**os.environ, "PYTHONPATH": os.path.join(project_root, "src")},
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
