"""Native SDK HTTP responses through the runner to actual exported OTLP bytes."""

from __future__ import annotations

import base64
import json
import struct
import zlib
from collections.abc import Callable
from contextlib import AsyncExitStack, ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from google.auth.credentials import AnonymousCredentials
from opentelemetry.proto.trace.v1.trace_pb2 import Status

from lightspeed_agentic import tls
from lightspeed_agentic.config import (
    AnthropicCredentialSource,
    BedrockCredentialSource,
    BedrockProviderParameters,
    ResolvedModelFamily,
    VertexCredentialSource,
    VertexProviderParameters,
)
from lightspeed_agentic.providers import cloud_credentials
from lightspeed_agentic.providers.deepagents import DeepAgentsProvider
from lightspeed_agentic.providers.model_router import resolve_model_route

from .helpers import attributes, invocation, messages, run_traced_query
from .helpers import otlp_requests as otlp_requests
from .helpers import wire_trace as wire_trace
from .test_azure_credentials import azure_identity_http as azure_identity_http
from .test_azure_deepagents import _route as azure_route
from .test_azure_models import azure_http as azure_http
from .test_openai_models import (
    _openai_chat_completion,
    _openai_response,
    _openai_response_stream,
)


def _aws_event(event: dict[str, Any]) -> bytes:
    """Encode the Bedrock HTTP event envelope, decoded by the real botocore SDK."""
    headers = b""
    for name, value in (
        (":message-type", "event"),
        (":event-type", "chunk"),
        (":content-type", "application/json"),
    ):
        key, text = name.encode(), value.encode()
        headers += bytes([len(key)]) + key + b"\x07" + struct.pack("!H", len(text)) + text
    payload = json.dumps({"bytes": base64.b64encode(json.dumps(event).encode()).decode()}).encode()
    prelude = struct.pack("!II", 16 + len(headers) + len(payload), len(headers))
    message = prelude + struct.pack("!I", zlib.crc32(prelude)) + headers + payload
    return message + struct.pack("!I", zlib.crc32(message))


def _anthropic_reply(
    request: httpx.Request, body: dict[str, Any], index: int
) -> tuple[httpx.Response, str]:
    shaped = index == 2
    tool_shape = shaped and any(tool.get("name") == "OutputModel" for tool in body.get("tools", []))
    stop = "tool_use" if tool_shape else "end_turn"
    if tool_shape:
        content = [
            {
                "type": "tool_use",
                "id": "sdk-shape-call",
                "name": "OutputModel",
                "input": {"success": True, "summary": "native answer"},
            }
        ]
    elif shaped:
        content = [{"type": "text", "text": '{"success":true,"summary":"native answer"}'}]
    else:
        content = [
            {"type": "thinking", "thinking": "native reasoning", "signature": "offline-signature"},
            {"type": "text", "text": "native answer"},
        ]
    usage = {"input_tokens": 3 if shaped else 0, "output_tokens": 2 if shaped else 4}
    message = {
        "id": f"sdk-response-{index}",
        "type": "message",
        "role": "assistant",
        "model": "served-claude-revision",
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": usage,
    }
    bedrock_stream = request.url.path.endswith("/invoke-with-response-stream")
    if not body.get("stream") and not bedrock_stream:
        return httpx.Response(200, json=message), stop
    events = [
        {
            "type": "message_start",
            "message": {
                **message,
                "content": [],
                "stop_reason": None,
                "usage": {**usage, "output_tokens": 0},
            },
        }
    ]
    for position, block in enumerate(content):
        kind = block["type"]
        start = {**block}
        if kind == "tool_use":
            start["input"] = {}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        elif kind == "thinking":
            start["thinking"] = ""
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        else:
            start["text"] = ""
            delta = {"type": "text_delta", "text": block["text"]}
        events.extend(
            [
                {"type": "content_block_start", "index": position, "content_block": start},
                {"type": "content_block_delta", "index": position, "delta": delta},
                {"type": "content_block_stop", "index": position},
            ]
        )
    events.extend(
        [
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": usage,
            },
            {"type": "message_stop"},
        ]
    )
    if bedrock_stream:
        return httpx.Response(
            200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            content=b"".join(_aws_event(event) for event in events),
        ), stop
    sse = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse), stop


def _family(route: str) -> ResolvedModelFamily:
    if route == "anthropic":
        return ResolvedModelFamily(
            name="anthropic",
            hosting="anthropic",
            model="claude-haiku-4-5",
            expected_envs=(),
            credentials=AnthropicCredentialSource(api_key="offline-key"),
        )
    if route == "bedrock/anthropic":
        return ResolvedModelFamily(
            name="anthropic",
            hosting="bedrock",
            model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
            expected_envs=(),
            provider=BedrockProviderParameters(region="us-west-2"),
            credentials=BedrockCredentialSource(
                access_key_id="offline-access",
                secret_access_key="offline-secret",  # noqa: S106 - offline SDK credential
            ),
        )
    if route.startswith("vertex/"):
        google = route.startswith("vertex/google")
        return ResolvedModelFamily(
            name="google" if google else "anthropic",
            hosting="vertex",
            model="gemini-2.5-flash" if google else "claude-sonnet-4-6@20260217",
            expected_envs=(),
            provider=VertexProviderParameters(project="test-project", region="us-central1"),
            credentials=VertexCredentialSource(Path("/unused/credentials.json")),
        )
    return ResolvedModelFamily(
        name="openai",
        hosting="openai",
        model="gpt-4.1" if route == "openai" else "requested-compatible-model",
        expected_envs=(),
        api_key="offline-key",
        base_url=None if route == "openai" else "https://compatible.example.test/v1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route_name",
    [
        "anthropic",
        "bedrock/anthropic",
        "vertex/anthropic",
        "vertex/google",
        "vertex/google-no-id",
        "openai",
        "openai-compatible",
        "azure-chat-key",
        "azure-chat-entra",
        "azure-responses-key",
        "azure-responses-entra",
    ],
)
@pytest.mark.parametrize("wire_trace", ["false", "true"], indirect=True)
async def test_native_sdk_metadata_and_shaping_are_reconstructable_from_otlp(
    wire_trace: Callable[[], list[Any]],
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    route_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Do not substitute fake AIMessage metadata for actual SDK-exposed fields."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    credentials = AnonymousCredentials()
    credentials.token = "offline-token"  # noqa: S105 - offline SDK credential
    monkeypatch.setattr(cloud_credentials, "create_vertex_credentials", lambda _source: credentials)
    http_requests: list[httpx.Request] = []
    stops: list[str] = []
    bodies: list[dict[str, Any]] = []
    schema = {
        "type": "object",
        "properties": {"success": {"type": "boolean"}, "summary": {"type": "string"}},
        "required": ["success", "summary"],
    }

    def respond(http_request: httpx.Request) -> httpx.Response:
        http_requests.append(http_request)
        body = json.loads(http_request.content)
        bodies.append(body)
        index = len(bodies)
        assert index <= 2, "Only main generation and structured shaping are scripted"
        shaped = index == 2
        host, path = http_request.url.host, http_request.url.path
        if route_name == "anthropic":
            assert (host, path) == ("api.anthropic.com", "/v1/messages")
        elif route_name == "bedrock/anthropic":
            assert host == "bedrock-runtime.us-west-2.amazonaws.com"
            assert path.endswith("/invoke" if shaped else "/invoke-with-response-stream")
        elif route_name.startswith("vertex/"):
            assert host == "us-central1-aiplatform.googleapis.com"
            assert "/projects/test-project/locations/us-central1/publishers/" in path
            assert path.endswith(
                (":rawPredict" if shaped else ":streamRawPredict")
                if "anthropic" in route_name
                else (":generateContent" if shaped else ":streamGenerateContent")
            )
        elif route_name == "openai":
            assert (host, path) == ("api.openai.com", "/v1/responses")
        elif route_name == "openai-compatible":
            assert (host, path) == ("compatible.example.test", "/v1/chat/completions")
        else:
            assert host == "resolved-resource.example.test"
            assert path.startswith("/proxy/openai/")
            assert path.endswith(
                "/responses"
                if "responses" in route_name
                else "/deployments/operator-deployment/chat/completions"
            )
            assert http_request.url.params["api-version"] == (
                "2025-03-01-preview" if "responses" in route_name else "2024-08-01-preview"
            )
            if route_name.endswith("entra"):
                assert http_request.headers["authorization"].startswith("Bearer entra-token-")
            else:
                assert http_request.headers["api-key"] == "resolved-azure-key"
        if "anthropic" in route_name:
            response, stop = _anthropic_reply(http_request, body, index)
            stops.append(stop)
            return response
        if route_name.startswith("vertex/google"):
            parts = (
                [{"text": '{"success":true,"summary":"native answer"}'}]
                if shaped
                else [{"text": "native reasoning", "thought": True}, {"text": "native answer"}]
            )
            payload = {
                "candidates": [
                    {"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}
                ],
                "usageMetadata": {
                    "promptTokenCount": 3 if shaped else 0,
                    "candidatesTokenCount": 2 if shaped else 4,
                    "totalTokenCount": 5 if shaped else 4,
                },
                "modelVersion": "served-gemini-revision",
            }
            if route_name != "vertex/google-no-id":
                payload["responseId"] = f"sdk-response-{index}"
            stops.append("STOP")
            if "streamGenerateContent" in http_request.url.path:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=f"data: {json.dumps(payload)}\n\n",
                )
            return httpx.Response(200, json=payload)
        text = '{"success":true,"summary":"native answer"}' if shaped else "native answer"
        if route_name == "openai" or "responses" in route_name:
            item = {
                "id": f"sdk-message-{index}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
            stops.append("completed")
            if body.get("stream"):
                return _openai_response_stream(
                    f"sdk-response-{index}",
                    item,
                    input_tokens=3 if shaped else 0,
                    output_tokens=2 if shaped else 4,
                    reasoning_text=None if shaped else "native reasoning",
                )
            return httpx.Response(
                200,
                json=_openai_response(
                    f"sdk-response-{index}",
                    item,
                    input_tokens=3 if shaped else 0,
                    output_tokens=2 if shaped else 4,
                ),
            )
        stops.append("stop")
        return _openai_chat_completion(
            f"sdk-response-{index}",
            {
                "role": "assistant",
                "content": text,
                **({"reasoning_content": "native reasoning"} if not shaped else {}),
            },
            stream=bool(body.get("stream")),
            input_tokens=3 if shaped else 0,
            output_tokens=2 if shaped else 4,
        )

    transport = httpx.MockTransport(respond)
    google_models: list[Any] = []
    sdk_results: list[Any] = []
    from langchain_core.callbacks import AsyncCallbackHandler

    class SourceObserver(AsyncCallbackHandler):
        async def on_llm_end(self, response: Any, **_kwargs: Any) -> None:
            sdk_results.append(response)

    with ExitStack() as sync_clients:
        async with AsyncExitStack() as async_clients:

            def sync_client(**kwargs: Any) -> httpx.Client:
                return sync_clients.enter_context(httpx.Client(transport=transport, **kwargs))

            def async_client(**kwargs: Any) -> httpx.AsyncClient:
                client = httpx.AsyncClient(transport=transport, **kwargs)
                async_clients.push_async_callback(client.aclose)
                return client

            if route_name.startswith("azure-"):
                azure = request.getfixturevalue("azure_http")
                azure.model_handler = respond
                route = azure_route(
                    monkeypatch,
                    tmp_path,
                    "2025-03-01-preview" if "responses" in route_name else "2024-08-01-preview",
                    "entra" if route_name.endswith("entra") else "api-key",
                )
            else:
                monkeypatch.setattr(tls, "create_http_client", sync_client)
                monkeypatch.setattr(tls, "create_async_http_client", async_client)
                if route_name.startswith("vertex/google"):
                    from google.genai import _api_client

                    monkeypatch.setattr(_api_client, "AsyncHttpxClient", async_client)
                route = resolve_model_route(model_family=_family(route_name))
            real_create = route.create_model

            def create_with_offline_transport(parameters: Any, reasoning_config: Any = None) -> Any:
                model = real_create(parameters, reasoning_config)
                model.callbacks = [SourceObserver()]
                if route.route == "vertex/anthropic":
                    model.async_client.access_token = "offline-token"  # noqa: S105
                    model.client.access_token = "offline-token"  # noqa: S105
                elif route.route == "vertex/google":
                    api = model.client._api_client
                    client = api._async_httpx_client
                    api._http_options.httpx_async_client = client
                    google_models.append(model)
                return model

            provider = DeepAgentsProvider(
                replace(route, create_model=create_with_offline_transport)
            )
            try:
                result = await run_traced_query(
                    provider, tmp_path, schema=schema, model=route.model
                )
            finally:
                await provider.aclose()
                for model in google_models:
                    model.client.close()
                    await model.client.aio.aclose()
    assert result.output == {"success": True, "summary": "native answer"}
    audit_records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"schema_version":1,')
    ]
    if request.node.callspec.params["wire_trace"] == "true":
        assert len(audit_records) == 4
        assert [row["event"] for row in audit_records] == [
            "outbound.request.started",
            "outbound.request.finished",
            "outbound.request.started",
            "outbound.request.finished",
        ]
        assert all(row["request.kind"] == "llm" for row in audit_records)
        assert len({row["attempt_id"] for row in audit_records}) == 2
        assert all(
            row["agenticrun.uid"] == "11111111-1111-4111-8111-111111111111"
            and row["agenticrun.phase"] == "analysis"
            for row in audit_records
        )
        assert all(row["http.request.method"] == "POST" for row in audit_records)
        assert all(
            row["server.address"] == http_requests[index // 2].url.host
            for index, row in enumerate(audit_records)
        )
        assert all(row["outcome"] == "success" for row in audit_records[1::2])
        assert "native answer" not in json.dumps(audit_records)
        assert "offline-token" not in json.dumps(audit_records)
    else:
        assert audit_records == []
    assert len(bodies) == len(sdk_results) == 2
    spans = wire_trace()
    root = invocation(spans)
    assert len(spans) == 3
    generations = sorted(
        (s for s in spans if s.name == f"chat {route.model}"), key=lambda s: s.start_time_unix_nano
    )
    assert len(generations) == 2
    observed_model = (
        "served-claude-revision"
        if "anthropic" in route_name
        else "served-gemini-revision"
        if route_name.startswith("vertex/google")
        else "gpt-4.1-mini"
        if route_name == "openai" or "responses" in route_name
        else "served-model-revision-7"
    )
    for index, span in enumerate(generations, start=1):
        attrs = attributes(span)
        sdk_result = sdk_results[index - 1]
        sdk_generation = sdk_result.generations[0][0]
        source_metadata = [
            sdk_generation.message.response_metadata,
            sdk_generation.generation_info or {},
            sdk_result.llm_output or {},
        ]
        assert span.status.code == Status.STATUS_CODE_UNSET
        assert attrs["gen_ai.request.model"] == route.model
        assert attrs["gen_ai.provider.name"] == route.telemetry_provider
        assert attrs["gen_ai.response.model"] == observed_model
        assert any(
            source.get(key) == observed_model
            for source in source_metadata
            for key in ("model", "model_name")
        )
        if route_name == "openai" or "responses" in route_name:
            # Responses status=completed is not a provider finish reason.
            assert "gen_ai.response.finish_reasons" not in attrs
            assert any(source.get("status") == "completed" for source in source_metadata)
            assert all(
                source.get(key) is None
                for source in source_metadata
                for key in ("stop_reason", "finish_reason")
            )
        else:
            assert attrs["gen_ai.response.finish_reasons"] == [stops[index - 1]]
        assert attrs["gen_ai.usage.input_tokens"] == (0 if index == 1 else 3)
        assert attrs["gen_ai.usage.output_tokens"] == (4 if index == 1 else 2)
        assert attrs["agenticrun.uid"] == "11111111-1111-4111-8111-111111111111"
        assert attrs["agenticrun.phase"] == "analysis"
        if route_name.startswith("vertex/google") or ("anthropic" in route_name and index == 1):
            # These SDK wrappers do not expose the HTTP response ID in the
            # callback metadata. Do not substitute framework-generated IDs.
            assert "gen_ai.response.id" not in attrs
            assert all(
                source.get(key) is None
                for source in source_metadata
                for key in ("id", "response_id")
            )
        else:
            assert attrs["gen_ai.response.id"] == f"sdk-response-{index}"
            assert any(
                source.get(key) == f"sdk-response-{index}"
                for source in source_metadata
                for key in ("id", "response_id")
            )
    parts = messages(generations[0])[0]["parts"]
    assert parts[-1] == {"type": "text", "content": "native answer"}
    assert parts[0] == {"type": "reasoning", "content": "native reasoning"}, (
        sdk_results[0].generations[0][0].message.content,
        sdk_results[0].generations[0][0].message.content_blocks,
        sdk_results[0].generations[0][0].message.response_metadata,
    )
    assert len(parts) == 2
    root_attrs = attributes(root)
    assert root.status.code == Status.STATUS_CODE_UNSET
    assert (root_attrs["gen_ai.usage.input_tokens"], root_attrs["gen_ai.usage.output_tokens"]) == (
        3,
        6,
    )
    assert generations[0].end_time_unix_nano <= generations[1].start_time_unix_nano
