"""Real SDK retries/classification and the native Gemini aiohttp backend."""

from __future__ import annotations

import json
from pathlib import Path
from time import monotonic
from typing import Any

import httpx
import pytest

from lightspeed_agentic import tls
from lightspeed_agentic.config import ModelParameters
from lightspeed_agentic.inspection.chunking import Utf8ByteCodec
from lightspeed_agentic.inspection.client import LangChainClassifierClient
from lightspeed_agentic.inspection.inspector import inspect_tool_result
from lightspeed_agentic.providers.deepagents import _close_model_clients
from lightspeed_agentic.providers.models.openai import create_openai_compatible_model
from tests.test_openai_models import _openai_chat_completion

pytestmark = pytest.mark.usefixtures("connection_audit_enabled")


def audit_rows(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"schema_version":1,')
    ]


@pytest.mark.asyncio
async def test_sdk_retries_and_classifier_retries_have_distinct_identities(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    def network(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["authorization"] == "Bearer PRIVATE-key"
        if calls == 1:
            return httpx.Response(503, json={"error": {"message": "PRIVATE-error"}})
        return _openai_chat_completion(
            f"response-{calls}",
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "classifier-native-call",
                        "type": "function",
                        "function": {
                            "name": "ClassifierDecision",
                            "arguments": '{"injectionDetected":false,"category":"none"}'
                            if calls == 3
                            else "PRIVATE-invalid",
                        },
                    }
                ],
            },
            stream=False,
            input_tokens=3,
            output_tokens=2,
        )

    async def no_sleep(_delay: float) -> None:
        pass

    transport = httpx.MockTransport(network)
    async with httpx.AsyncClient(transport=transport) as client:
        monkeypatch.setattr(tls, "create_async_http_client", lambda: client)
        model = create_openai_compatible_model(
            ModelParameters(
                model="served-model",
                base_url="https://model.example/v1",
                api_key="PRIVATE-key",
            )
        )
        model.root_async_client._calculate_retry_timeout = lambda *_a, **_k: 0
        try:
            result = await inspect_tool_result(
                LangChainClassifierClient(model),
                tool_name="probe",
                tool_call_id="native-source-call",
                result_type="result",
                value="PRIVATE-result",
                codec=Utf8ByteCodec(),
                context_window_tokens=100_000,
                instruction_tokens=512,
                deadline=monotonic() + 30,
                sleep=no_sleep,
            )
        finally:
            await _close_model_clients(model)
    assert result.passed is True
    observed = audit_rows(capsys)
    assert len(observed) == 6
    starts, ends = observed[::2], observed[1::2]
    assert calls == 3
    assert len({r["attempt_id"] for r in starts}) == 3
    assert [r["outcome"] for r in ends] == ["http_error", "success", "success"]
    assert [r["http.response.status_code"] for r in ends] == [503, 200, 200]
    assert all(r["request.kind"] == "llm" for r in observed)
    assert all("gen_ai.tool.name" not in r for r in observed)
    assert "PRIVATE" not in json.dumps(observed)


@pytest.mark.asyncio
async def test_native_subagent_calls_are_audited_without_payloads(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from lightspeed_agentic.config import ResolvedModelFamily
    from lightspeed_agentic.providers.deepagents import DeepAgentsProvider
    from lightspeed_agentic.providers.model_router import resolve_model_route
    from lightspeed_agentic.run_agent import run_agent_query

    calls = 0

    def network(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        message = {
            "role": "assistant",
            "content": "PRIVATE-subagent" if calls == 2 else "PRIVATE-answer",
        }
        if calls == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "delegate-call",
                        "type": "function",
                        "function": {
                            "name": "task",
                            "arguments": json.dumps(
                                {
                                    "description": "Return PRIVATE-subagent",
                                    "subagent_type": "general-purpose",
                                }
                            ),
                        },
                    }
                ],
            }
        assert calls <= 3
        return _openai_chat_completion(
            f"response-{calls}",
            message,
            stream=bool(body.get("stream")),
            input_tokens=3,
            output_tokens=2,
        )

    transport = httpx.MockTransport(network)
    with httpx.Client(transport=transport) as sync_client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            monkeypatch.setattr(tls, "create_http_client", lambda: sync_client)
            monkeypatch.setattr(tls, "create_async_http_client", lambda: async_client)
            route = resolve_model_route(
                model_family=ResolvedModelFamily(
                    name="openai",
                    hosting="openai",
                    model="served-model",
                    expected_envs=(),
                    base_url="https://model.example/v1",
                    api_key="PRIVATE-key",
                )
            )
            monkeypatch.setenv("HOME", str(tmp_path))
            provider = DeepAgentsProvider(route)
            result = await run_agent_query(
                provider,
                prompt="Delegate the task and report the answer.",
                system_prompt="Report observed results.",
                skills_dir=str(tmp_path),
                output_schema=None,
                context=None,
                model="served-model",
                max_turns=20,
                timeout_seconds=10,
                tool_output_inspection_enabled=False,
                audit_enabled=False,
                capture_content=False,
                agenticrun_uid="11111111-1111-4111-8111-111111111111",
                step="verification",
            )
    assert result.output == {"success": True, "summary": "PRIVATE-answer"}
    observed = audit_rows(capsys)
    assert len(observed) == 6
    assert all(
        row["agenticrun.uid"] == "11111111-1111-4111-8111-111111111111"
        and row["agenticrun.phase"] == "verification"
        for row in observed
    )
    assert len({row["attempt_id"] for row in observed}) == 3
    assert all(row["request.kind"] == "llm" for row in observed)
    assert all(row["outcome"] == "success" for row in observed[1::2])
    assert "PRIVATE" not in json.dumps(observed)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["complete", "read-error", "cancel", "early-close", "cross-context-close"]
)
async def test_native_openai_stream_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    import asyncio

    from langchain_openai.chat_models.base import OpenAITimeoutError

    from lightspeed_agentic.providers.models.openai import create_native_openai_model
    from tests.test_openai_models import _openai_response_stream

    source = _openai_response_stream(
        "response-stream",
        {
            "id": "message-stream",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "PRIVATE-output", "annotations": []}],
        },
        input_tokens=3,
        output_tokens=2,
    )
    failure = (
        asyncio.CancelledError("PRIVATE") if mode == "cancel" else httpx.ReadTimeout("PRIVATE")
    )

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            payload = source.content
            if mode == "complete":
                yield payload
            else:
                yield payload[: payload.index(b"event: response.completed")]
                raise failure

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=Stream()))
    ) as client:
        monkeypatch.setattr(tls, "create_async_http_client", lambda: client)
        model = create_native_openai_model(
            ModelParameters(model="gpt-4.1-mini", api_key="PRIVATE-key")
        )
        assert audit_rows(capsys) == []  # Constructors are not logical LLM calls.
        stream = model.astream("PRIVATE-prompt")
        try:
            if mode in {"early-close", "cross-context-close"}:
                async for chunk in stream:
                    if chunk.content:
                        break
                if mode == "cross-context-close":
                    from lightspeed_agentic import connection_audit as audit

                    assert (
                        audit.start_attempt(audit.Destination("https", "outside.example", 443))
                        is None
                    )
                    await asyncio.create_task(stream.aclose())
                else:
                    await stream.aclose()
            elif mode == "complete":
                assert "PRIVATE-output" in "".join([chunk.text async for chunk in stream])
            else:
                expected = asyncio.CancelledError if mode == "cancel" else OpenAITimeoutError
                with pytest.raises(expected) as raised:
                    async for _ in stream:
                        pass
                if mode == "cancel":
                    assert raised.value is failure
                else:
                    assert raised.value.__cause__.__cause__ is failure
        finally:
            await stream.aclose()
            await _close_model_clients(model)
    observed = audit_rows(capsys)
    assert len(observed) == 2
    end = observed[1]
    assert end["outcome"] == (
        "success" if mode == "complete" else "cancelled" if mode == "cancel" else "transport_error"
    )
    if mode != "complete":
        assert end["error.type"] == (
            "cancelled"
            if mode == "cancel"
            else "timeout"
            if mode == "read-error"
            else "protocol_error"
        )
    assert "PRIVATE" not in json.dumps(observed)


@pytest.mark.asyncio
@pytest.mark.parametrize("authorized", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_google_native_aiohttp_and_recreated_sessions_exclude_token_requests(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    authorized: bool,
    streaming: bool,
) -> None:
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from google.auth.aio.credentials import Credentials
    from google.auth.aio.transport.sessions import AsyncAuthorizedSession
    from google.auth.credentials import AnonymousCredentials

    from lightspeed_agentic.config import VertexCredentialSource, VertexProviderParameters
    from lightspeed_agentic.providers import cloud_credentials
    from lightspeed_agentic.providers.models.gemini import create_vertex_gemini_model

    urls = []

    async def network(request):
        urls.append(request.path)
        if request.path == "/token":
            return web.json_response({"access_token": "PRIVATE-token"})
        assert "PRIVATE" in await request.text()
        reply = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "PRIVATE-answer"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 2,
                "totalTokenCount": 5,
            },
        }
        if streaming:
            assert ":streamGenerateContent" in request.path
            return web.Response(
                text=f"data: {json.dumps(reply)}\n\n", content_type="text/event-stream"
            )
        return web.json_response(reply)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", network)
    credentials = AnonymousCredentials()
    credentials.token = "PRIVATE-token"  # noqa: S105 - offline credential sentinel
    monkeypatch.setattr(cloud_credentials, "create_vertex_credentials", lambda _source: credentials)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    async with TestServer(app) as server:

        class AsyncCredentials(Credentials):
            async def refresh(self, request):
                await request(
                    str(server.make_url("/token")), method="POST", body=b"PRIVATE-credential"
                )
                self.token = "PRIVATE-token"  # noqa: S105 - offline credential sentinel

            async def before_request(self, request, method, url, headers):
                if self.token is None:
                    await self.refresh(request)
                await super().before_request(request, method, url, headers)

        model = create_vertex_gemini_model(
            ModelParameters(
                model="gemini-2.5-flash",
                provider=VertexProviderParameters(project="test-project", region="us-central1"),
                credentials=VertexCredentialSource(tmp_path / "unused"),
            )
        )
        api = model.client._api_client
        # Change only the external server authority, retaining native SDK execution.
        api._http_options.base_url = str(server.make_url("/"))
        assert api._use_aiohttp() is True
        auth_sessions = []
        try:
            for _ in range(2):
                if authorized:
                    session = AsyncAuthorizedSession(AsyncCredentials())
                    auth_sessions.append(session)
                    api._http_options.aiohttp_client = session
                if streaming:
                    assert (
                        "".join([chunk.text async for chunk in model.astream("PRIVATE-prompt")])
                        == "PRIVATE-answer"
                    )
                else:
                    assert (await model.ainvoke("PRIVATE-prompt")).content == "PRIVATE-answer"
                current = await api._get_aiohttp_session()
                await current.close()  # The next native call must recreate/observe its session.
        finally:
            for session in auth_sessions:
                await session.close()
            model.client.close()
            await model.client.aio.aclose()
    observed = audit_rows(capsys)
    assert len(observed) == 4
    assert all(r["request.kind"] == "llm" for r in observed)
    assert all(r["http.request.method"] == "POST" for r in observed)
    assert all(r["server.address"] == "127.0.0.1" for r in observed)
    assert all(r["outcome"] == "success" for r in observed[1::2])
    assert len([url for url in urls if url != "/token"]) == 2
    assert urls.count("/token") == (2 if authorized else 0)
    assert "PRIVATE" not in json.dumps(observed)
