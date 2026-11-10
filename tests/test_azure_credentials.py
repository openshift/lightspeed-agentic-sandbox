"""Offline Azure credential helpers with real SDK acquisition and refresh."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.helpers import otlp_requests as otlp_requests


@pytest.fixture
def azure_identity_http(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Script only the HTTP boundary; keep Azure's pipeline/cache and sessions real."""
    import aiohttp
    from aiohttp.helpers import TimerNoop
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL

    from lightspeed_agentic import tls

    state = SimpleNamespace(clock=2_000_000_000, responses=[], requests=[], sessions=[])
    monkeypatch.setattr(time, "time", lambda: state.clock)
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    monkeypatch.setattr(tls, "_runtime_ssl_context", context)
    state.ssl_context = context
    monkeypatch.delenv("AZURE_AUTHORITY_HOST", raising=False)
    for name in (
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "AZURE_CLIENT_ID",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_SECRET",
    ):
        monkeypatch.setenv(name, "ambient-must-not-win")

    def reject_network(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Unscripted network access in Azure credential test")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)
    original_init = aiohttp.ClientSession.__init__

    def track_session(session: Any, *args: Any, **kwargs: Any) -> None:
        original_init(session, *args, **kwargs)
        state.sessions.append(session)

    async def request(_session: Any, method: str, url: str, **kwargs: Any) -> Any:
        state.requests.append((method, str(url), kwargs))
        assert state.responses, "Unexpected token HTTP request"
        scripted = state.responses.pop(0)
        await asyncio.sleep(0)  # Exercise concurrent callbacks through the SDK lock.
        if isinstance(scripted, BaseException):
            raise scripted
        status, payload = scripted
        response = aiohttp.ClientResponse(
            method,
            URL(url),
            writer=None,
            continue100=None,
            timer=TimerNoop(),
            request_info=None,
            traces=[],
            loop=asyncio.get_running_loop(),
            session=_session,
            stream_writer=SimpleNamespace(output_size=0),
        )
        response.status = status
        response.reason = "OK" if status == 200 else "Unauthorized"
        response._headers = CIMultiDictProxy(CIMultiDict({"Content-Type": "application/json"}))
        response._body = json.dumps(payload).encode()
        return response

    monkeypatch.setattr(aiohttp.ClientSession, "__init__", track_session)
    monkeypatch.setattr(aiohttp.ClientSession, "_request", request)
    return state


@pytest.mark.asyncio
async def test_azure_api_key_helper_uses_only_resolved_key(azure_identity_http: Any) -> None:
    from lightspeed_agentic.config import AzureApiKeyCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

    environment = dict(os.environ)
    auth = create_azure_credentials(AzureApiKeyCredentialSource("resolved-azure-key"))
    assert auth.api_key == "resolved-azure-key"
    assert auth.token_provider is None
    assert auth.credential is None
    assert "resolved-azure-key" not in repr(auth)
    await auth.aclose()
    assert dict(os.environ) == environment
    assert azure_identity_http.requests == []
    assert azure_identity_http.sessions == []


def test_azure_entra_helper_constructs_lazily_without_event_loop(azure_identity_http: Any) -> None:
    from azure.identity.aio import ClientSecretCredential

    from lightspeed_agentic.config import AzureEntraCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

    auth = create_azure_credentials(
        AzureEntraCredentialSource("resolved-tenant", "resolved-client", "resolved-secret")
    )
    assert isinstance(auth.credential, ClientSecretCredential)
    assert auth.token_provider is not None
    assert auth.api_key == "<missing API key>"
    assert all(
        value not in repr(auth)
        for value in ("resolved-tenant", "resolved-client", "resolved-secret")
    )
    assert azure_identity_http.requests == []
    assert azure_identity_http.sessions == []
    asyncio.run(auth.aclose())
    asyncio.run(auth.aclose())
    assert azure_identity_http.sessions == []


@pytest.mark.asyncio
async def test_azure_native_token_callback_refreshes_expired_token(
    azure_identity_http: Any,
) -> None:
    from lightspeed_agentic.config import AzureEntraCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

    http = azure_identity_http
    environment = dict(os.environ)
    http.responses.extend(
        [
            (
                200,
                {"access_token": "identity-token-one", "expires_in": 3600, "token_type": "Bearer"},
            ),
            (
                200,
                {"access_token": "identity-token-two", "expires_in": 3600, "token_type": "Bearer"},
            ),
        ]
    )
    auth = create_azure_credentials(
        AzureEntraCredentialSource("resolved-tenant", "resolved-client", "resolved-secret")
    )
    assert auth.token_provider is not None
    try:
        assert (
            await asyncio.gather(*(auth.token_provider() for _ in range(4)))
            == ["identity-token-one"] * 4
        )
        assert len(http.requests) == 1
        http.clock += 3601
        assert await auth.token_provider() == "identity-token-two"
        assert await auth.token_provider() == "identity-token-two"
        assert len(http.requests) == 2
        for method, url, kwargs in http.requests:
            assert method == "POST"
            assert url == "https://login.microsoftonline.com/resolved-tenant/oauth2/v2.0/token"
            assert kwargs["data"]["client_id"] == "resolved-client"
            assert kwargs["data"]["client_secret"] == "resolved-secret"  # noqa: S105 - offline fixture
            assert kwargs["data"]["grant_type"] == "client_credentials"
            assert kwargs["data"]["scope"] == "https://cognitiveservices.azure.com/.default"
            assert kwargs["ssl"] is http.ssl_context
            assert kwargs["ssl"].check_hostname
            assert kwargs["ssl"].verify_mode == ssl.CERT_REQUIRED
            assert "api-key" not in kwargs["headers"]
            assert "Authorization" not in kwargs["headers"]
        assert len(http.sessions) == 1
        assert not http.sessions[0].closed
    finally:
        await auth.aclose()
    assert http.sessions[0].closed
    assert dict(os.environ) == environment
    await auth.aclose()


@pytest.mark.parametrize("status", [400, 401, 403])
@pytest.mark.parametrize(
    "after_expiry", [False, True], ids=["initial-acquisition", "expired-refresh"]
)
@pytest.mark.asyncio
async def test_azure_token_rejection_propagates_without_api_key_fallback(
    azure_identity_http: Any, status: int, after_expiry: bool
) -> None:
    from azure.core.exceptions import ClientAuthenticationError

    from lightspeed_agentic.config import AzureEntraCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

    http = azure_identity_http
    if after_expiry:
        http.responses.append(
            (
                200,
                {"access_token": "identity-token-one", "expires_in": 3600, "token_type": "Bearer"},
            )
        )
    http.responses.append(
        (status, {"error": "invalid_client", "error_description": "offline credential rejected"})
    )
    auth = create_azure_credentials(
        AzureEntraCredentialSource("resolved-tenant", "resolved-client", "resolved-secret")
    )
    assert auth.token_provider is not None
    try:
        if after_expiry:
            assert await auth.token_provider() == "identity-token-one"
            http.clock += 3601
        with pytest.raises(ClientAuthenticationError, match="offline credential rejected") as exc:
            await auth.token_provider()
        assert all(
            value not in str(exc.value) for value in ("resolved-secret", "ambient-must-not-win")
        )
        assert auth.api_key == "<missing API key>"
        assert len(http.requests) == (2 if after_expiry else 1)
        assert http.responses == []
    finally:
        await auth.aclose()
    assert len(http.sessions) == 1
    assert http.sessions[0].closed


@pytest.mark.asyncio
async def test_azure_credential_can_close_after_cancelled_acquisition(
    azure_identity_http: Any,
) -> None:
    from lightspeed_agentic.config import AzureEntraCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

    http = azure_identity_http
    http.responses.append(asyncio.CancelledError())
    auth = create_azure_credentials(
        AzureEntraCredentialSource("resolved-tenant", "resolved-client", "resolved-secret")
    )
    assert auth.token_provider is not None
    try:
        with pytest.raises(asyncio.CancelledError):
            await auth.token_provider()
    finally:
        await auth.aclose()
    assert len(http.requests) == 1
    assert len(http.sessions) == 1
    assert http.sessions[0].closed


def test_azure_invalid_tenant_fails_without_allocating_session(azure_identity_http: Any) -> None:
    from lightspeed_agentic.config import AzureEntraCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

    with pytest.raises(ValueError, match="tenant") as exc:
        create_azure_credentials(
            AzureEntraCredentialSource("invalid/tenant", "resolved-client", "resolved-secret")
        )
    assert "resolved-secret" not in str(exc.value)
    assert azure_identity_http.requests == []
    assert azure_identity_http.sessions == []


def test_azure_api_key_helper_does_not_import_optional_sdks() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = r"""
import asyncio
import builtins

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if any(name == package or name.startswith(package + ".")
           for package in ("azure", "aiohttp", "openai", "google", "botocore")):
        raise AssertionError(f"Optional SDK import attempted: {name}")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from lightspeed_agentic.config import AzureApiKeyCredentialSource
from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

auth = create_azure_credentials(AzureApiKeyCredentialSource("resolved-azure-key"))
assert auth.api_key == "resolved-azure-key"
assert auth.credential is None
assert auth.token_provider is None
asyncio.run(auth.aclose())
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(  # noqa: S603 - controlled inline import-isolation test
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def azure_audit_wire(monkeypatch, capsys):
    """Keep native session middleware, pipeline/cache and body consumption real."""
    monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
    import aiohttp
    from aiohttp.base_protocol import BaseProtocol
    from aiohttp.helpers import TimerNoop
    from aiohttp.streams import StreamReader
    from multidict import CIMultiDict, CIMultiDictProxy

    state = SimpleNamespace(responses=[], requests=[], events=[], sessions=[])

    def capture():
        state.events.extend(
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if line.startswith('{"schema_version":1,')
        )
        return state.events

    original_init = aiohttp.ClientSession.__init__

    def session_init(session, *args, **kwargs):
        original_init(session, *args, **kwargs)
        state.sessions.append(session)

    async def connect(_connector, request, **_kwargs):
        events = capture()
        assert events
        assert events[-1]["event"] == "outbound.request.started"
        state.requests.append(request)
        scripted = state.responses.pop(0)
        if isinstance(scripted, BaseException):
            raise scripted
        protocol = SimpleNamespace(set_response_params=lambda **_kwargs: None)
        return SimpleNamespace(protocol=protocol, close=lambda: None, response=scripted)

    async def send(request, conn):
        status, payload = conn.response
        loop = asyncio.get_running_loop()
        response = aiohttp.ClientResponse(
            request.method,
            request.url,
            writer=None,
            continue100=None,
            timer=TimerNoop(),
            request_info=request.request_info,
            traces=[],
            loop=loop,
            session=request._session,
            stream_writer=SimpleNamespace(output_size=0),
        )
        response.status = status
        response.reason = "OK"
        response._headers = CIMultiDictProxy(CIMultiDict({"Content-Type": "application/json"}))
        response.content = StreamReader(BaseProtocol(loop), limit=2**16, loop=loop)
        response.content.feed_data(json.dumps(payload).encode())
        response.content.feed_eof()

        async def start(_conn):
            return response

        response.start = start
        return response

    def reject(*_args, **_kwargs):
        pytest.fail("Unscripted Azure auth network access")

    monkeypatch.setattr(aiohttp.ClientSession, "__init__", session_init)
    monkeypatch.setattr(aiohttp.TCPConnector, "connect", connect)
    monkeypatch.setattr(aiohttp.ClientRequest, "send", send)
    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket, "getaddrinfo", reject)
    state.capture = capture
    return state


@pytest.mark.asyncio
async def test_azure_native_auth_audit_cache_concurrency_refresh(azure_audit_wire, monkeypatch):
    from lightspeed_agentic.config import AzureEntraCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

    wire = azure_audit_wire
    clock = [2_000_000_000]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    wire.responses.extend(
        [
            (
                200,
                {"access_token": f"PRIVATE_TOKEN_{i}", "expires_in": 3600, "token_type": "Bearer"},
            )
            for i in range(2)
        ]
    )
    auth = create_azure_credentials(
        AzureEntraCredentialSource("PRIVATE-TENANT", "PRIVATE_CLIENT", "PRIVATE_SECRET")
    )
    assert wire.requests == []
    assert wire.sessions == []
    try:
        assert (
            await asyncio.gather(*(auth.token_provider() for _ in range(4)))
            == ["PRIVATE_TOKEN_0"] * 4
        )
        assert len(wire.requests) == 1
        clock[0] += 3601
        assert await auth.token_provider() == "PRIVATE_TOKEN_1"
        assert await auth.token_provider() == "PRIVATE_TOKEN_1"
        assert len(wire.requests) == 2
    finally:
        await auth.aclose()
    events = wire.capture()
    assert [row["event"] for row in events] == [
        "outbound.request.started",
        "outbound.request.finished",
    ] * 2
    assert {row["request.kind"] for row in events} == {"credential"}
    assert len({row["attempt_id"] for row in events}) == 2
    assert all(row["outcome"] == "success" for row in events[1::2])
    assert all(row["http.request.method"] == "POST" for row in events)
    assert all(session.closed for session in wire.sessions)
    assert "PRIVATE" not in json.dumps(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["2024-08-01-preview", "2025-03-01-preview"])
async def test_native_azure_model_auth_json_otlp_parity(
    azure_audit_wire, monkeypatch, otlp_requests, version
):
    import httpx
    from opentelemetry import trace
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest

    from lightspeed_agentic import tls, tracing
    from lightspeed_agentic.config import (
        AzureEntraCredentialSource,
        AzureProviderParameters,
        ModelParameters,
    )
    from lightspeed_agentic.providers.deepagents import _close_model_clients
    from lightspeed_agentic.providers.models.openai import create_azure_model
    from tests.test_openai_models import _openai_chat_completion, _openai_response

    wire = azure_audit_wire
    wire.responses.append((200, {"access_token": "PRIVATE_TOKEN", "expires_in": 3600}))
    monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
    monkeypatch.setenv("LIGHTSPEED_CAPTURE_CONTENT", "false")
    monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_STEP", "analysis")

    def model_http(request):
        assert request.headers["authorization"] == "Bearer PRIVATE_TOKEN"
        if "responses" in request.url.path:
            return httpx.Response(
                200,
                json=_openai_response(
                    "response-1",
                    {
                        "id": "message-1",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "answer", "annotations": []}],
                    },
                    input_tokens=3,
                    output_tokens=2,
                ),
            )
        return _openai_chat_completion("response-1", {"role": "assistant", "content": "answer"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(model_http)) as client:
        monkeypatch.setattr(tls, "create_async_http_client", lambda: client)
        tracing.init_tracer()
        model = None
        try:
            with trace.get_tracer("auth-sdk-test").start_as_current_span("invoke_agent") as span:
                span_id = span.get_span_context().span_id
                model = create_azure_model(
                    ModelParameters(
                        model="deployment",
                        base_url="https://model.example",
                        provider=AzureProviderParameters(version),
                        credentials=AzureEntraCredentialSource(
                            "PRIVATE-TENANT", "PRIVATE_CLIENT", "PRIVATE_SECRET"
                        ),
                    )
                )
                for _ in range(2):
                    assert (await model.ainvoke("PRIVATE prompt")).text == "answer"
        finally:
            if model is not None:
                await _close_model_clients(model)
            tracing.shutdown_tracer()
    local = wire.capture()
    auth = [row for row in local if row["request.kind"] == "credential"]
    llm = [row for row in local if row["request.kind"] == "llm"]
    assert len(auth) == 2
    assert len(llm) == 4
    assert auth[0]["server.address"] == "login.microsoftonline.com"
    assert all(row["server.address"] == "model.example" for row in llm)
    assert all(
        row["agenticrun.uid"] == "11111111-1111-4111-8111-111111111111"
        and row["span_id"] == f"{span_id:016x}"
        for row in local
    )
    remote = []
    while not otlp_requests.logs.empty():
        request = ExportLogsServiceRequest.FromString(otlp_requests.logs.get_nowait())
        remote.extend(
            record
            for resource in request.resource_logs
            for scope in resource.scope_logs
            for record in scope.log_records
            if record.body.string_value.startswith('{"schema_version":1')
        )
    assert [json.loads(record.body.string_value) for record in remote] == local
    assert all(int.from_bytes(record.span_id, "big") == span_id for record in remote)
    assert all(session.closed for session in wire.sessions)
    assert "PRIVATE" not in json.dumps(local)


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["rejected", "retry", "cancel", "timeout", "tls"])
async def test_azure_auth_audit_failures_and_native_retry(azure_audit_wire, monkeypatch, result):
    import aiohttp

    from lightspeed_agentic.config import AzureEntraCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

    wire = azure_audit_wire
    if result == "rejected":
        wire.responses.append((403, {"error": "invalid_client", "error_description": "PRIVATE"}))
    elif result == "retry":
        wire.responses.extend(
            [
                (503, {"error": "temporarily_unavailable"}),
                (200, {"access_token": "PRIVATE_TOKEN", "expires_in": 3600}),
            ]
        )
    elif result == "cancel":
        wire.responses.append(asyncio.CancelledError("PRIVATE"))
    elif result == "timeout":
        wire.responses.extend([TimeoutError("PRIVATE") for _ in range(4)])
    else:
        wire.responses.extend(
            [
                aiohttp.ClientConnectorCertificateError(
                    SimpleNamespace(host="PRIVATE", port=443, ssl=True, is_ssl=True),
                    ssl.SSLCertVerificationError("PRIVATE"),
                )
                for _ in range(4)
            ]
        )
    # Keep the retry policy real; skip only its native backoff wait.
    from azure.core.exceptions import (
        ClientAuthenticationError,
        ServiceRequestError,
        ServiceResponseError,
    )
    from azure.core.pipeline.transport import AioHttpTransport

    async def no_sleep(self, duration):
        pass

    monkeypatch.setattr(AioHttpTransport, "sleep", no_sleep)
    auth = create_azure_credentials(
        AzureEntraCredentialSource("PRIVATE-TENANT", "PRIVATE_CLIENT", "PRIVATE_SECRET")
    )
    try:
        if result == "retry":
            assert await auth.token_provider() == "PRIVATE_TOKEN"
        else:
            expected = (
                asyncio.CancelledError
                if result == "cancel"
                else ClientAuthenticationError
                if result == "rejected"
                else (ServiceRequestError, ServiceResponseError)
            )
            with pytest.raises(expected):
                await auth.token_provider()
    finally:
        await auth.aclose()
    events = wire.capture()
    starts = [row for row in events if row["event"] == "outbound.request.started"]
    ends = [row for row in events if row["event"] == "outbound.request.finished"]
    assert len(starts) == len(ends) == len(wire.requests)
    assert len(starts) >= 1
    assert len({row["attempt_id"] for row in starts}) == len(starts)
    assert {row["attempt_id"] for row in starts} == {row["attempt_id"] for row in ends}
    assert (
        ends[0]["error.type"]
        == {
            "rejected": "http_error",
            "retry": "http_error",
            "cancel": "cancelled",
            "timeout": "timeout",
            "tls": "tls_error",
        }[result]
    )
    assert "decision" not in ends[0]
    if result == "retry":
        assert len(starts) == 2
        assert ends[-1]["outcome"] == "success"
    assert all(session.closed for session in wire.sessions)
    assert "PRIVATE" not in json.dumps(events)
