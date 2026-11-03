"""Exercise actual HTTP client routing and body ownership with scripted I/O."""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import pytest_asyncio

from lightspeed_agentic import connection_audit as audit

pytestmark = pytest.mark.usefixtures("connection_audit_enabled")


def scope():
    return audit.llm_call(model_key=1, logical_call_id="logical")


def rows(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("audit_setting", "record_count"),
    [("true", 2), (" TrUe ", 2), ("false", 0), (None, 0), ("", 0), ("1", 0)],
)
async def test_audit_gate_preserves_requests_without_otlp(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    asynchronous: bool,
    audit_setting: str | None,
    record_count: int,
) -> None:
    from lightspeed_agentic.providers.audit_support import instrument_httpx_client

    if audit_setting is None:
        monkeypatch.delenv("LIGHTSPEED_AUDIT_ENABLED", raising=False)
    else:
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", audit_setting)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("LIGHTSPEED_CAPTURE_CONTENT", "true")
    requests = []

    def network(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"response")

    with scope():
        if asynchronous:
            async with httpx.AsyncClient(transport=httpx.MockTransport(network)) as client:
                instrument_httpx_client(client)
                response = await client.post("https://model.example", content=b"PRIVATE-request")
        else:
            with httpx.Client(transport=httpx.MockTransport(network)) as sync_client:
                instrument_httpx_client(sync_client)
                response = sync_client.post(  # noqa: ASYNC212 - scripted I/O
                    "https://model.example", content=b"PRIVATE-request"
                )
    assert len(requests) == 1
    assert requests[0].content == b"PRIVATE-request"
    assert response.status_code == 200
    assert response.content == b"response"
    observed = rows(capsys)
    assert len(observed) == record_count
    if record_count:
        assert [row["event"] for row in observed] == [
            "outbound.request.started",
            "outbound.request.finished",
        ]
        assert "PRIVATE-request" not in json.dumps(observed)


class Body(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(
        self, error: BaseException | None = None, close_error: BaseException | None = None
    ):
        self.error = error
        self.close_error = close_error
        self.closes = 0

    def __iter__(self) -> Iterator[bytes]:
        yield b"first"
        if self.error:
            raise self.error
        yield b"second"

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self:
            yield chunk

    def close(self) -> None:
        self.closes += 1
        if self.close_error:
            raise self.close_error

    async def aclose(self) -> None:
        self.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_transport_context_ownership_is_preserved(
    capsys: pytest.CaptureFixture[str], asynchronous: bool
) -> None:
    from lightspeed_agentic.providers.audit_support import instrument_httpx_client

    class Transport(httpx.MockTransport):
        entries = 0
        exits = 0

        def __enter__(self):
            self.entries += 1
            return super().__enter__()

        def __exit__(self, *args):
            self.exits += 1
            return super().__exit__(*args)

        async def __aenter__(self):
            self.entries += 1
            return await super().__aenter__()

        async def __aexit__(self, *args):
            self.exits += 1
            return await super().__aexit__(*args)

    def network(_request):
        assert transport.entries == 1
        return httpx.Response(200, content=b"ok")

    transport = Transport(network)
    with scope():
        if asynchronous:
            client = httpx.AsyncClient(transport=transport)
            instrument_httpx_client(client)
            async with client:
                assert (await client.get("https://model.example")).content == b"ok"
        else:
            sync_client = httpx.Client(transport=transport)
            instrument_httpx_client(sync_client)
            with sync_client:
                assert sync_client.get("https://model.example").content == b"ok"  # noqa: ASYNC212 - scripted I/O
    assert transport.exits == 1
    assert len(rows(capsys)) == 2


@pytest.mark.parametrize("status", [200, 401, 429, 503])
def test_http_attempt_pairs_for_success_and_http_errors(
    capsys: pytest.CaptureFixture[str], status: int
) -> None:
    from lightspeed_agentic.providers.audit_support import instrument_httpx_client

    def network(request: httpx.Request) -> httpx.Response:
        assert request.content == b"PRIVATE-request"
        assert len(rows(capsys)) == 1  # Start exists before any external I/O.
        return httpx.Response(status, stream=Body())

    with httpx.Client(transport=httpx.MockTransport(network)) as client, scope():
        instrument_httpx_client(client)
        instrument_httpx_client(client)
        response = client.post(
            "https://user:PRIVATE@model.example/path?key=PRIVATE", content=b"PRIVATE-request"
        )
        assert response.content == b"firstsecond"
    (end,) = rows(capsys)
    assert end["event"] == "outbound.request.finished"
    assert end["outcome"] == ("success" if status == 200 else "http_error")
    assert "request.body.size" not in end
    assert end["http.request.method"] == "POST"
    assert end["server.address"] == "model.example"
    assert "PRIVATE" not in json.dumps(end)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["connect-error", "early-response"])
async def test_buffered_length_does_not_prove_transmission(capsys, asynchronous, mode):
    from lightspeed_agentic.providers.audit_support import instrument_httpx_client

    error = httpx.ConnectError("PRIVATE connection failure")

    class Wire(httpx.BaseTransport, httpx.AsyncBaseTransport):
        def handle_request(self, request):
            assert request.content == b"PRIVATE-buffered"
            if mode == "connect-error":
                raise error
            # Return an early rejection without consuming the outbound stream.
            return httpx.Response(413, content=b"rejected")

        async def handle_async_request(self, request):
            return self.handle_request(request)

    async def send():
        if asynchronous:
            async with httpx.AsyncClient(transport=Wire()) as client:
                instrument_httpx_client(client)
                return await client.post("https://model.example", content=b"PRIVATE-buffered")
        with httpx.Client(transport=Wire()) as client:
            instrument_httpx_client(client)
            return client.post("https://model.example", content=b"PRIVATE-buffered")

    with scope():
        if mode == "connect-error":
            with pytest.raises(httpx.ConnectError) as raised:
                await send()
            assert raised.value is error
        else:
            response = await send()
            assert response.status_code == 413
            assert response.content == b"rejected"
    start, end = rows(capsys)
    assert start["attempt_id"] == end["attempt_id"]
    assert end["outcome"] == ("transport_error" if mode == "connect-error" else "http_error")
    assert all("request.body.size" not in record for record in (start, end))
    assert "PRIVATE" not in json.dumps([start, end])


@pytest.mark.asyncio
async def test_headers_are_not_stream_completion(capsys: pytest.CaptureFixture[str]) -> None:
    from lightspeed_agentic.providers.audit_support import instrument_httpx_client

    body = Body()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(200, stream=body))
    ) as client:
        instrument_httpx_client(client)
        with scope():
            async with client.stream("POST", "https://model.example", content=b"x") as response:
                (start,) = rows(capsys)
                assert start["event"] == "outbound.request.started"
                assert b"".join([part async for part in response.aiter_bytes()]) == b"firstsecond"
                (end,) = rows(capsys)
                assert end["outcome"] == "success"
            assert rows(capsys) == []
    assert body.closes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["error", "cancel", "close", "client-close"])
async def test_stream_failure_cancellation_and_early_close(
    capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    from lightspeed_agentic.providers.audit_support import instrument_httpx_client

    error = asyncio.CancelledError("PRIVATE") if mode == "cancel" else RuntimeError("PRIVATE")
    body = Body(error if mode in {"error", "cancel"} else None)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(200, stream=body))
    ) as client:
        instrument_httpx_client(client)
        with scope():
            response = await client.send(
                client.build_request("POST", "https://model.example"), stream=True
            )
            if mode in {"error", "cancel"}:
                with pytest.raises(type(error)) as raised:
                    await response.aread()
                assert raised.value is error
            if mode != "client-close":
                await response.aclose()
                await response.aclose()
    start, end = rows(capsys)
    assert end["attempt_id"] == start["attempt_id"]
    assert end["outcome"] == ("cancelled" if mode == "cancel" else "transport_error")
    assert end["error.type"] == ("cancelled" if mode == "cancel" else "protocol_error")
    assert "PRIVATE" not in json.dumps(end)


def test_sync_close_failure_is_preserved(capsys: pytest.CaptureFixture[str]) -> None:
    from lightspeed_agentic.providers.audit_support import instrument_httpx_client

    error = RuntimeError("PRIVATE cleanup failure")
    body = Body(close_error=error)
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _req: httpx.Response(200, stream=body))
        ) as client,
        scope(),
    ):
        instrument_httpx_client(client)
        response = client.send(client.build_request("POST", "https://model.example"), stream=True)
        with pytest.raises(RuntimeError, match="PRIVATE cleanup failure") as raised:
            response.close()
        assert raised.value is error
    assert len(rows(capsys)) == 2


@pytest.mark.asyncio
async def test_cleanup_failure_cannot_replace_request_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from lightspeed_agentic.providers.audit_support import instrument_httpx_client

    error = asyncio.CancelledError("primary")
    body = Body(error, RuntimeError("cleanup"))
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(200, stream=body))
    ) as client:
        instrument_httpx_client(client)
        with scope(), pytest.raises(asyncio.CancelledError) as raised:
            async with client.stream("POST", "https://model.example") as response:
                await response.aread()
    assert raised.value is error
    assert len(rows(capsys)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (socket.gaierror("PRIVATE"), "dns_error"),
        (httpx.ConnectError("PRIVATE"), "connect_error"),
        (ssl.SSLCertVerificationError("PRIVATE"), "tls_error"),
        (ssl.SSLError("PRIVATE"), "tls_error"),
        (httpx.ReadTimeout("PRIVATE"), "timeout"),
        (asyncio.CancelledError("PRIVATE"), "cancelled"),
    ],
)
async def test_transport_failure_categories(
    capsys: pytest.CaptureFixture[str], error: BaseException, reason: str
) -> None:
    from lightspeed_agentic.providers.audit_support import instrument_httpx_client

    def network(_request: httpx.Request) -> httpx.Response:
        raise error

    async with httpx.AsyncClient(transport=httpx.MockTransport(network)) as client:
        instrument_httpx_client(client)
        with scope(), pytest.raises(type(error)) as raised:
            await client.post("https://model.example")
    assert raised.value is error
    start, end = rows(capsys)
    assert end["error.type"] == reason
    assert "decision" not in end
    assert "deny_reason" not in end
    assert "PRIVATE" not in json.dumps([start, end])


@pytest.mark.parametrize("causality", ["explicit", "implicit", "suppressed"])
@pytest.mark.parametrize("kind", ["timeout", "connect"])
def test_tls_error_requires_a_causal_verification_error(
    capsys: pytest.CaptureFixture[str], causality: str, kind: str
) -> None:
    from lightspeed_agentic.providers.audit_support import instrument_httpx_client

    error = httpx.ReadTimeout("PRIVATE") if kind == "timeout" else httpx.ConnectError("PRIVATE")

    def network(_request: httpx.Request) -> httpx.Response:
        try:
            raise ssl.SSLCertVerificationError("PRIVATE previous failure")
        except ssl.SSLCertVerificationError as certificate:
            if causality == "explicit":
                raise error from certificate
            if causality == "suppressed":
                raise error from None
            raise error  # noqa: B904 - deliberately exercise implicit handling context

    with httpx.Client(transport=httpx.MockTransport(network)) as client:
        instrument_httpx_client(client)
        with scope(), pytest.raises(type(error)) as raised:
            client.post("https://model.example")
    assert raised.value is error
    start, end = rows(capsys)
    denied = causality == "explicit"
    assert end["error.type"] == (
        "tls_error" if denied else "connect_error" if kind == "connect" else "timeout"
    )
    assert "decision" not in end
    assert "deny_reason" not in end
    assert "PRIVATE" not in json.dumps([start, end])


@pytest.mark.parametrize(
    ("url", "host", "port"),
    [
        ("https://192.0.2.1:8443", "192.0.2.1", 8443),
        ("https://[2001:db8::1]:8443", "2001:db8::1", 8443),
    ],
)
def test_destination_preserves_ip_literals_and_explicit_ports(
    capsys: pytest.CaptureFixture[str],
    url: str,
    host: str,
    port: int,
) -> None:
    from lightspeed_agentic.providers.audit_support import instrument_httpx_client

    client = httpx.Client(
        transport=httpx.MockTransport(lambda _req: httpx.Response(200, content=b"ok"))
    )
    instrument_httpx_client(client)
    with client, scope():
        client.post(url)
    observed = rows(capsys)
    assert len(observed) == 2
    assert all(row["server.address"] == host and row["server.port"] == port for row in observed)


def test_body_size_does_not_read_a_request_stream(capsys: pytest.CaptureFixture[str]) -> None:
    from lightspeed_agentic.providers.audit_support import instrument_httpx_client

    class RequestBody(httpx.SyncByteStream):
        def __iter__(self):
            assert "request.body.size" not in rows(capsys)[0]
            yield b"secret"

    class Wire(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            assert b"".join(request.stream) == b"secret"
            return httpx.Response(200, stream=Body())

    with httpx.Client(transport=Wire()) as client, scope():
        instrument_httpx_client(client)
        client.post("https://model.example", content=RequestBody())
    (end,) = rows(capsys)
    assert end["outcome"] == "success"
    assert "request.body.size" not in end


def test_effective_destination_for_proxy_bypass_and_reuse(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpcore

    from lightspeed_agentic.providers.audit_support import instrument_httpx_client

    monkeypatch.setenv("HTTPS_PROXY", "http://user:PRIVATE@proxy.example:8080")
    monkeypatch.setenv("NO_PROXY", "bypass.example")

    routes = []

    def wire(_pool: Any, _request: Any) -> Any:
        routes.append(type(_pool).__name__)
        return httpcore.Response(200, content=b"ok")

    monkeypatch.setattr(httpcore.ConnectionPool, "handle_request", wire)
    monkeypatch.setattr(httpcore.HTTPProxy, "handle_request", wire)
    with httpx.Client() as client, scope():
        instrument_httpx_client(client)
        for host in ["model.example", "bypass.example", "model.example"]:
            assert client.post(f"https://{host}").content == b"ok"
    observed = rows(capsys)
    assert len(observed) == 6
    assert [r["server.address"] for r in observed[::2]] == [
        "model.example",
        "bypass.example",
        "model.example",
    ]
    assert observed[0]["server.port"] == 443
    assert routes == ["HTTPProxy", "ConnectionPool", "HTTPProxy"]
    assert all("network.peer.address" not in row for row in observed)
    assert "PRIVATE" not in json.dumps(observed)


@pytest.mark.asyncio
async def test_aiohttp_preserves_requested_total_timeout(
    aiohttp_server, capsys: pytest.CaptureFixture[str]
) -> None:
    import aiohttp

    from lightspeed_agentic.providers.audit_support import instrument_aiohttp_session

    url, _gate = aiohttp_server
    async with aiohttp.ClientSession() as session:
        instrument_aiohttp_session(session)
        with scope(), pytest.raises(TimeoutError):
            async with session.post(
                url + "stream", timeout=aiohttp.ClientTimeout(total=0.02)
            ) as response:
                await response.read()
    start, end = rows(capsys)
    assert start["http.request.method"] == "POST"
    assert end["outcome"] == "transport_error"
    assert end["error.type"] == "timeout"


@pytest_asyncio.fixture
async def aiohttp_server():
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    gate = asyncio.Event()

    async def normal(_request):
        return web.Response(body=b"firstsecond")

    async def redirect(request):
        return web.Response(
            status=int(request.query.get("status", "307")),
            headers={"Location": "/normal"},
            body=b"",
        )

    async def streaming(request):
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"first\n")
        await gate.wait()
        return response

    app = web.Application()
    app.router.add_route("*", "/normal", normal)
    app.router.add_route("*", "/redirect", redirect)
    app.router.add_route("*", "/stream", streaming)
    async with TestServer(app) as server:
        try:
            yield str(server.make_url("/")), gate
        finally:
            gate.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [303, 307])
async def test_aiohttp_redirects_pair_actual_attempts(
    aiohttp_server, capsys: pytest.CaptureFixture[str], status: int
) -> None:
    import aiohttp

    from lightspeed_agentic.providers.audit_support import instrument_aiohttp_session

    url, _gate = aiohttp_server
    async with aiohttp.ClientSession() as session:
        with scope():
            instrument_aiohttp_session(session)
            instrument_aiohttp_session(session)
            async with session.post(url + f"redirect?status={status}", data=b"PRIVATE") as response:
                assert isinstance(response, aiohttp.ClientResponse)
                assert await response.read() == b"firstsecond"
    observed = rows(capsys)
    starts = [r for r in observed if r["event"] == "outbound.request.started"]
    ends = [r for r in observed if r["event"] == "outbound.request.finished"]
    assert len(starts) == len(ends) == 2
    assert {r["attempt_id"] for r in starts} == {r["attempt_id"] for r in ends}
    assert len({r["attempt_id"] for r in starts}) == 2
    assert [r["http.request.method"] for r in starts] == [
        "POST",
        "GET" if status == 303 else "POST",
    ]
    assert all("request.body.size" not in record for record in observed)
    assert ends[-1]["outcome"] == "success"
    assert "PRIVATE" not in json.dumps(observed)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["read", "iterate", "release", "close", "generator-close", "cancel", "session-close"]
)
async def test_aiohttp_stream_consumption_and_closure(
    aiohttp_server, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    import aiohttp

    from lightspeed_agentic.providers.audit_support import instrument_aiohttp_session

    url, gate = aiohttp_server
    async with aiohttp.ClientSession() as session:
        instrument_aiohttp_session(session)
        with scope():
            response = await session.get(url + "stream")
            (start,) = rows(capsys)
            assert start["event"] == "outbound.request.started"
            if mode in {"read", "iterate"}:
                gate.set()
                if mode == "read":
                    assert await response.read() == b"first\n"
                else:
                    assert [line async for line in response.content] == [b"first\n"]
            elif mode == "generator-close":

                async def stream():
                    try:
                        yield await response.content.readline()
                    finally:
                        response.release()

                iterator = stream()
                assert await anext(iterator) == b"first\n"
                await iterator.aclose()
            elif mode == "cancel":
                task = asyncio.create_task(response.read())
                await asyncio.sleep(0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif mode == "release":
                response.release()
            elif mode == "close":
                response.close()
    (end,) = rows(capsys)
    assert end["attempt_id"] == start["attempt_id"]
    assert end["outcome"] == (
        "success"
        if mode in {"read", "iterate"}
        else "cancelled"
        if mode == "cancel"
        else "transport_error"
    )
    assert "request.body.size" not in end
    if mode == "generator-close":
        assert end["error.type"] == "protocol_error"
    response.close()
    assert rows(capsys) == []


@pytest.mark.asyncio
async def test_aiohttp_start_precedes_dns_or_connection_failure(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import aiohttp

    from lightspeed_agentic.providers.audit_support import instrument_aiohttp_session

    error = socket.gaierror("PRIVATE DNS failure")

    async def connect(*_args, **_kwargs):
        (start,) = rows(capsys)
        assert start["server.address"] == "model.example"
        raise error

    async with aiohttp.ClientSession() as session:
        with scope():
            instrument_aiohttp_session(session)
            monkeypatch.setattr(session.connector, "connect", connect)
            with pytest.raises(aiohttp.ClientOSError) as raised:
                await session.post("https://model.example", data=b"PRIVATE-request")
            assert raised.value.__cause__ is error
    (end,) = rows(capsys)
    assert end["error.type"] == "dns_error"
    assert "request.body.size" not in end
    assert "decision" not in end
