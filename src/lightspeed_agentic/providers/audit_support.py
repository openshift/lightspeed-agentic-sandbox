"""Native model scopes and HTTP transport observers for connection auditing."""

from __future__ import annotations

import asyncio
import socket
import ssl
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

import httpx

from lightspeed_agentic.connection_audit import (
    Destination,
    RequestAttempt,
    llm_call_scope,
    start_attempt,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


# HTTPX transports and byte streams.


def destination_from_url(url: Any) -> Destination:
    """Project only an authority from HTTPX/httpcore or aiohttp native URLs."""

    def text(value: str | bytes) -> str:
        return value.decode("ascii") if isinstance(value, bytes) else value

    scheme = text(url.scheme)
    port = url.port if url.port is not None else (443 if scheme == "https" else 80)
    return Destination(scheme, text(url.host), port)


def _begin(request: httpx.Request, transport: Any) -> RequestAttempt | None:
    # Inspect the pool selected by HTTPX's existing proxy/mount/bypass routing.
    proxy_url = getattr(getattr(transport, "_pool", None), "_proxy_url", None)
    target = destination_from_url(request.url)
    return start_attempt(
        destination_from_url(proxy_url) if proxy_url is not None else target,
        target=target if proxy_url is not None else None,
        method=request.method,
    )


class _BodyState:
    def __init__(self, attempt: RequestAttempt, active: set[RequestAttempt]) -> None:
        self.attempt = attempt
        self.active = active
        self.failed = False
        self.closed = False

    def finish(self, complete: bool, error: BaseException | None = None) -> None:
        self.failed = self.failed or error is not None
        self.active.discard(self.attempt)
        self.attempt.finish(response_complete=complete, error=error)


class _SyncBody(httpx.SyncByteStream, _BodyState):
    def __init__(
        self, stream: httpx.SyncByteStream, attempt: RequestAttempt, active: set[RequestAttempt]
    ) -> None:
        _BodyState.__init__(self, attempt, active)
        self.stream = stream

    def __iter__(self) -> Iterator[bytes]:
        try:
            yield from self.stream
        except GeneratorExit:
            self.finish(False)
            raise
        except BaseException as error:
            self.finish(False, error)
            raise
        else:
            self.finish(True)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        primary = sys.exc_info()[1]
        failed = self.failed
        try:
            self.stream.close()
        except BaseException as error:
            self.finish(False, error)
            if primary is None and not failed:
                raise
        finally:
            self.finish(False)


class _AsyncBody(httpx.AsyncByteStream, _BodyState):
    def __init__(
        self, stream: httpx.AsyncByteStream, attempt: RequestAttempt, active: set[RequestAttempt]
    ) -> None:
        _BodyState.__init__(self, attempt, active)
        self.stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self.stream:
                yield chunk
        except GeneratorExit:
            self.finish(False)
            raise
        except BaseException as error:
            self.finish(False, error)
            raise
        else:
            self.finish(True)

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        primary = sys.exc_info()[1]
        failed = self.failed
        try:
            await self.stream.aclose()
        except BaseException as error:
            self.finish(False, error)
            if primary is None and not failed:
                raise
        finally:
            self.finish(False)


def _headers(
    response: httpx.Response, attempt: RequestAttempt, active: set[RequestAttempt]
) -> bool:
    attempt.response_headers(response.status_code)
    if response.is_stream_consumed:
        # Custom/mock transports may return an already buffered response.
        attempt.finish(response_complete=True)
        active.discard(attempt)
        return False
    return True


@contextmanager
def _closing(active: set[RequestAttempt], primary: BaseException | None) -> Iterator[None]:
    error = primary
    try:
        yield
    except BaseException as cleanup:
        error = primary or cleanup
        if primary is None:
            raise
    finally:
        for attempt in tuple(active):
            attempt.finish(response_complete=False, error=error)
        active.clear()


class _SyncTransport(httpx.BaseTransport):
    def __init__(self, transport: httpx.BaseTransport) -> None:
        self.transport = transport
        self.active: set[RequestAttempt] = set()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        attempt = _begin(request, self.transport)
        if attempt is None:
            return self.transport.handle_request(request)
        self.active.add(attempt)
        try:
            response = self.transport.handle_request(request)
        except BaseException as error:
            self.active.discard(attempt)
            attempt.finish(response_complete=False, error=error)
            raise
        if _headers(response, attempt, self.active):
            response.stream = _SyncBody(
                cast(httpx.SyncByteStream, response.stream), attempt, self.active
            )
        return response

    def __enter__(self) -> _SyncTransport:
        self.transport.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        with _closing(self.active, exc_value):
            self.transport.__exit__(exc_type, exc_value, traceback)

    def close(self) -> None:
        with _closing(self.active, sys.exc_info()[1]):
            self.transport.close()


class _AsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self.transport = transport
        self.active: set[RequestAttempt] = set()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        attempt = _begin(request, self.transport)
        if attempt is None:
            return await self.transport.handle_async_request(request)
        self.active.add(attempt)
        try:
            response = await self.transport.handle_async_request(request)
        except BaseException as error:
            self.active.discard(attempt)
            attempt.finish(response_complete=False, error=error)
            raise
        if _headers(response, attempt, self.active):
            response.stream = _AsyncBody(
                cast(httpx.AsyncByteStream, response.stream), attempt, self.active
            )
        return response

    async def __aenter__(self) -> _AsyncTransport:
        await self.transport.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        with _closing(self.active, exc_value):
            await self.transport.__aexit__(exc_type, exc_value, traceback)

    async def aclose(self) -> None:
        with _closing(self.active, sys.exc_info()[1]):
            await self.transport.aclose()


def instrument_httpx_client(client: httpx.Client | httpx.AsyncClient) -> None:
    """Wrap this client's selected transports once, retaining native routing/ownership.

    HTTPX has no public transport decorator installer. Keep the two private
    client fields and the httpcore proxy authority observation isolated here.
    """
    wrappers: dict[int, Any] = {}

    def wrap(transport: Any) -> Any:
        if transport is None or isinstance(transport, (_SyncTransport, _AsyncTransport)):
            return transport
        if id(transport) not in wrappers:
            wrappers[id(transport)] = (
                _AsyncTransport(transport)
                if isinstance(client, httpx.AsyncClient)
                else _SyncTransport(transport)
            )
        return wrappers[id(transport)]

    client._transport = wrap(client._transport)
    client._mounts = {pattern: wrap(transport) for pattern, transport in client._mounts.items()}


# Native aiohttp middleware and body observation.


class _ObservedContent:
    """Delegate StreamReader API and observe when the consumer reaches EOF."""

    def __init__(self, content: Any, attempt: RequestAttempt, active: set[RequestAttempt]) -> None:
        self.content = content
        self.attempt = attempt
        self.active = active
        self.error: BaseException | None = None

    def finish(self, *, error: BaseException | None = None) -> None:
        self.error = self.error or error
        self.active.discard(self.attempt)
        self.attempt.finish(
            response_complete=self.content.at_eof() and self.error is None, error=self.error
        )

    def observed(self, data: Any) -> Any:
        if self.content.at_eof():
            self.finish()
        return data

    def __getattr__(self, name: str) -> Any:
        delegate = getattr(self.content, name)
        if name not in {"read", "readany", "readline", "readuntil", "readchunk", "readexactly"}:
            return delegate

        async def read(*args: Any, **kwargs: Any) -> Any:
            try:
                return self.observed(await delegate(*args, **kwargs))
            except BaseException as error:
                self.finish(error=error)
                raise

        return read

    def read_nowait(self, *args: Any, **kwargs: Any) -> bytes:
        try:
            return cast(bytes, self.observed(self.content.read_nowait(*args, **kwargs)))
        except BaseException as error:
            self.finish(error=error)
            raise

    def __aiter__(self) -> AsyncIterator[bytes]:
        from aiohttp.streams import AsyncStreamIterator

        return AsyncStreamIterator(self.readline)

    def iter_chunked(self, size: int) -> AsyncIterator[bytes]:
        from aiohttp.streams import AsyncStreamIterator

        return AsyncStreamIterator(lambda: self.read(size))

    def iter_any(self) -> AsyncIterator[bytes]:
        from aiohttp.streams import AsyncStreamIterator

        return AsyncStreamIterator(self.readany)

    def iter_chunks(self) -> AsyncIterator[tuple[bytes, bool]]:
        from aiohttp.streams import ChunkTupleAsyncStreamIterator

        return ChunkTupleAsyncStreamIterator(cast(Any, self))


def _track_response(response: Any, attempt: RequestAttempt, active: set[RequestAttempt]) -> None:
    attempt.response_headers(response.status)
    content = _ObservedContent(response.content, attempt, active)
    response.content = content

    def wrap(delegate: Callable[..., Any]) -> Callable[..., Any]:
        def close(*args: Any, **kwargs: Any) -> Any:
            primary = sys.exc_info()[1] or content.error
            try:
                return delegate(*args, **kwargs)
            except BaseException as error:
                content.finish(error=primary or error)
                if primary is None:
                    raise
                return None
            finally:
                content.finish(error=primary)

        return close

    response.close = wrap(response.close)
    response.release = wrap(response.release)
    if content.at_eof():
        content.finish()


def instrument_aiohttp_session(session: Any) -> None:
    """Instrument an owned session once; preserve its class, pool and auth.

    Middleware runs for *each* native redirect/persistent-connection retry and
    before connector/DNS/TLS work. Request-start/header-sent callbacks alone
    cannot meet those boundaries. Google authorized sessions also receive this
    middleware through model-only request kwargs, never token request kwargs.
    """
    if "_lightspeed_audit_middleware" in vars(session):
        return
    active: set[RequestAttempt] = set()

    async def middleware(request: Any, handler: Any) -> Any:
        target = destination_from_url(request.url)
        proxy = request.proxy
        attempt = start_attempt(
            destination_from_url(proxy) if proxy else target,
            target=target if proxy else None,
            method=request.method,
        )
        if attempt is None:
            return await handler(request)
        active.add(attempt)
        try:
            response = await handler(request)
        except BaseException as error:
            active.discard(attempt)
            attempt.finish(response_complete=False, error=error)
            raise
        _track_response(response, attempt, active)
        return response

    session._lightspeed_audit_middleware = middleware
    # Native sessions use their middleware stack. AsyncAuthorizedSession accepts
    # per-model-request kwargs instead; its credential transport is not modified.
    if hasattr(session, "_middlewares"):
        session._middlewares = (*session._middlewares, middleware)
    original_close = session.close

    async def close() -> None:
        primary = sys.exc_info()[1]
        error = primary
        try:
            await original_close()
        except BaseException as cleanup:
            error = primary or cleanup
            if primary is None:
                raise
        finally:
            for attempt in tuple(active):
                attempt.finish(response_complete=False, error=error)
            active.clear()

    session.close = close


# Credential-owned urllib3 pools and responses.


_attempt: ContextVar[RequestAttempt | None] = ContextVar("auth_urllib3_attempt", default=None)
_install_lock = RLock()


def _audit_error(error: BaseException) -> BaseException:
    """Recognize native typed causes, never messages or implicit handling context."""
    from urllib3 import exceptions

    if isinstance(error, asyncio.CancelledError):
        return error
    pending = [error]
    seen: set[int] = set()
    causes: list[BaseException] = []
    while pending and len(causes) < 16:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        causes.append(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if isinstance(current, exceptions.SSLError):
            pending.extend(arg for arg in current.args if isinstance(arg, BaseException))
        if isinstance(current, exceptions.MaxRetryError) and isinstance(
            current.reason, BaseException
        ):
            pending.append(current.reason)
        if isinstance(current, exceptions.ProxyError):
            pending.append(current.original_error)
    for kind, normalized in (
        (ssl.SSLCertVerificationError, None),
        (socket.gaierror, None),
        ((ssl.SSLError, exceptions.SSLError), ssl.SSLError),
        ((ConnectionError, exceptions.NewConnectionError), ConnectionError),
        ((TimeoutError, exceptions.TimeoutError), TimeoutError),
    ):
        for cause in causes:
            if isinstance(cause, kind):
                return cause if normalized is None else normalized()
    return error


def _observe_body(response: Any, attempt: RequestAttempt) -> None:
    """Retain the native response type and observe when its body is exhausted."""
    if isinstance(response._body, bytes):
        attempt.finish(response_complete=True)
        return
    original_read = response.read
    original_close = response.close
    original_stream = response.stream
    streaming = False

    def read(*args: Any, **kwargs: Any) -> Any:
        try:
            data = original_read(*args, **kwargs)
        except BaseException as error:
            attempt.finish(response_complete=False, error=_audit_error(error))
            raise
        amt = kwargs.get("amt", args[0] if args else None)
        if not streaming and (
            amt is None
            or (response.length_remaining == 0 and response._decoder is None)
            or (amt != 0 and not data)
        ):
            attempt.finish(response_complete=True)
        return data

    def stream(*args: Any, **kwargs: Any) -> Any:
        nonlocal streaming
        iterator = iter(original_stream(*args, **kwargs))
        try:
            while True:
                streaming = True
                try:
                    data = next(iterator)
                except StopIteration:
                    attempt.finish(response_complete=True)
                    return
                finally:
                    streaming = False
                yield data
        except BaseException as error:
            attempt.finish(response_complete=False, error=_audit_error(error))
            raise
        finally:
            iterator.close()

    def close() -> Any:
        try:
            return original_close()
        except BaseException as error:
            attempt.finish(response_complete=False, error=_audit_error(error))
            raise
        finally:
            attempt.finish(response_complete=False)

    response.read = read
    response.stream = stream
    response.close = close


def _instrument_pool(pool: Any) -> None:
    if getattr(pool, "_lightspeed_auth_audit", False):
        return
    original_urlopen = pool.urlopen
    original_make_request = pool._make_request

    def urlopen(method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        from urllib3.util import parse_url

        parsed = parse_url(url)
        target = (
            destination_from_url(parsed)
            if parsed.scheme and parsed.host
            else Destination(
                pool.scheme,
                pool.host,
                pool.port if pool.port is not None else (443 if pool.scheme == "https" else 80),
            )
        )
        proxy = pool.proxy
        attempt = start_attempt(
            destination_from_url(proxy) if proxy else target,
            target=target if proxy else None,
            method=method,
        )
        token = _attempt.set(attempt)
        try:
            return original_urlopen(method, url, *args, **kwargs)
        except BaseException as error:
            if attempt is not None:
                attempt.finish(response_complete=False, error=_audit_error(error))
            raise
        finally:
            _attempt.reset(token)

    def make_request(conn: Any, *args: Any, **kwargs: Any) -> Any:
        attempt = _attempt.get()
        try:
            response = original_make_request(conn, *args, **kwargs)
        except BaseException as error:
            if attempt is not None:
                attempt.finish(response_complete=False, error=_audit_error(error))
            raise
        if attempt is not None:
            attempt.response_headers(response.status)
            _observe_body(response, attempt)
        return response

    if hasattr(pool, "_prepare_proxy"):
        original_prepare = pool._prepare_proxy

        def prepare_proxy(*args: Any, **kwargs: Any) -> Any:
            try:
                return original_prepare(*args, **kwargs)
            except BaseException as error:
                attempt = _attempt.get()
                if attempt is not None:
                    attempt.finish(response_complete=False, error=_audit_error(error))
                raise

        pool._prepare_proxy = prepare_proxy
    pool.urlopen = urlopen
    pool._make_request = make_request
    pool._lightspeed_auth_audit = True


def instrument_urllib3_manager(manager: Any) -> None:
    """Observe existing manager-selected pools, including native retries/redirects.

    Private pool hooks are isolated here: urlopen starts before proxy CONNECT;
    _make_request sees failures before urllib3 swallows them for a retry. Reads
    retain their native timing, response class and pool-release behavior.
    """
    with _install_lock:
        if getattr(manager, "_lightspeed_auth_audit", False):
            return
        original = manager.connection_from_host

        def connection_from_host(*args: Any, **kwargs: Any) -> Any:
            pool = original(*args, **kwargs)
            with _install_lock:
                _instrument_pool(pool)
            return pool

        manager.connection_from_host = connection_from_host
        manager._lightspeed_auth_audit = True


def instrument_google_request(request: Any) -> None:
    """Decorate only the native requests session supplied to credential refresh."""
    session = getattr(request, "session", None)
    if session is None or getattr(session, "_lightspeed_auth_audit", False):
        return
    original = session.get_adapter

    def get_adapter(*args: Any, **kwargs: Any) -> Any:
        adapter = original(*args, **kwargs)
        with _install_lock:
            if not getattr(adapter, "_lightspeed_auth_audit", False):
                instrument_urllib3_manager(adapter.poolmanager)
                original_proxy = adapter.proxy_manager_for

                def proxy_manager_for(*args: Any, **kwargs: Any) -> Any:
                    manager = original_proxy(*args, **kwargs)
                    instrument_urllib3_manager(manager)
                    return manager

                adapter.proxy_manager_for = proxy_manager_for
                adapter._lightspeed_auth_audit = True
        return adapter

    session.get_adapter = get_adapter
    session._lightspeed_auth_audit = True


# Native model-call scopes and client installation.


def instrument_model_http_clients(model: Any) -> None:
    """Observe explicit clients only; never trigger a lazy credential/client getter."""
    for value in vars(model).values():
        if isinstance(value, (httpx.Client, httpx.AsyncClient)):
            instrument_httpx_client(value)


def instrument_google_client(client: Any) -> None:
    """Observe the SDK-selected backend, including loop-local recreated sessions."""
    api = client._api_client
    for name in ("_httpx_client", "_async_httpx_client"):
        native = getattr(api, name, None)
        if isinstance(native, (httpx.Client, httpx.AsyncClient)):
            instrument_httpx_client(native)
    if "_lightspeed_audit_installed" in vars(api):
        return
    api._lightspeed_audit_installed = True
    original_get = api._get_aiohttp_session
    previous_audit_middleware: Any = None

    async def get_session() -> Any:
        nonlocal previous_audit_middleware

        session = await original_get()
        instrument_aiohttp_session(session)
        middleware = session._lightspeed_audit_middleware
        args = api._async_client_session_request_args
        previous = args.get("middlewares", getattr(session, "_middlewares", ()))
        # The SDK reconstructs these kwargs on its native connection retry.
        # Per-request middleware also reaches mTLS authorized sessions without
        # instrumenting the credential transport's own token requests.
        args["middlewares"] = (
            *(
                item
                for item in previous
                if item is not previous_audit_middleware and item is not middleware
            ),
            middleware,
        )
        previous_audit_middleware = middleware
        return session

    api._get_aiohttp_session = get_session


def audited_model_class(model_class: Any) -> type[BaseChatModel]:
    """Scope execution, not construction; preserve native methods and callbacks."""

    def scope(model: Any, run_manager: Any) -> Any:
        run_id = getattr(run_manager, "run_id", None)
        return llm_call_scope(
            model_key=id(model),
            logical_call_id=str(run_id) if run_id is not None else None,
        )

    class AuditedModel(model_class):  # type: ignore[misc]  # Optional native SDK base.
        # LangChain checks these signatures before supplying the run manager.
        def _generate(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> Any:
            with scope(self, run_manager)():
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        async def _agenerate(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> Any:
            with scope(self, run_manager)():
                return await super()._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )

        def _stream(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> Iterator[Any]:
            activate = scope(self, run_manager)
            iterator = super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            try:
                while True:
                    with activate():
                        try:
                            chunk = next(iterator)
                        except StopIteration:
                            return
                    yield chunk
            finally:
                with activate():
                    iterator.close()

        async def _astream(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> AsyncIterator[Any]:
            activate = scope(self, run_manager)
            iterator = super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
            try:
                while True:
                    with activate():
                        try:
                            chunk = await anext(iterator)
                        except StopAsyncIteration:
                            return
                    yield chunk
            finally:
                with activate():
                    await iterator.aclose()

    return cast("type[BaseChatModel]", AuditedModel)
