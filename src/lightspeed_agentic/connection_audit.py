"""Payload-free audit of scoped model, MCP and credential HTTP attempts."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import math
import os
import re
import socket
import ssl
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import Any, Literal
from uuid import UUID, uuid4

import httpx
from opentelemetry import context, trace

AuthProvider = Literal["azure", "google", "aws"]
_LOGGER = logging.getLogger("lightspeed_agentic.connection_audit")
_run_identity: tuple[str | None, str | None] | None = None
_FIELDS = frozenset(
    [
        "schema_version",
        "event",
        "timestamp",
        "service.name",
        "attempt_id",
        "request.kind",
        "server.address",
        "server.port",
        "network.peer.address",
        "network.transport",
        "http.request.method",
        "trace_id",
        "span_id",
        "gen_ai.conversation.id",
        "agenticrun.uid",
        "agenticrun.phase",
        "gen_ai.tool.name",
        "outcome",
        "error.type",
        "http.response.status_code",
        "duration_ms",
        "request.body.size",
    ]
)


@dataclass(frozen=True)
class Destination:
    """An authority only, never a full URL."""

    scheme: str
    host: str
    port: int


@dataclass(frozen=True)
class _Call:
    kind: Literal["llm", "mcp", "auth"]
    model_key: int | None
    logical_call_id: str | None
    span_context: trace.SpanContext
    provider: AuthProvider | None = None
    tool_name: str | None = None


_call: ContextVar[_Call | None] = ContextVar("llm_audit_call", default=None)
_invocation: ContextVar[tuple[str | None, str | None] | None] = ContextVar(
    "llm_audit_invocation", default=None
)


def llm_call_scope(
    *, model_key: int, logical_call_id: str | None
) -> Callable[[], AbstractContextManager[None]]:
    """Capture a dispatch once; activate it only during execution, never caller yields."""
    previous = _call.get()
    if (
        previous is not None
        and previous.kind == "llm"
        and previous.model_key == model_key
        and (logical_call_id is None or logical_call_id == previous.logical_call_id)
    ):
        current = previous
    else:
        current = _Call(
            "llm",
            model_key,
            logical_call_id or uuid4().hex,
            trace.get_current_span().get_span_context(),
        )

    @contextmanager
    def activate() -> Iterator[None]:
        token = _call.set(current)
        try:
            yield
        finally:
            _call.reset(token)

    return activate


@contextmanager
def llm_call(*, model_key: int, logical_call_id: str | None) -> Iterator[None]:
    """Scope a native non-streaming dispatch or nested fallback."""
    with llm_call_scope(model_key=model_key, logical_call_id=logical_call_id)():
        yield


@contextmanager
def mcp_request(tool_name: str | None = None) -> Iterator[None]:
    """Scope MCP discovery or one tool call, including SDK-owned requests."""
    token = _call.set(
        _Call(
            "mcp",
            None,
            None,
            trace.get_current_span().get_span_context(),
            tool_name=tool_name,
        )
    )
    try:
        yield
    finally:
        _call.reset(token)


@contextmanager
def auth_request(provider: AuthProvider) -> Iterator[None]:
    """Scope native acquisition/refresh, sharing nested same-provider exchanges."""
    previous = _call.get()
    if previous is not None and previous.kind == "auth" and previous.provider == provider:
        yield
        return
    token = _call.set(
        _Call(
            "auth",
            None,
            None,
            trace.get_current_span().get_span_context(),
            provider=provider,
        )
    )
    try:
        yield
    finally:
        _call.reset(token)


@contextmanager
def invocation_context(uid: str | None, phase: str | None) -> Iterator[None]:
    """Carry caller-supplied identity even when the trace runtime is disabled."""
    token = _invocation.set((uid or None, phase or None))
    try:
        yield
    finally:
        _invocation.reset(token)


class _JSONHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Resolve stdout at emission time: do not retain a closed capture stream.
        with suppress(Exception):
            sys.stdout.write(record.getMessage() + "\n")
            sys.stdout.flush()


def configure_connection_audit() -> None:
    """Keep security JSON separate from developer/content log formatting."""
    _LOGGER.propagate = False
    _LOGGER.disabled = False
    _LOGGER.setLevel(logging.INFO)
    if not any(isinstance(handler, _JSONHandler) for handler in _LOGGER.handlers):
        _LOGGER.addHandler(_JSONHandler())


def set_run_identity(uid: str | None, phase: str | None) -> None:
    """Use the tracing runtime's already-resolved correlation, or restore env fallback."""
    global _run_identity
    _run_identity = None if uid is None and phase is None else (uid or None, phase or None)


def attach_otlp_handler(handler: logging.Handler) -> None:
    """Share the existing runtime's handler, without duplicate root propagation."""
    configure_connection_audit()
    if handler not in _LOGGER.handlers:
        _LOGGER.addHandler(handler)


def detach_otlp_handler(handler: logging.Handler) -> None:
    """Detach only the handler owned by the shutting-down tracing runtime."""
    _LOGGER.removeHandler(handler)


def _uuid(value: Any) -> str | None:
    if type(value) is not str or len(value) not in (32, 36):
        return None
    if not re.fullmatch(
        r"[0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value
    ):
        return None
    return str(UUID(value))


def _host(value: Any, *, peer: bool = False) -> str | None:
    """Normalize bounded authorities without DNS lookups or URL diagnostics."""
    if type(value) is not str or not 0 < len(value) <= 254 or "%" in value:
        return None
    host = value.lower() if peer else value.removesuffix(".").lower()
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        if peer:
            return None
    if len(host) > 253 or not host.isascii():
        return None
    if "." in host and host.replace(".", "").isdigit():
        return None  # Do not reinterpret malformed IP literals as DNS names.
    if not all(
        re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in host.split(".")
    ):
        return None
    return host


def _validated_metadata(name: str, value: Any) -> Any:
    """Omit invalid metadata; never stringify it or echo rejected values."""
    if name in {"agenticrun.uid", "gen_ai.conversation.id", "attempt_id"}:
        return _uuid(value)
    if name in {"trace_id", "span_id"}:
        width = 32 if name == "trace_id" else 16
        if (
            type(value) is str
            and re.fullmatch(rf"[0-9a-fA-F]{{{width}}}", value)
            and int(value, 16)
        ):
            return value.lower()
        return None
    if name == "agenticrun.phase":
        return value if value in {"analysis", "execution", "verification", "escalation"} else None
    if name == "server.address":
        return _host(value)
    if name == "network.peer.address":
        return _host(value, peer=True)
    if name in {"server.port", "http.response.status_code"}:
        low, high = (1, 65535) if name == "server.port" else (100, 599)
        return value if type(value) is int and low <= value <= high else None
    if name == "gen_ai.tool.name":
        return (
            value if type(value) is str and 0 < len(value) <= 256 and value.isprintable() else None
        )
    if name == "http.request.method":
        return (
            value
            if type(value) is str and re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,32}", value)
            else None
        )
    if name == "request.body.size":
        return value if type(value) is int and value >= 0 else None
    if name == "duration_ms":
        return value if type(value) in (int, float) and value >= 0 else None
    return value


def emit(record: dict[str, Any]) -> None:
    """Emit an explicitly constructed scalar record only when audit is enabled."""
    if os.environ.get("LIGHTSPEED_AUDIT_ENABLED", "").strip().lower() != "true":
        return
    configure_connection_audit()
    # Adapter bookkeeping is private. Project it onto the shared v1 wire contract
    # rather than serializing model, classifier, proxy or TLS-decision metadata.
    event = record.get("event")
    if isinstance(event, str) and event.startswith(
        ("llm.request.", "mcp.request.", "auth.request.")
    ):
        domain, stage = event.split(".request.", 1)
        category = {
            "dns": "dns_error",
            "connect": "connect_error",
            "tls": "tls_error",
            "timeout": "timeout",
            "protocol_error": "protocol_error",
            "stream_error": "protocol_error",
            "incomplete_response": "protocol_error",
            "http_error": "http_error",
            "cancelled": "cancelled",
            "transport": "unknown_error",
        }.get(record.get("failure_category") or "")
        wire = {
            "schema_version": 1,
            "event": "outbound.request.started"
            if stage == "started"
            else "outbound.request.finished",
            "timestamp": record.get("timestamp"),
            "service.name": "lightspeed-agentic-sandbox",
            "attempt_id": record.get("attempt_id"),
            "request.kind": "credential" if domain == "auth" else domain,
            "server.address": record.get("target_host") or record.get("destination_host"),
            "server.port": record.get("target_port")
            if record.get("target_host")
            else record.get("destination_port"),
            "network.transport": "tcp",
            "http.request.method": record.get("http.request.method"),
            "trace_id": record.get("trace_id"),
            "span_id": record.get("span_id"),
            "agenticrun.uid": record.get("run_id"),
            "agenticrun.phase": record.get("phase"),
            "gen_ai.tool.name": record.get("tool_name") if domain == "mcp" else None,
        }
        if stage == "completed":
            wire.update(
                {
                    "outcome": ("http_error" if category == "http_error" else "transport_error")
                    if record.get("outcome") == "failure"
                    else record.get("outcome"),
                    "error.type": category,
                    "duration_ms": record.get("duration_ms"),
                    "http.response.status_code": record.get("http_status"),
                }
            )
    else:
        wire = record
    safe = {}
    for name, value in wire.items():
        if (
            name not in _FIELDS
            or type(value) not in (str, bool, int, float)
            or (type(value) is float and not math.isfinite(value))
        ):
            continue
        validated = _validated_metadata(name, value)
        if validated is not None:
            safe[name] = validated
    if safe.get("request.kind") != "mcp":
        safe.pop("gen_ai.tool.name", None)
    _LOGGER.info(
        json.dumps(safe, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
        extra={
            name: safe[name]
            for name in ("event", "agenticrun.uid", "agenticrun.phase")
            if name in safe
        },
    )


def _failure(error: BaseException | None) -> str:
    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    seen: set[int] = set()
    errors: list[BaseException] = []
    while error is not None and id(error) not in seen and len(errors) < 16:
        seen.add(id(error))
        errors.append(error)
        # Handling context can belong to an earlier attempt, not this failure.
        error = error.__cause__
    for kind, category in (
        (ssl.SSLCertVerificationError, "tls_verification_failed"),
        (socket.gaierror, "dns"),
        (ssl.SSLError, "tls"),
        ((httpx.TimeoutException, TimeoutError), "timeout"),
        ((httpx.ConnectError, ConnectionError), "connect"),
        (httpx.ProtocolError, "protocol_error"),
        (asyncio.CancelledError, "cancelled"),
    ):
        if any(isinstance(exc, kind) for exc in errors):
            return category
    return "transport"


class RequestAttempt:
    """One attempt; body/close paths may finish it more than once safely."""

    def __init__(
        self,
        record: dict[str, Any],
        started: float,
        span_context: trace.SpanContext,
        *,
        event_prefix: Literal["llm", "mcp", "auth"] = "llm",
    ) -> None:
        self._event_prefix = event_prefix
        self._record = record
        self._started = started
        self._context = trace.set_span_in_context(trace.NonRecordingSpan(span_context))
        self._finished = False
        self._status: int | None = None
        self._send(f"{self._event_prefix}.request.started")

    def _send(self, event: str, **fields: Any) -> None:
        token = context.attach(self._context)
        try:
            # No raw diagnostic fallback; audit I/O cannot replace request errors.
            with suppress(Exception):
                emit(
                    {
                        **self._record,
                        "event": event,
                        "timestamp": datetime.now(UTC).isoformat(),
                        **fields,
                    }
                )
        finally:
            context.detach(token)

    def response_headers(self, status: int) -> None:
        """Record headers without declaring a streamed response complete."""
        self._status = status

    def finish(
        self,
        *,
        response_complete: bool,
        error: BaseException | None = None,
        cancelled: bool = False,
    ) -> None:
        """Finish once, retaining only a safe failure category."""
        if self._finished:
            return
        self._finished = True
        if isinstance(error, GeneratorExit):
            error = None  # Consumer closure, not a transport failure.
        category: str | None = None
        if error is not None:
            category = _failure(error)
            if category == "transport" and self._status is not None:
                category = "stream_error"
        if category == "tls_verification_failed":
            category = "tls"
        if cancelled or category == "cancelled":
            outcome, category = "cancelled", "cancelled"
        elif category is not None:
            outcome = "failure"
        elif self._status is not None and self._status >= 400:
            outcome, category = "failure", "http_error"
        elif not response_complete or self._status is None:
            outcome, category = "failure", "incomplete_response"
        else:
            outcome = "success"
        self._send(
            f"{self._event_prefix}.request.completed",
            outcome=outcome,
            duration_ms=(monotonic() - self._started) * 1000,
            http_status=self._status,
            failure_category=category,
        )


def start_attempt(
    destination: Destination,
    *,
    target: Destination | None = None,
    method: str | None = None,
) -> RequestAttempt | None:
    """Begin only scoped model/MCP/auth requests, never arbitrary client traffic."""
    call = _call.get()
    if call is None:
        return None
    started = monotonic()
    span = call.span_context
    mcp = call.kind == "mcp"
    auth = call.kind == "auth"
    uid, phase = (
        _run_identity
        if _run_identity is not None
        else (
            os.environ.get("LIGHTSPEED_AGENTICRUN_UID") or None,
            os.environ.get("LIGHTSPEED_AGENTICRUN_STEP") or None,
        )
    )
    invocation = _invocation.get()
    if invocation is not None:
        uid, phase = invocation[0] or uid, invocation[1] or phase
    record = {
        "schema_version": 1,
        "attempt_id": uuid4().hex,
        "run_id": uid,
        "phase": phase,
        "http.request.method": method,
        "trace_id": format(span.trace_id, "032x") if span.is_valid else None,
        "span_id": format(span.span_id, "016x") if span.is_valid else None,
        "destination_host": destination.host,
        "destination_port": destination.port,
        "target_host": target.host if target else None,
        "target_port": target.port if target else None,
        "tool_name": call.tool_name if mcp else None,
    }
    return RequestAttempt(
        record, started, span, event_prefix="mcp" if mcp else "auth" if auth else "llm"
    )
