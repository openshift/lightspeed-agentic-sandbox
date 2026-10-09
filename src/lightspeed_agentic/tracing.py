"""OTEL tracing and logging — provider initialization and traceparent parsing.

At startup we configure both signals from env: TracerProvider (traces) and
LoggerProvider (logs), sharing one Resource from ``Resource.create()``
with ``service.name`` set to ``lightspeed-agentic-sandbox``. Stdout audit
remains the existing span → OTLP-JSON exporter when audit is enabled.
When an OTLP endpoint is set **and** audit is enabled, a span processor
forwards AuditLogger span events (e.g. ``gen_ai.choice``) through stdlib
``logging`` so ``LoggingHandler`` dual-ships them to stderr and OTLP
(templog). When an OTLP endpoint is set, all stdlib ``logging`` is
dual-shipped the same way.

Templog (lightspeed-otel-collector postgresexporter) reads log **record**
attributes, so bridged logs stamp ``agenticrun.uid`` / ``agenticrun.phase`` /
``event`` via ``logging`` ``extra``. Phase comes from ``result-template.kind``
via ``init_tracer(agenticrun_phase=…)``; uid from env when set.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]
from opentelemetry import _logs, trace
from opentelemetry.context import Context, attach, detach
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter as GrpcLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter as HttpLogExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HttpSpanExporter,
)
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import NonRecordingSpan, Span, SpanContext, SpanKind, TraceFlags

_DEFAULT_SERVICE_NAME = "lightspeed-agentic-sandbox"
_TRACER_NAME = "lightspeed_agentic"
_ATTR_AGENTICRUN_UID = "agenticrun.uid"
_ATTR_AGENTICRUN_PHASE = "agenticrun.phase"
_logger = logging.getLogger(__name__)
_audit_bridge_logger = logging.getLogger("lightspeed_agentic.audit")


@dataclass
class _OtelState:
    tracer_provider: TracerProvider | None = None
    logger_provider: LoggerProvider | None = None
    logging_handler: LoggingHandler | None = None


_state = _OtelState()


def otel_runtime_enabled() -> bool:
    """Return True when stdout audit or OTLP export should be configured."""
    audit = os.environ.get("LIGHTSPEED_AUDIT_ENABLED", "").strip().lower() == "true"
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    return audit or bool(endpoint)


class OTLPJsonStdoutExporter(SpanExporter):
    """Exports spans as OTLP JSON wire format to stdout (one line per batch)."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        pb = encode_spans(spans)
        line = json.dumps(MessageToDict(pb, preserving_proto_field_name=True))
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


class _AdkSpanFilteringExporter(SpanExporter):
    """Exclude native ADK spans and delegate remaining exporter operations."""

    def __init__(self, exporter: SpanExporter) -> None:
        self._exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        filtered_spans = [
            span
            for span in spans
            if span.instrumentation_scope is None
            or span.instrumentation_scope.name != "gcp.vertex.agent"
        ]
        if not filtered_spans:
            return SpanExportResult.SUCCESS
        return self._exporter.export(filtered_spans)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._exporter.shutdown()


class _SpanEventsToLogsProcessor(SpanProcessor):
    """Forward span events through stdlib logging (templog via LoggingHandler).

    AuditLogger still emits once (span events). This is only a second
    destination — one stdlib log record per span event, body = event attrs
    JSON. ``LoggingHandler`` dual-ships to stderr and OTLP.

    Stamps ``agenticrun.uid`` / ``agenticrun.phase`` / ``event`` on each
    record via ``logging`` ``extra`` for the collector postgresexporter.

    Skips OTel automatic ``exception`` events (stack traces) so templog stays
    intentional audit events without hard-coding a gen_ai.* allowlist.
    """

    def __init__(self, *, agenticrun_uid: str = "", agenticrun_phase: str = "") -> None:
        self._attrs: dict[str, str] = {}
        if agenticrun_uid:
            self._attrs[_ATTR_AGENTICRUN_UID] = agenticrun_uid
        if agenticrun_phase:
            self._attrs[_ATTR_AGENTICRUN_PHASE] = agenticrun_phase

    def on_end(self, span: ReadableSpan) -> None:
        events = span.events
        if not events:
            return
        try:
            self._emit_events(span, events)
        except Exception:
            # Never break span export / request path if log bridging fails.
            _logger.exception("failed to forward span events to OTLP logs")

    def _emit_events(self, span: ReadableSpan, events: Sequence[Event]) -> None:
        # Attach ended span context so LoggingHandler stamps TraceID.
        token = attach(_span_context_for_logs(span))
        try:
            for event in events:
                if event.name == "exception":
                    continue
                _audit_bridge_logger.info(
                    json.dumps(dict(event.attributes or {}), default=str),
                    extra={"event": event.name, **self._attrs},
                )
        finally:
            detach(token)


class _AgenticRunFilter(logging.Filter):
    """Stamp agenticrun.uid and agenticrun.phase on every log record.

    Attached to the ``LoggingHandler`` so all stdlib log records forwarded
    to OTLP carry the run identity attributes the collector needs.
    """

    def __init__(self, *, agenticrun_uid: str = "", agenticrun_phase: str = "") -> None:
        super().__init__()
        self._uid = agenticrun_uid
        self._phase = agenticrun_phase

    def filter(self, record: logging.LogRecord) -> bool:
        if self._uid:
            setattr(record, _ATTR_AGENTICRUN_UID, self._uid)
        if self._phase:
            setattr(record, _ATTR_AGENTICRUN_PHASE, self._phase)
        return True


def _span_context_for_logs(span: ReadableSpan) -> Context:
    sc = span.get_span_context()
    if sc is None or not sc.is_valid:
        return Context()
    return trace.set_span_in_context(
        NonRecordingSpan(
            SpanContext(
                trace_id=sc.trace_id,
                span_id=sc.span_id,
                is_remote=sc.is_remote,
                trace_flags=sc.trace_flags,
                trace_state=sc.trace_state,
            )
        )
    )


def init_tracer(
    *,
    agenticrun_uid: str | None = None,
    agenticrun_phase: str | None = None,
) -> None:
    """Initialize OTEL TracerProvider and LoggerProvider from env.

    ``agenticrun_phase`` should be the workflow step from ``result-template.kind``
    (analysis, execution, verification, escalation). ``agenticrun_uid`` defaults
    to ``LIGHTSPEED_AGENTICRUN_UID`` when omitted.

    Traces:
    - Stdout OTLP-JSON exporter when ``LIGHTSPEED_AUDIT_ENABLED=true``.
    - OTLP span exporter when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set.

    Logs:
    - OTLP log exporter when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set.
    - Stdlib ``logging`` dual-shipped to stderr and OTLP via ``LoggingHandler``
      when the endpoint is set.
    - Span-event → log bridge when endpoint is set **and** audit is enabled
      (same gate as stdout): emits via stdlib so LoggingHandler dual-ships;
      stamps uid/phase/event in ``extra`` for templog.
    """
    if _state.tracer_provider is not None or _state.logger_provider is not None:
        raise RuntimeError("OTEL providers already initialized; call shutdown_tracer() first")

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").strip().lower() or "grpc"
    if protocol not in ("grpc", "http/protobuf"):
        _logger.warning("unsupported OTEL_EXPORTER_OTLP_PROTOCOL=%r, defaulting to grpc", protocol)
        protocol = "grpc"

    resource = Resource.create().merge(Resource({SERVICE_NAME: _DEFAULT_SERVICE_NAME}))
    if agenticrun_uid is None:
        agenticrun_uid = os.environ.get("LIGHTSPEED_AGENTICRUN_UID", "").strip()
    if agenticrun_phase is None:
        agenticrun_phase = os.environ.get("LIGHTSPEED_AGENTICRUN_STEP", "").strip()
    audit = os.environ.get("LIGHTSPEED_AUDIT_ENABLED", "").strip().lower() == "true"

    if endpoint and audit:
        missing = [
            name
            for name, value in (
                ("LIGHTSPEED_AGENTICRUN_UID", agenticrun_uid),
                ("LIGHTSPEED_AGENTICRUN_STEP", agenticrun_phase),
            )
            if not value
        ]
        if missing:
            _logger.warning(
                "OTLP audit/templog enabled but cannot resolve env %s; "
                "bridged log records will lack those attributes and the "
                "collector will skip records missing agenticrun.uid",
                ", ".join(missing),
            )

    _state.logger_provider = LoggerProvider(resource=resource)
    _configure_log_exporter(_state.logger_provider, endpoint=endpoint, protocol=protocol)
    _logs.set_logger_provider(_state.logger_provider)

    if endpoint:
        _state.logging_handler = LoggingHandler(logger_provider=_state.logger_provider)
        # Stamp agenticrun.uid / agenticrun.phase on every log record
        # so the collector postgresexporter can index them.
        stamp = _AgenticRunFilter(agenticrun_uid=agenticrun_uid, agenticrun_phase=agenticrun_phase)
        _state.logging_handler.addFilter(stamp)
        root = logging.getLogger()
        root.addHandler(_state.logging_handler)
        # LoggingHandler only sees records that pass the root effective level.
        # App startup uses INFO; pytest often leaves root at WARNING.
        if root.getEffectiveLevel() > logging.INFO:
            root.setLevel(logging.INFO)
        _audit_bridge_logger.setLevel(logging.INFO)

    _state.tracer_provider = TracerProvider(resource=resource)
    if audit:
        _state.tracer_provider.add_span_processor(
            SimpleSpanProcessor(_AdkSpanFilteringExporter(OTLPJsonStdoutExporter()))
        )
    if endpoint and audit:
        # Same gate as stdout audit: only forward when audit is enabled.
        _state.tracer_provider.add_span_processor(
            _SpanEventsToLogsProcessor(
                agenticrun_uid=agenticrun_uid,
                agenticrun_phase=agenticrun_phase,
            )
        )
    _configure_trace_exporter(_state.tracer_provider, endpoint=endpoint, protocol=protocol)
    trace.set_tracer_provider(_state.tracer_provider)


def _configure_trace_exporter(provider: TracerProvider, *, endpoint: str, protocol: str) -> None:
    if not endpoint:
        return

    exporter: SpanExporter
    if protocol == "http/protobuf":
        exporter = HttpSpanExporter(endpoint=endpoint)
    else:
        exporter = GrpcSpanExporter(endpoint=endpoint)

    provider.add_span_processor(BatchSpanProcessor(_AdkSpanFilteringExporter(exporter)))


def _configure_log_exporter(provider: LoggerProvider, *, endpoint: str, protocol: str) -> None:
    if not endpoint:
        return

    if protocol == "http/protobuf":
        provider.add_log_record_processor(
            BatchLogRecordProcessor(HttpLogExporter(endpoint=endpoint))
        )
    else:
        provider.add_log_record_processor(
            BatchLogRecordProcessor(GrpcLogExporter(endpoint=endpoint))
        )


def shutdown_tracer() -> None:
    """Shutdown tracer and logger providers, flushing pending exports."""
    if _state.logging_handler is not None:
        logging.getLogger().removeHandler(_state.logging_handler)
        _state.logging_handler = None
    if _state.tracer_provider:
        try:
            _state.tracer_provider.shutdown()
        finally:
            _state.tracer_provider = None
    if _state.logger_provider:
        try:
            _state.logger_provider.shutdown()
        finally:
            _state.logger_provider = None


def get_tracer() -> trace.Tracer:
    """Get a tracer instance for creating spans."""
    return trace.get_tracer(_TRACER_NAME)


def start_generation_span(
    operation: str, model: str, provider: str, *, parent_context: Context
) -> Span:
    """Start a model generation without changing the active SDK/log context."""
    attributes = {
        "gen_ai.operation.name": operation,
        "gen_ai.request.model": model,
        "gen_ai.provider.name": provider,
    }
    parent = trace.get_current_span(parent_context)
    parent_attributes = getattr(parent, "attributes", None) or {}
    for key in (_ATTR_AGENTICRUN_UID, _ATTR_AGENTICRUN_PHASE):
        if value := parent_attributes.get(key):
            attributes[key] = value
    return get_tracer().start_span(
        f"{operation} {model}",
        kind=SpanKind.CLIENT,
        context=parent_context,
        attributes=attributes,
    )


def set_json_span_attribute(span: Span, name: str, value: object) -> None:
    """Set a span attribute to compact UTF-8-safe JSON when recording."""
    if span.is_recording():
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        span.set_attribute(
            name,
            encoded.encode("utf-8", errors="backslashreplace").decode("utf-8"),
        )


def parse_traceparent(header: str | None) -> tuple[str, Context | None]:
    """Parse W3C traceparent header and return (trace_id, context).

    If the header is invalid or missing, generates a new trace ID.
    """
    if header:
        parts = header.split("-")
        if len(parts) >= 4:
            trace_id_hex = parts[1]
            parent_id_hex = parts[2]
            flags_hex = parts[3]
            if (
                len(trace_id_hex) == 32
                and trace_id_hex != "0" * 32
                and len(parent_id_hex) == 16
                and parent_id_hex != "0" * 16
            ):
                try:
                    trace_id = int(trace_id_hex, 16)
                    parent_id = int(parent_id_hex, 16)
                    flags = int(flags_hex, 16)
                except ValueError:
                    return _generate_trace_id()
                span_ctx = SpanContext(
                    trace_id=trace_id,
                    span_id=parent_id,
                    is_remote=True,
                    trace_flags=TraceFlags(flags),
                )
                ctx = trace.set_span_in_context(NonRecordingSpan(span_ctx))
                return trace_id_hex, ctx
    return _generate_trace_id()


def _generate_trace_id() -> tuple[str, Context]:
    """Generate a new trace ID and root context."""
    trace_id_hex = secrets.token_hex(16)
    span_id_hex = secrets.token_hex(8)
    span_ctx = SpanContext(
        trace_id=int(trace_id_hex, 16),
        span_id=int(span_id_hex, 16),
        is_remote=False,
        trace_flags=TraceFlags(1),
    )
    ctx = trace.set_span_in_context(NonRecordingSpan(span_ctx))
    return trace_id_hex, ctx
