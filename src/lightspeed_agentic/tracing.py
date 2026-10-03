"""OTEL tracing and logging — provider initialization and traceparent parsing.

The tracer and logger providers share a Resource with ``service.name`` set to
``lightspeed-agentic-sandbox``. Compliance stdout emits an OTLP JSON projection
for each ended span when audit is enabled. GenAI spans become one templog record
whose body contains span attributes; non-GenAI span events retain generic event
bridging through stdlib ``logging`` and ``LoggingHandler``.

Templog records stamp ``agenticrun.uid`` / ``agenticrun.phase`` / ``event`` via
``logging`` ``extra``. Phase comes from ``result-template.kind`` via
``init_tracer(agenticrun_phase=…)``; uid from env when set. Compliance content
filtering applies only to derived stdout/log copies; trace endpoint export keeps
the original source spans.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

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
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

_DEFAULT_SERVICE_NAME = "lightspeed-agentic-sandbox"
_TRACER_NAME = "lightspeed_agentic"
_SCHEMA_URL = "https://opentelemetry.io/schemas/1.41.0"
_ATTR_AGENTICRUN_UID = "agenticrun.uid"
_ATTR_AGENTICRUN_PHASE = "agenticrun.phase"
_ATTR_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
_CONTENT_ATTRIBUTES = frozenset(
    {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.definitions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
    }
)
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
    """Exports each ended span as OTLP JSON to stdout."""

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            request = encode_spans((span,))
            if not self._capture_content:
                _remove_content_attributes_from_request(request)
            line = json.dumps(MessageToDict(request, preserving_proto_field_name=True))
            sys.stdout.write(line + "\n")
        if spans:
            sys.stdout.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


class _ComplianceSpanToLogsProcessor(SpanProcessor):
    """Forward compliance span records through stdlib logging.

    GenAI spans produce one record with the span attributes as a JSON body and
    ``gen_ai.operation.name`` as the ``event`` record attribute. Non-GenAI spans
    retain generic event forwarding. ``LoggingHandler`` dual-ships these records
    to stderr and OTLP.
    """

    def __init__(
        self,
        *,
        agenticrun_uid: str = "",
        agenticrun_phase: str = "",
        capture_content: bool = True,
    ) -> None:
        self._attrs: dict[str, str] = {}
        if agenticrun_uid:
            self._attrs[_ATTR_AGENTICRUN_UID] = agenticrun_uid
        if agenticrun_phase:
            self._attrs[_ATTR_AGENTICRUN_PHASE] = agenticrun_phase
        self._capture_content = capture_content

    def on_end(self, span: ReadableSpan) -> None:
        span_attributes: Mapping[str, object] = span.attributes or {}
        operation = span_attributes.get(_ATTR_GEN_AI_OPERATION_NAME)
        try:
            if isinstance(operation, str) and operation:
                self._emit_record(
                    span,
                    operation,
                    _compliance_attributes(span_attributes, self._capture_content),
                    span_attributes,
                )
                return
            self._emit_events(span, span_attributes, span.events or ())
        except Exception:
            # Never break span export / request path if log bridging fails.
            _logger.exception("failed to forward compliance spans to OTLP logs")

    def _emit_events(
        self,
        span: ReadableSpan,
        span_attributes: Mapping[str, object],
        events: Sequence[Event],
    ) -> None:
        for event in events:
            if event.name == "exception" or event.name.startswith("gen_ai."):
                continue
            self._emit_record(
                span,
                event.name,
                _compliance_attributes(event.attributes, self._capture_content),
                span_attributes,
            )

    def _emit_record(
        self,
        span: ReadableSpan,
        event_name: str,
        body_attributes: dict[str, object],
        span_attributes: Mapping[str, object],
    ) -> None:
        token = attach(_span_context_for_logs(span))
        try:
            _audit_bridge_logger.info(
                json.dumps(body_attributes, default=str),
                extra={
                    "event": event_name,
                    **self._record_correlation_attributes(span_attributes),
                },
            )
        finally:
            detach(token)

    def _record_correlation_attributes(
        self, span_attributes: Mapping[str, object]
    ) -> dict[str, object]:
        record_attributes: dict[str, object] = dict(self._attrs)
        for key in (_ATTR_AGENTICRUN_UID, _ATTR_AGENTICRUN_PHASE):
            if value := span_attributes.get(key):
                record_attributes[key] = value
        return record_attributes


def _compliance_attributes(
    attributes: Mapping[str, object] | None, capture_content: bool
) -> dict[str, object]:
    compliance_attributes = dict(attributes or {})
    if not capture_content:
        for key in _CONTENT_ATTRIBUTES:
            compliance_attributes.pop(key, None)
    return compliance_attributes


def _remove_content_attributes_from_request(request: Any) -> None:
    for resource_spans in request.resource_spans:
        for scope_spans in resource_spans.scope_spans:
            for span in scope_spans.spans:
                _remove_encoded_content_attributes(span.attributes)
                for event in span.events:
                    _remove_encoded_content_attributes(event.attributes)
                for link in span.links:
                    _remove_encoded_content_attributes(link.attributes)


def _remove_encoded_content_attributes(attributes: Any) -> None:
    for index in range(len(attributes) - 1, -1, -1):
        if attributes[index].key in _CONTENT_ATTRIBUTES:
            del attributes[index]


def _resolve_capture_content(audit_enabled: bool) -> bool:
    raw = os.environ.get("LIGHTSPEED_CAPTURE_CONTENT", "").strip().lower()
    if raw == "false":
        return False
    if raw == "true":
        return True
    return audit_enabled


class _AgenticRunFilter(logging.Filter):
    """Fill missing run correlation on logs forwarded to the collector.

    Configured UID and phase values are fallbacks; values supplied by the
    completed span take precedence.
    """

    def __init__(self, *, agenticrun_uid: str = "", agenticrun_phase: str = "") -> None:
        super().__init__()
        self._uid = agenticrun_uid
        self._phase = agenticrun_phase

    def filter(self, record: logging.LogRecord) -> bool:
        if self._uid and not record.__dict__.get(_ATTR_AGENTICRUN_UID):
            setattr(record, _ATTR_AGENTICRUN_UID, self._uid)
        if self._phase and not record.__dict__.get(_ATTR_AGENTICRUN_PHASE):
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
    - Ended GenAI spans become one attribute-body log record when endpoint and
      audit are enabled; non-GenAI span events keep generic event forwarding.
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
    capture_content = _resolve_capture_content(audit)

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
            SimpleSpanProcessor(OTLPJsonStdoutExporter(capture_content=capture_content))
        )
    if endpoint and audit:
        _state.tracer_provider.add_span_processor(
            _ComplianceSpanToLogsProcessor(
                agenticrun_uid=agenticrun_uid,
                agenticrun_phase=agenticrun_phase,
                capture_content=capture_content,
            )
        )
    _configure_trace_exporter(_state.tracer_provider, endpoint=endpoint, protocol=protocol)
    trace.set_tracer_provider(_state.tracer_provider)


def _http_signal_endpoint(endpoint: str, signal: str) -> str:
    base = urlsplit(endpoint)
    return base._replace(path=f"{base.path.rstrip('/')}/v1/{signal}").geturl()


def _configure_trace_exporter(provider: TracerProvider, *, endpoint: str, protocol: str) -> None:
    if not endpoint:
        return

    exporter: SpanExporter
    if protocol == "http/protobuf":
        exporter = HttpSpanExporter(endpoint=_http_signal_endpoint(endpoint, "traces"))
    else:
        exporter = GrpcSpanExporter(endpoint=endpoint)

    provider.add_span_processor(BatchSpanProcessor(exporter))


def _configure_log_exporter(provider: LoggerProvider, *, endpoint: str, protocol: str) -> None:
    if not endpoint:
        return

    if protocol == "http/protobuf":
        provider.add_log_record_processor(
            BatchLogRecordProcessor(
                HttpLogExporter(endpoint=_http_signal_endpoint(endpoint, "logs"))
            )
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
    return trace.get_tracer(_TRACER_NAME, schema_url=_SCHEMA_URL)


def parse_traceparent(header: str | None) -> tuple[str | None, Context | None]:
    """Parse W3C traceparent as (trace_id, parent_context).

    Invalid or missing headers return (None, None), signaling a root span.
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
                    return None, None
                span_ctx = SpanContext(
                    trace_id=trace_id,
                    span_id=parent_id,
                    is_remote=True,
                    trace_flags=TraceFlags(flags),
                )
                ctx = trace.set_span_in_context(NonRecordingSpan(span_ctx))
                return trace_id_hex, ctx
    return None, None
