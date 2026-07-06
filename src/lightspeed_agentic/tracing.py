"""OTEL tracing — tracer initialization and traceparent parsing.

Uses standard OTEL SDK environment variable configuration. All standard
OTEL_* environment variables are respected:
- OTEL_SERVICE_NAME: Override service name (default: lightspeed-agentic-sandbox)
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (empty = no export)
- OTEL_EXPORTER_OTLP_INSECURE: Explicit TLS control (default: false)
- OTEL_EXPORTER_OTLP_PROTOCOL: Protocol selection (grpc/http/protobuf)
- OTEL_EXPORTER_OTLP_HEADERS: Custom headers (auth tokens)
- OTEL_RESOURCE_ATTRIBUTES: Additional resource attributes
- OTEL_TRACES_EXPORTER: Exporter selection (otlp/none)
"""

from __future__ import annotations

import os
import secrets
from urllib.parse import unquote

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

_DEFAULT_SERVICE_NAME = "lightspeed-agentic-sandbox"
_TRACER_NAME = "lightspeed_agentic"
_tracer_provider: TracerProvider | None = None


def init_tracer() -> None:
    """Initialize the OTEL tracer provider using standard SDK environment variables.

    Respects all standard OTEL_* environment variables. If OTEL_EXPORTER_OTLP_ENDPOINT
    is not set, creates a no-op tracer (no spans exported).
    """
    global _tracer_provider

    # Build resource with service name from env or default
    service_name = os.environ.get("OTEL_SERVICE_NAME", _DEFAULT_SERVICE_NAME)

    # Parse additional resource attributes from OTEL_RESOURCE_ATTRIBUTES
    # Format: key1=value1,key2=value2
    resource_attrs: dict[str, str] = {SERVICE_NAME: service_name}
    if extra_attrs := os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "").strip():
        for pair in extra_attrs.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                resource_attrs[key.strip()] = value.strip()

    resource = Resource.create(resource_attrs)
    _tracer_provider = TracerProvider(resource=resource)

    # Check if exporter is configured
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    traces_exporter = os.environ.get("OTEL_TRACES_EXPORTER", "otlp").strip().lower()

    if endpoint and traces_exporter != "none":
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Determine protocol (grpc is default)
        protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").strip().lower()

        # Determine TLS mode: explicit env var or auto-detect from endpoint scheme
        insecure_env = os.environ.get("OTEL_EXPORTER_OTLP_INSECURE", "").strip().lower()
        insecure = insecure_env == "true" if insecure_env else not endpoint.startswith("https")

        # Parse custom headers (values are URL-encoded per OTEL spec)
        headers_env = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "").strip()
        headers: list[tuple[str, str]] | None = None
        if headers_env:
            headers = []
            for pair in headers_env.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    headers.append((key.strip(), unquote(value.strip())))

        if protocol in ("grpc", ""):
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as GrpcExporter,
            )

            grpc_exporter = GrpcExporter(
                endpoint=endpoint,
                insecure=insecure,
                headers=headers,
            )
            _tracer_provider.add_span_processor(BatchSpanProcessor(grpc_exporter))
        else:
            # http/protobuf protocol
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter as HttpExporter,
            )

            http_exporter = HttpExporter(
                endpoint=endpoint,
                headers=dict(headers) if headers else None,
            )
            _tracer_provider.add_span_processor(BatchSpanProcessor(http_exporter))

    trace.set_tracer_provider(_tracer_provider)


def shutdown_tracer() -> None:
    """Shutdown the tracer provider, flushing any pending spans."""
    if _tracer_provider:
        _tracer_provider.shutdown()


def get_tracer() -> trace.Tracer:
    """Get a tracer instance for creating spans."""
    return trace.get_tracer(_TRACER_NAME)


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
