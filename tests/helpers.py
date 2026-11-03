"""Shared offline model, runner, and OTLP helpers for tests."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from typing import Any

import grpc
import pytest
from opentelemetry import trace
from opentelemetry._logs import _internal as log_globals
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceResponse
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

from lightspeed_agentic import tracing
from lightspeed_agentic.providers.model_router import (
    ModelCreator,
    ModelParameters,
    ResolvedModelRoute,
)

# Workflow budgets include SDK middleware graph steps, not just model calls.
TEST_WORKFLOW_MAX_TURNS = 20


def make_model_route(
    create_model: ModelCreator, *, model: str = "claude-sonnet-4-6"
) -> ResolvedModelRoute:
    """Build an explicit route around an offline test model constructor."""
    return ResolvedModelRoute(
        route="anthropic",
        model=model,
        create_model=create_model,
        model_parameters=ModelParameters(model=model),
        telemetry_provider="anthropic",
        structured_output_method="function_calling",
    )


class OTLPRequests(Queue[bytes]):
    """Separate trace and log wire payloads from the same local receiver."""

    def __init__(self) -> None:
        super().__init__()
        self.logs: Queue[bytes] = Queue()


@pytest.fixture
def otlp_requests(monkeypatch: pytest.MonkeyPatch) -> Iterator[OTLPRequests]:
    """Receive actual gRPC wire payloads; keep exporter and encoding code real."""
    assert tracing._state.tracer_provider is None
    assert tracing._state.logger_provider is None
    requests = OTLPRequests()

    def receive_trace(payload: bytes, _context: Any) -> ExportTraceServiceResponse:
        requests.put(payload)
        return ExportTraceServiceResponse()

    def receive_logs(payload: bytes, _context: Any) -> ExportLogsServiceResponse:
        requests.logs.put(payload)
        return ExportLogsServiceResponse()

    with ThreadPoolExecutor(max_workers=2) as executor:
        server = grpc.server(executor)
        server.add_generic_rpc_handlers(
            [
                grpc.method_handlers_generic_handler(
                    service,
                    {
                        "Export": grpc.unary_unary_rpc_method_handler(
                            handler,
                            response_serializer=lambda response: response.SerializeToString(),
                        )
                    },
                )
                for service, handler in (
                    ("opentelemetry.proto.collector.trace.v1.TraceService", receive_trace),
                    ("opentelemetry.proto.collector.logs.v1.LogsService", receive_logs),
                )
            ]
        )
        port = server.add_insecure_port("127.0.0.1:0")
        assert port, "Could not bind the local OTLP receiver"
        server.start()
        monkeypatch.setattr(trace, "_TRACER_PROVIDER", None)
        monkeypatch.setattr(trace._TRACER_PROVIDER_SET_ONCE, "_done", False)
        monkeypatch.setattr(log_globals, "_LOGGER_PROVIDER", None)
        monkeypatch.setattr(log_globals._LOGGER_PROVIDER_SET_ONCE, "_done", False)
        logging_levels = [
            (logger, logger.level)
            for logger in (logging.getLogger(), logging.getLogger("lightspeed_agentic.audit"))
        ]
        for signal in ("", "_TRACES", "_LOGS"):
            prefix = f"OTEL_EXPORTER_OTLP{signal}"
            monkeypatch.setenv(f"{prefix}_ENDPOINT", f"http://127.0.0.1:{port}")
            monkeypatch.setenv(f"{prefix}_PROTOCOL", "grpc")
            monkeypatch.setenv(f"{prefix}_INSECURE", "true")
            monkeypatch.setenv(f"{prefix}_TIMEOUT", "2")
            monkeypatch.delenv(f"{prefix}_HEADERS", raising=False)
        monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_on")
        try:
            yield requests
        finally:
            try:
                tracing.shutdown_tracer()
            finally:
                for logger, level in logging_levels:
                    logger.setLevel(level)
                assert server.stop(grace=0).wait(timeout=5), "OTLP receiver failed to stop"


def _value(value: Any) -> Any:
    field = value.WhichOneof("value")
    assert field is not None, "Exported attribute has no value"
    if field == "array_value":
        return [_value(item) for item in value.array_value.values]
    if field == "kvlist_value":
        return {item.key: _value(item.value) for item in value.kvlist_value.values}
    return getattr(value, field)


def attributes(span: Any) -> dict[str, Any]:
    """Preserve protobuf scalar types, explicit zero, and missing attributes."""
    return {attribute.key: _value(attribute.value) for attribute in span.attributes}


def exported_spans(requests: Queue[bytes]) -> list[Any]:
    spans = []
    while not requests.empty():
        request = ExportTraceServiceRequest.FromString(requests.get_nowait())
        for resource in request.resource_spans:
            assert attributes(resource.resource)["service.name"] == "lightspeed-agentic-sandbox"
            for scope in resource.scope_spans:
                assert scope.scope.name != "gcp.vertex.agent"
                spans.extend(scope.spans)
    return spans


@pytest.fixture
def wire_trace(
    otlp_requests: Queue[bytes], monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Callable[[], list[Any]]:
    """Start the production trace runtime and expose flushed, decoded OTLP spans."""
    monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", getattr(request, "param", "false"))
    monkeypatch.setenv("LIGHTSPEED_CAPTURE_CONTENT", "false")
    tracing.init_tracer(
        agenticrun_uid="11111111-1111-4111-8111-111111111111", agenticrun_phase="analysis"
    )

    def collect() -> list[Any]:
        assert tracing._state.tracer_provider.force_flush(timeout_millis=5000)
        return exported_spans(otlp_requests)

    return collect


def scripted_model(responses: list[Any], *, inputs: list[list[Any]] | None = None) -> Any:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import ToolMessage

    class ScriptedModel(FakeMessagesListChatModel):
        def bind_tools(self, _tools: Any, **_kwargs: Any) -> ScriptedModel:
            return self

        def _generate(self, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
            if inputs is not None:
                inputs.append(list(messages))
            for message in messages:
                if isinstance(message, ToolMessage):
                    assert message.status == "success", message.content
            return super()._generate(messages, *args, **kwargs)

    return ScriptedModel(responses=responses)


def messages(span: Any) -> list[dict[str, Any]]:
    return json.loads(attributes(span)["gen_ai.output.messages"])


async def run_traced_query(
    provider: Any,
    cwd: Path,
    *,
    schema: dict[str, Any] | None = None,
    model: str = "requested-model",
) -> Any:
    from lightspeed_agentic.run_agent import run_agent_query

    return await run_agent_query(
        provider,
        prompt="Inspect the available evidence and report it.",
        system_prompt="Report only observed evidence.",
        output_schema=schema,
        context=None,
        skills_dir=str(cwd),
        model=model,
        max_turns=TEST_WORKFLOW_MAX_TURNS,
        timeout_seconds=10,
        tool_output_inspection_enabled=False,
        audit_enabled=False,
        capture_content=False,
        agenticrun_uid="11111111-1111-4111-8111-111111111111",
        step="analysis",
        traceparent="00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
    )


def invocation(spans: list[Any]) -> Any:
    roots = [span for span in spans if span.name == "invoke_agent"]
    assert len(roots) == 1
    root = roots[0]
    assert root.trace_id.hex() == "0123456789abcdef0123456789abcdef"
    assert root.parent_span_id.hex() == "0123456789abcdef"
    for span in spans:
        assert span.trace_id == root.trace_id
        if span is not root:
            assert span.parent_span_id == root.span_id
    return root


def _json_value(data: Any) -> dict[str, Any]:
    """Build OTLP JSON fixtures independently of the decoder under test."""
    if data is None:
        return {}
    if type(data) is bool:
        return {"boolValue": data}
    if type(data) is int:
        return {"intValue": str(data)}
    if type(data) is float:
        return {"doubleValue": data}
    if isinstance(data, list):
        return {"arrayValue": {"values": [_json_value(item) for item in data]}}
    if isinstance(data, dict):
        return {"kvlistValue": {"values": json_attributes(data)}}
    return {"stringValue": data}


def json_attributes(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": key, "value": _json_value(item)} for key, item in data.items()]


def _json_nanos(timestamp: str) -> str:
    date, fraction = timestamp.removesuffix(" +0000 UTC").removesuffix("Z").split(".")
    seconds = int(datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp())
    return str(seconds * 1_000_000_000 + int(fraction.ljust(9, "0")))


def trace_export(spans: list[dict[str, Any]], *, resource_attrs: bool = True) -> str:
    scopes = []
    for span in spans:
        wire: dict[str, Any] = {
            native: span[old]
            for old, native in {
                "Trace ID": "traceId",
                "ID": "spanId",
                "Parent ID": "parentSpanId",
                "Name": "name",
            }.items()
            if old in span
        }
        wire["attributes"] = json_attributes(span.get("attrs", {}))
        if "Kind" in span:
            wire["kind"] = {"Internal": 1, "Server": 2, "Client": 3}[span["Kind"]]
        if "Status code" in span:
            wire["status"] = {
                "code": {"Unset": 0, "Ok": 1, "Error": 2}.get(span["Status code"], 999)
            }
        for old, native in (("Start time", "startTimeUnixNano"), ("End time", "endTimeUnixNano")):
            if span.get(old):
                wire[native] = _json_nanos(span[old])
        if "DroppedAttributesCount" in span:
            wire["droppedAttributesCount"] = int(span["DroppedAttributesCount"])
        for key in ("events", "links"):
            if key in span:
                wire[key] = span[key]
        if "wire_attributes" in span:
            wire["attributes"] = span["wire_attributes"]
        scopes.append({"scope": {"name": span.get("scope", "lightspeed_agentic")}, "spans": [wire]})
    resource = (
        {"attributes": json_attributes({"service.name": "lightspeed-agentic-sandbox"})}
        if resource_attrs
        else {}
    )
    return (
        json.dumps(
            {"resourceSpans": [{"resource": resource, "scopeSpans": scopes}]}, ensure_ascii=False
        )
        + "\n"
    )


def log_export(records: list[dict[str, Any]], *, resource_attrs: bool = True) -> str:
    scopes = []
    for record in records:
        body = record["body"]
        if isinstance(body, dict):
            body = json.dumps(body)
        scopes.append(
            {
                "scope": {"name": record["scope"]},
                "logRecords": [
                    {
                        "body": _json_value(body),
                        "attributes": json_attributes(record["attrs"]),
                        "traceId": record["trace_id"],
                        "spanId": record["span_id"],
                    }
                ],
            }
        )
    resource = (
        {"attributes": json_attributes({"service.name": "lightspeed-agentic-sandbox"})}
        if resource_attrs
        else {}
    )
    return (
        json.dumps(
            {"resourceLogs": [{"resource": resource, "scopeLogs": scopes}]}, ensure_ascii=False
        )
        + "\n"
    )
