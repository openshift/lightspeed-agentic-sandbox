from __future__ import annotations

import logging
from typing import ClassVar

import pytest

from lightspeed_agentic.inspection.errors import InspectionError
from lightspeed_agentic.inspection.inspector import inspect_tool_result


class CharacterCodec:
    def encode(self, text: str) -> list[int]:
        return list(text.encode())

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode()


class FakeSpan:
    def __init__(self, attributes: dict[str, object]) -> None:
        self.attributes = dict(attributes)
        self.status = None

    def __enter__(self) -> FakeSpan:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_status(self, status: object) -> None:
        self.status = status


class FakeTracer:
    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []

    def start_as_current_span(
        self,
        name: str,
        *,
        attributes: dict[str, object],
        **_: object,
    ) -> FakeSpan:
        assert name == "tool_result.inspection"
        span = FakeSpan(attributes)
        self.spans.append(span)
        return span


class FakeClient:
    def __init__(self, response: object) -> None:
        self.response = response

    async def classify(self, _request: object, *, deadline: float | None = None) -> object:
        del deadline
        return self.response


async def no_sleep(_: float) -> None:
    pass


@pytest.mark.asyncio
async def test_benign_span_contains_only_controlled_metadata() -> None:
    tracer = FakeTracer()

    await inspect_tool_result(
        FakeClient({"injectionDetected": False, "category": "none"}),
        tool_name="get_pods",
        result_type="result",
        value="SECRET-RESULT-CONTENT",
        tool_call_id="opaque-call-id-1",
        codec=CharacterCodec(),
        context_window_tokens=640,
        instruction_tokens=20,
        output_tokens=20,
        tracer=tracer,
        provider="anthropic",
        model="model-name",
        sleep=no_sleep,
    )

    attrs = tracer.spans[0].attributes
    assert attrs["inspection.outcome"] == "benign"
    assert attrs["inspection.runtime"] == "deepagents"
    assert attrs["inspection.result_type"] == "result"
    assert attrs["llm.provider"] == "anthropic"
    assert attrs["llm.model"] == "model-name"
    assert attrs["gen_ai.tool.call.id"] == "opaque-call-id-1"
    assert "SECRET-RESULT-CONTENT" not in repr(attrs)


@pytest.mark.asyncio
async def test_malicious_span_has_category_but_no_content(caplog: pytest.LogCaptureFixture) -> None:
    tracer = FakeTracer()
    caplog.set_level(logging.WARNING)

    result = await inspect_tool_result(
        FakeClient({"injectionDetected": True, "category": "unknown"}),
        tool_name="get_pods",
        result_type="error",
        value="DO-NOT-LOG-THIS",
        codec=CharacterCodec(),
        context_window_tokens=640,
        instruction_tokens=20,
        output_tokens=20,
        tracer=tracer,
        sleep=no_sleep,
    )

    assert result.passed is False
    assert tracer.spans[0].attributes["inspection.outcome"] == "malicious"
    assert tracer.spans[0].attributes["inspection.category"] == "unknown"
    assert "gen_ai.tool.call.id" not in tracer.spans[0].attributes
    assert "DO-NOT-LOG-THIS" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("malicious", [False, True])
async def test_real_inspection_decisions_are_unset(
    span_exporter,
    malicious: bool,
) -> None:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    category = "instruction_override" if malicious else "none"
    result = await inspect_tool_result(
        FakeClient({"injectionDetected": malicious, "category": category}),
        tool_name="execute",
        result_type="result",
        value="RAW-INSPECTION-SECRET",
        tool_call_id="inspected-call",
        codec=CharacterCodec(),
        context_window_tokens=640,
        instruction_tokens=20,
        output_tokens=20,
        correlation_attributes={
            "agenticrun.uid": " run-inspected ",
            "agenticrun.phase": "execution",
            "unrelated.attribute": "not-copied",
        },
        tracer=trace.get_tracer("test-inspection"),
        sleep=no_sleep,
    )

    span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "tool_result.inspection"
    )
    attributes = dict(span.attributes)
    assert result.passed is (not malicious)
    assert result.category == category
    assert span.status.status_code == StatusCode.UNSET
    assert attributes["inspection.outcome"] == ("malicious" if malicious else "benign")
    assert attributes["gen_ai.tool.call.id"] == "inspected-call"
    assert attributes["agenticrun.uid"] == " run-inspected "
    assert attributes["agenticrun.phase"] == "execution"
    assert "unrelated.attribute" not in attributes
    assert "error.type" not in attributes
    assert "RAW-INSPECTION-SECRET" not in repr(attributes)


@pytest.mark.asyncio
async def test_inspection_span_does_not_fallback_to_environment_resource_or_parent(
    span_exporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "environment-uid")
    monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_STEP", "environment-phase")
    provider = TracerProvider(
        resource=Resource.create(
            {
                "agenticrun.uid": "resource-uid",
                "agenticrun.phase": "resource-phase",
            }
        )
    )
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    tracer = provider.get_tracer("test-inspection")

    with tracer.start_as_current_span(
        "parent",
        attributes={
            "agenticrun.uid": "parent-uid",
            "agenticrun.phase": "parent-phase",
        },
    ):
        await inspect_tool_result(
            FakeClient({"injectionDetected": False, "category": "none"}),
            tool_name="execute",
            result_type="result",
            value="INSPECTED-CONTENT",
            codec=CharacterCodec(),
            context_window_tokens=640,
            instruction_tokens=20,
            output_tokens=20,
            correlation_attributes={
                "agenticrun.uid": "",
                "agenticrun.phase": "",
            },
            tracer=tracer,
            sleep=no_sleep,
        )

    span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "tool_result.inspection"
    )
    attributes = dict(span.attributes)
    assert span.parent is not None
    assert span.resource.attributes["agenticrun.uid"] == "resource-uid"
    assert span.resource.attributes["agenticrun.phase"] == "resource-phase"
    assert "agenticrun.uid" not in attributes
    assert "agenticrun.phase" not in attributes


@pytest.mark.asyncio
async def test_classifier_error_span_does_not_export_raw_exception(span_exporter) -> None:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    class FailingClient:
        async def classify(self, _request: object, *, deadline: float | None = None) -> object:
            del deadline
            raise RuntimeError("CLASSIFIER-RAW-OUTPUT")

    with pytest.raises(InspectionError):
        await inspect_tool_result(
            FailingClient(),
            tool_name="get_pods",
            result_type="result",
            value="TOOL-RESULT-SECRET",
            codec=CharacterCodec(),
            context_window_tokens=640,
            instruction_tokens=20,
            output_tokens=20,
            tracer=trace.get_tracer("test-inspection"),
            sleep=no_sleep,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes)
    assert span.status.status_code == StatusCode.ERROR
    assert attributes["inspection.outcome"] == "classifier_error"
    assert attributes["inspection.failure_type"] == "provider_error"
    assert attributes["error.type"] == "provider_error"
    assert "CLASSIFIER-RAW-OUTPUT" not in (span.status.description or "")
    assert "CLASSIFIER-RAW-OUTPUT" not in repr(attributes)
    assert "TOOL-RESULT-SECRET" not in repr(attributes)
    assert not span.events


@pytest.mark.asyncio
async def test_classifier_error_telemetry_is_controlled(caplog: pytest.LogCaptureFixture) -> None:
    tracer = FakeTracer()
    caplog.set_level(logging.WARNING)

    class FailingClient:
        async def classify(self, _request: object, *, deadline: float | None = None) -> object:
            del deadline
            raise RuntimeError("CLASSIFIER-RAW-OUTPUT")

    with pytest.raises(InspectionError, match="failed the safety inspection"):
        await inspect_tool_result(
            FailingClient(),
            tool_name="get_pods",
            result_type="result",
            value="TOOL-RESULT-SECRET",
            codec=CharacterCodec(),
            context_window_tokens=640,
            instruction_tokens=20,
            output_tokens=20,
            tracer=tracer,
            sleep=no_sleep,
        )

    assert tracer.spans[0].attributes["inspection.outcome"] == "classifier_error"
    assert tracer.spans[0].attributes["inspection.failure_type"] == "provider_error"
    assert "inspection.provider_status_code" not in tracer.spans[0].attributes
    assert "inspection.provider_error_type" not in tracer.spans[0].attributes
    assert "CLASSIFIER-RAW-OUTPUT" not in caplog.text
    assert "TOOL-RESULT-SECRET" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "issue"),
    [
        ('{"flagged":false}{"flagged":false}', "non_json"),
        ('{"unrecognized":"RAW-CLASSIFIER-SECRET"}', "schema_mismatch"),
        ([], "missing_text"),
    ],
)
async def test_invalid_response_records_only_bounded_shape(
    response: object, issue: str, caplog: pytest.LogCaptureFixture
) -> None:
    from lightspeed_agentic.inspection.client import LangChainClassifierClient

    class Model:
        async def ainvoke(self, _messages: object, **_kwargs: object) -> object:
            return response

    tracer = FakeTracer()
    caplog.set_level(logging.WARNING)
    with pytest.raises(InspectionError) as error:
        await inspect_tool_result(
            LangChainClassifierClient(Model()),
            tool_name="execute",
            result_type="result",
            value="TOOL-RESULT-SECRET",
            codec=CharacterCodec(),
            context_window_tokens=640,
            instruction_tokens=20,
            output_tokens=20,
            tracer=tracer,
            sleep=no_sleep,
        )

    assert error.value.failure_type == "invalid_response"
    assert tracer.spans[0].attributes["inspection.response_issue"] == issue
    assert "RAW-CLASSIFIER-SECRET" not in caplog.text
    assert "TOOL-RESULT-SECRET" not in caplog.text
    assert "RAW-CLASSIFIER-SECRET" not in repr(tracer.spans[0].attributes)


@pytest.mark.asyncio
async def test_provider_error_telemetry_contains_safe_http_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tracer = FakeTracer()
    caplog.set_level(logging.WARNING)

    class ProviderError(RuntimeError):
        status_code: ClassVar[int] = 400
        body: ClassVar[dict[str, object]] = {
            "error": {
                "type": "invalid_request_error",
                "message": "request message contains sensitive data",
            }
        }

    class FailingClient:
        async def classify(self, _request: object, *, deadline: float | None = None) -> object:
            del deadline
            raise ProviderError("raw exception")

    with pytest.raises(InspectionError):
        await inspect_tool_result(
            FailingClient(),
            tool_name="execute",
            result_type="result",
            value="TOOL-RESULT-SECRET",
            codec=CharacterCodec(),
            context_window_tokens=640,
            instruction_tokens=20,
            output_tokens=20,
            tracer=tracer,
            sleep=no_sleep,
        )

    assert tracer.spans[0].attributes["inspection.provider_status_code"] == 400
    assert tracer.spans[0].attributes["inspection.provider_error_type"] == "invalid_request_error"
    assert tracer.spans[0].attributes["inspection.provider_error_reason"] == "message_invalid"
    assert "CLASSIFIER-RAW-OUTPUT" not in caplog.text
    assert "raw exception" not in caplog.text
    assert "TOOL-RESULT-SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_classifier_cancellation_errors_span_without_payload_and_propagates(
    span_exporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import asyncio

    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    caplog.set_level(logging.WARNING)
    cancellation = asyncio.CancelledError("CLASSIFIER-CANCEL-SECRET")

    class CancelledClient:
        async def classify(self, _request: object, *, deadline: float | None = None) -> object:
            del deadline
            raise cancellation

    with pytest.raises(asyncio.CancelledError) as error:
        await inspect_tool_result(
            CancelledClient(),
            tool_name="execute",
            result_type="result",
            value="TOOL-RESULT-SECRET",
            tool_call_id="cancelled-call-id",
            codec=CharacterCodec(),
            context_window_tokens=640,
            instruction_tokens=20,
            output_tokens=20,
            tracer=trace.get_tracer("test-inspection"),
            sleep=no_sleep,
        )

    assert error.value is cancellation
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes)
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description is None
    assert attributes["inspection.outcome"] == "classifier_error"
    assert attributes["inspection.failure_type"] == "cancelled"
    assert attributes["error.type"] == "cancelled"
    assert "inspection.attempt_count" not in attributes
    assert not span.events
    for secret in ("CLASSIFIER-CANCEL-SECRET", "TOOL-RESULT-SECRET"):
        assert secret not in repr(attributes)
        assert secret not in repr(span.events)
        assert secret not in caplog.text
