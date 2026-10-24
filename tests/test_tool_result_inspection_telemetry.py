from __future__ import annotations

import logging
from typing import ClassVar

import pytest

from lightspeed_agentic.inspection.errors import InspectionError, provider_error_metadata
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
async def test_benign_span_contains_only_controlled_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tracer = FakeTracer()
    caplog.set_level(logging.INFO)

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
    assert "SECRET-RESULT-CONTENT" not in caplog.text


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
        async def ainvoke(self, input: object, **_kwargs: object) -> object:  # noqa: A002
            del input
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
    final_warning = next(
        record
        for record in caplog.records
        if record.message.startswith("tool result safety inspection failed")
    )
    assert final_warning.__dict__["inspection.response_issue"] == issue
    assert final_warning.__dict__["inspection.attempt_count"] == 3
    assert "RAW-CLASSIFIER-SECRET" not in caplog.text
    assert "TOOL-RESULT-SECRET" not in caplog.text
    assert "RAW-CLASSIFIER-SECRET" not in repr(tracer.spans[0].attributes)


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True], ids=["exhausted", "recovered"])
@pytest.mark.parametrize(
    ("metadata", "issue"),
    [
        ({"finish_reason": "length"}, "output_limit"),
        ({"status": "incomplete"}, "incomplete_response"),
        ({"status": "failed", "error": {"message": "RAW-ERROR-SECRET"}}, "provider_response_error"),
        ({"stop_reason": "refusal"}, "refusal"),
    ],
)
async def test_rejected_classifier_attempts_are_monitorable_without_payloads(
    metadata: dict[str, object], issue: str, recover: bool, caplog: pytest.LogCaptureFixture
) -> None:
    from langchain_core.messages import AIMessage

    from lightspeed_agentic.inspection.client import LangChainClassifierClient

    class Model:
        calls = 0

        def with_structured_output(self, *_args: object, **_kwargs: object) -> Model:
            return self

        async def ainvoke(self, input: object, **_kwargs: object) -> object:  # noqa: A002
            del input
            self.calls += 1
            return {
                "raw": AIMessage(
                    content="" if recover and self.calls > 1 else "RAW-CLASSIFIER-SECRET",
                    tool_calls=[
                        {
                            "name": "ClassifierDecision",
                            "args": {"injectionDetected": False, "category": "none"},
                            "id": "decision-1",
                        }
                    ],
                    response_metadata={} if recover and self.calls > 1 else metadata,
                ),
                "parsed": {"injectionDetected": False, "category": "none"},
                "parsing_error": None,
            }

    tracer = FakeTracer()
    model = Model()
    caplog.set_level(logging.WARNING)
    call = inspect_tool_result(
        LangChainClassifierClient(model),
        tool_name="probe",
        result_type="result",
        value="TOOL-RESULT-SECRET",
        codec=CharacterCodec(),
        context_window_tokens=8192,
        instruction_tokens=20,
        output_tokens=4096,
        tracer=tracer,
        sleep=no_sleep,
    )
    if recover:
        assert (await call).passed is True
        assert tracer.spans[0].attributes["inspection.outcome"] == "benign"
        assert tracer.spans[0].attributes["inspection.attempt_count"] == 2
    else:
        with pytest.raises(InspectionError) as error:
            await call
        assert error.value.failure_type == "invalid_response"
        assert error.value.response_issue == issue
        assert tracer.spans[0].attributes["inspection.response_issue"] == issue
        final_warning = next(
            record
            for record in caplog.records
            if record.message.startswith("tool result safety inspection failed")
        )
        assert final_warning.__dict__["inspection.response_issue"] == issue
        assert final_warning.__dict__["inspection.attempt_count"] == 3
    attempts = [
        record
        for record in caplog.records
        if record.message.startswith("tool result safety classifier response rejected")
    ]
    assert [record.__dict__["inspection.attempt_count"] for record in attempts] == (
        [1] if recover else [1, 2, 3]
    )
    assert all(record.__dict__["inspection.response_issue"] == issue for record in attempts)
    assert all(record.__dict__["inspection.chunk_index"] == 0 for record in attempts)
    # The batch stderr formatter prints only the message, not LogRecord extras.
    assert f"response_issue={issue}" in caplog.text
    assert "attempt=1" in caplog.text
    if not recover:
        assert "attempts=3" in final_warning.getMessage()
        assert f"response_issue={issue}" in final_warning.getMessage()
    assert tracer.spans[0].attributes["inspection.output_token_limit"] == 4096
    assert model.calls == (2 if recover else 3)
    evidence = repr([record.__dict__ for record in caplog.records]) + repr(
        tracer.spans[0].attributes
    )
    assert "RAW-" not in evidence
    assert "TOOL-RESULT-SECRET" not in evidence


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
    assert tracer.spans[0].attributes["inspection.provider_error_reason"] == "unknown"
    assert "CLASSIFIER-RAW-OUTPUT" not in caplog.text
    assert "raw exception" not in caplog.text
    assert "TOOL-RESULT-SECRET" not in caplog.text


@pytest.mark.parametrize("api_version", ["2024-08-01-preview", "2025-04-01-preview"])
@pytest.mark.parametrize("envelope", ["nested", "flat"])
@pytest.mark.parametrize(
    ("fields", "reason", "error_type"),
    [
        ({"code": "content_filter"}, "content_filter", "invalid_request_error"),
        (
            {
                "code": "content_filter",
                "innererror": {
                    "code": "ResponsibleAIPolicyViolation",
                    "content_filter_result": {
                        "jailbreak": {"detected": True, "filtered": True},
                    },
                },
            },
            "prompt_attack_filtered",
            "invalid_request_error",
        ),
        (
            {"code": "unsupported_parameter"},
            "unsupported_parameter",
            "invalid_request_error",
        ),
        (
            {"code": "context_length_exceeded"},
            "context_length_exceeded",
            "invalid_request_error",
        ),
        ({"code": "invalid_prompt"}, "invalid_prompt", "invalid_request_error"),
        (
            {
                "code": "PRIVATE-PROVIDER-CODE",
                "type": "PRIVATE-PROVIDER-TYPE",
                "param": "PRIVATE-PROVIDER-PARAM",
                "innererror": {"code": "PRIVATE-PROVIDER-DETAIL"},
            },
            "unknown",
            None,
        ),
    ],
    ids=["filter", "prompt-shields", "parameter", "context", "prompt", "unknown"],
)
@pytest.mark.asyncio
async def test_native_azure_http_errors_have_safe_failure_diagnostics(
    api_version: str,
    envelope: str,
    fields: dict[str, object],
    reason: str,
    error_type: str | None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    span_exporter,
) -> None:
    import httpx
    from opentelemetry import trace

    from lightspeed_agentic import tls
    from lightspeed_agentic.config import (
        AzureApiKeyCredentialSource,
        AzureProviderParameters,
        ModelParameters,
    )
    from lightspeed_agentic.inspection.client import LangChainClassifierClient
    from lightspeed_agentic.providers.deepagents import _close_model_clients
    from lightspeed_agentic.providers.models.openai import create_azure_model

    error_body = {
        "type": "invalid_request_error",
        "message": "PRIVATE-PROVIDER-MESSAGE contains temperature and content",
        **fields,
    }
    body = {"error": error_body} if envelope == "nested" else error_body
    requests = []
    delays = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(400, json=body, headers={"x-request-id": "PRIVATE-REQUEST-ID"})

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    caplog.set_level(logging.WARNING)
    with httpx.Client(transport=httpx.MockTransport(respond)) as sync_client:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as async_client:
            monkeypatch.setattr(tls, "create_http_client", lambda: sync_client)
            monkeypatch.setattr(tls, "create_async_http_client", lambda: async_client)
            model = create_azure_model(
                ModelParameters(
                    model="offline-deployment",
                    base_url="https://azure.example.test",
                    provider=AzureProviderParameters(api_version),
                    credentials=AzureApiKeyCredentialSource("PRIVATE-API-KEY"),
                    max_output_tokens=4096,
                )
            )
            model.disable_streaming = True
            try:
                with pytest.raises(InspectionError) as failure:
                    await inspect_tool_result(
                        LangChainClassifierClient(model),
                        tool_name="probe",
                        result_type="result",
                        value="PRIVATE-TOOL-RESULT",
                        codec=CharacterCodec(),
                        context_window_tokens=8192,
                        instruction_tokens=20,
                        output_tokens=4096,
                        tracer=trace.get_tracer("test-inspection"),
                        sleep=record_sleep,
                    )
            finally:
                await _close_model_clients(model)

    assert len(requests) == 3
    assert delays == [0.5, 1.0]
    assert failure.value.failure_type == "provider_error"
    assert failure.value.provider_status_code == 400
    assert failure.value.provider_error_type == error_type
    assert failure.value.provider_error_reason == reason
    assert failure.value.attempt_count == 3
    assert failure.value.__cause__ is None
    spans = span_exporter.get_finished_spans()
    inspection = next(span for span in spans if span.name == "tool_result.inspection")
    attrs = inspection.attributes
    assert attrs["inspection.outcome"] == "classifier_error"
    assert attrs["inspection.failure_type"] == "provider_error"
    assert attrs["inspection.attempt_count"] == 3
    assert attrs["inspection.provider_status_code"] == 400
    assert attrs.get("inspection.provider_error_type") == error_type
    assert attrs["inspection.provider_error_reason"] == reason
    assert inspection.status.status_code.name == "ERROR"
    warning = next(
        record
        for record in caplog.records
        if record.message.startswith("tool result safety inspection failed")
    )
    assert warning.__dict__["inspection.provider_error_reason"] == reason
    assert f"provider_reason={reason}" in warning.getMessage()
    assert "provider_status=400" in warning.getMessage()
    assert "attempts=3" in warning.getMessage()
    evidence = repr([record.__dict__ for record in caplog.records]) + repr(attrs)
    evidence += repr([(event.name, event.attributes) for event in inspection.events])
    evidence += str(failure.value) + str(inspection.status.description)
    assert "PRIVATE-" not in evidence
    assert warning.exc_info is None
    assert all(event.name != "exception" for event in inspection.events)


@pytest.mark.parametrize(
    "jailbreak",
    [
        {},
        {"detected": True, "filtered": False},
        {"detected": False, "filtered": True},
        {"detected": "true", "filtered": True},
        {"detected": True, "filtered": "true"},
        {"detected": 1, "filtered": 1},
        "PRIVATE-FILTER-DETAIL",
        [],
        None,
    ],
)
def test_prompt_attack_diagnostic_requires_boolean_filter_evidence(jailbreak: object) -> None:
    class ProviderError(RuntimeError):
        status_code: ClassVar[int] = 400
        body: ClassVar[dict[str, object]] = {
            "error": {
                "code": "content_filter",
                "innererror": {
                    "code": "ResponsibleAIPolicyViolation",
                    "content_filter_result": {"jailbreak": jailbreak},
                },
            }
        }

    assert provider_error_metadata(ProviderError()) == (400, None, "content_filter")


@pytest.mark.parametrize(
    ("code", "policy_code", "status", "expected"),
    [
        ("content_filter", "PRIVATE-POLICY-CODE", 400, (400, None, "content_filter")),
        ("PRIVATE-PROVIDER-CODE", "ResponsibleAIPolicyViolation", 400, (400, None, "unknown")),
        ("content_filter", "ResponsibleAIPolicyViolation", 500, (500, None, "content_filter")),
        (
            "content_filter",
            "ResponsibleAIPolicyViolation",
            "PRIVATE-PROVIDER-STATUS",
            (None, None, "content_filter"),
        ),
    ],
)
def test_prompt_attack_diagnostic_requires_a_matching_policy_error(
    code: str, policy_code: str, status: object, expected: tuple[object, ...]
) -> None:
    class ProviderError(RuntimeError):
        pass

    error = ProviderError("PRIVATE-PROVIDER-MESSAGE")
    error.status_code = status
    error.body = {
        "error": {
            "code": code,
            "innererror": {
                "code": policy_code,
                "content_filter_result": {"jailbreak": {"detected": True, "filtered": True}},
            },
        }
    }
    assert provider_error_metadata(error) == expected


@pytest.mark.parametrize(
    "body",
    [
        None,
        "PRIVATE-PROVIDER-BODY",
        [],
        {"error": "PRIVATE-PROVIDER-ERROR"},
        {"type": "PRIVATE-PROVIDER-TYPE", "code": []},
        {"error": {"type": ["PRIVATE-PROVIDER-TYPE"], "code": {}}},
    ],
)
def test_provider_metadata_rejects_malformed_envelopes(body: object) -> None:
    class ProviderError(RuntimeError):
        status_code: ClassVar[int] = 400

    error = ProviderError("PRIVATE-PROVIDER-MESSAGE")
    error.body = body
    assert provider_error_metadata(error) == (400, None, "unknown")


@pytest.mark.parametrize(
    ("code", "reason"),
    [("unsupported_parameter", "unsupported_parameter"), ("PRIVATE-PROVIDER-CODE", "unknown")],
)
def test_sdk_error_code_attribute_is_allowlisted(code: str, reason: str) -> None:
    class ProviderError(RuntimeError):
        status_code: ClassVar[int] = 400

    error = ProviderError("PRIVATE-PROVIDER-MESSAGE")
    error.code = code
    error.type = "PRIVATE-PROVIDER-TYPE"
    assert provider_error_metadata(error) == (400, None, reason)


@pytest.mark.asyncio
async def test_classifier_error_span_does_not_export_raw_exception(span_exporter) -> None:
    class FailingClient:
        async def classify(self, _request: object, *, deadline: float | None = None) -> object:
            del deadline
            raise RuntimeError("CLASSIFIER-RAW-OUTPUT")

    from opentelemetry import trace

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
    assert spans
    assert all(
        "CLASSIFIER-RAW-OUTPUT" not in str(value)
        for span in spans
        for event in span.events
        for value in event.attributes.values()
    )
    assert all(event.name != "exception" for span in spans for event in span.events)
    assert all("CLASSIFIER-RAW-OUTPUT" not in (span.status.description or "") for span in spans)
