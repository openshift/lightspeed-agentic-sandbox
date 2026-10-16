from __future__ import annotations

from typing import ClassVar

import pytest

from lightspeed_agentic.inspection.chunking import TokenCodec
from lightspeed_agentic.inspection.inspector import (
    CLASSIFIER_FAILURE_MESSAGE,
    ClassifierClient,
    InspectionError,
    inspect_tool_result,
)


class CharacterCodec(TokenCodec):
    def encode(self, text: str) -> list[int]:
        return list(text.encode())

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode()


class FakeClient(ClassifierClient):
    def __init__(self, responses: list[object]) -> None:
        self.responses = iter(responses)
        self.requests = []

    async def classify(self, request: object, *, deadline: float | None = None) -> object:
        del deadline
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


async def no_sleep(_: float) -> None:
    pass


def _decision(*, malicious: bool = False, category: str = "none") -> dict[str, object]:
    return {"injectionDetected": malicious, "category": category}


@pytest.mark.asyncio
async def test_inspects_all_chunks_in_source_order() -> None:
    client = FakeClient([_decision(), _decision(), _decision()])

    result = await inspect_tool_result(
        client,
        tool_name="get_pod_logs",
        result_type="result",
        value="x" * 1100,
        codec=CharacterCodec(),
        context_window_tokens=640,
        instruction_tokens=20,
        output_tokens=20,
        sleep=no_sleep,
    )

    assert result.passed is True
    assert [request.chunk_index for request in client.requests] == [0, 1, 2]
    assert all(request.tool_name == "get_pod_logs" for request in client.requests)
    assert all(request.result_type == "result" for request in client.requests)
    assert all(
        set(request.model_dump(by_alias=True))
        == {"toolName", "resultType", "chunkIndex", "chunkCount", "content"}
        for request in client.requests
    )


@pytest.mark.asyncio
async def test_malicious_first_chunk_stops_before_later_chunks() -> None:
    client = FakeClient([_decision(malicious=True, category="instruction_override")])

    result = await inspect_tool_result(
        client,
        tool_name="execute_command",
        result_type="error",
        value="malicious content " * 100,
        codec=CharacterCodec(),
        context_window_tokens=640,
        instruction_tokens=20,
        output_tokens=20,
        sleep=no_sleep,
    )

    assert result.passed is False
    assert result.category == "instruction_override"
    assert len(client.requests) == 1


@pytest.mark.parametrize("malicious_index", [1, 2])
@pytest.mark.asyncio
async def test_malicious_middle_or_last_chunk_stops_at_that_chunk(malicious_index: int) -> None:
    responses = [_decision() for _ in range(malicious_index)]
    responses.append(_decision(malicious=True, category="unknown"))
    client = FakeClient(responses)

    result = await inspect_tool_result(
        client,
        tool_name="get_pods",
        result_type="result",
        value="x" * 1100,
        codec=CharacterCodec(),
        context_window_tokens=640,
        instruction_tokens=20,
        output_tokens=20,
        sleep=no_sleep,
    )

    assert result.passed is False
    assert result.category == "unknown"
    assert [request.chunk_index for request in client.requests] == list(range(malicious_index + 1))


@pytest.mark.asyncio
async def test_classifier_failures_retry_three_total_attempts_with_exact_delays() -> None:
    client = FakeClient([RuntimeError("provider secret"), TimeoutError(), ValueError("raw output")])
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    with pytest.raises(InspectionError) as error:
        await inspect_tool_result(
            client,
            tool_name="get_secret",
            result_type="result",
            value="sensitive tool result",
            codec=CharacterCodec(),
            context_window_tokens=640,
            instruction_tokens=20,
            output_tokens=20,
            sleep=record_sleep,
        )

    assert len(client.requests) == 3
    assert delays == [0.5, 1.0]
    assert str(error.value) == CLASSIFIER_FAILURE_MESSAGE
    assert error.value.__cause__ is None
    assert "provider secret" not in repr(error.value)
    assert "raw output" not in repr(error.value)
    assert "sensitive" not in str(error.value)
    assert "secret" not in str(error.value)
    assert "raw output" not in str(error.value)


@pytest.mark.asyncio
async def test_malformed_classifier_reply_fails_closed_as_invalid_response() -> None:
    from lightspeed_agentic.inspection.client import LangChainClassifierClient

    class MalformedModel:
        calls = 0

        async def ainvoke(self, input: object, **_kwargs: object) -> str:  # noqa: A002
            del input
            self.calls += 1
            return '[]{{"flagged":false}"flagged": false}[]'

    model = MalformedModel()
    with pytest.raises(InspectionError) as error:
        await inspect_tool_result(
            LangChainClassifierClient(model),
            tool_name="execute",
            result_type="result",
            value="normal status output",
            codec=CharacterCodec(),
            context_window_tokens=640,
            instruction_tokens=20,
            output_tokens=20,
            sleep=no_sleep,
        )

    assert model.calls == 3
    assert error.value.failure_type == "invalid_response"
    assert str(error.value) == CLASSIFIER_FAILURE_MESSAGE
    assert "flagged" not in repr(error.value)


@pytest.mark.asyncio
async def test_provider_error_metadata_is_safe_and_preserved() -> None:
    class ProviderError(RuntimeError):
        status_code: ClassVar[int] = 400
        body: ClassVar[dict[str, object]] = {
            "error": {
                "type": "invalid_request_error",
                "message": "contains classifier content",
            }
        }

    with pytest.raises(InspectionError) as error:
        await inspect_tool_result(
            FakeClient([ProviderError(), ProviderError(), ProviderError()]),
            tool_name="execute",
            result_type="result",
            value="sensitive tool result",
            codec=CharacterCodec(),
            context_window_tokens=640,
            instruction_tokens=20,
            output_tokens=20,
            sleep=no_sleep,
        )

    assert error.value.provider_status_code == 400
    assert error.value.provider_error_type == "invalid_request_error"
    assert error.value.provider_error_reason == "message_invalid"
    assert "contains classifier content" not in repr(error.value)


@pytest.mark.asyncio
async def test_invalid_classifier_response_is_retried() -> None:
    client = FakeClient(
        [
            {"injectionDetected": "false", "category": "none"},
            _decision(),
        ]
    )
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    result = await inspect_tool_result(
        client,
        tool_name="get_pods",
        result_type="result",
        value="normal output",
        codec=CharacterCodec(),
        context_window_tokens=640,
        instruction_tokens=20,
        output_tokens=20,
        sleep=record_sleep,
    )

    assert result.passed is True
    assert len(client.requests) == 2
    assert delays == [0.5]


@pytest.mark.asyncio
async def test_disabled_inspection_makes_no_classifier_call() -> None:
    client = FakeClient([])

    result = await inspect_tool_result(
        client,
        tool_name="get_pods",
        result_type="result",
        value="untrusted output",
        codec=CharacterCodec(),
        context_window_tokens=640,
        instruction_tokens=20,
        output_tokens=20,
        enabled=False,
        sleep=no_sleep,
    )

    assert result.passed is True
    assert result.category == "none"
    assert client.requests == []


@pytest.mark.asyncio
async def test_deadline_prevents_retry_sleep_and_fails_closed() -> None:
    client = FakeClient([TimeoutError()])
    clock = iter([10.0, 10.0, 10.1])
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    with pytest.raises(InspectionError):
        await inspect_tool_result(
            client,
            tool_name="get_pods",
            result_type="result",
            value="output",
            codec=CharacterCodec(),
            context_window_tokens=640,
            instruction_tokens=20,
            output_tokens=20,
            deadline=10.2,
            monotonic=lambda: next(clock),
            sleep=record_sleep,
        )

    assert len(client.requests) == 1
    assert sleeps == []
