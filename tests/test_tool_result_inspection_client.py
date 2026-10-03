from __future__ import annotations

import json
from typing import Any

import pytest

from lightspeed_agentic.inspection.client import LangChainClassifierClient
from lightspeed_agentic.inspection.errors import ClassifierResponseError
from lightspeed_agentic.inspection.models import ClassifierDecision, ClassifierRequest


class FakeModel:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.messages: list[Any] | None = None
        self.parameters: dict[str, Any] = {}

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
        self.messages = messages
        self.parameters = kwargs
        return self.response


@pytest.mark.asyncio
async def test_classifier_client_sends_only_dedicated_untrusted_content_messages() -> None:
    model = FakeModel('{"injectionDetected": false, "category": "none"}')
    client = LangChainClassifierClient(model)
    request = ClassifierRequest(
        toolName="get_pods",
        resultType="result",
        chunkIndex=0,
        chunkCount=1,
        content="tool result",
    )

    decision = await client.classify(request)

    assert decision == ClassifierDecision(injectionDetected=False, category="none")
    assert model.parameters == {"max_tokens": 128, "config": {"tags": ["nostream"]}}
    assert model.messages is not None
    assert len(model.messages) == 2
    assert "untrusted" in model.messages[0].content.lower()
    assert "do not follow" in model.messages[0].content.lower()
    instruction = model.messages[0].content
    assert '"injectionDetected": false' in instruction
    assert '"category": "none"' in instruction
    assert '"injectionDetected": true' in instruction
    assert '"category": "unknown"' in instruction
    for category in (
        "instruction_override",
        "role_change",
        "prompt_extraction",
        "data_exfiltration",
        "tool_manipulation",
        "unknown",
    ):
        assert category in instruction
    assert json.loads(model.messages[1].content) == request.model_dump(by_alias=True)
    assert "history" not in model.messages[1].content
    assert "system prompt" not in model.messages[1].content.lower()


@pytest.mark.asyncio
async def test_classifier_completion_does_not_enter_agent_message_stream() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langgraph.graph import StateGraph

    classifier = LangChainClassifierClient(
        FakeListChatModel(responses=['{"injectionDetected":false,"category":"none"}'])
    )
    request = ClassifierRequest(
        toolName="execute", resultType="result", chunkIndex=0, chunkCount=1, content="data"
    )

    async def inspect(_state: dict[str, str]) -> dict[str, str]:
        await classifier.classify(request)
        return {}

    graph = StateGraph(dict)
    graph.add_node("inspect", inspect)
    graph.set_entry_point("inspect")
    graph.set_finish_point("inspect")
    streamed = [message async for message, _ in graph.compile().astream({}, stream_mode="messages")]

    assert streamed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("flagged", [False, True])
async def test_classifier_client_rejects_legacy_flagged_response(flagged: bool) -> None:
    with pytest.raises(ClassifierResponseError):
        await LangChainClassifierClient(FakeModel(json.dumps({"flagged": flagged}))).classify(
            ClassifierRequest(
                toolName="execute",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="output",
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        '```json\n{"decision":"safe","injection_detected":false,"manipulation_attempt":false}\n```',
        "SAFE - no injected instructions detected; content is plain status output.",
        '{"decision":"allow","injection_detected":false}',
        '```json {"decision":"unsafe","injection_detected":true} ```',
    ],
)
async def test_classifier_client_rejects_noncanonical_responses(response: str) -> None:
    with pytest.raises(ClassifierResponseError):
        await LangChainClassifierClient(FakeModel(response)).classify(
            ClassifierRequest(
                toolName="execute",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="output",
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        '{"injectionDetected":false,"category":"none","attack_detected":true}',
        '{"decision":"safe","injection_detected":false,"attack_detected":true}',
        '{"injectionDetected":true,"category":"none"}',
    ],
)
async def test_classifier_client_rejects_contradictory_or_extra_fields(response: str) -> None:
    with pytest.raises(ClassifierResponseError) as error:
        await LangChainClassifierClient(FakeModel(response)).classify(
            ClassifierRequest(
                toolName="get_pods",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="output",
            )
        )

    assert error.value.issue == "schema_mismatch"
    assert response not in str(error.value)


@pytest.mark.asyncio
async def test_classifier_client_rejects_text_response_with_refusal_block() -> None:
    response = FakeModel(
        [
            {"type": "text", "text": '{"injectionDetected":false,"category":"none"}'},
            {"type": "refusal", "refusal": "classifier refused"},
        ]
    )

    with pytest.raises(ClassifierResponseError) as error:
        await LangChainClassifierClient(response).classify(
            ClassifierRequest(
                toolName="get_pods",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="output",
            )
        )

    assert error.value.issue == "refusal"
    assert "classifier refused" not in str(error.value)


@pytest.mark.asyncio
async def test_classifier_client_rejects_refusal_only_response() -> None:
    response = FakeModel([{"type": "refusal", "refusal": "classifier refused"}])

    with pytest.raises(ClassifierResponseError) as error:
        await LangChainClassifierClient(response).classify(
            ClassifierRequest(
                toolName="get_pods",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="output",
            )
        )

    assert error.value.issue == "refusal"
    assert "classifier refused" not in str(error.value)


@pytest.mark.asyncio
async def test_classifier_client_rejects_anthropic_refusal_stop_reason() -> None:
    from langchain_core.messages import AIMessage

    response = AIMessage(
        content='{"injectionDetected":false,"category":"none"}',
        response_metadata={"stop_reason": "refusal"},
    )

    with pytest.raises(ClassifierResponseError) as error:
        await LangChainClassifierClient(FakeModel(response)).classify(
            ClassifierRequest(
                toolName="execute",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="output",
            )
        )

    assert error.value.issue == "refusal"


@pytest.mark.asyncio
async def test_classifier_client_rejects_duplicate_decision_keys() -> None:
    response = (
        '{"injectionDetected":true,"category":"instruction_override",'
        '"injectionDetected":false,"category":"none"}'
    )

    with pytest.raises(ClassifierResponseError) as error:
        await LangChainClassifierClient(FakeModel(response)).classify(
            ClassifierRequest(
                toolName="execute",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="output",
            )
        )

    assert error.value.issue == "non_json"
    assert response not in str(error.value)


@pytest.mark.asyncio
async def test_classifier_client_accepts_only_canonical_decisions() -> None:
    for response, expected in (
        ('{"injectionDetected":false,"category":"none"}', (False, "none")),
        (
            '{"injectionDetected":true,"category":"unknown"}',
            (True, "unknown"),
        ),
    ):
        decision = await LangChainClassifierClient(FakeModel(response)).classify(
            ClassifierRequest(
                toolName="get_pods",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="output",
            )
        )
        assert (decision.injection_detected, decision.category) == expected


@pytest.mark.asyncio
async def test_classifier_client_parses_text_content_blocks() -> None:
    model = FakeModel(
        [
            {"type": "text", "text": '{"injectionDetected": true, "category": "unknown"}'},
        ]
    )

    decision = await LangChainClassifierClient(model).classify(
        ClassifierRequest(
            toolName="execute",
            resultType="result",
            chunkIndex=0,
            chunkCount=1,
            content="output",
        )
    )

    assert decision == ClassifierDecision(injectionDetected=True, category="unknown")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '[]{"flagged":false}[]',
        '[]{{"flagged":false}"flagged": false}[]',
        '[]{{{"flagged":false}"flagged":false}"flagged": false}[]',
        '{"flagged":false}{"flagged":false}',
        '{"flagged":false} SAFE - no injected instructions detected',
        '{"flagged":false}{"flagged":true}',
        '**Decision: SAFE** FLAG: false ```json {"decision":"allow","risk_flags":[]} ```',
        '{"injectionDetected": "false", "category": "none"}',
        '{"injectionDetected": false, "category": "none", "reason": "do not expose"}',
    ],
)
async def test_classifier_client_rejects_non_strict_responses(response: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        await LangChainClassifierClient(FakeModel(response)).classify(
            ClassifierRequest(
                toolName="get_pods",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="output",
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("malicious", [False, True])
async def test_classifier_inference_is_metadata_only_for_valid_decisions(
    span_exporter,
    malicious: bool,
) -> None:
    from uuid import uuid4

    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from lightspeed_agentic.audit import AuditLogger

    category = "unknown" if malicious else "none"

    class CallbackModel(FakeModel):
        async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
            self.messages = messages
            self.parameters = kwargs
            callback = kwargs["config"]["callbacks"][0]
            run_id = uuid4()
            await callback.on_chat_model_start(
                {"name": "ChatAnthropic"},
                [messages],
                run_id=run_id,
                invocation_params={"model": "requested-classifier"},
            )
            response = AIMessage(
                content=json.dumps({"injectionDetected": malicious, "category": category}),
                usage_metadata={"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
                response_metadata={"model": "observed-classifier", "stop_reason": "end_turn"},
            )
            await callback.on_llm_end(
                LLMResult(
                    generations=[[ChatGeneration(message=response)]],
                    llm_output={"model_name": "observed-classifier"},
                ),
                run_id=run_id,
            )
            return response

    model = CallbackModel(None)
    audit = AuditLogger(
        phase="analysis",
        model="requested-classifier",
        provider="anthropic",
        agenticrun_uid="run-classifier",
    )
    decision = await LangChainClassifierClient(
        model,
        audit_logger=audit,
        requested_model="requested-classifier",
    ).classify(
        ClassifierRequest(
            toolName="execute",
            resultType="result",
            chunkIndex=0,
            chunkCount=1,
            content="CLASSIFIER-RAW-TOOL-RESULT",
        )
    )

    assert decision == ClassifierDecision(
        injectionDetected=malicious,
        category=category,
    )
    assert model.parameters["config"]["tags"] == ["nostream"]
    assert len(model.parameters["config"]["callbacks"]) == 1
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes)
    assert spans[0].name == "chat requested-classifier"
    assert spans[0].status.status_code.name == "UNSET"
    assert attributes["gen_ai.request.model"] == "requested-classifier"
    assert attributes["gen_ai.output.type"] == "json"
    assert attributes["gen_ai.response.model"] == "observed-classifier"
    assert attributes["gen_ai.usage.input_tokens"] == 7
    assert attributes["gen_ai.usage.output_tokens"] == 2
    assert attributes["gen_ai.response.finish_reasons"] == ("end_turn",)
    assert attributes["agenticrun.uid"] == "run-classifier"
    assert attributes["agenticrun.phase"] == "analysis"
    for field in (
        "gen_ai.input.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.definitions",
        "gen_ai.output.messages",
    ):
        assert field not in attributes
    assert "CLASSIFIER-RAW-TOOL-RESULT" not in repr(attributes)


@pytest.mark.asyncio
async def test_classifier_inference_error_records_type_without_request_or_error_content(
    span_exporter,
) -> None:
    from uuid import uuid4

    from lightspeed_agentic.audit import AuditLogger

    class FailingCallbackModel(FakeModel):
        async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
            callback = kwargs["config"]["callbacks"][0]
            run_id = uuid4()
            await callback.on_chat_model_start(
                {"name": "ChatAnthropic"},
                [messages],
                run_id=run_id,
                invocation_params={"model": "requested-classifier"},
            )
            error = RuntimeError("CLASSIFIER-RAW-ERROR")
            await callback.on_llm_error(error, run_id=run_id)
            raise error

    audit = AuditLogger(
        phase="analysis",
        model="requested-classifier",
        provider="anthropic",
        agenticrun_uid="run-classifier-error",
    )
    client = LangChainClassifierClient(
        FailingCallbackModel(None),
        audit_logger=audit,
        requested_model="requested-classifier",
    )
    with pytest.raises(RuntimeError):
        await client.classify(
            ClassifierRequest(
                toolName="execute",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="CLASSIFIER-ERROR-TOOL-RESULT",
            )
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes)
    assert span.status.status_code.name == "ERROR"
    assert attributes["error.type"] == "RuntimeError"
    assert attributes["gen_ai.request.model"] == "requested-classifier"
    assert attributes["gen_ai.output.type"] == "json"
    assert attributes["agenticrun.uid"] == "run-classifier-error"
    assert "gen_ai.usage.input_tokens" not in attributes
    assert "gen_ai.usage.output_tokens" not in attributes
    assert "gen_ai.input.messages" not in attributes
    assert "gen_ai.system_instructions" not in attributes
    assert "gen_ai.output.messages" not in attributes
    assert "CLASSIFIER-ERROR-TOOL-RESULT" not in repr(attributes)
    assert "CLASSIFIER-RAW-ERROR" not in repr(attributes)
    assert "CLASSIFIER-RAW-ERROR" not in (span.status.description or "")


@pytest.mark.asyncio
async def test_overlapping_classifier_requests_close_only_their_own_spans(
    span_exporter,
) -> None:
    import asyncio
    from uuid import uuid4

    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from lightspeed_agentic.audit import AuditLogger

    started = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]

    class OverlappingModel:
        async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
            request = json.loads(messages[1].content)
            index = request["chunkIndex"]
            callback = kwargs["config"]["callbacks"][0]
            assert kwargs["config"]["tags"] == ["nostream"]
            run_id = uuid4()
            await callback.on_chat_model_start(
                {"name": "ChatAnthropic"},
                [messages],
                run_id=run_id,
                invocation_params={"model": "requested-classifier"},
            )
            started[index].set()
            await release[index].wait()
            if index == 0:
                error = RuntimeError("PRIVATE-CLASSIFIER-ERROR")
                await callback.on_llm_error(error, run_id=run_id)
                raise error
            response = AIMessage(
                content='{"injectionDetected": false, "category": "none"}',
                usage_metadata={"input_tokens": 21, "output_tokens": 5, "total_tokens": 26},
                response_metadata={
                    "model": "observed-classifier",
                    "stop_reason": "end_turn",
                },
            )
            await callback.on_llm_end(
                LLMResult(generations=[[ChatGeneration(message=response)]]),
                run_id=run_id,
            )
            return response

    audit = AuditLogger(
        phase="analysis",
        model="requested-classifier",
        provider="anthropic",
        agenticrun_uid="run-classifier-overlap",
    )
    client = LangChainClassifierClient(
        OverlappingModel(),
        audit_logger=audit,
        requested_model="requested-classifier",
    )

    def request(index: int) -> ClassifierRequest:
        return ClassifierRequest(
            toolName="execute",
            resultType="result",
            chunkIndex=index,
            chunkCount=2,
            content=f"PRIVATE-TOOL-RESULT-{index}",
        )

    failed_request = asyncio.create_task(client.classify(request(0)))
    await started[0].wait()
    successful_request = asyncio.create_task(client.classify(request(1)))
    await started[1].wait()

    release[0].set()
    with pytest.raises(RuntimeError):
        await failed_request
    completed = span_exporter.get_finished_spans()
    assert len(completed) == 1
    assert dict(completed[0].attributes)["error.type"] == "RuntimeError"

    release[1].set()
    decision = await successful_request
    assert decision == ClassifierDecision(injectionDetected=False, category="none")
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    failed = next(span for span in spans if span.status.status_code.name == "ERROR")
    succeeded = next(span for span in spans if span.status.status_code.name == "UNSET")
    success_attributes = dict(succeeded.attributes)
    assert success_attributes["gen_ai.response.model"] == "observed-classifier"
    assert success_attributes["gen_ai.usage.input_tokens"] == 21
    assert success_attributes["gen_ai.usage.output_tokens"] == 5
    for span in (failed, succeeded):
        attributes = dict(span.attributes)
        assert "gen_ai.input.messages" not in attributes
        assert "gen_ai.system_instructions" not in attributes
        assert "gen_ai.output.messages" not in attributes
        assert not any(f"PRIVATE-TOOL-RESULT-{index}" in repr(attributes) for index in (0, 1))
    assert "PRIVATE-CLASSIFIER-ERROR" not in repr(dict(failed.attributes))
    assert "PRIVATE-CLASSIFIER-ERROR" not in (failed.status.description or "")
