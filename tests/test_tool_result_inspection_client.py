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


class StructuredFakeModel(FakeModel):
    def with_structured_output(
        self,
        schema: Any,
        *,
        method: str,
        include_raw: bool,
    ) -> FakeModel:
        assert schema is ClassifierDecision
        assert method == "function_calling"
        assert include_raw is True
        return _StructuredBoundModel(self.response, self)


class _StructuredBoundModel:
    def __init__(self, response: Any, parent: FakeModel) -> None:
        self.response = response
        self.parent = parent

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
        self.parent.messages = messages
        self.parent.parameters = kwargs
        return {"raw": None, "parsed": self.response, "parsing_error": None}


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
async def test_classifier_uses_function_calling_when_model_supports_structured_output() -> None:
    decision = await LangChainClassifierClient(
        StructuredFakeModel(ClassifierDecision(injectionDetected=False, category="none"))
    ).classify(
        ClassifierRequest(
            toolName="list_namespaces",
            resultType="result",
            chunkIndex=0,
            chunkCount=1,
            content="Namespaces: default",
        )
    )

    assert decision == ClassifierDecision(injectionDetected=False, category="none")


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
