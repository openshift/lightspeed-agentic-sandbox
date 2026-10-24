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

    async def ainvoke(self, input: list[Any], **kwargs: Any) -> Any:  # noqa: A002
        self.messages = input
        self.parameters = kwargs
        return self.response


class StructuredEnvelopeFakeModel(FakeModel):
    def with_structured_output(self, *_args: Any, **_kwargs: Any) -> FakeModel:
        return self


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

    async def ainvoke(self, input: list[Any], **kwargs: Any) -> Any:  # noqa: A002
        self.parent.messages = input
        self.parent.parameters = kwargs
        from langchain_core.messages import AIMessage

        return {
            "raw": AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ClassifierDecision",
                        "args": self.response.model_dump(by_alias=True),
                        "id": "decision-1",
                    }
                ],
                response_metadata={"finish_reason": "tool_calls"},
            ),
            "parsed": self.response,
            "parsing_error": None,
        }


class StructuredErrorFakeModel(FakeModel):
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
        return _StructuredErrorBoundModel(self)


class _StructuredErrorBoundModel:
    def __init__(self, parent: FakeModel) -> None:
        self.parent = parent

    async def ainvoke(self, input: list[Any], **kwargs: Any) -> Any:  # noqa: A002
        self.parent.messages = input
        self.parent.parameters = kwargs
        from langchain_core.messages import AIMessage

        return {
            "raw": AIMessage(content=""),
            "parsed": ClassifierDecision(injectionDetected=False, category="none"),
            "parsing_error": ValueError("invalid structured response"),
        }


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
    assert model.parameters == {"config": {"tags": ["nostream"]}}
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
@pytest.mark.parametrize("parsed", [False, True], ids=["text", "parsed"])
@pytest.mark.parametrize(
    ("overrides", "issue"),
    [
        ({"response_metadata": {"finish_reason": "length"}}, "output_limit"),
        ({"response_metadata": {"finish_reason": "MAX_TOKENS"}}, "output_limit"),
        ({"response_metadata": {"stop_reason": "max_tokens"}}, "output_limit"),
        ({"response_metadata": {"stop_reason": "model_context_window_exceeded"}}, "output_limit"),
        (
            {
                "response_metadata": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                }
            },
            "output_limit",
        ),
        (
            {
                "response_metadata": {
                    "status": "completed",
                    "incomplete_details": {"reason": "max_output_tokens"},
                }
            },
            "output_limit",
        ),
        ({"response_metadata": {"status": "incomplete"}}, "incomplete_response"),
        ({"response_metadata": {"status": "in_progress"}}, "incomplete_response"),
        ({"response_metadata": {"status": "queued"}}, "incomplete_response"),
        ({"response_metadata": {"status": "failed"}}, "provider_response_error"),
        ({"response_metadata": {"status": "cancelled"}}, "provider_response_error"),
        (
            {"response_metadata": {"error": {"message": "RAW-PROVIDER-SECRET"}}},
            "provider_response_error",
        ),
        ({"response_metadata": {"stop_reason": "refusal"}}, "refusal"),
        ({"response_metadata": {"finish_reason": "content_filter"}}, "refusal"),
        ({"response_metadata": {"finish_reason": "SAFETY"}}, "refusal"),
        ({"response_metadata": {"finish_reason": "RAW-UNKNOWN-STATUS"}}, "invalid_envelope"),
        ({"additional_kwargs": {"refusal": "RAW-REFUSAL-SECRET"}}, "refusal"),
        ({"content": [{"type": "refusal", "refusal": "RAW-REFUSAL-SECRET"}]}, "refusal"),
        (
            {
                "content": [
                    {
                        "type": "function_call",
                        "status": "incomplete",
                        "arguments": "RAW-PROVIDER-SECRET",
                    }
                ]
            },
            "incomplete_response",
        ),
        (
            {
                "invalid_tool_calls": [
                    {
                        "name": "ClassifierDecision",
                        "args": "{",
                        "id": "bad-call",
                        "error": "RAW-PARSER-SECRET",
                    }
                ]
            },
            "invalid_envelope",
        ),
    ],
)
async def test_classifier_checks_raw_safety_metadata_even_for_parsed_decisions(
    parsed: bool, overrides: dict[str, Any], issue: str
) -> None:
    from langchain_core.messages import AIMessage

    decision = {"injectionDetected": False, "category": "none"}
    raw = AIMessage(**{"content": json.dumps(decision), **overrides})
    model = StructuredEnvelopeFakeModel(
        {"raw": raw, "parsed": decision if parsed else None, "parsing_error": None}
    )
    with pytest.raises(ClassifierResponseError) as error:
        await LangChainClassifierClient(model).classify(
            ClassifierRequest(
                toolName="probe",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="TOOL-SECRET",
            )
        )
    assert error.value.issue == issue
    assert "RAW-" not in str(error.value)
    assert "TOOL-SECRET" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"finish_reason": "stop"},
        {"finish_reason": "tool_calls"},
        {"finish_reason": "function_call"},
        {"finish_reason": "STOP"},
        {"stop_reason": "tool_use"},
        {"stop_reason": "end_turn"},
        {"stop_reason": "stop_sequence"},
        {"status": "completed"},
    ],
)
async def test_classifier_preserves_completed_native_function_decisions(
    metadata: dict[str, Any],
) -> None:
    from langchain_core.messages import AIMessage

    decision = {"injectionDetected": False, "category": "none"}
    raw = AIMessage(
        content="",
        tool_calls=[{"name": "ClassifierDecision", "args": decision, "id": "decision-1"}],
        response_metadata=metadata,
    )
    result = await LangChainClassifierClient(
        StructuredEnvelopeFakeModel({"raw": raw, "parsed": decision, "parsing_error": None})
    ).classify(
        ClassifierRequest(
            toolName="probe", resultType="result", chunkIndex=0, chunkCount=1, content="safe"
        )
    )
    assert result == ClassifierDecision(injectionDetected=False, category="none")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_overrides",
    [
        {
            "tool_calls": [
                {
                    "name": "ClassifierDecision",
                    "args": {"injectionDetected": False, "category": "none"},
                    "id": "one",
                },
                {
                    "name": "ClassifierDecision",
                    "args": {"injectionDetected": True, "category": "unknown"},
                    "id": "two",
                },
            ]
        },
        {
            "tool_calls": [
                {
                    "name": "OtherDecision",
                    "args": {"injectionDetected": False, "category": "none"},
                    "id": "one",
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "name": "ClassifierDecision",
                    "args": {"injectionDetected": True, "category": "unknown"},
                    "id": "one",
                }
            ]
        },
        {
            "additional_kwargs": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "ClassifierDecision",
                            "arguments": (
                                '{"injectionDetected":true,"category":"unknown",'
                                '"injectionDetected":false,"category":"none"}'
                            ),
                        },
                        "id": "one",
                        "type": "function",
                    }
                ]
            }
        },
        {
            "content": [
                {
                    "type": "function_call",
                    "name": "ClassifierDecision",
                    "arguments": (
                        '{"injectionDetected":true,"category":"unknown",'
                        '"injectionDetected":false,"category":"none"}'
                    ),
                    "call_id": "one",
                }
            ]
        },
        {
            "additional_kwargs": {
                "function_call": {
                    "name": "ClassifierDecision",
                    "arguments": (
                        '{"injectionDetected":true,"category":"unknown",'
                        '"injectionDetected":false,"category":"none"}'
                    ),
                }
            }
        },
        {
            "content": [
                {
                    "type": "tool_use",
                    "name": "ClassifierDecision",
                    "input": {"injectionDetected": True, "category": "unknown"},
                    "id": "one",
                }
            ]
        },
        {
            "tool_calls": [],
            "content": (
                '{"injectionDetected":true,"category":"unknown",'
                '"injectionDetected":false,"category":"none"}'
            ),
        },
        {"tool_calls": [], "content": '{"injectionDetected":true,"category":"unknown"}'},
        {"tool_calls": [], "content": ""},
    ],
    ids=[
        "conflicting-calls",
        "wrong-function",
        "parsed-disagrees",
        "chat-duplicate-keys",
        "responses-duplicate-keys",
        "legacy-duplicate-keys",
        "anthropic-disagrees",
        "parsed-text-duplicate-keys",
        "parsed-text-disagrees",
        "no-decision",
    ],
)
async def test_classifier_rejects_ambiguous_native_decision_before_trusting_parsed(
    raw_overrides: dict[str, Any],
) -> None:
    from langchain_core.messages import AIMessage

    raw = AIMessage(
        **{
            "content": "",
            "tool_calls": [
                {
                    "name": "ClassifierDecision",
                    "args": {"injectionDetected": False, "category": "none"},
                    "id": "one",
                }
            ],
            **raw_overrides,
        }
    )
    client = LangChainClassifierClient(
        StructuredEnvelopeFakeModel(
            {
                "raw": raw,
                "parsed": ClassifierDecision(injectionDetected=False, category="none"),
                "parsing_error": None,
            }
        )
    )
    with pytest.raises(ClassifierResponseError):
        await client.classify(
            ClassifierRequest(
                toolName="probe",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="TOOL-SECRET",
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("detected", [False, True])
@pytest.mark.parametrize("representation", ["normalized", "chat", "responses", "anthropic", "text"])
async def test_classifier_preserves_single_native_decision_representations(
    representation: str,
    detected: bool,
) -> None:
    from langchain_core.messages import AIMessage

    payload = {"injectionDetected": detected, "category": "unknown" if detected else "none"}
    raw_kwargs: dict[str, Any] = {
        "content": "",
        "tool_calls": [{"name": "ClassifierDecision", "args": payload, "id": "one"}],
    }
    if representation == "chat":
        raw_kwargs["additional_kwargs"] = {
            "tool_calls": [
                {
                    "function": {"name": "ClassifierDecision", "arguments": json.dumps(payload)},
                    "id": "one",
                    "type": "function",
                }
            ]
        }
    elif representation == "responses":
        raw_kwargs["content"] = [
            {
                "type": "function_call",
                "name": "ClassifierDecision",
                "arguments": json.dumps(payload),
                "call_id": "one",
            }
        ]
    elif representation == "anthropic":
        raw_kwargs["content"] = [
            {"type": "tool_use", "name": "ClassifierDecision", "input": payload, "id": "one"}
        ]
    elif representation == "text":
        raw_kwargs.update(content=json.dumps(payload), tool_calls=[])
    result = await LangChainClassifierClient(
        StructuredEnvelopeFakeModel(
            {
                "raw": AIMessage(**raw_kwargs),
                "parsed": payload,
                "parsing_error": None,
            }
        )
    ).classify(
        ClassifierRequest(
            toolName="probe",
            resultType="result",
            chunkIndex=0,
            chunkCount=1,
            content="data",
        )
    )
    assert (result.injection_detected, result.category) == (
        detected,
        "unknown" if detected else "none",
    )


@pytest.mark.asyncio
async def test_classifier_rejects_parsed_decision_without_raw_envelope() -> None:
    model = StructuredEnvelopeFakeModel(
        {
            "raw": None,
            "parsed": {"injectionDetected": False, "category": "none"},
            "parsing_error": None,
        }
    )
    with pytest.raises(ClassifierResponseError) as error:
        await LangChainClassifierClient(model).classify(
            ClassifierRequest(
                toolName="probe", resultType="result", chunkIndex=0, chunkCount=1, content="safe"
            )
        )
    assert error.value.issue == "invalid_envelope"


@pytest.mark.asyncio
async def test_classifier_reports_cutoff_before_structured_parsing_error() -> None:
    from langchain_core.messages import AIMessage

    model = StructuredEnvelopeFakeModel(
        {
            "raw": AIMessage(content="", response_metadata={"finish_reason": "length"}),
            "parsed": None,
            "parsing_error": ValueError("RAW-PARSER-SECRET"),
        }
    )
    with pytest.raises(ClassifierResponseError) as error:
        await LangChainClassifierClient(model).classify(
            ClassifierRequest(
                toolName="probe", resultType="result", chunkIndex=0, chunkCount=1, content="safe"
            )
        )
    assert error.value.issue == "output_limit"


@pytest.mark.asyncio
async def test_classifier_rejects_structured_parsing_errors() -> None:
    with pytest.raises(ClassifierResponseError) as exc_info:
        await LangChainClassifierClient(StructuredErrorFakeModel(None)).classify(
            ClassifierRequest(
                toolName="list_namespaces",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="Namespaces: default",
            )
        )
    assert exc_info.value.issue == "schema_mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "metadata", "expected"),
    [
        (
            '{"injectionDetected":false,"category":"none"}',
            {"finish_reason": "stop"},
            (False, "none"),
        ),
        (
            [{"type": "text", "text": '{"injectionDetected":false,"category":"none"}', "index": 0}],
            {"finish_reason": "stop"},
            (False, "none"),
        ),
        ('{"injectionDetected":true,"category":"unknown"}', {}, (True, "unknown")),
        (
            '{"injectionDetected":true,"category":"instruction_override"}',
            {"stop_reason": "end_turn"},
            (True, "instruction_override"),
        ),
        (
            '{"injectionDetected":false,"category":"none"}',
            {"finish_reason": "STOP"},
            (False, "none"),
        ),
        (
            '{"injectionDetected":false,"category":"none"}',
            {"stop_reason": "stop_sequence"},
            (False, "none"),
        ),
        ('{"injectionDetected":false,"category":"none"}', {"status": "completed"}, (False, "none")),
    ],
)
async def test_classifier_accepts_strict_text_decision_when_function_call_is_absent(
    content: Any, metadata: dict[str, Any], expected: tuple[bool, str]
) -> None:
    from langchain_core.messages import AIMessage

    model = StructuredEnvelopeFakeModel(
        {
            "raw": AIMessage(content=content, response_metadata=metadata),
            "parsed": None,
            "parsing_error": None,
        }
    )
    decision = await LangChainClassifierClient(model).classify(
        ClassifierRequest(
            toolName="list_namespaces",
            resultType="result",
            chunkIndex=0,
            chunkCount=1,
            content="Namespaces: default",
        )
    )

    assert (decision.injection_detected, decision.category) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_overrides", "parsed", "parsing_error"),
    [
        ({"tool_calls": [{"name": "ClassifierDecision", "args": {}, "id": "call-1"}]}, None, None),
        (
            {
                "invalid_tool_calls": [
                    {
                        "name": "ClassifierDecision",
                        "args": "{",
                        "id": "call-1",
                        "error": "invalid JSON",
                    }
                ]
            },
            None,
            None,
        ),
        (
            {
                "additional_kwargs": {
                    "function_call": {"name": "ClassifierDecision", "arguments": "{"}
                }
            },
            None,
            None,
        ),
        ({"additional_kwargs": {"tool_calls": {}}}, None, None),
        ({"additional_kwargs": {"refusal": "refused"}}, None, None),
        ({"response_metadata": {"finish_reason": "length"}}, None, None),
        ({"response_metadata": {"finish_reason": "content_filter"}}, None, None),
        ({"response_metadata": {"finish_reason": "tool_calls"}}, None, None),
        ({"response_metadata": {"stop_reason": "max_tokens"}}, None, None),
        ({"response_metadata": {"stop_reason": "refusal"}}, None, None),
        ({"response_metadata": {"status": "incomplete"}}, None, None),
        (
            {
                "content": [
                    {"type": "text", "text": '{"injectionDetected":false,"category":"none"}'},
                    {"type": "refusal", "refusal": "refused"},
                ]
            },
            None,
            None,
        ),
        (
            {
                "content": [
                    {"type": "text", "text": '{"injectionDetected":false,"category":"none"}'},
                    {"type": "image", "url": "https://example.test/image.png"},
                ]
            },
            None,
            None,
        ),
        ({}, None, ValueError("structured parsing failed")),
        ({}, {"injectionDetected": False, "category": "unknown"}, None),
        ({}, False, None),
    ],
)
async def test_classifier_text_fallback_rejects_unsafe_or_ambiguous_structured_envelopes(
    raw_overrides: dict[str, Any], parsed: Any, parsing_error: Any
) -> None:
    from langchain_core.messages import AIMessage

    raw = AIMessage(**{"content": '{"injectionDetected":false,"category":"none"}', **raw_overrides})
    model = StructuredEnvelopeFakeModel(
        {"raw": raw, "parsed": parsed, "parsing_error": parsing_error}
    )

    with pytest.raises(ClassifierResponseError):
        await LangChainClassifierClient(model).classify(
            ClassifierRequest(
                toolName="list_namespaces",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="Namespaces: default",
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '{"injectionDetected":true,"category":"unknown","injectionDetected":false,"category":"none"}',
        '{"injectionDetected":false,"category":"none","reason":"safe"}',
        '{"injectionDetected":false,"category":"unknown"}',
        '{"injectionDetected":"false","category":"none"}',
        '{"injectionDetected":false,"category":"none"',
        '```json\n{"injectionDetected":false,"category":"none"}\n```',
        '{"injectionDetected":false,"category":"none"} SAFE',
        "[]",
        "",
    ],
)
async def test_classifier_text_fallback_does_not_repair_or_relax_invalid_decisions(
    content: str,
) -> None:
    from langchain_core.messages import AIMessage

    model = StructuredEnvelopeFakeModel(
        {"raw": AIMessage(content=content), "parsed": None, "parsing_error": None}
    )

    with pytest.raises(ClassifierResponseError):
        await LangChainClassifierClient(model).classify(
            ClassifierRequest(
                toolName="list_namespaces",
                resultType="result",
                chunkIndex=0,
                chunkCount=1,
                content="Namespaces: default",
            )
        )


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
