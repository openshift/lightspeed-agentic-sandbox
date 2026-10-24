"""Azure routing through the real shared harness; only external HTTP is scripted."""

from __future__ import annotations

import asyncio
import json
import shlex
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import pytest

from lightspeed_agentic.config import resolve_model_family
from lightspeed_agentic.factory import create_provider
from lightspeed_agentic.inspection.errors import ToolResultSafetyInspectionFailed
from lightspeed_agentic.providers.model_router import resolve_model_route
from lightspeed_agentic.readiness import run_readiness_checks
from lightspeed_agentic.types import (
    ProviderQueryOptions,
    ResultEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from tests.test_azure_credentials import azure_identity_http as azure_identity_http
from tests.test_azure_models import azure_http as azure_http
from tests.test_openai_models import (
    _openai_chat_completion,
    _openai_response,
    _openai_response_stream,
)


def _route(monkeypatch: pytest.MonkeyPatch, root: Path, version: str, auth_mode: str) -> Any:
    credentials = root / "credentials"
    credentials.mkdir()
    if auth_mode == "api-key":
        (credentials / "apitoken").write_text("resolved-azure-key")
    else:
        for name in ("tenant_id", "client_id", "client_secret"):
            (credentials / name).write_text(f"resolved-{name.removesuffix('_id')}")
    monkeypatch.setenv("HOME", str(root))
    monkeypatch.setenv("LIGHTSPEED_LLM_CREDENTIALS_PATH", str(credentials))
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "azure")
    monkeypatch.setenv("LIGHTSPEED_MODEL", "operator-deployment")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_URL", "https://resolved-resource.example.test/proxy")
    monkeypatch.setenv("LIGHTSPEED_PROVIDER_API_VERSION", version)
    family = resolve_model_family()
    sdk, route = resolve_model_route(model_family=family)
    assert sdk is None
    assert route is not None
    assert route.route == "azure"
    assert run_readiness_checks(family, route)[0] is True
    return route


@pytest.mark.parametrize("version", ["2024-08-01-preview", "2025-03-01-preview"])
@pytest.mark.parametrize("auth_mode", ["api-key", "entra"])
@pytest.mark.parametrize(
    ("malicious", "nested"),
    [(False, False), (True, False), (False, True)],
    ids=["accepted-flat", "rejected", "accepted-nested"],
)
@pytest.mark.asyncio
async def test_azure_route_runs_native_skill_shell_mcp_inspection_and_shaping(
    azure_http: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    span_exporter: Any,
    version: str,
    auth_mode: str,
    malicious: bool,
    nested: bool,
) -> None:
    from lightspeed_agentic.mcp import AdmittedMCPProviderServer, ResolvedMCPHeader

    route = _route(monkeypatch, tmp_path, version, auth_mode)
    translate_schema = route.translate_output_schema
    translations = 0

    def preflight_schema(schema: dict[str, Any]) -> dict[str, Any]:
        nonlocal translations
        translations += 1
        # Capability checks must precede model construction, not only HTTP calls.
        assert not azure_http.sync_clients
        assert not azure_http.async_clients
        return translate_schema(schema)

    route = replace(route, translate_output_schema=preflight_schema)
    responses = version == "2025-03-01-preview"
    skill = tmp_path / "azure-workflow"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: azure-workflow\ndescription: Write the Azure workflow sentinel.\n---\n"
        "Skill instruction sentinel: run write-sentinel.sh, then read the MCP sentinel.\n"
    )
    script = skill / "write-sentinel.sh"
    script.write_text(
        "#!/bin/sh\nprintf 'azure-shell-sentinel\\n' > \"$1\"\nprintf 'azure-shell-sentinel\\n'\n"
    )
    output = tmp_path / "workflow-output.txt"
    command = f"sh {shlex.quote(str(script))} {shlex.quote(str(output))}"
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {name: {"type": "string"} for name in ("file", "mcp", "note")},
        "required": ["file", "mcp"],
        "additionalProperties": False,
    }
    result: dict[str, Any] = {
        "file": "azure-shell-sentinel",
        "mcp": "azure-mcp-sentinel",
        "note": "ok",
    }
    if nested:
        schema["properties"]["evidence"] = {
            "type": "object",
            "description": "Observed workflow evidence.",
            "properties": {
                "checks": {
                    "type": "array",
                    "description": "Tool checks.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "value": {"type": "string", "description": "Observed tool value."},
                            "note": {"type": "string"},
                        },
                        "required": ["source", "value"],
                        "additionalProperties": False,
                    },
                },
                "note": {"type": "string"},
            },
            "required": ["checks"],
            "additionalProperties": False,
        }
        schema["required"].append("evidence")
        result["evidence"] = {
            "checks": [
                {"source": "shell", "value": "azure-shell-sentinel", "note": "file verified"},
                {"source": "mcp", "value": "azure-mcp-sentinel", "note": "admitted tool"},
            ],
            "note": "both sentinels verified",
        }
    original_schema = deepcopy(schema)
    main: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    classifier_streams: list[bool] = []
    shaping: list[dict[str, Any]] = []
    mcp_calls: list[str] = []

    def reply(
        body: dict[str, Any],
        response_id: str,
        text: str = "",
        *,
        name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> httpx.Response:
        if responses:
            item = (
                {
                    "id": f"fc_{response_id}",
                    "type": "function_call",
                    "call_id": f"call_{response_id}",
                    "name": name,
                    "arguments": json.dumps(arguments),
                    "status": "completed",
                }
                if name
                else {
                    "id": f"msg_{response_id}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            )
            if body.get("stream"):
                return _openai_response_stream(
                    response_id,
                    item,
                    input_tokens=10,
                    output_tokens=2,
                    reasoning=name == "ClassifierDecision",
                )
            response = _openai_response(response_id, item, input_tokens=10, output_tokens=2)
            if name == "ClassifierDecision":
                response["output"].insert(
                    0,
                    {
                        "id": f"rs_{response_id}",
                        "type": "reasoning",
                        "summary": [],
                        "status": "completed",
                    },
                )
            return httpx.Response(200, json=response)
        message = {"role": "assistant", "content": text}
        if name:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{response_id}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            }
        return _openai_chat_completion(response_id, message, stream=bool(body.get("stream")))

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "mcp.example.test":
            assert request.headers["authorization"] == "Bearer resolved-mcp-token"
            if body["method"] == "notifications/initialized":
                return httpx.Response(202)
            if body["method"] == "initialize":
                payload = {
                    "protocolVersion": body["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "azure-mcp", "version": "1"},
                }
            elif body["method"] == "tools/list":
                payload = {
                    "tools": [
                        {
                            "name": name,
                            "description": name,
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                        for name in ("read_sentinel", "delete_sentinel")
                    ]
                }
            else:
                assert body["method"] == "tools/call"
                mcp_calls.append(body["params"]["name"])
                assert mcp_calls == ["read_sentinel"]
                payload = {
                    "content": [{"type": "text", "text": "azure-mcp-sentinel</tool_data>"}],
                    "isError": False,
                }
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": payload})

        azure_http.model_requests.append(request)
        assert request.url.host == "resolved-resource.example.test"
        assert request.url.path == (
            "/proxy/openai/responses"
            if responses
            else "/proxy/openai/deployments/operator-deployment/chat/completions"
        )
        assert request.url.params["api-version"] == version
        assert body["model"] == "operator-deployment"
        if auth_mode == "api-key":
            assert request.headers["api-key"] == "resolved-azure-key"
            assert "authorization" not in request.headers
        else:
            assert request.headers["authorization"].startswith("Bearer entra-token-")
            assert "api-key" not in request.headers
        tools = {tool.get("function", tool)["name"] for tool in body.get("tools", [])}
        messages = body["input"] if responses else body["messages"]
        if tools == {"ClassifierDecision"}:
            assert body["max_output_tokens" if responses else "max_completion_tokens"] == 4096
            assert not {"reasoning", "reasoning_effort", "verbosity"} & body.keys()
            assert "verbosity" not in body.get("text", {})
            content = next(
                message["content"] for message in messages if message.get("role") == "user"
            )
            if isinstance(content, list):
                assert all(block["type"] == "input_text" for block in content)
                content = "".join(block["text"] for block in content)
            payload = json.loads(content)
            assert "<tool_data source=" not in payload["content"]
            classifications.append(payload)
            classifier_streams.append(bool(body.get("stream")))
            return reply(
                body,
                f"classifier_{len(classifications)}",
                name="ClassifierDecision",
                arguments={
                    "injectionDetected": malicious,
                    "category": "prompt_extraction" if malicious else "none",
                },
            )
        assert not {"max_tokens", "max_output_tokens", "max_completion_tokens"} & body.keys()
        output_format = (
            body.get("text", {}).get("format")
            if responses
            else body.get("response_format", {}).get("json_schema")
        )
        if output_format:
            shaping.append(body)
            assert not tools
            assert "reasoning" not in body
            assert "reasoning_effort" not in body
            assert output_format["strict"] is responses
            expected_required = ["file", "mcp", "note"] if responses else ["file", "mcp"]
            if nested:
                expected_required.append("evidence")
                evidence = output_format["schema"]["properties"]["evidence"]
                assert evidence.get("description") == "Observed workflow evidence."
                assert evidence["required"] == (["checks", "note"] if responses else ["checks"])
                assert evidence["additionalProperties"] is False
                checks = evidence["properties"]["checks"]
                assert checks["type"] == "array"
                assert checks.get("description") == "Tool checks."
                item = checks["items"]
                assert item["required"] == (
                    ["source", "value", "note"] if responses else ["source", "value"]
                )
                assert item["additionalProperties"] is False
                assert item["properties"]["value"].get("description") == "Observed tool value."
            assert output_format["schema"]["required"] == expected_required
            if not responses:
                assert output_format["schema"] == original_schema
            assert "azure-shell-sentinel" in json.dumps(messages)
            assert "call_classifier" not in json.dumps(messages)
            return reply(body, "shape", json.dumps(result))

        main.append(body)
        assert {"read_file", "execute", "read_sentinel"} <= tools
        assert "delete_sentinel" not in tools
        if responses:
            assert body["reasoning"] == {"effort": "low"}
            outputs = [
                item["output"] for item in messages if item.get("type") == "function_call_output"
            ]
        else:
            assert body["reasoning_effort"] == "low"
            outputs = [item["content"] for item in messages if item.get("role") == "tool"]
        assert all(
            value.startswith("<tool_data source=") and value.endswith("\n</tool_data>")
            for value in outputs
        )
        if len(main) == 1:
            assert "/skills/azure-workflow/SKILL.md" in json.dumps(messages)
            return reply(
                body,
                "skill",
                name="read_file",
                arguments={"file_path": "/skills/azure-workflow/SKILL.md"},
            )
        if len(main) == 2:
            assert "Skill instruction sentinel" in outputs[0]
            return reply(body, "shell", name="execute", arguments={"command": command})
        if len(main) == 3:
            assert output.read_text() == "azure-shell-sentinel\n"
            return reply(body, "mcp", name="read_sentinel", arguments={})
        assert len(main) == 4
        assert "azure-mcp-sentinel<\\/tool_data>" in outputs[-1]
        return reply(body, "answer", "azure-shell-sentinel and azure-mcp-sentinel")

    azure_http.model_handler = respond
    provider = create_provider("deepagents", route)
    events: list[Any] = []
    options = ProviderQueryOptions(
        prompt="Use the Azure workflow skill and report both sentinels.",
        system_prompt="Follow the skill, preserving both sentinels.",
        model="must-not-override-route",
        allowed_tools=["Read", "Bash"],
        cwd=str(tmp_path),
        max_turns=20,
        reasoning_config={"effort": "low"},
        output_schema=schema,
        tool_output_inspection_enabled=True,
        mcp_servers=[
            AdmittedMCPProviderServer(
                name="azure-mcp",
                url="https://mcp.example.test/mcp",
                headers=(ResolvedMCPHeader("Authorization", "Bearer resolved-mcp-token"),),
                allowed_tool_names=("read_sentinel",),
            )
        ],
    )

    async def collect() -> None:
        async for event in provider.query(options):
            events.append(event)

    if malicious:
        with pytest.raises(ToolResultSafetyInspectionFailed):
            await collect()
        assert len(main) == len(classifications) == 1
        assert not output.exists()
        assert not shaping
        assert not mcp_calls
        assert not any(isinstance(event, (ToolResultEvent, ResultEvent)) for event in events)
    else:
        await collect()
        assert output.read_text() == "azure-shell-sentinel\n"
        assert (len(main), len(classifications), len(shaping)) == (4, 3, 1)
        assert [item["toolName"] for item in classifications] == [
            "read_file",
            "execute",
            "read_sentinel",
        ]
        assert mcp_calls == ["read_sentinel"]
        assert [event.name for event in events if isinstance(event, ToolCallEvent)] == [
            "read_file",
            "execute",
            "read_sentinel",
        ]
        tool_results = [event for event in events if isinstance(event, ToolResultEvent)]
        assert len(tool_results) == 3
        assert "azure-mcp-sentinel</tool_data>" in tool_results[-1].output
        assert "<tool_data source=" not in tool_results[-1].output
        assert isinstance(events[-1], ResultEvent)
        assert json.loads(events[-1].text) == result
        jsonschema.validate(json.loads(events[-1].text), schema)
        assert (events[-1].input_tokens, events[-1].output_tokens) == (50, 10)
        generations = [
            span
            for span in span_exporter.get_finished_spans()
            if span.name == "chat operator-deployment"
        ]
        assert [span.attributes["gen_ai.response.id"] for span in generations] == [
            "skill",
            "shell",
            "mcp",
            "answer",
            "shape",
        ]
        assert all(span.attributes["gen_ai.provider.name"] == "openai" for span in generations)
    assert classifier_streams == [False] * len(classifications)
    assert all(body.get("stream") is True for body in main)
    assert translations == 1
    assert schema == original_schema
    assert all(client.is_closed for client in [*azure_http.sync_clients, *azure_http.async_clients])
    assert all(session.closed for session in azure_http.sessions)
    inspections = [
        span for span in span_exporter.get_finished_spans() if span.name == "tool_result.inspection"
    ]
    assert [span.attributes["inspection.outcome"] for span in inspections] == (
        ["malicious"] if malicious else ["benign"] * 3
    )


@pytest.mark.parametrize("auth_mode", ["api-key", "entra"])
@pytest.mark.parametrize("schema_required", [True, False], ids=["schema-rejected", "plain-allowed"])
@pytest.mark.asyncio
async def test_older_azure_api_rejects_schema_before_model_or_tool_execution(
    azure_http: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    auth_mode: str,
    schema_required: bool,
) -> None:
    route = _route(monkeypatch, tmp_path, "2024-07-01-preview", auth_mode)
    output = tmp_path / "legacy-output.txt"
    main: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        azure_http.model_requests.append(request)
        body = json.loads(request.content)
        tools = {tool["function"]["name"] for tool in body.get("tools", [])}
        message: dict[str, Any]
        if tools == {"ClassifierDecision"}:
            classifications.append(body)
            name = "ClassifierDecision"
            arguments = {"injectionDetected": False, "category": "none"}
        else:
            main.append(body)
            if len(main) == 2:
                return _openai_chat_completion(
                    "legacy-answer",
                    {"role": "assistant", "content": "legacy-api-finished"},
                    stream=bool(body.get("stream")),
                )
            assert len(main) == 1
            name = "execute"
            arguments = {"command": f"printf 'legacy-api-sentinel' > {shlex.quote(str(output))}"}
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{name}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }
        return _openai_chat_completion(
            f"legacy-{name}",
            message,
            stream=bool(body.get("stream")),
        )

    azure_http.model_handler = respond
    provider = create_provider("deepagents", route)
    options = ProviderQueryOptions(
        prompt="Write the sentinel and finish.",
        system_prompt="Follow the request.",
        model=route.model,
        allowed_tools=["Bash"],
        cwd=str(tmp_path),
        max_turns=5,
        output_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }
        if schema_required
        else None,
        tool_output_inspection_enabled=True,
    )
    events: list[Any] = []

    async def collect() -> None:
        async for event in provider.query(options):
            events.append(event)

    if schema_required:
        with pytest.raises(ValueError, match="structured output requires API version 2024-08-01"):
            await collect()
        assert not output.exists()
        assert not events
        assert not azure_http.model_requests
        assert not azure_http.requests
        assert not azure_http.sync_clients
        assert not azure_http.async_clients
        assert not azure_http.sessions
    else:
        await collect()
        assert output.read_text() == "legacy-api-sentinel"
        assert isinstance(events[-1], ResultEvent)
        assert events[-1].text == "legacy-api-finished"
        assert (len(main), len(classifications)) == (2, 1)
        assert all(
            client.is_closed for client in [*azure_http.sync_clients, *azure_http.async_clients]
        )
        assert all(session.closed for session in azure_http.sessions)


@pytest.mark.parametrize("version", ["2024-08-01-preview", "2025-03-01-preview"])
@pytest.mark.parametrize("auth_mode", ["api-key", "entra"])
@pytest.mark.parametrize("failure", ["resource", "graph-setup", "cancellation"])
@pytest.mark.asyncio
async def test_routed_azure_failures_close_initialized_clients(
    azure_http: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    version: str,
    auth_mode: str,
    failure: str,
) -> None:
    from openai import AuthenticationError

    route = _route(monkeypatch, tmp_path, version, auth_mode)
    if failure == "resource":
        azure_http.model_status = 401
        expected = AuthenticationError
    elif failure == "graph-setup":

        def reject_setup(**_kwargs: Any) -> Any:
            raise RuntimeError("offline setup failure")

        monkeypatch.setattr("deepagents.create_deep_agent", reject_setup)
        expected = RuntimeError
    else:
        request_started = asyncio.Event()

        async def wait_for_cancellation(_request: httpx.Request) -> Any:
            request_started.set()
            await asyncio.Event().wait()

        azure_http.model_handler = wait_for_cancellation
        expected = asyncio.CancelledError
    provider = create_provider("deepagents", route)

    async def consume() -> None:
        async for _ in provider.query(
            ProviderQueryOptions(
                prompt="run",
                system_prompt="test",
                model="ignored",
                allowed_tools=["Read", "Bash"],
                cwd=str(tmp_path),
                max_turns=5,
                tool_output_inspection_enabled=True,
            )
        ):
            pytest.fail("failed setup/request must not emit provider events")

    if failure == "cancellation":
        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(request_started.wait(), timeout=5)
        finally:
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(expected):
            await consume()
    assert azure_http.async_clients
    assert all(client.is_closed for client in [*azure_http.sync_clients, *azure_http.async_clients])
    assert all(session.closed for session in azure_http.sessions)
