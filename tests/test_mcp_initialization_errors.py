"""Admitted MCP setup failures are operational errors, not raw SDK diagnostics."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from .helpers import make_model_route

pytestmark = pytest.mark.usefixtures("connection_audit_enabled")


@pytest.mark.asyncio
@pytest.mark.parametrize("inspection_enabled", [False, True])
@pytest.mark.parametrize(
    "failure", ["client-construction", "initialize", "tools/list", "invalid-tool-schema"]
)
async def test_admitted_mcp_failure_has_safe_logs_and_result(
    capsys: pytest.CaptureFixture[str],
    failure: str,
    inspection_enabled: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    span_exporter: Any,
) -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from lightspeed_agentic.mcp import AdmittedMCPProviderServer, ResolvedMCPHeader
    from lightspeed_agentic.providers.deepagents import DeepAgentsProvider
    from lightspeed_agentic.publish_results.status import build_status
    from lightspeed_agentic.run_agent import run_agent_query

    class NoModelCalls(FakeMessagesListChatModel):
        def _generate(self, *_args: Any, **_kwargs: Any) -> Any:
            pytest.fail(
                "An admitted server's setup failure must stop the query before any model call"
            )

    model = NoModelCalls(responses=[AIMessage(content="must not run")])
    requests: list[tuple[str, str]] = []
    clients: list[httpx.AsyncClient] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method = body["method"]
        requests.append((request.url.host, method))
        broken = request.url.host == "broken-mcp.example"
        if broken and method == failure:
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "error": {"code": -32000, "message": "PRIVATE-MCP-BODY: reflected credentials"},
                },
            )
        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "offline-mcp", "version": "1"},
            }
        elif method == "notifications/initialized":
            return httpx.Response(202)
        else:
            assert method == "tools/list"
            result = {
                "tools": [
                    {
                        "name": "lookup",
                        "description": "PRIVATE-MCP-DESCRIPTION",
                        "inputSchema": ["PRIVATE-MCP-BODY"]
                        if broken
                        else {
                            "type": "object",
                            "properties": {},
                        },
                    }
                ]
            }
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    def create_client(**kwargs: Any) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    if failure == "client-construction":

        def reject_client(*_args: Any, **_kwargs: Any) -> Any:
            raise ValueError("PRIVATE-MCP-BODY: PRIVATE-MCP-CREDENTIAL")

        monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", reject_client)
    monkeypatch.setattr("lightspeed_agentic.tls.create_async_http_client", create_client)
    monkeypatch.setenv("HOME", str(tmp_path))
    caplog.set_level("INFO")
    result = await run_agent_query(
        DeepAgentsProvider(make_model_route(lambda _parameters, _reasoning: model)),
        prompt="Look up findings.",
        system_prompt="Report findings.",
        output_schema=None,
        context=None,
        skills_dir=str(tmp_path),
        model="offline-model",
        max_turns=10,
        timeout_seconds=30,
        tool_output_inspection_enabled=inspection_enabled,
        mcp_servers=[
            AdmittedMCPProviderServer(
                name=f"{name}-server",
                url=f"https://{name}-mcp.example/mcp",
                headers=(ResolvedMCPHeader("Authorization", "Bearer PRIVATE-MCP-CREDENTIAL"),),
                allowed_tool_names=("lookup",),
            )
            for name in ("healthy", "broken")
        ],
    )
    assert result.output == {"success": False, "summary": "Agent error: MCP initialization failed."}
    now = datetime.now(UTC)
    status = build_status("AnalysisResult", result.output, started_at=now, completed_at=now)
    assert status["failureReason"] == "Agent error: MCP initialization failed."
    assert next(c for c in status["conditions"] if c["type"] == "Completed")["reason"] == "Failed"
    for marker in ("PRIVATE-MCP-BODY", "PRIVATE-MCP-CREDENTIAL", "PRIVATE-MCP-DESCRIPTION"):
        assert marker not in caplog.text
        assert marker not in repr([record.__dict__ for record in caplog.records])
        assert marker not in json.dumps(status)
    errors = [record for record in caplog.records if record.levelname == "ERROR"]
    assert len(errors) == 1
    assert errors[0].exc_info is None
    assert "broken-server" in errors[0].getMessage()
    assert ("ValueError" if failure == "client-construction" else "ExceptionGroup") in errors[
        0
    ].getMessage()
    assert all(client.is_closed for client in clients)
    if failure == "client-construction":
        assert not requests
        assert "healthy-server" in errors[0].getMessage()
    else:
        assert [method for host, method in requests if host == "healthy-mcp.example"] == [
            "initialize",
            "notifications/initialized",
            "tools/list",
        ]
        assert [method for host, method in requests if host == "broken-mcp.example"] == (
            ["initialize"]
            if failure == "initialize"
            else [
                "initialize",
                "notifications/initialized",
                "tools/list",
            ]
        )
        assert "healthy-server" not in errors[0].getMessage()
    spans = span_exporter.get_finished_spans()
    invocation = next(s for s in spans if s.name == "invoke_agent")
    assert invocation.status.status_code.name == "ERROR"
    assert invocation.attributes["error.type"] == "MCPInitializationError"
    assert "gen_ai.output.messages" not in invocation.attributes
    assert not invocation.events
    assert not any(s.name == "tool_result.inspection" for s in spans)
    observed = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"schema_version":1,')
    ]
    starts = [row for row in observed if row["event"] == "outbound.request.started"]
    ends = [row for row in observed if row["event"] == "outbound.request.finished"]
    assert len(starts) == len(ends) == len(requests)
    assert all(row["request.kind"] == "mcp" for row in observed)
    assert all("gen_ai.tool.name" not in row for row in observed)
    assert all(row["outcome"] == "success" for row in ends)
    assert "PRIVATE" not in json.dumps(observed)


@pytest.mark.asyncio
async def test_mcp_initialization_cancellation_is_not_an_operational_failure(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    span_exporter: Any,
) -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from lightspeed_agentic.mcp import AdmittedMCPProviderServer
    from lightspeed_agentic.providers.deepagents import DeepAgentsProvider
    from lightspeed_agentic.run_agent import run_agent_query

    clients: list[httpx.AsyncClient] = []

    started = asyncio.Event()

    async def blocked_request(_request: httpx.Request) -> httpx.Response:
        started.set()
        return await asyncio.Future()

    def create_client(**kwargs: Any) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(blocked_request), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("lightspeed_agentic.tls.create_async_http_client", create_client)
    monkeypatch.setenv("HOME", str(tmp_path))
    model = FakeMessagesListChatModel(responses=[AIMessage(content="must not run")])
    task = asyncio.create_task(
        run_agent_query(
            DeepAgentsProvider(make_model_route(lambda _parameters, _reasoning: model)),
            prompt="Look up findings.",
            system_prompt="Report findings.",
            output_schema=None,
            context=None,
            skills_dir=str(tmp_path),
            model="offline-model",
            max_turns=10,
            timeout_seconds=30,
            mcp_servers=[
                AdmittedMCPProviderServer(
                    name="cancelled-server",
                    url="https://cancelled-mcp.example/mcp",
                    allowed_tool_names=("lookup",),
                )
            ],
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert clients
    assert all(client.is_closed for client in clients)
    invocation = next(s for s in span_exporter.get_finished_spans() if s.name == "invoke_agent")
    assert invocation.attributes["error.type"] == "CancelledError"
    assert invocation.status.status_code.name == "ERROR"
    assert not any(record.levelname == "ERROR" for record in caplog.records)
    observed = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"schema_version":1,')
    ]
    start, end = observed
    assert start["event"] == "outbound.request.started"
    assert end["event"] == "outbound.request.finished"
    assert start["attempt_id"] == end["attempt_id"]
    assert end["server.address"] == "cancelled-mcp.example"
    assert end["outcome"] == "cancelled"
    assert end["error.type"] == "cancelled"
