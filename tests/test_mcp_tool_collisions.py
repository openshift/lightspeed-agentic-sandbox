"""Reject ambiguous admitted MCP namespaces before the SDK can overwrite tools."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from .helpers import make_model_route

pytestmark = pytest.mark.usefixtures("connection_audit_enabled")


@pytest.mark.asyncio
@pytest.mark.parametrize("inspection_enabled", [False, True])
@pytest.mark.parametrize(
    "scenario",
    [
        "within-server",
        "across-servers",
        "unique",
        "unadmitted-duplicates",
        "tool-error",
        "streamed-tool-error",
    ],
)
async def test_admitted_mcp_tool_namespace_is_unambiguous(
    scenario: str,
    inspection_enabled: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    span_exporter: Any,
) -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from lightspeed_agentic.mcp import (
        ResolvedMCPHeader,
        ResolvedMCPServer,
        discover_and_admit_mcp_servers,
        split_admitted_mcp_servers,
    )
    from lightspeed_agentic.providers.deepagents import DeepAgentsProvider
    from lightspeed_agentic.run_agent import run_agent_query

    collisions = scenario in ("within-server", "across-servers")
    catalogs = (
        {"server-a": ["lookup", "lookup"]}
        if scenario == "within-server"
        else {"server-a": ["lookup"], "server-b": ["lookup"]}
        if collisions
        else {"server-a": ["lookup_a"], "server-b": ["lookup_b"]}
    )
    catalog_reads = {name: 0 for name in catalogs}
    executed: list[tuple[str, str]] = []
    clients: list[httpx.AsyncClient] = []

    class ToolStream(httpx.AsyncByteStream):
        def __init__(self, payload: bytes):
            self.payload = payload

        async def __aiter__(self):
            yield self.payload
            await asyncio.Future()  # SDK closes HTTP early once it receives the MCP result.

    def respond(request: httpx.Request) -> httpx.Response:
        server = request.url.host.split(".")[0]
        assert request.headers["authorization"] == "Bearer PRIVATE-MCP-CREDENTIAL"
        body = json.loads(request.content)
        method = body["method"]
        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": server, "version": "1"},
            }
        elif method == "notifications/initialized":
            return httpx.Response(202)
        elif method == "tools/list":
            catalog_reads[server] += 1
            names = list(catalogs[server])
            if scenario == "unadmitted-duplicates" and catalog_reads[server] > 1:
                names.extend(["not_admitted", "not_admitted"])
            result = {
                "tools": [
                    {
                        "name": name,
                        "description": "PRIVATE-MCP-DESCRIPTION",
                        "annotations": {"readOnlyHint": True},
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "PRIVATE-MCP-SCHEMA"},
                            },
                        },
                    }
                    for name in names
                ]
            }
        else:
            assert method == "tools/call"
            name = body["params"]["name"]
            executed.append((server, name))
            result = {
                "content": [{"type": "text", "text": f"findings from {server}"}],
                "isError": scenario in {"tool-error", "streamed-tool-error"},
            }
            if scenario == "streamed-tool-error":
                payload = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result})
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream"},
                    stream=ToolStream(f"event: message\ndata: {payload}\n\n".encode()),
                )
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    def create_client(**kwargs: Any) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("lightspeed_agentic.tls.create_async_http_client", create_client)
    monkeypatch.setenv("HOME", str(tmp_path))
    caplog.set_level("INFO")
    admitted = await discover_and_admit_mcp_servers(
        [
            ResolvedMCPServer(
                name=name,
                url=f"https://{name}.example/mcp",
                headers=[
                    ResolvedMCPHeader("Authorization", "Bearer PRIVATE-MCP-CREDENTIAL"),
                ],
            )
            for name in catalogs
        ]
    )
    servers, _policies = split_admitted_mcp_servers(admitted)
    assert len(servers) == len(catalogs)
    call_names = ("lookup",) if collisions else ("lookup_a", "lookup_b")
    calls = [
        {"type": "tool_call", "id": f"call-{name}", "name": name, "args": {"query": "findings"}}
        for name in call_names
    ]
    model_inputs: list[Any] = []

    class MainModel(FakeMessagesListChatModel):
        def bind_tools(self, _tools: Any, **_kwargs: Any) -> Any:
            return self

        def _generate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> Any:
            model_inputs.append(messages)
            return super()._generate(messages, stop, run_manager, **kwargs)

    main = MainModel(responses=[AIMessage(content="", tool_calls=calls), AIMessage(content="done")])
    classifier = FakeMessagesListChatModel(
        responses=[AIMessage(content=('{"injectionDetected":false,"category":"none"}'))]
    )
    models = iter((main, classifier))
    result = await run_agent_query(
        DeepAgentsProvider(make_model_route(lambda _parameters, _reasoning: next(models))),
        prompt="Look up findings.",
        system_prompt="Report findings.",
        output_schema=None,
        context=None,
        skills_dir=str(tmp_path),
        model="offline-model",
        max_turns=10,
        timeout_seconds=30,
        mcp_servers=servers,
        tool_output_inspection_enabled=inspection_enabled,
        agenticrun_uid="11111111-1111-4111-8111-111111111111",
        step="verification",
    )
    if collisions:
        assert result.output == {
            "success": False,
            "summary": "Agent error: MCP tool name collision.",
        }
        assert executed == []
        assert model_inputs == []
        errors = [record for record in caplog.records if record.levelname == "ERROR"]
        assert len(errors) == 1
        assert errors[0].exc_info is None
        message = errors[0].getMessage()
        assert "tool=lookup" in message
        assert "reason=duplicate_tool_name" in message
        assert (
            "servers=server-a,server-a"
            if scenario == "within-server"
            else "servers=server-a,server-b"
        ) in message
        root = next(s for s in span_exporter.get_finished_spans() if s.name == "invoke_agent")
        assert root.status.status_code.name == "ERROR"
        assert root.attributes["error.type"] == "MCPToolNameCollisionError"
        assert "gen_ai.output.messages" not in root.attributes
    else:
        assert result.output == {"success": True, "summary": "done"}
        assert sorted(executed) == [("server-a", "lookup_a"), ("server-b", "lookup_b")]
        assert len(model_inputs) == 2
    for marker in ("PRIVATE-MCP-CREDENTIAL", "PRIVATE-MCP-DESCRIPTION", "PRIVATE-MCP-SCHEMA"):
        assert marker not in caplog.text
        assert marker not in repr([record.__dict__ for record in caplog.records])
        assert marker not in json.dumps(result.output)
    assert all(client.is_closed for client in clients)
    observed = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"schema_version":1,')
    ]
    assert all(row.get("request.kind") == "mcp" for row in observed)
    assert {row["server.address"] for row in observed} <= {"server-a.example", "server-b.example"}
    starts = [row for row in observed if row["event"] == "outbound.request.started"]
    ends = [row for row in observed if row["event"] == "outbound.request.finished"]
    assert len(starts) == len(ends)
    assert {row["attempt_id"] for row in starts} == {row["attempt_id"] for row in ends}
    incomplete = [row for row in ends if row["outcome"] == "transport_error"]
    assert len(incomplete) == (2 if scenario == "streamed-tool-error" else 0)
    for row in ends:
        assert row["outcome"] == ("transport_error" if row in incomplete else "success")
        assert row.get("error.type") == ("protocol_error" if row in incomplete else None)
    discovery = [row for row in starts if "gen_ai.tool.name" not in row]
    assert len(discovery) == 6 * len(catalogs)  # Admission plus runtime reload, three HTTPs each.
    calls = [row for row in starts if "gen_ai.tool.name" in row]
    if collisions:
        assert calls == []
    else:
        # Successful results trigger an extra schema-cache tools/list in the native SDK.
        assert len(calls) == (6 if scenario in {"tool-error", "streamed-tool-error"} else 8)
        assert {(row["server.address"], row["gen_ai.tool.name"]) for row in calls} == {
            ("server-a.example", "lookup_a"),
            ("server-b.example", "lookup_b"),
        }
        assert all(
            row["agenticrun.uid"] == "11111111-1111-4111-8111-111111111111"
            and row["agenticrun.phase"] == "verification"
            for row in calls
        )
        assert len({row["attempt_id"] for row in starts}) == len(starts)
    assert "PRIVATE" not in json.dumps(observed)
