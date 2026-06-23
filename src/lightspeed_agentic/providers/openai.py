"""OpenAI provider — wraps openai-agents SDK.

Uses SandboxAgent with native Shell, Filesystem, and Skills capabilities.
The SDK handles tool registration, skill discovery, and command execution.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:

    class AgentOutputSchemaBase:
        """Type-checker stub for the optional openai-agents base class."""

        pass

else:
    try:
        from agents.agent_output import AgentOutputSchemaBase
    except ImportError:

        class AgentOutputSchemaBase:  # pragma: no cover - optional SDK fallback
            """Fallback base so the module imports without the openai extra."""

            pass


from lightspeed_agentic.types import (
    TOOL_INPUT_MAX_CHARS,
    TOOL_OUTPUT_MAX_CHARS,
    AgentProvider,
    ContentBlockStopEvent,
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    stringify,
)


def _make_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Add additionalProperties:false and required:[all props] to all objects recursively."""
    if not isinstance(schema, dict):
        return schema
    schema = dict(schema)
    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        schema["required"] = list(schema["properties"].keys())
        schema["properties"] = {k: _make_strict(v) for k, v in schema["properties"].items()}
    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _make_strict(schema["items"])
    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema and isinstance(schema[keyword], list):
            schema[keyword] = [_make_strict(s) for s in schema[keyword]]
    if "not" in schema and isinstance(schema["not"], dict):
        schema["not"] = _make_strict(schema["not"])
    for defs_key in ("$defs", "definitions"):
        if defs_key in schema and isinstance(schema[defs_key], dict):
            schema[defs_key] = {k: _make_strict(v) for k, v in schema[defs_key].items()}
    return schema


_OPENAI_HOSTS = ("api.openai.com",)


def _is_native_openai() -> bool:
    """True when talking to api.openai.com (explicitly or by default)."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        return True
    try:
        from urllib.parse import urlparse

        return urlparse(base_url).hostname in _OPENAI_HOSTS
    except Exception:
        return False


class _RawJsonSchema(AgentOutputSchemaBase):
    """Wraps an operator-provided JSON schema dict for the openai-agents SDK.

    Strict mode is enabled for native OpenAI (guarantees schema conformance)
    but disabled for custom endpoints like vLLM that don't support it.
    """

    def __init__(self, schema: dict[str, Any]) -> None:
        self._strict = _is_native_openai()
        self._schema = _make_strict(schema) if self._strict else schema

    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return "raw_json_schema"

    def json_schema(self) -> dict[str, Any]:
        return self._schema

    def is_strict_json_schema(self) -> bool:
        return self._strict

    def validate_json(self, json_str: str) -> Any:
        return json.loads(json_str)


logger = logging.getLogger(__name__)

_openai_initialized = False


def _ensure_openai_init() -> None:
    global _openai_initialized
    if _openai_initialized:
        return
    from agents import enable_verbose_stdout_logging
    from agents.tracing import set_tracing_disabled

    set_tracing_disabled(True)
    enable_verbose_stdout_logging()  # type: ignore[no-untyped-call]
    _openai_initialized = True


def _validated_e2e_output_dir() -> str | None:
    """Return E2E_OUTPUT_DIR when it resolves under the system temp directory."""
    raw = os.environ.get("E2E_OUTPUT_DIR", "").strip()
    if not raw:
        return None
    try:
        resolved = Path(raw).resolve()
    except OSError:
        logger.warning("E2E_OUTPUT_DIR is not a valid path: %s", raw)
        return None
    temp_root = Path(tempfile.gettempdir()).resolve()
    if resolved != temp_root and not str(resolved).startswith(str(temp_root) + os.sep):
        logger.warning(
            "E2E_OUTPUT_DIR outside temp root %s: %s",
            temp_root,
            resolved,
        )
        return None
    return str(resolved)


def _build_manifest(cwd: str) -> Any:
    """Build sandbox manifest, optionally granting write access to E2E_OUTPUT_DIR."""
    from agents.sandbox.manifest import Manifest, SandboxPathGrant  # type: ignore[attr-defined]

    kwargs: dict[str, Any] = {"root": cwd}
    output_dir = _validated_e2e_output_dir()
    if output_dir:
        kwargs["extra_path_grants"] = (
            SandboxPathGrant(
                path=output_dir,
                read_only=False,
                description="e2e skill token output",
            ),
        )
    return Manifest(**kwargs)


class OpenAIProvider(AgentProvider):
    _client: Any = None

    @property
    def name(self) -> str:
        return "openai"

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        from agents import (
            RawResponsesStreamEvent,
            RunItemStreamEvent,
            Runner,
        )
        from agents.items import ToolCallItem, ToolCallOutputItem
        from agents.models.openai_responses import OpenAIResponsesModel
        from agents.run_config import RunConfig, SandboxRunConfig
        from agents.sandbox import SandboxAgent
        from agents.sandbox.capabilities import Filesystem, Shell, Skills
        from agents.sandbox.capabilities.skills import LocalDirLazySkillSource
        from agents.sandbox.entries import LocalDir
        from agents.sandbox.sandboxes.unix_local import (
            UnixLocalSandboxClient,
        )
        from openai.types.responses import ResponseTextDeltaEvent

        _ensure_openai_init()

        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))
        model = OpenAIResponsesModel(model=options.model, openai_client=self._client)

        capabilities = [
            Shell(),
            Filesystem(),
            Skills(
                lazy_from=LocalDirLazySkillSource(
                    source=LocalDir(src=Path(options.cwd)),
                )
            ),
        ]

        manifest = _build_manifest(options.cwd)

        mcp_server_instances: list[Any] = []
        if options.mcp_servers:
            from agents.mcp import MCPServerStreamableHttp

            for server in options.mcp_servers:
                params: dict[str, Any] = {"url": server.url}
                if server.headers:
                    params["headers"] = server.headers
                mcp_srv = MCPServerStreamableHttp(name=server.name, params=params)
                mcp_server_instances.append(mcp_srv)

        agent_kwargs: dict[str, Any] = {
            "name": "lightspeed",
            "instructions": options.system_prompt,
            "model": model,
            "capabilities": capabilities,
            "default_manifest": manifest,
        }
        if mcp_server_instances:
            agent_kwargs["mcp_servers"] = mcp_server_instances

        if options.output_schema:
            agent_kwargs["output_type"] = _RawJsonSchema(options.output_schema)

        agent = SandboxAgent(**agent_kwargs)

        for srv in mcp_server_instances:
            await srv.connect()

        try:
            run_config = RunConfig(
                sandbox=SandboxRunConfig(
                    client=UnixLocalSandboxClient(),
                ),
            )

            result = Runner.run_streamed(
                agent,
                options.prompt,
                max_turns=options.max_turns,
                run_config=run_config,
            )

            async for event in result.stream_events():
                if isinstance(event, RawResponsesStreamEvent):
                    if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                        yield TextDeltaEvent(text=event.data.delta)
                elif isinstance(event, RunItemStreamEvent):
                    if isinstance(event.item, ToolCallItem):
                        raw = event.item.raw_item
                        name = (
                            getattr(raw, "name", None)
                            or (raw.get("name") if isinstance(raw, dict) else "")
                            or ""
                        )
                        args = getattr(raw, "arguments", None) or ""
                        yield ToolCallEvent(name=name, input=args[:TOOL_INPUT_MAX_CHARS])
                    elif isinstance(event.item, ToolCallOutputItem):
                        yield ToolResultEvent(
                            output=stringify(event.item.output)[:TOOL_OUTPUT_MAX_CHARS]
                        )

            yield ContentBlockStopEvent()

            usage = getattr(result, "usage", None) or {}
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)

            yield ResultEvent(
                text=stringify(result.final_output),
                cost_usd=0,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        finally:
            for srv in mcp_server_instances:
                await srv.cleanup()
