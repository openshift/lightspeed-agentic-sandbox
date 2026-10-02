"""MCP server configuration parsing and header resolution.

Reads LIGHTSPEED_MCP_SERVERS env var (JSON array) and resolves header values
from Kubernetes-mounted secrets and projected service account tokens.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from lightspeed_agentic.readiness import read_mounted_secret

logger = logging.getLogger("lightspeed_agentic")

SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"  # noqa: S105
MCP_SECRET_MOUNT_ROOT = "/var/secrets/mcp"  # noqa: S105


class MCPConfigError(ValueError):
    """``LIGHTSPEED_MCP_SERVERS`` is set but not valid JSON array configuration."""


class MCPServerResolutionError(ValueError):
    """A mounted credential prevents one MCP server from being configured."""


class MCPAuthClass(StrEnum):
    KUBERNETES = "kubernetes"
    NON_KUBERNETES = "non_kubernetes"


@dataclass(frozen=True)
class ResolvedMCPHeader:
    name: str
    value: str


@dataclass(frozen=True)
class ResolvedMCPServer:
    name: str
    url: str
    auth_class: MCPAuthClass = MCPAuthClass.NON_KUBERNETES
    timeout: float = 60
    headers: list[ResolvedMCPHeader] = field(default_factory=list)


@dataclass(frozen=True)
class AdmittedMCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool
    rbac_metadata: dict[str, Any] | None


@dataclass(frozen=True)
class AdmittedMCPServer:
    name: str
    url: str
    auth_class: MCPAuthClass
    timeout: float = 60
    headers: tuple[ResolvedMCPHeader, ...] = ()
    tools: tuple[AdmittedMCPTool, ...] = ()


@dataclass(frozen=True)
class AdmittedMCPProviderServer:
    """Provider-facing MCP connection with its admitted tool names."""

    name: str
    url: str
    timeout: float = 60
    headers: tuple[ResolvedMCPHeader, ...] = ()
    allowed_tool_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class MCPPolicyEntry:
    """Model-facing policy data for one admitted mutating tool."""

    server_name: str
    tool_name: str
    rbac_metadata: dict[str, Any]


def _read_secret_value(secret_path: Path) -> str:
    """Read exactly one non-empty value from a Secret mount."""
    if secret_path.is_file():
        value = read_mounted_secret(secret_path)
    elif secret_path.is_dir():
        try:
            files = sorted(
                (path for path in secret_path.iterdir() if path.is_file()),
                key=lambda path: path.name,
            )
        except OSError as exc:
            raise MCPServerResolutionError("secret directory is unreadable") from exc
        if len(files) != 1:
            raise MCPServerResolutionError("secret mount must contain exactly one file")
        value = read_mounted_secret(files[0])
    else:
        value = None

    if value is None:
        raise MCPServerResolutionError("secret value is missing or unreadable")
    return value


def _resolve_header(header: dict[str, str]) -> ResolvedMCPHeader | None:
    """Resolve one configured header from the projected token or Secret mount.

    ``Secret`` sources contain the complete header value in one mounted
    Secret. Invalid Secret mounts raise a server-level resolution error;
    service-account token absence omits only that header.
    """
    name = header["name"]
    source = header["source"]

    if source == "ServiceAccountToken":
        token = read_mounted_secret(Path(SA_TOKEN_PATH))
        if token is None:
            logger.warning("SA token not found at %s for header %s", SA_TOKEN_PATH, name)
            return None
        return ResolvedMCPHeader(name=name, value=f"Bearer {token}")

    if source == "Secret":
        secret_name = header.get("secretName", "")
        if not isinstance(secret_name, str):
            raise MCPServerResolutionError("secretName must be a string")
        root = Path(MCP_SECRET_MOUNT_ROOT).resolve()
        secret_path = (root / secret_name).resolve()
        if not secret_name or not secret_path.is_relative_to(root):
            raise MCPServerResolutionError("invalid secret path")
        return ResolvedMCPHeader(name=name, value=_read_secret_value(secret_path))

    raise MCPConfigError(
        f"LIGHTSPEED_MCP_SERVERS header {name!r} has unsupported source {source!r}"
    )


def _parse_server_entry(entry: Any, index: int) -> ResolvedMCPServer | None:
    """Validate and resolve one raw server entry.

    Authentication classification is derived from raw header sources before
    resolution, so a missing token cannot turn a Kubernetes server into a
    non-Kubernetes server.
    """
    if not isinstance(entry, dict):
        raise MCPConfigError(
            f"LIGHTSPEED_MCP_SERVERS[{index}] must be a JSON object, got {type(entry).__name__}"
        )

    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise MCPConfigError(f"LIGHTSPEED_MCP_SERVERS[{index}] missing or invalid name")

    url = entry.get("url")
    if not isinstance(url, str) or not url.strip():
        raise MCPConfigError(f"LIGHTSPEED_MCP_SERVERS[{index}] missing or invalid url")

    raw_headers = entry.get("headers")
    if raw_headers is None:
        raw_headers = []
    elif not isinstance(raw_headers, list):
        raise MCPConfigError(f"LIGHTSPEED_MCP_SERVERS[{index}] headers must be a JSON array")

    auth_class = (
        MCPAuthClass.KUBERNETES
        if any(
            isinstance(header, dict) and header.get("source") == "ServiceAccountToken"
            for header in raw_headers
        )
        else MCPAuthClass.NON_KUBERNETES
    )
    resolved_headers: list[ResolvedMCPHeader] = []
    for header_index, header in enumerate(raw_headers):
        if not isinstance(header, dict):
            raise MCPConfigError(
                f"LIGHTSPEED_MCP_SERVERS[{index}].headers[{header_index}] must be a JSON object"
            )
        if "name" not in header or "source" not in header:
            raise MCPConfigError(
                f"LIGHTSPEED_MCP_SERVERS[{index}].headers[{header_index}] "
                "must include name and source"
            )
        header_name = header["name"]
        if not isinstance(header_name, str) or not header_name.strip():
            raise MCPConfigError(
                f"LIGHTSPEED_MCP_SERVERS[{index}].headers[{header_index}] missing or invalid name"
            )
        header_source = header["source"]
        if not isinstance(header_source, str) or not header_source.strip():
            raise MCPConfigError(
                f"LIGHTSPEED_MCP_SERVERS[{index}].headers[{header_index}] missing or invalid source"
            )
        try:
            resolved = _resolve_header(header)
        except MCPServerResolutionError as exc:
            logger.warning("Omitting MCP server name=%s reason=%s", name, exc)
            return None
        if resolved is not None:
            resolved_headers.append(resolved)

    timeout = entry.get("timeout", 60)
    if not isinstance(timeout, int | float) or isinstance(timeout, bool):
        logger.warning("Invalid timeout in server %r, using default", name)
        timeout = 60

    return ResolvedMCPServer(
        name=name,
        url=url,
        auth_class=auth_class,
        timeout=timeout,
        headers=resolved_headers,
    )


def parse_mcp_servers() -> list[ResolvedMCPServer]:
    """Parse configured MCP servers and resolve their header values.

    Invalid top-level configuration raises ``MCPConfigError``. Invalid
    credential mounts are logged and cause only their server to be omitted.
    """
    raw = os.environ.get("LIGHTSPEED_MCP_SERVERS", "").strip()
    if not raw:
        return []

    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MCPConfigError(f"LIGHTSPEED_MCP_SERVERS contains invalid JSON: {exc}") from exc

    if not isinstance(entries, list):
        raise MCPConfigError(
            f"LIGHTSPEED_MCP_SERVERS must be a JSON array, got {type(entries).__name__}"
        )

    servers: list[ResolvedMCPServer] = []
    seen_names: set[str] = set()
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str) and name.strip() and name in seen_names:
                raise MCPConfigError(f"duplicate MCP server name '{name}' at index {index}")
            if isinstance(name, str) and name.strip():
                seen_names.add(name)
        server = _parse_server_entry(entry, index)
        if server is not None:
            servers.append(server)

    if servers:
        logger.info("Resolved %d MCP server(s): %s", len(servers), [s.name for s in servers])
    return servers


def _has_valid_rbac_metadata(metadata: Any) -> bool:
    """Check the supported top-level RBAC declaration shape.

    This deliberately does not interpret the contents of a declaration; the
    analysis agent resolves those contents against the eventual tool call.
    """
    if not isinstance(metadata, dict):
        return False
    if any(
        isinstance(metadata.get(key), bool) and metadata[key] for key in ("noRbac", "unbounded")
    ):
        return False

    forms = [
        metadata[key]
        for key in ("rules", "deriveFromArgs", "deriveFromManifest")
        if key in metadata
    ]
    if len(forms) != 1:
        return False
    return isinstance(forms[0], dict | list) and bool(forms[0])


def _admit_discovered_tools(
    server: ResolvedMCPServer,
    tools: list[Any],
) -> list[AdmittedMCPTool]:
    """Convert and filter one server's discovered SDK tools."""
    admitted_tools: list[AdmittedMCPTool] = []
    for tool in tools:
        metadata = getattr(tool, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
        read_only_hint = bool(metadata.get("readOnlyHint"))
        destructive_hint = bool(metadata.get("destructiveHint"))
        if server.auth_class is MCPAuthClass.KUBERNETES and read_only_hint and destructive_hint:
            logger.info(
                "Filtered MCP tool server=%s tool=%s auth_class=%s "
                "reason=contradictory_annotations",
                server.name,
                tool.name,
                server.auth_class.value,
            )
            continue
        read_only = read_only_hint and not destructive_hint
        tool_meta = metadata.get("_meta")
        rbac_metadata = tool_meta.get("openshift.io/rbac") if isinstance(tool_meta, dict) else None
        has_valid_rbac = _has_valid_rbac_metadata(rbac_metadata)
        if server.auth_class is MCPAuthClass.KUBERNETES and not (read_only or has_valid_rbac):
            logger.info(
                "Filtered MCP tool server=%s tool=%s auth_class=%s reason=missing_rbac_metadata",
                server.name,
                tool.name,
                server.auth_class.value,
            )
            continue

        input_schema = getattr(tool, "args_schema", {})
        if not isinstance(input_schema, dict):
            input_schema = input_schema.model_json_schema()
        admitted_tools.append(
            AdmittedMCPTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=input_schema,
                read_only=read_only,
                rbac_metadata=rbac_metadata if isinstance(rbac_metadata, dict) else None,
            )
        )
    return admitted_tools


async def discover_and_admit_mcp_server(
    server: ResolvedMCPServer,
) -> AdmittedMCPServer | None:
    """Discover and admit tools from one server.

    A connection or ``tools/list`` failure omits this server rather than
    failing the batch. Filtered tools are never included in the result, and a
    server with no remaining tools returns ``None``.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from lightspeed_agentic.tls import create_async_http_client

    connection = {
        "transport": "http",
        "url": server.url,
        "headers": _headers_dict(server),
        "timeout": server.timeout,
        "httpx_client_factory": create_async_http_client,
    }
    try:
        client = cast(Any, MultiServerMCPClient)({server.name: connection})
        tools = await client.get_tools(server_name=server.name)
        admitted_tools = _admit_discovered_tools(server, tools)
    except Exception as exc:
        logger.warning(
            "MCP server admission failed server=%s auth_class=%s reason=%s",
            server.name,
            server.auth_class.value,
            type(exc).__name__,
        )
        return None

    if not admitted_tools:
        logger.info(
            "Removed MCP server=%s auth_class=%s reason=no_admitted_tools",
            server.name,
            server.auth_class.value,
        )
        return None

    return AdmittedMCPServer(
        name=server.name,
        url=server.url,
        auth_class=server.auth_class,
        timeout=server.timeout,
        headers=tuple(server.headers),
        tools=tuple(admitted_tools),
    )


async def discover_and_admit_mcp_servers(
    servers: list[ResolvedMCPServer],
) -> list[AdmittedMCPServer]:
    """Discover independent MCP servers concurrently and keep admitted ones.

    ``asyncio.gather`` preserves input order; failed or empty servers are
    represented by ``None`` and removed from the returned mutable list.
    """
    results = await asyncio.gather(*(discover_and_admit_mcp_server(server) for server in servers))
    return [server for server in results if server is not None]


def split_admitted_mcp_servers(
    servers: list[AdmittedMCPServer],
) -> tuple[list[AdmittedMCPProviderServer], list[MCPPolicyEntry]]:
    """Project admitted servers into provider and model-policy views."""
    provider_servers: list[AdmittedMCPProviderServer] = []
    policies: list[MCPPolicyEntry] = []
    for server in servers:
        provider_servers.append(
            AdmittedMCPProviderServer(
                name=server.name,
                url=server.url,
                timeout=server.timeout,
                headers=server.headers,
                allowed_tool_names=tuple(tool.name for tool in server.tools),
            )
        )
        policies.extend(
            MCPPolicyEntry(
                server_name=server.name,
                tool_name=tool.name,
                rbac_metadata=tool.rbac_metadata,
            )
            for tool in server.tools
            if not tool.read_only and tool.rbac_metadata is not None
        )
    return provider_servers, policies


MCP_POLICY_CONTEXT_TEMPLATE = """\
<MCP_MUTATING_TOOL_POLICIES>
{payload}
</MCP_MUTATING_TOOL_POLICIES>"""


def render_mcp_policy_context(policies: list[MCPPolicyEntry]) -> str:
    """Render admitted mutating-tool RBAC data for the model system prompt."""
    if not policies:
        return ""

    policy_data = [
        {
            "server": policy.server_name,
            "tool": policy.tool_name,
            "rbac": policy.rbac_metadata,
        }
        for policy in policies
    ]
    payload = (
        json.dumps(policy_data, sort_keys=True).replace("<", "\\u003c").replace(">", "\\u003e")
    )
    return MCP_POLICY_CONTEXT_TEMPLATE.format(payload=payload)


def _headers_dict(server: ResolvedMCPServer | AdmittedMCPProviderServer) -> dict[str, str]:
    """Convert already-resolved headers to the SDK connection format."""
    return {h.name: h.value for h in server.headers}


def to_gemini_mcp_toolsets(servers: list[AdmittedMCPProviderServer]) -> list[Any]:
    """Convert resolved servers to Google ADK MCP toolsets."""
    from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

    from lightspeed_agentic.tls import create_async_http_client

    toolsets: list[Any] = []
    for s in servers:
        params = StreamableHTTPConnectionParams(
            url=s.url,
            headers=_headers_dict(s) if s.headers else None,
            timeout=s.timeout,
            httpx_client_factory=create_async_http_client,
        )
        toolsets.append(
            McpToolset(
                connection_params=params,
                tool_filter=list(s.allowed_tool_names),
            )
        )
    return toolsets


def to_openai_mcp_servers(servers: list[AdmittedMCPProviderServer]) -> list[Any]:
    """Convert resolved servers to OpenAI Agents MCP server instances."""
    from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams

    from lightspeed_agentic.tls import create_async_http_client

    result: list[Any] = []
    for s in servers:
        params = MCPServerStreamableHttpParams(
            url=s.url,
            timeout=s.timeout,
            httpx_client_factory=create_async_http_client,
        )
        if s.headers:
            params["headers"] = _headers_dict(s)
        result.append(
            MCPServerStreamableHttp(
                params=params,
                name=s.name,
                tool_filter={"allowed_tool_names": list(s.allowed_tool_names)},
            )
        )
    return result
