"""MCP server configuration parsing and header resolution.

Reads LIGHTSPEED_MCP_SERVERS env var (JSON array) and resolves header values
from Kubernetes-mounted secrets and projected service account tokens.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("lightspeed_agentic")

SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"  # noqa: S105
MCP_SECRET_MOUNT_ROOT = "/var/secrets/mcp"  # noqa: S105


@dataclass(frozen=True)
class ResolvedMCPHeader:
    name: str
    value: str


@dataclass(frozen=True)
class ResolvedMCPServer:
    name: str
    url: str
    timeout: int = 60
    headers: list[ResolvedMCPHeader] = field(default_factory=list)
    ca_file: str | None = None


def _resolve_header(header: dict[str, str]) -> ResolvedMCPHeader | None:
    """Resolve a single header entry based on its source type."""
    name = header["name"]
    source = header["source"]

    if source == "ServiceAccountToken":
        try:
            token = Path(SA_TOKEN_PATH).read_text().strip()
        except OSError:
            logger.warning("SA token not found at %s for header %s", SA_TOKEN_PATH, name)
            return None
        return ResolvedMCPHeader(name=name, value=f"Bearer {token}")

    if source == "Secret":
        secret_name = header.get("secretName", "")
        if not isinstance(secret_name, str):
            logger.warning("secretName must be a string for header %s", name)
            return None
        root = Path(MCP_SECRET_MOUNT_ROOT).resolve()
        secret_dir = (root / secret_name).resolve()
        if not secret_name or not secret_dir.is_relative_to(root):
            logger.warning("Invalid secret path: %s for header %s", secret_dir, name)
            return None
        if not secret_dir.is_dir():
            logger.warning("Secret dir not found: %s for header %s", secret_dir, name)
            return None
        try:
            files = sorted((f for f in secret_dir.iterdir() if f.is_file()), key=lambda f: f.name)
        except OSError:
            logger.warning("Cannot list secret dir %s for header %s", secret_dir, name)
            return None
        if not files:
            logger.warning("No files in secret dir %s for header %s", secret_dir, name)
            return None
        try:
            value = files[0].read_text().strip()
        except OSError:
            logger.warning("Cannot read secret file %s for header %s", files[0], name)
            return None
        return ResolvedMCPHeader(name=name, value=value)

    if source == "Client":
        return None

    logger.warning("Unknown header source %r for header %s, skipping", source, name)
    return None


def _resolve_ca_file(raw: Any) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("caFile must be a non-empty absolute path")

    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError("caFile must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("caFile must reference a regular file")
        context = ssl.create_default_context()
        context.load_verify_locations(cafile=str(resolved))
    except (OSError, ssl.SSLError) as exc:
        raise ValueError("caFile is missing, unreadable, or invalid") from exc
    return str(resolved)


def mcp_http_client_factory(
    ca_file: str | None,
) -> Callable[..., httpx.AsyncClient] | None:
    """Return an MCP HTTP client factory that adds a server-specific CA."""
    if ca_file is None:
        return None

    context = ssl.create_default_context()
    context.load_verify_locations(cafile=ca_file)

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            auth=auth,
            verify=context,
            follow_redirects=False,
        )

    return factory


def parse_mcp_servers() -> list[ResolvedMCPServer]:
    """Parse LIGHTSPEED_MCP_SERVERS env var and resolve all header values."""
    raw = os.environ.get("LIGHTSPEED_MCP_SERVERS", "").strip()
    if not raw:
        return []

    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Invalid JSON in LIGHTSPEED_MCP_SERVERS")
        return []

    if not isinstance(entries, list):
        logger.error("LIGHTSPEED_MCP_SERVERS must be a JSON array")
        return []

    servers: list[ResolvedMCPServer] = []
    for entry in entries:
        if not isinstance(entry, dict) or "name" not in entry or "url" not in entry:
            logger.warning("Skipping invalid MCP server entry: %r", entry)
            continue
        resolved_headers: list[ResolvedMCPHeader] = []
        raw_headers = entry.get("headers") or []
        if not isinstance(raw_headers, list):
            logger.warning("headers is not a list in server %r, skipping", entry["name"])
            raw_headers = []
        for h in raw_headers:
            if not isinstance(h, dict) or "name" not in h or "source" not in h:
                logger.warning("Skipping invalid header in server %r: %r", entry.get("name"), h)
                continue
            resolved = _resolve_header(h)
            if resolved is not None:
                resolved_headers.append(resolved)

        timeout = entry.get("timeout", 60)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            logger.warning("Invalid timeout in server %r, using default", entry["name"])
            timeout = 60

        try:
            ca_file = _resolve_ca_file(entry.get("caFile"))
        except ValueError as exc:
            logger.error("Skipping MCP server %r: %s", entry["name"], exc)
            continue

        servers.append(
            ResolvedMCPServer(
                name=entry["name"],
                url=entry["url"],
                timeout=timeout,
                headers=resolved_headers,
                ca_file=ca_file,
            )
        )

    if servers:
        logger.info("Resolved %d MCP server(s): %s", len(servers), [s.name for s in servers])
    return servers


def _headers_dict(server: ResolvedMCPServer) -> dict[str, str]:
    return {h.name: h.value for h in server.headers}


def to_gemini_mcp_toolsets(servers: list[ResolvedMCPServer]) -> list[Any]:
    """Convert to google-adk McpToolset instances."""
    from google.adk.tools.mcp_tool.mcp_toolset import (  # type: ignore[attr-defined]
        McpToolset,
        StreamableHTTPConnectionParams,
    )

    toolsets: list[Any] = []
    for s in servers:
        params_kwargs: dict[str, Any] = {
            "url": s.url,
            "headers": _headers_dict(s) if s.headers else None,
            "timeout": float(s.timeout),
        }
        client_factory = mcp_http_client_factory(s.ca_file)
        if client_factory is not None:
            params_kwargs["httpx_client_factory"] = client_factory
        params = StreamableHTTPConnectionParams(**params_kwargs)
        toolsets.append(McpToolset(connection_params=params))
    return toolsets


def to_openai_mcp_servers(servers: list[ResolvedMCPServer]) -> list[Any]:
    """Convert to openai-agents MCPServerStreamableHttp instances."""
    from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams

    result: list[Any] = []
    for s in servers:
        params = MCPServerStreamableHttpParams(url=s.url, timeout=float(s.timeout))
        if s.headers:
            params["headers"] = _headers_dict(s)
        client_factory = mcp_http_client_factory(s.ca_file)
        if client_factory is not None:
            params["httpx_client_factory"] = client_factory
        result.append(MCPServerStreamableHttp(params=params, name=s.name))
    return result
