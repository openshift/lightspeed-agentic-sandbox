"""Health and readiness probe handlers."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from lightspeed_agentic.config import McpServerConfig, ResolvedSDK

PROBE_TIMEOUT_SEC = 3.0


def health_payload() -> dict[str, str]:
    return {"status": "ok"}


def register_health_routes(app: FastAPI) -> None:
    """Register GET /health (liveness)."""

    @app.get("/health")
    def health() -> dict[str, str]:
        return health_payload()


def check_provider_env(expected_envs: tuple[str, ...]) -> str:
    """R1: all required credential env var(s) must be present and non-empty."""
    missing = [var for var in expected_envs if not os.environ.get(var, "").strip()]
    if not missing:
        return "ok"
    return f"error: missing {', '.join(missing)}"


def probe_provider_endpoint(url: str, timeout: float = PROBE_TIMEOUT_SEC) -> str:
    """R2: HTTP GET; any HTTP response (including 4xx) means reachable."""
    scheme = urlparse(url).scheme
    if scheme not in ("https", "http"):
        return f"error: unsupported URL scheme {scheme!r}"
    try:
        request = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(request, timeout=timeout):  # noqa: S310
            return "ok"
    except urllib.error.HTTPError:
        return "ok"
    except Exception as exc:
        return f"error: {exc}"


def check_provider_endpoint(probe_url: str) -> str:
    url = probe_url.strip()
    if not url:
        return "error: empty probe URL"
    return probe_provider_endpoint(url)


def check_mcp_endpoints(
    servers: tuple[McpServerConfig, ...],
    timeout: float = PROBE_TIMEOUT_SEC,
) -> dict[str, str]:
    """R3: probe each configured MCP server URL for reachability."""
    results: dict[str, str] = {}
    for server in servers:
        results[f"mcp_{server.name}"] = probe_provider_endpoint(server.url, timeout)
    return results


def run_readiness_checks(sdk: ResolvedSDK) -> tuple[bool, dict[str, str]]:
    checks = {
        "provider_env": check_provider_env(sdk.expected_envs),
        "provider_endpoint": check_provider_endpoint(sdk.probe_url),
    }
    if sdk.mcp_servers:
        checks.update(check_mcp_endpoints(sdk.mcp_servers))
    return all(status == "ok" for status in checks.values()), checks


def ready_response(sdk: ResolvedSDK) -> tuple[int, dict[str, object]]:
    ok, checks = run_readiness_checks(sdk)
    if ok:
        return 200, {"status": "ok"}
    return 503, {"status": "error", "checks": checks}


def register_ready_route(app: FastAPI, *, sdk: ResolvedSDK) -> None:
    """Register GET /ready (readiness)."""

    @app.get("/ready")
    def ready() -> JSONResponse:
        status_code, body = ready_response(sdk)
        return JSONResponse(status_code=status_code, content=body)
