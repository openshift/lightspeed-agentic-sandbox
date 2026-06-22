"""FastAPI routers — mount into any FastAPI app.

Usage:
    from lightspeed_agentic.routes import build_router
    app.include_router(build_router(provider), prefix="/v1/agent")
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from fastapi import APIRouter

from lightspeed_agentic.routes.query import register_query_routes
from lightspeed_agentic.types import DEFAULT_MODEL, AgentProvider

if TYPE_CHECKING:
    from lightspeed_agentic.config import McpServerConfig

_MODEL_ENV_VARS = {
    "claude": "ANTHROPIC_MODEL",
    "gemini": "GEMINI_MODEL",
    "openai": "OPENAI_MODEL",
}


def resolve_startup_model(provider_name: str) -> str | None:
    """Startup model hint for logging and build_router; None when only defaults apply."""
    lightspeed_model = os.environ.get("LIGHTSPEED_MODEL", "").strip()
    if lightspeed_model:
        return lightspeed_model
    env_var = _MODEL_ENV_VARS.get(provider_name, "ANTHROPIC_MODEL")
    sdk_model = os.environ.get(env_var, "").strip()
    return sdk_model or None


def _resolve_router_model(provider_name: str, model: str | None = None) -> str:
    """Resolve model per configuration.md rule 5."""
    if model:
        return model
    lightspeed_model = os.environ.get("LIGHTSPEED_MODEL", "").strip()
    if lightspeed_model:
        return lightspeed_model
    env_var = _MODEL_ENV_VARS.get(provider_name, "ANTHROPIC_MODEL")
    sdk_model = os.environ.get(env_var, "").strip()
    if sdk_model:
        return sdk_model
    return DEFAULT_MODEL


def build_router(
    provider: AgentProvider,
    *,
    skills_dir: str = "/app/skills",
    model: str | None = None,
    max_turns: int = 200,
    default_timeout_ms: int = 300_000,
    audit_enabled: bool = False,
    mcp_servers: tuple[McpServerConfig, ...] = (),
) -> APIRouter:
    resolved_model = _resolve_router_model(provider.name, model)

    router = APIRouter()
    register_query_routes(
        router,
        provider=provider,
        skills_dir=skills_dir,
        model=resolved_model,
        max_turns=max_turns,
        default_timeout_ms=default_timeout_ms,
        audit_enabled=audit_enabled,
        mcp_servers=mcp_servers,
    )
    return router
