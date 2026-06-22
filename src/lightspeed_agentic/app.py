"""FastAPI application — production entry point.

Usage: uvicorn lightspeed_agentic.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from lightspeed_agentic.config import resolve_sdk
from lightspeed_agentic.factory import create_provider
from lightspeed_agentic.health import register_health_routes, register_ready_route
from lightspeed_agentic.routes import build_router, resolve_startup_model
from lightspeed_agentic.tracing import init_tracer, shutdown_tracer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

audit_enabled = os.environ.get("LIGHTSPEED_AUDIT_ENABLED", "").strip().lower() == "true"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    init_tracer()
    yield
    shutdown_tracer()


app = FastAPI(title="lightspeed-agentic-sandbox", lifespan=lifespan)

sdk = resolve_sdk()
provider = create_provider(sdk.name)
startup_model = resolve_startup_model(sdk.name)
logger.info(
    "Starting app (sdk=%s, model=%s, LIGHTSPEED_MODEL=%s, audit=%s)",
    sdk.name,
    startup_model,
    os.environ.get("LIGHTSPEED_MODEL", ""),
    audit_enabled,
)
router = build_router(
    provider,
    skills_dir=os.environ.get("LIGHTSPEED_SKILLS_DIR", "/app/skills"),
    model=startup_model,
    audit_enabled=audit_enabled,
    mcp_servers=sdk.mcp_servers,
)
app.include_router(router, prefix="/v1/agent")

register_health_routes(app)
register_ready_route(app, sdk=sdk)
