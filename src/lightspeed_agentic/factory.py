"""Provider factory — maps to lightspeed-agent/src/providers/factory.ts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lightspeed_agentic.types import AgentProvider

if TYPE_CHECKING:
    from lightspeed_agentic.providers.model_router import ResolvedModelRoute


def create_provider(name: str, model_route: ResolvedModelRoute | None = None) -> AgentProvider:
    match name:
        case "deepagents":
            if model_route is None:
                raise ValueError("DeepAgents requires a resolved model route")
            from lightspeed_agentic.providers.deepagents import DeepAgentsProvider

            return DeepAgentsProvider(model_route)
        case "gemini":
            from lightspeed_agentic.providers.gemini import GeminiProvider

            return GeminiProvider()
        case "openai":
            from lightspeed_agentic.providers.openai import OpenAIProvider

            return OpenAIProvider()
        case _:
            raise ValueError(f"Unknown provider: {name}. Supported: deepagents, gemini, openai")
