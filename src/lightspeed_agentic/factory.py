"""Provider factory — maps to lightspeed-agent/src/providers/factory.ts."""

from __future__ import annotations

from typing import Any

from lightspeed_agentic.types import AgentProvider


def create_provider(name: str, guardrails_config: Any | None = None) -> AgentProvider:
    match name:
        case "deepagents":
            from lightspeed_agentic.providers.deepagents import DeepAgentsProvider

            return DeepAgentsProvider(guardrails_config=guardrails_config)
        case "gemini":
            from lightspeed_agentic.providers.gemini import GeminiProvider

            return GeminiProvider()
        case "openai":
            from lightspeed_agentic.providers.openai import OpenAIProvider

            return OpenAIProvider()
        case _:
            raise ValueError(f"Unknown provider: {name}. Supported: deepagents, gemini, openai")
