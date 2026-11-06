"""Vertex Gemini chat model construction for DeepAgents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from lightspeed_agentic.config import ModelParameters


def create_vertex_gemini_model(
    parameters: ModelParameters,
    reasoning_config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Construct a Vertex Gemini model with shared TLS and optional impersonation."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    from lightspeed_agentic.config import (
        VertexCredentialSource,
        VertexProviderParameters,
    )
    from lightspeed_agentic.providers.audit_support import (
        audited_model_class,
        instrument_google_client,
    )
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials
    from lightspeed_agentic.tls import get_ssl_context

    provider = cast(VertexProviderParameters, parameters.provider)
    credential_source = cast(VertexCredentialSource, parameters.credentials)
    kwargs: dict[str, Any] = {
        "model": parameters.model,
        "vertexai": True,
        "project": provider.project,
        "location": provider.region,
        "client_args": {"verify": get_ssl_context()},
    }
    if parameters.base_url is not None:
        kwargs["base_url"] = parameters.base_url
    kwargs["credentials"] = create_vertex_credentials(credential_source)
    if parameters.max_output_tokens is not None:
        kwargs["max_output_tokens"] = parameters.max_output_tokens
    if reasoning_config:
        kwargs["thinking_config"] = reasoning_config
    model_class = audited_model_class(ChatGoogleGenerativeAI)
    model = model_class(**kwargs)
    instrument_google_client(cast(Any, model).client)
    return model
