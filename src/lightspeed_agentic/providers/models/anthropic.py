"""Anthropic chat model constructors for direct, Bedrock, and Vertex hosting.

Routing inputs are explicit; native SDKs retain their credential loading and
refresh behavior. Provider SDK imports stay local to the selected constructor.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from lightspeed_agentic.providers.model_router import ModelParameters


def _tls_model_class(
    model_class: Any,
    client_class: Any,
    async_client_class: Any,
    **client_kwargs: Any,
) -> type[BaseChatModel]:
    """Wrap lazy native clients with shared TLS-configured HTTP clients."""
    from lightspeed_agentic.tls import create_async_http_client, create_http_client

    class TLSChatModel(model_class):  # type: ignore[misc]  # Dynamic optional SDK base class.
        @cached_property
        def _client(self) -> Any:
            return client_class(
                **self._client_params,
                **client_kwargs,
                http_client=create_http_client(),
            )

        @cached_property
        def _async_client(self) -> Any:
            return async_client_class(
                **self._client_params,
                **client_kwargs,
                http_client=create_async_http_client(),
            )

    return cast("type[BaseChatModel]", TLSChatModel)


def create_direct_model(
    parameters: ModelParameters,
    reasoning_config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Construct a direct Anthropic client with shared TLS and optional bearer auth."""
    from anthropic import Anthropic, AsyncAnthropic
    from langchain_anthropic import ChatAnthropic

    model_class = _tls_model_class(ChatAnthropic, Anthropic, AsyncAnthropic)
    kwargs: dict[str, Any] = {"model": parameters.model}
    if parameters.base_url is not None:
        kwargs["base_url"] = parameters.base_url
    thinking = reasoning_config.get("thinking") if reasoning_config else None
    if thinking:
        kwargs["thinking"] = thinking
    if parameters.auth_token:
        kwargs["default_headers"] = {"Authorization": f"Bearer {parameters.auth_token}"}
    return model_class(**kwargs)


def create_bedrock_model(
    parameters: ModelParameters,
    reasoning_config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Construct an Anthropic Bedrock client without changing the model/profile ID."""
    if not parameters.region:
        raise ValueError("Bedrock requires a region")

    from anthropic import AnthropicBedrock, AsyncAnthropicBedrock
    from langchain_aws import ChatAnthropicBedrock

    # The integration has no stable HTTPX injection point; keep the native-client
    # override aligned with the installed Anthropic SDK.
    model_class = _tls_model_class(
        ChatAnthropicBedrock,
        AnthropicBedrock,
        AsyncAnthropicBedrock,
        base_url=parameters.base_url,
    )
    kwargs: dict[str, Any] = {"model": parameters.model, "region_name": parameters.region}
    thinking = reasoning_config.get("thinking") if reasoning_config else None
    if thinking:
        kwargs["thinking"] = thinking
    return model_class(**kwargs)


def create_vertex_model(
    parameters: ModelParameters,
    reasoning_config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Construct an Anthropic Vertex client for the supplied project and region."""
    if not parameters.project:
        raise ValueError("Vertex requires a project")
    if not parameters.region:
        raise ValueError("Vertex requires a region")

    from langchain_google_vertexai.model_garden import ChatAnthropicVertex

    from lightspeed_agentic.tls import create_async_http_client, create_http_client

    kwargs: dict[str, Any] = {
        "model_name": parameters.model,
        "project": parameters.project,
        "location": parameters.region,
        "http_client": create_http_client(),
        "async_http_client": create_async_http_client(),
    }
    if parameters.base_url is not None:
        # Vertex SDK overrides replace its versioned default base URL entirely.
        endpoint = urlsplit(parameters.base_url)
        if endpoint.path in ("", "/"):
            endpoint = endpoint._replace(path="/v1")
        kwargs["base_url"] = endpoint.geturl()
    thinking = reasoning_config.get("thinking") if reasoning_config else None
    if thinking:
        kwargs["thinking"] = thinking
    return ChatAnthropicVertex(**kwargs)
