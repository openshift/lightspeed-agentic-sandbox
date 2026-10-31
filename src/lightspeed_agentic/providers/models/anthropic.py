"""Anthropic chat model constructors for direct, Bedrock, and Vertex hosting.

Routing inputs are explicit; native SDKs retain their credential loading and
refresh behavior. Provider SDK imports stay local to the selected constructor.
"""

from __future__ import annotations

from functools import cached_property
from operator import itemgetter
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from lightspeed_agentic.config import ModelParameters


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

    from lightspeed_agentic.config import AnthropicCredentialSource

    model_class = _tls_model_class(ChatAnthropic, Anthropic, AsyncAnthropic)
    credential_source = cast(AnthropicCredentialSource, parameters.credentials)
    kwargs: dict[str, Any] = {
        "model": parameters.model,
        "api_key": credential_source.api_key,
    }
    if parameters.base_url is not None:
        kwargs["base_url"] = parameters.base_url
    if parameters.max_output_tokens is not None:
        kwargs["max_tokens"] = parameters.max_output_tokens
    thinking = reasoning_config.get("thinking") if reasoning_config else None
    if thinking:
        kwargs["thinking"] = thinking
    if credential_source.auth_token:
        kwargs["default_headers"] = {"Authorization": f"Bearer {credential_source.auth_token}"}
    return model_class(**kwargs)


def create_bedrock_model(
    parameters: ModelParameters,
    reasoning_config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Construct an Anthropic Bedrock model with static or refreshable AWS credentials."""
    from lightspeed_agentic.config import BedrockCredentialSource, BedrockProviderParameters
    from lightspeed_agentic.providers.cloud_credentials import configure_bedrock_credentials

    provider = cast(BedrockProviderParameters, parameters.provider)
    credential_source = cast(BedrockCredentialSource, parameters.credentials)

    from anthropic import AnthropicBedrock, AsyncAnthropicBedrock
    from langchain_aws import ChatAnthropicBedrock

    client_class, async_client_class, client_kwargs, credential_kwargs = (
        configure_bedrock_credentials(
            credential_source,
            provider.region,
            AnthropicBedrock,
            AsyncAnthropicBedrock,
            client_kwargs={"base_url": parameters.base_url},
        )
    )

    # The integration has no stable HTTPX injection point; keep the native-client
    # override aligned with the installed Anthropic SDK.
    model_class = _tls_model_class(
        ChatAnthropicBedrock,
        client_class,
        async_client_class,
        **client_kwargs,
    )
    kwargs: dict[str, Any] = {
        "model": parameters.model,
        "region_name": provider.region,
        **credential_kwargs,
    }
    if parameters.max_output_tokens is not None:
        kwargs["max_tokens"] = parameters.max_output_tokens
    else:
        kwargs["max_tokens"] = _model_profile_max_tokens(parameters.model)
    thinking = reasoning_config.get("thinking") if reasoning_config else None
    if thinking:
        kwargs["thinking"] = thinking
    return model_class(**kwargs)


def _model_profile_max_tokens(parameters_model: str) -> int:
    """Model-profile output limit, matching direct ChatAnthropic's default.

    The Vertex and Bedrock adapters do not fill an unset output limit from the
    model profile: Vertex hard-defaults to 4096 and Bedrock leaves it to the
    client's low cap. Both silently truncate large single-response structured
    output, so apply the profile default explicitly when unset.
    """
    from langchain_anthropic.chat_models import (
        _FALLBACK_MAX_OUTPUT_TOKENS,
        _get_default_model_profile,
    )

    # Bedrock passes inference-profile IDs (for example
    # global.anthropic.claude-sonnet-5); profiles are keyed by bare model name.
    bare_model = parameters_model.split(":")[0].rsplit(".", 1)[-1]
    profile = _get_default_model_profile(bare_model) or {}
    return int(profile.get("max_output_tokens", _FALLBACK_MAX_OUTPUT_TOKENS))


def anthropic_tool_schema(schema: Any) -> dict[str, Any]:
    """Prepare the Anthropic shaping tool during schema preparation.

    Running this conversion at preparation time surfaces deterministic local
    schema errors before the agent executes tools, instead of during the
    post-run shaping pass.
    """
    from langchain_google_vertexai._anthropic_utils import convert_to_anthropic_tool

    return cast(dict[str, Any], convert_to_anthropic_tool(schema))


def bind_anthropic_output(
    model: Any,
    schema: Any,
) -> Any:
    """Shape through unforced tool calling on every Anthropic route.

    Anthropic models such as Opus 5.5 and Sonnet 5.5 reject a forced
    ``tool_choice`` with HTTP 400. The pinned LangChain adapters force it in
    ``with_structured_output`` anyway — the Vertex adapter always, and the
    direct/Bedrock adapter for every model except a hard-coded name list. The
    native ``output_config.format`` grammar is not an alternative for the
    large shaping schema: Vertex rejects oversized compiled grammars with
    HTTP 400. Bind the shaping tool without a forced choice on all Anthropic
    routes and accept either a tool call or one strict JSON object; the shared
    shaping pass validates the parsed value against the original schema.
    """
    import json

    from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough
    from langchain_google_vertexai._anthropic_parsers import ToolsOutputParser
    from langchain_google_vertexai._anthropic_utils import convert_to_anthropic_tool

    tool_parser = ToolsOutputParser(first_tool_only=True, args_only=True)

    def parse(message: Any) -> Any:
        if getattr(message, "tool_calls", None):
            return tool_parser.invoke(message)
        content = getattr(message, "content", message)
        if not isinstance(content, str):
            raise ValueError("shaping response is not a tool call or JSON text")
        return json.loads(content)

    parsed = RunnablePassthrough.assign(
        parsed=itemgetter("raw") | RunnableLambda(parse),
        parsing_error=lambda _: None,
    ).with_fallbacks(
        [RunnablePassthrough.assign(parsed=lambda _: None)], exception_key="parsing_error"
    )
    if isinstance(schema, dict) and "input_schema" in schema:
        tool: dict[str, Any] = schema
    else:
        tool = cast(dict[str, Any], convert_to_anthropic_tool(schema))
    return RunnableMap(raw=model.bind_tools([tool], tool_choice=None)) | parsed


def create_vertex_model(
    parameters: ModelParameters,
    reasoning_config: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Construct an Anthropic Vertex client for the supplied project and region."""
    from langchain_google_vertexai.model_garden import ChatAnthropicVertex

    from lightspeed_agentic.config import (
        VertexCredentialSource,
        VertexProviderParameters,
    )
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials
    from lightspeed_agentic.tls import create_async_http_client, create_http_client

    provider = cast(VertexProviderParameters, parameters.provider)
    credential_source = cast(VertexCredentialSource, parameters.credentials)
    credentials = create_vertex_credentials(credential_source)
    kwargs: dict[str, Any] = {
        "model_name": parameters.model,
        "project": provider.project,
        "location": provider.region,
        "http_client": create_http_client(),
        "async_http_client": create_async_http_client(),
    }
    kwargs["credentials"] = credentials
    if parameters.max_output_tokens is not None:
        kwargs["max_tokens"] = parameters.max_output_tokens
    else:
        kwargs["max_tokens"] = _model_profile_max_tokens(parameters.model)
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
