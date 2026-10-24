"""Cloud provider credential loading and SDK-managed refresh."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from azure.core.credentials_async import AsyncTokenCredential
    from google.auth.credentials import Credentials

    from lightspeed_agentic.config import (
        AzureCredentialSource,
        BedrockCredentialSource,
        VertexCredentialSource,
    )

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_TARGET_SERVICE_ACCOUNT_FILE = "target_service_account"


@dataclass(frozen=True)
class AzureModelCredentials:
    """Native Azure auth settings and the identity resource the caller must close."""

    api_key: str = field(repr=False)
    token_provider: Callable[[], Awaitable[str]] | None = field(default=None, repr=False)
    credential: AsyncTokenCredential | None = field(default=None, repr=False)

    async def aclose(self) -> None:
        """Release the native identity transport, including after failed acquisition."""
        if self.credential is not None:
            await self.credential.close()


def _configure_azure_identity_tls(request: Any) -> None:
    """Supply shared TLS through Azure's native aiohttp request options."""
    from lightspeed_agentic.tls import get_ssl_context

    request.context.options["ssl"] = get_ssl_context()


def create_azure_credentials(source: AzureCredentialSource) -> AzureModelCredentials:
    """Use the resolved key or create a lazy native Entra credential/callback.

    Azure Identity owns token acquisition, caching, refresh, and its lazy HTTP
    session. The caller owns aclose(), even if model construction or execution
    subsequently fails. This helper neither reads credentials from environment
    nor constructs a model client or acquires a token.
    """
    from lightspeed_agentic.config import AzureApiKeyCredentialSource

    if isinstance(source, AzureApiKeyCredentialSource):
        return AzureModelCredentials(api_key=source.api_key)

    from azure.identity.aio import ClientSecretCredential, get_bearer_token_provider
    from openai.lib.azure import API_KEY_SENTINEL

    credential = ClientSecretCredential(
        source.tenant_id,
        source.client_id,
        source.client_secret,
        raw_request_hook=_configure_azure_identity_tls,
    )
    return AzureModelCredentials(
        api_key=API_KEY_SENTINEL,
        token_provider=get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        ),
        credential=credential,
    )


def _read_target_service_account(source_path: Path) -> str | None:
    """Read the optional target account next to the mounted ADC credential file."""
    target_path = source_path.parent / _TARGET_SERVICE_ACCOUNT_FILE
    try:
        target = target_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("Unable to read Vertex target_service_account credential") from exc

    if not target:
        raise ValueError("Vertex target_service_account credential is empty")
    return target


def create_bedrock_credentials(
    source: BedrockCredentialSource,
    region: str,
) -> Any:
    """Create static Bedrock credentials or lazy, refreshable role credentials.

    Botocore performs STS AssumeRole on first use and refreshes temporary
    credentials before expiration. Role source keys are installed on a private
    botocore session, never in process-global environment.
    """
    import botocore.credentials  # type: ignore[import-untyped]
    import botocore.session  # type: ignore[import-untyped]

    if not source.role_arn:
        return botocore.credentials.Credentials(
            source.access_key_id,
            source.secret_access_key,
            source.session_token,
        )

    session = botocore.session.Session()
    session.set_credentials(
        source.access_key_id,
        source.secret_access_key,
        source.session_token,
    )
    source_credentials = cast(Any, session.get_credentials())

    fetcher = botocore.credentials.AssumeRoleCredentialFetcher(
        client_creator=partial(session.create_client, region_name=region),
        source_credentials=source_credentials,
        role_arn=source.role_arn,
        extra_args={"RoleSessionName": "lightspeed-agentic"},
    )
    return botocore.credentials.DeferredRefreshableCredentials(
        refresh_using=fetcher.fetch_credentials,
        method="assume-role",
    )


def _role_auth_headers(client: Any, request: Any) -> dict[str, str]:
    """Sign one Bedrock request using the current botocore role credentials."""
    from anthropic.lib.bedrock._auth import get_auth_headers

    credentials = client._role_credentials.get_frozen_credentials()
    return get_auth_headers(
        method=request.method,
        url=str(request.url),
        headers=request.headers,
        aws_access_key=credentials.access_key,
        aws_secret_key=credentials.secret_key,
        aws_session_token=credentials.token,
        region=client.aws_region,
        profile=None,
        data=request.read().decode(),
    )


def _role_assuming_bedrock_clients(
    client_class: Any,
    async_client_class: Any,
) -> tuple[Any, Any]:
    """Wrap Anthropic's clients to sign with botocore-refreshable credentials."""

    class RoleAssumingAnthropicBedrock(client_class):  # type: ignore[misc]
        def __init__(self, *args: Any, _role_credentials: Any, **kwargs: Any) -> None:
            self._role_credentials = _role_credentials
            super().__init__(*args, **kwargs)

        def _prepare_request(self, request: Any) -> None:
            request.headers.update(_role_auth_headers(self, request))

    class AsyncRoleAssumingAnthropicBedrock(async_client_class):  # type: ignore[misc]
        def __init__(self, *args: Any, _role_credentials: Any, **kwargs: Any) -> None:
            self._role_credentials = _role_credentials
            super().__init__(*args, **kwargs)

        async def _prepare_request(self, request: Any) -> None:
            from anthropic._utils import asyncify

            headers = await asyncify(_role_auth_headers)(self, request)
            request.headers.update(headers)

    return RoleAssumingAnthropicBedrock, AsyncRoleAssumingAnthropicBedrock


def configure_bedrock_credentials(
    source: BedrockCredentialSource,
    region: str,
    client_class: Any,
    async_client_class: Any,
    *,
    client_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Configure static boto environment or role-aware Bedrock SDK clients."""
    import os

    credentials = create_bedrock_credentials(source, region)
    sync_class = client_class
    async_class = async_client_class
    configured_client_kwargs = dict(client_kwargs or {})
    model_kwargs: dict[str, Any] = {}

    if source.role_arn:
        sync_class, async_class = _role_assuming_bedrock_clients(sync_class, async_class)
        configured_client_kwargs["_role_credentials"] = credentials
        # Avoid LangChain loading unrelated static credentials from process env.
        model_kwargs.update(
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
        )
    else:
        frozen_credentials = credentials.get_frozen_credentials()
        os.environ["AWS_ACCESS_KEY_ID"] = frozen_credentials.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = frozen_credentials.secret_key
        if frozen_credentials.token:
            os.environ["AWS_SESSION_TOKEN"] = frozen_credentials.token
        else:
            os.environ.pop("AWS_SESSION_TOKEN", None)

    return sync_class, async_class, configured_client_kwargs, model_kwargs


def create_vertex_credentials(source: VertexCredentialSource) -> Credentials:
    """Load mounted Vertex credentials and optionally wrap them for impersonation.

    Credential material is loaded only when the selected Vertex model is
    constructed. Google's auth library owns token refresh for both source and
    impersonated credentials.
    """
    import google.auth
    from google.auth.credentials import Credentials

    source_credentials, _project_id = google.auth.load_credentials_from_file(
        str(source.credentials_file), scopes=[_CLOUD_PLATFORM_SCOPE]
    )  # type: ignore[no-untyped-call]
    source_credentials = cast(Credentials, source_credentials)
    target_service_account = _read_target_service_account(source.credentials_file)
    if target_service_account is None:
        return source_credentials

    from google.auth import impersonated_credentials

    return impersonated_credentials.Credentials(  # type: ignore[no-untyped-call]
        source_credentials=source_credentials,
        target_principal=target_service_account,
        target_scopes=[_CLOUD_PLATFORM_SCOPE],
    )
