"""Offline Azure credential helpers with real SDK acquisition and refresh."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def azure_identity_http(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Script only the HTTP boundary; keep Azure's pipeline/cache and sessions real."""
    import aiohttp
    from aiohttp.helpers import TimerNoop
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL

    from lightspeed_agentic import tls

    state = SimpleNamespace(clock=2_000_000_000, responses=[], requests=[], sessions=[])
    monkeypatch.setattr(time, "time", lambda: state.clock)
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    monkeypatch.setattr(tls, "_runtime_ssl_context", context)
    state.ssl_context = context
    monkeypatch.delenv("AZURE_AUTHORITY_HOST", raising=False)
    for name in (
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "AZURE_CLIENT_ID",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_SECRET",
    ):
        monkeypatch.setenv(name, "ambient-must-not-win")

    def reject_network(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Unscripted network access in Azure credential test")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)
    original_init = aiohttp.ClientSession.__init__

    def track_session(session: Any, *args: Any, **kwargs: Any) -> None:
        original_init(session, *args, **kwargs)
        state.sessions.append(session)

    async def request(_session: Any, method: str, url: str, **kwargs: Any) -> Any:
        state.requests.append((method, str(url), kwargs))
        assert state.responses, "Unexpected token HTTP request"
        scripted = state.responses.pop(0)
        await asyncio.sleep(0)  # Exercise concurrent callbacks through the SDK lock.
        if isinstance(scripted, BaseException):
            raise scripted
        status, payload = scripted
        response = aiohttp.ClientResponse(
            method,
            URL(url),
            writer=None,
            continue100=None,
            timer=TimerNoop(),
            request_info=None,
            traces=[],
            loop=asyncio.get_running_loop(),
            session=_session,
            stream_writer=SimpleNamespace(output_size=0),
        )
        response.status = status
        response.reason = "OK" if status == 200 else "Unauthorized"
        response._headers = CIMultiDictProxy(CIMultiDict({"Content-Type": "application/json"}))
        response._body = json.dumps(payload).encode()
        return response

    monkeypatch.setattr(aiohttp.ClientSession, "__init__", track_session)
    monkeypatch.setattr(aiohttp.ClientSession, "_request", request)
    return state


@pytest.mark.asyncio
async def test_azure_api_key_helper_uses_only_resolved_key(azure_identity_http: Any) -> None:
    from lightspeed_agentic.config import AzureApiKeyCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

    environment = dict(os.environ)
    auth = create_azure_credentials(AzureApiKeyCredentialSource("resolved-azure-key"))
    assert auth.api_key == "resolved-azure-key"
    assert auth.token_provider is None
    assert auth.credential is None
    assert "resolved-azure-key" not in repr(auth)
    await auth.aclose()
    assert dict(os.environ) == environment
    assert azure_identity_http.requests == []
    assert azure_identity_http.sessions == []


def test_azure_entra_helper_constructs_lazily_without_event_loop(azure_identity_http: Any) -> None:
    from azure.identity.aio import ClientSecretCredential

    from lightspeed_agentic.config import AzureEntraCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

    auth = create_azure_credentials(
        AzureEntraCredentialSource("resolved-tenant", "resolved-client", "resolved-secret")
    )
    assert isinstance(auth.credential, ClientSecretCredential)
    assert auth.token_provider is not None
    assert auth.api_key == "<missing API key>"
    assert all(
        value not in repr(auth)
        for value in ("resolved-tenant", "resolved-client", "resolved-secret")
    )
    assert azure_identity_http.requests == []
    assert azure_identity_http.sessions == []
    asyncio.run(auth.aclose())
    asyncio.run(auth.aclose())
    assert azure_identity_http.sessions == []


@pytest.mark.asyncio
async def test_azure_native_token_callback_refreshes_expired_token(
    azure_identity_http: Any,
) -> None:
    from lightspeed_agentic.config import AzureEntraCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

    http = azure_identity_http
    environment = dict(os.environ)
    http.responses.extend(
        [
            (
                200,
                {"access_token": "identity-token-one", "expires_in": 3600, "token_type": "Bearer"},
            ),
            (
                200,
                {"access_token": "identity-token-two", "expires_in": 3600, "token_type": "Bearer"},
            ),
        ]
    )
    auth = create_azure_credentials(
        AzureEntraCredentialSource("resolved-tenant", "resolved-client", "resolved-secret")
    )
    assert auth.token_provider is not None
    try:
        assert (
            await asyncio.gather(*(auth.token_provider() for _ in range(4)))
            == ["identity-token-one"] * 4
        )
        assert len(http.requests) == 1
        http.clock += 3601
        assert await auth.token_provider() == "identity-token-two"
        assert await auth.token_provider() == "identity-token-two"
        assert len(http.requests) == 2
        for method, url, kwargs in http.requests:
            assert method == "POST"
            assert url == "https://login.microsoftonline.com/resolved-tenant/oauth2/v2.0/token"
            assert kwargs["data"]["client_id"] == "resolved-client"
            assert kwargs["data"]["client_secret"] == "resolved-secret"  # noqa: S105 - offline fixture
            assert kwargs["data"]["grant_type"] == "client_credentials"
            assert kwargs["data"]["scope"] == "https://cognitiveservices.azure.com/.default"
            assert kwargs["ssl"] is http.ssl_context
            assert kwargs["ssl"].check_hostname
            assert kwargs["ssl"].verify_mode == ssl.CERT_REQUIRED
            assert "api-key" not in kwargs["headers"]
            assert "Authorization" not in kwargs["headers"]
        assert len(http.sessions) == 1
        assert not http.sessions[0].closed
    finally:
        await auth.aclose()
    assert http.sessions[0].closed
    assert dict(os.environ) == environment
    await auth.aclose()


@pytest.mark.parametrize("status", [400, 401, 403])
@pytest.mark.parametrize(
    "after_expiry", [False, True], ids=["initial-acquisition", "expired-refresh"]
)
@pytest.mark.asyncio
async def test_azure_token_rejection_propagates_without_api_key_fallback(
    azure_identity_http: Any, status: int, after_expiry: bool
) -> None:
    from azure.core.exceptions import ClientAuthenticationError

    from lightspeed_agentic.config import AzureEntraCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

    http = azure_identity_http
    if after_expiry:
        http.responses.append(
            (
                200,
                {"access_token": "identity-token-one", "expires_in": 3600, "token_type": "Bearer"},
            )
        )
    http.responses.append(
        (status, {"error": "invalid_client", "error_description": "offline credential rejected"})
    )
    auth = create_azure_credentials(
        AzureEntraCredentialSource("resolved-tenant", "resolved-client", "resolved-secret")
    )
    assert auth.token_provider is not None
    try:
        if after_expiry:
            assert await auth.token_provider() == "identity-token-one"
            http.clock += 3601
        with pytest.raises(ClientAuthenticationError, match="offline credential rejected") as exc:
            await auth.token_provider()
        assert all(
            value not in str(exc.value) for value in ("resolved-secret", "ambient-must-not-win")
        )
        assert auth.api_key == "<missing API key>"
        assert len(http.requests) == (2 if after_expiry else 1)
        assert http.responses == []
    finally:
        await auth.aclose()
    assert len(http.sessions) == 1
    assert http.sessions[0].closed


@pytest.mark.asyncio
async def test_azure_credential_can_close_after_cancelled_acquisition(
    azure_identity_http: Any,
) -> None:
    from lightspeed_agentic.config import AzureEntraCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

    http = azure_identity_http
    http.responses.append(asyncio.CancelledError())
    auth = create_azure_credentials(
        AzureEntraCredentialSource("resolved-tenant", "resolved-client", "resolved-secret")
    )
    assert auth.token_provider is not None
    try:
        with pytest.raises(asyncio.CancelledError):
            await auth.token_provider()
    finally:
        await auth.aclose()
    assert len(http.requests) == 1
    assert len(http.sessions) == 1
    assert http.sessions[0].closed


def test_azure_invalid_tenant_fails_without_allocating_session(azure_identity_http: Any) -> None:
    from lightspeed_agentic.config import AzureEntraCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

    with pytest.raises(ValueError, match="tenant") as exc:
        create_azure_credentials(
            AzureEntraCredentialSource("invalid/tenant", "resolved-client", "resolved-secret")
        )
    assert "resolved-secret" not in str(exc.value)
    assert azure_identity_http.requests == []
    assert azure_identity_http.sessions == []


def test_azure_api_key_helper_does_not_import_optional_sdks() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = r"""
import asyncio
import builtins

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if any(name == package or name.startswith(package + ".")
           for package in ("azure", "aiohttp", "openai", "google", "botocore")):
        raise AssertionError(f"Optional SDK import attempted: {name}")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from lightspeed_agentic.config import AzureApiKeyCredentialSource
from lightspeed_agentic.providers.cloud_credentials import create_azure_credentials

auth = create_azure_credentials(AzureApiKeyCredentialSource("resolved-azure-key"))
assert auth.api_key == "resolved-azure-key"
assert auth.credential is None
assert auth.token_provider is None
asyncio.run(auth.aclose())
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(  # noqa: S603 - controlled inline import-isolation test
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
