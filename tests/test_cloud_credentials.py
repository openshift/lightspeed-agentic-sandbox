"""Tests for cloud credential loading and refresh."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import pytest


def _mounted_credentials(tmp_path: Any, target: str | None = None) -> None:
    source_file = tmp_path / "GOOGLE_APPLICATION_CREDENTIALS"
    source_file.write_text("{}", encoding="utf-8")
    if target is not None:
        (tmp_path / "target_service_account").write_text(target, encoding="utf-8")


def test_vertex_credentials_load_only_when_the_vertex_route_constructs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth
    from google.oauth2.credentials import Credentials

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    source = Credentials(token="source-token")  # noqa: S106 - Fake offline credential.
    credential_file = tmp_path / "GOOGLE_APPLICATION_CREDENTIALS"
    credential_file.write_text("invalid JSON is not parsed during routing", encoding="utf-8")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        lambda _filename, **_kwargs: (source, "source-project"),
    )

    credentials = create_vertex_credentials(VertexCredentialSource(credential_file))

    assert credentials is source


def test_without_target_loads_the_configured_adc_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth
    from google.oauth2.credentials import Credentials

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path)
    source = Credentials(token="source-token")  # noqa: S106 - Fake offline credential.
    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        lambda _filename, **_kwargs: (source, "source-project"),
    )

    credentials = create_vertex_credentials(
        VertexCredentialSource(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
    )

    assert credentials is source


def test_target_creates_impersonated_credentials_from_adc_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth
    from google.auth import credentials, impersonated_credentials

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    target = "vertex-agent@example.iam.gserviceaccount.com"
    _mounted_credentials(tmp_path, target)
    source = credentials.AnonymousCredentials()
    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        lambda _filename, **_kwargs: (source, "source-project"),
    )

    impersonated = create_vertex_credentials(
        VertexCredentialSource(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
    )

    assert isinstance(impersonated, impersonated_credentials.Credentials)
    assert impersonated.service_account_email == target


def test_empty_target_fails_instead_of_using_source_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth
    from google.auth import credentials

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path, "   \n")
    source = credentials.AnonymousCredentials()
    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        lambda _filename, **_kwargs: (source, "source-project"),
    )

    with pytest.raises(ValueError, match=r"target_service_account.*empty"):
        create_vertex_credentials(
            VertexCredentialSource(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
        )


def test_adc_failure_propagates_when_target_is_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path, "vertex-agent@example.iam.gserviceaccount.com")
    failure = RuntimeError("source credentials unavailable")

    def fail_file_load(_filename: str, **_kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(google.auth, "load_credentials_from_file", fail_file_load)

    with pytest.raises(RuntimeError, match="source credentials unavailable"):
        create_vertex_credentials(
            VertexCredentialSource(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
        )


def test_google_auth_refreshes_the_returned_impersonated_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth
    from google.auth import impersonated_credentials
    from google.oauth2.credentials import Credentials

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path, "vertex-agent@example.iam.gserviceaccount.com")
    source = Credentials(
        token="source-token",  # noqa: S106 - Fake source credential for offline refresh test.
        expiry=datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1),
    )
    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        lambda _filename, **_kwargs: (source, "source-project"),
    )
    requests: list[tuple[str, str, bytes, dict[str, str]]] = []

    class Response:
        status = 200
        data = json.dumps(
            {
                "accessToken": "target-token",
                "expireTime": (datetime.now(UTC) + timedelta(hours=1)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            }
        ).encode("utf-8")
        headers: ClassVar[dict[str, str]] = {}

    def fake_request(url: str, method: str, *, body: bytes, headers: dict[str, str]) -> Response:
        requests.append((url, method, body, headers))
        return Response()

    impersonated = create_vertex_credentials(
        VertexCredentialSource(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
    )
    assert isinstance(impersonated, impersonated_credentials.Credentials)

    impersonated.refresh(fake_request)

    assert impersonated.token == "target-token"  # noqa: S105 - Fake token from the mock response.
    assert len(requests) == 1
    url, method, body, headers = requests[0]
    assert method == "POST"
    assert url.endswith(
        "/v1/projects/-/serviceAccounts/vertex-agent@example.iam.gserviceaccount.com:generateAccessToken"
    )
    assert json.loads(body) == {
        "delegates": None,
        "scope": ["https://www.googleapis.com/auth/cloud-platform"],
        "lifetime": "3600s",
    }
    assert headers["authorization"] == "Bearer source-token"


def test_impersonation_refresh_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import google.auth
    from google.auth import exceptions, impersonated_credentials
    from google.oauth2.credentials import Credentials

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    _mounted_credentials(tmp_path, "vertex-agent@example.iam.gserviceaccount.com")
    source = Credentials(
        token="source-token",  # noqa: S106 - Fake source credential in an offline failure test.
        expiry=datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1),
    )
    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        lambda _filename, **_kwargs: (source, "source-project"),
    )

    class Response:
        status = 403
        data = b'{"error":"permissionDenied"}'
        headers: ClassVar[dict[str, str]] = {}

    def forbidden_request(*_args: Any, **_kwargs: Any) -> Response:
        return Response()

    impersonated = create_vertex_credentials(
        VertexCredentialSource(tmp_path / "GOOGLE_APPLICATION_CREDENTIALS")
    )
    assert isinstance(impersonated, impersonated_credentials.Credentials)

    with pytest.raises(exceptions.RefreshError):
        impersonated.refresh(forbidden_request)


def test_bedrock_credentials_without_role_return_static_botocore_credentials() -> None:
    from botocore.credentials import Credentials, RefreshableCredentials

    from lightspeed_agentic.config import BedrockCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_bedrock_credentials

    credentials = create_bedrock_credentials(
        BedrockCredentialSource(
            access_key_id="static-access",
            secret_access_key="static-secret",  # noqa: S106 - test credential
            session_token="static-session",  # noqa: S106 - test credential
        ),
        "us-east-1",
    )

    assert isinstance(credentials, Credentials)
    assert not isinstance(credentials, RefreshableCredentials)
    frozen = credentials.get_frozen_credentials()
    assert frozen.access_key == "static-access"
    assert frozen.secret_key == "static-secret"  # noqa: S105
    assert frozen.token == "static-session"  # noqa: S105


def test_bedrock_credentials_with_role_assume_and_refresh(monkeypatch: Any) -> None:
    import botocore.credentials
    import botocore.session

    from lightspeed_agentic.config import BedrockCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_bedrock_credentials

    now = datetime.now(UTC)
    clock = [now]
    assume_calls: list[dict[str, Any]] = []

    class FakeSTS:
        def assume_role(self, **kwargs: Any) -> dict[str, Any]:
            assume_calls.append(kwargs)
            suffix = str(len(assume_calls))
            return {
                "Credentials": {
                    "AccessKeyId": f"assumed-access-{suffix}",
                    "SecretAccessKey": f"assumed-secret-{suffix}",
                    "SessionToken": f"assumed-token-{suffix}",
                    "Expiration": clock[0] + timedelta(hours=1),
                }
            }

    client_calls: list[tuple[str, dict[str, Any]]] = []

    def create_client(_session: Any, service_name: str, **kwargs: Any) -> FakeSTS:
        client_calls.append((service_name, kwargs))
        return FakeSTS()

    monkeypatch.setattr(botocore.session.Session, "create_client", create_client)
    monkeypatch.setattr(botocore.credentials, "_local_now", lambda: clock[0])
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ambient-access-must-not-win")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-secret-must-not-win")
    source = BedrockCredentialSource(
        access_key_id="source-access",
        secret_access_key="source-secret",  # noqa: S106 - test credential
        session_token="source-session",  # noqa: S106 - test credential
        role_arn="arn:aws:iam::123456789012:role/target",
    )

    credentials = create_bedrock_credentials(source, "us-west-2")
    credentials._time_fetcher = lambda: clock[0]
    first = credentials.get_frozen_credentials()
    clock[0] += timedelta(minutes=50)
    second = credentials.get_frozen_credentials()

    assert first.access_key == "assumed-access-1"
    assert first.token == "assumed-token-1"  # noqa: S105
    assert second.access_key == "assumed-access-2"
    assert second.token == "assumed-token-2"  # noqa: S105
    assert len(assume_calls) == 2
    assert assume_calls[0]["RoleArn"] == "arn:aws:iam::123456789012:role/target"
    assert client_calls[0] == (
        "sts",
        {
            "aws_access_key_id": "source-access",
            "aws_secret_access_key": "source-secret",
            "aws_session_token": "source-session",
            "region_name": "us-west-2",
        },
    )


def test_bedrock_role_assumption_error_propagates_without_fallback(
    monkeypatch: Any,
) -> None:
    import botocore.session
    from botocore.exceptions import ClientError

    from lightspeed_agentic.config import BedrockCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_bedrock_credentials

    assume_calls: list[dict[str, Any]] = []

    class FakeSTS:
        def assume_role(self, **kwargs: Any) -> dict[str, Any]:
            assume_calls.append(kwargs)
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "role assumption denied"}},
                "AssumeRole",
            )

    def create_client(_session: Any, _service_name: str, **_kwargs: Any) -> FakeSTS:
        return FakeSTS()

    monkeypatch.setattr(botocore.session.Session, "create_client", create_client)
    source = BedrockCredentialSource(
        access_key_id="source-access",
        secret_access_key="source-secret",  # noqa: S106 - test credential
        role_arn="arn:aws:iam::123456789012:role/target",
    )
    credentials = create_bedrock_credentials(source, "us-west-2")

    with pytest.raises(ClientError) as exc_info:
        credentials.get_frozen_credentials()

    assert exc_info.value.response["Error"]["Code"] == "AccessDenied"
    assert len(assume_calls) == 1
    assert assume_calls[0]["RoleArn"] == "arn:aws:iam::123456789012:role/target"


@pytest.fixture
def auth_wire(monkeypatch, capsys):
    """Keep SDK/cache/retry/HTTP response handling real; script the pool I/O only."""
    monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
    import io
    import socket
    import time
    from types import SimpleNamespace

    import urllib3

    state = SimpleNamespace(responses=[], requests=[], events=[])
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def capture():
        state.events.extend(json.loads(line) for line in capsys.readouterr().out.splitlines())
        return state.events

    def network(pool, _conn, method, url, **kwargs):
        events = capture()
        # Removing credential/transport observation must fail before any external I/O.
        assert events
        assert events[-1]["event"] == "outbound.request.started"
        state.requests.append((pool.host, method, url, kwargs))
        assert state.responses, "Unscripted credential request"
        scripted = state.responses.pop(0)
        if isinstance(scripted, BaseException):
            raise scripted
        if isinstance(scripted, urllib3.response.HTTPResponse):
            return scripted
        status, data = scripted
        if not isinstance(data, bytes):
            data = json.dumps(data).encode()
        return urllib3.response.HTTPResponse(
            body=io.BytesIO(data),
            status=status,
            headers={"Content-Type": "application/json", "Content-Length": str(len(data))},
            preload_content=False,
        )

    def reject(*_args, **_kwargs):
        pytest.fail("Unscripted auth socket/DNS access")

    def prepare_proxy(_pool, _conn):
        events = capture()
        assert events
        assert events[-1]["event"] == "outbound.request.started"
        if getattr(state, "proxy_error", None) is not None:
            raise state.proxy_error

    monkeypatch.setattr(urllib3.connectionpool.HTTPSConnectionPool, "_prepare_proxy", prepare_proxy)
    monkeypatch.setattr(urllib3.connectionpool.HTTPConnectionPool, "_make_request", network)
    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket, "getaddrinfo", reject)
    state.capture = capture
    return state


def _google_adc(tmp_path, kind):
    path = tmp_path / "GOOGLE_APPLICATION_CREDENTIALS"
    if kind == "authorized_user":
        data = {
            "type": kind,
            "client_id": "PRIVATE_CLIENT",
            "client_secret": "PRIVATE_SECRET",
            "refresh_token": "PRIVATE_REFRESH",
        }
    elif kind == "service_account":
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        data = {
            "type": kind,
            "project_id": "project",
            "client_email": "PRIVATE@example.com",
            "token_uri": "https://oauth2.googleapis.com/token",
            "private_key": key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ).decode(),
        }
    else:
        data = {
            "type": "external_account",
            "audience": (
                "//iam.googleapis.com/projects/123/locations/global/"
                "workloadIdentityPools/pool/providers/provider"
            ),
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "token_url": "https://sts.googleapis.com/v1/token",
            "credential_source": {"url": "http://metadata.example/PRIVATE_SOURCE?secret=PRIVATE"},
        }
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("kind", ["authorized_user", "service_account", "external_account"])
@pytest.mark.parametrize("impersonate", [False, True])
def test_google_native_refresh_http_audit_and_cache(
    auth_wire, monkeypatch, tmp_path, kind, impersonate
):
    from google.auth import _helpers
    from google.auth.transport.requests import Request

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    clock = [datetime.now(UTC).replace(tzinfo=None)]
    monkeypatch.setattr(_helpers, "utcnow", lambda: clock[0])
    path = _google_adc(tmp_path, kind)
    if impersonate:
        (tmp_path / "target_service_account").write_text("PRIVATE_TARGET@example.com")
    for _ in range(2):
        if kind == "external_account":
            auth_wire.responses.append((200, b"PRIVATE_SUBJECT"))
        auth_wire.responses.append(
            (200, {"access_token": "PRIVATE_TOKEN", "expires_in": 3600, "token_type": "Bearer"})
        )
        # google-auth resolves the external-account project during native ADC loading.
        if kind == "external_account" and _ == 0:
            auth_wire.responses.append((200, {"projectId": "project"}))
            if impersonate:
                # Native impersonation scopes a copy of the source; its token is reset.
                auth_wire.responses.extend(
                    [
                        (200, b"PRIVATE_SUBJECT"),
                        (
                            200,
                            {
                                "access_token": "PRIVATE_TOKEN",
                                "expires_in": 3600,
                                "token_type": "Bearer",
                            },
                        ),
                    ]
                )
        if impersonate:
            auth_wire.responses.append(
                (
                    200,
                    {
                        "accessToken": "PRIVATE_TARGET_TOKEN",
                        "expireTime": (clock[0] + timedelta(hours=1, seconds=3601 * _)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                    },
                )
            )
    credentials = create_vertex_credentials(VertexCredentialSource(path))
    request = Request()
    try:
        for _cycle in range(2):
            headers = {}
            credentials.before_request(request, "POST", "https://model.example", headers)
            assert headers["authorization"] == (
                "Bearer PRIVATE_TARGET_TOKEN" if impersonate else "Bearer PRIVATE_TOKEN"
            )
            count = len(auth_wire.requests)
            credentials.before_request(request, "POST", "https://model.example", {})
            assert len(auth_wire.requests) == count  # Native cache hit emits no request.
            clock[0] += timedelta(seconds=3601)
    finally:
        request.session.close()
    events = auth_wire.capture()
    starts = [row for row in events if row["event"] == "outbound.request.started"]
    ends = [row for row in events if row["event"] == "outbound.request.finished"]
    assert len(starts) == len(ends) == len(auth_wire.requests)
    assert auth_wire.responses == []
    assert {row["request.kind"] for row in starts} == {"credential"}
    assert all(row["outcome"] == "success" for row in ends)
    assert all("request.body.size" not in row for row in events)
    assert len({row["attempt_id"] for row in starts}) == len(starts)
    assert "PRIVATE" not in json.dumps(events)


def _sts_reply(expiry):
    return (
        f'<AssumeRoleResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">'
        "<AssumeRoleResult><Credentials><AccessKeyId>PRIVATE_ACCESS</AccessKeyId>"
        "<SecretAccessKey>PRIVATE_SECRET</SecretAccessKey><SessionToken>PRIVATE_TOKEN</SessionToken>"
        f"<Expiration>{expiry.strftime('%Y-%m-%dT%H:%M:%SZ')}</Expiration></Credentials>"
        "</AssumeRoleResult><ResponseMetadata><RequestId>PRIVATE_ID</RequestId></ResponseMetadata>"
        "</AssumeRoleResponse>"
    ).encode()


def test_sts_native_refresh_http_audit_and_cache(auth_wire, monkeypatch):
    import botocore.credentials

    from lightspeed_agentic.config import BedrockCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_bedrock_credentials

    clock = [datetime.now(UTC)]
    monkeypatch.setattr(botocore.credentials, "_local_now", lambda: clock[0])
    for cycle in range(2):
        auth_wire.responses.append((200, _sts_reply(clock[0] + timedelta(hours=1 + cycle))))
    credential = create_bedrock_credentials(
        BedrockCredentialSource(
            "PRIVATE_SOURCE_ACCESS",
            "PRIVATE_SOURCE_SECRET",
            role_arn="arn:aws:iam::123456789012:role/PRIVATE",
        ),
        "us-east-1",
    )
    credential._time_fetcher = lambda: clock[0]
    assert auth_wire.requests == []
    assert credential.get_frozen_credentials().token == "PRIVATE_TOKEN"  # noqa: S105 - offline
    assert credential.get_frozen_credentials().token == "PRIVATE_TOKEN"  # noqa: S105 - offline
    assert len(auth_wire.requests) == 1
    clock[0] += timedelta(hours=1)
    assert credential.get_frozen_credentials().token == "PRIVATE_TOKEN"  # noqa: S105 - offline
    assert len(auth_wire.requests) == 2
    events = auth_wire.capture()
    assert [row["event"] for row in events] == [
        "outbound.request.started",
        "outbound.request.finished",
    ] * 2
    assert all(row["request.kind"] == "credential" for row in events)
    assert events[1]["outcome"] == "success"
    assert events[1]["http.request.method"] == "POST"
    assert "PRIVATE" not in json.dumps(events)


@pytest.mark.parametrize("status", [200, 403, 503])
def test_google_auth_retry_http_outcome_and_proxy(auth_wire, tmp_path, status):
    from google.auth import exceptions
    from google.auth.transport.requests import Request
    from requests.adapters import HTTPAdapter
    from urllib3.util import Retry

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    credentials = create_vertex_credentials(
        VertexCredentialSource(_google_adc(tmp_path, "authorized_user"))
    )
    request = Request()
    request.session.proxies = {"https": "http://user:PRIVATE@proxy.example:8080"}
    request.session.mount(
        "https://", HTTPAdapter(max_retries=Retry(total=1, status_forcelist=[503]))
    )
    auth_wire.responses.append((status, {"error": "invalid_grant", "error_description": "PRIVATE"}))
    if status == 503:
        auth_wire.responses.append((200, {"access_token": "PRIVATE_TOKEN", "expires_in": 3600}))
    try:
        if status == 503:
            credentials.refresh(request)
            assert credentials.token == "PRIVATE_TOKEN"  # noqa: S105 - offline
        else:
            with pytest.raises(exceptions.RefreshError):
                credentials.refresh(request)
    finally:
        request.session.close()
    events = auth_wire.capture()
    starts = events[::2]
    ends = events[1::2]
    assert len(starts) == (2 if status == 503 else 1)
    assert len({row["attempt_id"] for row in starts}) == len(starts)
    assert all(row["server.address"] == "oauth2.googleapis.com" for row in starts)
    assert all(row["server.port"] == 443 for row in starts)
    assert ends[0]["http.response.status_code"] == status
    assert ends[0]["outcome"] == ("success" if status == 200 else "http_error")
    assert "decision" not in ends[0]
    assert "PRIVATE" not in json.dumps(events)


@pytest.mark.parametrize("provider", ["google", "aws"])
def test_native_auth_environment_proxy_bypass_and_pool_selection(
    auth_wire, monkeypatch, tmp_path, provider
):
    from google.auth.transport.requests import Request

    from lightspeed_agentic.config import BedrockCredentialSource, VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import (
        create_bedrock_credentials,
        create_vertex_credentials,
    )

    host = "oauth2.googleapis.com" if provider == "google" else "sts.us-east-1.amazonaws.com"
    monkeypatch.setenv("HTTPS_PROXY", "http://user:PRIVATE@proxy.example:8080")
    monkeypatch.setenv("NO_PROXY", host)
    if provider == "google":
        credential = create_vertex_credentials(
            VertexCredentialSource(_google_adc(tmp_path, "authorized_user"))
        )
        auth_wire.responses.extend(
            [(200, {"access_token": "PRIVATE_TOKEN", "expires_in": 3600})] * 2
        )
        request = Request()
        try:
            credential.refresh(request)
            monkeypatch.delenv("NO_PROXY")
            credential.refresh(request)
        finally:
            request.session.close()
    else:
        import botocore.credentials

        clock = [datetime.now(UTC)]
        monkeypatch.setattr(botocore.credentials, "_local_now", lambda: clock[0])
        auth_wire.responses.extend(
            [(200, _sts_reply(clock[0] + timedelta(hours=i))) for i in (1, 3)]
        )
        credential = create_bedrock_credentials(
            BedrockCredentialSource(
                "PRIVATE_ACCESS",
                "PRIVATE_SECRET",
                role_arn="arn:aws:iam::123456789012:role/PRIVATE",
            ),
            "us-east-1",
        )
        credential._time_fetcher = lambda: clock[0]
        credential.get_frozen_credentials()
        monkeypatch.delenv("NO_PROXY")
        clock[0] += timedelta(hours=2)
        credential.get_frozen_credentials()
    events = auth_wire.capture()
    starts = events[::2]
    assert len(starts) == 2
    assert [row["server.address"] for row in starts] == [host, host]
    assert starts[1]["server.port"] == 443
    assert all(row["outcome"] == "success" for row in events[1::2])
    assert "PRIVATE" not in json.dumps(events)


def test_native_sts_rejection_is_http_failure_not_network_denial(auth_wire):
    from botocore.exceptions import ClientError

    from lightspeed_agentic.config import BedrockCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_bedrock_credentials

    auth_wire.responses.append(
        (
            403,
            b"<ErrorResponse><Error><Type>Sender</Type>"
            b"<Code>AccessDenied</Code><Message>PRIVATE</Message></Error>"
            b"<RequestId>PRIVATE</RequestId></ErrorResponse>",
        )
    )
    credential = create_bedrock_credentials(
        BedrockCredentialSource(
            "PRIVATE_ACCESS", "PRIVATE_SECRET", role_arn="arn:aws:iam::123456789012:role/PRIVATE"
        ),
        "us-east-1",
    )
    with pytest.raises(ClientError) as raised:
        credential.get_frozen_credentials()
    assert raised.value.response["Error"]["Code"] == "AccessDenied"
    start, end = auth_wire.capture()
    assert end["http.response.status_code"] == 403
    assert end["error.type"] == "http_error"
    assert end["outcome"] == "http_error"
    assert "decision" not in end
    assert "PRIVATE" not in json.dumps([start, end])


@pytest.mark.parametrize("provider", ["google", "aws"])
@pytest.mark.parametrize("failure", ["timeout", "connect", "dns", "tls", "context", "cancel"])
def test_native_auth_transport_failure_metadata(auth_wire, tmp_path, provider, failure):
    import asyncio
    import socket
    import ssl

    from google.auth import exceptions as google_errors
    from google.auth.transport.requests import Request
    from urllib3 import exceptions

    from lightspeed_agentic.config import BedrockCredentialSource, VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import (
        create_bedrock_credentials,
        create_vertex_credentials,
    )

    if failure == "dns":
        reason = socket.gaierror("PRIVATE")
        error = exceptions.NameResolutionError("PRIVATE", None, reason)
        error.__cause__ = reason  # Native HTTPConnection._new_conn raises explicitly from it.
    elif failure == "connect":
        error = exceptions.NewConnectionError(None, "PRIVATE")
    elif failure == "tls":
        error = exceptions.SSLError(ssl.SSLCertVerificationError("PRIVATE"))
    elif failure == "cancel":
        error = asyncio.CancelledError("PRIVATE")
        error.__cause__ = ssl.SSLCertVerificationError("PRIVATE previous failure")
    else:
        error = exceptions.ReadTimeoutError(None, "PRIVATE", "PRIVATE")
        if failure == "context":
            error.__context__ = ssl.SSLCertVerificationError("PRIVATE previous failure")
    auth_wire.responses.extend([error] * 10)
    if provider == "google":
        credential = create_vertex_credentials(
            VertexCredentialSource(_google_adc(tmp_path, "authorized_user"))
        )
        request = Request()
        try:
            with pytest.raises(
                asyncio.CancelledError if failure == "cancel" else google_errors.TransportError
            ):
                credential.refresh(request)
        finally:
            request.session.close()
    else:
        from botocore.exceptions import BotoCoreError

        credential = create_bedrock_credentials(
            BedrockCredentialSource(
                "PRIVATE_ACCESS",
                "PRIVATE_SECRET",
                role_arn="arn:aws:iam::123456789012:role/PRIVATE",
            ),
            "us-east-1",
        )
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else BotoCoreError):
            credential.get_frozen_credentials()
    events = auth_wire.capture()
    starts = [row for row in events if row["event"] == "outbound.request.started"]
    ends = [row for row in events if row["event"] == "outbound.request.finished"]
    assert len(starts) == len(ends) == len(auth_wire.requests)
    assert len(starts) > 0
    assert len({row["attempt_id"] for row in starts}) == len(starts)
    assert {row["error.type"] for row in ends} == {
        {
            "context": "timeout",
            "cancel": "cancelled",
            "dns": "dns_error",
            "connect": "connect_error",
            "tls": "tls_error",
        }.get(failure, failure)
    }
    assert all("decision" not in row for row in ends)
    assert all("request.body.size" not in row for row in events)
    assert "PRIVATE" not in json.dumps(events)


def test_google_auth_compressed_response_preserves_native_decoding(auth_wire, tmp_path):
    import gzip
    import io

    from google.auth.transport.requests import Request
    from urllib3.response import HTTPResponse

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    body = json.dumps({"access_token": "PRIVATE" * 10000, "expires_in": 3600}).encode()
    encoded = gzip.compress(body)
    auth_wire.responses.append(
        HTTPResponse(
            body=io.BytesIO(encoded),
            status=200,
            headers={"Content-Encoding": "gzip", "Content-Length": str(len(encoded))},
            preload_content=False,
        )
    )
    credential = create_vertex_credentials(
        VertexCredentialSource(_google_adc(tmp_path, "authorized_user"))
    )
    request = Request()
    try:
        credential.refresh(request)
        assert credential.token == "PRIVATE" * 10000
    finally:
        request.session.close()
    start, end = auth_wire.capture()
    assert end["outcome"] == "success"
    assert "response_body_bytes" not in end
    assert "PRIVATE" not in json.dumps([start, end])


def test_google_loading_cleanup_cannot_replace_acquisition_error(auth_wire, monkeypatch, tmp_path):
    import requests
    from google.auth.exceptions import OAuthError

    from lightspeed_agentic.config import VertexCredentialSource
    from lightspeed_agentic.providers.cloud_credentials import create_vertex_credentials

    original_close = requests.Session.close
    closed = []

    def close(session):
        if not closed:
            closed.append(session)
            raise OSError("PRIVATE cleanup error")
        original_close(session)

    monkeypatch.setattr(requests.Session, "close", close)
    auth_wire.responses.extend([(200, b"PRIVATE_SUBJECT"), (403, {"error": "invalid_grant"})])
    with pytest.raises(OAuthError):
        create_vertex_credentials(VertexCredentialSource(_google_adc(tmp_path, "external_account")))
    assert closed
    events = auth_wire.capture()
    assert events[-1]["http.response.status_code"] == 403
    assert "PRIVATE" not in json.dumps(events)


@pytest.mark.parametrize("mode", ["close", "exhaust", "chunked", "body-error", "proxy-error"])
def test_auth_native_lazy_body_and_proxy_connect(auth_wire, mode):
    import io
    import ssl
    from http.client import HTTPResponse as WireResponse

    from google.auth.transport.requests import Request
    from urllib3.response import HTTPResponse

    from lightspeed_agentic.connection_audit import auth_request
    from lightspeed_agentic.providers.audit_support import instrument_google_request

    request = Request()
    instrument_google_request(request)
    instrument_google_request(request)
    if mode == "proxy-error":
        request.session.proxies = {"https": "http://user:PRIVATE@proxy.example:8080"}
        auth_wire.proxy_error = ssl.SSLCertVerificationError("PRIVATE")
    elif mode == "chunked":

        class Socket:
            def makefile(self, *_args, **_kwargs):
                return io.BytesIO(
                    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                    b"7\r\nPRIVATE\r\n0\r\n\r\n"
                )

        wire_response = WireResponse(Socket(), method="GET")
        wire_response.begin()
        auth_wire.responses.append(
            HTTPResponse(
                body=wire_response,
                status=200,
                original_response=wire_response,
                headers={"Transfer-Encoding": "chunked"},
                preload_content=False,
            )
        )
    elif mode == "body-error":

        class Broken(io.BytesIO):
            def read(self, *_args, **_kwargs):
                raise TimeoutError("PRIVATE")

        auth_wire.responses.append(
            HTTPResponse(
                body=Broken(), status=200, headers={"Content-Length": "7"}, preload_content=False
            )
        )
    else:
        auth_wire.responses.append((200, b"PRIVATE"))
    try:
        with auth_request("google"):
            if mode == "proxy-error":
                from google.auth.exceptions import TransportError

                with pytest.raises(TransportError):
                    request("https://token.example/PRIVATE?secret=PRIVATE", stream=True)
            else:
                response = request("https://token.example/PRIVATE?secret=PRIVATE", stream=True)
                assert len(auth_wire.capture()) == 1  # Headers alone never finish the attempt.
                if mode == "close":
                    response._response.close()
                elif mode == "body-error":
                    from requests.exceptions import ConnectionError as RequestsConnectionError

                    with pytest.raises(RequestsConnectionError):
                        _ = response.data
                    response._response.close()
                else:
                    assert response.data == b"PRIVATE"
                    response._response.close()
    finally:
        request.session.close()
    start, end = auth_wire.capture()
    assert "request.body.size" not in end
    assert end["outcome"] == ("success" if mode in {"exhaust", "chunked"} else "transport_error")
    assert end.get("error.type") == {
        "close": "protocol_error",
        "body-error": "timeout",
        "proxy-error": "tls_error",
    }.get(mode)
    if mode == "proxy-error":
        assert start["server.address"] == "token.example"
        assert "network.peer.address" not in start
        assert "decision" not in end
        assert auth_wire.requests == []
    assert "PRIVATE" not in json.dumps([start, end])


@pytest.mark.asyncio
@pytest.mark.parametrize("route_name", ["vertex/google", "vertex/anthropic", "bedrock/anthropic"])
async def test_selected_native_model_auth_keeps_audit_domains_separate(
    auth_wire,
    monkeypatch,
    tmp_path,
    route_name,
):
    from dataclasses import replace

    import httpx

    from lightspeed_agentic import tls
    from lightspeed_agentic.config import BedrockCredentialSource, VertexCredentialSource
    from lightspeed_agentic.connection_audit import invocation_context
    from lightspeed_agentic.providers.deepagents import _close_model_clients
    from lightspeed_agentic.providers.model_router import resolve_model_route
    from tests.test_trace_sdk_routes import _anthropic_reply, _family

    family = _family(route_name)
    if route_name.startswith("vertex/"):
        family = replace(
            family, credentials=VertexCredentialSource(_google_adc(tmp_path, "authorized_user"))
        )
        auth_wire.responses.append((200, {"access_token": "PRIVATE_TOKEN", "expires_in": 3600}))
    else:
        family = replace(
            family,
            credentials=BedrockCredentialSource(
                "PRIVATE_ACCESS",
                "PRIVATE_SECRET",
                role_arn="arn:aws:iam::123456789012:role/PRIVATE",
            ),
        )
        auth_wire.responses.append((200, _sts_reply(datetime.now(UTC) + timedelta(hours=1))))
    calls = []

    def model_http(request):
        calls.append(request)
        if route_name.startswith("vertex/"):
            assert request.headers["authorization"] == "Bearer PRIVATE_TOKEN"
        else:
            assert request.headers["x-amz-security-token"] == "PRIVATE_TOKEN"
            assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256")
        if "anthropic" in route_name:
            return _anthropic_reply(request, json.loads(request.content), 1)[0]
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "answer"}]},
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(model_http)
    with httpx.Client(transport=transport) as sync_client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            monkeypatch.setattr(tls, "create_http_client", lambda **_kwargs: sync_client)
            monkeypatch.setattr(tls, "create_async_http_client", lambda **_kwargs: async_client)
            if route_name == "vertex/google":
                from google.genai import _api_client

                monkeypatch.setattr(_api_client, "AsyncHttpxClient", lambda **_kwargs: async_client)
            route = resolve_model_route(model_family=family)
            model = route.create_model(route.model_parameters)
            if route_name == "vertex/google":
                api = model.client._api_client
                api._http_options.httpx_async_client = api._async_httpx_client
            try:
                with invocation_context("11111111-1111-4111-8111-111111111111", "analysis"):
                    for _ in range(2):
                        await model.ainvoke("PRIVATE prompt")
            finally:
                await _close_model_clients(model)
    events = auth_wire.capture()
    auth = [row for row in events if row["request.kind"] == "credential"]
    llm = [row for row in events if row["request.kind"] == "llm"]
    assert len(auth) == 2  # Second model invocation uses the SDK's cached token.
    assert len(llm) == 4
    assert len(calls) == 2
    assert all(row["http.request.method"] == "POST" for row in events)
    assert all(row["agenticrun.uid"] == "11111111-1111-4111-8111-111111111111" for row in events)
    assert all("gen_ai.tool.name" not in row for row in auth)
    assert "PRIVATE" not in json.dumps(events)
