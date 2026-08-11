"""Tests for MCP server configuration parsing and provider adapters."""

from __future__ import annotations

import json
import os
import ssl
from pathlib import Path
from unittest.mock import patch

from lightspeed_agentic.mcp import (
    ResolvedMCPHeader,
    ResolvedMCPServer,
    mcp_http_client_factory,
    parse_mcp_servers,
    to_gemini_mcp_toolsets,
    to_openai_mcp_servers,
)

TEST_CA_CERT = """-----BEGIN CERTIFICATE-----
MIIEMDCCAxigAwIBAgIJANqb7HHzA7AZMA0GCSqGSIb3DQEBCwUAMIGkMQswCQYD
VQQGEwJQQTEPMA0GA1UECAwGUGFuYW1hMRQwEgYDVQQHDAtQYW5hbWEgQ2l0eTEk
MCIGA1UECgwbVHJ1c3RDb3IgU3lzdGVtcyBTLiBkZSBSLkwuMScwJQYDVQQLDB5U
cnVzdENvciBDZXJ0aWZpY2F0ZSBBdXRob3JpdHkxHzAdBgNVBAMMFlRydXN0Q29y
IFJvb3RDZXJ0IENBLTEwHhcNMTYwMjA0MTIzMjE2WhcNMjkxMjMxMTcyMzE2WjCB
pDELMAkGA1UEBhMCUEExDzANBgNVBAgMBlBhbmFtYTEUMBIGA1UEBwwLUGFuYW1h
IENpdHkxJDAiBgNVBAoMG1RydXN0Q29yIFN5c3RlbXMgUy4gZGUgUi5MLjEnMCUG
A1UECwweVHJ1c3RDb3IgQ2VydGlmaWNhdGUgQXV0aG9yaXR5MR8wHQYDVQQDDBZU
cnVzdENvciBSb290Q2VydCBDQS0xMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIB
CgKCAQEAv463leLCJhJrMxnHQFgKq1mqjQCj/IDHUHuO1CAmujIS2CNUSSUQIpid
RtLByZ5OGy4sDjjzGiVoHKZaBeYei0i/mJZ0PmnK6bV4pQa81QBeCQryJ3pS/C3V
seq0iWEk8xoT26nPUu0MJLq5nux+AHT6k61sKZKuUbS701e/s/OojZz0JEsq1pme
9J7+wH5COucLlVPat2gOkEz7cD+PSiyU8ybdY2mplNgQTsVHCJCZGxdNuWxu72CV
EY4hgLW9oHPY0LJ3xEXqWib7ZnZ2+AYfYW0PVcWDtxBWcgYHpfOxGgMFZA6dWorW
hnAbJN7+KIor0Gqw/Hqi3LJ5DotlDwIDAQABo2MwYTAdBgNVHQ4EFgQU7mtJPHo/
DeOxCbeKyKsZn3MzUOcwHwYDVR0jBBgwFoAU7mtJPHo/DeOxCbeKyKsZn3MzUOcw
DwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8BAf8EBAMCAYYwDQYJKoZIhvcNAQELBQAD
ggEBACUY1JGPE+6PHh0RU9otRCkZoB5rMZ5NDp6tPVxBb5UrJKF5mDo4Nvu7Zp5I
/5CQ7z3UuJu0h3U/IJvOcs+hVcFNZKIZBqEHMwwLKeXx6quj7LUKdJDHfXLy11yf
ke+Ri7fc7Waiz45mO7yfOgLgJ90WmMCV1Aqk5IGadZQ1nJBfiDcGrVmVCrDRZ9MZ
yonnMlo2HD6CqFqTvsbQZJG2z9m2GM/bftJlo6bEjhcxwft+dtvTheNYsnd6djts
L1Ac59v2Z3kf9YKVmgenFK+P3CghZwnS1k1aHBkcjndcw5QkPTJrS37UeJSDvjdN
zl/HHk484IkzlQsPpTLWPFp5LBk=
-----END CERTIFICATE-----
"""


class TestParseMCPServers:
    def test_empty_env_returns_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            assert parse_mcp_servers() == []

    def test_empty_string_returns_empty(self):
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": ""}):
            assert parse_mcp_servers() == []

    def test_whitespace_returns_empty(self):
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": "   "}):
            assert parse_mcp_servers() == []

    def test_invalid_json_returns_empty(self):
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": "not-json"}):
            assert parse_mcp_servers() == []

    def test_non_array_returns_empty(self):
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": '{"key": "val"}'}):
            assert parse_mcp_servers() == []

    def test_basic_server_no_headers(self):
        servers_json = json.dumps([{"name": "test", "url": "http://test:8080/mcp"}])
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}):
            result = parse_mcp_servers()
            assert len(result) == 1
            assert result[0] == ResolvedMCPServer(
                name="test", url="http://test:8080/mcp", timeout=60, headers=[]
            )

    def test_custom_timeout(self):
        servers_json = json.dumps([{"name": "test", "url": "http://test:8080/mcp", "timeout": 120}])
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}):
            result = parse_mcp_servers()
            assert result[0].timeout == 120

    def test_valid_ca_file(self, tmp_path: Path):
        ca_file = tmp_path / "service-ca.crt"
        ca_file.write_text(TEST_CA_CERT)
        servers_json = json.dumps(
            [{"name": "test", "url": "https://test:8443/mcp", "caFile": str(ca_file)}]
        )
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}):
            result = parse_mcp_servers()
        assert result[0].ca_file == str(ca_file.resolve())

    def test_missing_ca_file_skips_server(self, tmp_path: Path):
        servers_json = json.dumps(
            [
                {
                    "name": "test",
                    "url": "https://test:8443/mcp",
                    "caFile": str(tmp_path / "missing.crt"),
                }
            ]
        )
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}):
            assert parse_mcp_servers() == []

    def test_invalid_ca_file_skips_server(self, tmp_path: Path):
        ca_file = tmp_path / "invalid.crt"
        ca_file.write_text("not a certificate")
        servers_json = json.dumps(
            [{"name": "test", "url": "https://test:8443/mcp", "caFile": str(ca_file)}]
        )
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}):
            assert parse_mcp_servers() == []

    def test_relative_ca_file_skips_server(self):
        servers_json = json.dumps(
            [{"name": "test", "url": "https://test:8443/mcp", "caFile": "ca.crt"}]
        )
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}):
            assert parse_mcp_servers() == []

    def test_service_account_token_header(self, tmp_path: Path):
        token_file = tmp_path / "token"
        token_file.write_text("my-sa-token")

        servers_json = json.dumps(
            [
                {
                    "name": "ocp",
                    "url": "http://mcp:8080/mcp",
                    "headers": [{"name": "Authorization", "source": "ServiceAccountToken"}],
                }
            ]
        )
        with (
            patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}),
            patch("lightspeed_agentic.mcp.SA_TOKEN_PATH", str(token_file)),
        ):
            result = parse_mcp_servers()
            assert len(result) == 1
            assert result[0].headers == [
                ResolvedMCPHeader(name="Authorization", value="Bearer my-sa-token")
            ]

    def test_service_account_token_missing(self):
        servers_json = json.dumps(
            [
                {
                    "name": "ocp",
                    "url": "http://mcp:8080/mcp",
                    "headers": [{"name": "Authorization", "source": "ServiceAccountToken"}],
                }
            ]
        )
        with (
            patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}),
            patch("lightspeed_agentic.mcp.SA_TOKEN_PATH", "/nonexistent/path"),
        ):
            result = parse_mcp_servers()
            assert len(result) == 1
            assert result[0].headers == []

    def test_secret_header(self, tmp_path: Path):
        secret_dir = tmp_path / "my-secret"
        secret_dir.mkdir()
        (secret_dir / "header").write_text("secret-value-123")

        servers_json = json.dumps(
            [
                {
                    "name": "ext",
                    "url": "http://ext:9090/mcp",
                    "headers": [
                        {"name": "X-Api-Key", "source": "Secret", "secretName": "my-secret"}
                    ],
                }
            ]
        )
        with (
            patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}),
            patch("lightspeed_agentic.mcp.MCP_SECRET_MOUNT_ROOT", str(tmp_path)),
        ):
            result = parse_mcp_servers()
            assert result[0].headers == [
                ResolvedMCPHeader(name="X-Api-Key", value="secret-value-123")
            ]

    def test_secret_dir_missing(self):
        servers_json = json.dumps(
            [
                {
                    "name": "ext",
                    "url": "http://ext:9090/mcp",
                    "headers": [
                        {"name": "X-Api-Key", "source": "Secret", "secretName": "no-such-secret"}
                    ],
                }
            ]
        )
        with (
            patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}),
            patch("lightspeed_agentic.mcp.MCP_SECRET_MOUNT_ROOT", "/nonexistent"),
        ):
            result = parse_mcp_servers()
            assert result[0].headers == []

    def test_client_source_skipped(self):
        servers_json = json.dumps(
            [
                {
                    "name": "ext",
                    "url": "http://ext:9090/mcp",
                    "headers": [{"name": "Authorization", "source": "Client"}],
                }
            ]
        )
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}):
            result = parse_mcp_servers()
            assert result[0].headers == []

    def test_multiple_servers(self, tmp_path: Path):
        token_file = tmp_path / "token"
        token_file.write_text("tok")

        servers_json = json.dumps(
            [
                {"name": "a", "url": "http://a:8080/mcp"},
                {"name": "b", "url": "http://b:8080/mcp", "timeout": 30},
            ]
        )
        with (
            patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}),
            patch("lightspeed_agentic.mcp.SA_TOKEN_PATH", str(token_file)),
        ):
            result = parse_mcp_servers()
            assert len(result) == 2
            assert result[0].name == "a"
            assert result[1].name == "b"
            assert result[1].timeout == 30

    def test_invalid_entry_skipped(self):
        servers_json = json.dumps([42, {"name": "ok", "url": "http://ok:8080/mcp"}])
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}):
            result = parse_mcp_servers()
            assert len(result) == 1
            assert result[0].name == "ok"

    def test_invalid_header_skipped(self):
        servers_json = json.dumps(
            [
                {
                    "name": "s",
                    "url": "http://s:8080/mcp",
                    "headers": [
                        "bad",
                        {"name": "X", "source": "Client"},
                    ],
                },
            ]
        )
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}):
            result = parse_mcp_servers()
            assert result[0].headers == []

    def test_path_traversal_rejected(self, tmp_path: Path):
        mount_root = tmp_path / "secrets"
        mount_root.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        (outside_dir / "leaked").write_text("should-not-be-read")

        servers_json = json.dumps(
            [
                {
                    "name": "evil",
                    "url": "http://x:8080/mcp",
                    "headers": [
                        {"name": "X", "source": "Secret", "secretName": "../outside"},
                    ],
                },
            ]
        )
        with (
            patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}),
            patch("lightspeed_agentic.mcp.MCP_SECRET_MOUNT_ROOT", str(mount_root)),
        ):
            result = parse_mcp_servers()
            assert result[0].headers == []

    def test_headers_null_treated_as_empty(self):
        servers_json = json.dumps(
            [
                {"name": "s", "url": "http://s:8080/mcp", "headers": None},
            ]
        )
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}):
            result = parse_mcp_servers()
            assert result[0].headers == []

    def test_headers_non_list_treated_as_empty(self):
        servers_json = json.dumps(
            [
                {"name": "s", "url": "http://s:8080/mcp", "headers": "bad"},
            ]
        )
        with patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}):
            result = parse_mcp_servers()
            assert result[0].headers == []


class TestMCPHTTPClientFactory:
    def test_adds_ca_without_disabling_verification(self, tmp_path: Path):
        ca_file = tmp_path / "service-ca.crt"
        ca_file.write_text(TEST_CA_CERT)
        factory = mcp_http_client_factory(str(ca_file))
        assert factory is not None

        with patch("lightspeed_agentic.mcp.httpx.AsyncClient") as client:
            factory(headers={"X-Test": "value"}, timeout=None, auth=None)

        verify = client.call_args.kwargs["verify"]
        assert isinstance(verify, ssl.SSLContext)
        assert verify.verify_mode == ssl.CERT_REQUIRED
        assert verify.check_hostname is True
        assert client.call_args.kwargs["follow_redirects"] is True


class TestGeminiAdapter:
    def test_creates_toolsets(self):
        servers = [ResolvedMCPServer(name="ocp-mcp", url="https://ocp:8443/mcp", timeout=90)]
        toolsets = to_gemini_mcp_toolsets(servers)
        assert len(toolsets) == 1
        from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

        assert isinstance(toolsets[0], McpToolset)

    def test_passes_connection_params(self):
        servers = [
            ResolvedMCPServer(
                name="s",
                url="http://test:8080/mcp",
                timeout=45,
                headers=[ResolvedMCPHeader(name="X-Key", value="val")],
            )
        ]
        toolsets = to_gemini_mcp_toolsets(servers)
        params = toolsets[0]._connection_params
        assert params.url == "http://test:8080/mcp"
        assert params.headers == {"X-Key": "val"}
        assert params.timeout == 45.0

    def test_passes_ca_client_factory(self, tmp_path: Path):
        ca_file = tmp_path / "service-ca.crt"
        ca_file.write_text(TEST_CA_CERT)
        servers = [
            ResolvedMCPServer(
                name="s",
                url="https://test:8443/mcp",
                ca_file=str(ca_file),
            )
        ]
        params = to_gemini_mcp_toolsets(servers)[0]._connection_params
        assert callable(params.httpx_client_factory)


class TestOpenAIAdapter:
    def test_creates_servers(self):
        servers = [ResolvedMCPServer(name="ocp-mcp", url="https://ocp:8443/mcp")]
        result = to_openai_mcp_servers(servers)
        assert len(result) == 1
        from agents.mcp import MCPServerStreamableHttp

        assert isinstance(result[0], MCPServerStreamableHttp)
        assert result[0].name == "ocp-mcp"

    def test_passes_headers(self):
        servers = [
            ResolvedMCPServer(
                name="ext",
                url="http://ext/mcp",
                headers=[ResolvedMCPHeader(name="Auth", value="Bearer x")],
            )
        ]
        result = to_openai_mcp_servers(servers)
        assert result[0].params["headers"] == {"Auth": "Bearer x"}

    def test_passes_ca_client_factory(self, tmp_path: Path):
        ca_file = tmp_path / "service-ca.crt"
        ca_file.write_text(TEST_CA_CERT)
        servers = [
            ResolvedMCPServer(
                name="s",
                url="https://test:8443/mcp",
                ca_file=str(ca_file),
            )
        ]
        result = to_openai_mcp_servers(servers)
        assert callable(result[0].params["httpx_client_factory"])
