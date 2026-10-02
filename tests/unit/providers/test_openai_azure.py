"""Tests for Azure OpenAI adapter construction in the OpenAI provider.

Verifies that the adapter constructs AsyncAzureOpenAI + OpenAIResponsesModel
for Azure (Responses API supported since api-version 2025-03-01-preview),
and correctly wires Entra ID token provider vs API key.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lightspeed_agentic.providers.openai import OpenAIProvider


@pytest.fixture(autouse=True)
def _clean_azure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove Azure-specific env vars to isolate tests."""
    for var in [
        "LIGHTSPEED_PROVIDER",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
    ]:
        monkeypatch.delenv(var, raising=False)


def _setup_azure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myresource.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")


class TestAzureClientConstruction:
    """Verify _build_azure_client returns AsyncAzureOpenAI with correct params."""

    def test_api_key_mode_builds_azure_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """API-key mode: AsyncAzureOpenAI with api_key, wrapped in OpenAIResponsesModel."""
        _setup_azure_env(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

        provider = OpenAIProvider()
        client, model_wrapper = provider._build_azure_client("gpt-4.1")

        # Verify the client type
        from openai import AsyncAzureOpenAI

        assert isinstance(client, AsyncAzureOpenAI)

        # Verify model wrapper type — Azure uses Responses API
        from agents.models.openai_responses import OpenAIResponsesModel

        assert isinstance(model_wrapper, OpenAIResponsesModel)

    def test_azure_client_preserves_tls_context_and_disables_redirects(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Azure HTTP client must preserve TLS config and avoid auth header redirects."""
        _setup_azure_env(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        shared_context = object()

        with (
            patch("lightspeed_agentic.tls.get_ssl_context", return_value=shared_context),
            patch("openai.DefaultAsyncHttpxClient") as http_client,
            patch("openai.AsyncAzureOpenAI.__init__", return_value=None),
        ):
            http_client.return_value = MagicMock()
            OpenAIProvider()._build_azure_client("gpt-4.1")

        http_client.assert_called_once_with(
            verify=shared_context,
            follow_redirects=False,
        )

    def test_rejects_non_https_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configured Azure endpoints must use HTTPS."""
        _setup_azure_env(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "http://myresource.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

        with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT must use https"):
            OpenAIProvider()._build_azure_client("gpt-4.1")

    def test_empty_endpoint_is_preserved_as_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty endpoint keeps existing None handling for SDK/env fallback."""
        _setup_azure_env(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

        with patch("openai.AsyncAzureOpenAI.__init__", return_value=None) as azure_init:
            OpenAIProvider()._build_azure_client("gpt-4.1")

        assert azure_init.call_args.kwargs["azure_endpoint"] is None

    def test_legacy_api_version_uses_chat_completions_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Azure API versions before 2025-03-01-preview use Chat Completions."""
        _setup_azure_env(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

        provider = OpenAIProvider()
        client, model_wrapper = provider._build_azure_client("gpt-4.1")

        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
        from openai import AsyncAzureOpenAI

        assert isinstance(client, AsyncAzureOpenAI)
        assert isinstance(model_wrapper, OpenAIChatCompletionsModel)

    def test_missing_api_version_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Azure client construction requires an explicit API version."""
        _setup_azure_env(monkeypatch)
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

        with pytest.raises(ValueError, match="AZURE_OPENAI_API_VERSION is required"):
            OpenAIProvider()._build_azure_client("gpt-4.1")

    def test_entra_id_mode_builds_azure_client_with_token_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Entra ID mode: AsyncAzureOpenAI with azure_ad_token_provider."""
        _setup_azure_env(monkeypatch)

        mock_credential = MagicMock()
        mock_token_provider = MagicMock(return_value="fake-token")

        with (
            patch(
                "azure.identity.ClientSecretCredential",
                return_value=mock_credential,
            ) as csc_cls,
            patch(
                "azure.identity.get_bearer_token_provider",
                return_value=mock_token_provider,
            ) as gbtp,
        ):
            provider = OpenAIProvider()
            provider._azure_credentials = {
                "client_id": "cid",
                "tenant_id": "tid",
                "client_secret": "csec",
            }
            client, model_wrapper = provider._build_azure_client("gpt-4.1")

        # Verify ClientSecretCredential was constructed
        csc_cls.assert_called_once_with("tid", "cid", "csec")

        # Verify get_bearer_token_provider was called with the credential
        gbtp.assert_called_once_with(
            mock_credential,
            "https://cognitiveservices.azure.com/.default",
        )

        from openai import AsyncAzureOpenAI

        assert isinstance(client, AsyncAzureOpenAI)

        from agents.models.openai_responses import OpenAIResponsesModel

        assert isinstance(model_wrapper, OpenAIResponsesModel)

    def test_entra_id_does_not_set_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Entra ID mode must NOT pass api_key to AsyncAzureOpenAI."""
        _setup_azure_env(monkeypatch)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

        mock_credential = MagicMock()
        mock_token_provider = MagicMock(return_value="fake-token")

        with (
            patch(
                "azure.identity.ClientSecretCredential",
                return_value=mock_credential,
            ),
            patch(
                "azure.identity.get_bearer_token_provider",
                return_value=mock_token_provider,
            ),
            patch(
                "openai.AsyncAzureOpenAI.__init__",
                return_value=None,
            ) as azure_init,
        ):
            provider = OpenAIProvider()
            provider._azure_credentials = {
                "client_id": "cid",
                "tenant_id": "tid",
                "client_secret": "csec",
            }
            provider._build_azure_client("gpt-4.1")

        # Verify api_key was not passed
        call_kwargs = azure_init.call_args
        assert "api_key" not in (call_kwargs.kwargs if call_kwargs.kwargs else {})
