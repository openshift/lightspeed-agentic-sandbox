"""Unit tests for E2E credential preflight (tests/e2e/credentials.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.e2e import credentials as creds


def _vertex_sa_json(project_id: str = "my-gcp-project") -> str:
    return json.dumps(
        {
            "type": "service_account",
            "project_id": project_id,
            "private_key_id": "key",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIE\n-----END PRIVATE KEY-----\n",
            "client_email": "svc@my-gcp-project.iam.gserviceaccount.com",
            "client_id": "1",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    )


@pytest.fixture
def cred_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(creds, "KONFLUX_CREDENTIAL_DIR", str(tmp_path))
    monkeypatch.setattr(creds, "KONFLUX_CREDENTIAL_MOUNT", str(tmp_path / "token"))
    for key in (
        "GOOGLE_PROVIDER_CREDENTIALS_PATH",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AWS_REGION",
        "CLOUD_ML_REGION",
    ):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


class TestOpenAIAgents:
    def test_openai_provider_key_path_file(
        self, cred_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        key_file = cred_dir / "openai.key"
        key_file.write_text("sk-test-key\n", encoding="utf-8")
        monkeypatch.setenv(creds.OPENAI_PROVIDER_KEY_PATH_ENV, str(key_file))

        status = creds.detect_credentials(creds.PROVIDER_OPENAI_AGENTS)
        assert status.available
        assert "openai.key" in status.reason


class TestAnthropicVertexDeepagents:
    def test_vertex_credential_file_sets_vertex_env(
        self, cred_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
        token = cred_dir / "token"
        token.write_text(_vertex_sa_json(), encoding="utf-8")

        status = creds.detect_credentials(creds.PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS)
        assert status.available
        assert status.env_vars["CLAUDE_CODE_USE_VERTEX"] == "1"
        assert status.env_vars["ANTHROPIC_VERTEX_PROJECT_ID"] == "my-gcp-project"
        assert status.env_vars[creds.GOOGLE_PROVIDER_CREDENTIALS_PATH_ENV] == str(token)

    def test_adc_authorized_user_quota_project_id(
        self, cred_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
        adc = cred_dir / "adc.json"
        adc.write_text(
            json.dumps(
                {
                    "account": "user@example.com",
                    "client_id": "1",
                    "client_secret": "secret",
                    "quota_project_id": "itpc-ca-99c692de8f",
                    "refresh_token": "token",
                    "type": "authorized_user",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv(creds.GOOGLE_PROVIDER_CREDENTIALS_PATH_ENV, str(adc))

        status = creds.detect_credentials(creds.PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS)
        assert status.available
        assert status.env_vars["ANTHROPIC_VERTEX_PROJECT_ID"] == "itpc-ca-99c692de8f"


class TestAnthropicBedrockDeepagents:
    def test_aws_env_vars(self, cred_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        assert cred_dir.is_dir()
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-value")

        status = creds.detect_credentials(creds.PROVIDER_ANTHROPIC_BEDROCK_DEEPAGENTS)
        assert status.available
        assert status.env_vars == {
            "AWS_ACCESS_KEY_ID": "AKIATEST",
            "AWS_SECRET_ACCESS_KEY": "secret-value",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_REGION": "us-east-1",
        }

    def test_konflux_bedrock_apitoken_mount(
        self, cred_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        (cred_dir / "aws_access_key_id").write_text("AKIAMOUNT\n", encoding="utf-8")
        (cred_dir / "aws_secret_access_key").write_text("mount-secret\n", encoding="utf-8")
        mount_secret = (cred_dir / "aws_secret_access_key").read_text().strip()

        status = creds.detect_credentials(creds.PROVIDER_ANTHROPIC_BEDROCK_DEEPAGENTS)
        assert status.available
        assert status.env_vars["AWS_ACCESS_KEY_ID"] == "AKIAMOUNT"
        assert status.env_vars["AWS_SECRET_ACCESS_KEY"] == mount_secret
        assert status.env_vars["CLAUDE_CODE_USE_BEDROCK"] == "1"
        assert status.env_vars["AWS_REGION"] == "us-east-1"
        assert "bedrock-apitoken" in status.reason

    def test_missing_credentials(self, cred_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        assert not list(cred_dir.iterdir())
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)

        status = creds.detect_credentials(creds.PROVIDER_ANTHROPIC_BEDROCK_DEEPAGENTS)
        assert not status.available
        assert "bedrock-apitoken" in status.reason

    def test_require_credentials_exports_env(
        self, cred_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)
        (cred_dir / "aws_access_key_id").write_text("AKIAREQ", encoding="utf-8")
        (cred_dir / "aws_secret_access_key").write_text("req-secret", encoding="utf-8")

        creds.require_credentials(creds.PROVIDER_ANTHROPIC_BEDROCK_DEEPAGENTS)
        assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIAREQ"
        assert os.environ["CLAUDE_CODE_USE_BEDROCK"] == "1"


class TestProviderMatrix:
    def test_all_providers_registered(self) -> None:
        names = set(creds.PROVIDER_NAMES)
        assert names == {
            creds.PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS,
            creds.PROVIDER_ANTHROPIC_BEDROCK_DEEPAGENTS,
            creds.PROVIDER_GEMINI_VERTEX_ADK,
            creds.PROVIDER_OPENAI_AGENTS,
        }
