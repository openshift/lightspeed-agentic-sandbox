"""E2E credential validation for live cluster BDD.

Resolution order per provider mirrors the deploy scripts in
lightspeed-operator/hack/ — env vars first, CLI tools as fallback.

Unlike evals credential checks, missing credentials raise instead of soft-skipping.

E2E matrix ids use {model-or-vendor}-{transport?}-{runtime} (not AgentProvider.name /
CR LIGHTSPEED_PROVIDER).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS = "anthropic-vertex-deepagents"
PROVIDER_ANTHROPIC_BEDROCK_DEEPAGENTS = "anthropic-bedrock-deepagents"
PROVIDER_GEMINI_VERTEX_ADK = "gemini-vertex-adk"
PROVIDER_OPENAI_AGENTS = "openai-agents"

# Konflux workspace secrets mount at /var/run/credentials/token (data key: token).
KONFLUX_CREDENTIAL_MOUNT = "/var/run/credentials/token"
KONFLUX_CREDENTIAL_DIR = "/var/run/credentials"
GOOGLE_PROVIDER_CREDENTIALS_PATH_ENV = "GOOGLE_PROVIDER_CREDENTIALS_PATH"
OPENAI_PROVIDER_KEY_PATH_ENV = "OPENAI_PROVIDER_KEY_PATH"


@dataclass(frozen=True)
class ProviderCredentialStatus:
    provider: str
    available: bool
    source: str
    reason: str
    env_vars: dict[str, str] = field(default_factory=dict)


def _run_quiet(cmd: list[str], timeout: int = 10) -> tuple[bool, str]:
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, ""


def _credential_file_path(*env_keys: str) -> str | None:
    for key in env_keys:
        path = os.environ.get(key, "").strip()
        if path:
            return path
    if os.path.isfile(KONFLUX_CREDENTIAL_MOUNT):
        return KONFLUX_CREDENTIAL_MOUNT
    return None


def _validate_plaintext_credential_file(path: str, label: str) -> tuple[bool, str]:
    if not os.path.isfile(path):
        return False, f"{label} file not found: {path}"
    try:
        with open(path, encoding="utf-8") as handle:
            content = handle.read().strip()
    except OSError as exc:
        return False, f"{label} unreadable ({path}): {exc}"
    if not content:
        return False, f"{label} file is empty: {path}"
    return True, path


def _validate_vertex_credential_file(path: str, label: str) -> tuple[bool, str, str]:
    ok, detail = _validate_plaintext_credential_file(path, label)
    if not ok:
        return False, detail, ""
    try:
        import json

        with open(path, encoding="utf-8") as handle:
            data = json.loads(handle.read())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"{label} is not valid Vertex JSON ({path}): {exc}", ""
    project_id = data.get("project_id") or data.get("quota_project_id") or ""
    if not project_id:
        return False, f"{label} missing project_id or quota_project_id ({path})", ""
    return True, path, str(project_id)


def _vertex_credential_status(
    name: str,
    *,
    source_label: str,
    path: str,
    project_id: str,
    anthropic: bool,
) -> ProviderCredentialStatus:
    env_vars: dict[str, str] = {
        "GOOGLE_APPLICATION_CREDENTIALS": path,
        GOOGLE_PROVIDER_CREDENTIALS_PATH_ENV: path,
    }
    env_vars.update(_vertex_env_vars())
    if anthropic:
        env_vars["CLAUDE_CODE_USE_VERTEX"] = "1"
        if not os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"):
            env_vars["ANTHROPIC_VERTEX_PROJECT_ID"] = project_id
    else:
        env_vars["GOOGLE_CLOUD_PROJECT"] = project_id
    return ProviderCredentialStatus(
        name,
        True,
        "env",
        f"{source_label} ({path})",
        env_vars=env_vars,
    )


def _check_vertex_credential_file(
    name: str,
    *,
    anthropic: bool,
) -> ProviderCredentialStatus | None:
    path = _credential_file_path(
        GOOGLE_PROVIDER_CREDENTIALS_PATH_ENV,
        "GOOGLE_APPLICATION_CREDENTIALS",
    )
    if not path:
        return None
    label = GOOGLE_PROVIDER_CREDENTIALS_PATH_ENV
    ok, detail, project_id = _validate_vertex_credential_file(path, label)
    if not ok:
        return ProviderCredentialStatus(name, False, "none", detail)
    return _vertex_credential_status(
        name,
        source_label=label,
        path=path,
        project_id=project_id,
        anthropic=anthropic,
    )


def _check_anthropic_vertex_deepagents() -> ProviderCredentialStatus:
    name = PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ProviderCredentialStatus(
            name,
            True,
            "env",
            "ANTHROPIC_API_KEY set",
        )

    vertex_file = _check_vertex_credential_file(name, anthropic=True)
    if vertex_file is not None:
        return vertex_file

    if os.environ.get("CLAUDE_CODE_USE_VERTEX") == "1":
        gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if gac and os.path.isfile(gac):
            return ProviderCredentialStatus(
                name,
                True,
                "env",
                "Vertex AI credentials file",
            )
        ok, _ = _run_quiet(["gcloud", "auth", "application-default", "print-access-token"])
        if ok:
            return ProviderCredentialStatus(
                name,
                True,
                "gcloud",
                "gcloud application-default credentials",
            )
        return ProviderCredentialStatus(
            name,
            False,
            "none",
            "CLAUDE_CODE_USE_VERTEX=1 but no credentials file or gcloud ADC",
        )

    if os.environ.get("CLAUDE_CODE_USE_BEDROCK") == "1":
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            return ProviderCredentialStatus(
                name,
                True,
                "env",
                "AWS Bedrock credentials via env vars",
            )
        ok, _ = _run_quiet(["aws", "configure", "get", "aws_access_key_id"])
        if ok:
            return ProviderCredentialStatus(
                name,
                True,
                "aws_cli",
                "AWS credentials via aws configure",
            )
        return ProviderCredentialStatus(
            name,
            False,
            "none",
            "CLAUDE_CODE_USE_BEDROCK=1 but no AWS credentials found",
        )

    return ProviderCredentialStatus(
        name,
        False,
        "none",
        "ANTHROPIC_API_KEY not set (or set CLAUDE_CODE_USE_VERTEX=1 / "
        f"CLAUDE_CODE_USE_BEDROCK=1 / {GOOGLE_PROVIDER_CREDENTIALS_PATH_ENV})",
    )


def _read_bedrock_aws_env() -> dict[str, str] | None:
    access = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
    if access and secret:
        return {"AWS_ACCESS_KEY_ID": access, "AWS_SECRET_ACCESS_KEY": secret}

    access_path = os.path.join(KONFLUX_CREDENTIAL_DIR, "aws_access_key_id")
    secret_path = os.path.join(KONFLUX_CREDENTIAL_DIR, "aws_secret_access_key")
    if os.path.isfile(access_path) and os.path.isfile(secret_path):
        try:
            with open(access_path, encoding="utf-8") as handle:
                access = handle.read().strip()
            with open(secret_path, encoding="utf-8") as handle:
                secret = handle.read().strip()
        except OSError:
            return None
        if access and secret:
            return {"AWS_ACCESS_KEY_ID": access, "AWS_SECRET_ACCESS_KEY": secret}
    return None


def _check_anthropic_bedrock_deepagents() -> ProviderCredentialStatus:
    name = PROVIDER_ANTHROPIC_BEDROCK_DEEPAGENTS
    aws_env = _read_bedrock_aws_env()
    if aws_env is None:
        if os.environ.get("CLAUDE_CODE_USE_BEDROCK") == "1":
            ok, _ = _run_quiet(["aws", "configure", "get", "aws_access_key_id"])
            if ok:
                return ProviderCredentialStatus(
                    name,
                    True,
                    "aws_cli",
                    "AWS credentials via aws configure",
                    env_vars={"CLAUDE_CODE_USE_BEDROCK": "1"},
                )
        return ProviderCredentialStatus(
            name,
            False,
            "none",
            "AWS Bedrock credentials not found (set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY, or mount Konflux bedrock-apitoken)",
        )

    env_vars = {**aws_env, "CLAUDE_CODE_USE_BEDROCK": "1"}
    if not os.environ.get("AWS_REGION"):
        env_vars["AWS_REGION"] = os.environ.get("CLOUD_ML_REGION", "us-east-1")
    return ProviderCredentialStatus(
        name,
        True,
        "env",
        "AWS Bedrock credentials (bedrock-apitoken or env)",
        env_vars=env_vars,
    )


def _vertex_env_vars() -> dict[str, str]:
    """Build env vars needed to switch google-genai / ADK to Vertex AI."""
    env: dict[str, str] = {"GOOGLE_GENAI_USE_VERTEXAI": "TRUE"}
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "")
        if not project:
            ok, project = _run_quiet(["gcloud", "config", "get-value", "project"])
            if not ok:
                project = ""
        if project:
            env["GOOGLE_CLOUD_PROJECT"] = project
    if not os.environ.get("GOOGLE_CLOUD_LOCATION"):
        env["GOOGLE_CLOUD_LOCATION"] = os.environ.get("CLOUD_ML_REGION", "global")
    return env


def _check_gemini_vertex_adk() -> ProviderCredentialStatus:
    name = PROVIDER_GEMINI_VERTEX_ADK
    if os.environ.get("GOOGLE_API_KEY"):
        return ProviderCredentialStatus(
            name,
            True,
            "env",
            "GOOGLE_API_KEY set",
        )

    # Bridge GEMINI_API_KEY → GOOGLE_API_KEY (ADK reads GOOGLE_API_KEY)
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        return ProviderCredentialStatus(
            name,
            True,
            "env",
            "GEMINI_API_KEY set",
            env_vars={"GOOGLE_API_KEY": gemini_key},
        )

    vertex_file = _check_vertex_credential_file(name, anthropic=False)
    if vertex_file is not None:
        return vertex_file

    # ADC file → Vertex AI mode (ADC is for GCP APIs, not the Gemini Developer API)
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if gac and os.path.isfile(gac):
        return ProviderCredentialStatus(
            name,
            True,
            "env",
            "GOOGLE_APPLICATION_CREDENTIALS file (Vertex AI)",
            env_vars=_vertex_env_vars(),
        )

    ok, _ = _run_quiet(["gcloud", "auth", "application-default", "print-access-token"])
    if ok:
        return ProviderCredentialStatus(
            name,
            True,
            "gcloud",
            "gcloud ADC (Vertex AI)",
            env_vars=_vertex_env_vars(),
        )

    return ProviderCredentialStatus(
        name,
        False,
        "none",
        "No Gemini credentials: set GOOGLE_API_KEY, GEMINI_API_KEY, "
        f"{GOOGLE_PROVIDER_CREDENTIALS_PATH_ENV}, GOOGLE_APPLICATION_CREDENTIALS, "
        "or configure gcloud ADC",
    )


def _check_openai_agents() -> ProviderCredentialStatus:
    name = PROVIDER_OPENAI_AGENTS
    if os.environ.get("OPENAI_API_KEY"):
        return ProviderCredentialStatus(
            name,
            True,
            "env",
            "OPENAI_API_KEY set",
        )

    key_path = os.environ.get(OPENAI_PROVIDER_KEY_PATH_ENV, "")
    if not key_path and os.path.isfile(KONFLUX_CREDENTIAL_MOUNT):
        key_path = KONFLUX_CREDENTIAL_MOUNT
    if key_path:
        ok, detail = _validate_plaintext_credential_file(
            key_path,
            OPENAI_PROVIDER_KEY_PATH_ENV,
        )
        if not ok:
            return ProviderCredentialStatus(name, False, "none", detail)
        return ProviderCredentialStatus(
            name,
            True,
            "env",
            f"{OPENAI_PROVIDER_KEY_PATH_ENV} file ({key_path})",
        )

    if os.environ.get("OPENAI_BASE_URL"):
        return ProviderCredentialStatus(
            name,
            True,
            "env",
            "OPENAI_BASE_URL set (custom endpoint, no key required)",
        )

    return ProviderCredentialStatus(
        name,
        False,
        "none",
        "OPENAI_API_KEY not set (or set OPENAI_BASE_URL for keyless endpoints, "
        f"or {OPENAI_PROVIDER_KEY_PATH_ENV})",
    )


_CHECKERS = {
    PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS: _check_anthropic_vertex_deepagents,
    PROVIDER_ANTHROPIC_BEDROCK_DEEPAGENTS: _check_anthropic_bedrock_deepagents,
    PROVIDER_GEMINI_VERTEX_ADK: _check_gemini_vertex_adk,
    PROVIDER_OPENAI_AGENTS: _check_openai_agents,
}

PROVIDER_NAMES = list(_CHECKERS.keys())


def detect_credentials(provider: str) -> ProviderCredentialStatus:
    checker = _CHECKERS.get(provider)
    if checker is None:
        return ProviderCredentialStatus(provider, False, "none", f"Unknown provider: {provider}")
    return checker()


def detect_all() -> dict[str, ProviderCredentialStatus]:
    return {name: detect_credentials(name) for name in PROVIDER_NAMES}


def require_credentials(provider: str) -> None:
    """Fail fast with a clear message when the host cannot run E2E for this provider."""
    status = detect_credentials(provider)
    if not status.available:
        msg = f"E2E credentials missing for provider {provider!r}: {status.reason}"
        raise RuntimeError(msg)
    for key, value in status.env_vars.items():
        if value and not os.environ.get(key):
            os.environ[key] = value


def main() -> None:
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "check":
        require_credentials(sys.argv[2])
        return
    raise SystemExit("usage: python credentials.py check <provider>")


if __name__ == "__main__":
    main()
