"""Batch E2E suite setup — cluster preflight and session lifecycle (OLS-3926).

Layer 2 between shell fixture install (``scripts/e2e-install-fixtures.sh``) and
per-scenario batch Jobs. Mirrors the operator ``TestMain`` pattern in pytest.
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from kubernetes.client import (  # type: ignore[import-untyped]
    ApiException,
    AppsV1Api,
    CoreV1Api,
)

from tests.e2e.credentials import (
    PROVIDER_ANTHROPIC_BEDROCK_DEEPAGENTS,
    PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS,
    PROVIDER_GEMINI_VERTEX_ADK,
    PROVIDER_OPENAI_AGENTS,
    require_credentials,
)

logger = logging.getLogger(__name__)

DEFAULT_NAMESPACE = "openshift-lightspeed"
DEFAULT_SANDBOX_IMAGE = (
    "quay.io/redhat-user-workloads/crt-nshift-lightspeed-tenant/lightspeed-agentic-sandbox:main"
)
DEFAULT_SANDBOX_SA = "lightspeed-sandbox-e2e"
DEFAULT_OTEL_CA_SECRET = "lightspeed-otel-ca"  # noqa: S105 — K8s Secret name, not a credential
DEFAULT_OTEL_DEPLOYMENT = "lightspeed-otel-collector"
DEFAULT_MOCK_MCP_DEPLOYMENT = "lightspeed-mock-mcp"
DEFAULT_MOCK_MCP_SERVICE = "lightspeed-mock-mcp"
DEFAULT_MOCK_MCP_PORT = 19090
E2E_RUN_LABEL = "e2e.openshift.io/session"
E2E_COMPONENT_LABEL = "agentic.openshift.io/component"
E2E_COMPONENT_VALUE = "sandbox-e2e"

_E2E_PROVIDER_TO_LIGHTSPEED: dict[str, str] = {
    PROVIDER_OPENAI_AGENTS: "openai",
    PROVIDER_GEMINI_VERTEX_ADK: "vertex",
    PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS: "vertex",
    PROVIDER_ANTHROPIC_BEDROCK_DEEPAGENTS: "bedrock",
}

_DEFAULT_LLM_SECRETS: dict[str, str] = {
    PROVIDER_OPENAI_AGENTS: "llm-creds-openai",
    PROVIDER_GEMINI_VERTEX_ADK: "llm-creds-vertex",
    PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS: "llm-creds-anthropic",
    PROVIDER_ANTHROPIC_BEDROCK_DEEPAGENTS: "llm-creds-bedrock",
}

_MODEL_ENV_BY_PROVIDER: dict[str, str] = {
    PROVIDER_OPENAI_AGENTS: "OPENAI_MODEL",
    PROVIDER_GEMINI_VERTEX_ADK: "GEMINI_MODEL",
    PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS: "ANTHROPIC_MODEL",
    PROVIDER_ANTHROPIC_BEDROCK_DEEPAGENTS: "ANTHROPIC_BEDROCK_MODEL",
}


@dataclass(frozen=True)
class BatchE2EConfig:
    """Resolved batch E2E configuration for one pytest session."""

    namespace: str
    sandbox_image: str
    service_account: str
    llm_secret: str
    lightspeed_provider: str
    model: str
    provider_name: str
    session_id: str
    otel_endpoint: str
    otel_ca_secret: str
    verify_full_fixtures: bool
    extra_env: dict[str, str] = field(default_factory=dict)
    job_env: dict[str, str] = field(default_factory=dict)


def _truthy(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_config_env_defaults() -> dict[str, str]:
    path = Path(__file__).parent / "config.env"
    defaults: dict[str, str] = {}
    if not path.is_file():
        return defaults
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        if value.startswith("${") and ":-" in value:
            value = value.split(":-", 1)[1].rstrip("}")
        defaults[key] = value
    return defaults


def resolve_model(provider_name: str) -> str:
    """Resolve LIGHTSPEED_MODEL for an E2E provider matrix id."""
    env_key = _MODEL_ENV_BY_PROVIDER.get(provider_name)
    if env_key is None:
        msg = f"unknown E2E provider for model resolution: {provider_name!r}"
        raise ValueError(msg)
    if os.environ.get(env_key, "").strip():
        return os.environ[env_key].strip()
    defaults = _load_config_env_defaults()
    return defaults.get(env_key, "").strip()


def resolve_llm_secret(provider_name: str) -> str:
    """Resolve the credentials Secret name for batch Jobs."""
    override = os.environ.get("LLM_SECRET", "").strip()
    if override:
        return override
    secret = _DEFAULT_LLM_SECRETS.get(provider_name)
    if secret is None:
        msg = f"unknown E2E provider for LLM secret: {provider_name!r}"
        raise ValueError(msg)
    return secret


def load_batch_e2e_config() -> BatchE2EConfig:
    """Load batch E2E config from environment."""
    provider_name = os.environ.get("E2E_PROVIDER", "").strip()
    if not provider_name:
        msg = "E2E_PROVIDER is required (e.g. openai-agents)"
        raise RuntimeError(msg)

    lightspeed_provider = _E2E_PROVIDER_TO_LIGHTSPEED.get(provider_name)
    if lightspeed_provider is None:
        msg = f"unsupported E2E_PROVIDER: {provider_name!r}"
        raise RuntimeError(msg)

    model = resolve_model(provider_name)
    if not model:
        msg = f"no model resolved for provider {provider_name!r}"
        raise RuntimeError(msg)

    namespace = os.environ.get("E2E_NAMESPACE", DEFAULT_NAMESPACE).strip()
    session_id = os.environ.get("E2E_BATCH_SESSION_ID", "").strip() or secrets.token_hex(8)

    return BatchE2EConfig(
        namespace=namespace,
        sandbox_image=os.environ.get("SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE).strip(),
        service_account=os.environ.get("SANDBOX_SA", DEFAULT_SANDBOX_SA).strip(),
        llm_secret=resolve_llm_secret(provider_name),
        lightspeed_provider=lightspeed_provider,
        model=model,
        provider_name=provider_name,
        session_id=session_id,
        otel_endpoint=os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            f"{DEFAULT_OTEL_DEPLOYMENT}.{namespace}.svc:4317",
        ).strip(),
        otel_ca_secret=os.environ.get("OTEL_CA_SECRET", DEFAULT_OTEL_CA_SECRET).strip(),
        verify_full_fixtures=_truthy("E2E_BATCH_VERIFY_FIXTURES", default=False),
        extra_env=_provider_extra_env(provider_name),
        job_env=_session_job_env(),
    )


def _session_job_env() -> dict[str, str]:
    """Env vars forwarded from the pytest host onto every batch Job."""
    job_env: dict[str, str] = {}
    for key in ("LIGHTSPEED_MCP_SERVERS", "LIGHTSPEED_REASONING_CONFIG"):
        raw = os.environ.get(key, "").strip()
        if raw:
            job_env[key] = raw
    return job_env


def mock_mcp_service_url(namespace: str) -> str:
    """In-cluster MCP URL for the e2e mock server."""
    return f"http://{DEFAULT_MOCK_MCP_SERVICE}.{namespace}.svc:{DEFAULT_MOCK_MCP_PORT}/mcp"


def _provider_extra_env(provider_name: str) -> dict[str, str]:
    extra: dict[str, str] = {}
    if provider_name == PROVIDER_GEMINI_VERTEX_ADK:
        extra["LIGHTSPEED_MODEL_PROVIDER"] = "google"
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        if project:
            extra["LIGHTSPEED_PROVIDER_PROJECT"] = project
    elif provider_name == PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS:
        extra["LIGHTSPEED_MODEL_PROVIDER"] = "anthropic"
        project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "").strip()
        if project:
            extra["LIGHTSPEED_PROVIDER_PROJECT"] = project
    elif provider_name == PROVIDER_ANTHROPIC_BEDROCK_DEEPAGENTS:
        region = os.environ.get("AWS_REGION", "").strip()
        if region:
            extra["LIGHTSPEED_PROVIDER_REGION"] = region
    if provider_name in (PROVIDER_GEMINI_VERTEX_ADK, PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS):
        region = os.environ.get("CLOUD_ML_REGION", "").strip()
        if region:
            extra["LIGHTSPEED_PROVIDER_REGION"] = region
    return extra


def verify_batch_cluster(
    core_api: CoreV1Api,
    apps_api: AppsV1Api,
    config: BatchE2EConfig,
) -> list[str]:
    """Verify cluster prerequisites. Returns list of missing resource descriptions."""
    missing: list[str] = []

    try:
        core_api.read_namespace(config.namespace)
    except ApiException as exc:
        if exc.status == 404:
            missing.append(f"namespace/{config.namespace}")
        else:
            raise

    try:
        core_api.read_namespaced_secret(config.llm_secret, config.namespace)
    except ApiException as exc:
        if exc.status == 404:
            missing.append(f"secret/{config.llm_secret}")
        else:
            raise

    if not config.verify_full_fixtures:
        return missing

    try:
        core_api.read_namespaced_service_account(config.service_account, config.namespace)
    except ApiException as exc:
        if exc.status == 404:
            missing.append(f"serviceaccount/{config.service_account}")
        else:
            raise

    try:
        core_api.read_namespaced_secret(config.otel_ca_secret, config.namespace)
    except ApiException as exc:
        if exc.status == 404:
            missing.append(f"secret/{config.otel_ca_secret}")
        else:
            raise

    try:
        apps_api.read_namespaced_deployment(DEFAULT_OTEL_DEPLOYMENT, config.namespace)
    except ApiException as exc:
        if exc.status == 404:
            missing.append(f"deployment/{DEFAULT_OTEL_DEPLOYMENT}")
        else:
            raise

    if "LIGHTSPEED_MCP_SERVERS" in config.job_env:
        try:
            apps_api.read_namespaced_deployment(
                DEFAULT_MOCK_MCP_DEPLOYMENT,
                config.namespace,
            )
        except ApiException as exc:
            if exc.status == 404:
                missing.append(f"deployment/{DEFAULT_MOCK_MCP_DEPLOYMENT}")
            else:
                raise

    return missing


def setup_batch_suite(
    core_api: CoreV1Api,
    apps_api: AppsV1Api,
    config: BatchE2EConfig,
) -> None:
    """Session setup: credential preflight and cluster verification."""
    require_credentials(config.provider_name)
    missing = verify_batch_cluster(core_api, apps_api, config)
    if missing:
        hint = "run scripts/e2e-install-fixtures.sh (set E2E_BATCH_VERIFY_FIXTURES=1)"
        missing_list = ", ".join(missing)
        msg = (
            f"batch e2e cluster prerequisites missing in {config.namespace}: "
            f"{missing_list} ({hint})"
        )
        raise RuntimeError(msg)

    logger.info(
        "batch e2e suite ready: namespace=%s provider=%s model=%s session=%s",
        config.namespace,
        config.provider_name,
        config.model,
        config.session_id,
    )
