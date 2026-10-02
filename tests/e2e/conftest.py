"""E2E fixtures — batch Job harness only (no HTTP server)."""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
from kubernetes import config as k8s_config  # type: ignore[import-untyped]
from kubernetes.client import (  # type: ignore[import-untyped]
    ApiClient,
    AppsV1Api,
    BatchV1Api,
    CoreV1Api,
    CustomObjectsApi,
)

from tests.e2e.batch_runner import run_batch_query
from tests.e2e.run_result import E2ERunResult, batch_to_run_result
from tests.e2e.suite_setup import (
    BatchE2EConfig,
    load_batch_e2e_config,
    setup_batch_suite,
)
from steps.given import *  # noqa: F403 — step fixtures must be in conftest namespace
from steps.when import *  # noqa: F403
from steps.then import *  # noqa: F403


@pytest.fixture
def bdd_context() -> dict[str, Any]:
    return {}


@pytest.fixture(scope="session")
def provider_name() -> str:
    name = os.environ.get("E2E_PROVIDER", "").strip()
    if not name:
        pytest.fail("E2E_PROVIDER is not set")
    return name


@pytest.fixture
def e2e_output_dir() -> Path | None:
    """Host-side output directory where skill tools write token files."""
    raw = os.environ.get("E2E_OUTPUT_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


@pytest.fixture(scope="session")
def _k8s_config_loaded() -> None:
    """Load kubeconfig once per session."""
    k8s_config.load_kube_config()


@pytest.fixture(scope="session")
def _k8s_api_client(_k8s_config_loaded: None) -> ApiClient:
    return ApiClient()


@pytest.fixture(scope="session")
def k8s_client(_k8s_api_client: ApiClient) -> CustomObjectsApi:
    """Authenticated CustomObjectsApi from KUBECONFIG."""
    return CustomObjectsApi(_k8s_api_client)


@pytest.fixture(scope="session")
def k8s_core_client(_k8s_api_client: ApiClient) -> CoreV1Api:
    """Authenticated CoreV1Api for pod log retrieval."""
    return CoreV1Api(_k8s_api_client)


@pytest.fixture(scope="session")
def k8s_batch_client(_k8s_api_client: ApiClient) -> BatchV1Api:
    """Authenticated BatchV1Api for sandbox batch Jobs."""
    return BatchV1Api(_k8s_api_client)


@pytest.fixture(scope="session")
def k8s_apps_client(_k8s_api_client: ApiClient) -> AppsV1Api:
    """Authenticated AppsV1Api for fixture verification."""
    return AppsV1Api(_k8s_api_client)


@pytest.fixture(scope="session")
def batch_e2e_config() -> BatchE2EConfig:
    """Resolved E2E configuration for batch Job runs."""
    return load_batch_e2e_config()


@pytest.fixture(scope="session", autouse=True)
def _batch_suite_lifecycle(
    batch_e2e_config: BatchE2EConfig,
    _k8s_api_client: ApiClient,
) -> None:
    """Session setup (operator TestMain equivalent)."""
    core_api = CoreV1Api(_k8s_api_client)
    apps_api = AppsV1Api(_k8s_api_client)
    setup_batch_suite(core_api, apps_api, batch_e2e_config)


@pytest.fixture
def run_runner(
    batch_e2e_config: BatchE2EConfig,
    k8s_core_client: CoreV1Api,
    k8s_batch_client: BatchV1Api,
    k8s_client: CustomObjectsApi,
) -> Callable[..., E2ERunResult]:
    """Launch a sandbox batch Job and return the assertion envelope."""

    def _run(
        query: str,
        *,
        system_prompt: str = "You are a helpful assistant. Follow instructions exactly.",
        output_schema: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        step: str = "analysis",
        wait_timeout_seconds: float = 600.0,
        timeout_ms: int | None = None,
        mount_skills: bool = False,
        job_env_overrides: dict[str, str] | None = None,
    ) -> E2ERunResult:
        batch = run_batch_query(
            batch_e2e_config,
            k8s_core_client,
            k8s_batch_client,
            k8s_client,
            query,
            system_prompt=system_prompt,
            output_schema=output_schema,
            context=context,
            step=step,
            wait_timeout_seconds=wait_timeout_seconds,
            timeout_ms=timeout_ms,
            mount_skills=mount_skills,
            job_env_overrides=job_env_overrides,
        )
        return batch_to_run_result(batch)

    return _run


@pytest.fixture
def scenario_cleanup() -> Generator[None, None, None]:
    """Yield fixture that runs cleanup.sh on test teardown.

    Expects SCENARIO_DIR env var to point to the scenario directory
    containing cleanup.sh. Skips cleanup if not set.
    """
    yield
    scenario_dir = os.environ.get("SCENARIO_DIR", "").strip()
    if not scenario_dir:
        return
    cleanup_script = Path(scenario_dir) / "cleanup.sh"
    if cleanup_script.is_file():
        result = subprocess.run(  # noqa: S603
            ["bash", str(cleanup_script)],  # noqa: S607
            check=False,
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger = logging.getLogger(__name__)
            logger.warning(
                "cleanup.sh exited %d\nstderr: %s",
                result.returncode,
                result.stderr.decode(errors="replace").strip(),
            )
