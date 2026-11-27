"""RBAC probe steps — SA allow/deny assertions (OLS-5022)."""

from __future__ import annotations

from typing import Any

import pytest
from kubernetes.client import ApiException, CoreV1Api, CustomObjectsApi  # type: ignore[import-untyped]
from pytest_bdd import given, then, when

from tests.e2e.rbac_probe import (
    cleanup_probe_result_cr,
    cleanup_probe_secret,
    create_sa_api_client,
    generate_probe_name,
    probe_create_result_cr,
    probe_create_secret,
    probe_read_result_cr,
)
from tests.e2e.suite_setup import BatchE2EConfig


@given("the sandbox service account is available")
def sa_available(
    bdd_context: dict[str, Any],
    batch_e2e_config: BatchE2EConfig,
    k8s_core_client: CoreV1Api,
) -> None:
    """Verify the SA exists and create an SA-scoped API client."""
    try:
        k8s_core_client.read_namespaced_service_account(
            batch_e2e_config.service_account,
            batch_e2e_config.namespace,
        )
    except ApiException as exc:
        if exc.status == 404:
            pytest.skip(
                f"ServiceAccount {batch_e2e_config.service_account} not found in "
                f"{batch_e2e_config.namespace} (run scripts/e2e-install-fixtures.sh)"
            )
        raise

    sa_client = create_sa_api_client(
        k8s_core_client.api_client,
        batch_e2e_config.namespace,
        batch_e2e_config.service_account,
    )
    bdd_context["sa_client"] = sa_client
    bdd_context["probe_namespace"] = batch_e2e_config.namespace


@when("the RBAC probe creates and reads an AnalysisResult using SA credentials")
def rbac_probe_create_read(
    bdd_context: dict[str, Any],
    k8s_client: CustomObjectsApi,
) -> None:
    """Create and read an AnalysisResult using SA credentials, then clean up."""
    sa_custom = CustomObjectsApi(bdd_context["sa_client"])
    namespace = bdd_context["probe_namespace"]
    result_name = generate_probe_name()

    try:
        probe_create_result_cr(sa_custom, namespace, result_name)
        bdd_context["probe_create_ok"] = True

        probe_read_result_cr(sa_custom, namespace, result_name)
        bdd_context["probe_read_ok"] = True
    finally:
        cleanup_probe_result_cr(k8s_client, namespace, result_name)


@when("the RBAC probe attempts to create a Secret using SA credentials")
def rbac_probe_create_secret(
    bdd_context: dict[str, Any],
    k8s_core_client: CoreV1Api,
) -> None:
    """Attempt to create a Secret using SA credentials."""
    sa_core = CoreV1Api(bdd_context["sa_client"])
    namespace = bdd_context["probe_namespace"]
    secret_name = generate_probe_name()

    status = probe_create_secret(sa_core, namespace, secret_name)
    bdd_context["denied_status"] = status

    # Defensive cleanup if the Secret was unexpectedly created
    if status == 200:
        cleanup_probe_secret(k8s_core_client, namespace, secret_name)


@then("the allowed Result CR operations succeed")
def assert_allowed_ops(bdd_context: dict[str, Any]) -> None:
    assert bdd_context.get("probe_create_ok") is True, "SA failed to create AnalysisResult"
    assert bdd_context.get("probe_read_ok") is True, "SA failed to read AnalysisResult"


@then("the operation is denied with HTTP 403 Forbidden")
def assert_denied_403(bdd_context: dict[str, Any]) -> None:
    status = bdd_context.get("denied_status")
    assert status == 403, (
        f"Expected HTTP 403 Forbidden but got {status}; "
        "the SA should not have permission to create Secrets"
    )
