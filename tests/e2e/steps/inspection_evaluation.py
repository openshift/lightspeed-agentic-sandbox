"""Classifier-only corpus measurements through the normal BDD fixtures."""

from __future__ import annotations

import os
from pathlib import Path

from kubernetes.client import BatchV1Api, CoreV1Api  # type: ignore[import-untyped]
from pytest_bdd import then, when

from tests.e2e.suite_setup import BatchE2EConfig
from tests.evaluation.batch import run_evaluation


@when(
    "I evaluate the labeled corpus with only the production classifier",
    target_fixture="corpus_evaluation_status",
)
def evaluate_inspection_corpus(
    batch_e2e_config: BatchE2EConfig,
    k8s_core_client: CoreV1Api,
    k8s_batch_client: BatchV1Api,
    tmp_path: Path,
) -> int:
    """Use the existing cluster clients/config; retain reports even on failure."""
    directory = os.environ.get("ARTIFACT_DIR", "").strip()
    return run_evaluation(
        batch_e2e_config,
        k8s_core_client,
        k8s_batch_client,
        output_dir=Path(directory) if directory else tmp_path,
        case_timeout=60,
    )


@then("the corpus evaluation completes without operational errors")
def assert_corpus_evaluation_complete(corpus_evaluation_status: int) -> None:
    """Operational success means retained measurements, not an accuracy gate."""
    assert corpus_evaluation_status == 0, (
        "Classifier corpus evaluation incomplete or failed; see inspection-evaluation JSON report"
    )
