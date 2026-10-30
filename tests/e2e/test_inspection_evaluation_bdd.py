"""BDD discovery for classifier-only corpus measurements in the sandbox image."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_bdd import scenarios

# Eighteen 60s cases, provisioning/polling allowance, and bounded observation/cleanup.
# The suite's ordinary 600s timeout would interrupt the helper before its deadline.
pytestmark = [pytest.mark.e2e, pytest.mark.timeout(1350)]

_FEATURE = Path(__file__).parent / "features" / "inspection_evaluation.feature"
scenarios(_FEATURE.as_posix())
