"""pytest-bdd glue for sandbox ServiceAccount RBAC probe scenarios (OLS-5022)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_bdd import scenarios

pytestmark = pytest.mark.e2e

_FEATURES = Path(__file__).parent / "features"
scenarios((_FEATURES / "rbac_probe.feature").as_posix())
