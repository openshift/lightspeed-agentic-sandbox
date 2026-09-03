"""Offline jsonschema tests for operator analysis shapes (not sent to provider APIs)."""

from __future__ import annotations

import jsonschema
import pytest

from tests.e2e.analysis_schemas import ANALYSIS_NO_ACTION_SCHEMA, ANALYSIS_WITH_COMPONENTS_SCHEMA

_MINIMAL_OPTION: dict = {
    "title": "Fix it",
    "diagnosis": {"summary": "s", "rootCause": "r"},
    "remediationPlan": {
        "description": "d",
        "actions": [{"command": "echo ok", "type": "pre-check", "description": "d"}],
        "reversible": "Reversible",
    },
    "verification": {"description": "d"},
    "components": [
        {
            "type": "t",
            "source": {"generator": "g", "timestamp": "ts"},
            "tokens": {
                "primary": {"value": "DIAG_x", "valid": True},
                "secondary": {"value": "VERIFY_x", "valid": True},
            },
            "audit": {
                "outcome": "pass",
                "checks_performed": ["c"],
                "findings": [{"check": "c", "result": "r", "severity": "info"}],
            },
        }
    ],
}

_DIAGNOSIS: dict = {"summary": "s", "rootCause": "r"}


def test_action_required_true_with_options():
    doc = {"actionRequired": "True", "options": [_MINIMAL_OPTION]}
    jsonschema.validate(doc, ANALYSIS_WITH_COMPONENTS_SCHEMA)


def test_action_required_true_rejects_empty_options():
    doc = {"actionRequired": "True", "options": []}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, ANALYSIS_WITH_COMPONENTS_SCHEMA)


def test_action_required_false_with_diagnosis():
    doc = {"actionRequired": "False", "options": [], "diagnosis": _DIAGNOSIS}
    jsonschema.validate(doc, ANALYSIS_NO_ACTION_SCHEMA)


def test_action_required_false_rejects_missing_diagnosis():
    doc = {"actionRequired": "False", "options": []}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, ANALYSIS_NO_ACTION_SCHEMA)


def test_false_alarm_doc_fails_live_action_required_schema():
    doc = {"actionRequired": "False", "options": [], "diagnosis": _DIAGNOSIS}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, ANALYSIS_WITH_COMPONENTS_SCHEMA)
