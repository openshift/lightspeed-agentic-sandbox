"""Unit tests for operator analysis token placement helpers."""

from __future__ import annotations

import pytest

from tests.e2e.analysis_tokens import assert_skill_tokens_in_response


def test_assert_skill_tokens_requires_values_in_components():
    raw = {
        "options": [
            {
                "components": [
                    {
                        "tokens": {
                            "primary": {"value": "DIAG_abc", "valid": True},
                            "secondary": {"value": "VERIFY_xyz", "valid": True},
                        }
                    }
                ]
            }
        ]
    }
    assert_skill_tokens_in_response(raw, "test")


def test_assert_skill_tokens_rejects_tokens_outside_components():
    raw = {
        "diagnosis": {"summary": "DIAG_leak", "rootCause": "VERIFY_leak"},
        "options": [
            {
                "components": [
                    {
                        "tokens": {
                            "primary": {"value": "other", "valid": True},
                            "secondary": {"value": "nope", "valid": True},
                        }
                    }
                ]
            }
        ],
    }
    with pytest.raises(AssertionError, match="DIAG_"):
        assert_skill_tokens_in_response(raw, "test")
