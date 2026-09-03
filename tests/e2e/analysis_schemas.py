"""Operator analysis JSON Schemas for batch BDD / eval find-token coverage.

Tokens (DIAG_/VERIFY_) live only under ``components`` so published AnalysisResult
status passes CRD validation.

Live batch Jobs use ``ANALYSIS_WITH_COMPONENTS_SCHEMA`` (flat, no allOf) because
structured-output APIs reject conditional JSON Schema. Offline tests also use
``ANALYSIS_NO_ACTION_SCHEMA`` for the operator false-alarm path (never sent to providers).
"""

from __future__ import annotations

from typing import Any

_S = {"type": "string"}
_DIAG = {
    "type": "object",
    "properties": {"summary": _S, "rootCause": _S},
    "required": ["summary", "rootCause"],
}
_TOKEN = {
    "type": "object",
    "properties": {"value": _S, "valid": {"type": "boolean"}},
    "required": ["value", "valid"],
}
_FINDING = {
    "type": "object",
    "properties": {
        "check": _S,
        "result": _S,
        "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
    },
    "required": ["check", "result", "severity"],
}
_COMPONENT = {
    "type": "object",
    "properties": {
        "type": _S,
        "source": {
            "type": "object",
            "properties": {"generator": _S, "timestamp": _S},
            "required": ["generator", "timestamp"],
        },
        "tokens": {
            "type": "object",
            "properties": {"primary": _TOKEN, "secondary": _TOKEN},
            "required": ["primary", "secondary"],
        },
        "audit": {
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "enum": ["pass", "fail", "partial"]},
                "checks_performed": {"type": "array", "items": _S},
                "findings": {"type": "array", "items": _FINDING},
            },
            "required": ["outcome", "checks_performed", "findings"],
        },
    },
    "required": ["type", "source", "tokens", "audit"],
}
_OPTION = {
    "type": "object",
    "properties": {
        "title": _S,
        "diagnosis": _DIAG,
        "remediationPlan": {
            "type": "object",
            "properties": {
                "description": _S,
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {"command": _S, "type": _S, "description": _S},
                        "required": ["command", "type", "description"],
                    },
                },
                "reversible": {
                    "type": "string",
                    "enum": ["Reversible", "Irreversible", "Partial"],
                },
            },
            "required": ["description", "actions", "reversible"],
        },
        "verification": {
            "type": "object",
            "properties": {"description": _S},
            "required": ["description"],
        },
        "components": {"type": "array", "minItems": 1, "items": _COMPONENT},
    },
    "required": ["title", "diagnosis", "remediationPlan", "verification", "components"],
}

ANALYSIS_WITH_COMPONENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "actionRequired": {"type": "string", "enum": ["True", "False"]},
        "diagnosis": _DIAG,
        "options": {"type": "array", "minItems": 1, "items": _OPTION},
    },
    "required": ["actionRequired", "options"],
}

# Offline validation only — operator false-alarm shape (not used on batch Job input).
ANALYSIS_NO_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "actionRequired": {"type": "string", "enum": ["False"]},
        "options": {"type": "array", "maxItems": 0},
        "diagnosis": _DIAG,
    },
    "required": ["actionRequired", "options", "diagnosis"],
}
