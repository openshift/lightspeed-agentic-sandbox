"""Tests for provider factory."""

import importlib
from unittest.mock import Mock

import pytest

from lightspeed_agentic.factory import create_provider

from .conftest import make_model_route


def test_create_provider_unknown():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider("nonexistent")


def test_create_provider_requires_name():
    with pytest.raises(TypeError):
        create_provider()  # type: ignore[call-arg]


def test_create_provider_explicit_name():
    # SDK might not be installed — just verify the right import is attempted
    for name in ("gemini", "openai"):
        try:
            provider = create_provider(name)
            assert provider.name == name
        except ImportError:
            pass


def test_deepagents_provider_receives_resolved_route():
    create_model = Mock()
    route = make_model_route(create_model)

    provider = create_provider("deepagents", route)

    assert provider.name == "deepagents"
    assert provider._model_route is route
    create_model.assert_not_called()


def test_deepagents_provider_requires_resolved_route():
    with pytest.raises(ValueError, match="resolved model route"):
        create_provider("deepagents")


def test_openai_provider_module_imports_without_eager_optional_sdk_imports():
    module = importlib.import_module("lightspeed_agentic.providers.openai")
    assert module.OpenAIProvider().name == "openai"
