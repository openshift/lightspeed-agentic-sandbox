"""Tests for provider factory."""

from unittest.mock import Mock

import pytest

from lightspeed_agentic.factory import create_provider

from .helpers import make_model_route


def test_create_provider_unknown():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider("nonexistent")


def test_create_provider_requires_name():
    with pytest.raises(TypeError):
        create_provider()  # type: ignore[call-arg]


@pytest.mark.parametrize("name", ["openai", "gemini"])
def test_legacy_provider_is_removed(name):
    with pytest.raises(ValueError, match=r"Unknown provider: .*Supported: deepagents$"):
        create_provider(name)


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
