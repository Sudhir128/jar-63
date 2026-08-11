"""Tests for the provider factory."""

from __future__ import annotations

import pytest

from app.llm.factory import ProviderFactory, UnknownProviderError
from app.llm.providers.ollama import OllamaClient
from app.llm.providers.openai_compatible import OpenAICompatibleClient
from tests.llm_helpers import make_llm_settings


def test_factory_creates_ollama_client() -> None:
    factory = ProviderFactory(make_llm_settings())
    client = factory.create("ollama")
    assert isinstance(client, OllamaClient)
    assert client.provider_name == "ollama"


def test_factory_creates_openai_compatible_client() -> None:
    settings = make_llm_settings(openai_compat_enabled=True)
    factory = ProviderFactory(settings)
    client = factory.create("openai_compatible")
    assert isinstance(client, OpenAICompatibleClient)


def test_factory_openai_compatible_not_configured_raises() -> None:
    factory = ProviderFactory(make_llm_settings(openai_compat_enabled=False))
    with pytest.raises(UnknownProviderError, match="not configured"):
        factory.create("openai_compatible")


def test_factory_unknown_provider_raises() -> None:
    factory = ProviderFactory(make_llm_settings())
    with pytest.raises(UnknownProviderError, match="Unknown LLM provider"):
        factory.create("bogus")


def test_factory_create_for_model() -> None:
    from app.llm.models import ModelDefinition

    factory = ProviderFactory(make_llm_settings())
    model = ModelDefinition(model_id="x", provider="ollama")
    client = factory.create_for_model(model)
    assert isinstance(client, OllamaClient)
