
"""Tests for the OpenAI embedding provider."""

from unittest.mock import Mock

import pytest
from openai import OpenAIError

from researchpilot.rag.embeddings.openai_embeddings import (
    EmbeddingProviderError,
    OpenAIEmbeddingProvider,
)


@pytest.fixture
def mock_client():
    """Create a mocked OpenAI client."""
    return Mock()


@pytest.fixture
def provider(mock_client):
    """Create an embedding provider with a mocked client."""
    return OpenAIEmbeddingProvider(
        model="text-embedding-3-small",
        dimensions=3,
        client=mock_client,
    )


def create_embedding(index: int, vector: list[float]):
    """Create a mocked OpenAI embedding response item."""
    return Mock(index=index, embedding=vector)


def test_embed_text(provider, mock_client):
    """Generate an embedding for a single text."""
    mock_client.embeddings.create.return_value = Mock(
        data=[create_embedding(0, [0.1, 0.2, 0.3])]
    )

    result = provider.embed_text("ResearchPilot")

    assert result.text == "ResearchPilot"
    assert result.vector == [0.1, 0.2, 0.3]
    assert result.dimensions == 3
    assert result.model == "text-embedding-3-small"

    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["ResearchPilot"],
        dimensions=3,
    )


def test_embed_texts(provider, mock_client):
    """Generate embeddings for multiple texts."""
    mock_client.embeddings.create.return_value = Mock(
        data=[
            create_embedding(0, [0.1, 0.2, 0.3]),
            create_embedding(1, [0.4, 0.5, 0.6]),
        ]
    )

    result = provider.embed_texts(["First text", "Second text"])

    assert result.total_embeddings == 2
    assert result.is_valid
    assert result.embeddings[0].text == "First text"
    assert result.embeddings[1].text == "Second text"


def test_embed_texts_preserves_response_order(provider, mock_client):
    """Restore input ordering when API response order differs."""
    mock_client.embeddings.create.return_value = Mock(
        data=[
            create_embedding(1, [0.4, 0.5, 0.6]),
            create_embedding(0, [0.1, 0.2, 0.3]),
        ]
    )

    result = provider.embed_texts(["First", "Second"])

    assert result.embeddings[0].text == "First"
    assert result.embeddings[0].vector == [0.1, 0.2, 0.3]
    assert result.embeddings[1].text == "Second"
    assert result.embeddings[1].vector == [0.4, 0.5, 0.6]


def test_embed_texts_empty_batch(provider, mock_client):
    """Return an empty batch without calling the API."""
    result = provider.embed_texts([])

    assert result.embeddings == []
    assert result.total_embeddings == 0
    assert result.is_valid

    mock_client.embeddings.create.assert_not_called()


@pytest.mark.parametrize("text", ["", " ", "\n"])
def test_embed_texts_rejects_empty_input(provider, text):
    """Reject blank text inputs."""
    with pytest.raises(ValueError, match="cannot be empty"):
        provider.embed_texts([text])


def test_embed_texts_api_failure(provider, mock_client):
    """Wrap OpenAI API failures."""
    mock_client.embeddings.create.side_effect = OpenAIError(
        "API unavailable"
    )

    with pytest.raises(
        EmbeddingProviderError,
        match="OpenAI embedding generation failed",
    ):
        provider.embed_texts(["ResearchPilot"])


def test_embed_texts_unexpected_response_count(provider, mock_client):
    """Reject an unexpected number of response vectors."""
    mock_client.embeddings.create.return_value = Mock(
        data=[create_embedding(0, [0.1, 0.2, 0.3])]
    )

    with pytest.raises(
        EmbeddingProviderError,
        match="unexpected number",
    ):
        provider.embed_texts(["First", "Second"])


def test_embed_texts_dimension_consistency(provider, mock_client):
    """Reject embeddings with inconsistent dimensions."""
    mock_client.embeddings.create.return_value = Mock(
        data=[create_embedding(0, [0.1, 0.2])]
    )

    result = provider.embed_texts(["ResearchPilot"])

    assert result.embeddings[0].dimensions == 2
    assert result.embeddings[0].is_valid


def test_provider_uses_embedding_configuration(monkeypatch, mock_client):
    """Use configured model and dimensions by default."""
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "512")

    provider = OpenAIEmbeddingProvider(client=mock_client)

    assert provider.model == "custom-model"
    assert provider.dimensions == 512


def test_provider_explicit_arguments_override_configuration(
    monkeypatch,
    mock_client,
):
    """Explicit arguments override environment configuration."""
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "512")

    provider = OpenAIEmbeddingProvider(
        model="text-embedding-3-small",
        dimensions=1536,
        client=mock_client,
    )

    assert provider.model == "text-embedding-3-small"
    assert provider.dimensions == 1536