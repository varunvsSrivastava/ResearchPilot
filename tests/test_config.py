
"""Tests for project configuration."""

import pytest

from researchpilot.config import (
    get_openai_embedding_dimensions,
    get_openai_embedding_model,
)


def test_embedding_model_default(monkeypatch):
    """Return the default embedding model."""
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)

    assert get_openai_embedding_model() == "text-embedding-3-small"


def test_embedding_model_override(monkeypatch):
    """Return the configured embedding model."""
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "custom-model")

    assert get_openai_embedding_model() == "custom-model"


def test_embedding_dimensions_default(monkeypatch):
    """Return the default embedding dimensions."""
    monkeypatch.delenv("OPENAI_EMBEDDING_DIMENSIONS", raising=False)

    assert get_openai_embedding_dimensions() == 1536


def test_embedding_dimensions_override(monkeypatch):
    """Return the configured embedding dimensions."""
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "512")

    assert get_openai_embedding_dimensions() == 512


def test_embedding_dimensions_reject_non_positive(monkeypatch):
    """Reject non-positive embedding dimensions."""
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "0")

    with pytest.raises(ValueError, match="must be positive"):
        get_openai_embedding_dimensions()


def test_embedding_dimensions_reject_invalid_value(monkeypatch):
    """Reject non-integer embedding dimensions."""
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "invalid")

    with pytest.raises(ValueError):
        get_openai_embedding_dimensions()
