
"""Embedding interfaces and models."""

from researchpilot.rag.embeddings.base import EmbeddingProvider
from researchpilot.rag.embeddings.models import (
    EmbeddingBatch,
    EmbeddingVector,
)
from researchpilot.rag.embeddings.openai_embeddings import (
    EmbeddingProviderError,
    OpenAIEmbeddingProvider,
)

__all__ = [
    "EmbeddingProvider",
    "EmbeddingVector",
    "EmbeddingBatch",
    "EmbeddingProviderError",
    "OpenAIEmbeddingProvider",
]
