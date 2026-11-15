
"""OpenAI embedding provider implementation."""

from openai import OpenAI, OpenAIError

from researchpilot.rag.embeddings.base import EmbeddingProvider
from researchpilot.rag.embeddings.models import (
    EmbeddingBatch,
    EmbeddingVector,
)

from researchpilot.config import (
    get_openai_embedding_dimensions,
    get_openai_embedding_model,
)

class EmbeddingProviderError(Exception):
    """Raised when embedding generation fails."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Generate text embeddings using OpenAI models."""

    def __init__(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model or get_openai_embedding_model()
        self.dimensions = (
            dimensions
            if dimensions is not None
            else get_openai_embedding_dimensions()
        )

        if not self.model.strip():
            raise ValueError("Embedding model cannot be empty.")

        if self.dimensions < 1:
            raise ValueError("Embedding dimensions must be positive.")

        self.client = client or OpenAI()

    def embed_text(self, text: str) -> EmbeddingVector:
        """Generate an embedding for a single text."""
        results = self.embed_texts([text])

        return results.embeddings[0]

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        """Generate embeddings for multiple texts."""

        if not texts:
            return EmbeddingBatch(
                embeddings=[],
                model=self.model,
                total_embeddings=0,
            )

        if any(not text.strip() for text in texts):
            raise ValueError("Embedding inputs cannot be empty.")

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
            )
        except OpenAIError as exc:
            raise EmbeddingProviderError(
                "OpenAI embedding generation failed."
            ) from exc

        ordered_data = sorted(
            response.data,
            key=lambda item: item.index,
        )

        if len(ordered_data) != len(texts):
            raise EmbeddingProviderError(
                "OpenAI returned an unexpected number of embeddings."
            )

        embeddings = [
            EmbeddingVector(
                text=texts[item.index],
                vector=item.embedding,
                dimensions=len(item.embedding),
                model=self.model,
            )
            for item in ordered_data
        ]

        if any(not embedding.is_valid for embedding in embeddings):
            raise EmbeddingProviderError(
                "Embedding vector dimensions are inconsistent."
            )

        return EmbeddingBatch(
            embeddings=embeddings,
            model=self.model,
            total_embeddings=len(embeddings),
        )
