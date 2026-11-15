
"""Semantic retrieval service implementation."""

from researchpilot.rag.embeddings.base import EmbeddingProvider
from researchpilot.rag.retrieval.base import RetrievalService
from researchpilot.rag.retrieval.models import (
    RetrievalRequest,
    RetrievalResponse,
)
from researchpilot.rag.vectorstore.base import VectorStore


class SemanticRetrievalService(RetrievalService):
    """Retrieve relevant evidence using semantic similarity."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        """Initialize the semantic retrieval service."""
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        """Retrieve relevant evidence for a research query."""
        query_embedding = self.embedding_provider.embed_text(
            request.query
        )

        if not query_embedding.is_valid:
            raise ValueError("Invalid query embedding dimensions.")

        search_response = self.vector_store.search(
            query_vector=query_embedding,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        )

        return RetrievalResponse(
            query=request.query,
            results=search_response.results,
            total_results=len(search_response.results),
        )
