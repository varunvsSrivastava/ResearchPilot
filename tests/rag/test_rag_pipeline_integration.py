
"""Integration tests for RAG pipeline orchestration."""

from unittest.mock import Mock

from researchpilot.rag.embeddings.models import EmbeddingVector
from researchpilot.rag.pipeline import (
    RAGPipelineRequest,
    RAGPipelineService,
)
from researchpilot.rag.retrieval.service import SemanticRetrievalService
from researchpilot.rag.vectorstore.models import (
    VectorRecord,
    VectorSearchResponse,
    VectorSearchResult,
)


def test_rag_pipeline_integrates_semantic_retrieval_and_context():
    """Verify retrieval, ranking, and context assembly work together."""
    embedding_provider = Mock()
    embedding_provider.embed_text.return_value = EmbeddingVector(
        text="machine learning",
        vector=[0.1, 0.2, 0.3],
        dimensions=3,
        model="test-model",
    )

    vector_store = Mock()
    vector_store.search.return_value = VectorSearchResponse(
        results=[
            VectorSearchResult(
                record=VectorRecord(
                    id="evidence-1",
                    text="Lower-ranked evidence.",
                    vector=[0.1, 0.2, 0.3],
                    metadata={"paper_id": "paper-1"},
                ),
                score=0.75,
            ),
            VectorSearchResult(
                record=VectorRecord(
                    id="evidence-2",
                    text="Higher-ranked evidence.",
                    vector=[0.2, 0.3, 0.4],
                    metadata={"paper_id": "paper-2"},
                ),
                score=0.95,
            ),
        ],
        total_results=2,
    )

    retrieval_service = SemanticRetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    pipeline = RAGPipelineService(retrieval_service)

    result = pipeline.execute(
        RAGPipelineRequest(
            query="machine learning",
            top_k=2,
        )
    )

    assert result.query == "machine learning"
    assert result.total_results == 2
    assert result.context == (
        "[Source 1]\n"
        "Evidence ID: evidence-2\n"
        "Paper ID: paper-2\n"
        "Evidence:\n"
        "Higher-ranked evidence.\n\n"
        "[Source 2]\n"
        "Evidence ID: evidence-1\n"
        "Paper ID: paper-1\n"
        "Evidence:\n"
        "Lower-ranked evidence."
    )
    assert [
        item.record.id
        for item in result.retrieval_response.results
    ] == ["evidence-2", "evidence-1"]

    embedding_provider.embed_text.assert_called_once_with(
        "machine learning"
    )
    vector_store.search.assert_called_once()
