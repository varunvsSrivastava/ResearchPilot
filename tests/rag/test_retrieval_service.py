
"""Tests for semantic retrieval service."""

from unittest.mock import Mock

import pytest

from researchpilot.rag.embeddings.models import EmbeddingVector
from researchpilot.rag.retrieval.models import (
    RetrievalRequest,
)
from researchpilot.rag.retrieval.service import (
    SemanticRetrievalService,
)
from researchpilot.rag.vectorstore.models import (
    VectorRecord,
    VectorSearchResponse,
    VectorSearchResult,
)


@pytest.fixture
def embedding_provider():
    provider = Mock()
    provider.embed_text.return_value = EmbeddingVector(
        text="machine learning",
        vector=[0.1, 0.2, 0.3],
        dimensions=3,
        model="test-model",
    )
    return provider


@pytest.fixture
def vector_store():
    store = Mock()
    store.search.return_value = VectorSearchResponse(
        results=[
            VectorSearchResult(
                record=VectorRecord(
                    id="evidence-1",
                    text="Machine learning research evidence",
                    vector=[0.1, 0.2, 0.3],
                    metadata={"paper_id": "paper-1"},
                ),
                score=0.95,
            )
        ],
        total_results=1,
    )
    return store


@pytest.fixture
def retrieval_service(embedding_provider, vector_store):
    return SemanticRetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )


def test_retrieve_returns_semantically_relevant_results(
    retrieval_service,
    embedding_provider,
    vector_store,
):
    request = RetrievalRequest(
        query="machine learning",
        top_k=3,
    )

    response = retrieval_service.retrieve(request)

    assert response.query == "machine learning"
    assert response.total_results == 1
    assert response.is_valid
    assert response.results[0].record.id == "evidence-1"
    assert response.results[0].score == 0.95

    embedding_provider.embed_text.assert_called_once_with(
        "machine learning"
    )
    vector_store.search.assert_called_once_with(
        query_vector=[0.1, 0.2, 0.3],
        top_k=3,
    )


def test_retrieve_returns_empty_results(
    retrieval_service,
    vector_store,
):
    vector_store.search.return_value = VectorSearchResponse(
        results=[],
        total_results=0,
    )

    response = retrieval_service.retrieve(
        RetrievalRequest(query="unknown topic")
    )

    assert response.results == []
    assert response.total_results == 0
    assert response.is_valid


def test_retrieve_rejects_invalid_embedding(
    retrieval_service,
    embedding_provider,
    vector_store,
):
    embedding_provider.embed_text.return_value = EmbeddingVector(
        text="machine learning",
        vector=[0.1, 0.2],
        dimensions=3,
        model="test-model",
    )

    with pytest.raises(
        ValueError,
        match="Invalid query embedding dimensions",
    ):
        retrieval_service.retrieve(
            RetrievalRequest(query="machine learning")
        )

    vector_store.search.assert_not_called()
