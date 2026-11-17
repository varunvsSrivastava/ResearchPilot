
"""Tests for semantic retrieval service."""

from unittest.mock import Mock
from researchpilot.rag.retrieval.ranking.base import RankingStrategy
from researchpilot.rag.retrieval.ranking.similarity_ranker import (
    SimilarityRanker,
)
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
        query_vector=embedding_provider.embed_text.return_value.vector,
        top_k=3,
        metadata_filter=None,
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

def test_retrieve_passes_metadata_filter(
    retrieval_service,
    embedding_provider,
    vector_store,
):
    from researchpilot.rag.vectorstore.filters import MetadataFilter

    metadata_filter = MetadataFilter(
        conditions={"source": "paper-1"}
    )

    request = RetrievalRequest(
        query="machine learning",
        top_k=3,
        metadata_filter=metadata_filter,
    )

    retrieval_service.retrieve(request)

    vector_store.search.assert_called_once_with(
        query_vector=embedding_provider.embed_text.return_value.vector,
        top_k=3,
        metadata_filter=metadata_filter,
    )


def test_retrieve_uses_custom_ranking_strategy(
    embedding_provider,
    vector_store,
):
    """Verify that retrieval delegates ranking to the injected strategy."""
    ranking_strategy = Mock(spec=RankingStrategy)

    search_results = vector_store.search.return_value.results
    ranking_strategy.rank.return_value = search_results

    service = SemanticRetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        ranking_strategy=ranking_strategy,
    )

    response = service.retrieve(
        RetrievalRequest(query="machine learning")
    )

    ranking_strategy.rank.assert_called_once_with(search_results)
    assert response.results == search_results


def test_retrieve_returns_custom_ranked_results(
    embedding_provider,
    vector_store,
):
    """Return results in the order provided by the ranking strategy."""
    search_results = vector_store.search.return_value.results

    ranking_strategy = Mock(spec=RankingStrategy)
    ranking_strategy.rank.return_value = list(reversed(search_results))

    service = SemanticRetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        ranking_strategy=ranking_strategy,
    )

    response = service.retrieve(
        RetrievalRequest(query="machine learning")
    )

    assert response.results == list(reversed(search_results))
    assert response.total_results == len(response.results)
    assert response.is_valid


def test_retrieve_uses_default_similarity_ranker(
    embedding_provider,
    vector_store,
):
    """Use SimilarityRanker when no ranking strategy is provided."""
    first_result = vector_store.search.return_value.results[0]

    second_result = VectorSearchResult(
        record=VectorRecord(
            id="evidence-2",
            text="Additional machine learning evidence",
            vector=[0.2, 0.3, 0.4],
        ),
        score=0.99,
    )

    vector_store.search.return_value = VectorSearchResponse(
        results=[first_result, second_result],
        total_results=2,
    )

    service = SemanticRetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    assert isinstance(service.ranking_strategy, SimilarityRanker)

    response = service.retrieve(
        RetrievalRequest(query="machine learning")
    )

    assert [result.record.id for result in response.results] == [
        "evidence-2",
        "evidence-1",
    ]
