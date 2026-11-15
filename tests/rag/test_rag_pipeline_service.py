
"""Tests for RAG pipeline orchestration."""

from unittest.mock import MagicMock

from researchpilot.rag.pipeline import (
    RAGPipelineRequest,
    RAGPipelineService,
)
from researchpilot.rag.retrieval.models import RetrievalResponse
from researchpilot.rag.vectorstore.filters import MetadataFilter
from researchpilot.rag.vectorstore.models import (
    VectorRecord,
    VectorSearchResult,
)


def create_result(record_id: str, text: str, score: float) -> VectorSearchResult:
    """Create a vector search result for testing."""
    return VectorSearchResult(
        record=VectorRecord(
            id=record_id,
            text=text,
            vector=[0.1, 0.2],
            metadata={"paper_id": "paper-1"},
        ),
        score=score,
    )


def test_pipeline_assembles_context_in_retrieval_order():
    """Context should preserve the ranking order of retrieved evidence."""
    retrieval_service = MagicMock()
    retrieval_service.retrieve.return_value = RetrievalResponse(
        query="AI research",
        results=[
            create_result("evidence-1", "First evidence.", 0.9),
            create_result("evidence-2", "Second evidence.", 0.8),
        ],
        total_results=2,
    )

    pipeline = RAGPipelineService(retrieval_service)

    result = pipeline.execute(
        RAGPipelineRequest(query="AI research", top_k=2)
    )

    assert result.query == "AI research"
    assert result.total_results == 2
    assert result.context == (
        "[Source 1]\n"
        "Evidence ID: evidence-1\n"
        "Paper ID: paper-1\n"
        "Evidence:\n"
        "First evidence.\n\n"
        "[Source 2]\n"
        "Evidence ID: evidence-2\n"
        "Paper ID: paper-1\n"
        "Evidence:\n"
        "Second evidence."
    )
    assert result.retrieval_response.total_results == 2

    retrieval_service.retrieve.assert_called_once()


def test_pipeline_forwards_metadata_filter():
    """Metadata filters should be forwarded to semantic retrieval."""
    retrieval_service = MagicMock()
    retrieval_service.retrieve.return_value = RetrievalResponse(
        query="AI research",
        results=[],
        total_results=0,
    )

    metadata_filter = MetadataFilter(
        conditions={"paper_id": "paper-1"}
    )
    pipeline = RAGPipelineService(retrieval_service)

    pipeline.execute(
        RAGPipelineRequest(
            query="AI research",
            metadata_filter=metadata_filter,
        )
    )

    retrieval_request = retrieval_service.retrieve.call_args.args[0]

    assert retrieval_request.query == "AI research"
    assert retrieval_request.top_k == 5
    assert retrieval_request.metadata_filter == metadata_filter


def test_pipeline_handles_empty_retrieval_results():
    """An empty retrieval response should produce an empty context."""
    retrieval_service = MagicMock()
    retrieval_service.retrieve.return_value = RetrievalResponse(
        query="Unknown topic",
        results=[],
        total_results=0,
    )

    pipeline = RAGPipelineService(retrieval_service)

    result = pipeline.execute(
        RAGPipelineRequest(query="Unknown topic")
    )

    assert result.context == ""
    assert result.total_results == 0
    assert result.retrieval_response.results == []
