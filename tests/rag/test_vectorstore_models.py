
"""Tests for vector store models and interface."""

import pytest
from pydantic import ValidationError

from researchpilot.rag.vectorstore import (
    VectorRecord,
    VectorSearchResponse,
    VectorSearchResult,
    VectorStore,
)


@pytest.fixture
def vector_record():
    """Create a sample vector record."""
    return VectorRecord(
        id="record-1",
        text="ResearchPilot evidence",
        vector=[0.1, 0.2, 0.3],
        metadata={
            "source": "paper.pdf",
            "page": 2,
            "verified": True,
        },
    )


def test_vector_record_creation(vector_record):
    """Create a vector record with metadata."""
    assert vector_record.id == "record-1"
    assert vector_record.text == "ResearchPilot evidence"
    assert vector_record.metadata["source"] == "paper.pdf"
    assert vector_record.metadata["page"] == 2
    assert vector_record.metadata["verified"] is True


def test_vector_record_dimensions(vector_record):
    """Return the vector dimensions."""
    assert vector_record.dimensions == 3


def test_vector_record_rejects_empty_vector():
    """Reject an empty vector."""
    with pytest.raises(ValidationError):
        VectorRecord(
            id="record-1",
            text="Evidence",
            vector=[],
        )


def test_vector_search_result_creation(vector_record):
    """Create a vector search result."""
    result = VectorSearchResult(
        record=vector_record,
        score=0.95,
    )

    assert result.record.id == "record-1"
    assert result.score == 0.95


def test_vector_search_response_creation(vector_record):
    """Create a valid vector search response."""
    result = VectorSearchResult(
        record=vector_record,
        score=0.95,
    )

    response = VectorSearchResponse(
        results=[result],
        total_results=1,
    )

    assert response.total_results == 1
    assert response.is_valid


def test_vector_search_response_empty():
    """Create a valid empty search response."""
    response = VectorSearchResponse()

    assert response.results == []
    assert response.total_results == 0
    assert response.is_valid


def test_vector_search_response_invalid_count(vector_record):
    """Detect inconsistent result counts."""
    result = VectorSearchResult(
        record=vector_record,
        score=0.95,
    )

    response = VectorSearchResponse(
        results=[result],
        total_results=2,
    )

    assert not response.is_valid


def test_vector_store_is_abstract():
    """Prevent direct instantiation of the abstract vector store."""
    with pytest.raises(TypeError, match="abstract"):
        VectorStore()
