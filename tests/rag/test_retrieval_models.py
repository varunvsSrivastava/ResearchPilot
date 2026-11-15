
"""Tests for semantic retrieval models."""

import pytest
from pydantic import ValidationError

from researchpilot.rag.retrieval.models import (
    RetrievalRequest,
    RetrievalResponse,
)


def test_retrieval_request_defaults():
    request = RetrievalRequest(query="machine learning")

    assert request.query == "machine learning"
    assert request.top_k == 5


def test_retrieval_request_custom_top_k():
    request = RetrievalRequest(query="deep learning", top_k=10)

    assert request.top_k == 10


def test_retrieval_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="")


def test_retrieval_request_rejects_invalid_top_k():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="AI", top_k=0)


def test_retrieval_response_defaults():
    response = RetrievalResponse(query="AI")

    assert response.results == []
    assert response.total_results == 0
    assert response.is_valid


def test_retrieval_response_detects_count_mismatch():
    response = RetrievalResponse(query="AI", total_results=2)

    assert not response.is_valid
