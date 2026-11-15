
"""Tests for RAG retrieval tool."""

import pytest
from pydantic import ValidationError

from unittest.mock import MagicMock
from researchpilot.tools.rag_retrieval import (
    RAGRetrievalInput,
    RAGRetrievalTool,
)

def test_valid_retrieval_input():
    arguments = RAGRetrievalInput(
        query="retrieval augmented generation",
        top_k=10,
    )

    assert arguments.query == "retrieval augmented generation"
    assert arguments.top_k == 10

def test_default_top_k():
    arguments = RAGRetrievalInput(
        query="large language models",
    )

    assert arguments.top_k == 5


def test_short_query_rejected():
    with pytest.raises(ValidationError):
        RAGRetrievalInput(query="AI")


def test_invalid_top_k_rejected():
    with pytest.raises(ValidationError):
        RAGRetrievalInput(
            query="machine learning",
            top_k=100,
        )


def test_zero_top_k_rejected():
    with pytest.raises(ValidationError):
        RAGRetrievalInput(
            query="machine learning",
            top_k=0,
        )


@pytest.fixture
def retrieval_service():
    service = MagicMock()
    service.retrieve.return_value = "retrieval_result"
    return service


@pytest.fixture
def rag_tool(retrieval_service):
    return RAGRetrievalTool(retrieval_service)


def test_retrieval_executes(rag_tool):
    result = rag_tool.execute(
        RAGRetrievalInput(query="retrieval augmented generation")
    )

    assert result == "retrieval_result"


def test_retrieval_service_called_once(
    rag_tool,
    retrieval_service,
):
    rag_tool.execute(
        RAGRetrievalInput(query="large language models")
    )

    retrieval_service.retrieve.assert_called_once()


def test_query_forwarded(rag_tool, retrieval_service):
    rag_tool.execute(
        RAGRetrievalInput(query="machine learning")
    )

    request = retrieval_service.retrieve.call_args.args[0]
    assert request.query == "machine learning"


def test_top_k_forwarded(rag_tool, retrieval_service):
    rag_tool.execute(
        RAGRetrievalInput(
            query="deep learning",
            top_k=10,
        )
    )

    request = retrieval_service.retrieve.call_args.args[0]
    assert request.top_k == 10


def test_retrieval_result_returned_unchanged(
    rag_tool,
    retrieval_service,
):
    expected = object()
    retrieval_service.retrieve.return_value = expected

    result = rag_tool.execute(
        RAGRetrievalInput(query="AI research")
    )

    assert result is expected
