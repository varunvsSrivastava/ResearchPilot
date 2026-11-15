
"""Tests for evidence indexing models and interface."""

import pytest
from pydantic import ValidationError

from researchpilot.rag.indexing import (
    EvidenceIndexer,
    IndexingRequest,
    IndexingResult,
)


def test_indexing_result_success():
    """Represent a successful indexing operation."""
    result = IndexingResult(
        total_evidence=10,
        indexed_evidence=10,
        failed_evidence=0,
    )

    assert result.is_successful
    assert result.total_evidence == 10


def test_indexing_result_failure():
    """Represent an indexing operation with failures."""
    result = IndexingResult(
        total_evidence=10,
        indexed_evidence=8,
        failed_evidence=2,
        errors=["Evidence indexing failed"],
    )

    assert not result.is_successful
    assert result.failed_evidence == 2


def test_indexing_result_partial_success():
    """Detect incomplete indexing without explicit failures."""
    result = IndexingResult(
        total_evidence=10,
        indexed_evidence=8,
        failed_evidence=0,
    )

    assert not result.is_successful


def test_indexing_result_empty():
    """Represent an empty indexing operation."""
    result = IndexingResult()

    assert result.total_evidence == 0
    assert result.indexed_evidence == 0
    assert result.failed_evidence == 0
    assert result.is_successful


def test_indexing_request_defaults():
    """Create an indexing request with default settings."""
    request = IndexingRequest()

    assert request.evidence_ids == []
    assert request.batch_size == 100
    assert request.overwrite is False


def test_indexing_request_custom_values():
    """Create an indexing request with custom settings."""
    request = IndexingRequest(
        evidence_ids=["evidence-1", "evidence-2"],
        batch_size=50,
        overwrite=True,
    )

    assert request.evidence_ids == ["evidence-1", "evidence-2"]
    assert request.batch_size == 50
    assert request.overwrite is True


def test_indexing_request_rejects_invalid_batch_size():
    """Reject a batch size below one."""
    with pytest.raises(ValidationError):
        IndexingRequest(batch_size=0)


def test_evidence_indexer_is_abstract():
    """Prevent direct instantiation of the abstract indexer."""
    with pytest.raises(TypeError, match="abstract"):
        EvidenceIndexer()
