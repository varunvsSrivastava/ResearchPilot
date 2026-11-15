
"""Tests for the FAISS vector store."""

import numpy as np
import pytest

from researchpilot.rag.vectorstore import (
    FAISSVectorStore,
    VectorRecord,
)


@pytest.fixture
def store():
    """Create an empty FAISS vector store."""
    return FAISSVectorStore(dimensions=3)


@pytest.fixture
def sample_records():
    """Create sample vector records."""
    return [
        VectorRecord(
            id="record-1",
            text="Machine learning research",
            vector=[1.0, 0.0, 0.0],
            metadata={"source": "paper-1"},
        ),
        VectorRecord(
            id="record-2",
            text="Deep learning research",
            vector=[0.0, 1.0, 0.0],
            metadata={"source": "paper-2"},
        ),
        VectorRecord(
            id="record-3",
            text="Neural network research",
            vector=[0.8, 0.2, 0.0],
            metadata={"source": "paper-3"},
        ),
    ]


def test_store_initialization(store):
    """Initialize an empty FAISS index."""
    assert store.dimensions == 3
    assert store.index.ntotal == 0
    assert store.records == {}
    assert store.record_ids == []


def test_store_rejects_invalid_dimensions():
    """Reject non-positive dimensions."""
    with pytest.raises(ValueError, match="positive"):
        FAISSVectorStore(dimensions=0)


def test_add_records(store, sample_records):
    """Add records to the vector store."""
    store.add_records(sample_records)

    assert store.index.ntotal == 3
    assert len(store.records) == 3
    assert store.record_ids == [
        "record-1",
        "record-2",
        "record-3",
    ]


def test_add_empty_records(store):
    """Adding an empty batch should be a no-op."""
    store.add_records([])

    assert store.index.ntotal == 0


def test_add_records_rejects_duplicate_batch_ids(store):
    """Reject duplicate IDs within one batch."""
    record = VectorRecord(
        id="record-1",
        text="Evidence",
        vector=[1.0, 0.0, 0.0],
    )

    with pytest.raises(ValueError, match="Duplicate"):
        store.add_records([record, record])


def test_add_records_rejects_existing_id(store, sample_records):
    """Reject IDs that already exist in the store."""
    store.add_records(sample_records[:1])

    with pytest.raises(ValueError, match="already exists"):
        store.add_records(sample_records[:1])


def test_add_records_rejects_dimension_mismatch(store):
    """Reject vectors with incompatible dimensions."""
    record = VectorRecord(
        id="record-1",
        text="Evidence",
        vector=[1.0, 0.0],
    )

    with pytest.raises(ValueError, match="dimensions"):
        store.add_records([record])


def test_search_returns_similar_records(store, sample_records):
    """Retrieve records using cosine similarity."""
    store.add_records(sample_records)

    response = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert response.is_valid
    assert response.total_results == 2
    assert response.results[0].record.id == "record-1"
    assert response.results[0].score == pytest.approx(1.0)
    assert response.results[1].record.id == "record-3"


def test_search_empty_store(store):
    """Return empty results for an empty index."""
    response = store.search([1.0, 0.0, 0.0])

    assert response.is_valid
    assert response.total_results == 0


def test_search_rejects_invalid_top_k(store):
    """Reject non-positive top_k."""
    with pytest.raises(ValueError, match="positive"):
        store.search([1.0, 0.0, 0.0], top_k=0)


def test_search_rejects_dimension_mismatch(store):
    """Reject query vectors with incompatible dimensions."""
    with pytest.raises(ValueError, match="dimensions"):
        store.search([1.0, 0.0])


def test_delete_records(store, sample_records):
    """Delete records and rebuild the index."""
    store.add_records(sample_records)

    store.delete_records(["record-2"])

    assert store.index.ntotal == 2
    assert "record-2" not in store.records
    assert store.record_ids == ["record-1", "record-3"]


def test_delete_empty_ids(store, sample_records):
    """Deleting an empty ID list is a no-op."""
    store.add_records(sample_records)

    store.delete_records([])

    assert store.index.ntotal == 3


def test_delete_unknown_record(store):
    """Reject unknown record IDs."""
    with pytest.raises(ValueError, match="Unknown"):
        store.delete_records(["missing-record"])


def test_save_and_load(store, sample_records, tmp_path):
    """Persist and restore the vector store."""
    store.add_records(sample_records)

    store.save(str(tmp_path))

    restored = FAISSVectorStore(dimensions=3)
    restored.load(str(tmp_path))

    assert restored.index.ntotal == 3
    assert restored.record_ids == store.record_ids
    assert restored.records == store.records

    response = restored.search([1.0, 0.0, 0.0], top_k=1)

    assert response.results[0].record.id == "record-1"


def test_load_missing_files(store, tmp_path):
    """Reject loading when persistence files are missing."""
    with pytest.raises(FileNotFoundError):
        store.load(str(tmp_path))


def test_search_top_k_larger_than_index(store, sample_records):
    """Return available records when top_k exceeds index size."""
    store.add_records(sample_records)

    response = store.search([1.0, 0.0, 0.0], top_k=10)

    assert response.total_results == 3


def test_search_filters_records_by_metadata(store, sample_records):
    from researchpilot.rag.vectorstore.filters import MetadataFilter

    store.add_records(sample_records)

    metadata_filter = MetadataFilter(
        conditions={"source": "paper-1"}
    )

    response = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=5,
        metadata_filter=metadata_filter,
    )

    assert response.total_results == 1
    assert response.results[0].record.id == "record-1"


def test_search_returns_empty_when_metadata_does_not_match(
    store,
    sample_records,
):
    from researchpilot.rag.vectorstore.filters import MetadataFilter

    store.add_records(sample_records)

    metadata_filter = MetadataFilter(
        conditions={"source": "missing-paper"}
    )

    response = store.search(
        query_vector=[1.0, 0.0, 0.0],
        metadata_filter=metadata_filter,
    )

    assert response.results == []
    assert response.total_results == 0


def test_search_applies_top_k_after_metadata_filter(
    store,
    sample_records,
):
    from researchpilot.rag.vectorstore.filters import MetadataFilter

    store.add_records(sample_records)

    metadata_filter = MetadataFilter(
        conditions={"source": "paper-1"}
    )

    response = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=1,
        metadata_filter=metadata_filter,
    )

    assert response.total_results == 1
    assert response.results[0].record.id == "record-1"


def test_search_supports_multiple_metadata_conditions(
    store,
    sample_records,
):
    from researchpilot.rag.vectorstore.filters import MetadataFilter

    store.add_records(sample_records)

    metadata_filter = MetadataFilter(
        conditions={
            "source": "paper-1",
            "missing_field": "value",
        }
    )

    response = store.search(
        query_vector=[1.0, 0.0, 0.0],
        metadata_filter=metadata_filter,
    )

    assert response.total_results == 0
