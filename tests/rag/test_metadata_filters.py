
"""Tests for metadata filtering models."""

from researchpilot.rag.vectorstore.filters import MetadataFilter


def test_metadata_filter_defaults():
    metadata_filter = MetadataFilter()

    assert metadata_filter.conditions == {}
    assert metadata_filter.is_empty


def test_metadata_filter_matches_all_conditions():
    metadata_filter = MetadataFilter(
        conditions={
            "paper_id": "paper-1",
            "page_number": 5,
        }
    )

    metadata = {
        "paper_id": "paper-1",
        "page_number": 5,
        "chunk_index": 2,
    }

    assert metadata_filter.matches(metadata)


def test_metadata_filter_rejects_missing_metadata():
    metadata_filter = MetadataFilter(
        conditions={"paper_id": "paper-1"}
    )

    assert not metadata_filter.matches({})


def test_metadata_filter_rejects_mismatched_metadata():
    metadata_filter = MetadataFilter(
        conditions={"paper_id": "paper-1"}
    )

    metadata = {"paper_id": "paper-2"}

    assert not metadata_filter.matches(metadata)


def test_metadata_filter_matches_boolean_values():
    metadata_filter = MetadataFilter(
        conditions={"is_open_access": True}
    )

    assert metadata_filter.matches({"is_open_access": True})
    assert not metadata_filter.matches({"is_open_access": False})
