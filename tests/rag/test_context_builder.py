
"""Tests for RAG context construction."""

from researchpilot.rag.pipeline.context import ContextBuilder
from researchpilot.rag.vectorstore.models import (
    VectorRecord,
    VectorSearchResult,
)


def create_result(
    evidence_id: str,
    text: str,
    metadata: dict | None = None,
    score: float = 0.9,
) -> VectorSearchResult:
    """Create a vector search result for testing."""
    return VectorSearchResult(
        record=VectorRecord(
            id=evidence_id,
            text=text,
            vector=[0.1, 0.2],
            metadata=metadata or {},
        ),
        score=score,
    )


def test_context_preserves_retrieval_order():
    """Context should preserve the order of retrieved evidence."""
    builder = ContextBuilder()

    results = [
        create_result("evidence-1", "First evidence."),
        create_result("evidence-2", "Second evidence."),
    ]

    context = builder.build(results)

    assert context.index("[Source 1]") < context.index("[Source 2]")
    assert context.index("First evidence.") < context.index(
        "Second evidence."
    )


def test_context_skips_blank_evidence():
    """Whitespace-only evidence should not be included."""
    builder = ContextBuilder()

    context = builder.build(
        [
            create_result("evidence-1", "   "),
            create_result("evidence-2", "Valid evidence."),
        ]
    )

    assert "[Source 1]" in context
    assert "Valid evidence." in context
    assert "evidence-1" not in context


def test_context_deduplicates_evidence_ids():
    """Repeated evidence IDs should only appear once."""
    builder = ContextBuilder()

    context = builder.build(
        [
            create_result("evidence-1", "First evidence."),
            create_result("evidence-1", "Duplicate evidence."),
            create_result("evidence-2", "Second evidence."),
        ]
    )

    assert context.count("Evidence ID: evidence-1") == 1
    assert "Duplicate evidence." not in context
    assert "[Source 2]" in context


def test_context_omits_missing_metadata():
    """Unavailable metadata should not produce empty metadata fields."""
    builder = ContextBuilder()

    context = builder.build(
        [create_result("evidence-1", "Evidence text.")]
    )

    assert "Evidence ID: evidence-1" in context
    assert "Paper ID:" not in context
    assert "Title:" not in context
    assert "DOI:" not in context
    assert "Page:" not in context


def test_context_handles_empty_results():
    """No results should produce an empty context."""
    builder = ContextBuilder()

    assert builder.build([]) == ""


def test_duplicate_id_keeps_first_occurrence():
    """The first occurrence of a duplicate evidence ID should win."""
    builder = ContextBuilder()

    context = builder.build(
        [
            create_result("evidence-1", "Original evidence."),
            create_result("evidence-2", "Other evidence."),
            create_result("evidence-1", "Repeated evidence."),
        ]
    )

    assert context.count("Evidence ID: evidence-1") == 1
    assert "Original evidence." in context
    assert "Repeated evidence." not in context
    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "[Source 3]" not in context


def test_identical_text_with_different_ids_is_preserved():
    """Identical text from distinct evidence IDs should be retained."""
    builder = ContextBuilder()

    context = builder.build(
        [
            create_result("evidence-1", "Shared evidence."),
            create_result("evidence-2", "Shared evidence."),
        ]
    )

    assert context.count("Shared evidence.") == 2
    assert context.count("Evidence ID:") == 2


def test_context_respects_character_limit():
    """Context should not exceed the configured character limit."""
    builder = ContextBuilder(max_context_chars=100)

    context = builder.build(
        [
            create_result("evidence-1", "A" * 30),
            create_result("evidence-2", "B" * 30),
        ]
    )

    assert len(context) <= 100
    assert "[Source 1]" in context


def test_context_skips_section_that_exceeds_limit():
    """An oversized first section should produce empty context."""
    builder = ContextBuilder(max_context_chars=20)

    context = builder.build(
        [create_result("evidence-1", "Evidence text.")]
    )

    assert context == ""


def test_context_includes_complete_sections_only():
    """Evidence text should not be truncated to fit the limit."""
    builder = ContextBuilder(max_context_chars=100)

    first = create_result("evidence-1", "First evidence.")
    second = create_result("evidence-2", "B" * 100)

    context = builder.build([first, second])

    assert "First evidence." in context
    assert "B" * 100 not in context


def test_context_builder_rejects_invalid_limit():
    """The maximum context size must be positive."""
    import pytest

    with pytest.raises(ValueError, match="must be positive"):
        ContextBuilder(max_context_chars=0)
