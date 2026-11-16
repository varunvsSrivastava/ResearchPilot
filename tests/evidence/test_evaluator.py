
"""Tests for evidence evaluation service."""

from researchpilot.evidence.evaluator import EvidenceEvaluator
from researchpilot.rag.vectorstore.models import (
    VectorRecord,
    VectorSearchResult,
)


def test_evaluate_converts_vector_results_to_evidence() -> None:
    evaluator = EvidenceEvaluator()

    result = VectorSearchResult(
        record=VectorRecord(
            id="doc-1",
            text="Machine learning improves prediction accuracy.",
            vector=[0.1, 0.2],
            metadata={"source": "paper.pdf", "page": 1},
        ),
        score=0.85,
    )

    evaluated = evaluator.evaluate(
        results=[result],
        research_question="machine learning prediction",
    )

    assert len(evaluated) == 1
    assert evaluated[0].evidence.source_id == "doc-1"
    assert evaluated[0].evidence.content == (
        "Machine learning improves prediction accuracy."
    )
    assert evaluated[0].evidence.retrieval_score == 0.85


def test_evaluate_preserves_metadata() -> None:
    evaluator = EvidenceEvaluator()

    result = VectorSearchResult(
        record=VectorRecord(
            id="doc-2",
            text="Employee attrition affects organizations.",
            vector=[0.1, 0.2],
            metadata={"source": "report.pdf", "year": 2025},
        ),
        score=0.75,
    )

    evaluated = evaluator.evaluate(
        results=[result],
        research_question="employee attrition",
    )

    assert evaluated[0].evidence.metadata == {
        "source": "report.pdf",
        "year": 2025,
    }


def test_evaluate_computes_quality_score() -> None:
    evaluator = EvidenceEvaluator()

    result = VectorSearchResult(
        record=VectorRecord(
            id="doc-3",
            text="Machine learning improves prediction accuracy.",
            vector=[0.1, 0.2],
        ),
        score=0.9,
    )

    evaluated = evaluator.evaluate(
        results=[result],
        research_question="machine learning prediction",
    )

    quality = evaluated[0].quality

    assert quality.relevance == 1.0
    assert 0.0 <= quality.completeness <= 1.0
    assert 0.0 <= quality.clarity <= 1.0
    assert 0.0 <= quality.overall_score <= 1.0


def test_evaluate_returns_empty_list_for_empty_results() -> None:
    evaluator = EvidenceEvaluator()

    evaluated = evaluator.evaluate(
        results=[],
        research_question="machine learning",
    )

    assert evaluated == []
