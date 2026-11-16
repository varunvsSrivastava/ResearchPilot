
"""Integration tests for the research report pipeline."""

from researchpilot.evidence.coverage_models import (
    EvidenceCoverageAssessment,
)
from researchpilot.evidence.contradiction_models import (
    ContradictionAssessment,
)
from researchpilot.evidence.synthesis_models import ResearchSynthesis
from researchpilot.generation import (
    ResearchReportPipeline,
    ResearchPipelineResult,
)
from researchpilot.rag.vectorstore.models import (
    VectorRecord,
    VectorSearchResult,
)
from researchpilot.retrieval.models import Paper


def make_retrieved_results():
    """Create representative vector search results."""

    return [
        VectorSearchResult(
            record=VectorRecord(
                id="1",
                text="AI improves diagnostic accuracy in healthcare.",
                vector=[0.1, 0.2],
                metadata={"title": "Healthcare AI Study"},
            ),
            score=0.95,
        ),
        VectorSearchResult(
            record=VectorRecord(
                id="2",
                text="AI supports clinical decision-making.",
                vector=[0.2, 0.3],
                metadata={"title": "Clinical AI Study"},
            ),
            score=0.90,
        ),
    ]


def make_papers():
    """Create representative academic papers."""

    return [
        Paper(
            paper_id="paper-1",
            provider="test-provider",
            title="Artificial Intelligence in Healthcare",
            abstract="AI applications in healthcare.",
            authors=["Author A"],
            publication_year=2025,
            doi="10.1234/example",
            source_url="https://example.com/paper-1",
            citation_count=25,
            is_open_access=True,
            search_query="AI healthcare",
        )
    ]


def make_context():
    """Create citation-aware retrieved context."""

    return (
        "[Source 1] AI improves diagnostic accuracy in healthcare.\n"
        "[Source 2] AI supports clinical decision-making."
    )


def test_pipeline_returns_complete_result():
    pipeline = ResearchReportPipeline()

    result = pipeline.run(
        research_question="How does AI improve healthcare?",
        retrieved_results=make_retrieved_results(),
        papers=make_papers(),
        context=make_context(),
        covered_aspects=["diagnostics"],
        all_aspects=["diagnostics", "clinical decision-making"],
        title="AI Healthcare Research",
    )

    assert isinstance(result, ResearchPipelineResult)
    assert len(result.evaluated_evidence) == 2
    assert len(result.evaluated_sources) == 1

    assert isinstance(result.coverage, EvidenceCoverageAssessment)
    assert result.coverage.evidence_count == 2

    assert isinstance(result.contradictions, ContradictionAssessment)
    assert result.contradictions.evidence_count == 2

    assert isinstance(result.synthesis, ResearchSynthesis)
    assert result.synthesis.evidence_count == 2

    assert result.report.title == "AI Healthcare Research"
    assert result.report.research_question == (
        "How does AI improve healthcare?"
    )

    assert result.citation_validation.is_valid
    assert not result.citation_validation.invalid_sources
    assert not result.citation_validation.missing_citations


def test_pipeline_handles_empty_retrieved_evidence():
    pipeline = ResearchReportPipeline()

    result = pipeline.run(
        research_question="Research question",
        retrieved_results=[],
        papers=[],
        context="",
        covered_aspects=[],
        all_aspects=[],
    )

    assert result.evaluated_evidence == []
    assert result.evaluated_sources == []
    assert result.synthesis.evidence_count == 0
    assert result.report.sections == []
    assert result.citation_validation.is_valid
