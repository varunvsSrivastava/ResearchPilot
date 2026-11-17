
"""Tests for the ResearchPilot API."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from researchpilot.api.dependencies import get_research_orchestrator
from researchpilot.api.main import app
from researchpilot.evidence.coverage_models import (
    EvidenceCoverageAssessment,
    EvidenceCoverageScore,
)
from researchpilot.evidence.contradiction_models import ContradictionAssessment
from researchpilot.evidence.synthesis_models import ResearchSynthesis
from researchpilot.evidence.models import (
    Evidence,
    EvidenceQualityScore,
    EvaluatedEvidence,
)
from researchpilot.evidence.source_models import (
    EvaluatedSource,
    SourceCredibilityScore,
)
from researchpilot.generation.pipeline_models import ResearchPipelineResult
from researchpilot.generation.report_models import ResearchReport, ReportSection
from researchpilot.generation.citation_models import CitationValidationResult

@pytest.fixture
def api_client():
    """Provide a test client with a mocked research orchestrator."""

    orchestrator = Mock()

    app.dependency_overrides[get_research_orchestrator] = (
        lambda: orchestrator
    )

    with TestClient(app) as client:
        yield client, orchestrator

    app.dependency_overrides.pop(get_research_orchestrator, None)


def create_mock_agent(
    research_question: str,
    status: str = "completed",
    final_answer: str | None = "Research completed successfully.",
    error: str | None = None,
):
    """Create a lightweight mock research agent."""

    state = SimpleNamespace(
        research_question=research_question,
        status=status,
        research_plan=None,
        steps=[],
        final_answer=final_answer,
        error=error,
    )

    return SimpleNamespace(
        state=state,
        research_result=None,
    )


def test_health_endpoint(api_client):
    """Verify that the health endpoint returns a healthy status."""

    client, _ = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_successful_research_execution(api_client):
    """Verify successful research execution."""

    client, orchestrator = api_client

    question = "How does retrieval augmented generation work?"

    agent = create_mock_agent(
        research_question=question,
    )

    orchestrator.execute_full_research.return_value = agent

    response = client.post(
        "/api/v1/research",
        json={"research_question": question},
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["research_question"] == question
    assert response_data["status"] == "completed"
    assert response_data["final_answer"] == (
        "Research completed successfully."
    )
    assert response_data["research_result"] is None
    assert response_data["error"] is None

    orchestrator.execute_full_research.assert_called_once_with(question)


def test_failed_research_execution(api_client):
    """Verify that failed research execution returns error details."""

    client, orchestrator = api_client

    question = "Explain multi-agent systems."

    agent = create_mock_agent(
        research_question=question,
        status="failed",
        final_answer=None,
        error="Academic retrieval failed.",
    )

    orchestrator.execute_full_research.return_value = agent

    response = client.post(
        "/api/v1/research",
        json={"research_question": question},
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["status"] == "failed"
    assert response_data["final_answer"] is None
    assert response_data["error"] == "Academic retrieval failed."


def test_invalid_research_question(api_client):
    """Verify validation of research questions."""

    client, orchestrator = api_client

    response = client.post(
        "/api/v1/research",
        json={"research_question": "AI"},
    )

    assert response.status_code == 422

    orchestrator.execute_full_research.assert_not_called()


def test_structured_research_result_serialization(api_client):
    """Verify structured pipeline results serialize into API JSON."""

    client, orchestrator = api_client

    question = "How does retrieval augmented generation work?"

    evidence = Evidence(
        content="RAG combines retrieval with language generation.",
        source_id="paper-001",
        metadata={"title": "RAG Research"},
        retrieval_score=0.95,
    )

    evaluated_evidence = EvaluatedEvidence(
        evidence=evidence,
        quality=EvidenceQualityScore(
            relevance=0.95,
            completeness=0.90,
            clarity=0.92,
            overall_score=0.92,
        ),
    )

    evaluated_source = EvaluatedSource(
        paper_id="paper-001",
        title="RAG Research",
        provider="OpenAlex",
        credibility=SourceCredibilityScore(
            doi_score=0.9,
            citation_score=0.8,
            recency_score=0.85,
            metadata_score=0.95,
            overall_score=0.875,
        ),
    )

    coverage = EvidenceCoverageAssessment(
        research_question=question,
        evidence_count=1,
        coverage=EvidenceCoverageScore(
            coverage_score=0.8,
            covered_aspects=["Retrieval"],
            uncovered_aspects=["Evaluation"],
            total_aspects=2,
            coverage_ratio=0.5,
        ),
    )

    contradictions = ContradictionAssessment(
        evidence_count=1,
        contradiction_count=0,
    )

    synthesis = ResearchSynthesis(
        research_question=question,
        findings=[],
        evidence_count=1,
        synthesis_summary="RAG combines retrieval and generation.",
    )

    report = ResearchReport(
        title="RAG Research Report",
        research_question=question,
        sections=[
            ReportSection(
                title="Introduction",
                content="RAG retrieves relevant evidence.",
                citations=[1],
            )
        ],
        conclusion="RAG improves evidence grounding.",
        metadata={"source_count": "1"},
    )

    citation_validation = CitationValidationResult(
        is_valid=True,
        cited_sources={1},
        invalid_sources=set(),
        missing_citations=[],
        unused_sources=set(),
    )

    pipeline_result = ResearchPipelineResult(
        evaluated_evidence=[evaluated_evidence],
        evaluated_sources=[evaluated_source],
        coverage=coverage,
        contradictions=contradictions,
        synthesis=synthesis,
        report=report,
        citation_validation=citation_validation,
    )

    agent = create_mock_agent(question)
    agent.research_result = pipeline_result

    orchestrator.execute_full_research.return_value = agent

    response = client.post(
        "/api/v1/research",
        json={"research_question": question},
    )

    assert response.status_code == 200

    response_data = response.json()
    result = response_data["research_result"]

    assert len(result["evaluated_evidence"]) == 1
    assert result["evaluated_evidence"][0]["evidence"]["source_id"] == (
        "paper-001"
    )

    assert len(result["evaluated_sources"]) == 1
    assert result["evaluated_sources"][0]["provider"] == "OpenAlex"

    assert result["coverage"]["evidence_count"] == 1
    assert result["contradictions"]["contradiction_count"] == 0

    assert result["synthesis"]["evidence_count"] == 1
    assert result["report"]["title"] == "RAG Research Report"

    assert result["citation_validation"]["is_valid"] is True
    assert result["citation_validation"]["cited_sources"] == [1]
    assert result["citation_validation"]["invalid_sources"] == []