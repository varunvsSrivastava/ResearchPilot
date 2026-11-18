
"""Tests for end-to-end research workflow."""

from unittest.mock import MagicMock, patch

from researchpilot.agent.orchestrator import ResearchOrchestrator
from researchpilot.tools.registry import ToolRegistry


def build_dependencies():
    """Build mocked dependencies for the full research workflow."""

    retrieval_pipeline = MagicMock()
    retrieval_pipeline.execute.return_value = MagicMock(
        papers=["paper"],
        evidence=["evidence"],
        total_papers=1,
        successful_papers=1,
        failed_papers=0,
        total_evidence=1,
    )

    indexing_service = MagicMock()
    indexing_service.index_evidence.return_value = MagicMock(
        is_successful=True,
        indexed_evidence=1,
        total_evidence=1,
        failed_evidence=0,
    )

    # Semantic retrieval result must expose record.text.
    retrieved_result = MagicMock()
    retrieved_result.record.text = "one evidence"
    
    rag_pipeline = MagicMock()
    rag_pipeline.execute.return_value = MagicMock(
        retrieval_response=MagicMock(
            results=[retrieved_result],
        ),
        context="Evidence context",
    )

    report_pipeline = MagicMock()

    structured_report = MagicMock()
    structured_report.conclusion = "Research conclusion based on evidence"

    report_pipeline.run.return_value = MagicMock(
        report=structured_report,
    )

    return (
        retrieval_pipeline,
        indexing_service,
        rag_pipeline,
        report_pipeline,
    )


@patch("researchpilot.agent.orchestrator.generate_research_plan")
def test_execute_full_research_completes_workflow(mock_plan):
    """Verify the full research workflow completes successfully."""

    plan = MagicMock()
    plan.search_queries = []
    plan.sub_questions = ["Aspect one", "Aspect two"]
    mock_plan.return_value = plan

    (
        retrieval_pipeline,
        indexing_service,
        rag_pipeline,
        report_pipeline,
    ) = build_dependencies()

    orchestrator = ResearchOrchestrator(
        tool_registry=ToolRegistry(),
        retrieval_pipeline=retrieval_pipeline,
        indexing_service=indexing_service,
        rag_pipeline_service=rag_pipeline,
        report_pipeline=report_pipeline,
    )

    agent = orchestrator.execute_full_research("Research question")

    assert agent.state.status == "completed"
    assert agent.state.final_answer == "Research conclusion based on evidence"
    assert agent.research_result is report_pipeline.run.return_value
    assert (
        agent.research_result.report
        is report_pipeline.run.return_value.report
    )
    assert "ResearchReport(" not in agent.state.final_answer

    retrieval_pipeline.execute.assert_called_once_with(plan)
    indexing_service.index_evidence.assert_called_once_with(["evidence"])
    rag_pipeline.execute.assert_called_once()
    report_pipeline.run.assert_called_once()

    # Verify aspect coverage is passed to the report pipeline.
    report_kwargs = report_pipeline.run.call_args.kwargs

    assert report_kwargs["covered_aspects"] == ["Aspect one"]
    assert report_kwargs["all_aspects"] == ["Aspect one", "Aspect two"]


@patch("researchpilot.agent.orchestrator.generate_research_plan")
def test_execute_full_research_fails_on_indexing_failure(mock_plan):
    """Verify the workflow stops when evidence indexing fails."""

    plan = MagicMock()
    plan.search_queries = []
    plan.sub_questions = []
    mock_plan.return_value = plan

    (
        retrieval_pipeline,
        indexing_service,
        rag_pipeline,
        report_pipeline,
    ) = build_dependencies()

    indexing_service.index_evidence.return_value.is_successful = False

    orchestrator = ResearchOrchestrator(
        tool_registry=ToolRegistry(),
        retrieval_pipeline=retrieval_pipeline,
        indexing_service=indexing_service,
        rag_pipeline_service=rag_pipeline,
        report_pipeline=report_pipeline,
    )

    agent = orchestrator.execute_full_research("Research question")

    assert agent.state.status == "failed"
    assert "index" in agent.state.error.lower()

    rag_pipeline.execute.assert_not_called()
    report_pipeline.run.assert_not_called()
