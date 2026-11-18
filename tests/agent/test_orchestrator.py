
"""Tests for research orchestration."""

import pytest
from unittest.mock import MagicMock, patch
from researchpilot.agent.research_agent import AgentExecutionLimitError
from researchpilot.agent.orchestrator import ResearchOrchestrator
from researchpilot.agent.research_agent import ResearchAgent
from researchpilot.tools.registry import ToolRegistry


def test_orchestrator_initialization():
    registry = ToolRegistry()
    orchestrator = ResearchOrchestrator(registry)

    assert orchestrator.tool_registry is registry
    assert orchestrator.max_steps == 10


def test_custom_max_steps():
    orchestrator = ResearchOrchestrator(
        ToolRegistry(),
        max_steps=5,
    )

    assert orchestrator.max_steps == 5


def test_create_agent():
    orchestrator = ResearchOrchestrator(ToolRegistry())

    agent = orchestrator.create_agent("Research question")

    assert isinstance(agent, ResearchAgent)
    assert agent.state.research_question == "Research question"


def test_agent_max_steps_forwarded():
    orchestrator = ResearchOrchestrator(
        ToolRegistry(),
        max_steps=4,
    )

    agent = orchestrator.create_agent("Research question")

    assert agent.max_steps == 4


@patch(
    "researchpilot.agent.orchestrator.generate_research_plan"
)
def test_plan_research(mock_generate_plan):
    expected_plan = MagicMock()
    mock_generate_plan.return_value = expected_plan

    orchestrator = ResearchOrchestrator(ToolRegistry())
    result = orchestrator.plan_research("Research question")

    mock_generate_plan.assert_called_once_with("Research question")
    assert result is expected_plan


@patch("researchpilot.agent.orchestrator.generate_research_plan")
def test_execute_research_success(mock_generate_plan):
    plan = MagicMock()
    plan.search_queries = []
    mock_generate_plan.return_value = plan

    registry = ToolRegistry()
    rag_tool = MagicMock()
    rag_tool.name = "rag_retrieval"
    rag_tool.args_schema.model_validate.return_value = MagicMock()
    rag_tool.execute.return_value = ["evidence"]
    registry.register(rag_tool)

    orchestrator = ResearchOrchestrator(registry)
    agent = orchestrator.execute_research("Research question")

    assert agent.state.status == "running"
    assert agent.state.step_count == 2
    assert agent.state.steps[0].action == "Generate research plan"


@patch("researchpilot.agent.orchestrator.generate_research_plan")
def test_execute_research_planning_failure(mock_generate_plan):
    mock_generate_plan.side_effect = RuntimeError("Planning failed")

    orchestrator = ResearchOrchestrator(ToolRegistry())
    agent = orchestrator.execute_research("Research question")

    assert agent.state.status == "failed"
    assert agent.state.error == "Planning failed"


def test_execute_tool_success():
    registry = ToolRegistry()
    tool = MagicMock()
    tool.name = "mock_tool"
    tool.args_schema.model_validate.return_value = MagicMock()
    tool.execute.return_value = "Research results"
    registry.register(tool)

    orchestrator = ResearchOrchestrator(registry)
    agent = orchestrator.create_agent("Research question")
    agent.start()

    result = orchestrator.execute_tool(
        agent,
        tool.name,
        {"query": "research query"},
    )

    assert result == "Research results"
    assert agent.state.step_count == 1
    assert agent.state.steps[0].status == "completed"


def test_execute_tool_invalid_arguments():
    registry = ToolRegistry()
    tool = MagicMock()
    tool.name = "mock_tool"
    tool.args_schema.model_validate.side_effect = ValueError("Invalid arguments")
    registry.register(tool)

    orchestrator = ResearchOrchestrator(registry)
    agent = orchestrator.create_agent("Research question")
    agent.start()

    with pytest.raises(ValueError, match="Invalid arguments"):
        orchestrator.execute_tool(agent, tool.name, {})

    assert agent.state.status == "failed"


def test_execute_tool_unknown_tool():
    orchestrator = ResearchOrchestrator(ToolRegistry())
    agent = orchestrator.create_agent("Research question")
    agent.start()

    with pytest.raises(KeyError):
        orchestrator.execute_tool(agent, "unknown_tool", {})

    assert agent.state.status == "failed"


def test_execute_tool_execution_failure():
    registry = ToolRegistry()
    tool = MagicMock()
    tool.name = "mock_tool"
    tool.args_schema.model_validate.return_value = MagicMock()
    tool.execute.side_effect = RuntimeError("Tool execution failed")
    registry.register(tool)

    orchestrator = ResearchOrchestrator(registry)
    agent = orchestrator.create_agent("Research question")
    agent.start()

    with pytest.raises(RuntimeError, match="Tool execution failed"):
        orchestrator.execute_tool(agent, tool.name, {})

    assert agent.state.status == "failed"
    assert agent.state.error == "Tool execution failed"


@patch("researchpilot.agent.orchestrator.generate_research_plan")
def test_execute_research_retains_plan(mock_generate_plan):
    plan = MagicMock()
    plan.search_queries = []
    mock_generate_plan.return_value = plan

    orchestrator = ResearchOrchestrator(ToolRegistry())
    agent = orchestrator.execute_research("Research question")

    assert agent.state.research_plan is plan


@patch("researchpilot.agent.orchestrator.generate_research_plan")
def test_execute_research_skips_academic_search_without_queries(
    mock_generate_plan,
):
    plan = MagicMock()
    plan.search_queries = []
    mock_generate_plan.return_value = plan

    registry = ToolRegistry()

    rag_tool = MagicMock()
    rag_tool.name = "rag_retrieval"
    rag_tool.args_schema.model_validate.return_value = MagicMock()
    rag_tool.execute.return_value = ["evidence"]
    registry.register(rag_tool)

    orchestrator = ResearchOrchestrator(registry)

    agent = orchestrator.execute_research("Research question")

    assert agent.state.status == "running"
    assert agent.state.step_count == 2


@patch("researchpilot.agent.orchestrator.generate_research_plan")
def test_execute_research_runs_academic_search(mock_generate_plan):
    plan = MagicMock()
    plan.search_queries = [
        MagicMock(query="query one", source_type="academic"),
        MagicMock(query="query two", source_type="academic"),
    ]
    mock_generate_plan.return_value = plan

    registry = ToolRegistry()
    tool = MagicMock()
    tool.name = "academic_search"
    tool.args_schema.model_validate.return_value = MagicMock()
    tool.execute.return_value = ["paper one", "paper two"]
    registry.register(tool)

    rag_tool = MagicMock()
    rag_tool.name = "rag_retrieval"
    rag_tool.args_schema.model_validate.return_value = MagicMock()
    rag_tool.execute.return_value = ["retrieved evidence"]
    registry.register(rag_tool)

    orchestrator = ResearchOrchestrator(registry)
    agent = orchestrator.execute_research("Research question")

    tool.args_schema.model_validate.assert_called_once_with(
        {"queries": ["query one", "query two"]}
    )
    tool.execute.assert_called_once()
    assert agent.state.status == "running"
    assert agent.state.step_count == 3
    assert agent.state.steps[-1].tool_name == "rag_retrieval"


@patch("researchpilot.agent.orchestrator.generate_research_plan")
def test_execute_research_runs_rag_retrieval(mock_generate_plan):
    plan = MagicMock()
    plan.search_queries = []
    mock_generate_plan.return_value = plan

    registry = ToolRegistry()
    rag_tool = MagicMock()
    rag_tool.name = "rag_retrieval"
    rag_tool.args_schema.model_validate.return_value = MagicMock()
    rag_tool.execute.return_value = ["evidence one", "evidence two"]
    registry.register(rag_tool)

    orchestrator = ResearchOrchestrator(registry)
    agent = orchestrator.execute_research("Research question")

    rag_tool.args_schema.model_validate.assert_called_once_with(
        {"query": "Research question"}
    )
    rag_tool.execute.assert_called_once()
    assert agent.state.status == "running"
    assert agent.state.step_count == 2
    assert agent.state.steps[-1].tool_name == "rag_retrieval"
    assert "evidence one" in agent.state.steps[-1].tool_output


@patch("researchpilot.agent.orchestrator.generate_research_plan")
def test_execute_research_fails_when_rag_retrieval_fails(mock_generate_plan):
    plan = MagicMock()
    plan.search_queries = []
    mock_generate_plan.return_value = plan

    registry = ToolRegistry()
    rag_tool = MagicMock()
    rag_tool.name = "rag_retrieval"
    rag_tool.args_schema.model_validate.return_value = MagicMock()
    rag_tool.execute.side_effect = RuntimeError("Retrieval failed")
    registry.register(rag_tool)

    orchestrator = ResearchOrchestrator(registry)
    agent = orchestrator.execute_research("Research question")

    assert agent.state.status == "failed"
    assert agent.state.error == "Retrieval failed"


def test_generate_answer_from_retrieval():
    registry = ToolRegistry()

    pipeline = MagicMock()
    pipeline.context_builder.build.return_value = "Evidence context"
    pipeline.answer_generation_service.generate.return_value = (
        "Grounded research answer"
    )

    orchestrator = ResearchOrchestrator(
        registry,
        rag_pipeline_service=pipeline,
    )

    retrieval_output = MagicMock()
    retrieval_output.results = ["evidence"]

    answer = orchestrator.generate_answer(
        "Research question",
        retrieval_output,
    )

    pipeline.context_builder.build.assert_called_once_with(
        ["evidence"]
    )
    pipeline.answer_generation_service.generate.assert_called_once_with(
        query="Research question",
        context="Evidence context",
    )
    assert answer == "Grounded research answer"


def test_generate_answer_without_pipeline_raises():
    orchestrator = ResearchOrchestrator(ToolRegistry())

    with pytest.raises(RuntimeError, match="RAG pipeline service is not configured"):
        orchestrator.generate_answer(
            "Research question",
            MagicMock(),
        )

def test_execute_research_completes_with_generated_answer():
    registry = ToolRegistry()

    retrieval_tool = MagicMock()
    retrieval_tool.name = "rag_retrieval"
    retrieval_tool.args_schema.model_validate.return_value = MagicMock()
    retrieval_tool.execute.return_value = MagicMock(results=["evidence"])

    registry.register(retrieval_tool)

    pipeline = MagicMock()
    pipeline.context_builder.build.return_value = "Evidence context"
    pipeline.answer_generation_service.generate.return_value = (
        "Grounded research answer"
    )

    orchestrator = ResearchOrchestrator(
        tool_registry=registry,
        rag_pipeline_service=pipeline,
    )

    with patch.object(
        orchestrator,
        "plan_research",
        return_value=MagicMock(search_queries=[]),
    ):
        agent = orchestrator.execute_research("Research question")

    assert agent.state.status == "completed"
    assert agent.state.final_answer == "Grounded research answer"

def test_execute_tool_rejects_terminal_agent():
    orchestrator = ResearchOrchestrator(ToolRegistry())
    agent = orchestrator.create_agent("Research question")
    agent.start()
    agent.complete("Final answer")

    with pytest.raises(
        RuntimeError,
        match="Cannot execute tools for a terminal agent",
    ):
        orchestrator.execute_tool(
            agent,
            "rag_retrieval",
            {"query": "Research question"},
        )

def test_execute_tool_respects_agent_step_limit():
    orchestrator = ResearchOrchestrator(
        ToolRegistry(),
        max_steps=1,
    )
    agent = orchestrator.create_agent("Research question")
    agent.start()

    agent.record_step(
        action="Initial step",
        status="completed",
    )

    with pytest.raises(AgentExecutionLimitError):
        agent.record_step(
            action="Exceed limit",
            status="completed",
        )

def test_execute_tool_rejects_pending_agent():
    registry = ToolRegistry()
    tool = MagicMock()
    tool.name = "mock_tool"
    registry.register(tool)

    orchestrator = ResearchOrchestrator(registry)
    agent = orchestrator.create_agent("Research question")

    with pytest.raises(
        RuntimeError,
        match="Agent must be running",
    ):
        orchestrator.execute_tool(agent, tool.name, {})

    tool.execute.assert_not_called()
    assert agent.state.status == "pending"
    assert agent.state.step_count == 0