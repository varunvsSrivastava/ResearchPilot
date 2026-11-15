
"""Tests for research agent execution models."""

import pytest
from pydantic import ValidationError
import pytest

from researchpilot.agent.research_agent import ResearchAgent

from researchpilot.agent.models import (
    AgentStep,
    ResearchAgentState,
)


def test_agent_state_initializes_with_pending_status():
    state = ResearchAgentState(
        research_question="How does RAG improve research?"
    )

    assert state.status == "pending"
    assert state.steps == []
    assert state.final_answer is None
    assert state.error is None


def test_agent_state_is_not_terminal_when_pending():
    state = ResearchAgentState(
        research_question="How does RAG improve research?"
    )

    assert state.is_terminal is False


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_agent_state_is_terminal_for_final_statuses(status):
    state = ResearchAgentState(
        research_question="How does RAG improve research?",
        status=status,
    )

    assert state.is_terminal is True


def test_agent_step_count_tracks_execution_history():
    state = ResearchAgentState(
        research_question="How does RAG improve research?",
        steps=[
            AgentStep(step_number=1, action="search"),
            AgentStep(step_number=2, action="retrieve"),
        ],
    )

    assert state.step_count == 2


def test_agent_step_records_tool_execution():
    step = AgentStep(
        step_number=1,
        action="execute_tool",
        tool_name="academic_search",
        tool_input={"query": "retrieval augmented generation"},
        tool_output="Found 5 papers.",
        status="completed",
    )

    assert step.tool_name == "academic_search"
    assert step.status == "completed"
    assert step.tool_output == "Found 5 papers."


def test_agent_state_rejects_short_research_question():
    with pytest.raises(ValidationError):
        ResearchAgentState(research_question="AI")


def test_agent_step_rejects_invalid_step_number():
    with pytest.raises(ValidationError):
        AgentStep(step_number=0, action="search")

def test_agent_cannot_start_twice():
    agent = ResearchAgent("Research question")
    agent.start()

    with pytest.raises(RuntimeError, match="already started"):
        agent.start()


def test_agent_cannot_complete_twice():
    agent = ResearchAgent("Research question")
    agent.start()
    agent.complete("Final answer")

    with pytest.raises(RuntimeError, match="must be running"):
        agent.complete("Another answer")


def test_agent_cannot_fail_after_completion():
    agent = ResearchAgent("Research question")
    agent.start()
    agent.complete("Final answer")

    with pytest.raises(RuntimeError, match="must be running"):
        agent.fail("Unexpected error")


def test_agent_cannot_complete_with_empty_answer():
    agent = ResearchAgent("Research question")
    agent.start()

    with pytest.raises(ValueError, match="Final answer cannot be empty"):
        agent.complete("   ")