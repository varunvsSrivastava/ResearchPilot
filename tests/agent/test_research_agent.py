
"""Tests for research agent execution controller."""

import pytest

from researchpilot.agent.models import AgentStep
from researchpilot.agent.research_agent import (
    AgentExecutionLimitError,
    ResearchAgent,
)


def test_agent_starts_execution():
    agent = ResearchAgent("How does RAG improve research?")

    state = agent.start()

    assert state.status == "running"


def test_agent_records_execution_step():
    agent = ResearchAgent("How does RAG improve research?")
    agent.start()

    step = agent.record_step(
        action="execute_tool",
        tool_name="academic_search",
        tool_input={"query": "RAG"},
        tool_output="Found 5 papers.",
    )

    assert isinstance(step, AgentStep)
    assert step.step_number == 1
    assert agent.state.step_count == 1


def test_agent_numbers_steps_sequentially():
    agent = ResearchAgent("How does RAG improve research?")
    agent.start()

    first = agent.record_step(action="search")
    second = agent.record_step(action="retrieve")

    assert first.step_number == 1
    assert second.step_number == 2


def test_agent_completes_with_final_answer():
    agent = ResearchAgent("How does RAG improve research?")
    agent.start()

    state = agent.complete("RAG improves research through grounding.")

    assert state.status == "completed"
    assert state.final_answer is not None
    assert state.is_terminal


def test_agent_fails_with_error():
    agent = ResearchAgent("How does RAG improve research?")
    agent.start()

    state = agent.fail("Tool execution failed.")

    assert state.status == "failed"
    assert state.error == "Tool execution failed."
    assert state.is_terminal


def test_agent_enforces_maximum_steps():
    agent = ResearchAgent(
        "How does RAG improve research?",
        max_steps=1,
    )
    agent.start()
    agent.record_step(action="search")

    with pytest.raises(AgentExecutionLimitError):
        agent.record_step(action="retrieve")


def test_agent_cannot_record_steps_before_start():
    agent = ResearchAgent("How does RAG improve research?")

    with pytest.raises(RuntimeError):
        agent.record_step(action="search")


def test_agent_cannot_start_twice():
    agent = ResearchAgent("How does RAG improve research?")
    agent.start()

    with pytest.raises(RuntimeError):
        agent.start()


def test_agent_rejects_empty_final_answer():
    agent = ResearchAgent("How does RAG improve research?")
    agent.start()

    with pytest.raises(ValueError):
        agent.complete("   ")


def test_agent_rejects_empty_failure_message():
    agent = ResearchAgent("How does RAG improve research?")
    agent.start()

    with pytest.raises(ValueError):
        agent.fail("   ")


def test_agent_rejects_zero_max_steps():
    with pytest.raises(ValueError):
        ResearchAgent(
            "How does RAG improve research?",
            max_steps=0,
        )
