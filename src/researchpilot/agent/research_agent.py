
"""Research agent execution controller."""

from researchpilot.agent.models import AgentStep, ResearchAgentState


class AgentExecutionLimitError(RuntimeError):
    """Raised when the agent exceeds its execution step limit."""


class ResearchAgent:
    """Manage research agent execution lifecycle and history."""

    def __init__(
        self,
        research_question: str,
        max_steps: int = 10,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        self.max_steps = max_steps
        self.state = ResearchAgentState(
            research_question=research_question
        )
        self.research_result = None

    def start(self) -> ResearchAgentState:
        """Start agent execution."""
        if self.state.status != "pending":
            raise RuntimeError("Agent execution has already started.")

        self.state.status = "running"
        return self.state

    def record_step(
        self,
        action: str,
        tool_name: str | None = None,
        tool_input: dict | None = None,
        tool_output: str | None = None,
        status: str = "completed",
        error: str | None = None,
    ) -> AgentStep:
        """Record one execution step."""
        if self.state.status != "running":
            raise RuntimeError("Agent must be running to record steps.")

        if self.state.step_count >= self.max_steps:
            raise AgentExecutionLimitError(
                f"Agent exceeded maximum steps: {self.max_steps}."
            )

        step = AgentStep(
            step_number=self.state.step_count + 1,
            action=action,
            tool_name=tool_name,
            tool_input=tool_input or {},
            tool_output=tool_output,
            status=status,
            error=error,
        )

        self.state.steps.append(step)
        return step

    def complete(self, final_answer: str) -> ResearchAgentState:
        """Mark execution as completed."""
        if self.state.status != "running":
            raise RuntimeError("Agent must be running to complete.")

        if not final_answer.strip():
            raise ValueError("Final answer cannot be empty.")

        self.state.final_answer = final_answer
        self.state.status = "completed"
        return self.state

    def fail(self, error: str) -> ResearchAgentState:
        """Mark execution as failed."""
        if self.state.status != "running":
            raise RuntimeError("Agent must be running to fail.")

        if not error.strip():
            raise ValueError("Error message cannot be empty.")

        self.state.error = error
        self.state.status = "failed"
        return self.state
