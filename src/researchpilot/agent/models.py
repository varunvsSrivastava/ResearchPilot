
"""Models for research agent execution."""

from typing import Literal

from pydantic import BaseModel, Field
from researchpilot.schemas.research_plan import ResearchPlan

AgentStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
]


class AgentStep(BaseModel):
    """Represent one step in the agent execution history."""

    step_number: int = Field(ge=1)
    action: str = Field(min_length=1)
    tool_name: str | None = None
    tool_input: dict = Field(default_factory=dict)
    tool_output: str | None = None
    status: Literal["pending", "running", "completed", "failed"] = (
        "pending"
    )
    error: str | None = None


class ResearchAgentState(BaseModel):
    """Track the state of a research agent execution."""

    research_plan: ResearchPlan | None = None
    research_question: str = Field(min_length=3)
    status: AgentStatus = "pending"
    steps: list[AgentStep] = Field(default_factory=list)
    final_answer: str | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        """Return whether the agent execution has terminated."""
        return self.status in {"completed", "failed"}

    @property
    def step_count(self) -> int:
        """Return the number of recorded execution steps."""
        return len(self.steps)
