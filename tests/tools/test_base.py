
"""Tests for the research tool interface."""

import pytest
from pydantic import BaseModel

from researchpilot.tools.base import ResearchTool


class SampleToolInput(BaseModel):
    query: str


class SampleTool(ResearchTool):
    name = "sample_tool"
    description = "A sample research tool."
    args_schema = SampleToolInput

    def execute(self, arguments: SampleToolInput) -> str:
        return f"Executed: {arguments.query}"


def test_tool_definition():
    tool = SampleTool()

    definition = tool.get_definition()

    assert definition["name"] == "sample_tool"
    assert definition["description"] == "A sample research tool."
    assert definition["parameters"]["properties"]["query"]["type"] == "string"


def test_tool_execution():
    tool = SampleTool()
    arguments = SampleToolInput(query="AI research")

    result = tool.execute(arguments)

    assert result == "Executed: AI research"


def test_tool_interface_is_abstract():
    with pytest.raises(TypeError):
        ResearchTool()
