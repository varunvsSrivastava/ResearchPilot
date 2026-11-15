
"""Tests for the tool registry."""

import pytest
from pydantic import BaseModel

from researchpilot.tools.base import ResearchTool
from researchpilot.tools.registry import ToolRegistry


class SampleInput(BaseModel):
    query: str


class SampleTool(ResearchTool):
    name = "sample_tool"
    description = "Sample tool."
    args_schema = SampleInput

    def execute(self, arguments: SampleInput) -> str:
        return arguments.query


@pytest.fixture
def registry():
    tool_registry = ToolRegistry()
    tool_registry.register(SampleTool())
    return tool_registry


def test_register_tool(registry):
    assert len(registry) == 1


def test_get_registered_tool(registry):
    tool = registry.get("sample_tool")
    assert isinstance(tool, SampleTool)


def test_duplicate_registration_rejected(registry):
    with pytest.raises(ValueError, match="already registered"):
        registry.register(SampleTool())


def test_unknown_tool_rejected(registry):
    with pytest.raises(KeyError, match="Tool not found"):
        registry.get("unknown_tool")


def test_list_tools(registry):
    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "sample_tool"


def test_get_tool_definitions(registry):
    definitions = registry.get_definitions()
    assert len(definitions) == 1
    assert definitions[0]["name"] == "sample_tool"


def test_empty_registry():
    registry = ToolRegistry()
    assert len(registry) == 0
    assert registry.list_tools() == []
    assert registry.get_definitions() == []
