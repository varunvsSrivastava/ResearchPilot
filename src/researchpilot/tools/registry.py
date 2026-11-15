
"""Registry for managing ResearchPilot tools."""

from researchpilot.tools.base import ResearchTool


class ToolRegistry:
    """Register, retrieve, and discover research tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ResearchTool] = {}

    def register(self, tool: ResearchTool) -> None:
        """Register a tool by its unique name."""
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ResearchTool:
        """Retrieve a registered tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def list_tools(self) -> list[ResearchTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_definitions(self) -> list[dict]:
        """Return definitions for all registered tools."""
        return [tool.get_definition() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)
