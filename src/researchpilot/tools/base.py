
"""Base interface for ResearchPilot tools."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ResearchTool(ABC):
    """Abstract interface for research tools."""

    name: str
    description: str
    args_schema: type[BaseModel]

    @abstractmethod
    def execute(self, arguments: BaseModel) -> Any:
        """Execute the tool using validated arguments."""
        raise NotImplementedError

    def get_definition(self) -> dict[str, Any]:
        """Return tool metadata and JSON schema for tool registration."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema(),
        }
