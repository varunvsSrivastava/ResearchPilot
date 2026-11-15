
"""Models for vector metadata filtering."""

from pydantic import BaseModel, Field


class MetadataFilter(BaseModel):
    """Represent metadata conditions for vector search."""

    conditions: dict[str, str | int | float | bool] = Field(
        default_factory=dict
    )

    @property
    def is_empty(self) -> bool:
        """Check whether the filter has any conditions."""
        return not self.conditions

    def matches(
        self,
        metadata: dict[str, str | int | float | bool],
    ) -> bool:
        """Check whether metadata satisfies all filter conditions."""
        return all(
            metadata.get(key) == value
            for key, value in self.conditions.items()
        )
