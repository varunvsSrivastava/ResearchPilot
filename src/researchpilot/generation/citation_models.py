
"""Models for report citation validation."""

from dataclasses import dataclass, field


@dataclass
class CitationValidationResult:
    """Represent the outcome of report citation validation."""

    is_valid: bool
    cited_sources: set[int] = field(default_factory=set)
    invalid_sources: set[int] = field(default_factory=set)
    missing_citations: list[str] = field(default_factory=list)
    unused_sources: set[int] = field(default_factory=set)
