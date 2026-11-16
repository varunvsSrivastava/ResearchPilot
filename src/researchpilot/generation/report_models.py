
"""Data models for structured research reports."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ReportSection:
    """A section within a structured research report."""

    title: str
    content: str
    citations: list[int] = field(default_factory=list)


@dataclass
class ResearchReport:
    """Structured research report."""

    title: str
    research_question: str
    sections: list[ReportSection] = field(default_factory=list)
    conclusion: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
