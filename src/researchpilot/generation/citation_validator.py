
"""Validate citations in structured research reports."""

import re

from researchpilot.rag.pipeline.citations import CitationValidator

from .citation_models import CitationValidationResult
from .report_models import ResearchReport


class ReportCitationValidator:
    """Validate citations across a structured research report."""

    CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]")

    def __init__(self) -> None:
        self.citation_validator = CitationValidator()

    def validate(
        self,
        report: ResearchReport,
        context: str,
    ) -> CitationValidationResult:
        """Validate report citations against retrieved context."""

        available_sources = (
            self.citation_validator.extract_available_sources(context)
        )

        cited_sources: set[int] = set()
        missing_citations: list[str] = []

        for section in report.sections:
            content_citations = set(
                int(match)
                for match in self.CITATION_PATTERN.findall(section.content)
            )

            cited_sources.update(section.citations)
            cited_sources.update(content_citations)

            if not section.citations and not content_citations:
                missing_citations.append(section.title)

        conclusion_citations = set(
            int(match)
            for match in self.CITATION_PATTERN.findall(report.conclusion)
        )
        cited_sources.update(conclusion_citations)

        if report.conclusion.strip() and not conclusion_citations:
            missing_citations.append("Conclusion")

        invalid_sources = cited_sources - available_sources
        unused_sources = available_sources - cited_sources

        return CitationValidationResult(
            is_valid=not invalid_sources and not missing_citations,
            cited_sources=cited_sources,
            invalid_sources=invalid_sources,
            missing_citations=missing_citations,
            unused_sources=unused_sources,
        )
