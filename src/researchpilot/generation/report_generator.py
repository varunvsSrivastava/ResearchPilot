
"""Generate structured research reports from evidence synthesis."""

import re

from researchpilot.evidence.synthesis_models import ResearchSynthesis
from researchpilot.rag.pipeline.citations import CitationValidator

from .report_models import ResearchReport, ReportSection


class ReportGenerator:
    """Generate and validate structured research reports."""

    CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]")
    EVIDENCE_ID_PATTERN = re.compile(r"Evidence ID:\s*(\S+)")

    def __init__(self) -> None:
        self.citation_validator = CitationValidator()

    def _extract_citations(self, text: str) -> list[int]:
        """Extract numeric source references from text."""

        citations = [
            int(match)
            for match in self.CITATION_PATTERN.findall(text)
        ]

        if not citations:
            citations = [
                int(source_id)
                for source_id in text.split()
                if source_id.isdigit()
            ]

        return list(dict.fromkeys(citations))

    def _build_source_mapping(self, context: str) -> dict[str, int]:
        """Map evidence record IDs to their numeric context citations."""

        source_mapping = {}
        current_source = None

        for line in context.splitlines():
            citation_match = self.CITATION_PATTERN.search(line)

            if citation_match:
                current_source = int(citation_match.group(1))
                continue

            evidence_match = self.EVIDENCE_ID_PATTERN.search(line)

            if evidence_match and current_source is not None:
                evidence_id = evidence_match.group(1)
                source_mapping[evidence_id] = current_source
                current_source = None

        return source_mapping

    def generate(
        self,
        synthesis: ResearchSynthesis,
        context: str,
        title: str | None = None,
    ) -> ResearchReport:
        """Generate a structured report with traceable citations."""

        source_mapping = self._build_source_mapping(context)
        available_sources = self.citation_validator.extract_available_sources(
            context
        )

        sections = []

        for index, finding in enumerate(synthesis.findings, start=1):
            citations = []

            for source_id in finding.supporting_sources:
                direct_citations = self._extract_citations(source_id)

                if direct_citations:
                    citations.extend(direct_citations)
                    continue

                citation_id = source_mapping.get(source_id)

                if citation_id is not None:
                    citations.append(citation_id)

            citations = list(dict.fromkeys(citations))

            content = finding.finding

            sections.append(
                ReportSection(
                    title=f"Finding {index}",
                    content=content,
                    citations=citations,
                )
            )

        report = ResearchReport(
            title=title or "Research Report",
            research_question=synthesis.research_question,
            sections=sections,
            conclusion=synthesis.synthesis_summary,
            metadata={
                "evidence_count": str(synthesis.evidence_count),
                "finding_count": str(len(synthesis.findings)),
                "conclusion_is_summary": "true",
            },
        )

        for section in report.sections:
            if not section.citations:
                raise ValueError(
                    f"Finding '{section.title}' has no valid source "
                    "references in the provided context."
                )

            invalid_sources = set(section.citations) - available_sources

            if invalid_sources:
                raise ValueError(
                    "Report contains citations that are not available "
                    "in the provided context."
                )

        return report
