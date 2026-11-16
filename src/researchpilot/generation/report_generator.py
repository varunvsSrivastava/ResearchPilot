
"""Generate structured research reports from evidence synthesis."""

import re

from researchpilot.evidence.synthesis_models import ResearchSynthesis
from researchpilot.rag.pipeline.citations import CitationValidator

from .report_models import ResearchReport, ReportSection


class ReportGenerator:
    """Generate and validate structured research reports."""

    CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]")

    def __init__(self) -> None:
        self.citation_validator = CitationValidator()

    def _extract_citations(self, text: str) -> list[int]:
        """Extract citations from labels or numeric source identifiers."""

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

    def generate(
        self,
        synthesis: ResearchSynthesis,
        context: str,
        title: str | None = None,
    ) -> ResearchReport:
        """Generate and validate a structured report."""

        sections = []

        for index, finding in enumerate(synthesis.findings, start=1):
            citations = self._extract_citations(
                " ".join(finding.supporting_sources)
            )

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

        report_text = "\n".join(
            f"{finding.finding} "
            f"{' '.join(finding.supporting_sources)}"
            for finding in synthesis.findings
        )

        if not self.citation_validator.validate(report_text, context):
            raise ValueError(
                "Report contains citations that are not available "
                "in the provided context."
            )

        return report
