
"""Integrated academic retrieval pipeline."""

from researchpilot.retrieval.fulltext.models import Evidence
from researchpilot.retrieval.fulltext.service import (
    FullTextAcquisitionError,
    FullTextAcquisitionService,
)
from researchpilot.retrieval.models import Paper
from researchpilot.retrieval.pipeline_models import (
    PaperRetrievalResult,
    RetrievalPipelineResult,
)
from researchpilot.retrieval.services.academic_search import (
    AcademicSearchService,
)
from researchpilot.schemas.research_plan import ResearchPlan


class RetrievalPipeline:
    """Orchestrate academic search and full-text evidence acquisition."""

    def __init__(
        self,
        academic_search_service: AcademicSearchService,
        fulltext_service: FullTextAcquisitionService,
    ) -> None:
        self.academic_search_service = academic_search_service
        self.fulltext_service = fulltext_service

    def execute(
        self,
        research_plan: ResearchPlan,
        max_results_per_query: int = 10,
    ) -> RetrievalPipelineResult:
        """Execute academic retrieval and collect evidence."""

        search_results = self.academic_search_service.search_plan(
            research_plan=research_plan,
            max_results_per_query=max_results_per_query,
        )

        papers = self._deduplicate_papers(
            [
                paper
                for result in search_results
                for paper in result.papers
            ]
        )

        paper_results: list[PaperRetrievalResult] = []
        all_evidence: list[Evidence] = []

        for paper in papers:
            try:
                evidence = self.fulltext_service.acquire(paper)
                paper_result = PaperRetrievalResult(
                    paper=paper,
                    evidence=evidence,
                )
            except FullTextAcquisitionError as exc:
                paper_result = PaperRetrievalResult(
                    paper=paper,
                    error=str(exc),
                )

            paper_results.append(paper_result)
            all_evidence.extend(paper_result.evidence)

        successful_papers = sum(
            result.success for result in paper_results
        )

        failed_papers = len(paper_results) - successful_papers

        return RetrievalPipelineResult(
            research_question=research_plan.original_question,
            papers=papers,
            paper_results=paper_results,
            evidence=all_evidence,
            total_papers=len(papers),
            successful_papers=successful_papers,
            failed_papers=failed_papers,
            total_evidence=len(all_evidence),
        )

    @staticmethod
    def _deduplicate_papers(papers: list[Paper]) -> list[Paper]:
        """Deduplicate papers by DOI, falling back to normalized title."""

        unique_papers: list[Paper] = []
        seen_dois: set[str] = set()
        seen_titles: set[str] = set()

        for paper in papers:
            doi = paper.doi.lower().strip() if paper.doi else None
            title = " ".join(paper.title.lower().split())

            if doi and doi in seen_dois:
                continue

            if title and title in seen_titles:
                continue

            if doi:
                seen_dois.add(doi)

            if title:
                seen_titles.add(title)

            unique_papers.append(paper)

        return unique_papers
