
from researchpilot.retrieval.academic.base import AcademicSearchProvider
from researchpilot.retrieval.models import Paper, SearchResult
from researchpilot.schemas.research_plan import ResearchPlan


class AcademicSearchService:
    """Execute academic queries from a research plan."""

    def __init__(self, provider: AcademicSearchProvider) -> None:
        self.provider = provider

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

    def search_plan(
        self,
        research_plan: ResearchPlan,
        max_results_per_query: int = 10,
    ) -> list[SearchResult]:
        """Execute academic queries and return query-level results."""
        if not 1 <= max_results_per_query <= 100:
            raise ValueError("Limit must be between 1 and 100.")

        results: list[SearchResult] = []

        for search_query in research_plan.search_queries:
            if search_query.source_type != "academic":
                continue

            papers = self.provider.search(
                query=search_query.query,
                limit=max_results_per_query,
            )

            unique_papers = self._deduplicate_papers(papers)

            results.append(
                SearchResult(
                    query=search_query.query,
                    provider="openalex",
                    papers=unique_papers,
                    total_results=len(unique_papers),
                )
            )

        return results
