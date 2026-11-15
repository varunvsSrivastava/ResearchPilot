
"""Academic search tool for research agent."""

from pydantic import BaseModel, Field

from researchpilot.tools.base import ResearchTool


class AcademicSearchInput(BaseModel):
    """Arguments for academic paper search."""

    queries: list[str] = Field(
        min_length=2,
        description="At least two academic search queries to execute.",
    )
    max_results_per_query: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results returned per query.",
    )


class AcademicSearchTool(ResearchTool):
    """Search academic literature through AcademicSearchService."""

    name = "academic_search"
    description = (
        "Search academic literature for research papers using scholarly queries."
    )
    args_schema = AcademicSearchInput

    def __init__(self, academic_search_service) -> None:
        self.academic_search_service = academic_search_service

    def execute(self, arguments: AcademicSearchInput) -> list:
        """Execute academic searches and return deduplicated papers."""
        from researchpilot.schemas.research_plan import (
            ResearchPlan,
            SearchQuery,
            SourcePreferences,
        )

        search_queries = [
            SearchQuery(
                query=query,
                purpose=f"Academic search for: {query}",
                source_type="academic",
            )
            for query in arguments.queries
        ]

        research_plan = ResearchPlan(
            original_question=arguments.queries[0],
            research_objective=(
                "Retrieve academic papers relevant to the research queries."
            ),
            sub_questions=[
                f"What research addresses {query}?"
                for query in arguments.queries[:2]
            ],
            search_queries=search_queries,
            source_preferences=SourcePreferences(
                primary_sources=["academic"],
                preferred_evidence=["peer-reviewed research"],
            ),
            research_strategy=(
                "Search academic literature using the provided scholarly queries."
            ),
        )

        return self.academic_search_service.search_plan(
            research_plan,
            max_results_per_query=arguments.max_results_per_query,
        )
