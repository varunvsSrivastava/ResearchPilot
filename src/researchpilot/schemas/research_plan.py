
from typing import Literal

from pydantic import BaseModel, Field


SourceType = Literal["academic", "web"]


class SearchQuery(BaseModel):
    query: str = Field(
        min_length=3,
        description="A focused search query for retrieving relevant evidence.",
    )

    purpose: str = Field(
        min_length=10,
        description="The evidence or information this query aims to retrieve.",
    )

    source_type: SourceType = Field(
        description="The preferred source type for executing this query.",
    )


class SourcePreferences(BaseModel):
    primary_sources: list[SourceType] = Field(
        min_length=1,
        description="Source types prioritized during research.",
    )

    preferred_evidence: list[str] = Field(
        min_length=1,
        description="Preferred evidence characteristics, such as peer-reviewed research or systematic reviews.",
    )

    supplementary_sources: list[SourceType] = Field(
        default_factory=list,
        description="Additional source types used to supplement primary evidence.",
    )


class ResearchPlan(BaseModel):
    original_question: str = Field(
        min_length=3,
        description="The original research question provided by the user.",
    )

    research_objective: str = Field(
        min_length=10,
        description="The main objective of the research.",
    )

    sub_questions: list[str] = Field(
        min_length=2,
        max_length=6,
        description="Two to six focused questions that collectively investigate the research objective.",
    )

    search_queries: list[SearchQuery] = Field(
        min_length=2,
        max_length=8,
        description="Focused search queries mapped to evidence retrieval purposes and source types.",
    )

    source_preferences: SourcePreferences = Field(
        description="Source priorities and preferred evidence characteristics.",
    )

    research_strategy: str = Field(
        min_length=20,
        description="A concise strategy explaining how the research should approach the question.",
    )
