from typing import Literal

from pydantic import BaseModel, Field

class ResearchPlan(BaseModel):
    original_question: str = Field(
        description = "The original research question provided by the user."
    )
    
    research_objective: str = Field(
        description = "The main objective of the research."
    )
    
    sub_questions: list[str] = Field(
        description = "Smaller questions that help investigate the topic."
    )
    
    search_queries: list[str] = Field(
        description = "Search queries for retrieving relevant evidence."
    )
    
    source_types: list[Literal["web","academic"]] = Field(
        description = "Types of sources required for this research."
    )