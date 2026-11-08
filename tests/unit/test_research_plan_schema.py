
import pytest
from pydantic import ValidationError

from researchpilot.schemas.research_plan import (
    ResearchPlan,
    SearchQuery,
    SourcePreferences,
)


@pytest.fixture
def valid_search_query():
    return SearchQuery(
        query="generative AI healthcare applications",
        purpose="Identify applications of generative AI in healthcare.",
        source_type="academic",
    )


@pytest.fixture
def valid_source_preferences():
    return SourcePreferences(
        primary_sources=["academic"],
        preferred_evidence=[
            "Peer-reviewed research",
            "Systematic reviews",
        ],
        supplementary_sources=["web"],
    )


@pytest.fixture
def valid_research_plan(
    valid_search_query,
    valid_source_preferences,
):
    return ResearchPlan(
        original_question="How is generative AI transforming healthcare?",
        research_objective=(
            "Analyze applications, benefits, and challenges "
            "of generative AI in healthcare."
        ),
        sub_questions=[
            "What are the major applications?",
            "What challenges does it introduce?",
        ],
        search_queries=[
            valid_search_query,
            SearchQuery(
                query="generative AI healthcare challenges",
                purpose="Identify limitations and risks.",
                source_type="web",
            ),
        ],
        source_preferences=valid_source_preferences,
        research_strategy=(
            "Prioritize academic evidence and supplement "
            "findings with reliable web sources."
        ),
    )


def test_search_query_valid(valid_search_query):
    assert valid_search_query.query
    assert valid_search_query.source_type == "academic"


def test_search_query_invalid_source_type():
    with pytest.raises(ValidationError):
        SearchQuery(
            query="generative AI healthcare",
            purpose="Identify relevant healthcare applications.",
            source_type="blog",
        )


def test_search_query_empty_query():
    with pytest.raises(ValidationError):
        SearchQuery(
            query="",
            purpose="Identify relevant healthcare applications.",
            source_type="academic",
        )


def test_source_preferences_valid(valid_source_preferences):
    assert valid_source_preferences.primary_sources == ["academic"]
    assert valid_source_preferences.supplementary_sources == ["web"]


def test_source_preferences_empty_primary_sources():
    with pytest.raises(ValidationError):
        SourcePreferences(
            primary_sources=[],
            preferred_evidence=["Peer-reviewed research"],
        )


def test_research_plan_valid(valid_research_plan):
    assert isinstance(valid_research_plan, ResearchPlan)
    assert len(valid_research_plan.sub_questions) == 2
    assert len(valid_research_plan.search_queries) == 2


def test_research_plan_too_few_sub_questions(
    valid_search_query,
    valid_source_preferences,
):
    with pytest.raises(ValidationError):
        ResearchPlan(
            original_question="Explain generative AI in healthcare.",
            research_objective="Analyze healthcare applications.",
            sub_questions=["What are its applications?"],
            search_queries=[
                valid_search_query,
                valid_search_query,
            ],
            source_preferences=valid_source_preferences,
            research_strategy="Prioritize academic research.",
        )


def test_research_plan_too_many_sub_questions(
    valid_search_query,
    valid_source_preferences,
):
    with pytest.raises(ValidationError):
        ResearchPlan(
            original_question="Explain generative AI in healthcare.",
            research_objective="Analyze healthcare applications.",
            sub_questions=[
                "Question one?",
                "Question two?",
                "Question three?",
                "Question four?",
                "Question five?",
                "Question six?",
                "Question seven?",
            ],
            search_queries=[
                valid_search_query,
                valid_search_query,
            ],
            source_preferences=valid_source_preferences,
            research_strategy="Prioritize academic research.",
        )


def test_research_plan_too_few_search_queries(
    valid_source_preferences,
):
    with pytest.raises(ValidationError):
        ResearchPlan(
            original_question="Explain generative AI in healthcare.",
            research_objective="Analyze healthcare applications.",
            sub_questions=[
                "What are its applications?",
                "What are its limitations?",
            ],
            search_queries=[],
            source_preferences=valid_source_preferences,
            research_strategy="Prioritize academic research.",
        )


def test_research_plan_invalid_nested_query(
    valid_source_preferences,
):
    with pytest.raises(ValidationError):
        ResearchPlan(
            original_question="Explain generative AI in healthcare.",
            research_objective="Analyze healthcare applications.",
            sub_questions=[
                "What are its applications?",
                "What are its limitations?",
            ],
            search_queries=[
                {
                    "query": "AI healthcare",
                    "purpose": "Evidence retrieval",
                    "source_type": "invalid",
                },
                {
                    "query": "AI healthcare challenges",
                    "purpose": "Identify healthcare challenges",
                    "source_type": "academic",
                },
            ],
            source_preferences=valid_source_preferences,
            research_strategy="Prioritize academic research.",
        )
