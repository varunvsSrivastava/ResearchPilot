
"""FastAPI application for ResearchPilot."""

from fastapi import FastAPI

from researchpilot.api.routes.research import router as research_router


app = FastAPI(
    title="ResearchPilot API",
    description="Evidence-driven AI research assistant API.",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Check API health."""
    return {"status": "healthy"}


app.include_router(research_router)
