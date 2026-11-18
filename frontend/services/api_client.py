
"""HTTP client for communicating with the ResearchPilot API."""

import os
from typing import Any

import httpx


DEFAULT_API_URL = "http://127.0.0.1:8000"


class ResearchPilotAPIError(Exception):
    """Raised when the ResearchPilot API request fails."""


class ResearchPilotClient:
    """Client for interacting with the ResearchPilot backend."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (
            base_url or os.getenv("RESEARCHPILOT_API_URL", DEFAULT_API_URL)
        ).rstrip("/")

    def health_check(self) -> dict[str, Any]:
        """Check whether the backend API is healthy."""
        return self._request("GET", "/health")

    def execute_research(self, research_question: str) -> dict[str, Any]:
        """Submit a research question to the backend."""
        if not research_question or len(research_question.strip()) < 3:
            raise ValueError("Research question must contain at least 3 characters.")

        payload = {"research_question": research_question.strip()}

        return self._request(
            "POST",
            "/api/v1/research",
            json=payload,
            timeout=300.0,
        )

    def _request(
        self,
        method: str,
        endpoint: str,
        timeout: float = 10.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an HTTP request and handle API errors."""
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{endpoint}",
                timeout=timeout,
                **kwargs,
            )

            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException as exc:
            raise ResearchPilotAPIError(
                "The API request timed out. Please try again."
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise ResearchPilotAPIError(
                f"API returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:500]}"
            ) from exc

        except httpx.RequestError as exc:
            raise ResearchPilotAPIError(
                "Unable to connect to the ResearchPilot API. "
                "Check whether the backend server is running."
            ) from exc

        except ValueError as exc:
            raise ResearchPilotAPIError(
                "The API returned an invalid JSON response."
            ) from exc
