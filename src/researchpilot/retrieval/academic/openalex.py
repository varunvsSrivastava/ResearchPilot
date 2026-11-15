
from typing import Any

import httpx

from researchpilot.retrieval.academic.base import AcademicSearchProvider
from researchpilot.retrieval.models import Paper


class OpenAlexProvider(AcademicSearchProvider):
    """Academic search provider backed by the OpenAlex Works API."""

    BASE_URL = "https://api.openalex.org/works"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout

    @staticmethod
    def _reconstruct_abstract(
        inverted_index: dict[str, list[int]] | None,
    ) -> str | None:
        """Reconstruct plaintext abstract from OpenAlex inverted index."""
        if not inverted_index:
            return None

        words: dict[int, str] = {}

        for word, positions in inverted_index.items():
            for position in positions:
                words[position] = word

        if not words:
            return None

        return " ".join(
            words[position]
            for position in sorted(words)
        )

    @staticmethod
    def _normalize_doi(doi: str | None) -> str | None:
        """Normalize DOI by removing its URL prefix."""
        if not doi:
            return None

        doi = doi.strip()

        for prefix in (
            "https://doi.org/",
            "http://doi.org/",
            "https://dx.doi.org/",
            "http://dx.doi.org/",
        ):
            if doi.lower().startswith(prefix):
                return doi[len(prefix):]

        return doi

    @staticmethod
    def _extract_pdf_urls(work: dict[str, Any]) -> list[str]:
        """Extract unique PDF URLs, prioritizing the best OA location."""
        pdf_urls: list[str] = []
        seen_urls: set[str] = set()

        best_oa_location = work.get("best_oa_location") or {}
        locations = work.get("locations") or []

        candidates = [
            best_oa_location.get("pdf_url"),
            *[
                location.get("pdf_url")
                for location in locations
                if isinstance(location, dict)
            ],
        ]

        for url in candidates:
            if not isinstance(url, str):
                continue

            url = url.strip()

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            pdf_urls.append(url)

        return pdf_urls

    @staticmethod
    def _parse_work(
        work: dict[str, Any],
        query: str,
    ) -> Paper:
        """Convert an OpenAlex work record into a Paper model."""
        authors = [
            authorship["author"]["display_name"]
            for authorship in work.get("authorships", [])
            if authorship.get("author", {}).get("display_name")
        ]

        open_access = work.get("open_access") or {}
        pdf_urls = OpenAlexProvider._extract_pdf_urls(work)

        return Paper(
            paper_id=work["id"].rsplit("/", maxsplit=1)[-1],
            provider="openalex",
            title=work.get("display_name") or work.get("title") or "Untitled",
            abstract=OpenAlexProvider._reconstruct_abstract(
                work.get("abstract_inverted_index")
            ),
            authors=authors,
            publication_year=work.get("publication_year"),
            doi=OpenAlexProvider._normalize_doi(work.get("doi")),
            source_url=work["id"],
            pdf_url=pdf_urls[0] if pdf_urls else None,
            pdf_urls=pdf_urls,
            citation_count=work.get("cited_by_count"),
            is_open_access=open_access.get("is_oa"),
            search_query=query,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[Paper]:
        """Search OpenAlex and return normalized academic papers."""
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if not 1 <= limit <= 100:
            raise ValueError("Limit must be between 1 and 100.")

        params: dict[str, Any] = {
            "search": query,
            "per_page": limit,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

        return [
            self._parse_work(work, query)
            for work in data.get("results", [])
        ]
