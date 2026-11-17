
"""OpenAlex academic search provider."""

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from researchpilot.retrieval.academic.base import AcademicSearchProvider
from researchpilot.retrieval.models import Paper


class OpenAlexProvider(AcademicSearchProvider):
    """Academic search provider backed by the OpenAlex Works API."""

    BASE_URL = "https://api.openalex.org/works"
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        if backoff_factor < 0:
            raise ValueError("backoff_factor cannot be negative.")

        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

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

    def _get_retry_delay(
        self,
        response: httpx.Response | None,
        attempt: int,
    ) -> float:
        """Calculate retry delay, respecting Retry-After when available."""
        if response is not None:
            retry_after = response.headers.get("Retry-After")

            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    try:
                        retry_at = parsedate_to_datetime(retry_after)

                        if retry_at.tzinfo is None:
                            retry_at = retry_at.replace(
                                tzinfo=timezone.utc
                            )

                        return max(
                            0.0,
                            (
                                retry_at
                                - datetime.now(timezone.utc)
                            ).total_seconds(),
                        )
                    except (TypeError, ValueError, OverflowError):
                        pass

        return self.backoff_factor * (2 ** attempt)

    def _get_with_retries(
        self,
        client: httpx.Client,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute an HTTP request with bounded retries."""
        for attempt in range(self.max_retries + 1):
            response = None

            try:
                response = client.get(self.BASE_URL, params=params)

                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    if attempt == self.max_retries:
                        response.raise_for_status()

                    delay = self._get_retry_delay(response, attempt)
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                return response.json()

            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == self.max_retries:
                    raise

                time.sleep(
                    self._get_retry_delay(None, attempt)
                )

        raise RuntimeError("OpenAlex request exhausted its retries.")

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
            data = self._get_with_retries(client, params)

        return [
            self._parse_work(work, query)
            for work in data.get("results", [])
        ]
