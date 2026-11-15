
"""Text normalization for extracted academic documents."""

import re

from researchpilot.retrieval.fulltext.models import (
    ExtractedDocument,
    ExtractedPage,
)


class TextNormalizer:
    """Normalize PDF-extracted text while preserving page metadata."""

    def normalize_page(self, text: str) -> str:
        """Normalize text extracted from a single PDF page."""
        if not text:
            return ""

        # Reconstruct words split by end-of-line hyphenation.
        text = re.sub(
            r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])",
            "",
            text,
        )

        # Normalize line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Convert tabs and non-breaking spaces.
        text = text.replace("\t", " ").replace("\u00a0", " ")

        # Preserve paragraph boundaries while joining wrapped lines.
        paragraphs = re.split(r"\n\s*\n+", text)

        normalized_paragraphs = []

        for paragraph in paragraphs:
            # Collapse whitespace inside each paragraph.
            paragraph = re.sub(r"\s+", " ", paragraph).strip()

            if paragraph:
                normalized_paragraphs.append(paragraph)

        return "\n\n".join(normalized_paragraphs)

    def normalize_document(
        self,
        document: ExtractedDocument,
    ) -> ExtractedDocument:
        """Normalize extracted text without losing page provenance."""
        normalized_pages = [
            ExtractedPage(
                page_number=page.page_number,
                text=self.normalize_page(page.text),
            )
            for page in document.pages
        ]

        return document.model_copy(
            update={"pages": normalized_pages},
        )
