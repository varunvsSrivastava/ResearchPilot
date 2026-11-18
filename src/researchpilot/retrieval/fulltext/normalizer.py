
"""Text normalization for extracted academic documents."""

import re

from researchpilot.retrieval.fulltext.models import (
    ExtractedDocument,
    ExtractedPage,
)

class TextNormalizer:
    """Normalize PDF-extracted text while preserving block boundaries."""

    def normalize_block(self, text: str) -> str:
        """Normalize a single extracted text block."""
        if not text:
            return ""

        # Normalize line endings and whitespace characters.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\t", " ").replace("\u00a0", " ")

        # Reconstruct words split by end-of-line hyphenation.
        text = re.sub(
            r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])",
            "",
            text,
        )

        # Join wrapped lines within the same block.
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def normalize_page(self, text: str) -> str:
        """Normalize page text while preserving block boundaries."""
        if not text:
            return ""

        blocks = re.split(r"\n\s*\n+", text)

        normalized_blocks = [
            normalized
            for block in blocks
            if (normalized := self.normalize_block(block))
        ]

        return "\n\n".join(normalized_blocks)

    def normalize_document(
        self,
        document: ExtractedDocument,
    ) -> ExtractedDocument:
        """Normalize extracted text without losing page metadata."""
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
