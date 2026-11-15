
"""Paragraph-aware text chunking for academic documents."""

from researchpilot.retrieval.fulltext.models import (
    ExtractedDocument,
    TextChunk,
)


class TextChunker:
    """Split normalized academic text into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if overlap < 0:
            raise ValueError("overlap cannot be negative.")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(
        self,
        document: ExtractedDocument,
    ) -> list[TextChunk]:
        """Chunk every page while preserving page provenance."""
        chunks: list[TextChunk] = []

        for page in document.pages:
            page_chunks = self._chunk_page(
                text=page.text,
                document=document,
                page_number=page.page_number,
            )
            chunks.extend(page_chunks)

        return chunks

    def _chunk_page(
        self,
        text: str,
        document: ExtractedDocument,
        page_number: int,
    ) -> list[TextChunk]:
        """Chunk a single page using paragraph-aware boundaries."""
        if not text.strip():
            return []

        chunks: list[TextChunk] = []
        start = 0
        chunk_index = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            if end < text_length:
                end = self._find_boundary(text, start, end)

            chunk_text = text[start:end].strip()

            if chunk_text:
                # Adjust offsets to exclude surrounding whitespace.
                leading_whitespace = len(text[start:end]) - len(
                    text[start:end].lstrip()
                )
                trailing_whitespace = len(text[start:end]) - len(
                    text[start:end].rstrip()
                )

                actual_start = start + leading_whitespace
                actual_end = end - trailing_whitespace

                chunks.append(
                    TextChunk(
                        paper_id=document.paper_id,
                        title=document.title,
                        doi=document.doi,
                        source_url=document.source_url,
                        pdf_url=document.pdf_url,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        start_char=actual_start,
                        end_char=actual_end,
                    )
                )

                chunk_index += 1

            if end >= text_length:
                break

            next_start = end - self.overlap

            # Guarantee forward progress.
            start = max(next_start, start + 1)

        return chunks

    def _find_boundary(
        self,
        text: str,
        start: int,
        end: int,
    ) -> int:
        """Prefer paragraph boundaries, then whitespace boundaries."""
        minimum_boundary = start + self.chunk_size // 2

        paragraph_boundary = text.rfind(
            "\n\n",
            minimum_boundary,
            end,
        )

        if paragraph_boundary != -1:
            return paragraph_boundary

        whitespace_boundary = text.rfind(
            " ",
            minimum_boundary,
            end,
        )

        if whitespace_boundary != -1:
            return whitespace_boundary

        return end
