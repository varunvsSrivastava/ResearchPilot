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

            raw_chunk = text[start:end]
            chunk_text = raw_chunk.strip()

            if chunk_text:
                leading_whitespace = len(raw_chunk) - len(
                    raw_chunk.lstrip()
                )
                trailing_whitespace = len(raw_chunk) - len(
                    raw_chunk.rstrip()
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

            target_start = end - self.overlap

            next_start = self._find_overlap_start(
                text=text,
                target=target_start,
                previous_end=end,
            )

            # Preserve word boundaries and guarantee forward progress.
            if next_start <= start:
                start = end
            else:
                start = next_start

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

    def _find_overlap_start(
        self,
        text: str,
        target: int,
        previous_end: int,
    ) -> int:
        """Find a safe overlap start at a word boundary."""
        if target <= 0:
            return 0

        if target >= previous_end:
            return target

        # If target is already at a whitespace boundary,
        # it is safe to start there.
        if text[target].isspace():
            return target

        # Move forward to the next whitespace boundary.
        next_whitespace = target

        while (
            next_whitespace < previous_end
            and not text[next_whitespace].isspace()
        ):
            next_whitespace += 1

        if next_whitespace < previous_end:
            return next_whitespace

        # If no whitespace exists before previous_end,
        # move backward to the preceding boundary.
        previous_whitespace = target

        while (
            previous_whitespace > 0
            and not text[previous_whitespace - 1].isspace()
        ):
            previous_whitespace -= 1

        return previous_whitespace
