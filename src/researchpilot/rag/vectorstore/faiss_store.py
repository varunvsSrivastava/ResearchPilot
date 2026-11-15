
"""FAISS vector store implementation."""

from pathlib import Path
import json
import faiss
import numpy as np

from researchpilot.rag.vectorstore.base import VectorStore
from researchpilot.rag.vectorstore.models import (
    VectorRecord,
    VectorSearchResponse,
    VectorSearchResult,
)


class FAISSVectorStore(VectorStore):
    """Store and retrieve vectors using FAISS."""

    def __init__(self, dimensions: int) -> None:
        if dimensions < 1:
            raise ValueError("Vector dimensions must be positive.")

        self.dimensions = dimensions
        self.index = faiss.IndexFlatIP(dimensions)
        self.records: dict[str, VectorRecord] = {}
        self.record_ids: list[str] = []


    def add_records(self, records: list[VectorRecord]) -> None:
        """Add records to the FAISS index."""
        if not records:
            return

        ids = [record.id for record in records]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate record IDs in batch.")

        if any(record_id in self.records for record_id in ids):
            raise ValueError("Record ID already exists.")

        if any(record.dimensions != self.dimensions for record in records):
            raise ValueError("Vector dimensions do not match the index.")

        vectors = np.asarray(
            [record.vector for record in records],
            dtype=np.float32,
        )

        faiss.normalize_L2(vectors)
        self.index.add(vectors)

        for record in records:
            self.records[record.id] = record
            self.record_ids.append(record.id)


    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> VectorSearchResponse:
        """Search for similar records using cosine similarity."""
        if top_k < 1:
            raise ValueError("top_k must be positive.")

        if len(query_vector) != self.dimensions:
            raise ValueError("Query vector dimensions do not match the index.")

        if self.index.ntotal == 0:
            return VectorSearchResponse()

        query = np.asarray([query_vector], dtype=np.float32)
        faiss.normalize_L2(query)

        limit = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, limit)

        results = []

        for score, index_position in zip(scores[0], indices[0]):
            if index_position < 0:
                continue

            record_id = self.record_ids[index_position]
            record = self.records[record_id]

            results.append(
                VectorSearchResult(
                    record=record,
                    score=float(score),
                )
            )

        return VectorSearchResponse(
            results=results,
            total_results=len(results),
        )


    def delete_records(self, record_ids: list[str]) -> None:
        """Delete records and rebuild the FAISS index."""
        if not record_ids:
            return

        ids_to_delete = set(record_ids)

        missing_ids = ids_to_delete - self.records.keys()

        if missing_ids:
            raise ValueError(
                f"Unknown record IDs: {sorted(missing_ids)}"
            )

        remaining_records = [
            self.records[record_id]
            for record_id in self.record_ids
            if record_id not in ids_to_delete
        ]

        self.index = faiss.IndexFlatIP(self.dimensions)
        self.records = {}
        self.record_ids = []

        self.add_records(remaining_records)
        
        
    def save(self, path: str) -> None:
        """Persist the FAISS index and record metadata."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.json"

        faiss.write_index(self.index, str(index_path))

        metadata = {
            "dimensions": self.dimensions,
            "record_ids": self.record_ids,
            "records": {
                record_id: record.model_dump()
                for record_id, record in self.records.items()
            },
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    
    def load(self, path: str) -> None:
        """Load a persisted FAISS index and record metadata."""
        directory = Path(path)
        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "FAISS index or metadata file not found."
            )

        index = faiss.read_index(str(index_path))

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        dimensions = metadata["dimensions"]

        if index.d != dimensions:
            raise ValueError(
                "FAISS index dimensions do not match metadata."
            )

        records = {
            record_id: VectorRecord.model_validate(record_data)
            for record_id, record_data in metadata["records"].items()
        }

        record_ids = metadata["record_ids"]

        if index.ntotal != len(record_ids) or len(records) != len(record_ids):
            raise ValueError(
                "FAISS index and metadata record counts do not match."
            )

        if set(record_ids) != set(records):
            raise ValueError(
                "FAISS record IDs do not match metadata."
            )

        self.dimensions = dimensions
        self.index = index
        self.records = records
        self.record_ids = record_ids

