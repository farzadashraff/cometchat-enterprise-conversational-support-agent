"""The local vector index: persistence and similarity search.

Storage format is deliberately simple: one JSON object per line
(`records.jsonl`), each a fully self-describing `IndexRecord` (chunk +
embedding + embedding model id). At this corpus's scale (tens of chunks)
there is no need for a binary format, a memory-mapped array, or a hosted
vector database — a flat JSONL file is human-inspectable, diff-friendly,
trivially rebuildable, and easy to reason about in a 6-8 hour assignment.

Every write is a full, atomic rebuild of the file (write to a temp file,
then rename over the target) — never an incremental append. That is what
makes re-running ingestion idempotent "for free": running it twice against
an unchanged corpus produces byte-for-byte the same file, and running it
against a changed corpus produces a fresh, correct snapshot with no
leftover stale records.

This module answers "what's the same as / close to this query in the
index" — nothing more. It has no opinion about which results are
authoritative, current, or safe to cite; that is the retrieval/authority
layer's job in a later phase (see docs/architecture.md §4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aster_row_agent.rag.embeddings import EmbeddingProvider
from aster_row_agent.rag.errors import IndexNotFoundError
from aster_row_agent.rag.models import IndexRecord, SearchResult

_RECORDS_FILENAME = "records.jsonl"


def index_records_path(index_dir: Path) -> Path:
    return index_dir / _RECORDS_FILENAME


def save_index(records: list[IndexRecord], index_dir: Path) -> Path:
    """Atomically (re)write the index as a full snapshot of `records`."""
    index_dir.mkdir(parents=True, exist_ok=True)
    target = index_records_path(index_dir)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")
    tmp.replace(target)
    return target


def load_index_records(index_dir: Path) -> list[IndexRecord]:
    """Load every persisted `IndexRecord` from disk."""
    path = index_records_path(index_dir)
    if not path.exists():
        raise IndexNotFoundError(path)
    records: list[IndexRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(IndexRecord.model_validate_json(line))
    return records


class VectorIndex:
    """An in-memory, load-once, cosine-similarity index over `IndexRecord`s."""

    def __init__(self, records: list[IndexRecord], embedding_provider: EmbeddingProvider) -> None:
        self._records = records
        self._embedding_provider = embedding_provider
        if records:
            matrix = np.array([record.embedding for record in records], dtype=float)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._normalized_matrix = matrix / norms
        else:
            self._normalized_matrix = np.zeros((0, embedding_provider.dimensions))

    @classmethod
    def load(cls, index_dir: Path, embedding_provider: EmbeddingProvider) -> VectorIndex:
        return cls(load_index_records(index_dir), embedding_provider)

    def __len__(self) -> int:
        return len(self._records)

    def score_all(self, query: str) -> list[float]:
        """Cosine similarity of `query` against every record, in record order.

        Used by the Phase 3 hybrid retriever, which needs a score for every
        candidate (not just a pre-truncated top-k) to normalize and fuse
        against the lexical index's scores. See `rag/retrieval.py`.
        """
        if not self._records:
            return []
        query_vector = np.array(self._embedding_provider.embed_query(query), dtype=float)
        query_norm = np.linalg.norm(query_vector) or 1.0
        query_normalized = query_vector / query_norm
        scores = self._normalized_matrix @ query_normalized
        return [float(score) for score in scores]

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the `top_k` chunks most similar to `query` by cosine similarity."""
        scores = self.score_all(query)
        if not scores:
            return []
        top_k = min(top_k, len(self._records))
        ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            SearchResult(chunk=self._records[i].chunk, score=scores[i]) for i in ranked_indices
        ]
