"""Hybrid (lexical + semantic) retrieval.

Neither method alone is sufficient (see the module-level rationale in
`rag/lexical.py`): BM25 is strong on exact terminology, product names, and
numbers; embeddings are strong on paraphrase and natural-language
concept-matching. Both are computed for *every* chunk on every query
(the corpus is small enough — tens of chunks — that this is cheap and
avoids the correctness pitfalls of merging two separately-truncated
top-k lists), then combined with a documented, deterministic fusion:

1. Score every chunk with BM25 (`lexical_score`) and cosine similarity
   (`semantic_score`).
2. Min-max normalize each score list independently to [0, 1] across the
   current candidate pool — the two raw score spaces are not comparable
   (BM25 is an unbounded corpus-relative score; cosine similarity is
   bounded but on a different scale), so combining them without
   normalizing first would let whichever method happens to produce larger
   raw numbers dominate regardless of actual relevance.
3. Fuse: `fused = lexical_weight * lexical_norm + semantic_weight * semantic_norm`,
   with configurable weights (default 0.5/0.5 — an even split, since
   neither signal is assumed more reliable than the other for this
   corpus; see `RetrievalOptions`).
4. Sort by fused score, descending, and return the top `top_k`.

The index (both the lexical and the vector side) is loaded once when the
`Retriever` is constructed, not per query — see `Retriever.load`.
"""

from __future__ import annotations

from pathlib import Path

from aster_row_agent.rag.embeddings import EmbeddingProvider
from aster_row_agent.rag.index import VectorIndex, load_index_records
from aster_row_agent.rag.lexical import LexicalIndex
from aster_row_agent.rag.models import IndexRecord
from aster_row_agent.rag.query import normalize_query
from aster_row_agent.rag.retrieval_models import (
    RetrievalCandidate,
    RetrievalOptions,
    RetrievalResult,
)


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Scale `scores` to [0, 1]; a degenerate (empty or constant) list maps to all zeros.

    Mapping a constant list to zero (rather than one, or leaving it
    unnormalized) means a method that found *no* discriminating signal at
    all — e.g. BM25 returning the same score for every chunk because none
    of the query's tokens appear anywhere in the corpus — contributes
    nothing to the fused score rather than a misleading uniform boost.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    spread = hi - lo
    if spread < 1e-9:
        return [0.0 for _ in scores]
    return [(score - lo) / spread for score in scores]


class Retriever:
    """Loads the index once; scores and fuses lexical + semantic results per query."""

    def __init__(self, records: list[IndexRecord], embedding_provider: EmbeddingProvider) -> None:
        self._records = records
        self._lexical_index = LexicalIndex(records)
        self._vector_index = VectorIndex(records, embedding_provider)

    @classmethod
    def load(cls, index_dir: Path, embedding_provider: EmbeddingProvider) -> Retriever:
        return cls(load_index_records(index_dir), embedding_provider)

    @property
    def records(self) -> list[IndexRecord]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def retrieve(self, query: str, options: RetrievalOptions | None = None) -> RetrievalResult:
        options = options or RetrievalOptions()
        normalized = normalize_query(query)

        if not self._records:
            return RetrievalResult(
                query=query, normalized_query=normalized.normalized, candidates=(), total_scored=0
            )

        lexical_scores = self._lexical_index.score_all(normalized.normalized)
        semantic_scores = self._vector_index.score_all(normalized.normalized)
        lexical_normalized = _min_max_normalize(lexical_scores)
        semantic_normalized = _min_max_normalize(semantic_scores)

        fused_scores = [
            options.lexical_weight * lexical_normalized[i]
            + options.semantic_weight * semantic_normalized[i]
            for i in range(len(self._records))
        ]
        ranked_indices = sorted(range(len(self._records)), key=lambda i: -fused_scores[i])
        top_indices = ranked_indices[: min(options.top_k, len(ranked_indices))]

        candidates = tuple(
            RetrievalCandidate(
                chunk=self._records[i].chunk,
                lexical_score=lexical_scores[i],
                lexical_score_normalized=lexical_normalized[i],
                semantic_score=semantic_scores[i],
                semantic_score_normalized=semantic_normalized[i],
                fused_score=fused_scores[i],
                rank=rank,
            )
            for rank, i in enumerate(top_indices)
        )
        return RetrievalResult(
            query=query,
            normalized_query=normalized.normalized,
            candidates=candidates,
            total_scored=len(self._records),
        )
