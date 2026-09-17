"""Lexical (BM25) retrieval.

Complements the dense/semantic index: exact product names, order-like
identifiers, country names, membership names, and specific numbers are
often better matched by literal token overlap than by embedding
similarity, especially for a small, terminology-dense policy corpus. See
`rag/retrieval.py` for how this is fused with semantic scores.

Built once from the corpus's `IndexRecord`s (loaded once, scored many
times) — never rebuilt per query, and never re-reads the source Markdown
files.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from aster_row_agent.rag.models import IndexRecord

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

# A small, standard English stopword list. Without this, BM25's IDF term can
# give pathologically high weight to a common function word that happens to
# be locally rare across this small corpus's ~53 short chunks (observed in
# practice: "how"/"does"/"what" once outranked the actual Canada-shipping
# document for a Canada shipping question, because those words happened to
# cluster in one unrelated internal chunk). Removing them is standard IR
# practice and does not affect the semantic/dense side of retrieval at all.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to", "in", "on",
        "at", "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "from", "up", "down", "out", "off",
        "over", "under", "again", "further", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "having", "do", "does", "did", "doing", "will",
        "would", "shall", "should", "can", "could", "may", "might", "must", "i", "you",
        "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
        "his", "its", "our", "their", "this", "that", "these", "those", "what", "which",
        "who", "whom", "how", "when", "where", "why", "as", "not", "no", "nor", "too",
        "very", "just", "also",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenization, with common English stopwords removed."""
    return [
        token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOPWORDS
    ]


class LexicalIndex:
    """A BM25 index over a fixed list of `IndexRecord`s, in a fixed order.

    `score_all` returns one score per record, in the same order the index
    was built with — callers align it against `records` positionally, the
    same convention `VectorIndex.score_all` uses.
    """

    def __init__(self, records: list[IndexRecord]) -> None:
        self._record_count = len(records)
        corpus_tokens = [tokenize(record.chunk.text) for record in records]
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def __len__(self) -> int:
        return self._record_count

    def score_all(self, query: str) -> list[float]:
        """BM25 score of `query` against every record, in record order."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        return [float(score) for score in scores]
