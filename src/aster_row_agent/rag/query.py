"""Conservative query normalization.

Deliberately does very little: Unicode normalization (so visually-identical
characters compare equal) and whitespace collapsing. It does NOT lowercase,
stem, expand abbreviations, correct spelling, or rewrite the query in any
way that could change its meaning or invent entities the user didn't
mention. Retrieval must retrieve what the user asked about — see
docs/architecture.md §Phase 3 and the assignment's explicit warning against
"aggressively rewriting" the query.

Case-folding, where useful, is left to each retrieval method individually
(BM25 tokenization lowercases; embeddings are case-insensitive by
construction) rather than baked into the shared normalized query string,
so a case-sensitive future consumer (e.g. an order-ID extractor in a later
phase) still sees the user's original casing.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, ConfigDict

_WHITESPACE_RE = re.compile(r"\s+")


class NormalizedQuery(BaseModel):
    """A query before and after conservative normalization."""

    model_config = ConfigDict(frozen=True)

    raw: str
    normalized: str


def normalize_query(raw: str) -> NormalizedQuery:
    """Apply only whitespace and Unicode normalization to `raw`.

    - Unicode NFKC normalization so equivalent characters (e.g. full-width
      vs. half-width forms, combining vs. precomposed accents) compare
      equal without altering the text a human would read.
    - Collapse runs of whitespace (including newlines/tabs) to a single
      space, and strip leading/trailing whitespace.

    Nothing else. No entities are invented, corrected, or expanded.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return NormalizedQuery(raw=raw, normalized=text)
