"""Structured claim extraction: the deterministic half of conflict detection.

This is NOT a general contradiction-detection engine — the Phase 3 brief
explicitly rules that out as unreliable for a 6-8 hour build. Instead, each
`ClaimExtractor` recognizes one narrow, well-defined policy concept via
pattern matching scoped to individual sentences, and produces a directly
comparable `ExtractedClaim` (a number, a boolean, ...). `rag/conflict.py`
only ever compares claims that share the same `concept` key, which is a
much stronger and more precise signal than generic text similarity.

This is deliberately a small, explicit registry (`CLAIM_EXTRACTORS`) rather
than a framework: adding a new concept means adding one function and
registering it, and every extractor is independently unit-tested against
both its true positives and its near-miss false positives in the real
corpus (see tests/unit/test_claims.py). The Phase 3 brief explicitly
allows a "curated configuration ... as a supplementary mechanism" as long
as it is not the *only* signal; here it is paired with the fully generic
metadata-driven authority/supersession gates in `rag/authority.py` and
`rag/conflict.py`, which apply to any concept this registry is ever
extended to cover.

Extraction operates on `chunk.text` only, which is untrusted corpus
content (see docs/security-model.md) — every extractor is a plain regex
match; nothing here executes, evaluates, or follows instructions that
might appear in that text, including the literal fake "SYSTEM
INSTRUCTION" payload in the migration scratchpad, which simply fails to
match any of these narrow patterns and produces no claim at all.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from aster_row_agent.rag.applicability import applicability_tags
from aster_row_agent.rag.models import DocumentChunk
from aster_row_agent.rag.retrieval_models import ExtractedClaim

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """A deliberately simple sentence splitter — good enough for short policy prose."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


ClaimExtractor = Callable[[DocumentChunk], list[ExtractedClaim]]

# --- return_window_days -------------------------------------------------------------

_RETURN_WORD_RE = re.compile(r"\breturn", re.IGNORECASE)
_DAY_COUNT_RE = re.compile(r"(\d+)[\s-]*(?:calendar[\s-]*)?days?\b", re.IGNORECASE)


def _extract_return_window_claims(chunk: DocumentChunk) -> list[ExtractedClaim]:
    """A day-count in a sentence that also mentions "return" is a return-window claim.

    Scoped to the sentence (not the whole chunk) so that, e.g., a refund-
    timing sentence ("allow 5-7 business days...") in the same chunk as a
    return-window sentence is not conflated with it, and so a chunk that
    merely cross-references another segment's policy without stating a
    number of its own contributes no claim.
    """
    claims: list[ExtractedClaim] = []
    for sentence in _split_sentences(chunk.text):
        if not _RETURN_WORD_RE.search(sentence):
            continue
        match = _DAY_COUNT_RE.search(sentence)
        if match is None:
            continue
        claims.append(
            ExtractedClaim(
                concept="return_window_days",
                value=int(match.group(1)),
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                applicability_tags=applicability_tags(sentence),
                evidence_span=sentence,
            )
        )
    return claims


# --- breeze_tumbler_body_dishwasher_safe --------------------------------------------

_TUMBLER_WORD_RE = re.compile(r"\btumbler\b", re.IGNORECASE)
_ALL_COMPONENTS_DISHWASHER_SAFE_RE = re.compile(
    r"all components?\s+(?:is|are)\s+dishwasher[\s-]*safe", re.IGNORECASE
)
_HAND_WASHED_RE = re.compile(r"\bhand-?washed\b", re.IGNORECASE)


def _extract_breeze_tumbler_dishwasher_claims(chunk: DocumentChunk) -> list[ExtractedClaim]:
    """Whether the tumbler *body* is dishwasher-safe, per this chunk's text.

    Gated on the word "tumbler" appearing somewhere in the chunk's title,
    heading path, or text (so an unrelated "tumble dry" instruction for
    bags elsewhere in the product-care guide is never mistaken for a claim
    about the tumbler), then per-sentence pattern matching for the two
    specific, contradictory phrasings the corpus actually uses.
    """
    haystack = f"{chunk.title} {chunk.heading_path} {chunk.text}"
    if not _TUMBLER_WORD_RE.search(haystack):
        return []

    claims: list[ExtractedClaim] = []
    for sentence in _split_sentences(chunk.text):
        if _ALL_COMPONENTS_DISHWASHER_SAFE_RE.search(sentence):
            value: bool | None = True
        elif _HAND_WASHED_RE.search(sentence):
            value = False
        else:
            value = None
        if value is None:
            continue
        claims.append(
            ExtractedClaim(
                concept="breeze_tumbler_body_dishwasher_safe",
                value=value,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                applicability_tags=applicability_tags(sentence),
                evidence_span=sentence,
            )
        )
    return claims


CLAIM_EXTRACTORS: tuple[ClaimExtractor, ...] = (
    _extract_return_window_claims,
    _extract_breeze_tumbler_dishwasher_claims,
)


def extract_claims(chunk: DocumentChunk) -> list[ExtractedClaim]:
    """Run every registered extractor over one chunk and collect their claims."""
    claims: list[ExtractedClaim] = []
    for extractor in CLAIM_EXTRACTORS:
        claims.extend(extractor(chunk))
    return claims
