"""Conflict detection over extracted claims.

Two claims are only ever compared if `rag/claims.py` assigned them the same
`concept` key — a same-policy-concept signal that is precise by
construction (see that module's docstring for why this is preferred over
general contradiction detection). This module adds three more of the
signals the Phase 3 brief calls for:

- **incompatible normalized values**: the actual claim values differ.
- **overlapping topic/heading**: computed and attached to every detected
  disagreement as `shared_topic_keywords`, for transparency/debugging —
  informational, not a gate, since a precise concept match is already
  strong enough evidence on its own that two chunks address the same
  point (see `_shared_topic_keywords`'s docstring for why this is not
  used as a hard filter).
- **explicit supersession relationships**: resolved via `rag/authority.py`,
  not re-derived here.

A disagreement between two claims from the *same* `document_id` is not a
cross-source conflict (a document does not conflict with itself) and is
skipped. Applicability tags are consulted before authority: two claims
that are both clearly scoped to *different, non-overlapping* segments
(e.g. "trailplus" vs. "standard") describe different situations, not a
disagreement, and are never turned into a `Conflict` at all — see
`rag/applicability.py`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

from aster_row_agent.rag.authority import DocumentIndex, evaluate_chunk_authority
from aster_row_agent.rag.claims import extract_claims
from aster_row_agent.rag.models import DocumentChunk
from aster_row_agent.rag.retrieval_models import (
    AuthorityDisposition,
    ChunkAuthority,
    Conflict,
    ConflictDisposition,
    ExtractedClaim,
)

_STOPWORDS = frozenset(
    {
        "this",
        "that",
        "with",
        "from",
        "your",
        "have",
        "will",
        "does",
        "when",
        "what",
        "which",
        "about",
        "policy",
        "information",
    }
)
_KEYWORD_RE = re.compile(r"[a-z]+")


def _significant_keywords(chunk: DocumentChunk) -> frozenset[str]:
    """Lowercase, length>=4, stopword-filtered, crudely-singularized keywords.

    Drawn from the chunk's title and heading path only (not its body text)
    so this stays a *topic* signal, not a second copy of the claim
    extractor. Used purely as diagnostic context on a detected `Conflict`
    (`shared_topic_keywords`) — never as a gate — because requiring topic
    overlap in addition to a concept match would (for example) suppress
    the corpus's legitimate non-authoritative disagreement between the
    current returns policy and the migration scratchpad, whose heading
    ("Unapproved legacy copy") shares no vocabulary with "Returns Policy"
    even though both make a claim about the same `return_window_days`
    concept.
    """
    haystack = f"{chunk.title} {chunk.heading_path}".lower()
    words = set()
    for word in _KEYWORD_RE.findall(haystack):
        if len(word) < 4 or word in _STOPWORDS:
            continue
        singular = word[:-1] if word.endswith("s") and len(word) > 4 else word
        words.add(singular)
    return frozenset(words)


def _applicability_conflict_excluded(claim_a: ExtractedClaim, claim_b: ExtractedClaim) -> bool:
    """True when the two claims are clearly scoped to different segments.

    Empty tags mean "no specific segment named" and are treated as
    potentially applying broadly, so they are never, by themselves, a
    reason to exclude a comparison — only two *non-empty, disjoint* tag
    sets (e.g. {"trailplus"} vs {"standard"}) represent a clear scope
    difference worth treating as "these describe different situations"
    rather than "these disagree."
    """
    if not claim_a.applicability_tags or not claim_b.applicability_tags:
        return False
    return claim_a.applicability_tags.isdisjoint(claim_b.applicability_tags)


def _classify_disagreement(
    authority_a: ChunkAuthority, authority_b: ChunkAuthority
) -> ConflictDisposition:
    """Turn two chunks' independent authority judgments into one disposition.

    Order matters and is deliberate: internal-operational content should
    never be reported as a customer *policy* conflict even if the other
    side is also non-authoritative, so it is checked first; a genuinely
    unapproved/draft side always prevents "genuine active conflict" status
    regardless of what it is being compared against; supersession is only
    reached once neither side is disqualified for those stronger reasons.
    """
    dispositions = {authority_a.disposition, authority_b.disposition}
    if (
        AuthorityDisposition.INTERNAL_OPERATIONAL in dispositions
        and AuthorityDisposition.NON_AUTHORITATIVE not in dispositions
    ):
        return ConflictDisposition.INTERNAL_OPERATIONAL_DISAGREEMENT
    if AuthorityDisposition.NON_AUTHORITATIVE in dispositions:
        return ConflictDisposition.NON_AUTHORITATIVE_DISAGREEMENT
    if AuthorityDisposition.SUPERSEDED in dispositions:
        return ConflictDisposition.RESOLVED_BY_SUPERSESSION
    if dispositions == {AuthorityDisposition.AUTHORITATIVE_CUSTOMER}:
        return ConflictDisposition.GENUINE_ACTIVE_CONFLICT
    return ConflictDisposition.UNCERTAIN


def _explain(concept: str, claim_a: ExtractedClaim, claim_b: ExtractedClaim) -> str:
    return (
        f"{concept}: document {claim_a.document_id} states {claim_a.value!r} "
        f"({claim_a.evidence_span!r}); document {claim_b.document_id} states "
        f"{claim_b.value!r} ({claim_b.evidence_span!r})."
    )


def detect_conflicts(
    chunks: list[DocumentChunk], document_index: DocumentIndex
) -> list[Conflict]:
    """Extract claims from every chunk and report every cross-document disagreement.

    Returns disagreements of *every* disposition (not just genuine active
    conflicts) — see `Conflict.disposition` — so callers can decide what
    each one means for their purposes (e.g. `rag/evidence.py` only lets
    GENUINE_ACTIVE_CONFLICT drive the bundle's top-level disposition, but
    keeps the others for transparency).
    """
    claims_by_concept: dict[str, list[ExtractedClaim]] = defaultdict(list)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for chunk in chunks:
        for claim in extract_claims(chunk):
            claims_by_concept[claim.concept].append(claim)

    conflicts: list[Conflict] = []
    for concept, claims in claims_by_concept.items():
        for claim_a, claim_b in combinations(claims, 2):
            if claim_a.document_id == claim_b.document_id:
                continue
            if claim_a.value == claim_b.value:
                continue
            if _applicability_conflict_excluded(claim_a, claim_b):
                continue

            chunk_a, chunk_b = chunk_by_id[claim_a.chunk_id], chunk_by_id[claim_b.chunk_id]
            authority_a = evaluate_chunk_authority(chunk_a, document_index)
            authority_b = evaluate_chunk_authority(chunk_b, document_index)
            disposition = _classify_disagreement(authority_a, authority_b)

            conflicts.append(
                Conflict(
                    concept=concept,
                    disposition=disposition,
                    claim_a=claim_a,
                    claim_b=claim_b,
                    shared_topic_keywords=tuple(
                        sorted(_significant_keywords(chunk_a) & _significant_keywords(chunk_b))
                    ),
                    explanation=_explain(concept, claim_a, claim_b),
                )
            )
    return conflicts
