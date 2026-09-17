"""Typed domain models for retrieval, authority, conflict, and evidence assembly.

Mirrors the split established in `rag/models.py` for ingestion: this module
holds pure data, no decision logic. The logic that produces and consumes
these models lives in `rag/query.py`, `rag/lexical.py`, `rag/retrieval.py`,
`rag/authority.py`, `rag/applicability.py`, `rag/claims.py`,
`rag/conflict.py`, and `rag/evidence.py`.

The central design invariant these models encode: **relevance, authority,
and customer-citability are three different questions**, and nothing here
lets one silently stand in for another. A `RetrievalCandidate` only ever
carries relevance signals (lexical/semantic/fused scores). Whether that
same chunk is *authoritative* is a separate `ChunkAuthority` judgment made
from metadata alone (see `rag/authority.py`), and whether it ends up
*customer-citable* is a further, narrower judgment
(`ChunkAuthority.customer_citable`). An `EvidenceBundle` is the only place
these three concerns are combined, and it combines them explicitly rather
than collapsing them into a single score.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from aster_row_agent.rag.models import DocumentChunk, HeadingPath

# --- retrieval ---------------------------------------------------------------


class RetrievalCandidate(BaseModel):
    """One chunk scored against a query, with every score component preserved.

    Provenance is never reconstructed from `text` after the fact — every
    field a citation could need is already here, copied from the
    `DocumentChunk` at retrieval time.
    """

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    lexical_score: float
    """Raw BM25 score against the normalized query (unbounded, corpus-relative)."""
    lexical_score_normalized: float
    """`lexical_score` min-max normalized to [0, 1] across this query's candidate pool."""
    semantic_score: float
    """Raw cosine similarity against the normalized query, in [-1, 1]."""
    semantic_score_normalized: float
    """`semantic_score` min-max normalized to [0, 1] across this query's candidate pool."""
    fused_score: float
    """Weighted combination of the two normalized scores. See `rag/retrieval.py`."""
    rank: int
    """0-based position in the fused ranking (0 = most relevant)."""

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class RetrievalOptions(BaseModel):
    """Tunable parameters for one retrieval call."""

    model_config = ConfigDict(frozen=True)

    top_k: int = 8
    """How many fused-ranked candidates to return. Small on purpose — see
    docs/architecture.md §6 (the entire corpus is never passed downstream)."""
    lexical_weight: float = 0.5
    semantic_weight: float = 0.5


class RetrievalResult(BaseModel):
    """Everything one `Retriever.retrieve()` call produced, for debugging and reuse."""

    model_config = ConfigDict(frozen=True)

    query: str
    """The original, unmodified user query."""
    normalized_query: str
    """After conservative normalization (whitespace/Unicode only) — see `rag/query.py`."""
    candidates: tuple[RetrievalCandidate, ...]
    """Top `top_k` candidates by fused score, descending."""
    total_scored: int
    """How many chunks in the corpus were scored (every chunk, at this corpus's scale)."""


# --- authority -----------------------------------------------------------------


class AuthorityDisposition(StrEnum):
    """What a chunk's metadata says about whether it can answer a customer, alone.

    Computed purely from `DocumentMetadata` — never from filename, never
    from retrieval relevance. See `rag/authority.py` for the derivation.
    """

    AUTHORITATIVE_CUSTOMER = "authoritative_customer"
    """active + official + customer audience: the current customer-facing answer."""
    SUPERSEDED = "superseded"
    """was official+customer, but a newer document now supersedes it."""
    NON_AUTHORITATIVE = "non_authoritative"
    """draft, or policy_authority is not "official" (e.g. the migration scratchpad)."""
    INTERNAL_OPERATIONAL = "internal_operational"
    """active + official, but audience is internal (e.g. escalation rules)."""
    UNRECOGNIZED_AUDIENCE = "unrecognized_audience"
    """active + official, but audience is neither "customer" nor "internal" —
    a data-quality situation this system has never seen; treated as
    non-citable rather than guessed at."""


class SupersessionInfo(BaseModel):
    """What a chunk's document claims about supersession, cross-checked against the corpus."""

    model_config = ConfigDict(frozen=True)

    is_superseded: bool
    superseded_by_document_id: str | None = None
    supersedes_document_id: str | None = None
    superseding_document_present: bool | None = None
    """True if `superseded_by_document_id` actually exists in the loaded corpus.
    None when `is_superseded` is False (the field is not applicable)."""
    superseding_document_active: bool | None = None
    """True if the superseding document is itself status=active. A False or
    None here (superseding doc missing or itself not active) is a corpus
    data-quality issue worth surfacing, not silently ignoring."""
    date_ordering_consistent: bool | None = None
    """True if the superseding document's `effective_date` is on or after
    this one's. This is the *only* place `effective_date` participates in
    authority evaluation at all, and only as a data-quality sanity check —
    never as a ranking or tie-breaking signal (see INVARIANT 3). None when
    either document's `effective_date` is unavailable or supersession does
    not apply, since consistency cannot be evaluated without both dates."""


class ChunkAuthority(BaseModel):
    """The full authority judgment for one chunk."""

    model_config = ConfigDict(frozen=True)

    disposition: AuthorityDisposition
    customer_citable: bool
    """True only for AUTHORITATIVE_CUSTOMER. Every other disposition is False —
    this is the single field the response layer should check before ever
    citing a chunk to a customer."""
    supersession: SupersessionInfo


# --- applicability ---------------------------------------------------------------

# Deliberately not an exhaustive enum: applicability scope is discovered from
# text via `rag/applicability.py`, not declared up front, since the corpus
# does not name a closed set of segments anywhere in its metadata.
ApplicabilityTags = frozenset[str]


# --- claims / conflict -------------------------------------------------------------


class ExtractedClaim(BaseModel):
    """One normalized, structured assertion pulled from a chunk's text.

    Claims are how the conflict detector compares documents without a
    general-purpose NLP contradiction engine: each `ClaimExtractor` (see
    `rag/claims.py`) recognizes one narrow, well-defined concept (a return
    window in days, a boolean dishwasher-safety statement, ...) and pulls
    out a directly comparable value. Two claims are only ever compared when
    they share the same `concept` string.
    """

    model_config = ConfigDict(frozen=True)

    concept: str
    value: bool | int | float | str
    chunk_id: str
    document_id: str
    applicability_tags: ApplicabilityTags
    evidence_span: str
    """The sentence (or similar short span) the claim was extracted from —
    kept for debugging/explanation, never re-parsed as an instruction."""


class ConflictDisposition(StrEnum):
    """What kind of disagreement two claims represent, once authority is applied."""

    GENUINE_ACTIVE_CONFLICT = "genuine_active_conflict"
    """Both claims come from AUTHORITATIVE_CUSTOMER chunks, from different
    documents, with no supersession relationship between them. This must
    stay unresolved — see docs/architecture.md §5 and INVARIANT 9."""
    RESOLVED_BY_SUPERSESSION = "resolved_by_supersession"
    """One claim's document has been superseded by the other's (directly or
    transitively via metadata) — not a live conflict."""
    NON_AUTHORITATIVE_DISAGREEMENT = "non_authoritative_disagreement"
    """At least one side is NON_AUTHORITATIVE (draft / non-official) — the
    disagreement is real text, but the non-authoritative side can never
    override the other regardless of the other side's disposition."""
    INTERNAL_OPERATIONAL_DISAGREEMENT = "internal_operational_disagreement"
    """At least one side is INTERNAL_OPERATIONAL and neither side is
    NON_AUTHORITATIVE — an internal/operational nuance, not a customer
    policy conflict."""
    UNCERTAIN = "uncertain"
    """The claims disagree and neither of the above rules cleanly applies
    (e.g. an AUTHORITATIVE_CUSTOMER claim vs. an UNRECOGNIZED_AUDIENCE
    claim) — preserved rather than forced into a stronger category."""


class Conflict(BaseModel):
    """One detected disagreement between two extracted claims."""

    model_config = ConfigDict(frozen=True)

    concept: str
    disposition: ConflictDisposition
    claim_a: ExtractedClaim
    claim_b: ExtractedClaim
    shared_topic_keywords: tuple[str, ...]
    """Heading/title keywords shared by the two source chunks — a secondary,
    diagnostic topic-overlap signal (see docs/architecture.md §Phase 3),
    not itself a gate on detection."""
    explanation: str
    """Generated from structured fields only (concept + both claims'
    citations and values) — never phrased by an LLM."""


# --- citations / evidence items -----------------------------------------------------


class Citation(BaseModel):
    """A structured source reference, sufficient to render without an LLM."""

    model_config = ConfigDict(frozen=True)

    filename: str
    heading: str
    document_id: str
    title: str

    def render(self) -> str:
        return f"{self.filename} — {self.heading}" if self.heading else self.filename


class EvidenceItem(BaseModel):
    """One chunk as it appears in an EvidenceBundle, with its authority judgment attached."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    title: str
    source_filename: str
    heading_path: HeadingPath
    text: str
    fused_score: float
    lexical_score: float
    semantic_score: float
    authority: ChunkAuthority
    citation: Citation


class EvidenceDisposition(StrEnum):
    """The bundle-level answer to "can this query be answered from evidence?"."""

    ANSWERABLE = "answerable"
    """At least one AUTHORITATIVE_CUSTOMER chunk is relevant and selected."""
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    """Nothing in the corpus was relevant enough to the query."""
    AUTHORITATIVE_CONFLICT = "authoritative_conflict"
    """A GENUINE_ACTIVE_CONFLICT touches the relevant evidence for this
    query — takes priority over ANSWERABLE even if one side alone would
    have been usable."""
    NON_AUTHORITATIVE_ONLY = "non_authoritative_only"
    """Something relevant was found, but none of it is customer-citable
    (only internal/superseded/non-authoritative/unrecognized-audience
    evidence exists for this topic)."""


class RetrievalMetadata(BaseModel):
    """Debug/observability summary of how one EvidenceBundle was assembled."""

    model_config = ConfigDict(frozen=True)

    total_scored: int
    candidate_count: int
    relevant_count: int
    top_fused_score: float | None
    min_semantic_score: float
    min_lexical_score: float
    relevance_margin: float
    lexical_weight: float
    semantic_weight: float
    embedding_model: str


class EvidenceAssemblyOptions(BaseModel):
    """Tunable parameters for evidence assembly, layered on top of `RetrievalOptions`.

    Sufficiency is decided in two stages, deliberately using different
    score spaces for each (see `rag/evidence.py::_passes_threshold`):

    1. An **absolute** floor (`min_semantic_score` / `min_lexical_score`)
       checked against each candidate's *raw*, un-normalized score — a
       real question's own scale, not relative to whatever else happened
       to be retrieved this turn.
    2. A **relative** margin (`relevance_margin`) checked against the
       fused, pool-normalized score, applied only among candidates that
       already cleared stage 1 — this trims a long tail of "technically
       above the floor but nowhere near as relevant as the top hit"
       candidates.

    Stage 1 must use raw scores, not the min-max-normalized fused score:
    min-max normalization always stretches the current pool's best score
    to 1.0 by construction, which means a purely relative threshold can
    never detect "nothing retrieved is actually relevant" — the least-bad
    candidate in a pool of uniformly irrelevant chunks would still
    normalize to the maximum. This was caught empirically while
    calibrating these defaults: see docs/architecture.md §Phase 3 for the
    raw cosine-similarity measurements (on-topic queries, including
    heavy paraphrases with zero lexical overlap, scored 0.64-0.80; a
    battery of clearly off-topic queries — weather, sports, recipes,
    unrelated small talk — scored 0.44-0.59 against the same corpus and
    embedding model). 0.60 sits in the gap between those two clusters.
    """

    model_config = ConfigDict(frozen=True)

    top_k: int = 8
    lexical_weight: float = 0.5
    semantic_weight: float = 0.5
    min_semantic_score: float = 0.60
    """Absolute floor on raw cosine similarity (embedding model:
    BAAI/bge-small-en-v1.5). See the class docstring for the calibration
    data this default was chosen from. A starting point for further
    empirical tuning in the evaluation phase (docs/evaluation-plan.md),
    not a final calibrated value — exposed as configuration specifically
    so it can be adjusted without a code change.
    """
    min_lexical_score: float = 2.0
    """Absolute floor on raw BM25 score. Only needs to catch a strong
    exact-term match (product names, numbers, country names); combined
    with `min_semantic_score` via OR, since either signal alone is good
    evidence of genuine relevance (see `rag/retrieval.py`'s module
    docstring on why neither method is used alone)."""
    relevance_margin: float = 0.5
    """Among candidates that already cleared the absolute floor, a
    candidate must also score at least `relevance_margin * top_fused_score`
    (fused, pool-normalized) to remain "relevant" — a *relative* cutoff
    that prevents a long tail of marginally-qualifying candidates from
    being treated as equally relevant as a clearly-dominant top hit.
    """
    max_selected_evidence: int = 4
    """Upper bound on how many chunks are ever selected into
    `selected_evidence` — the entire corpus is never passed downstream
    merely because a question was asked (INVARIANT 10)."""


class EvidenceBundle(BaseModel):
    """The complete, explicit evidence decision for one query.

    This is what the (future) agent/LLM layer consumes — never raw
    retrieval results. Every field here is structured data; nothing in
    this bundle asks an LLM to infer whether evidence is sufficient,
    authoritative, or conflicting.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    normalized_query: str
    disposition: EvidenceDisposition
    candidates: tuple[RetrievalCandidate, ...]
    """Every candidate retrieval considered (bounded by top_k), before thresholding."""
    selected_evidence: tuple[EvidenceItem, ...]
    """The small, bounded, safe-to-cite-or-reason-about set actually chosen."""
    authoritative_evidence: tuple[EvidenceItem, ...]
    """Subset of `selected_evidence` that is customer-citable."""
    internal_evidence: tuple[EvidenceItem, ...]
    """Subset of `selected_evidence` that is internal-operational only — may
    inform a handoff decision, must never be cited to a customer."""
    superseded_evidence: tuple[EvidenceItem, ...]
    """Superseded chunks that were relevant to the query, kept for
    diagnostics even though they were not selected as an answer source."""
    non_authoritative_evidence: tuple[EvidenceItem, ...]
    """Draft/non-official chunks that were relevant to the query, kept for
    diagnostics (e.g. to show *why* a fabricated claim must be rejected)."""
    conflicts: tuple[Conflict, ...]
    """Every detected disagreement among the retrieved candidates, of any
    disposition — not just genuine ones. See `Conflict.disposition`."""
    customer_citable_sources: tuple[Citation, ...]
    """Citations for `authoritative_evidence` only — this is the exact list
    a response is allowed to cite."""
    retrieval_metadata: RetrievalMetadata

    @property
    def has_unresolved_conflict(self) -> bool:
        return self.disposition is EvidenceDisposition.AUTHORITATIVE_CONFLICT
