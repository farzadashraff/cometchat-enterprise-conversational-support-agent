"""Evidence assembly: the Stage A -> B -> C pipeline the Phase 3 brief calls for.

    Stage A: retrieve relevant candidates          (rag/retrieval.py)
    Stage B: analyze metadata and authority         (rag/authority.py, rag/conflict.py)
    Stage C: construct safe evidence                (this module)

`EvidenceAssembler.assemble()` is the single entry point a future agent
layer should call; it returns one `EvidenceBundle` and nothing else. The
agent/LLM layer is not implemented yet (out of scope for this phase) and
must not need to re-derive any of these judgments from prose — every
decision (authority, conflict, sufficiency) is already a typed field on
the bundle.

Retrieval relevance is deliberately kept separate from authority
throughout: Stage A retrieves from the *entire* candidate pool, including
internal and non-authoritative chunks (the Phase 3 brief is explicit that
excluding them at retrieval time would prevent legitimate uses, like an
internal escalation chunk informing a handoff decision, or a
non-authoritative chunk being visible in diagnostics for exactly the kind
of prompt-injection/fabricated-policy case the corpus is designed to
test). Authority and citability are only decided in Stage B, per chunk,
from metadata alone.
"""

from __future__ import annotations

from pathlib import Path

from aster_row_agent.rag.authority import build_document_index, evaluate_chunk_authority
from aster_row_agent.rag.conflict import detect_conflicts
from aster_row_agent.rag.embeddings import EmbeddingProvider
from aster_row_agent.rag.models import DocumentChunk
from aster_row_agent.rag.retrieval import Retriever
from aster_row_agent.rag.retrieval_models import (
    AuthorityDisposition,
    ChunkAuthority,
    Citation,
    Conflict,
    ConflictDisposition,
    EvidenceAssemblyOptions,
    EvidenceBundle,
    EvidenceDisposition,
    EvidenceItem,
    RetrievalCandidate,
    RetrievalMetadata,
    RetrievalOptions,
)


def _citation_for(chunk: DocumentChunk) -> Citation:
    return Citation(
        filename=chunk.filename,
        heading=str(chunk.heading_path),
        document_id=chunk.document_id,
        title=chunk.title,
    )


def _evidence_item(candidate: RetrievalCandidate, authority: ChunkAuthority) -> EvidenceItem:
    chunk = candidate.chunk
    return EvidenceItem(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        title=chunk.title,
        source_filename=chunk.filename,
        heading_path=chunk.heading_path,
        text=chunk.text,
        fused_score=candidate.fused_score,
        lexical_score=candidate.lexical_score,
        semantic_score=candidate.semantic_score,
        authority=authority,
        citation=_citation_for(chunk),
    )


def _meets_absolute_floor(candidate: RetrievalCandidate, options: EvidenceAssemblyOptions) -> bool:
    """Stage 1: does this candidate look relevant on its own, absolute terms?

    Uses `candidate.semantic_score`/`lexical_score` — the *raw*, un-normalized
    values — never the pool-relative `fused_score`. See
    `EvidenceAssemblyOptions`'s docstring for why a purely relative
    threshold cannot detect "nothing retrieved is actually relevant."
    """
    return (
        candidate.semantic_score >= options.min_semantic_score
        or candidate.lexical_score >= options.min_lexical_score
    )


def select_relevant_candidates(
    candidates: list[RetrievalCandidate], options: EvidenceAssemblyOptions
) -> list[RetrievalCandidate]:
    """Apply both sufficiency stages and return the candidates worth acting on.

    Stage 1 (absolute) is applied first, over every candidate. Stage 2
    (relative) is then applied only among the stage-1 survivors, using the
    *best fused score among them* as its reference point — not the fused
    score of a candidate that stage 1 already rejected as insufficient.
    """
    absolute_qualified = [c for c in candidates if _meets_absolute_floor(c, options)]
    if not absolute_qualified:
        return []
    top_score = absolute_qualified[0].fused_score  # candidates are already fused-score-sorted
    relative_floor = top_score * options.relevance_margin
    return [c for c in absolute_qualified if top_score <= 0 or c.fused_score >= relative_floor]


def select_evidence(
    relevant: list[RetrievalCandidate],
    authorities: dict[str, ChunkAuthority],
    conflicts: list[Conflict],
    *,
    max_items: int,
) -> tuple[EvidenceItem, ...]:
    """Choose a small, bounded, safe-to-cite-or-reason-about evidence set.

    Priority order:
    1. Both sides of every GENUINE_ACTIVE_CONFLICT fully present in
       `relevant` are always included together — an unresolved conflict
       is never presented as if only one side existed (INVARIANT 9).
    2. Remaining slots go to AUTHORITATIVE_CUSTOMER candidates, by fused
       score, since those are what a normal answer should cite.
    3. If nothing authoritative was found at all, at most one
       INTERNAL_OPERATIONAL candidate is let through — for handoff
       reasoning only (see `EvidenceBundle.internal_evidence`), never as
       a citable source.

    Superseded and non-authoritative candidates are never selected here,
    even when relevant — they remain visible via
    `EvidenceBundle.superseded_evidence` / `.non_authoritative_evidence`
    for diagnostics, but are not treated as answer material.
    """
    selected: list[RetrievalCandidate] = []
    selected_ids: set[str] = set()

    for conflict in conflicts:
        if conflict.disposition is not ConflictDisposition.GENUINE_ACTIVE_CONFLICT:
            continue
        pair_ids = {conflict.claim_a.chunk_id, conflict.claim_b.chunk_id}
        matches = [c for c in relevant if c.chunk_id in pair_ids]
        if len(matches) != len(pair_ids):
            continue  # only part of the conflict pair is relevant to this query
        for candidate in matches:
            if candidate.chunk_id not in selected_ids:
                selected.append(candidate)
                selected_ids.add(candidate.chunk_id)

    for candidate in relevant:
        if len(selected) >= max_items:
            break
        if candidate.chunk_id in selected_ids:
            continue
        disposition = authorities[candidate.chunk_id].disposition
        if disposition is AuthorityDisposition.AUTHORITATIVE_CUSTOMER:
            selected.append(candidate)
            selected_ids.add(candidate.chunk_id)

    has_authoritative = any(
        authorities[c.chunk_id].disposition is AuthorityDisposition.AUTHORITATIVE_CUSTOMER
        for c in selected
    )
    if not has_authoritative:
        for candidate in relevant:
            if len(selected) >= max_items:
                break
            if candidate.chunk_id in selected_ids:
                continue
            disposition = authorities[candidate.chunk_id].disposition
            if disposition is AuthorityDisposition.INTERNAL_OPERATIONAL:
                selected.append(candidate)
                selected_ids.add(candidate.chunk_id)
                break

    return tuple(_evidence_item(c, authorities[c.chunk_id]) for c in selected)


def _filter_conflicts_to_relevant(
    conflicts: list[Conflict], relevant: list[RetrievalCandidate]
) -> list[Conflict]:
    """Keep only disagreements where *both* sides were relevant to this query.

    `detect_conflicts` runs over every retrieved candidate regardless of
    relevance (it needs the actual chunk objects, and the corpus is small
    enough that this is cheap) — but a latent disagreement between two
    chunks that both merely happened to land in the raw top-k, with one of
    them scoring nowhere near this query's actual topic, is not a
    conflict *for this query*. Filtering here (rather than not detecting
    it at all) keeps `rag/conflict.py` a general-purpose, corpus-wide
    mechanism while keeping each bundle's reported conflicts scoped to
    what is actually relevant right now — see the Phase 3 brief §15
    (topic contamination).
    """
    relevant_ids = {c.chunk_id for c in relevant}
    return [
        conflict
        for conflict in conflicts
        if {conflict.claim_a.chunk_id, conflict.claim_b.chunk_id} <= relevant_ids
    ]


def _determine_disposition(
    relevant: list[RetrievalCandidate],
    authorities: dict[str, ChunkAuthority],
    conflicts: list[Conflict],
) -> EvidenceDisposition:
    """`conflicts` must already be filtered to this query's relevant set (see above)."""
    genuine_conflict_is_live = any(
        conflict.disposition is ConflictDisposition.GENUINE_ACTIVE_CONFLICT
        for conflict in conflicts
    )
    if genuine_conflict_is_live:
        return EvidenceDisposition.AUTHORITATIVE_CONFLICT
    if not relevant:
        return EvidenceDisposition.INSUFFICIENT_EVIDENCE
    if any(
        authorities[c.chunk_id].disposition is AuthorityDisposition.AUTHORITATIVE_CUSTOMER
        for c in relevant
    ):
        return EvidenceDisposition.ANSWERABLE
    return EvidenceDisposition.NON_AUTHORITATIVE_ONLY


class EvidenceAssembler:
    """Owns a loaded `Retriever` and turns a query into an `EvidenceBundle`."""

    def __init__(
        self, retriever: Retriever, *, options: EvidenceAssemblyOptions | None = None
    ) -> None:
        self._retriever = retriever
        self._options = options or EvidenceAssemblyOptions()
        self._document_index = build_document_index(record.chunk for record in retriever.records)

    @classmethod
    def load(
        cls,
        index_dir: Path,
        embedding_provider: EmbeddingProvider,
        *,
        options: EvidenceAssemblyOptions | None = None,
    ) -> EvidenceAssembler:
        return cls(Retriever.load(index_dir, embedding_provider), options=options)

    def assemble(self, query: str) -> EvidenceBundle:
        retrieval_options = RetrievalOptions(
            top_k=self._options.top_k,
            lexical_weight=self._options.lexical_weight,
            semantic_weight=self._options.semantic_weight,
        )
        result = self._retriever.retrieve(query, retrieval_options)
        candidates = list(result.candidates)

        authorities = {
            candidate.chunk_id: evaluate_chunk_authority(candidate.chunk, self._document_index)
            for candidate in candidates
        }

        relevant = select_relevant_candidates(candidates, self._options)

        all_conflicts = detect_conflicts([c.chunk for c in candidates], self._document_index)
        conflicts = _filter_conflicts_to_relevant(all_conflicts, relevant)

        selected_evidence = select_evidence(
            relevant, authorities, conflicts, max_items=self._options.max_selected_evidence
        )
        disposition = _determine_disposition(relevant, authorities, conflicts)

        authoritative_evidence = tuple(
            item for item in selected_evidence
            if item.authority.disposition is AuthorityDisposition.AUTHORITATIVE_CUSTOMER
        )
        internal_evidence = tuple(
            item for item in selected_evidence
            if item.authority.disposition is AuthorityDisposition.INTERNAL_OPERATIONAL
        )
        superseded_evidence = tuple(
            _evidence_item(c, authorities[c.chunk_id])
            for c in relevant
            if authorities[c.chunk_id].disposition is AuthorityDisposition.SUPERSEDED
        )
        non_authoritative_evidence = tuple(
            _evidence_item(c, authorities[c.chunk_id])
            for c in relevant
            if authorities[c.chunk_id].disposition is AuthorityDisposition.NON_AUTHORITATIVE
        )
        customer_citable_sources = tuple(item.citation for item in authoritative_evidence)

        retrieval_metadata = RetrievalMetadata(
            total_scored=result.total_scored,
            candidate_count=len(candidates),
            relevant_count=len(relevant),
            top_fused_score=candidates[0].fused_score if candidates else None,
            min_semantic_score=self._options.min_semantic_score,
            min_lexical_score=self._options.min_lexical_score,
            relevance_margin=self._options.relevance_margin,
            lexical_weight=self._options.lexical_weight,
            semantic_weight=self._options.semantic_weight,
            embedding_model=self._retriever_embedding_model_id(),
        )

        return EvidenceBundle(
            query=result.query,
            normalized_query=result.normalized_query,
            disposition=disposition,
            candidates=tuple(candidates),
            selected_evidence=selected_evidence,
            authoritative_evidence=authoritative_evidence,
            internal_evidence=internal_evidence,
            superseded_evidence=superseded_evidence,
            non_authoritative_evidence=non_authoritative_evidence,
            conflicts=tuple(conflicts),
            customer_citable_sources=customer_citable_sources,
            retrieval_metadata=retrieval_metadata,
        )

    def _retriever_embedding_model_id(self) -> str:
        records = self._retriever.records
        return records[0].embedding_model if records else "unknown"
