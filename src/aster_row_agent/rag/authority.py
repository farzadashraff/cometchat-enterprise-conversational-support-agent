"""Metadata-driven authority evaluation.

INVARIANT (see docs/architecture.md and the Phase 3 brief): filename never
determines authority, and retrieval relevance never determines authority.
Every function here takes a `DocumentMetadata` (or a chunk's metadata) and
returns a judgment derived purely from its fields — `status`,
`policy_authority`, `audience`, `supersedes`/`superseded_by`. Nothing in
this module ever branches on a document's file path or name, and there
must never be such a branch;
`tests/unit/test_authority.py::test_no_filename_based_authority_logic` is
a standing guard against that regressing.

A newer `effective_date` is never used to break a tie between two active,
official documents — see `analyze_supersession`'s `date_ordering_consistent`
field for the one place `effective_date` is consulted at all, and it is a
data-quality sanity check (does a supersession claim's dates make sense?),
not a ranking signal.
"""

from __future__ import annotations

from collections.abc import Iterable

from aster_row_agent.rag.models import DocumentChunk, DocumentMetadata
from aster_row_agent.rag.retrieval_models import (
    AuthorityDisposition,
    ChunkAuthority,
    SupersessionInfo,
)

DocumentIndex = dict[str, DocumentMetadata]


def build_document_index(chunks: Iterable[DocumentChunk]) -> DocumentIndex:
    """Collapse a chunk list into one `DocumentMetadata` per `document_id`.

    Needed for supersession cross-checks, which must ask "does the
    document this one claims to supersede/be superseded by actually exist,
    and is it active?" — a question that spans documents, not one chunk.
    """
    index: DocumentIndex = {}
    for chunk in chunks:
        index.setdefault(chunk.document_id, chunk.metadata)
    return index


def classify_chunk_authority(metadata: DocumentMetadata) -> AuthorityDisposition:
    """The core authority gate pipeline, in priority order.

    1. Not active/superseded, or not officially authored -> NON_AUTHORITATIVE.
       This is checked first so a draft/unapproved internal document (like
       the migration scratchpad, which is *also* audience=internal) is
       classified by what makes it unsafe to use as policy — its
       draft/non-official status — rather than being lumped in with
       legitimate internal operational guidance.
    2. Internal audience (and not already excluded above) -> INTERNAL_OPERATIONAL.
    3. Superseded (and not already excluded above) -> SUPERSEDED.
    4. Customer audience, active, official -> AUTHORITATIVE_CUSTOMER.
    5. Anything else (active + official + some other audience value) ->
       UNRECOGNIZED_AUDIENCE — a data-quality situation this corpus never
       actually produces, handled by refusing to treat it as citable
       rather than guessing.
    """
    if metadata.status not in ("active", "superseded") or metadata.policy_authority != "official":
        return AuthorityDisposition.NON_AUTHORITATIVE
    if metadata.audience == "internal":
        return AuthorityDisposition.INTERNAL_OPERATIONAL
    if metadata.status == "superseded":
        return AuthorityDisposition.SUPERSEDED
    if metadata.audience == "customer":
        return AuthorityDisposition.AUTHORITATIVE_CUSTOMER
    return AuthorityDisposition.UNRECOGNIZED_AUDIENCE


def analyze_supersession(
    metadata: DocumentMetadata, document_index: DocumentIndex
) -> SupersessionInfo:
    """Cross-check a document's supersession claims against the loaded corpus.

    `superseded_by`/`supersedes` are trusted as authored (this is not a
    place to second-guess the corpus's own claims), but whether the
    referenced document actually exists and is active is verified rather
    than assumed — a dangling or stale supersession pointer is a corpus
    data-quality issue worth surfacing, not silently trusting.
    """
    is_superseded = metadata.status == "superseded"
    superseding_id = metadata.superseded_by if is_superseded else None

    superseding_present: bool | None = None
    superseding_active: bool | None = None
    date_ordering_consistent: bool | None = None
    if superseding_id is not None:
        superseding_metadata = document_index.get(superseding_id)
        superseding_present = superseding_metadata is not None
        superseding_active = (
            superseding_metadata.status == "active" if superseding_metadata is not None else False
        )
        if (
            superseding_metadata is not None
            and superseding_metadata.effective_date is not None
            and metadata.effective_date is not None
        ):
            date_ordering_consistent = (
                superseding_metadata.effective_date >= metadata.effective_date
            )

    return SupersessionInfo(
        is_superseded=is_superseded,
        superseded_by_document_id=superseding_id,
        supersedes_document_id=metadata.supersedes,
        superseding_document_present=superseding_present,
        superseding_document_active=superseding_active,
        date_ordering_consistent=date_ordering_consistent,
    )


def evaluate_chunk_authority(chunk: DocumentChunk, document_index: DocumentIndex) -> ChunkAuthority:
    """The full authority judgment for one chunk: disposition + supersession + citability."""
    disposition = classify_chunk_authority(chunk.metadata)
    supersession = analyze_supersession(chunk.metadata, document_index)
    return ChunkAuthority(
        disposition=disposition,
        customer_citable=disposition is AuthorityDisposition.AUTHORITATIVE_CUSTOMER,
        supersession=supersession,
    )
