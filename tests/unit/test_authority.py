from __future__ import annotations

import inspect
from pathlib import Path

from aster_row_agent.rag import authority, conflict
from aster_row_agent.rag.authority import (
    analyze_supersession,
    build_document_index,
    classify_chunk_authority,
    evaluate_chunk_authority,
)
from aster_row_agent.rag.chunking import build_document_chunks
from aster_row_agent.rag.models import Document, DocumentMetadata
from aster_row_agent.rag.retrieval_models import AuthorityDisposition

# The real corpus's filenames and document_ids — used ONLY to assert they
# never appear in the authority/conflict modules' source code, never to
# special-case behavior for them.
_REAL_FILENAMES = [
    "01-returns-policy-current.md",
    "02-returns-policy-legacy.md",
    "03-final-sale-and-promotions.md",
    "04-damaged-or-wrong-items.md",
    "05-domestic-shipping.md",
    "06-international-shipping.md",
    "07-warranty.md",
    "08-order-changes-and-cancellations.md",
    "09-trailplus-membership.md",
    "10-gift-cards-and-price-adjustments.md",
    "11-product-care.md",
    "12-breeze-tumbler-product-card.md",
    "13-support-escalation.md",
    "14-internal-content-migration-notes.md",
]
_REAL_DOCUMENT_IDS = [
    "RET-2026-01",
    "RET-2024-01",
    "RET-2026-02",
    "OPS-2026-04",
    "SHIP-2026-US",
    "SHIP-2026-INTL",
    "WAR-2026-01",
    "ORD-2026-01",
    "MEM-2026-01",
    "PAY-2026-03",
    "CARE-2026-01",
    "PROD-BREEZE-20",
    "SUP-2026-01",
    "MIG-TEST-04",
]


def _metadata(**overrides: object) -> DocumentMetadata:
    base: dict[str, object] = {
        "document_id": "DOC-1",
        "title": "Doc",
        "status": "active",
        "audience": "customer",
        "policy_authority": "official",
    }
    base.update(overrides)
    return DocumentMetadata.model_validate(base)


# --- classify_chunk_authority --------------------------------------------------


def test_active_official_customer_is_authoritative() -> None:
    assert classify_chunk_authority(_metadata()) is AuthorityDisposition.AUTHORITATIVE_CUSTOMER


def test_superseded_official_customer_is_superseded() -> None:
    metadata = _metadata(status="superseded", superseded_by="DOC-2")
    assert classify_chunk_authority(metadata) is AuthorityDisposition.SUPERSEDED


def test_draft_is_non_authoritative_regardless_of_audience() -> None:
    metadata = _metadata(status="draft", policy_authority="none", audience="internal")
    assert classify_chunk_authority(metadata) is AuthorityDisposition.NON_AUTHORITATIVE


def test_non_official_authority_is_non_authoritative_even_if_active() -> None:
    metadata = _metadata(status="active", policy_authority="none")
    assert classify_chunk_authority(metadata) is AuthorityDisposition.NON_AUTHORITATIVE


def test_active_official_internal_is_internal_operational_not_non_authoritative() -> None:
    """A legitimate internal ops doc (like the real support-escalation policy)
    must be distinguished from a draft/unapproved one — both are
    audience=internal, but only the draft one is NON_AUTHORITATIVE."""
    metadata = _metadata(audience="internal")
    assert classify_chunk_authority(metadata) is AuthorityDisposition.INTERNAL_OPERATIONAL


def test_active_official_unrecognized_audience_is_flagged_not_guessed() -> None:
    metadata = _metadata(audience="partner")
    assert classify_chunk_authority(metadata) is AuthorityDisposition.UNRECOGNIZED_AUDIENCE


def test_unknown_status_is_non_authoritative() -> None:
    """A future status this system has never seen must fail safe, not be assumed active."""
    metadata = _metadata(status="archived")
    assert classify_chunk_authority(metadata) is AuthorityDisposition.NON_AUTHORITATIVE


def test_effective_date_never_influences_authority_classification() -> None:
    """INVARIANT: a newer effective_date must not change the authority outcome."""
    older = _metadata(effective_date="2020-01-01")
    newer = _metadata(effective_date="2030-01-01")
    assert classify_chunk_authority(older) == classify_chunk_authority(newer)


# --- supersession -------------------------------------------------------------


def test_analyze_supersession_not_superseded() -> None:
    info = analyze_supersession(_metadata(), document_index={})
    assert info.is_superseded is False
    assert info.superseded_by_document_id is None
    assert info.superseding_document_present is None


def test_analyze_supersession_superseding_doc_present_and_active() -> None:
    superseded = _metadata(status="superseded", superseded_by="DOC-NEW")
    index = {"DOC-NEW": _metadata(document_id="DOC-NEW", status="active")}
    info = analyze_supersession(superseded, document_index=index)
    assert info.is_superseded is True
    assert info.superseding_document_present is True
    assert info.superseding_document_active is True


def test_analyze_supersession_superseding_doc_missing_from_corpus() -> None:
    superseded = _metadata(status="superseded", superseded_by="DOC-GHOST")
    info = analyze_supersession(superseded, document_index={})
    assert info.superseding_document_present is False
    assert info.superseding_document_active is False


def test_analyze_supersession_superseding_doc_present_but_not_active() -> None:
    """A data-quality situation: the doc that supposedly supersedes this one
    isn't itself active — surfaced rather than silently trusted."""
    superseded = _metadata(status="superseded", superseded_by="DOC-STALE")
    index = {"DOC-STALE": _metadata(document_id="DOC-STALE", status="draft")}
    info = analyze_supersession(superseded, document_index=index)
    assert info.superseding_document_present is True
    assert info.superseding_document_active is False


def test_analyze_supersession_date_ordering_consistent() -> None:
    superseded = _metadata(
        status="superseded", superseded_by="DOC-NEW", effective_date="2024-01-01"
    )
    index = {
        "DOC-NEW": _metadata(document_id="DOC-NEW", status="active", effective_date="2026-04-01")
    }
    info = analyze_supersession(superseded, document_index=index)
    assert info.date_ordering_consistent is True


def test_analyze_supersession_date_ordering_inconsistent() -> None:
    """A data-quality problem: the document claiming to supersede this one has
    an *earlier* effective_date, which should never happen — surfaced, not hidden."""
    superseded = _metadata(
        status="superseded", superseded_by="DOC-NEW", effective_date="2026-01-01"
    )
    index = {
        "DOC-NEW": _metadata(document_id="DOC-NEW", status="active", effective_date="2020-01-01")
    }
    info = analyze_supersession(superseded, document_index=index)
    assert info.date_ordering_consistent is False


def test_analyze_supersession_date_ordering_unknown_when_dates_missing() -> None:
    superseded = _metadata(status="superseded", superseded_by="DOC-NEW")
    index = {"DOC-NEW": _metadata(document_id="DOC-NEW", status="active")}
    info = analyze_supersession(superseded, document_index=index)
    assert info.date_ordering_consistent is None


def test_build_document_index_dedups_by_document_id() -> None:
    document = Document(
        source_path="knowledge-base/doc.md",
        metadata=_metadata(),
        body="# Doc\n\n## Section\n\nText one.\n\n## Section Two\n\nText two.\n",
        content_hash="irrelevant",
    )
    chunks = build_document_chunks(document, max_chunk_chars=1200)
    assert len(chunks) == 2  # same document_id on both chunks
    index = build_document_index(chunks)
    assert list(index.keys()) == ["DOC-1"]


# --- evaluate_chunk_authority combines disposition + citability + supersession --


def test_evaluate_chunk_authority_customer_citable_only_for_authoritative_customer() -> None:
    document = Document(
        source_path="knowledge-base/doc.md",
        metadata=_metadata(),
        body="# Doc\n\n## Section\n\nText.\n",
        content_hash="irrelevant",
    )
    chunk = build_document_chunks(document, max_chunk_chars=1200)[0]
    authority_result = evaluate_chunk_authority(chunk, document_index={"DOC-1": chunk.metadata})
    assert authority_result.disposition is AuthorityDisposition.AUTHORITATIVE_CUSTOMER
    assert authority_result.customer_citable is True


def test_evaluate_chunk_authority_internal_is_not_customer_citable() -> None:
    document = Document(
        source_path="knowledge-base/doc.md",
        metadata=_metadata(audience="internal"),
        body="# Doc\n\n## Section\n\nText.\n",
        content_hash="irrelevant",
    )
    chunk = build_document_chunks(document, max_chunk_chars=1200)[0]
    authority_result = evaluate_chunk_authority(chunk, document_index={"DOC-1": chunk.metadata})
    assert authority_result.customer_citable is False


# --- no filename hardcoding (standing guard) -----------------------------------


def test_no_filename_based_authority_logic() -> None:
    """The authority module must derive every decision from metadata fields
    alone — never a specific filename or document_id. This reads the
    module's own source and fails if any real corpus filename or
    document_id literal appears in it at all."""
    source = inspect.getsource(authority)
    for filename in _REAL_FILENAMES:
        assert filename not in source, f"authority.py must not reference {filename}"
    for document_id in _REAL_DOCUMENT_IDS:
        assert document_id not in source, f"authority.py must not reference {document_id}"


def test_no_filename_based_conflict_logic() -> None:
    """Same guard for the conflict module: conflicts must be found via
    metadata + extracted claims, never a specific filename/document_id
    comparison (a "curated registry of two specific files" would fail
    this test)."""
    source = inspect.getsource(conflict)
    for filename in _REAL_FILENAMES:
        assert filename not in source, f"conflict.py must not reference {filename}"
    for document_id in _REAL_DOCUMENT_IDS:
        assert document_id not in source, f"conflict.py must not reference {document_id}"


def test_authority_module_does_not_access_chunk_source_path_or_filename() -> None:
    """A slightly stronger static check: `chunk.source_path`/`chunk.filename`
    should not be *read* by the authority module at all — it has no
    legitimate reason to look at where a chunk came from, only what its
    metadata says."""
    source = Path(inspect.getfile(authority)).read_text()
    assert "chunk.source_path" not in source
    assert "chunk.filename" not in source
