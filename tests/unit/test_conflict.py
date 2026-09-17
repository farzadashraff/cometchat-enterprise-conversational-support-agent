from __future__ import annotations

from pathlib import Path

from aster_row_agent.rag.authority import build_document_index
from aster_row_agent.rag.chunking import build_document_chunks
from aster_row_agent.rag.conflict import detect_conflicts
from aster_row_agent.rag.models import Document, DocumentChunk, DocumentMetadata
from aster_row_agent.rag.retrieval_models import ConflictDisposition


def _doc(body: str, **metadata_overrides: object) -> Document:
    base: dict[str, object] = {
        "document_id": metadata_overrides.pop("document_id", "DOC-A"),
        "title": "Doc",
        "status": "active",
        "audience": "customer",
        "policy_authority": "official",
    }
    base.update(metadata_overrides)
    metadata = DocumentMetadata.model_validate(base)
    return Document(
        source_path=f"knowledge-base/{base['document_id']}.md",
        metadata=metadata,
        body=body,
        content_hash="irrelevant",
    )


def _chunks(*docs: Document) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for doc in docs:
        chunks.extend(build_document_chunks(doc, max_chunk_chars=4000))
    return chunks


_RETURN_BODY_30 = (
    "# Returns\n\n## Standard return window\n\nCustomers may return within 30 calendar days.\n"
)
_RETURN_BODY_45 = (
    "# Legacy Returns\n\n## Return window\n\nCustomers could return within 45 calendar days.\n"
)


def test_two_active_official_customer_docs_disagreeing_is_genuine_conflict() -> None:
    doc_a = _doc(_RETURN_BODY_30, document_id="DOC-A")
    doc_b = _doc(_RETURN_BODY_45, document_id="DOC-B")
    chunks = _chunks(doc_a, doc_b)
    conflicts = detect_conflicts(chunks, build_document_index(chunks))
    assert len(conflicts) == 1
    assert conflicts[0].disposition is ConflictDisposition.GENUINE_ACTIVE_CONFLICT
    assert conflicts[0].concept == "return_window_days"
    assert {conflicts[0].claim_a.value, conflicts[0].claim_b.value} == {30, 45}


def test_same_document_id_is_never_a_conflict_with_itself() -> None:
    body = (
        _RETURN_BODY_30
        + "\n## Another mention\n\nA different section could return within 45 calendar days.\n"
    )
    doc = _doc(body, document_id="DOC-A")
    chunks = _chunks(doc)
    conflicts = detect_conflicts(chunks, build_document_index(chunks))
    assert conflicts == []


def test_identical_values_are_not_a_conflict() -> None:
    doc_a = _doc(_RETURN_BODY_30, document_id="DOC-A")
    doc_b = _doc(
        "# Other\n\n## Return window\n\nCustomers may return within 30 calendar days.\n",
        document_id="DOC-B",
    )
    chunks = _chunks(doc_a, doc_b)
    conflicts = detect_conflicts(chunks, build_document_index(chunks))
    assert conflicts == []


def test_superseded_doc_disagreeing_with_its_successor_is_resolved() -> None:
    doc_a = _doc(_RETURN_BODY_30, document_id="DOC-A")
    doc_b = _doc(_RETURN_BODY_45, document_id="DOC-B", status="superseded", superseded_by="DOC-A")
    chunks = _chunks(doc_a, doc_b)
    conflicts = detect_conflicts(chunks, build_document_index(chunks))
    assert len(conflicts) == 1
    assert conflicts[0].disposition is ConflictDisposition.RESOLVED_BY_SUPERSESSION


def test_draft_non_authoritative_disagreement_is_never_genuine() -> None:
    doc_a = _doc(_RETURN_BODY_30, document_id="DOC-A")
    doc_b = _doc(_RETURN_BODY_45, document_id="DOC-B", status="draft", policy_authority="none")
    chunks = _chunks(doc_a, doc_b)
    conflicts = detect_conflicts(chunks, build_document_index(chunks))
    assert len(conflicts) == 1
    assert conflicts[0].disposition is ConflictDisposition.NON_AUTHORITATIVE_DISAGREEMENT


def test_internal_operational_disagreement_is_not_a_customer_policy_conflict() -> None:
    doc_a = _doc(_RETURN_BODY_30, document_id="DOC-A")
    doc_b = _doc(_RETURN_BODY_45, document_id="DOC-B", audience="internal")
    chunks = _chunks(doc_a, doc_b)
    conflicts = detect_conflicts(chunks, build_document_index(chunks))
    assert len(conflicts) == 1
    assert conflicts[0].disposition is ConflictDisposition.INTERNAL_OPERATIONAL_DISAGREEMENT


def test_draft_takes_priority_over_internal_when_both_apply() -> None:
    """A draft+internal doc (like the real migration scratchpad) disagreeing
    with an active/official/internal doc must be classified by its draft
    status, not diluted into a merely-internal disagreement."""
    doc_a = _doc(_RETURN_BODY_30, document_id="DOC-A", audience="internal")
    doc_b = _doc(
        _RETURN_BODY_45,
        document_id="DOC-B",
        audience="internal",
        status="draft",
        policy_authority="none",
    )
    chunks = _chunks(doc_a, doc_b)
    conflicts = detect_conflicts(chunks, build_document_index(chunks))
    assert len(conflicts) == 1
    assert conflicts[0].disposition is ConflictDisposition.NON_AUTHORITATIVE_DISAGREEMENT


def test_disjoint_applicability_tags_are_not_a_conflict() -> None:
    """Standard vs. TrailPlus windows legitimately differ — not a conflict."""
    doc_a = _doc(
        "# Returns\n\n## Standard return window\n\n"
        "Customers on the standard plan may return within 30 calendar days.\n",
        document_id="DOC-A",
    )
    doc_b = _doc(
        "# TrailPlus\n\n## Return window\n\n"
        "A TrailPlus member receives a 45-calendar-day return window.\n",
        document_id="DOC-B",
    )
    chunks = _chunks(doc_a, doc_b)
    conflicts = detect_conflicts(chunks, build_document_index(chunks))
    assert conflicts == []


def test_broadly_scoped_claim_still_conflicts_with_a_segment_scoped_one() -> None:
    """An unscoped (no segment named) claim is compared normally — only a
    *disjoint, both-non-empty* tag pair is excluded."""
    doc_a = _doc(
        "# Returns\n\n## Standard return window\n\n"
        "Customers on the standard plan may return within 30 calendar days.\n",
        document_id="DOC-A",
    )
    doc_b = _doc(_RETURN_BODY_45, document_id="DOC-B")  # no segment word at all
    chunks = _chunks(doc_a, doc_b)
    conflicts = detect_conflicts(chunks, build_document_index(chunks))
    assert len(conflicts) == 1
    assert conflicts[0].disposition is ConflictDisposition.GENUINE_ACTIVE_CONFLICT


def test_conflict_records_shared_topic_keywords() -> None:
    doc_a = _doc(_RETURN_BODY_30, document_id="DOC-A")
    doc_b = _doc(_RETURN_BODY_45, document_id="DOC-B")
    chunks = _chunks(doc_a, doc_b)
    conflicts = detect_conflicts(chunks, build_document_index(chunks))
    assert "return" in conflicts[0].shared_topic_keywords


def test_conflict_explanation_is_generated_from_structured_fields() -> None:
    doc_a = _doc(_RETURN_BODY_30, document_id="DOC-A")
    doc_b = _doc(_RETURN_BODY_45, document_id="DOC-B")
    chunks = _chunks(doc_a, doc_b)
    conflicts = detect_conflicts(chunks, build_document_index(chunks))
    assert "DOC-A" in conflicts[0].explanation
    assert "DOC-B" in conflicts[0].explanation
    assert "30" in conflicts[0].explanation
    assert "45" in conflicts[0].explanation


def test_no_conflict_when_no_shared_concept() -> None:
    doc_a = _doc(_RETURN_BODY_30, document_id="DOC-A")
    doc_b = _doc(
        "# Warranty\n\n## Periods\n\nBags carry a 2-year warranty.\n", document_id="DOC-B"
    )
    chunks = _chunks(doc_a, doc_b)
    assert detect_conflicts(chunks, build_document_index(chunks)) == []


def test_detect_conflicts_over_real_corpus_finds_expected_dispositions() -> None:
    """End-to-end (no embeddings needed — conflict detection is text/metadata
    only) against the real 14-document corpus, asserting the exact set of
    dispositions this corpus is designed to produce."""
    from aster_row_agent.rag.ingest import discover_documents, parse_corpus

    knowledge_base_dir = Path(__file__).resolve().parents[2] / "knowledge-base"
    paths = discover_documents(knowledge_base_dir)
    documents = parse_corpus(paths, knowledge_base_dir=knowledge_base_dir)
    all_chunks = [c for doc in documents for c in build_document_chunks(doc, max_chunk_chars=1200)]
    document_index = build_document_index(all_chunks)

    conflicts = detect_conflicts(all_chunks, document_index)
    dispositions = {c.disposition for c in conflicts}

    assert ConflictDisposition.GENUINE_ACTIVE_CONFLICT in dispositions
    assert ConflictDisposition.RESOLVED_BY_SUPERSESSION in dispositions
    assert ConflictDisposition.NON_AUTHORITATIVE_DISAGREEMENT in dispositions

    genuine = [c for c in conflicts if c.disposition is ConflictDisposition.GENUINE_ACTIVE_CONFLICT]
    assert len(genuine) == 1
    assert genuine[0].concept == "breeze_tumbler_body_dishwasher_safe"
    assert {genuine[0].claim_a.document_id, genuine[0].claim_b.document_id} == {
        "CARE-2026-01",
        "PROD-BREEZE-20",
    }
