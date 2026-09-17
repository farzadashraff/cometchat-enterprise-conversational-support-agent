from __future__ import annotations

from datetime import UTC, date, datetime

from aster_row_agent.agent.llm import StructuredLLMOutput
from aster_row_agent.agent.validation import validate_response
from aster_row_agent.orders.models import (
    CustomerSafeOrder,
    CustomerSafeOrderItem,
    OrderLookupOutcome,
    OrderLookupResult,
)
from aster_row_agent.rag.authority import evaluate_chunk_authority
from aster_row_agent.rag.models import DocumentChunk, DocumentMetadata, HeadingPath
from aster_row_agent.rag.retrieval_models import (
    Citation,
    Conflict,
    ConflictDisposition,
    EvidenceBundle,
    EvidenceDisposition,
    EvidenceItem,
    ExtractedClaim,
    RetrievalMetadata,
)


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


def _chunk(chunk_id: str, filename: str = "01.md", heading: str = "Section") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="DOC-1",
        source_path=f"knowledge-base/{filename}",
        title="Doc",
        heading_path=HeadingPath(parts=("Doc", heading)),
        chunk_index=0,
        text="Some policy text.",
        content_hash="irrelevant",
        metadata=_metadata(),
    )


def _evidence_item(
    chunk_id: str, filename: str = "01.md", text: str = "Some policy text."
) -> EvidenceItem:
    chunk = _chunk(chunk_id, filename)
    authority = evaluate_chunk_authority(chunk, {"DOC-1": chunk.metadata})
    return EvidenceItem(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        title=chunk.title,
        source_filename=chunk.filename,
        heading_path=chunk.heading_path,
        text=text,
        fused_score=1.0,
        lexical_score=1.0,
        semantic_score=1.0,
        authority=authority,
        citation=Citation(
            filename=chunk.filename,
            heading=str(chunk.heading_path),
            document_id="DOC-1",
            title="Doc",
        ),
    )


def _retrieval_metadata() -> RetrievalMetadata:
    return RetrievalMetadata(
        total_scored=1,
        candidate_count=1,
        relevant_count=1,
        top_fused_score=1.0,
        min_semantic_score=0.6,
        min_lexical_score=2.0,
        relevance_margin=0.5,
        lexical_weight=0.5,
        semantic_weight=0.5,
        embedding_model="test-model",
    )


def _bundle(
    disposition: EvidenceDisposition,
    authoritative: tuple[EvidenceItem, ...] = (),
    conflicts: tuple[Conflict, ...] = (),
) -> EvidenceBundle:
    return EvidenceBundle(
        query="q",
        normalized_query="q",
        disposition=disposition,
        candidates=(),
        selected_evidence=authoritative,
        authoritative_evidence=authoritative,
        internal_evidence=(),
        superseded_evidence=(),
        non_authoritative_evidence=(),
        conflicts=conflicts,
        customer_citable_sources=tuple(item.citation for item in authoritative),
        retrieval_metadata=_retrieval_metadata(),
    )


def _order(
    status: str = "shipped",
    *,
    stale_suppressed: bool = False,
    eta: date | None = date(2026, 8, 22),
) -> OrderLookupResult:
    order = CustomerSafeOrder(
        order_id="ORD-1000",
        membership_tier="standard",
        items=(CustomerSafeOrderItem(name="Item", quantity=1, final_sale=False),),
        placed_at=datetime(2026, 8, 1, tzinfo=UTC),
        status=status,
        status_updated_at=datetime(2026, 8, 2, tzinfo=UTC),
        shipped_at=None,
        delivered_at=None,
        carrier=None,
        tracking_number=None,
        estimated_delivery=eta,
        customer_safe_message="msg",
        requires_support_review=(status == "exception"),
        stale_delivery_fields_suppressed=stale_suppressed,
    )
    return OrderLookupResult(
        outcome=OrderLookupOutcome.FOUND,
        requested_input="ORD-1000",
        canonical_order_id="ORD-1000",
        order=order,
    )


def _output(answer: str, cited: tuple[str, ...] = ()) -> StructuredLLMOutput:
    return StructuredLLMOutput(answer=answer, cited_filenames=cited)


def _conflict_pair() -> tuple[EvidenceItem, EvidenceItem, Conflict]:
    item_a = _evidence_item("chunk-a", "11.md")
    item_b = _evidence_item("chunk-b", "12.md")
    claim_a = ExtractedClaim(
        concept="c",
        value=True,
        chunk_id="chunk-a",
        document_id="DOC-1",
        applicability_tags=frozenset(),
        evidence_span="x",
    )
    claim_b = ExtractedClaim(
        concept="c",
        value=False,
        chunk_id="chunk-b",
        document_id="DOC-1",
        applicability_tags=frozenset(),
        evidence_span="y",
    )
    conflict = Conflict(
        concept="c",
        disposition=ConflictDisposition.GENUINE_ACTIVE_CONFLICT,
        claim_a=claim_a,
        claim_b=claim_b,
        shared_topic_keywords=(),
        explanation="conflict",
    )
    return item_a, item_b, conflict


# --- action-completion claims ------------------------------------------------------------


def test_valid_answer_with_no_issues_passes() -> None:
    result = validate_response(
        _output("The standard return window is 30 calendar days."),
        question="What is your return policy?",
        evidence_bundle=None,
        order_result=None,
    )
    assert result.passed is True
    assert result.issues == ()


def test_rejects_fake_refund_completed_claim() -> None:
    result = validate_response(
        _output("I've refunded your order."),
        question="q",
        evidence_bundle=None,
        order_result=None,
    )
    assert result.passed is False
    assert "action_completion_claim" in result.issue_codes


def test_rejects_fake_cancellation_claim() -> None:
    result = validate_response(
        _output("I have cancelled your order for you."),
        question="q",
        evidence_bundle=None,
        order_result=None,
    )
    assert result.passed is False
    assert "action_completion_claim" in result.issue_codes


def test_rejects_fake_replacement_issued_claim() -> None:
    result = validate_response(
        _output("Your replacement has been issued."),
        question="q",
        evidence_bundle=None,
        order_result=None,
    )
    assert result.passed is False


def test_rejects_fake_address_change_claim() -> None:
    result = validate_response(
        _output("I've updated your address."),
        question="q",
        evidence_bundle=None,
        order_result=None,
    )
    assert result.passed is False


def test_does_not_reject_order_status_report_that_happens_to_say_cancelled() -> None:
    """Reporting the order's own status ('your order is cancelled') is not
    an action-completion claim and must not be rejected."""
    result = validate_response(
        _output("Your order is cancelled and will not be shipped."),
        question="q",
        evidence_bundle=None,
        order_result=None,
    )
    assert result.passed is True


# --- order claims --------------------------------------------------------------------------


def test_rejects_stale_delivery_claim_for_cancelled_order() -> None:
    result = validate_response(
        _output("Your order is on its way and will arrive soon."),
        question="q",
        evidence_bundle=None,
        order_result=_order("cancelled", stale_suppressed=True, eta=None),
    )
    assert result.passed is False
    assert "stale_delivery_claim" in result.issue_codes


def test_rejects_fabricated_eta_when_none_available() -> None:
    result = validate_response(
        _output("It should arrive on August 25, 2026."),
        question="q",
        evidence_bundle=None,
        order_result=_order("shipped", eta=None),
    )
    assert result.passed is False
    assert "fabricated_eta" in result.issue_codes


def test_accepts_real_eta_when_available() -> None:
    result = validate_response(
        _output("It is estimated to arrive on 2026-08-22."),
        question="q",
        evidence_bundle=None,
        order_result=_order("shipped", eta=date(2026, 8, 22)),
    )
    assert result.passed is True


def test_no_order_claims_issue_when_order_not_found() -> None:
    not_found = OrderLookupResult(
        outcome=OrderLookupOutcome.NOT_FOUND,
        requested_input="ORD-9999",
        canonical_order_id="ORD-9999",
    )
    result = validate_response(
        _output("I couldn't find that order."),
        question="q",
        evidence_bundle=None,
        order_result=not_found,
    )
    assert result.passed is True


# --- internal-data leakage -----------------------------------------------------------------


def test_rejects_email_leak() -> None:
    result = validate_response(
        _output("Your order was sent to ava.morgan@example.test."),
        question="q",
        evidence_bundle=None,
        order_result=None,
    )
    assert result.passed is False
    assert "email_leak" in result.issue_codes


def test_rejects_risk_score_mention() -> None:
    result = validate_response(
        _output("Your risk score is 82."), question="q", evidence_bundle=None, order_result=None
    )
    assert result.passed is False
    assert "internal_data_leak" in result.issue_codes


def test_rejects_warehouse_note_mention() -> None:
    result = validate_response(
        _output("The warehouse note says fraud review cleared."),
        question="q",
        evidence_bundle=None,
        order_result=None,
    )
    assert result.passed is False


def test_rejects_system_prompt_echo() -> None:
    answer = (
        "You are the Aster & Row customer support assistant, and here is the "
        "untrusted data boundary..."
    )
    result = validate_response(
        _output(answer), question="q", evidence_bundle=None, order_result=None
    )
    assert result.passed is False
    assert "system_prompt_leak" in result.issue_codes


# --- citations --------------------------------------------------------------------------------


def test_rejects_invented_citation_not_in_evidence() -> None:
    bundle = _bundle(EvidenceDisposition.ANSWERABLE, authoritative=(_evidence_item("c1", "01.md"),))
    result = validate_response(
        _output("Here's the policy.", cited=("99-made-up-file.md",)),
        question="q",
        evidence_bundle=bundle,
        order_result=None,
    )
    assert result.passed is False
    assert "invented_citation" in result.issue_codes


def test_accepts_citation_that_is_in_evidence() -> None:
    bundle = _bundle(EvidenceDisposition.ANSWERABLE, authoritative=(_evidence_item("c1", "01.md"),))
    result = validate_response(
        _output("Here's the policy.", cited=("01.md",)),
        question="q",
        evidence_bundle=bundle,
        order_result=None,
    )
    assert result.passed is True


def test_no_evidence_bundle_means_any_citation_is_invented() -> None:
    result = validate_response(
        _output("Here's the policy.", cited=("01.md",)),
        question="q",
        evidence_bundle=None,
        order_result=None,
    )
    assert result.passed is False


# --- disposition consistency ------------------------------------------------------------------


def test_rejects_citation_under_insufficient_evidence_disposition() -> None:
    bundle = _bundle(EvidenceDisposition.INSUFFICIENT_EVIDENCE)
    result = validate_response(
        _output("Here's the answer.", cited=("01.md",)),
        question="q",
        evidence_bundle=bundle,
        order_result=None,
    )
    assert result.passed is False
    assert "citation_without_evidence" in result.issue_codes


def test_rejects_conflict_answer_citing_only_one_side() -> None:
    item_a, item_b, conflict = _conflict_pair()
    bundle = _bundle(
        EvidenceDisposition.AUTHORITATIVE_CONFLICT,
        authoritative=(item_a, item_b),
        conflicts=(conflict,),
    )
    result = validate_response(
        _output("Here's one side of it.", cited=("11.md",)),
        question="q",
        evidence_bundle=bundle,
        order_result=None,
    )
    assert result.passed is False
    assert "conflict_not_fully_cited" in result.issue_codes


def test_accepts_conflict_answer_citing_both_sides() -> None:
    item_a, item_b, conflict = _conflict_pair()
    bundle = _bundle(
        EvidenceDisposition.AUTHORITATIVE_CONFLICT,
        authoritative=(item_a, item_b),
        conflicts=(conflict,),
    )
    result = validate_response(
        _output("Both sources disagree.", cited=("11.md", "12.md")),
        question="q",
        evidence_bundle=bundle,
        order_result=None,
    )
    assert result.passed is True


# --- unsupported claim regression (vegan) ------------------------------------------------------


def test_rejects_confident_vegan_claim_when_evidence_never_mentions_it() -> None:
    evidence_text = "Spot-clean fabric bags with mild soap."
    bundle = _bundle(
        EvidenceDisposition.ANSWERABLE,
        authoritative=(_evidence_item("c1", "11.md", text=evidence_text),),
    )
    result = validate_response(
        _output("Yes, all our bags are vegan.", cited=("11.md",)),
        question="Are all fabrics and adhesives in your bags vegan?",
        evidence_bundle=bundle,
        order_result=None,
    )
    assert result.passed is False
    assert "unsupported_claim" in result.issue_codes


def test_accepts_hedged_response_to_vegan_question() -> None:
    evidence_text = "Spot-clean fabric bags with mild soap."
    bundle = _bundle(
        EvidenceDisposition.ANSWERABLE,
        authoritative=(_evidence_item("c1", "11.md", text=evidence_text),),
    )
    result = validate_response(
        _output("I don't have information confirming whether our materials are vegan."),
        question="Are all fabrics and adhesives in your bags vegan?",
        evidence_bundle=bundle,
        order_result=None,
    )
    assert result.passed is True


def test_does_not_flag_vegan_claim_when_evidence_actually_discusses_it() -> None:
    evidence_text = "All our bags are certified vegan materials."
    bundle = _bundle(
        EvidenceDisposition.ANSWERABLE,
        authoritative=(_evidence_item("c1", "11.md", text=evidence_text),),
    )
    result = validate_response(
        _output("Yes, all our bags are vegan.", cited=("11.md",)),
        question="Are your bags vegan?",
        evidence_bundle=bundle,
        order_result=None,
    )
    assert result.passed is True


def test_unrelated_question_is_not_affected_by_unsupported_claim_check() -> None:
    bundle = _bundle(EvidenceDisposition.ANSWERABLE, authoritative=(_evidence_item("c1", "01.md"),))
    result = validate_response(
        _output("The return window is 30 days.", cited=("01.md",)),
        question="What is your return policy?",
        evidence_bundle=bundle,
        order_result=None,
    )
    assert result.passed is True
