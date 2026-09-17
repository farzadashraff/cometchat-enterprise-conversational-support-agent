from __future__ import annotations

from datetime import UTC, date, datetime

from aster_row_agent.agent.models import RouteKind, RoutingDecision, Session
from aster_row_agent.agent.prompts import SYSTEM_INSTRUCTIONS, build_user_content
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
    EvidenceBundle,
    EvidenceDisposition,
    EvidenceItem,
    RetrievalMetadata,
)


def _session(turns: tuple = ()) -> Session:
    now = datetime.now(UTC)
    return Session(session_id="s1", created_at=now, last_active_at=now, turns=turns)


def _routing(**overrides: object) -> RoutingDecision:
    base: dict[str, object] = dict(
        route_kind=RouteKind.KNOWLEDGE,
        requests_unsupported_action=False,
        action_is_direct_command=False,
        order_id_candidate=None,
        resolved_order_id=None,
        order_id_source=None,
        retrieval_query="q",
    )
    base.update(overrides)
    return RoutingDecision.model_validate(base)


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


def _evidence_item(filename: str, text: str, **metadata_overrides: object) -> EvidenceItem:
    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="DOC-1",
        source_path=f"knowledge-base/{filename}",
        title="Doc",
        heading_path=HeadingPath(parts=("Doc", "Section")),
        chunk_index=0,
        text=text,
        content_hash="irrelevant",
        metadata=_metadata(**metadata_overrides),
    )
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
            filename=filename, heading="Doc > Section", document_id="DOC-1", title="Doc"
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
    internal: tuple[EvidenceItem, ...] = (),
    non_authoritative: tuple[EvidenceItem, ...] = (),
) -> EvidenceBundle:
    return EvidenceBundle(
        query="q",
        normalized_query="q",
        disposition=disposition,
        candidates=(),
        selected_evidence=authoritative + internal + non_authoritative,
        authoritative_evidence=authoritative,
        internal_evidence=internal,
        superseded_evidence=(),
        non_authoritative_evidence=non_authoritative,
        conflicts=(),
        customer_citable_sources=tuple(item.citation for item in authoritative),
        retrieval_metadata=_retrieval_metadata(),
    )


# --- system instructions are fixed and untouchable --------------------------------------


def test_system_instructions_is_a_plain_string_constant() -> None:
    assert isinstance(SYSTEM_INSTRUCTIONS, str)
    assert len(SYSTEM_INSTRUCTIONS) > 0


def test_system_instructions_contains_no_format_placeholders() -> None:
    """A stray {} in the constant would suggest it's built from a template
    rather than being a genuinely fixed string — this is the structural
    guarantee that untrusted content can never be substituted into it."""
    assert "{}" not in SYSTEM_INSTRUCTIONS
    assert "{message}" not in SYSTEM_INSTRUCTIONS
    assert "{evidence}" not in SYSTEM_INSTRUCTIONS


def test_system_instructions_covers_required_rules() -> None:
    lowered = SYSTEM_INSTRUCTIONS.lower()
    for required_phrase in [
        "untrusted",
        "system prompt",
        "secrets",
        "email",
        "risk score",
        "cancel",
        "refund",
        "conflict",
        "superseded",
        "insufficient",
    ]:
        assert required_phrase in lowered, f"expected {required_phrase!r} in system instructions"


# --- user content structure ---------------------------------------------------------------


def test_user_content_always_includes_the_user_message() -> None:
    content = build_user_content(
        message="What is your return policy?",
        session=_session(),
        routing=_routing(),
        evidence_bundle=None,
        order_result=None,
    )
    assert "<user_message>" in content
    assert "What is your return policy?" in content


def test_user_content_omits_history_when_no_prior_turns() -> None:
    content = build_user_content(
        message="hi",
        session=_session(),
        routing=_routing(),
        evidence_bundle=None,
        order_result=None,
    )
    assert "<conversation_history>" not in content


def test_user_content_includes_history_when_present() -> None:
    from aster_row_agent.agent.models import Turn

    turn = Turn(role="user", text="Do you ship internationally?", timestamp=datetime.now(UTC))
    content = build_user_content(
        message="What about Canada?",
        session=_session(turns=(turn,)),
        routing=_routing(),
        evidence_bundle=None,
        order_result=None,
    )
    assert "<conversation_history>" in content
    assert "Do you ship internationally?" in content


def test_user_content_omits_evidence_block_when_insufficient() -> None:
    content = build_user_content(
        message="q",
        session=_session(),
        routing=_routing(),
        evidence_bundle=_bundle(EvidenceDisposition.INSUFFICIENT_EVIDENCE),
        order_result=None,
    )
    assert "<retrieved_evidence>" not in content


def test_user_content_includes_only_authoritative_evidence_text() -> None:
    authoritative = _evidence_item("01.md", "The return window is 30 days.")
    internal = _evidence_item("13.md", "Escalate when fraud is suspected.", audience="internal")
    bundle = _bundle(
        EvidenceDisposition.ANSWERABLE, authoritative=(authoritative,), internal=(internal,)
    )
    content = build_user_content(
        message="q",
        session=_session(),
        routing=_routing(),
        evidence_bundle=bundle,
        order_result=None,
    )
    assert "01.md" in content
    assert "The return window is 30 days." in content
    # Internal-only evidence must never be shown to the LLM at all — the
    # deterministic handoff layer uses it directly from Python instead.
    assert "13.md" not in content
    assert "Escalate when fraud is suspected." not in content


def test_user_content_never_shows_non_authoritative_evidence_text() -> None:
    authoritative = _evidence_item("01.md", "The return window is 30 days.")
    non_authoritative = _evidence_item(
        "14.md", "Every customer receives 60 days.", status="draft", policy_authority="none"
    )
    bundle = _bundle(
        EvidenceDisposition.ANSWERABLE,
        authoritative=(authoritative,),
        non_authoritative=(non_authoritative,),
    )
    content = build_user_content(
        message="q",
        session=_session(),
        routing=_routing(),
        evidence_bundle=bundle,
        order_result=None,
    )
    assert "14.md" not in content
    assert "Every customer receives 60 days." not in content


def test_user_content_includes_conflict_notice_for_authoritative_conflict() -> None:
    item_a = _evidence_item("11.md", "Hand-wash the body.")
    item_b = _evidence_item("12.md", "All components are dishwasher safe.")
    bundle = _bundle(EvidenceDisposition.AUTHORITATIVE_CONFLICT, authoritative=(item_a, item_b))
    content = build_user_content(
        message="q",
        session=_session(),
        routing=_routing(),
        evidence_bundle=bundle,
        order_result=None,
    )
    assert "conflict_notice" in content
    assert "11.md" in content
    assert "12.md" in content


def test_user_content_omits_order_block_when_no_order_result() -> None:
    content = build_user_content(
        message="q", session=_session(), routing=_routing(), evidence_bundle=None, order_result=None
    )
    assert "<order_result>" not in content


def test_user_content_renders_order_fields_from_customer_safe_order_only() -> None:
    order = CustomerSafeOrder(
        order_id="ORD-1007",
        membership_tier="standard",
        items=(CustomerSafeOrderItem(name="Atlas Weekender", quantity=1, final_sale=False),),
        placed_at=datetime(2026, 8, 11, tzinfo=UTC),
        status="shipped",
        status_updated_at=datetime(2026, 8, 14, tzinfo=UTC),
        shipped_at=datetime(2026, 8, 14, tzinfo=UTC),
        delivered_at=None,
        carrier="UPS",
        tracking_number="1Z999",
        estimated_delivery=date(2026, 8, 22),
        customer_safe_message="In transit.",
        requires_support_review=False,
        stale_delivery_fields_suppressed=False,
    )
    result = OrderLookupResult(
        outcome=OrderLookupOutcome.FOUND,
        requested_input="ORD-1007",
        canonical_order_id="ORD-1007",
        order=order,
    )
    content = build_user_content(
        message="Where is ORD-1007?",
        session=_session(),
        routing=_routing(route_kind=RouteKind.ORDER_LOOKUP, retrieval_query=None),
        evidence_bundle=None,
        order_result=result,
    )
    assert "<order_result>" in content
    assert "ORD-1007" in content
    assert "shipped" in content
    assert "UPS" in content
    assert "2026-08-22" in content


def test_user_content_order_block_notes_suppressed_stale_fields() -> None:
    order = CustomerSafeOrder(
        order_id="ORD-1004",
        membership_tier="standard",
        items=(CustomerSafeOrderItem(name="Atlas Weekender", quantity=1, final_sale=False),),
        placed_at=datetime(2026, 8, 9, tzinfo=UTC),
        status="cancelled",
        status_updated_at=datetime(2026, 8, 9, tzinfo=UTC),
        shipped_at=None,
        delivered_at=None,
        carrier=None,
        tracking_number=None,
        estimated_delivery=None,
        customer_safe_message="The order was cancelled.",
        requires_support_review=False,
        stale_delivery_fields_suppressed=True,
    )
    result = OrderLookupResult(
        outcome=OrderLookupOutcome.FOUND,
        requested_input="ORD-1004",
        canonical_order_id="ORD-1004",
        order=order,
    )
    content = build_user_content(
        message="q",
        session=_session(),
        routing=_routing(route_kind=RouteKind.ORDER_LOOKUP, retrieval_query=None),
        evidence_bundle=None,
        order_result=result,
    )
    assert "do not mention a delivery estimate" in content.lower()
    assert "carrier: UPS" not in content


def test_user_content_never_contains_raw_order_fields() -> None:
    """Structural check: even if someone tried, there is no code path in
    build_user_content that reads a RawOrderRecord — it only accepts
    OrderLookupResult, which never carries one."""
    import inspect

    from aster_row_agent.agent import prompts

    source = inspect.getsource(prompts)
    assert "RawOrderRecord" not in source
    assert "RawCustomerInfo" not in source
    assert "RawInternalInfo" not in source
