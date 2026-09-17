from __future__ import annotations

from datetime import UTC, date, datetime

from aster_row_agent.agent.handoff import decide_pre_llm_handoff
from aster_row_agent.agent.models import HandoffReason, RouteKind, RoutingDecision
from aster_row_agent.orders.models import (
    CustomerSafeOrder,
    CustomerSafeOrderItem,
    OrderLookupOutcome,
    OrderLookupResult,
)
from aster_row_agent.rag.retrieval_models import (
    EvidenceBundle,
    EvidenceDisposition,
    RetrievalMetadata,
)


def _routing(**overrides: object) -> RoutingDecision:
    base: dict[str, object] = dict(
        route_kind=RouteKind.KNOWLEDGE,
        requests_unsupported_action=False,
        action_is_direct_command=False,
        order_id_candidate=None,
        resolved_order_id=None,
        order_id_source=None,
        retrieval_query="test query",
        reports_item_problem=False,
    )
    base.update(overrides)
    return RoutingDecision.model_validate(base)


def _retrieval_metadata() -> RetrievalMetadata:
    return RetrievalMetadata(
        total_scored=10,
        candidate_count=5,
        relevant_count=1,
        top_fused_score=0.9,
        min_semantic_score=0.6,
        min_lexical_score=2.0,
        relevance_margin=0.5,
        lexical_weight=0.5,
        semantic_weight=0.5,
        embedding_model="test-model",
    )


def _evidence_bundle(disposition: EvidenceDisposition) -> EvidenceBundle:
    return EvidenceBundle(
        query="q",
        normalized_query="q",
        disposition=disposition,
        candidates=(),
        selected_evidence=(),
        authoritative_evidence=(),
        internal_evidence=(),
        superseded_evidence=(),
        non_authoritative_evidence=(),
        conflicts=(),
        customer_citable_sources=(),
        retrieval_metadata=_retrieval_metadata(),
    )


def _order_result(
    outcome: OrderLookupOutcome, *, requires_support_review: bool = False
) -> OrderLookupResult:
    order = None
    if outcome is OrderLookupOutcome.FOUND:
        order = CustomerSafeOrder(
            order_id="ORD-1000",
            membership_tier="standard",
            items=(CustomerSafeOrderItem(name="Item", quantity=1, final_sale=False),),
            placed_at=datetime(2026, 8, 1, tzinfo=UTC),
            status="exception" if requires_support_review else "shipped",
            status_updated_at=datetime(2026, 8, 2, tzinfo=UTC),
            shipped_at=None,
            delivered_at=None,
            carrier=None,
            tracking_number=None,
            estimated_delivery=date(2026, 8, 22),
            customer_safe_message="msg",
            requires_support_review=requires_support_review,
            stale_delivery_fields_suppressed=False,
        )
    return OrderLookupResult(
        outcome=outcome,
        requested_input="ORD-1000",
        canonical_order_id="ORD-1000" if outcome != OrderLookupOutcome.MISSING_ORDER_ID else None,
        order=order,
    )


# --- priority rules ----------------------------------------------------------------------


def test_sensitive_request_always_forces_handoff() -> None:
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(route_kind=RouteKind.SENSITIVE_REQUEST),
        evidence_bundle=None,
        order_result=None,
    )
    assert handoff is True
    assert reason is HandoffReason.SENSITIVE_DATA_REQUEST


def test_authoritative_conflict_forces_handoff() -> None:
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(),
        evidence_bundle=_evidence_bundle(EvidenceDisposition.AUTHORITATIVE_CONFLICT),
        order_result=None,
    )
    assert handoff is True
    assert reason is HandoffReason.AUTHORITATIVE_CONFLICT


def test_non_authoritative_only_forces_handoff() -> None:
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(),
        evidence_bundle=_evidence_bundle(EvidenceDisposition.NON_AUTHORITATIVE_ONLY),
        order_result=None,
    )
    assert handoff is True
    assert reason is HandoffReason.NON_AUTHORITATIVE_ONLY


def test_answerable_evidence_does_not_force_handoff() -> None:
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(),
        evidence_bundle=_evidence_bundle(EvidenceDisposition.ANSWERABLE),
        order_result=None,
    )
    assert handoff is False
    assert reason is None


def test_reported_item_problem_forces_handoff_even_when_evidence_is_answerable() -> None:
    """BUG-004 (docs/architecture.md §22): a reported damaged/defective/
    wrong-item receipt must force handoff unconditionally, independent
    of evidence disposition — see `routing.py::_reports_item_problem`."""
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(reports_item_problem=True),
        evidence_bundle=_evidence_bundle(EvidenceDisposition.ANSWERABLE),
        order_result=None,
    )
    assert handoff is True
    assert reason is HandoffReason.ITEM_PROBLEM_REQUIRES_REVIEW


def test_sensitive_request_takes_priority_over_reported_item_problem() -> None:
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(route_kind=RouteKind.SENSITIVE_REQUEST, reports_item_problem=True),
        evidence_bundle=None,
        order_result=None,
    )
    assert handoff is True
    assert reason is HandoffReason.SENSITIVE_DATA_REQUEST


def test_insufficient_evidence_forces_handoff_when_no_order_fallback() -> None:
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(),
        evidence_bundle=_evidence_bundle(EvidenceDisposition.INSUFFICIENT_EVIDENCE),
        order_result=None,
    )
    assert handoff is True
    assert reason is HandoffReason.INSUFFICIENT_EVIDENCE


def test_insufficient_evidence_does_not_force_handoff_when_order_found() -> None:
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(route_kind=RouteKind.ORDER_AND_KNOWLEDGE),
        evidence_bundle=_evidence_bundle(EvidenceDisposition.INSUFFICIENT_EVIDENCE),
        order_result=_order_result(OrderLookupOutcome.FOUND),
    )
    assert handoff is False


def test_order_not_found_forces_handoff() -> None:
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(route_kind=RouteKind.ORDER_LOOKUP, retrieval_query=None),
        evidence_bundle=None,
        order_result=_order_result(OrderLookupOutcome.NOT_FOUND),
    )
    assert handoff is True
    assert reason is HandoffReason.ORDER_NOT_FOUND


def test_order_dataset_error_forces_handoff() -> None:
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(route_kind=RouteKind.ORDER_LOOKUP, retrieval_query=None),
        evidence_bundle=None,
        order_result=_order_result(OrderLookupOutcome.DATASET_ERROR),
    )
    assert handoff is True
    assert reason is HandoffReason.ORDER_DATASET_ERROR


def test_order_requiring_support_review_forces_handoff() -> None:
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(route_kind=RouteKind.ORDER_LOOKUP, retrieval_query=None),
        evidence_bundle=None,
        order_result=_order_result(OrderLookupOutcome.FOUND, requires_support_review=True),
    )
    assert handoff is True
    assert reason is HandoffReason.ORDER_REQUIRES_SUPPORT_REVIEW


def test_order_found_without_review_does_not_force_handoff() -> None:
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(route_kind=RouteKind.ORDER_LOOKUP, retrieval_query=None),
        evidence_bundle=None,
        order_result=_order_result(OrderLookupOutcome.FOUND, requires_support_review=False),
    )
    assert handoff is False
    assert reason is None


def test_no_signals_at_all_means_no_handoff() -> None:
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(), evidence_bundle=None, order_result=None
    )
    assert handoff is False
    assert reason is None


# --- priority ordering ---------------------------------------------------------------------


def test_sensitive_request_takes_priority_over_everything_else() -> None:
    """Even if evidence/order signals would also justify handoff, the
    reported reason should be the most specific one — sensitive request
    is checked first."""
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(route_kind=RouteKind.SENSITIVE_REQUEST),
        evidence_bundle=_evidence_bundle(EvidenceDisposition.AUTHORITATIVE_CONFLICT),
        order_result=_order_result(OrderLookupOutcome.NOT_FOUND),
    )
    assert reason is HandoffReason.SENSITIVE_DATA_REQUEST


def test_authoritative_conflict_takes_priority_over_order_not_found() -> None:
    handoff, reason = decide_pre_llm_handoff(
        routing=_routing(route_kind=RouteKind.ORDER_AND_KNOWLEDGE),
        evidence_bundle=_evidence_bundle(EvidenceDisposition.AUTHORITATIVE_CONFLICT),
        order_result=_order_result(OrderLookupOutcome.NOT_FOUND),
    )
    assert reason is HandoffReason.AUTHORITATIVE_CONFLICT
