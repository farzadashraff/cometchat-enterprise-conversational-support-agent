"""Deterministic handoff decisions.

Sourced from doc 13 (`support-escalation.md`, compiled into code per
Phase 1's architecture — see docs/architecture.md §7) and the Phase 3/4
disposition contracts. This is plain Python evaluated over typed inputs;
nothing here calls the LLM, and nothing the LLM produces can change the
outcome (`orchestration.py` computes this *before* generation, and only
adds a `VALIDATION_FAILED`/`LLM_UNAVAILABLE` reason afterward if the
generation step itself has a problem — see `orchestration.py`).

Priority order matters: the first matching rule wins, so a turn's
`handoff_reason` reflects the most specific/important cause rather than
an arbitrary one when several technically apply.
"""

from __future__ import annotations

from aster_row_agent.agent.models import HandoffReason, RouteKind, RoutingDecision
from aster_row_agent.orders.models import OrderLookupOutcome, OrderLookupResult
from aster_row_agent.rag.retrieval_models import EvidenceBundle, EvidenceDisposition


def decide_pre_llm_handoff(
    *,
    routing: RoutingDecision,
    evidence_bundle: EvidenceBundle | None,
    order_result: OrderLookupResult | None,
) -> tuple[bool, HandoffReason | None]:
    """The handoff decision knowable before (and independent of) generation."""
    if routing.route_kind is RouteKind.SENSITIVE_REQUEST:
        return True, HandoffReason.SENSITIVE_DATA_REQUEST

    if evidence_bundle is not None:
        if evidence_bundle.disposition is EvidenceDisposition.AUTHORITATIVE_CONFLICT:
            return True, HandoffReason.AUTHORITATIVE_CONFLICT
        if evidence_bundle.disposition is EvidenceDisposition.NON_AUTHORITATIVE_ONLY:
            return True, HandoffReason.NON_AUTHORITATIVE_ONLY

    if order_result is not None:
        if order_result.outcome is OrderLookupOutcome.DATASET_ERROR:
            return True, HandoffReason.ORDER_DATASET_ERROR
        if order_result.outcome is OrderLookupOutcome.NOT_FOUND:
            return True, HandoffReason.ORDER_NOT_FOUND
        if order_result.order is not None and order_result.order.requires_support_review:
            return True, HandoffReason.ORDER_REQUIRES_SUPPORT_REVIEW

    evidence_insufficient = (
        evidence_bundle is not None
        and evidence_bundle.disposition is EvidenceDisposition.INSUFFICIENT_EVIDENCE
    )
    if evidence_insufficient:
        order_found = order_result is not None and order_result.outcome is OrderLookupOutcome.FOUND
        if not order_found:
            return True, HandoffReason.INSUFFICIENT_EVIDENCE

    return False, None
