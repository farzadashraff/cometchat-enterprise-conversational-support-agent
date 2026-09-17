"""The agent orchestrator: the single entry point that ties every layer together.

    user message
        -> session state            (session.py)
        -> deterministic routing    (routing.py)
        -> RAG and/or order lookup  (rag.evidence / orders.service — Phases 3-4, untouched)
        -> safety/disposition gate  (handoff.py)
        -> prompt construction      (prompts.py)
        -> LLM                      (llm.py)
        -> response validation      (validation.py)
        -> citation rendering       (this module)
        -> AgentResponse + session/trace update

Deterministic short-circuits (no LLM call at all) are used wherever the
correct response is already fully determined by typed data — sensitive
requests, missing/malformed order IDs, unknown orders, dataset errors,
and fully insufficient evidence. This is a direct application of the
project's stated precedence (deterministic logic > LLM generation): the
LLM is invoked only when there is something to synthesize in language
from evidence/order data that has already been selected and gated by
code.
"""

from __future__ import annotations

import logging
import re
import time

from aster_row_agent.agent.errors import LLMError, LLMMalformedResponseError
from aster_row_agent.agent.handoff import decide_pre_llm_handoff
from aster_row_agent.agent.llm import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    LLMClient,
    LLMRequest,
    StructuredLLMOutput,
    parse_structured_output,
)
from aster_row_agent.agent.models import (
    AgentResponse,
    HandoffReason,
    ResponseDisposition,
    RouteKind,
    RoutingDecision,
    Session,
)
from aster_row_agent.agent.prompts import SYSTEM_INSTRUCTIONS, build_user_content
from aster_row_agent.agent.routing import classify_message
from aster_row_agent.agent.session import SessionStore
from aster_row_agent.agent.validation import ValidationResult, validate_response
from aster_row_agent.logging_setup import log_event
from aster_row_agent.orders.models import OrderLookupOutcome, OrderLookupResult
from aster_row_agent.orders.service import OrderLookupService
from aster_row_agent.rag.evidence import EvidenceAssembler
from aster_row_agent.rag.retrieval_models import Citation, EvidenceBundle, EvidenceDisposition

logger = logging.getLogger(__name__)

# A genuinely ambiguous follow-up leans on an anaphor because it has no
# complete subject of its own ("what about Canada?", "does that include
# Canada?", "how about the same for TrailPlus?") — used only to decide
# whether the two-pass topic-hint augmentation below may run (BUG-003,
# see docs/architecture.md §20). Deliberately narrow and easy to extend
# rather than an attempt at general anaphora resolution.
_REFERENTIAL_FOLLOWUP_RE = re.compile(
    r"\b(that|this|those|it|same|either|what about|how about)\b", re.IGNORECASE
)

_SENSITIVE_REFUSAL_TEXT = (
    "I can't share system instructions, internal notes, or another customer's "
    "information. I'm happy to help with a policy question or your own order status."
)
_INSUFFICIENT_EVIDENCE_TEXT = (
    "I don't have enough information in our current documentation to answer that "
    "confidently. I'd recommend confirming with our support team."
)
_MISSING_ORDER_ID_TEXT = (
    "Could you share your order ID (for example, ORD-1007) so I can look that up?"
)
_MALFORMED_ORDER_ID_TEXT = (
    "That doesn't look like a valid order ID. Order IDs look like ORD-1007 — could "
    "you double-check and send it again?"
)
_ORDER_NOT_FOUND_TEXT = (
    "I couldn't find an order matching that ID. Please double-check the order ID, or "
    "our support team can help locate it."
)
_ORDER_DATASET_ERROR_TEXT = (
    "I'm having trouble accessing order information right now. Our support team can "
    "help you directly in the meantime."
)
_LLM_UNAVAILABLE_TEXT = (
    "I'm having trouble generating a response right now. Our support team can help "
    "you directly in the meantime."
)


class Agent:
    """Owns the wiring between session state, evidence, orders, and the LLM."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        evidence_assembler: EvidenceAssembler,
        order_lookup_service: OrderLookupService,
        llm_client: LLMClient,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._sessions = session_store
        self._evidence = evidence_assembler
        self._orders = order_lookup_service
        self._llm = llm_client
        self._model = model
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds

    def handle_message(self, session_id: str, message: str) -> AgentResponse:
        turn_start = time.monotonic()
        session_before = self._sessions.get_or_create(session_id)
        routing = classify_message(message, session_before)
        session = self._sessions.record_turn(session_id, "user", message)
        turn_index = session.turn_count

        log_event(
            logger,
            logging.INFO,
            "agent.turn.started",
            session_id=session_id,
            turn_index=turn_index,
            route_kind=routing.route_kind.value,
            requests_unsupported_action=routing.requests_unsupported_action,
            order_id_source=routing.order_id_source,
        )

        response = self._route_and_respond(session_id, turn_index, message, session, routing)

        self._sessions.record_turn(session_id, "agent", response.answer)
        log_event(
            logger,
            logging.INFO,
            "agent.turn.completed",
            session_id=session_id,
            turn_index=turn_index,
            disposition=response.disposition.value,
            handoff=response.handoff,
            handoff_reason=response.handoff_reason.value if response.handoff_reason else None,
            validation_passed=response.validation_passed,
            used_llm=response.used_llm,
            source_count=len(response.citable_sources),
            duration_ms=round((time.monotonic() - turn_start) * 1000, 2),
        )
        return response

    # --- routing / gathering ------------------------------------------------------------

    def _route_and_respond(
        self,
        session_id: str,
        turn_index: int,
        message: str,
        session: Session,
        routing: RoutingDecision,
    ) -> AgentResponse:
        if routing.route_kind is RouteKind.SENSITIVE_REQUEST:
            return self._respond(
                session_id,
                turn_index,
                disposition=ResponseDisposition.REFUSAL,
                answer=_SENSITIVE_REFUSAL_TEXT,
                citable_sources=(),
                handoff=True,
                handoff_reason=HandoffReason.SENSITIVE_DATA_REQUEST,
                validation_passed=True,
                used_llm=False,
            )

        evidence_bundle = self._maybe_assemble_evidence(routing, session)
        order_result, order_id_missing, order_id_malformed = self._maybe_lookup_order(routing)

        if routing.order_id_source == "message" and routing.resolved_order_id is not None:
            session = self._sessions.set_last_order_id(session_id, routing.resolved_order_id)
        if routing.retrieval_query is not None:
            session = self._sessions.set_last_topic_hint(session_id, message)

        # Deterministic short-circuits — no LLM call.
        if order_id_missing:
            return self._respond(
                session_id,
                turn_index,
                disposition=ResponseDisposition.CLARIFICATION_REQUIRED,
                answer=_MISSING_ORDER_ID_TEXT,
                citable_sources=(),
                handoff=False,
                handoff_reason=None,
                validation_passed=True,
                used_llm=False,
            )
        if order_id_malformed:
            return self._respond(
                session_id,
                turn_index,
                disposition=ResponseDisposition.CLARIFICATION_REQUIRED,
                answer=_MALFORMED_ORDER_ID_TEXT,
                citable_sources=(),
                handoff=False,
                handoff_reason=None,
                validation_passed=True,
                used_llm=False,
            )
        if order_result is not None and order_result.outcome is OrderLookupOutcome.NOT_FOUND:
            return self._respond(
                session_id,
                turn_index,
                disposition=ResponseDisposition.HANDOFF_REQUIRED,
                answer=_ORDER_NOT_FOUND_TEXT,
                citable_sources=(),
                handoff=True,
                handoff_reason=HandoffReason.ORDER_NOT_FOUND,
                validation_passed=True,
                used_llm=False,
            )
        if order_result is not None and order_result.outcome is OrderLookupOutcome.DATASET_ERROR:
            return self._respond(
                session_id,
                turn_index,
                disposition=ResponseDisposition.ERROR,
                answer=_ORDER_DATASET_ERROR_TEXT,
                citable_sources=(),
                handoff=True,
                handoff_reason=HandoffReason.ORDER_DATASET_ERROR,
                validation_passed=True,
                used_llm=False,
            )
        evidence_insufficient = (
            evidence_bundle is not None
            and evidence_bundle.disposition is EvidenceDisposition.INSUFFICIENT_EVIDENCE
        )
        if evidence_insufficient:
            order_found = (
                order_result is not None and order_result.outcome is OrderLookupOutcome.FOUND
            )
            if not order_found:
                return self._respond(
                    session_id,
                    turn_index,
                    disposition=ResponseDisposition.HANDOFF_REQUIRED,
                    answer=_INSUFFICIENT_EVIDENCE_TEXT,
                    citable_sources=(),
                    handoff=True,
                    handoff_reason=HandoffReason.INSUFFICIENT_EVIDENCE,
                    validation_passed=True,
                    used_llm=False,
                )

        # Everything else needs language synthesis from the (already
        # gated, already sanitized) evidence and/or order data.
        return self._generate_and_validate(
            session_id, turn_index, message, session, routing, evidence_bundle, order_result
        )

    def _maybe_assemble_evidence(
        self, routing: RoutingDecision, session: Session
    ) -> EvidenceBundle | None:
        if routing.retrieval_query is None:
            return None
        bundle = self._evidence.assemble(routing.retrieval_query)
        if bundle.disposition is not EvidenceDisposition.INSUFFICIENT_EVIDENCE:
            return bundle
        # Two-pass augmentation: only retried when the query alone found
        # nothing, and only using the *immediately preceding* user
        # message as a bounded hint — never the whole history — so a
        # strong standalone query is never diluted, and an unrelated new
        # topic never inherits stale context (see docs/architecture.md
        # §Phase 5, "topic contamination").
        #
        # Also restricted to genuinely short, fragment-like queries that
        # look like they refer back to something (BUG-003, docs/architecture.md
        # §20): evaluation found that a complete, self-contained, unrelated
        # question that legitimately has no evidence (e.g. "Which of your
        # products are vegan?") was being "rescued" into a false ANSWERABLE
        # disposition by augmenting it with a prior turn's unrelated topic
        # hint (e.g. "Do you ship internationally?") — silently
        # contaminating a question that should have triggered
        # abstention/handoff.
        #
        # A word-count-only cutoff was tried first and found insufficient
        # during Phase 8's final smoke test: "Which products are vegan?"
        # is only 4 words — short enough to pass a length-only threshold —
        # yet is just as unrelated and self-contained as its 6-word
        # phrasing. A genuine ambiguous follow-up is not just short, it is
        # *referential* ("what about Canada?", "does that include
        # Canada?") — it leans on an anaphor because it has no complete
        # subject of its own. Requiring both signals together is a
        # tighter, still-cheap discriminator that closes this gap without
        # guessing at semantic relatedness.
        is_short = len(routing.retrieval_query.split()) <= 5
        is_referential = bool(_REFERENTIAL_FOLLOWUP_RE.search(routing.retrieval_query))
        if session.last_topic_hint is None or not (is_short and is_referential):
            return bundle
        augmented_query = f"{session.last_topic_hint}\n{routing.retrieval_query}"
        augmented_bundle = self._evidence.assemble(augmented_query)
        return augmented_bundle

    def _maybe_lookup_order(
        self, routing: RoutingDecision
    ) -> tuple[OrderLookupResult | None, bool, bool]:
        if routing.route_kind not in (RouteKind.ORDER_LOOKUP, RouteKind.ORDER_AND_KNOWLEDGE):
            return None, False, False
        if routing.resolved_order_id is not None:
            return self._orders.lookup(routing.resolved_order_id), False, False
        # Only a pure order question hard-stops on a missing/malformed ID;
        # a mixed order+knowledge question still gets a knowledge answer
        # (see prompts.py's response-requirements rendering) with no
        # order-specific data attached.
        if routing.route_kind is not RouteKind.ORDER_LOOKUP:
            return None, False, False
        if routing.order_id_candidate is not None:
            return None, False, True  # a candidate was found but failed normalization
        return None, True, False

    # --- LLM path -----------------------------------------------------------------------

    def _generate_and_validate(
        self,
        session_id: str,
        turn_index: int,
        message: str,
        session: Session,
        routing: RoutingDecision,
        evidence_bundle: EvidenceBundle | None,
        order_result: OrderLookupResult | None,
    ) -> AgentResponse:
        user_content = build_user_content(
            message=message,
            session=session,
            routing=routing,
            evidence_bundle=evidence_bundle,
            order_result=order_result,
        )
        request = LLMRequest(
            model=self._model,
            system=SYSTEM_INSTRUCTIONS,
            user_content=user_content,
            max_tokens=self._max_tokens,
            timeout_seconds=self._timeout_seconds,
        )

        try:
            llm_response = self._llm.generate(request)
            output = parse_structured_output(llm_response)
        except LLMMalformedResponseError:
            return self._llm_fallback(session_id, turn_index)
        except LLMError as exc:
            log_event(logger, logging.WARNING, "agent.llm_error", error=type(exc).__name__)
            return self._llm_fallback(session_id, turn_index)

        validation = validate_response(
            output, question=message, evidence_bundle=evidence_bundle, order_result=order_result
        )
        pre_llm_handoff, pre_llm_reason = decide_pre_llm_handoff(
            routing=routing, evidence_bundle=evidence_bundle, order_result=order_result
        )

        if not validation.passed:
            return self._validation_failure_fallback(
                session_id, turn_index, validation, pre_llm_reason
            )

        citable_sources = self._resolve_citations(output, evidence_bundle)
        # A per-answer insufficiency (the model correctly hedged on a fact
        # the evidence never establishes) still requires handoff even when
        # the evidence *bundle* as a whole was answerable — bundle-level
        # and per-answer sufficiency are different questions (BUG-001, see
        # docs/architecture.md §20).
        handoff = pre_llm_handoff or validation.fact_sensitive_hedge
        handoff_reason = pre_llm_reason or (
            HandoffReason.INSUFFICIENT_EVIDENCE if validation.fact_sensitive_hedge else None
        )
        disposition = (
            ResponseDisposition.HANDOFF_REQUIRED if handoff else ResponseDisposition.ANSWER
        )

        return self._respond(
            session_id,
            turn_index,
            disposition=disposition,
            answer=_append_citations(output.answer, citable_sources),
            citable_sources=citable_sources,
            handoff=handoff,
            handoff_reason=handoff_reason,
            validation_passed=True,
            used_llm=True,
        )

    def _resolve_citations(
        self, output: StructuredLLMOutput, evidence_bundle: EvidenceBundle | None
    ) -> tuple[Citation, ...]:
        """Render citations from structured provenance only — the model's
        `cited_filenames` selects *which* of the already-vetted
        `customer_citable_sources` to show, it never supplies the
        filename/heading text itself (INVARIANT: citations are
        application-rendered, never LLM-invented)."""
        if evidence_bundle is None:
            return ()
        by_filename = {c.filename: c for c in evidence_bundle.customer_citable_sources}
        return tuple(by_filename[name] for name in output.cited_filenames if name in by_filename)

    def _validation_failure_fallback(
        self,
        session_id: str,
        turn_index: int,
        validation: ValidationResult,
        pre_llm_reason: HandoffReason | None,
    ) -> AgentResponse:
        log_event(
            logger,
            logging.WARNING,
            "agent.validation_failed",
            issue_codes=sorted(validation.issue_codes),
        )
        return self._respond(
            session_id,
            turn_index,
            disposition=ResponseDisposition.HANDOFF_REQUIRED,
            answer=_INSUFFICIENT_EVIDENCE_TEXT,
            citable_sources=(),
            handoff=True,
            handoff_reason=pre_llm_reason or HandoffReason.VALIDATION_FAILED,
            validation_passed=False,
            used_llm=True,
        )

    def _llm_fallback(self, session_id: str, turn_index: int) -> AgentResponse:
        return self._respond(
            session_id,
            turn_index,
            disposition=ResponseDisposition.ERROR,
            answer=_LLM_UNAVAILABLE_TEXT,
            citable_sources=(),
            handoff=True,
            handoff_reason=HandoffReason.LLM_UNAVAILABLE,
            validation_passed=True,
            used_llm=False,
        )

    def _respond(
        self,
        session_id: str,
        turn_index: int,
        *,
        disposition: ResponseDisposition,
        answer: str,
        citable_sources: tuple[Citation, ...],
        handoff: bool,
        handoff_reason: HandoffReason | None,
        validation_passed: bool,
        used_llm: bool,
    ) -> AgentResponse:
        return AgentResponse(
            session_id=session_id,
            turn_index=turn_index,
            disposition=disposition,
            answer=answer,
            citable_sources=citable_sources,
            handoff=handoff,
            handoff_reason=handoff_reason,
            validation_passed=validation_passed,
            used_llm=used_llm,
        )


def _append_citations(answer: str, citations: tuple[Citation, ...]) -> str:
    if not citations:
        return answer
    rendered = "\n".join(f"- {c.render()}" for c in citations)
    return f"{answer}\n\nSources:\n{rendered}"
