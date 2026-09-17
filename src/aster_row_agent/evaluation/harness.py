"""Runs one evaluation case against the real Agent pipeline.

Deliberately exercises the *application-level* interface
(`Agent.handle_message`), not internal functions in isolation — per the
Phase 7 brief, "exercise the application behavior, not merely test
individual internal functions." The evidence and order layers are the
real, production ones (real embeddings, real knowledge base, real
`data/orders.json`); only the LLM is a `FakeLLMClient` scripted per case
(see `scripts.py`), so the suite never needs a real API key and never
makes a network call (Phase 7 brief §23).

Two thin recording wrappers (`RecordingOrderLookupService`,
`RecordingEvidenceAssembler`) subclass the real Phase 3/4 classes purely
to observe what was called, with no behavior change — this gives the
evaluator a "trace" for tool-call and source-selection assertions without
inventing a new observability mechanism or reading anything the real
pipeline wouldn't have read anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aster_row_agent.agent.llm import FakeLLMClient, LLMRequest, LLMResponse
from aster_row_agent.agent.models import AgentResponse
from aster_row_agent.agent.orchestration import Agent
from aster_row_agent.agent.session import SessionStore
from aster_row_agent.orders.models import OrderLookupResult
from aster_row_agent.orders.repository import OrderRepository
from aster_row_agent.orders.service import OrderLookupService
from aster_row_agent.rag.evidence import EvidenceAssembler
from aster_row_agent.rag.retrieval import Retriever
from aster_row_agent.rag.retrieval_models import EvidenceAssemblyOptions, EvidenceBundle


class RecordingOrderLookupService(OrderLookupService):
    """Delegates to the real service; records every `lookup()` call's raw
    input and result so the evaluator can assert tool/tool_arguments
    without changing lookup behavior at all."""

    def __init__(self, repository: OrderRepository) -> None:
        super().__init__(repository)
        self.calls: list[str | None] = []
        self.results: list[OrderLookupResult] = []

    def lookup(self, raw_order_id: str | None) -> OrderLookupResult:
        result = super().lookup(raw_order_id)
        self.calls.append(raw_order_id)
        self.results.append(result)
        return result


class RecordingEvidenceAssembler(EvidenceAssembler):
    """Delegates to the real assembler; records every `EvidenceBundle`
    produced (including two-pass augmentation retries) so retrieval/
    authority/conflict assertions can inspect the real, structured
    decision rather than parsing an LLM's prose citation of it."""

    def __init__(self, retriever: Retriever, *, options: EvidenceAssemblyOptions | None = None):
        super().__init__(retriever, options=options)
        self.bundles: list[EvidenceBundle] = []

    def assemble(self, query: str) -> EvidenceBundle:
        bundle = super().assemble(query)
        self.bundles.append(bundle)
        return bundle


@dataclass
class TurnObservation:
    """Everything the evaluator can deterministically inspect about one turn."""

    message: str
    response: AgentResponse
    llm_request: LLMRequest | None
    order_calls: list[str | None]
    """Raw inputs passed to `OrderLookupService.lookup()` during this turn."""
    order_results: list[OrderLookupResult]
    evidence_bundles: list[EvidenceBundle]
    """Every `EvidenceBundle` assembled during this turn (0, 1, or 2 for a
    two-pass-augmented turn)."""


@dataclass
class CaseObservation:
    case_id: str
    turns: list[TurnObservation] = field(default_factory=list)

    @property
    def last(self) -> TurnObservation:
        return self.turns[-1]


def run_session(
    messages: list[str],
    *,
    evidence_assembler: EvidenceAssembler,
    order_repository: OrderRepository,
    scripted_responses: list[LLMResponse],
    session_id: str,
) -> list[TurnObservation]:
    """Replay `messages` against one fresh session of a real `Agent`,
    wired to real evidence/order layers and a `FakeLLMClient` scripted
    with `scripted_responses` (one entry per LLM call actually expected
    across the whole session — an unscripted extra call raises loudly,
    turning "the code called the LLM when it shouldn't have" into a
    directly visible evaluation failure rather than a silent pass)."""
    recording_orders = RecordingOrderLookupService(order_repository)
    llm = FakeLLMClient(list(scripted_responses))
    agent = Agent(
        session_store=SessionStore(),
        evidence_assembler=evidence_assembler,
        order_lookup_service=recording_orders,
        llm_client=llm,
    )

    observations: list[TurnObservation] = []
    for message in messages:
        order_calls_before = len(recording_orders.calls)
        llm_calls_before = len(llm.requests)
        bundles_before = (
            len(evidence_assembler.bundles)
            if isinstance(evidence_assembler, RecordingEvidenceAssembler)
            else 0
        )

        response = agent.handle_message(session_id, message)

        llm_request = (
            llm.requests[llm_calls_before] if len(llm.requests) > llm_calls_before else None
        )
        turn_bundles = (
            evidence_assembler.bundles[bundles_before:]
            if isinstance(evidence_assembler, RecordingEvidenceAssembler)
            else []
        )
        observations.append(
            TurnObservation(
                message=message,
                response=response,
                llm_request=llm_request,
                order_calls=recording_orders.calls[order_calls_before:],
                order_results=recording_orders.results[order_calls_before:],
                evidence_bundles=turn_bundles,
            )
        )
    return observations
