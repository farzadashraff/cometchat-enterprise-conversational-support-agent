"""Typed domain models for the agent layer: session state, routing, and response.

Mirrors the split established in `rag/models.py` and `orders/models.py`:
pure data here, decision logic in `session.py`, `routing.py`,
`handoff.py`, `orchestration.py`. All models are frozen — an "update" is
always a new object (`model_copy`), never in-place mutation, matching the
immutable-model convention used everywhere else in this project.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from aster_row_agent.rag.retrieval_models import Citation

# --- session -----------------------------------------------------------------------

TurnRole = Literal["user", "agent"]


class Turn(BaseModel):
    """One message in a session's bounded recent history.

    `text` is truncated to a configured maximum length before storage
    (see `session.py`) — this is conversational context, not an
    unrestricted transcript store, and it never contains anything beyond
    what the user typed or the agent's own rendered answer (never a raw
    order record, never retrieved document text).
    """

    model_config = ConfigDict(frozen=True)

    role: TurnRole
    text: str
    timestamp: datetime


class Session(BaseModel):
    """Bounded, in-memory conversational state for one session.

    This is NOT long-term memory: `turns` is capped (see `session.py`),
    and only the minimum fields needed for the assignment's multi-turn
    cases are retained. Notably absent: any raw or sanitized order
    record, any retrieved document text, any customer PII — a session
    only ever stores an order *ID string* and a short topic *hint
    string*, never order data itself (INVARIANT 17).
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    turns: tuple[Turn, ...] = ()
    turn_count: int = 0
    """Monotonically increasing count of user turns handled in this
    session's lifetime — distinct from `len(turns)`, which is capped and
    drops the oldest entries. Used only for observability/response
    numbering, never for routing or security decisions."""
    last_order_id: str | None = None
    """Canonical (`ORD-####`) order ID, set only when the user explicitly
    supplied a well-formed order ID in an order-related turn — never
    guessed, never inherited from another session (INVARIANT 16)."""
    last_topic_hint: str | None = None
    """The previous turn's user message text, used only as a bounded,
    opt-in retrieval-query augmentation hint for genuinely ambiguous
    follow-ups (see `orchestration.py`) — never injected into the prompt
    as if it were new evidence."""
    created_at: datetime
    last_active_at: datetime


# --- routing -----------------------------------------------------------------------


class RouteKind(StrEnum):
    """The deterministic, top-level classification of one user message."""

    KNOWLEDGE = "knowledge"
    """A policy/product question — answered from the RAG evidence layer."""
    ORDER_LOOKUP = "order_lookup"
    """A pure order-status question — answered from the order layer."""
    ORDER_AND_KNOWLEDGE = "order_and_knowledge"
    """Both an order reference and a policy question (e.g. "Can I return
    ORD-1001?") — both layers are consulted."""
    SENSITIVE_REQUEST = "sensitive_request"
    """The user is directly asking to reveal the system prompt, hidden
    instructions, secrets, or another customer's data — refused
    deterministically, no retrieval, no LLM call at all."""
    AMBIGUOUS = "ambiguous"
    """No order reference and no clear policy-question signal — treated
    as a knowledge query and left to the evidence layer's own
    insufficient-evidence handling rather than guessed at further."""


OrderIdSource = Literal["message", "session"]


class RoutingDecision(BaseModel):
    """The deterministic routing decision for one turn."""

    model_config = ConfigDict(frozen=True)

    route_kind: RouteKind
    requests_unsupported_action: bool
    """True if the message asks the agent to perform an action this
    system cannot complete (cancel, refund, replace, change address, ...).
    Orthogonal to `route_kind` — a message can be ORDER_AND_KNOWLEDGE
    *and* request an unsupported action at the same time."""
    action_is_direct_command: bool
    """True when `requests_unsupported_action` is a direct command
    ("Cancel it right now.") rather than an eligibility question ("Can I
    cancel it?", answerable from policy). Used only by `prompts.py` to
    adjust tone (lead with "I can't process that directly" for a command
    vs. explaining eligibility first for a question) — deliberately NOT
    used to force a handoff by itself: the assignment's own
    `retrieved-prompt-injection` visible case phrases a request
    imperatively ("...approve my return") yet expects `handoff: false`
    once the real policy is correctly explained, so handoff is driven by
    evidence/order disposition (see `handoff.py`), not by command-vs-
    question phrasing. Always False when `requests_unsupported_action`
    is False."""
    order_id_candidate: str | None
    """The raw order-ID-shaped substring found in the message, if any,
    before Phase 4 normalization."""
    resolved_order_id: str | None
    """The canonical order ID to actually look up, if one could be
    resolved — from the message (normalized via
    `orders.normalize.normalize_order_id`) or from `Session.last_order_id`.
    None if order-related intent was detected but no ID is available from
    either source."""
    order_id_source: OrderIdSource | None
    retrieval_query: str | None
    """The text to send to the Phase 3 `EvidenceAssembler`, when
    `route_kind` implies a knowledge lookup. None otherwise."""


# --- response ------------------------------------------------------------------------


class ResponseDisposition(StrEnum):
    """The top-level shape of one agent response."""

    ANSWER = "answer"
    CLARIFICATION_REQUIRED = "clarification_required"
    HANDOFF_REQUIRED = "handoff_required"
    REFUSAL = "refusal"
    ERROR = "error"


class HandoffReason(StrEnum):
    """Why a turn was routed to human handoff, when it was."""

    SENSITIVE_DATA_REQUEST = "sensitive_data_request"
    AUTHORITATIVE_CONFLICT = "authoritative_conflict"
    NON_AUTHORITATIVE_ONLY = "non_authoritative_only"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ORDER_NOT_FOUND = "order_not_found"
    ORDER_DATASET_ERROR = "order_dataset_error"
    ORDER_REQUIRES_SUPPORT_REVIEW = "order_requires_support_review"
    UNSUPPORTED_ACTION_REQUESTED = "unsupported_action_requested"
    VALIDATION_FAILED = "validation_failed"
    LLM_UNAVAILABLE = "llm_unavailable"


class AgentResponse(BaseModel):
    """The complete, typed result of handling one user message.

    This is the only thing a caller (CLI, future API) ever needs — it
    carries the final answer text, the sources it is safe to render, and
    every disposition/handoff signal needed to present the turn
    correctly, all as structured data rather than something to parse out
    of the answer text.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    turn_index: int
    disposition: ResponseDisposition
    answer: str
    citable_sources: tuple[Citation, ...]
    handoff: bool
    handoff_reason: HandoffReason | None
    validation_passed: bool
    used_llm: bool
    """False for turns answered entirely by deterministic templates (e.g.
    missing order ID, insufficient evidence) — useful for observability
    and for tests that assert "no LLM call happened for this case"."""
