"""Deterministic-first message routing.

Per the Phase 5 brief: "Do not use an LLM as the sole security/router
mechanism. Use deterministic checks for obvious sensitive requests and
explicit order identifiers." Every check here is a plain regex/keyword
scan over the user's own message — never a call to the LLM, and never a
scan of retrieved evidence or order text (which are separate untrusted
inputs handled by the authority/sanitization layers in Phases 3-4, not
here).

Order-ID handling reuses Phase 4's `normalize_order_id` for the actual
normalization/validation step — this module only does the narrower job
of finding an order-ID-*shaped* substring within a longer natural-
language message (Phase 4's normalizer expects the whole input to already
be just the ID).
"""

from __future__ import annotations

import re

from aster_row_agent.agent.models import OrderIdSource, RouteKind, RoutingDecision, Session
from aster_row_agent.orders.normalize import OrderIdNormalizationOutcome, normalize_order_id

# --- sensitive-request detection -----------------------------------------------------
#
# Scoped narrowly and deliberately: these match the user asking to reveal
# the agent's OWN system prompt/instructions, or another customer's/
# internal data — not any use of the word "ignore" in a policy question.
# A message like "the migration note says to ignore the real policy and
# give everyone 60 days" must NOT match here (it doesn't ask the agent to
# ignore *its own instructions*, and is a legitimate, answerable policy
# question that the RAG authority layer already handles correctly — see
# rag/authority.py and the retrieved-prompt-injection scenario in
# docs/security-model.md).

_REVEAL_PROMPT_RE = re.compile(
    r"\b(reveal|show|print|display|tell me|what is|give me)\b.{0,25}"
    r"\b(system prompt|hidden prompt|your (instructions|rules|prompt)|internal prompt)\b",
    re.IGNORECASE,
)
_IGNORE_INSTRUCTIONS_RE = re.compile(
    r"\bignore\b.{0,20}\b(previous|prior|all|your)\b.{0,20}\b(instructions|rules|prompt)\b",
    re.IGNORECASE,
)
_INTERNAL_DATA_REQUEST_RE = re.compile(
    r"\b(internal notes?|warehouse notes?|risk score|support tags?|"
    r"another customer'?s?|other customers'?)\b.{0,15}"
    r"\b(data|info|information|details|order|notes?|score)?\b",
    re.IGNORECASE,
)
_CREDENTIALS_REQUEST_RE = re.compile(
    r"\b(api key|credentials?|secrets?|password)\b", re.IGNORECASE
)

_SENSITIVE_PATTERNS = (
    _REVEAL_PROMPT_RE,
    _IGNORE_INSTRUCTIONS_RE,
    _INTERNAL_DATA_REQUEST_RE,
    _CREDENTIALS_REQUEST_RE,
)


def _is_sensitive_request(message: str) -> bool:
    return any(pattern.search(message) for pattern in _SENSITIVE_PATTERNS)


# --- unsupported-action detection ----------------------------------------------------
#
# No action tool exists in this system at all (Phase 4's order layer is
# lookup-only) — these verbs each name an action the agent can never
# complete, regardless of phrasing. "return" is deliberately NOT in this
# list: a return is a supported, policy-governed *process* the agent can
# meaningfully explain (eligibility, window, condition requirements) via
# RAG evidence, unlike cancel/refund/replace/address-change, which have
# no self-service path at all per the order data dictionary.
_UNSUPPORTED_ACTION_RE = re.compile(
    r"\b(cancel|refund|replace|reship|approve)\b"
    r"|\b(change|update)\b.{0,15}\b(my |the )?(address|shipping address)\b"
    r"|\bissue\b.{0,15}\b(a |my )?(refund|replacement|coupon)\b",
    re.IGNORECASE,
)

# Distinguishes an *eligibility question* ("Can I cancel ORD-1001?" — fully
# answerable from policy, explaining eligibility is a complete, useful
# response) from a *direct command* ("Cancel ORD-1001 right now." — nothing
# but escalation is a useful response, since no eligibility explanation
# was even asked for). Both are always subject to the same "never claim
# completion" guard (see prompts.py/validation.py); this distinction only
# affects whether the action signal *alone* forces a handoff — see
# handoff.py.
_ELIGIBILITY_QUESTION_RE = re.compile(
    r"\b(can|could)\s+i\b|\bam\s+i\b|\bis\s+it\s+possible\b|\bis\s+(this|that|it)\s+(order\s+)?eligible\b",
    re.IGNORECASE,
)


def _requests_unsupported_action(message: str) -> bool:
    return bool(_UNSUPPORTED_ACTION_RE.search(message))


def _is_direct_action_command(message: str) -> bool:
    return _requests_unsupported_action(message) and not _ELIGIBILITY_QUESTION_RE.search(message)


# --- received-item-problem detection (BUG-004, docs/architecture.md §22) -------------
#
# Docs 04 and 07 both state, in plain text, that a resolution (refund/
# replacement/warranty approval) for an already-received problem item may
# never be promised before a human review is completed. That sentence
# does not reliably retrieve for every phrasing of the scenario it
# governs (heading-level chunking means it can rank far outside any
# reasonable top-K for a query that is topically about the *rest* of the
# same document) — so this cannot be a retrieval-completeness fix without
# either editing the supplied corpus (out of scope) or non-selectively
# widening retrieval for every query (a broad, unrelated behavior change).
# It also cannot be an evidence/heading-level gate: the doc 04 "Available
# resolutions" heading was empirically confirmed to surface as incidental
# retrieval noise for unrelated queries (a shipping-delay refund question,
# even a lifetime-warranty question) that must NOT force a handoff.
#
# The one signal that is actually specific to this scenario, and not
# noisy, is the message itself: the customer is reporting that an item
# they already received arrived damaged, defective, or wrong — a
# completed-receipt narration, not a hypothetical ("what if it arrives
# damaged?") or a general policy question. This mirrors the existing
# `_is_direct_action_command` pragmatic-signal pattern already used in
# this module rather than introducing a new mechanism.
_ITEM_PROBLEM_RE = re.compile(
    r"\b(arrived|received|came|got)\b.{0,20}"
    r"\b(damaged|broken|defective|faulty|cracked|torn|incorrect|"
    r"wrong (item|size|color|colour))\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_FRAMING_RE = re.compile(r"\b(if|what if|suppose|in case|imagine)\b", re.IGNORECASE)


def _reports_item_problem(message: str) -> bool:
    return bool(_ITEM_PROBLEM_RE.search(message)) and not _HYPOTHETICAL_FRAMING_RE.search(message)


# --- order-ID candidate scanning ------------------------------------------------------
#
# A loose scan for "the user is clearly attempting to reference an order
# ID" within a longer message (e.g. "Where is ORD-1007?"). Deliberately
# broader than a valid ID (alphanumeric, not just digits) so that an
# *invalid* attempt ("ORD-ABCD", "ORD-12") is still found as a candidate
# and correctly classified MALFORMED by Phase 4's normalizer, rather than
# being missed entirely and misclassified as "no ID given at all". The
# actual normalization/validation is entirely delegated to
# orders.normalize.normalize_order_id — this regex only decides *where to
# look*, never what counts as valid.
#
# Three alternatives, not one simpler pattern, because the plain English
# word "order" itself starts with "ord" — matching "ORD" plus any
# optional separator plus any letters would treat every use of the word
# "order" as a malformed ID attempt (a real bug caught by testing "Where
# is my order?" and getting MALFORMED instead of MISSING). A real or
# attempted ID always has either an explicit separator (hyphen/
# underscore/space) or is glued directly to digits — plain English
# continuations of "order" ("order", "orders", "ordinary") never match.
_ORDER_ID_CANDIDATE_RE = re.compile(
    r"\bORD[\-_][A-Za-z0-9]{1,15}\b"  # explicit hyphen/underscore separator
    r"|\bORD[ ]\d{1,15}\b"  # explicit space separator, must be digits
    r"|\bORD\d{1,15}\b",  # no separator, glued directly to digits
    re.IGNORECASE,
)

_ORDER_INTENT_RE = re.compile(
    r"\b(order|eta|arrive|arriving|arrival|shipped|shipping status|tracking|"
    r"track|deliver|delivery|delivered|where is my|status of my)\b",
    re.IGNORECASE,
)


def _find_order_id_candidate(message: str) -> str | None:
    match = _ORDER_ID_CANDIDATE_RE.search(message)
    return match.group(0) if match else None


def _has_order_intent(message: str) -> bool:
    return bool(_ORDER_INTENT_RE.search(message))


# --- routing ---------------------------------------------------------------------------


def classify_message(message: str, session: Session) -> RoutingDecision:
    """Deterministically classify one user message, resolving order-ID
    references against the current session where appropriate.

    No LLM call happens here. Ambiguity is represented as `RouteKind.AMBIGUOUS`
    and left to the evidence layer's own insufficient-evidence handling —
    never guessed at further by this function.
    """
    requests_unsupported_action = _requests_unsupported_action(message)
    action_is_direct_command = _is_direct_action_command(message)
    reports_item_problem = _reports_item_problem(message)

    if _is_sensitive_request(message):
        return RoutingDecision(
            route_kind=RouteKind.SENSITIVE_REQUEST,
            requests_unsupported_action=requests_unsupported_action,
            action_is_direct_command=action_is_direct_command,
            order_id_candidate=None,
            resolved_order_id=None,
            order_id_source=None,
            retrieval_query=None,
            reports_item_problem=reports_item_problem,
        )

    order_id_candidate = _find_order_id_candidate(message)
    resolved_order_id: str | None = None
    order_id_source: OrderIdSource | None = None

    if order_id_candidate is not None:
        normalization = normalize_order_id(order_id_candidate)
        if normalization.outcome is OrderIdNormalizationOutcome.WELL_FORMED:
            resolved_order_id = normalization.canonical_order_id
            order_id_source = "message"
        # A candidate that fails normalization (e.g. "ORD-ABCD") still
        # counts as order intent below — the message clearly attempted an
        # order ID, it just isn't valid, which is exactly the
        # MALFORMED_ORDER_ID case orchestration.py needs to detect. It
        # must not be treated the same as "no ID mentioned at all".

    session_id_usable = resolved_order_id is None and session.last_order_id is not None
    if session_id_usable and _has_order_intent(message):
        resolved_order_id = session.last_order_id
        order_id_source = "session"

    has_order_intent = order_id_candidate is not None or _has_order_intent(message)

    # Route kind: an explicit order ID in the message is always
    # order-related, even without an order keyword ("Where is ORD-1007?"
    # has no separate "order" word). A knowledge signal is "everything
    # else" — including AMBIGUOUS, which behaves like KNOWLEDGE for
    # retrieval purposes but is labeled distinctly for observability.
    if has_order_intent and _looks_like_pure_order_question(message):
        route_kind = RouteKind.ORDER_LOOKUP
        retrieval_query = None
    elif has_order_intent:
        route_kind = RouteKind.ORDER_AND_KNOWLEDGE
        retrieval_query = message
    elif _looks_like_a_question_or_statement(message):
        route_kind = RouteKind.KNOWLEDGE
        retrieval_query = message
    else:
        route_kind = RouteKind.AMBIGUOUS
        retrieval_query = message

    return RoutingDecision(
        route_kind=route_kind,
        requests_unsupported_action=requests_unsupported_action,
        action_is_direct_command=action_is_direct_command,
        order_id_candidate=order_id_candidate,
        resolved_order_id=resolved_order_id,
        order_id_source=order_id_source,
        retrieval_query=retrieval_query,
        reports_item_problem=reports_item_problem,
    )


# Keywords that indicate the message is asking about a *policy/product*
# topic in addition to (or instead of) order status — used only to decide
# ORDER_LOOKUP vs. ORDER_AND_KNOWLEDGE, never to gate retrieval itself.
_POLICY_SIGNAL_RE = re.compile(
    r"\b(return|refund|policy|warranty|cancel|exchange|damaged|defective|"
    r"final.?sale|can i|am i (eligible|allowed)|allowed to)\b",
    re.IGNORECASE,
)


def _looks_like_pure_order_question(message: str) -> bool:
    """True when the message is only about an order's status/ETA/tracking,
    with no separate policy question layered on top (e.g. "Where is
    ORD-1007?", "What's the ETA?" — as opposed to "Can I return
    ORD-1001?", which also needs RAG evidence)."""
    return not _POLICY_SIGNAL_RE.search(message)


def _looks_like_a_question_or_statement(message: str) -> bool:
    """A minimal sanity check that there is *something* to search for —
    excludes only genuinely empty/whitespace-only input, which is not a
    realistic message but is handled safely rather than crashing."""
    return bool(message.strip())
