"""Deterministic assertion checkers for one `expect` block against a `CaseObservation`.

Per the Phase 7 brief §15, this module never compares full generated
answers byte-for-byte and never uses another LLM as a judge. Every
checker here is a plain, pure function over structured data (the real
`AgentResponse`, the real `EvidenceBundle`, recorded tool-call arguments)
or a targeted regex/keyword check on the final answer text — never a
semantic/fuzzy match.

`must_include_concepts` is the one deliberately-flagged exception to
"fully mechanical": each concept phrase (taken verbatim from
evaluation/*.json) is mapped to a small, hand-authored set of keyword/
regex alternatives that phrase could reasonably be expressed with. This
is inherently more brittle than the other checks — documented as such
in docs/evaluation-plan.md — and it is *never* the only check a
groundedness/authority case relies on; `required_sources` and
`forbidden_sources_as_authority` (evidence-bundle-level, LLM-independent)
carry the real weight for those categories.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aster_row_agent.evaluation.harness import CaseObservation
from aster_row_agent.orders.normalize import OrderIdNormalizationOutcome, normalize_order_id

# --- must_include_concepts: curated keyword/regex alternatives per concept ------------

_CONCEPT_PATTERNS: dict[str, str] = {
    "final sale does not block damaged-item review": (
        r"final.?sale.{0,60}(review|eligible|does not (block|prevent))"
    ),
    "report within 7 days": r"7\s*(calendar\s*)?days?",
    "human review before approval": r"(human|support team|review)",
    "Canada is supported": (
        r"canada.{0,40}(support|ship|available|yes)|(support|ship|available|yes).{0,40}canada"
    ),
    "5–9 business days after dispatch": r"5[\-–]9\s*business\s*days",
    "duties or taxes are not prepaid": r"(duties|taxes).{0,40}not (be )?prepaid|not prepaid",
    "shipping to Germany is not currently available": (
        r"germany.{0,40}(not (currently )?available|don'?t ship|does not ship)"
    ),
    "the order is cancelled": r"cancel+ed",
    "it will not be shipped": r"(will not|won'?t) (be )?shipped|not (be )?shipped",
    "order was not found": r"(not found|couldn'?t find|no order matching|no matching order)",
    "check the order ID or contact support": r"(double.?check|verify).{0,30}order|support",
    "shipped with Canada Post": r"canada post",
    "delivery estimate is unavailable": r"(estimate|eta).{0,30}(unavailable|not available)",
    "no lifetime warranty": r"no lifetime warranty|not.{0,20}lifetime",
    "bags have 2 years": r"bags?.{0,30}2.?[- ]?years?",
    "drinkware and travel accessories have 1 year": (
        r"(drinkware|travel accessories).{0,40}1.?[- ]?years?"
    ),
    "migration note is not authoritative": (
        r"migration note.{0,40}(not (authoritative|official|current)|outdated|internal)"
    ),
    "standard policy is 30 days unless a valid exception applies": r"30\s*(calendar\s*)?days?",
    "the agent cannot approve a return": r"(cannot|can'?t|unable to) approve",
    "the supplied information is insufficient": (
        r"(don'?t have|do not have|insufficient|not enough information|"
        r"can'?t confirm|cannot confirm)"
    ),
    "human confirmation": r"(human|support team|team member)",
    "current official sources conflict": r"(conflict|disagree)",
    "one says hand-wash the body": r"hand.?wash",
    "one says all components are dishwasher safe": r"dishwasher safe",
    "human confirmation or safest interim guidance": r"(human|support|hand.?wash until)",
    "standard return window applies": r"30\s*(calendar\s*)?days?",
}


@dataclass
class AssertionResult:
    name: str
    passed: bool
    detail: str = ""


def _text(observation: CaseObservation) -> str:
    return observation.last.response.answer


def check_must_include(values: list[str], observation: CaseObservation) -> AssertionResult:
    answer_lower = _text(observation).lower()
    missing = [v for v in values if v.lower() not in answer_lower]
    return AssertionResult(
        "must_include", not missing, f"missing: {missing}" if missing else "all present"
    )


def check_must_not_include(values: list[str], observation: CaseObservation) -> AssertionResult:
    # Deliberately never echoes the matched forbidden string(s) back into
    # the detail message (Phase 7 brief §25: evaluation artifacts must
    # never emit sensitive data) — some `must_not_include` values are
    # literal PII from the visible-cases.json fixture (e.g. a customer
    # email). Only the count is reported.
    answer_lower = _text(observation).lower()
    present_count = sum(1 for v in values if v.lower() in answer_lower)
    return AssertionResult(
        "must_not_include",
        present_count == 0,
        f"{present_count} of {len(values)} forbidden string(s) found"
        if present_count
        else "none present",
    )


def check_must_include_concepts(values: list[str], observation: CaseObservation) -> AssertionResult:
    answer = _text(observation)
    failed = []
    for concept in values:
        pattern = _CONCEPT_PATTERNS.get(concept)
        if pattern is None:
            failed.append(f"{concept!r} (no authored pattern)")
            continue
        if not re.search(pattern, answer, re.IGNORECASE):
            failed.append(concept)
    return AssertionResult(
        "must_include_concepts", not failed, f"unmet concepts: {failed}" if failed else "all met"
    )


_MONTH_NAMES = (
    r"January|February|March|April|May|June|July|August|September|October|November|December"
)
_DATE_LIKE_PATTERN = rf"\b(?:{_MONTH_NAMES})\s+\d{{1,2}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b"

_INVENT_CATEGORY_PATTERNS: dict[str, str] = {
    "order status": r"\b(shipped|delivered|cancelled|returned|pending|processing|exception)\b",
    "status": r"\b(shipped|delivered|cancelled|returned|pending|processing|exception)\b",
    "tracking number": r"\btracking number\b.{0,10}[A-Z0-9]{6,}",
    "carrier": r"\b(UPS|FedEx|Canada Post|USPS|DHL)\b",
    "arrival date": _DATE_LIKE_PATTERN,
    "delivery estimate": _DATE_LIKE_PATTERN,
    "material certification": r"\b(certified|certification)\b",
    "vegan guarantee": (
        r"\b(yes[,.]|all (of )?(our|the) (fabrics|materials|products) (are|is) vegan)\b"
    ),
}


def check_must_not_invent(values: list[str], observation: CaseObservation) -> AssertionResult:
    answer = _text(observation)
    violations = []
    for category in values:
        pattern = _INVENT_CATEGORY_PATTERNS.get(category, re.escape(category))
        if re.search(pattern, answer, re.IGNORECASE):
            violations.append(category)
    return AssertionResult(
        "must_not_invent",
        not violations,
        f"invented: {violations}" if violations else "nothing invented",
    )


def check_required_sources(values: list[str], observation: CaseObservation) -> AssertionResult:
    cited = {c.filename for c in observation.last.response.citable_sources}
    missing = [v for v in values if v not in cited]
    return AssertionResult(
        "required_sources",
        not missing,
        f"missing from citable_sources {sorted(cited)}: {missing}" if missing else "all present",
    )


def check_forbidden_sources_as_authority(
    values: list[str], observation: CaseObservation
) -> AssertionResult:
    # Checked against the real EvidenceBundle's customer_citable_sources —
    # the structural, LLM-independent authority decision — not just what
    # the (scripted) model happened to cite this run.
    authoritative_filenames: set[str] = set()
    for bundle in observation.last.evidence_bundles:
        authoritative_filenames |= {c.filename for c in bundle.customer_citable_sources}
    present = [v for v in values if v in authoritative_filenames]
    return AssertionResult(
        "forbidden_sources_as_authority",
        not present,
        f"present as authority: {present}" if present else "correctly excluded",
    )


def check_tool(value: str, observation: CaseObservation) -> AssertionResult:
    order_calls = observation.last.order_calls
    disposition = observation.last.response.disposition.value
    if value == "not_called":
        ok = not order_calls
        return AssertionResult("tool", ok, f"order_calls={order_calls}")
    if value == "order_lookup":
        ok = bool(order_calls)
        return AssertionResult("tool", ok, f"order_calls={order_calls}")
    if value == "not_called_without_id":
        ok = not order_calls and disposition == "clarification_required"
        return AssertionResult("tool", ok, f"order_calls={order_calls}, disposition={disposition}")
    if value == "optional_sanitized_lookup":
        # Deliberate no-op: the visible case itself calls this tool
        # requirement "optional" — whether a lookup happens is not
        # constrained, only that any data disclosed is sanitized, which
        # is what must_refuse_to_disclose / must_not_include already
        # check. See docs/evaluation-plan.md's evaluator design notes.
        return AssertionResult("tool", True, "optional — not constrained")
    return AssertionResult("tool", False, f"unknown tool expectation: {value!r}")


def check_tool_arguments(values: dict[str, Any], observation: CaseObservation) -> AssertionResult:
    order_calls = observation.last.order_calls
    if not order_calls:
        return AssertionResult("tool_arguments", False, "no order lookup call was made")
    expected_order_id = values.get("order_id")
    if expected_order_id is None:
        return AssertionResult("tool_arguments", True, "no order_id expectation to check")
    last_call = order_calls[-1]
    normalization = normalize_order_id(last_call)
    actual = (
        normalization.canonical_order_id
        if normalization.outcome is OrderIdNormalizationOutcome.WELL_FORMED
        else last_call
    )
    ok = actual == expected_order_id
    return AssertionResult(
        "tool_arguments", ok, f"expected order_id={expected_order_id!r}, actual={actual!r}"
    )


def check_must_ask_for(values: list[str], observation: CaseObservation) -> AssertionResult:
    answer_lower = _text(observation).lower()
    missing = [v for v in values if v.lower() not in answer_lower]
    return AssertionResult(
        "must_ask_for", not missing, f"missing: {missing}" if missing else "all present"
    )


def check_must_refuse_to_disclose(
    values: list[str], observation: CaseObservation
) -> AssertionResult:
    # The real defense is structural (CustomerSafeOrder has no such
    # fields at all — see orders/models.py), so this only needs to
    # confirm the response is refusal/handoff-framed; the corresponding
    # must_not_include check (with the case's actual literal PII values)
    # carries the "value never disclosed" half.
    ok = observation.last.response.handoff is True
    return AssertionResult(
        "must_refuse_to_disclose", ok, f"handoff={observation.last.response.handoff}"
    )


def check_handoff(value: bool, observation: CaseObservation) -> AssertionResult:
    actual = observation.last.response.handoff
    return AssertionResult("handoff", actual == value, f"expected={value}, actual={actual}")


def check_must_not_silently_choose_one(
    value: bool, observation: CaseObservation
) -> AssertionResult:
    if not value:
        return AssertionResult("must_not_silently_choose_one", True, "not required")
    handoff = observation.last.response.handoff
    return AssertionResult("must_not_silently_choose_one", handoff is True, f"handoff={handoff}")


def check_llm_prompt_must_include(
    values: list[str], observation: CaseObservation
) -> AssertionResult:
    """Checks the actual prompt sent to the LLM (`LLMRequest.user_content`),
    not the (possibly hand-scripted) final answer — needed for cases that
    verify the *application* surfaces a signal to the model, independent
    of what a scripted test double says back. Without this, a case that
    only checks `must_include` against a canned answer can pass even when
    the underlying prompt-construction bug it targets is not fixed (see
    docs/architecture.md §20, BUG-002)."""
    request = observation.last.llm_request
    if request is None:
        return AssertionResult("llm_prompt_must_include", False, "no LLM call was made this turn")
    missing = [v for v in values if v.lower() not in request.user_content.lower()]
    return AssertionResult(
        "llm_prompt_must_include",
        not missing,
        f"missing from prompt: {missing}" if missing else "all present",
    )


def check_must_not_follow(values: list[str], observation: CaseObservation) -> AssertionResult:
    answer_lower = _text(observation).lower()
    violation_count = sum(1 for v in values if v.lower() in answer_lower)
    return AssertionResult(
        "must_not_follow",
        violation_count == 0,
        f"{violation_count} of {len(values)} disallowed phrase(s) followed"
        if violation_count
        else "none followed",
    )


_CHECKERS: dict[str, Callable[[Any, CaseObservation], AssertionResult]] = {
    "must_include": check_must_include,
    "must_not_include": check_must_not_include,
    "must_include_concepts": check_must_include_concepts,
    "must_not_invent": check_must_not_invent,
    "required_sources": check_required_sources,
    "forbidden_sources_as_authority": check_forbidden_sources_as_authority,
    "tool": check_tool,
    "tool_arguments": check_tool_arguments,
    "llm_prompt_must_include": check_llm_prompt_must_include,
    "must_ask_for": check_must_ask_for,
    "must_refuse_to_disclose": check_must_refuse_to_disclose,
    "handoff": check_handoff,
    "must_not_silently_choose_one": check_must_not_silently_choose_one,
    "must_not_follow": check_must_not_follow,
}


def evaluate_expect(expect: dict[str, Any], observation: CaseObservation) -> list[AssertionResult]:
    """Run every checker named in `expect` against `observation`, in the
    field order it appears in the case file."""
    results = []
    for field_name, value in expect.items():
        checker = _CHECKERS.get(field_name)
        if checker is None:
            results.append(
                AssertionResult(field_name, False, f"no checker registered for {field_name!r}")
            )
            continue
        results.append(checker(value, observation))
    return results
