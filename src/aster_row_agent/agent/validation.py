"""Deterministic response validation.

This is an initial validator, not a perfect semantic fact-checker (per
the Phase 5 brief). It runs after LLM generation and before a response
is returned, checking the model's structured output against the same
typed inputs it was given — never trusting the model's own account of
what it did.

On any failure, `orchestration.py` discards the LLM's answer entirely and
substitutes a deterministic fallback (never a "repaired" version of the
model's text — repairing free text reliably is itself an unsolved
problem, and a wrong repair is worse than an honest fallback).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from aster_row_agent.agent.llm import StructuredLLMOutput
from aster_row_agent.orders.models import OrderLookupOutcome, OrderLookupResult
from aster_row_agent.rag.retrieval_models import (
    ConflictDisposition,
    EvidenceBundle,
    EvidenceDisposition,
)


class ValidationIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    detail: str


class ValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    issues: tuple[ValidationIssue, ...]
    fact_sensitive_hedge: bool = False
    """True when the answer correctly hedged on a fact-sensitive term (e.g.
    "vegan") that the evidence never establishes, for a question whose
    evidence bundle was otherwise ANSWERABLE (see `_fact_sensitive_hedge_detected`).
    This is not a validation *issue* — the answer is honest and not wrong —
    but it signals a per-answer insufficiency that `orchestration.py` uses
    to force a human-handoff, independent of the evidence bundle's own
    disposition. Found via Phase 7 evaluation: see docs/architecture.md
    §20, BUG-001."""

    @property
    def issue_codes(self) -> frozenset[str]:
        return frozenset(issue.code for issue in self.issues)


# --- B. action-completion claims -----------------------------------------------------
#
# No action tool exists anywhere in this system (Phase 4 is lookup-only).
# These patterns catch first-person/agent-voice claims that an action was
# completed or is underway — not the order's own *status* being reported
# (e.g. "your order is cancelled" reporting fact is fine; "I've cancelled
# your order" claiming the agent just did it is not).
_ACTION_COMPLETION_PATTERNS = (
    re.compile(
        r"\bI(?:'ve| have)?\s*(?:just\s+)?(cancel+ed|refunded|replaced|reshipped)\b", re.IGNORECASE
    ),
    re.compile(
        r"\bwe(?:'ve| have)?\s*(?:just\s+)?(cancel+ed|refunded|replaced|reshipped)\b", re.IGNORECASE
    ),
    re.compile(
        r"\byour (refund|replacement|cancellation) (has been|is) (issued|processed|complete)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(has been|is being) (cancelled|canceled|refunded|replaced) "
        r"(per your request|for you|as requested)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bI(?:'ve| have)?\s*(?:just\s+)?(updated|changed) your (address|shipping address)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\byour address (has been|is now) (updated|changed)\b", re.IGNORECASE),
    re.compile(
        r"\bI(?:'ve| have)?\s*(?:just\s+)?issued\b.{0,15}\b(coupon|refund|credit)\b", re.IGNORECASE
    ),
)


def _find_action_completion_claim(answer: str) -> str | None:
    for pattern in _ACTION_COMPLETION_PATTERNS:
        match = pattern.search(answer)
        if match:
            return match.group(0)
    return None


# --- C. order claims -------------------------------------------------------------------

_STILL_ARRIVING_RE = re.compile(
    r"\b(on its way|arriving|will arrive|is being delivered|out for delivery)\b", re.IGNORECASE
)
_MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
_DATE_LIKE_RE = re.compile(
    rf"\b(?:{_MONTH_NAMES})\s+\d{{1,2}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b", re.IGNORECASE
)


def _order_claims_issue(
    answer: str, order_result: OrderLookupResult | None
) -> ValidationIssue | None:
    if order_result is None or order_result.outcome is not OrderLookupOutcome.FOUND:
        return None
    order = order_result.order
    assert order is not None

    if order.stale_delivery_fields_suppressed and _STILL_ARRIVING_RE.search(answer):
        return ValidationIssue(
            code="stale_delivery_claim",
            detail=f"answer implies order {order.order_id} is still arriving "
            f"despite status={order.status}",
        )
    if order.estimated_delivery is None and _DATE_LIKE_RE.search(answer):
        return ValidationIssue(
            code="fabricated_eta",
            detail=f"answer contains a date-like token but order {order.order_id} "
            "has no estimated_delivery",
        )
    return None


# --- E. internal-data leakage ------------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_INTERNAL_PHRASE_RE = re.compile(
    r"\b(risk score|warehouse note|internal note|support tag|internal-only)\b", re.IGNORECASE
)
_SYSTEM_PROMPT_LEAK_RE = re.compile(
    r"\b(you are the aster ?& ?row customer support assistant|untrusted data boundary)\b",
    re.IGNORECASE,
)


def _internal_leakage_issue(answer: str) -> ValidationIssue | None:
    if _EMAIL_RE.search(answer):
        return ValidationIssue(code="email_leak", detail="answer contains an email-like string")
    if _INTERNAL_PHRASE_RE.search(answer):
        return ValidationIssue(
            code="internal_data_leak", detail="answer references an internal-only field by name"
        )
    if _SYSTEM_PROMPT_LEAK_RE.search(answer):
        return ValidationIssue(
            code="system_prompt_leak", detail="answer echoes system-instruction text"
        )
    return None


# --- D. citations ------------------------------------------------------------------------


def _citation_issue(
    cited_filenames: tuple[str, ...], evidence_bundle: EvidenceBundle | None
) -> ValidationIssue | None:
    allowed: set[str] = (
        {c.filename for c in evidence_bundle.customer_citable_sources}
        if evidence_bundle is not None
        else set()
    )
    invented = [name for name in cited_filenames if name not in allowed]
    if invented:
        return ValidationIssue(
            code="invented_citation", detail=f"cited filename(s) not in evidence: {invented}"
        )
    return None


# --- A. disposition consistency -----------------------------------------------------------


def _filename_for_chunk(evidence_bundle: EvidenceBundle, chunk_id: str) -> str | None:
    for item in evidence_bundle.authoritative_evidence:
        if item.chunk_id == chunk_id:
            return item.source_filename
    return None


def _disposition_issue(
    cited_filenames: tuple[str, ...], evidence_bundle: EvidenceBundle | None
) -> ValidationIssue | None:
    if evidence_bundle is None:
        return None
    if evidence_bundle.disposition is EvidenceDisposition.INSUFFICIENT_EVIDENCE and cited_filenames:
        return ValidationIssue(
            code="citation_without_evidence",
            detail="cited a source despite insufficient-evidence disposition",
        )
    if evidence_bundle.disposition is EvidenceDisposition.AUTHORITATIVE_CONFLICT:
        genuine = [
            c
            for c in evidence_bundle.conflicts
            if c.disposition is ConflictDisposition.GENUINE_ACTIVE_CONFLICT
        ]
        for conflict in genuine:
            required = {
                _filename_for_chunk(evidence_bundle, conflict.claim_a.chunk_id),
                _filename_for_chunk(evidence_bundle, conflict.claim_b.chunk_id),
            }
            required.discard(None)
            if not required.issubset(set(cited_filenames)):
                return ValidationIssue(
                    code="conflict_not_fully_cited",
                    detail="an authoritative conflict must cite both conflicting sources",
                )
    return None


# --- F. unsupported-claim regression guard (retrieval relevance != evidence sufficiency) --
#
# Explicitly NOT solved by keyword matching alone: the primary defense is
# architectural — an ANSWERABLE disposition already requires genuinely
# relevant, authoritative evidence (see rag/evidence.py), and a fully
# unrelated question is rejected upstream as INSUFFICIENT_EVIDENCE before
# the LLM is even called (see orchestration.py). This check is a narrow,
# supplementary guard for the specific gap Phase 3 identified and
# documented as a known limitation (docs/architecture.md §16, "risks
# discovered"): a question can be topically on-target (evidence is
# genuinely about the right product/policy area) while asking about one
# specific fact that evidence never actually states. A small, curated set
# of fact-sensitive terms is checked for *presence in the evidence text*
# as a necessary condition for a confident answer — this is one signal
# among several in a disposition-driven architecture, not the system's
# only defense, and is documented here as an explicit, extensible list
# rather than a hidden heuristic.
_FACT_SENSITIVE_TERMS = (
    "vegan",
    "organic",
    "sustainable",
    "recycled",
    "cruelty-free",
    "cruelty free",
    "hypoallergenic",
    "certified",
    "gluten-free",
    "carbon neutral",
)
_HEDGING_RE = re.compile(
    r"\b(don'?t have|do not have|cannot confirm|can'?t confirm|unable to confirm|"
    r"not specified|insufficient information|no information)\b",
    re.IGNORECASE,
)
_CONFIDENT_AFFIRMATIVE_RE = re.compile(
    r"\b(yes[,.]|are (all )?(vegan|organic|sustainable|certified|recycled)|"
    r"is (vegan|organic|sustainable|certified))\b",
    re.IGNORECASE,
)


def _unestablished_fact_sensitive_terms(
    question: str, evidence_bundle: EvidenceBundle | None
) -> list[str]:
    """Fact-sensitive terms the question asks about that the evidence
    bundle's authoritative text never actually establishes. Shared by
    `_unsupported_claim_issue` (the confident-claim guard) and
    `_fact_sensitive_hedge_detected` (the handoff signal below) so the two
    checks can never disagree about which terms are unestablished."""
    if evidence_bundle is None:
        return []
    question_lower = question.lower()
    mentioned_terms = [term for term in _FACT_SENSITIVE_TERMS if term in question_lower]
    if not mentioned_terms:
        return []
    evidence_text = " ".join(item.text for item in evidence_bundle.authoritative_evidence).lower()
    return [term for term in mentioned_terms if term not in evidence_text]


def _unsupported_claim_issue(
    question: str, answer: str, evidence_bundle: EvidenceBundle | None
) -> ValidationIssue | None:
    unestablished_terms = _unestablished_fact_sensitive_terms(question, evidence_bundle)
    if not unestablished_terms:
        return None  # no fact-sensitive term asked about, or evidence does discuss it

    if _HEDGING_RE.search(answer):
        return None  # the model already correctly hedged/abstained

    if _CONFIDENT_AFFIRMATIVE_RE.search(answer):
        return ValidationIssue(
            code="unsupported_claim",
            detail=f"answer confidently addresses {unestablished_terms} "
            "which the evidence never establishes",
        )
    return None


def _fact_sensitive_hedge_detected(
    question: str, answer: str, evidence_bundle: EvidenceBundle | None
) -> bool:
    """True when the model correctly hedged on a fact-sensitive term the
    evidence never establishes — the answer is fine, but per the
    assignment's "recommend human assistance when...the data is
    insufficient" rule, this specific fact still warrants a handoff even
    though the evidence bundle as a whole was ANSWERABLE (see
    docs/architecture.md §20, BUG-001: an evidence bundle can be relevant
    to the general topic while never establishing the one fact asked
    about — bundle-level ANSWERABLE and per-answer insufficiency are
    different questions)."""
    unestablished_terms = _unestablished_fact_sensitive_terms(question, evidence_bundle)
    if not unestablished_terms:
        return False
    return bool(_HEDGING_RE.search(answer))


# --- entry point ---------------------------------------------------------------------------


def validate_response(
    output: StructuredLLMOutput,
    *,
    question: str,
    evidence_bundle: EvidenceBundle | None,
    order_result: OrderLookupResult | None,
) -> ValidationResult:
    issues: list[ValidationIssue] = []

    action_claim = _find_action_completion_claim(output.answer)
    if action_claim:
        issues.append(
            ValidationIssue(code="action_completion_claim", detail=f"matched: {action_claim!r}")
        )

    order_issue = _order_claims_issue(output.answer, order_result)
    if order_issue:
        issues.append(order_issue)

    leakage_issue = _internal_leakage_issue(output.answer)
    if leakage_issue:
        issues.append(leakage_issue)

    citation_issue = _citation_issue(output.cited_filenames, evidence_bundle)
    if citation_issue:
        issues.append(citation_issue)

    disposition_issue = _disposition_issue(output.cited_filenames, evidence_bundle)
    if disposition_issue:
        issues.append(disposition_issue)

    unsupported_issue = _unsupported_claim_issue(question, output.answer, evidence_bundle)
    if unsupported_issue:
        issues.append(unsupported_issue)

    hedge_detected = _fact_sensitive_hedge_detected(question, output.answer, evidence_bundle)

    return ValidationResult(
        passed=not issues, issues=tuple(issues), fact_sensitive_hedge=hedge_detected
    )
