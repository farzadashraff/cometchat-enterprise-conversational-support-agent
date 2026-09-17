"""Applicability tagging: which customer/order segment a piece of text scopes to.

This does not implement general applicability *reasoning* (the corpus does
not give enough structure for that, and inventing rules it doesn't support
is explicitly out of scope — see the Phase 3 brief §8). It implements one
narrow, useful primitive: given a short span of text (a sentence an
`ExtractedClaim` was pulled from), which known segment keywords does it
mention? Two claims about the same concept whose applicability tags are
disjoint and both non-empty are describing *different* situations (e.g.
the standard 30-day window vs. the TrailPlus 45-day window) rather than
disagreeing about the same one — see `rag/conflict.py`.

Tags are computed per claim span, not per whole chunk: a chunk can state a
standard-scope number in one sentence while merely cross-referencing an
unrelated segment ("TrailPlus members receive a different return window")
in the next. Tagging the whole chunk would incorrectly scope the standard
number to both segments.
"""

from __future__ import annotations

import re

_APPLICABILITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "trailplus": re.compile(r"\btrailplus\b", re.IGNORECASE),
    "standard": re.compile(r"\bstandard\s+(?:plan|customers?)\b", re.IGNORECASE),
    "final_sale": re.compile(r"\bfinal[- ]sale\b", re.IGNORECASE),
    "domestic": re.compile(r"\bdomestic\b", re.IGNORECASE),
    "international": re.compile(r"\binternational\b", re.IGNORECASE),
    "canada": re.compile(r"\bcanad(?:a|ian)\b", re.IGNORECASE),
}

# A sentence explicitly claiming universal scope ("every item", "all
# customers") is describing a broader situation than any single named
# segment, even if it happens to *mention* a segment in passing (e.g. "...
# to return every item, including gift cards and final-sale merchandise" —
# the claim is about *every* item, not specifically final-sale ones). Such
# a sentence is treated as unscoped (empty tags) rather than narrowly
# scoped to whichever segment word it happens to contain, so it is not
# incorrectly excluded from comparison against a genuinely segment-scoped
# claim on the same concept.
_UNIVERSAL_SCOPE_RE = re.compile(
    r"\bevery\s+(?:item|customer|order)\b|\ball\s+(?:items|customers|orders)\b", re.IGNORECASE
)


def applicability_tags(text: str) -> frozenset[str]:
    """Which known segment keywords appear in `text`.

    An empty result means the text does not name a specific segment (and
    is treated as broadly scoped when compared against a tagged claim —
    see `rag/conflict.py`), not that applicability is unknown; genuine
    inability to determine scope is represented by the conflict layer's
    `UNCERTAIN` disposition, not by this function.
    """
    if _UNIVERSAL_SCOPE_RE.search(text):
        return frozenset()
    return frozenset(
        tag for tag, pattern in _APPLICABILITY_PATTERNS.items() if pattern.search(text)
    )
