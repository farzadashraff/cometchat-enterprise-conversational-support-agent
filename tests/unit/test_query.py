from __future__ import annotations

from aster_row_agent.rag.query import normalize_query


def test_normalize_query_collapses_internal_whitespace() -> None:
    result = normalize_query("Do   you\tship\n\nto Canada?")
    assert result.normalized == "Do you ship to Canada?"


def test_normalize_query_strips_leading_and_trailing_whitespace() -> None:
    result = normalize_query("   What about Canada?   ")
    assert result.normalized == "What about Canada?"


def test_normalize_query_preserves_original_casing() -> None:
    """Case-folding is a per-method concern (BM25 lowercases internally,
    embeddings are case-insensitive by construction) — the shared
    normalized query must not destroy casing a future consumer might need."""
    result = normalize_query("ORD-1007 TrailPlus Canada")
    assert result.normalized == "ORD-1007 TrailPlus Canada"


def test_normalize_query_preserves_raw_field() -> None:
    raw = "  What   about Canada?  "
    result = normalize_query(raw)
    assert result.raw == raw


def test_normalize_query_applies_unicode_nfkc_normalization() -> None:
    # U+FF23 U+FF41 U+FF4E U+FF41 U+FF44 U+FF41 = fullwidth "Canada"
    fullwidth_canada = "Ｃａｎａｄａ"
    result = normalize_query(f"Do you ship to {fullwidth_canada}?")
    assert "Canada" in result.normalized


def test_normalize_query_does_not_invent_or_rewrite_words() -> None:
    """Conservative: normalization must not add, remove, or substitute words."""
    raw = "what about canada"
    result = normalize_query(raw)
    assert result.normalized.split() == raw.split()


def test_normalize_query_handles_empty_string() -> None:
    result = normalize_query("")
    assert result.normalized == ""


def test_normalize_query_is_deterministic() -> None:
    raw = "  Do you ship\tto Canada?  "
    assert normalize_query(raw).normalized == normalize_query(raw).normalized
