"""Retrieval-quality tests against the real corpus and real embedding model.

These deliberately do NOT use `FakeDeterministicEmbeddingProvider` — it has
no notion of semantic similarity, so it cannot meaningfully test whether a
paraphrase actually ranks the right document. The `real_retriever` fixture
(see tests/conftest.py) loads the local embedding model and embeds the
real 14-document corpus exactly once per test session.
"""

from __future__ import annotations

from aster_row_agent.rag.embeddings import FakeDeterministicEmbeddingProvider
from aster_row_agent.rag.models import IndexRecord
from aster_row_agent.rag.retrieval import Retriever
from aster_row_agent.rag.retrieval_models import RetrievalOptions


def test_retrieve_returns_at_most_top_k_candidates(real_retriever: Retriever) -> None:
    result = real_retriever.retrieve("How long is the return window?", RetrievalOptions(top_k=5))
    assert len(result.candidates) == 5


def test_retrieve_never_returns_more_than_corpus_size(real_retriever: Retriever) -> None:
    result = real_retriever.retrieve("returns", RetrievalOptions(top_k=1000))
    assert len(result.candidates) == len(real_retriever)


def test_retrieve_candidates_are_sorted_by_fused_score_descending(
    real_retriever: Retriever,
) -> None:
    result = real_retriever.retrieve("Can I put the Breeze Tumbler in the dishwasher?")
    scores = [c.fused_score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_candidate_rank_matches_position(real_retriever: Retriever) -> None:
    result = real_retriever.retrieve("What is your return policy?")
    assert [c.rank for c in result.candidates] == list(range(len(result.candidates)))


def test_retrieve_preserves_full_provenance_on_every_candidate(real_retriever: Retriever) -> None:
    result = real_retriever.retrieve("How long do I have to return a backpack?")
    for candidate in result.candidates:
        chunk = candidate.chunk
        assert chunk.chunk_id
        assert chunk.document_id
        assert chunk.filename.endswith(".md")
        assert chunk.heading_path is not None
        assert chunk.metadata.status
        assert chunk.metadata.audience
        assert chunk.metadata.policy_authority


def test_exact_terminology_query_ranks_matching_document_highly(real_retriever: Retriever) -> None:
    result = real_retriever.retrieve("TrailPlus membership return window")
    top_document_ids = {c.chunk.document_id for c in result.candidates[:2]}
    assert "MEM-2026-01" in top_document_ids


def test_country_specific_query_ranks_international_shipping_document(
    real_retriever: Retriever,
) -> None:
    result = real_retriever.retrieve("Do you ship to Canada?")
    top_document_ids = {c.chunk.document_id for c in result.candidates[:2]}
    assert "SHIP-2026-INTL" in top_document_ids


def test_paraphrase_without_lexical_overlap_still_finds_relevant_document(
    real_retriever: Retriever,
) -> None:
    """No shared vocabulary with the corpus's own wording — must rely on
    the semantic half of the hybrid retriever."""
    result = real_retriever.retrieve("if I dislike the color of my bag can I send it back")
    top_document_ids = {c.chunk.document_id for c in result.candidates[:3]}
    assert "RET-2026-01" in top_document_ids or "RET-2026-02" in top_document_ids


def test_retrieve_is_deterministic_for_the_same_query(real_retriever: Retriever) -> None:
    query = "What is the warranty on bags?"
    first = real_retriever.retrieve(query)
    second = real_retriever.retrieve(query)
    first_ids = [c.chunk.chunk_id for c in first.candidates]
    second_ids = [c.chunk.chunk_id for c in second.candidates]
    assert first_ids == second_ids
    first_scores = [c.fused_score for c in first.candidates]
    second_scores = [c.fused_score for c in second.candidates]
    assert first_scores == second_scores


def test_retrieve_normalizes_the_query() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=8)
    retriever = Retriever([], provider)
    result = retriever.retrieve("  What   about   Canada?  ")
    assert result.normalized_query == "What about Canada?"
    assert result.query == "  What   about   Canada?  "


def test_retrieve_on_empty_index_returns_empty_result() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=8)
    retriever = Retriever([], provider)
    result = retriever.retrieve("anything")
    assert result.candidates == ()
    assert result.total_scored == 0


def test_retrieve_candidate_carries_all_four_score_components(real_retriever: Retriever) -> None:
    result = real_retriever.retrieve("Can I put the Breeze Tumbler in the dishwasher?")
    top = result.candidates[0]
    assert isinstance(top.lexical_score, float)
    assert isinstance(top.semantic_score, float)
    assert 0.0 <= top.lexical_score_normalized <= 1.0
    assert 0.0 <= top.semantic_score_normalized <= 1.0
    assert 0.0 <= top.fused_score <= 1.0


def test_retriever_length_matches_loaded_record_count(
    real_retriever: Retriever, real_index_records: list[IndexRecord]
) -> None:
    assert len(real_retriever) == len(real_index_records)
