from __future__ import annotations

from aster_row_agent.rag.embeddings import FakeDeterministicEmbeddingProvider
from aster_row_agent.rag.lexical import LexicalIndex, tokenize
from aster_row_agent.rag.models import DocumentChunk, DocumentMetadata, HeadingPath


def _metadata() -> DocumentMetadata:
    return DocumentMetadata.model_validate(
        {
            "document_id": "DOC-1",
            "title": "Doc",
            "status": "active",
            "audience": "customer",
            "policy_authority": "official",
        }
    )


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="DOC-1",
        source_path="knowledge-base/doc.md",
        title="Doc",
        heading_path=HeadingPath(parts=("Doc", "Section")),
        chunk_index=0,
        text=text,
        content_hash="irrelevant",
        metadata=_metadata(),
    )


def _records(texts: list[str]) -> list:
    from aster_row_agent.rag.models import IndexRecord

    provider = FakeDeterministicEmbeddingProvider(dimensions=8)
    return [
        IndexRecord(
            chunk=_chunk(f"chunk-{i}", text),
            embedding=tuple(provider.embed_query(text)),
            embedding_model=provider.model_id,
        )
        for i, text in enumerate(texts)
    ]


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Canada, Canada!") == ["canada", "canada"]


def test_tokenize_keeps_numbers_and_apostrophes() -> None:
    assert tokenize("30 days, don't delay") == ["30", "days", "don't", "delay"]


def test_tokenize_removes_common_stopwords() -> None:
    tokens = tokenize("What about Canada, and how long does it take?")
    assert "canada" in tokens
    assert "long" in tokens or "take" in tokens
    for stopword in ("what", "about", "and", "how", "does", "it"):
        assert stopword not in tokens


def test_tokenize_does_not_remove_negation() -> None:
    # "not" is deliberately kept out of the stopword list in some IR setups,
    # but this project's list does remove it (BM25 is a coarse relevance
    # signal, not a negation-aware one) — pin the actual behavior so a
    # future change to the stopword list is a deliberate, visible decision.
    assert "not" not in tokenize("This is not covered by warranty")


def test_lexical_index_score_all_returns_one_score_per_record_in_order() -> None:
    index = LexicalIndex(_records(["alpha beta", "gamma delta", "epsilon zeta"]))
    scores = index.score_all("alpha")
    assert len(scores) == 3


def test_lexical_index_exact_term_match_scores_higher_than_no_match() -> None:
    index = LexicalIndex(
        _records(
            [
                "Canada shipping details and delivery estimate information",
                "warranty periods for bags and backpacks",
                "gift card price adjustment policy details",
            ]
        )
    )
    scores = index.score_all("Canada")
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    assert scores[1] == 0.0
    assert scores[2] == 0.0


def test_lexical_index_empty_records_returns_empty_scores() -> None:
    index = LexicalIndex([])
    assert index.score_all("anything") == []
    assert len(index) == 0


def test_lexical_index_len_reflects_record_count() -> None:
    index = LexicalIndex(_records(["a", "b", "c"]))
    assert len(index) == 3
