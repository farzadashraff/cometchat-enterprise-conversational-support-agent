from __future__ import annotations

from pathlib import Path

import pytest

from aster_row_agent.rag.embeddings import FakeDeterministicEmbeddingProvider
from aster_row_agent.rag.errors import IndexNotFoundError
from aster_row_agent.rag.index import VectorIndex, load_index_records, save_index
from aster_row_agent.rag.models import DocumentChunk, DocumentMetadata, HeadingPath, IndexRecord


def _chunk(chunk_id: str, text: str, heading: str = "Section") -> DocumentChunk:
    metadata = DocumentMetadata.model_validate(
        {
            "document_id": "DOC-1",
            "title": "Doc",
            "status": "active",
            "audience": "customer",
            "policy_authority": "official",
        }
    )
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="DOC-1",
        source_path="knowledge-base/doc.md",
        title="Doc",
        heading_path=HeadingPath(parts=("Doc", heading)),
        chunk_index=0,
        text=text,
        content_hash="irrelevant",
        metadata=metadata,
    )


def _records(provider: FakeDeterministicEmbeddingProvider, texts: list[str]) -> list[IndexRecord]:
    return [
        IndexRecord(
            chunk=_chunk(f"chunk-{i}", text, heading=f"Section {i}"),
            embedding=tuple(provider.embed_query(text)),
            embedding_model=provider.model_id,
        )
        for i, text in enumerate(texts)
    ]


def test_save_and_load_index_roundtrip(tmp_path: Path) -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=16)
    records = _records(provider, ["alpha text", "beta text", "gamma text"])
    save_index(records, tmp_path)

    loaded = load_index_records(tmp_path)
    assert len(loaded) == 3
    assert [r.chunk.chunk_id for r in loaded] == [r.chunk.chunk_id for r in records]
    assert [r.embedding for r in loaded] == [r.embedding for r in records]


def test_save_index_is_a_full_rebuild_not_an_append(tmp_path: Path) -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=16)
    save_index(_records(provider, ["first batch a", "first batch b"]), tmp_path)
    save_index(_records(provider, ["second batch only"]), tmp_path)

    loaded = load_index_records(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].chunk.text == "second batch only"


def test_load_index_missing_raises_index_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError):
        load_index_records(tmp_path / "does-not-exist")


def test_vector_index_search_returns_exact_text_match_as_top_result() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=16)
    texts = ["completely unrelated topic", "another unrelated topic", "the exact target text"]
    records = _records(provider, texts)
    index = VectorIndex(records, provider)

    results = index.search("the exact target text", top_k=2)
    assert results[0].chunk.text == "the exact target text"
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_vector_index_search_respects_top_k() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=16)
    records = _records(provider, [f"text number {i}" for i in range(10)])
    index = VectorIndex(records, provider)
    assert len(index.search("text number 3", top_k=3)) == 3


def test_vector_index_search_on_empty_index_returns_empty_list() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=16)
    index = VectorIndex([], provider)
    assert index.search("anything", top_k=5) == []


def test_vector_index_len_reflects_record_count() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=16)
    records = _records(provider, ["a", "b"])
    index = VectorIndex(records, provider)
    assert len(index) == 2
