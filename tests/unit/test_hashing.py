from __future__ import annotations

from aster_row_agent.rag.hashing import compute_chunk_id, hash_bytes, hash_text

_DOCUMENT_ID = "RET-2026-01"
_SOURCE_PATH = "knowledge-base/01-returns-policy-current.md"
_HEADING_PATH = ("Returns Policy", "Standard return window")


def test_hash_bytes_is_deterministic() -> None:
    assert hash_bytes(b"hello world") == hash_bytes(b"hello world")


def test_hash_bytes_differs_for_different_input() -> None:
    assert hash_bytes(b"hello world") != hash_bytes(b"hello world!")


def test_hash_text_is_deterministic() -> None:
    assert hash_text("Returns Policy") == hash_text("Returns Policy")


def test_hash_text_matches_utf8_hash_bytes() -> None:
    assert hash_text("café") == hash_bytes("café".encode())


def test_compute_chunk_id_is_deterministic() -> None:
    content_hash = hash_text("Customers may return within 30 days.")
    id_first = compute_chunk_id(
        document_id=_DOCUMENT_ID,
        source_path=_SOURCE_PATH,
        heading_path=_HEADING_PATH,
        chunk_index=0,
        content_hash=content_hash,
    )
    id_second = compute_chunk_id(
        document_id=_DOCUMENT_ID,
        source_path=_SOURCE_PATH,
        heading_path=_HEADING_PATH,
        chunk_index=0,
        content_hash=content_hash,
    )
    assert id_first == id_second


def test_compute_chunk_id_changes_when_content_hash_changes() -> None:
    id_a = compute_chunk_id(
        document_id=_DOCUMENT_ID,
        source_path=_SOURCE_PATH,
        heading_path=_HEADING_PATH,
        chunk_index=0,
        content_hash=hash_text("version A"),
    )
    id_b = compute_chunk_id(
        document_id=_DOCUMENT_ID,
        source_path=_SOURCE_PATH,
        heading_path=_HEADING_PATH,
        chunk_index=0,
        content_hash=hash_text("version B"),
    )
    assert id_a != id_b


def test_compute_chunk_id_changes_when_chunk_index_changes() -> None:
    content_hash = hash_text("same text")
    id_first = compute_chunk_id(
        document_id=_DOCUMENT_ID,
        source_path=_SOURCE_PATH,
        heading_path=_HEADING_PATH,
        chunk_index=0,
        content_hash=content_hash,
    )
    id_second = compute_chunk_id(
        document_id=_DOCUMENT_ID,
        source_path=_SOURCE_PATH,
        heading_path=_HEADING_PATH,
        chunk_index=1,
        content_hash=content_hash,
    )
    assert id_first != id_second


def test_compute_chunk_id_changes_when_document_id_changes() -> None:
    content_hash = hash_text("same text")
    id_a = compute_chunk_id(
        document_id="RET-2026-01",
        source_path=_SOURCE_PATH,
        heading_path=_HEADING_PATH,
        chunk_index=0,
        content_hash=content_hash,
    )
    id_b = compute_chunk_id(
        document_id="RET-2024-01",
        source_path=_SOURCE_PATH,
        heading_path=_HEADING_PATH,
        chunk_index=0,
        content_hash=content_hash,
    )
    assert id_a != id_b


def test_compute_chunk_id_is_not_a_random_uuid() -> None:
    # A stable ID must be reproducible without any per-process random state.
    ids = {
        compute_chunk_id(
            document_id=_DOCUMENT_ID,
            source_path=_SOURCE_PATH,
            heading_path=("Returns Policy",),
            chunk_index=0,
            content_hash=hash_text("x"),
        )
        for _ in range(5)
    }
    assert len(ids) == 1
