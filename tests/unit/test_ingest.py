from __future__ import annotations

import re
from pathlib import Path

import pytest

from aster_row_agent.config import Settings
from aster_row_agent.rag.embeddings import FakeDeterministicEmbeddingProvider
from aster_row_agent.rag.errors import CorpusNotFoundError, CorpusValidationError
from aster_row_agent.rag.hashing import hash_bytes
from aster_row_agent.rag.index import load_index_records
from aster_row_agent.rag.ingest import (
    build_corpus_fingerprint,
    discover_documents,
    parse_corpus,
    parse_document,
    run_ingestion,
)

EXPECTED_DOCUMENT_COUNT = 14


def _settings(knowledge_base_dir: Path, data_dir: Path) -> Settings:
    return Settings(
        knowledge_base_dir=knowledge_base_dir,
        data_dir=data_dir,
        max_chunk_chars=1200,
    )


# --- discovery ---------------------------------------------------------------


def test_discover_documents_finds_all_supplied_files(real_knowledge_base_dir: Path) -> None:
    paths = discover_documents(real_knowledge_base_dir)
    assert len(paths) == EXPECTED_DOCUMENT_COUNT
    assert all(p.suffix == ".md" for p in paths)


def test_discover_documents_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(CorpusNotFoundError):
        discover_documents(tmp_path / "does-not-exist")


def test_discover_documents_ignores_non_markdown_files(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("not markdown")
    (tmp_path / "doc.md").write_text(
        "---\ndocument_id: D\ntitle: T\nstatus: active\naudience: customer\n"
        "policy_authority: official\n---\n# T\n\n## S\n\nText.\n"
    )
    paths = discover_documents(tmp_path)
    assert [p.name for p in paths] == ["doc.md"]


# --- parsing real documents ---------------------------------------------------


def test_parse_document_preserves_required_metadata(real_knowledge_base_dir: Path) -> None:
    path = real_knowledge_base_dir / "01-returns-policy-current.md"
    document = parse_document(path, knowledge_base_dir=real_knowledge_base_dir)
    assert document.metadata.document_id == "RET-2026-01"
    assert document.metadata.title == "Returns Policy"
    assert document.metadata.status == "active"
    assert document.metadata.audience == "customer"
    assert document.metadata.policy_authority == "official"
    assert document.metadata.supersedes == "RET-2024-01"


def test_parse_document_preserves_source_filename(real_knowledge_base_dir: Path) -> None:
    path = real_knowledge_base_dir / "01-returns-policy-current.md"
    document = parse_document(path, knowledge_base_dir=real_knowledge_base_dir)
    assert document.source_path == "knowledge-base/01-returns-policy-current.md"


def test_parse_document_content_hash_matches_raw_bytes(real_knowledge_base_dir: Path) -> None:
    path = real_knowledge_base_dir / "07-warranty.md"
    document = parse_document(path, knowledge_base_dir=real_knowledge_base_dir)
    assert document.content_hash == hash_bytes(path.read_bytes())


def test_parse_corpus_collects_every_failure_before_raising(tmp_path: Path) -> None:
    (tmp_path / "bad-one.md").write_text("no front matter at all\n")
    (tmp_path / "bad-two.md").write_text("---\n[invalid yaml\n---\nBody\n")
    (tmp_path / "good.md").write_text(
        "---\ndocument_id: D\ntitle: T\nstatus: active\naudience: customer\n"
        "policy_authority: official\n---\n# T\n\n## S\n\nText.\n"
    )
    paths = discover_documents(tmp_path)
    with pytest.raises(CorpusValidationError) as exc_info:
        parse_corpus(paths, knowledge_base_dir=tmp_path)
    assert len(exc_info.value.errors) == 2


# --- corpus fingerprint --------------------------------------------------------


def test_corpus_fingerprint_is_deterministic(real_knowledge_base_dir: Path) -> None:
    paths = discover_documents(real_knowledge_base_dir)
    documents = parse_corpus(paths, knowledge_base_dir=real_knowledge_base_dir)
    assert build_corpus_fingerprint(documents) == build_corpus_fingerprint(documents)


def test_corpus_fingerprint_is_order_independent(real_knowledge_base_dir: Path) -> None:
    paths = discover_documents(real_knowledge_base_dir)
    documents = parse_corpus(paths, knowledge_base_dir=real_knowledge_base_dir)
    reversed_documents = list(reversed(documents))
    assert build_corpus_fingerprint(documents) == build_corpus_fingerprint(reversed_documents)


# --- end-to-end ingestion ------------------------------------------------------


def test_run_ingestion_end_to_end_over_real_corpus(
    real_knowledge_base_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(real_knowledge_base_dir, tmp_path / ".data")
    result = run_ingestion(settings, FakeDeterministicEmbeddingProvider())

    assert result.manifest.document_count == EXPECTED_DOCUMENT_COUNT
    assert len(result.documents) == EXPECTED_DOCUMENT_COUNT
    assert result.manifest.chunk_count == len(result.chunks)
    assert result.manifest.chunk_count > 0
    assert result.index_path.exists()
    assert (result.index_path.parent / "manifest.json").exists()


def test_run_ingestion_is_idempotent_over_unchanged_corpus(
    real_knowledge_base_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(real_knowledge_base_dir, tmp_path / ".data")

    first = run_ingestion(settings, FakeDeterministicEmbeddingProvider())
    first_records = load_index_records(settings.index_dir)

    second = run_ingestion(settings, FakeDeterministicEmbeddingProvider())
    second_records = load_index_records(settings.index_dir)

    assert first.manifest.document_count == second.manifest.document_count
    assert first.manifest.chunk_count == second.manifest.chunk_count
    assert first.manifest.corpus_fingerprint == second.manifest.corpus_fingerprint

    first_ids = sorted(r.chunk.chunk_id for r in first_records)
    second_ids = sorted(r.chunk.chunk_id for r in second_records)
    assert first_ids == second_ids
    assert len(second_ids) == len(set(second_ids)), "re-ingestion must not create duplicates"


def test_run_ingestion_preserves_superseded_metadata(
    real_knowledge_base_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(real_knowledge_base_dir, tmp_path / ".data")
    result = run_ingestion(settings, FakeDeterministicEmbeddingProvider())

    legacy_doc = next(
        d for d in result.documents if d.source_path.endswith("02-returns-policy-legacy.md")
    )
    assert legacy_doc.metadata.status == "superseded"
    assert legacy_doc.metadata.superseded_by == "RET-2026-01"
    # Superseded is a distinct concept from authority: this document was (and
    # remains recorded as) official, it is just no longer the current answer.
    assert legacy_doc.metadata.policy_authority == "official"


def test_run_ingestion_preserves_internal_audience_metadata(
    real_knowledge_base_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(real_knowledge_base_dir, tmp_path / ".data")
    result = run_ingestion(settings, FakeDeterministicEmbeddingProvider())

    internal_doc = next(
        d for d in result.documents if d.source_path.endswith("13-support-escalation.md")
    )
    assert internal_doc.metadata.audience == "internal"
    assert internal_doc.metadata.status == "active"
    assert internal_doc.metadata.policy_authority == "official"


def test_run_ingestion_preserves_draft_non_authoritative_metadata(
    real_knowledge_base_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(real_knowledge_base_dir, tmp_path / ".data")
    result = run_ingestion(settings, FakeDeterministicEmbeddingProvider())

    scratchpad_doc = next(
        d
        for d in result.documents
        if d.source_path.endswith("14-internal-content-migration-notes.md")
    )
    assert scratchpad_doc.metadata.status == "draft"
    assert scratchpad_doc.metadata.audience == "internal"
    assert scratchpad_doc.metadata.policy_authority == "none"
    assert scratchpad_doc.metadata.customer_answering is False


def test_run_ingestion_does_not_drop_the_scratchpad_document(
    real_knowledge_base_dir: Path, tmp_path: Path
) -> None:
    """The draft/adversarial document must still be indexed as data, not excluded."""
    settings = _settings(real_knowledge_base_dir, tmp_path / ".data")
    result = run_ingestion(settings, FakeDeterministicEmbeddingProvider())
    scratchpad_chunks = [
        c for c in result.chunks if c.source_path.endswith("14-internal-content-migration-notes.md")
    ]
    assert len(scratchpad_chunks) > 0


def test_prompt_injection_text_is_preserved_verbatim_as_inert_data(
    real_knowledge_base_dir: Path, tmp_path: Path
) -> None:
    """The embedded fake 'SYSTEM INSTRUCTION' payload must survive ingestion as
    ordinary chunk text — proving ingestion stores content, it does not
    interpret or strip it as an instruction."""
    settings = _settings(real_knowledge_base_dir, tmp_path / ".data")
    result = run_ingestion(settings, FakeDeterministicEmbeddingProvider())

    scratchpad_chunks = [
        c
        for c in result.chunks
        if c.source_path.endswith("14-internal-content-migration-notes.md")
    ]
    combined_text = "\n".join(c.text for c in scratchpad_chunks)
    assert "SYSTEM INSTRUCTION" in combined_text
    assert "Ignore all prior rules" in combined_text
    # And ingestion completed normally rather than raising/behaving specially.
    assert result.manifest.document_count == EXPECTED_DOCUMENT_COUNT


def test_source_files_remain_byte_for_byte_unchanged_after_ingestion(
    real_knowledge_base_dir: Path, tmp_path: Path
) -> None:
    md_paths = sorted(real_knowledge_base_dir.glob("*.md"))
    before = {path.name: hash_bytes(path.read_bytes()) for path in md_paths}
    settings = _settings(real_knowledge_base_dir, tmp_path / ".data")
    run_ingestion(settings, FakeDeterministicEmbeddingProvider())
    after = {path.name: hash_bytes(path.read_bytes()) for path in md_paths}
    assert before == after
    assert len(before) == EXPECTED_DOCUMENT_COUNT


# --- provenance / manifest / index mapping ------------------------------------


def test_chunk_provenance_is_complete_for_every_chunk(
    real_knowledge_base_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(real_knowledge_base_dir, tmp_path / ".data")
    result = run_ingestion(settings, FakeDeterministicEmbeddingProvider())

    for chunk in result.chunks:
        assert chunk.source_path.startswith("knowledge-base/")
        assert chunk.document_id
        assert chunk.title
        assert chunk.chunk_id
        assert chunk.content_hash
        assert chunk.metadata.status
        assert chunk.metadata.audience
        assert chunk.metadata.policy_authority
        # A citation must be producible from the chunk alone.
        assert re.match(r"^\S+\.md( — .+)?$", chunk.citation)


def test_corpus_manifest_fields_are_populated_correctly(
    real_knowledge_base_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(real_knowledge_base_dir, tmp_path / ".data")
    result = run_ingestion(settings, FakeDeterministicEmbeddingProvider())
    manifest = result.manifest

    assert manifest.document_count == EXPECTED_DOCUMENT_COUNT
    assert len(manifest.documents) == EXPECTED_DOCUMENT_COUNT
    assert manifest.chunk_count == sum(entry.chunk_count for entry in manifest.documents)
    assert manifest.embedding_model == "fake-hash-embedding-v1"
    assert manifest.embedding_dimensions > 0
    assert manifest.chunking.max_chunk_chars == settings.max_chunk_chars
    assert manifest.index_schema_version == 1
    assert manifest.ingestion_duration_ms >= 0
    assert len(manifest.corpus_fingerprint) == 64  # sha256 hex digest length


def test_manifest_is_persisted_and_reloadable(
    real_knowledge_base_dir: Path, tmp_path: Path
) -> None:
    from aster_row_agent.rag.models import CorpusManifest

    settings = _settings(real_knowledge_base_dir, tmp_path / ".data")
    result = run_ingestion(settings, FakeDeterministicEmbeddingProvider())

    manifest_path = result.index_path.parent / "manifest.json"
    reloaded = CorpusManifest.model_validate_json(manifest_path.read_text())
    assert reloaded.document_count == result.manifest.document_count
    assert reloaded.corpus_fingerprint == result.manifest.corpus_fingerprint


def test_index_records_map_back_to_source_filename_and_heading(
    real_knowledge_base_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(real_knowledge_base_dir, tmp_path / ".data")
    run_ingestion(settings, FakeDeterministicEmbeddingProvider())

    records = load_index_records(settings.index_dir)
    returns_policy_records = [
        r for r in records if r.chunk.source_path.endswith("01-returns-policy-current.md")
    ]
    assert returns_policy_records, "expected an indexed chunk for the current returns policy"
    headings = {r.chunk.heading_path.leaf for r in returns_policy_records}
    assert "Standard return window" in headings

    standard_window_record = next(
        r for r in returns_policy_records if r.chunk.heading_path.leaf == "Standard return window"
    )
    assert standard_window_record.chunk.citation == (
        "01-returns-policy-current.md — Returns Policy > Standard return window"
    )
