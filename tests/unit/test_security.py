"""Security-focused ingestion tests: path safety and prompt-injection inertness.

These complement the broader assertions in test_ingest.py with tests that
exist specifically to prove a security property, not just a functional one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aster_row_agent.rag.errors import UnsafeCorpusPathError
from aster_row_agent.rag.ingest import discover_documents


def _write_minimal_doc(path: Path, document_id: str) -> None:
    path.write_text(
        f"---\ndocument_id: {document_id}\ntitle: T\nstatus: active\n"
        "audience: customer\npolicy_authority: official\n---\n# T\n\n## S\n\nText.\n"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="symlink permissions differ on Windows")
def test_discover_documents_rejects_symlink_escaping_corpus_root(tmp_path: Path) -> None:
    corpus_root = tmp_path / "knowledge-base"
    corpus_root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret.md"
    _write_minimal_doc(outside_file, "SECRET-1")

    escaping_symlink = corpus_root / "escape.md"
    escaping_symlink.symlink_to(outside_file)

    with pytest.raises(UnsafeCorpusPathError):
        discover_documents(corpus_root)


def test_discover_documents_accepts_symlink_within_corpus_root(tmp_path: Path) -> None:
    corpus_root = tmp_path / "knowledge-base"
    corpus_root.mkdir()
    real_file = corpus_root / "real.md"
    _write_minimal_doc(real_file, "REAL-1")
    internal_symlink = corpus_root / "alias.md"
    internal_symlink.symlink_to(real_file)

    paths = discover_documents(corpus_root)
    assert {p.name for p in paths} == {"real.md", "alias.md"}


def test_discover_documents_does_not_recurse_into_subdirectories(tmp_path: Path) -> None:
    corpus_root = tmp_path / "knowledge-base"
    corpus_root.mkdir()
    _write_minimal_doc(corpus_root / "top-level.md", "TOP-1")
    nested_dir = corpus_root / "nested"
    nested_dir.mkdir()
    _write_minimal_doc(nested_dir / "nested.md", "NESTED-1")

    paths = discover_documents(corpus_root)
    assert [p.name for p in paths] == ["top-level.md"]
