"""Ingestion orchestration: corpus → documents → chunks → embeddings → index + manifest.

This module wires together `frontmatter`, `chunking`, `embeddings`, and
`index` into one reproducible pipeline. It contains no business-answer
logic (no authority ranking, no conflict detection, no citation
selection) — see the module docstring in `models.py` for why that
boundary matters.

Documents that are internal, draft, superseded, or contain adversarial
text (prompt-injection payloads, fabricated policy claims) are ingested
exactly like every other document. This module's only job is to preserve
their content and metadata faithfully as data; deciding whether a piece of
evidence is *authoritative* is explicitly out of scope here (see
docs/architecture.md §4) and is never used as a reason to skip, alter, or
drop a source file during ingestion.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import pydantic

from aster_row_agent.config import Settings
from aster_row_agent.logging_setup import log_event
from aster_row_agent.rag.chunking import build_document_chunks
from aster_row_agent.rag.embeddings import EmbeddingProvider
from aster_row_agent.rag.errors import (
    CorpusNotFoundError,
    CorpusValidationError,
    DocumentParsingError,
    FrontMatterError,
    UnsafeCorpusPathError,
)
from aster_row_agent.rag.frontmatter import split_front_matter
from aster_row_agent.rag.hashing import hash_bytes, hash_text
from aster_row_agent.rag.index import save_index
from aster_row_agent.rag.models import (
    ChunkingConfig,
    CorpusManifest,
    Document,
    DocumentChunk,
    DocumentManifestEntry,
    DocumentMetadata,
    IndexRecord,
)

logger = logging.getLogger(__name__)

CHUNKING_STRATEGY_VERSION = 1


class IngestionResult(pydantic.BaseModel):
    """Everything produced by one call to `run_ingestion`."""

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    documents: tuple[Document, ...]
    chunks: tuple[DocumentChunk, ...]
    manifest: CorpusManifest
    index_path: Path


def discover_documents(knowledge_base_dir: Path) -> list[Path]:
    """List the corpus's Markdown files, sorted, with basic path-safety checks.

    Only direct children of `knowledge_base_dir` are considered (no
    recursive walk), and any entry that resolves outside the corpus root
    (e.g. a symlink pointing elsewhere on disk) is rejected rather than
    silently followed — the corpus is untrusted input and must not be able
    to make ingestion read arbitrary filesystem paths.
    """
    resolved_root = knowledge_base_dir.resolve()
    if not resolved_root.is_dir():
        raise CorpusNotFoundError(knowledge_base_dir)

    paths: list[Path] = []
    for entry in sorted(resolved_root.iterdir()):
        if entry.suffix != ".md":
            continue
        resolved_entry = entry.resolve()
        if not resolved_entry.is_relative_to(resolved_root):
            raise UnsafeCorpusPathError(resolved_entry, resolved_root)
        paths.append(entry)
    return paths


def parse_document(path: Path, *, knowledge_base_dir: Path) -> Document:
    """Parse one Markdown file into a `Document`, without modifying it on disk."""
    raw_bytes = path.read_bytes()
    raw_text = raw_bytes.decode("utf-8")
    front_matter, body = split_front_matter(raw_text, path=path)
    metadata = DocumentMetadata.model_validate(front_matter)
    source_path = f"{knowledge_base_dir.name}/{path.name}"
    return Document(
        source_path=source_path,
        metadata=metadata,
        body=body,
        content_hash=hash_bytes(raw_bytes),
    )


def parse_corpus(paths: list[Path], *, knowledge_base_dir: Path) -> list[Document]:
    """Parse every discovered document, collecting (not stopping at) every failure.

    Reporting every bad document in one `CorpusValidationError` is more
    useful for debugging a corpus than failing on the first one and hiding
    the rest.
    """
    documents: list[Document] = []
    failures: list[DocumentParsingError] = []
    for path in paths:
        try:
            documents.append(parse_document(path, knowledge_base_dir=knowledge_base_dir))
        except (UnicodeDecodeError, pydantic.ValidationError, FrontMatterError) as exc:
            # Each of these means "this one document is malformed", not "ingestion
            # itself is broken" — collected so a single run reports every bad
            # document instead of stopping at the first one.
            failures.append(DocumentParsingError(path, exc))

    if failures:
        raise CorpusValidationError(failures)
    return documents


def build_corpus_fingerprint(documents: list[Document]) -> str:
    """A single deterministic hash summarizing every document's identity + content."""
    ordered = sorted(documents, key=lambda d: d.source_path)
    canonical = "\n".join(f"{doc.source_path}:{doc.content_hash}" for doc in ordered)
    return hash_text(canonical)


def build_manifest(
    *,
    documents: list[Document],
    chunks_by_document: dict[str, list[DocumentChunk]],
    embedding_provider: EmbeddingProvider,
    knowledge_base_dir: Path,
    max_chunk_chars: int,
    ingestion_duration_ms: float,
) -> CorpusManifest:
    entries = tuple(
        DocumentManifestEntry(
            source_path=doc.source_path,
            document_id=doc.metadata.document_id,
            title=doc.metadata.title,
            status=doc.metadata.status,
            audience=doc.metadata.audience,
            policy_authority=doc.metadata.policy_authority,
            content_hash=doc.content_hash,
            chunk_count=len(chunks_by_document[doc.source_path]),
        )
        for doc in sorted(documents, key=lambda d: d.source_path)
    )
    total_chunks = sum(entry.chunk_count for entry in entries)
    return CorpusManifest(
        generated_at=datetime.now(UTC),
        corpus_fingerprint=build_corpus_fingerprint(documents),
        knowledge_base_dir=str(knowledge_base_dir),
        document_count=len(documents),
        chunk_count=total_chunks,
        documents=entries,
        embedding_model=embedding_provider.model_id,
        embedding_dimensions=embedding_provider.dimensions,
        chunking=ChunkingConfig(
            max_chunk_chars=max_chunk_chars,
            version=CHUNKING_STRATEGY_VERSION,
        ),
        ingestion_duration_ms=ingestion_duration_ms,
    )


def write_manifest(manifest: CorpusManifest, index_dir: Path) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / "manifest.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def run_ingestion(settings: Settings, embedding_provider: EmbeddingProvider) -> IngestionResult:
    """Run the full ingestion pipeline and return everything it produced.

    Steps (matching the documented `ingest` CLI command): validate the
    corpus directory, discover files, parse + validate every document,
    chunk them, embed the chunks, persist the index, and generate the
    corpus manifest. Any failure raises a specific `IngestionError`
    subclass with enough context to diagnose it — nothing is caught and
    hidden.
    """
    start = time.monotonic()
    log_event(
        logger,
        logging.INFO,
        "ingestion.started",
        knowledge_base_dir=str(settings.knowledge_base_dir),
        embedding_model=embedding_provider.model_id,
    )

    paths = discover_documents(settings.knowledge_base_dir)
    log_event(logger, logging.INFO, "ingestion.documents_discovered", count=len(paths))

    documents = parse_corpus(paths, knowledge_base_dir=settings.knowledge_base_dir)
    for doc in documents:
        log_event(
            logger,
            logging.INFO,
            "ingestion.document_parsed",
            source_path=doc.source_path,
            document_id=doc.metadata.document_id,
            status=doc.metadata.status,
            audience=doc.metadata.audience,
            policy_authority=doc.metadata.policy_authority,
        )

    chunks_by_document: dict[str, list[DocumentChunk]] = {}
    all_chunks: list[DocumentChunk] = []
    for doc in documents:
        doc_chunks = build_document_chunks(doc, settings.max_chunk_chars)
        chunks_by_document[doc.source_path] = doc_chunks
        all_chunks.extend(doc_chunks)
        log_event(
            logger,
            logging.INFO,
            "ingestion.document_chunked",
            source_path=doc.source_path,
            chunk_count=len(doc_chunks),
        )

    texts = [chunk.text for chunk in all_chunks]
    embeddings = embedding_provider.embed_documents(texts) if texts else []
    log_event(
        logger,
        logging.INFO,
        "ingestion.embeddings_generated",
        chunk_count=len(all_chunks),
        embedding_model=embedding_provider.model_id,
        dimensions=embedding_provider.dimensions,
    )

    records = [
        IndexRecord(
            chunk=chunk,
            embedding=tuple(vector),
            embedding_model=embedding_provider.model_id,
        )
        for chunk, vector in zip(all_chunks, embeddings, strict=True)
    ]
    index_path = save_index(records, settings.index_dir)

    duration_ms = (time.monotonic() - start) * 1000
    manifest = build_manifest(
        documents=documents,
        chunks_by_document=chunks_by_document,
        embedding_provider=embedding_provider,
        knowledge_base_dir=settings.knowledge_base_dir,
        max_chunk_chars=settings.max_chunk_chars,
        ingestion_duration_ms=duration_ms,
    )
    write_manifest(manifest, settings.index_dir)

    log_event(
        logger,
        logging.INFO,
        "ingestion.completed",
        document_count=manifest.document_count,
        chunk_count=manifest.chunk_count,
        corpus_fingerprint=manifest.corpus_fingerprint,
        duration_ms=round(duration_ms, 2),
        index_path=str(index_path),
    )

    return IngestionResult(
        documents=tuple(documents),
        chunks=tuple(all_chunks),
        manifest=manifest,
        index_path=index_path,
    )
