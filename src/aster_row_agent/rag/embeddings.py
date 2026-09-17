"""Embedding provider abstraction.

The rest of the pipeline depends only on the `EmbeddingProvider` protocol,
never on a specific library — so the real provider can be swapped (or
skipped entirely in fast/offline tests) without touching ingestion,
indexing, or search code.

Two implementations are provided:

- `FastEmbedProvider`: the real, local, offline embedding model
  (`BAAI/bge-small-en-v1.5` via `fastembed`, ONNX runtime — no torch, no
  external API, no API key). Weights are downloaded once and cached under
  the configured embedding cache directory.
- `FakeDeterministicEmbeddingProvider`: a pure hash-based embedding used in
  unit tests and any fast/offline run. It has no notion of semantic
  similarity, but it is completely deterministic, requires no network
  access or model download, and is enough to exercise every piece of the
  pipeline (chunking, indexing, persistence, search plumbing) without an
  expensive external call.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that can turn text into fixed-dimension vectors."""

    model_id: str
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of chunk texts (indexing time)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (search time)."""
        ...


class FakeDeterministicEmbeddingProvider:
    """Deterministic, dependency-free embedding for tests and offline runs.

    Each vector is derived purely from a SHA-256 hash of the input text,
    expanded to `dimensions` floats in [-1, 1] and L2-normalized. The same
    text always produces the exact same vector, and different texts produce
    different (uncorrelated, but not semantically meaningful) vectors —
    sufficient to prove that indexing/search plumbing works without paying
    for a real model.
    """

    model_id = "fake-hash-embedding-v1"

    def __init__(self, dimensions: int = 32) -> None:
        self.dimensions = dimensions

    def _embed_one(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        counter = 0
        while len(values) < self.dimensions:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for offset in range(0, len(digest), 4):
                if len(values) >= self.dimensions:
                    break
                raw = int.from_bytes(digest[offset : offset + 4], "big")
                values.append((raw / 0xFFFFFFFF) * 2 - 1)
            counter += 1
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class FastEmbedProvider:
    """Local ONNX embedding model via `fastembed` (no API key, offline after first download)."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        # Imported lazily so that code paths using only the fake provider
        # (e.g. most unit tests) never need `fastembed`/`onnxruntime` importable.
        from fastembed import TextEmbedding

        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        self.model_id = model_name
        # fastembed doesn't expose dimensionality as a static attribute across
        # versions; determine it once, deterministically, from a probe embed.
        probe = next(iter(self._model.embed(["dimension probe"])))
        self.dimensions = len(probe)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([text]))).tolist()
