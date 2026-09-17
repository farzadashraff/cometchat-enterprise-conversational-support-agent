from __future__ import annotations

import math

from aster_row_agent.rag.embeddings import FakeDeterministicEmbeddingProvider


def test_fake_embedding_is_deterministic_across_instances() -> None:
    provider_a = FakeDeterministicEmbeddingProvider(dimensions=16)
    provider_b = FakeDeterministicEmbeddingProvider(dimensions=16)
    assert provider_a.embed_query("hello") == provider_b.embed_query("hello")


def test_fake_embedding_different_text_gives_different_vector() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=16)
    assert provider.embed_query("hello") != provider.embed_query("goodbye")


def test_fake_embedding_respects_configured_dimensions() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=48)
    vector = provider.embed_query("anything")
    assert len(vector) == 48
    assert provider.dimensions == 48


def test_fake_embedding_is_l2_normalized() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=32)
    vector = provider.embed_query("normalize me")
    norm = math.sqrt(sum(v * v for v in vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-6)


def test_fake_embedding_batch_matches_individual_calls() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=16)
    texts = ["one", "two", "three"]
    batch = provider.embed_documents(texts)
    individual = [provider.embed_query(t) for t in texts]
    assert batch == individual


def test_fake_embedding_has_no_network_dependency_at_import_time() -> None:
    # If this import required network access or a model download, constructing
    # the provider (not just importing the module) would be slow/fail in a
    # sandboxed CI runner. This is a smoke test for that property.
    provider = FakeDeterministicEmbeddingProvider()
    assert provider.model_id == "fake-hash-embedding-v1"
