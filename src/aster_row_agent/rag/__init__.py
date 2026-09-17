"""RAG evidence layer: ingestion, chunking, embeddings, and the local index.

This package intentionally stops at "evidence" — it has no knowledge of
conversation state, order lookup, or LLM prompting. Those concerns live in
sibling packages added in later phases so that the ingestion pipeline can be
built, tested, and reasoned about in isolation.
"""

__all__: list[str] = []
