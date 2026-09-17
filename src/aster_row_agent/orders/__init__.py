"""Order lookup: repository, normalization, sanitization, and the lookup service.

Mirrors the module-boundary discipline established in `rag/`: pure data
models in `models.py`, one concern per module, and a hard security
invariant that this package's docstrings and tests both enforce —
`RawOrderRecord` (and anything nested under it: `customer`, `internal`)
never crosses into an LLM-facing type. `CustomerSafeOrder` is the only
type any future agent/tool layer is allowed to see, and it is built by an
explicit allowlist (`sanitize.py`), never by removing known-bad fields
from the raw record.
"""

__all__: list[str] = []
