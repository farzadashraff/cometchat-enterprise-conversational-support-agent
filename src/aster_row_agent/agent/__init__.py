"""The application/agent layer: session state, routing, prompting, LLM
integration, response validation, and handoff.

This package is the only place `rag.EvidenceBundle` (Phase 3) and
`orders.OrderLookupResult` (Phase 4) are consumed together. It never
reimplements retrieval, authority, conflict detection, order lookup, or
sanitization — those boundaries are frozen and this package treats them
as trusted, typed inputs.

Guiding precedence for every decision in this package (see
docs/architecture.md §18):

    deterministic application logic
        > trust boundaries
        > structured evidence / tool results
        > LLM generation

The LLM is used only for language synthesis (phrasing an answer from
evidence/order data that has already been selected and validated by
deterministic code) and, in a narrow and bounded way, for interpreting
genuinely ambiguous routing — never for deciding security, privacy,
policy precedence, or whether an action actually occurred.
"""

__all__: list[str] = []
